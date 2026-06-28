#!/usr/bin/env python3
"""Phase 17D v22 — LEAN single-perm builder (memory-efficient, resume-friendly).

Memory-conscious rewrite of the earlier P17d_v22_optimized_permutation.py:
  - Processes ONE perm per invocation (caller loops; Python re-spawn frees RAM)
  - Does NOT cache 22-chr annot templates in RAM; reads each chr fresh from disk
  - bin_to_snps built from a streamed master (chr-by-chr concat then dropped)
  - Resume: if perm_NNN dir already has 22 annot files, skip

Usage:  python3 P17d_v22_lean_perm.py PERM_INDEX
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import sys, gc, warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
BASELINE_V22 = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"
P14E_FRQ = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_frq"
P14E_CLUSTER = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17d_v22"

if len(sys.argv) < 2:
    sys.exit("Usage: P17d_v22_lean_perm.py PERM_INDEX")
PERM = int(sys.argv[1])

perm_dir = OUT / f"perm_{PERM:03d}"
if perm_dir.exists() and len(list(perm_dir.glob("cluster_perm.*.annot.gz"))) == 22:
    print(f"  perm {PERM}: already complete, skipping")
    sys.exit(0)
perm_dir.mkdir(parents=True, exist_ok=True)

t0 = datetime.now()
print(f"P17d v22 LEAN perm {PERM} — {t0:%H:%M:%S}", flush=True)

# ---- Build HM3 master (v2.2 L2-bin × MAF-bin) ----
print("  [1] master bin map...", flush=True)
master_pieces = []
for chrom in range(1, 23):
    base_ld = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.l2.ldscore.gz",
                          sep="\t", compression="gzip",
                          usecols=lambda c: c == "SNP" or c.endswith("L2"))
    l2_col = "baseL2" if "baseL2" in base_ld.columns else \
             [c for c in base_ld.columns if c.endswith("L2")][0]
    base_ld = base_ld[["SNP", l2_col]].rename(columns={l2_col: "baseL2_v22"})
    frq = pd.read_csv(P14E_FRQ / f"1000G.EUR.QC.{chrom}.frq",
                      delim_whitespace=True, usecols=["SNP", "MAF"])
    master_pieces.append(base_ld.merge(frq, on="SNP", how="inner"))
    del base_ld, frq
master = pd.concat(master_pieces, ignore_index=True)
del master_pieces; gc.collect()
master["maf_bin"] = pd.qcut(master["MAF"], 10, labels=False, duplicates="drop")
master["l2_bin"]  = pd.qcut(master["baseL2_v22"], 10, labels=False, duplicates="drop")
master["bin_id"]  = master["maf_bin"].astype(str) + "_" + master["l2_bin"].astype(str)
master = master[["SNP", "bin_id"]]  # drop heavy cols
gc.collect()

# ---- C0 SNPs ----
print("  [2] C0 SNPs...", flush=True)
c0 = []
for chrom in range(1, 23):
    a = pd.read_csv(P14E_CLUSTER / f"cluster.{chrom}.annot.gz", sep="\t",
                    compression="gzip", usecols=["SNP", "C0"])
    c0.append(a.loc[a["C0"] == 1, ["SNP"]])
c0_df = pd.concat(c0, ignore_index=True).merge(master, on="SNP", how="inner")
c0_bin_counts = c0_df["bin_id"].value_counts().to_dict()
del c0, c0_df; gc.collect()

# ---- Index master by bin ----
print("  [3] bin index...", flush=True)
bin_to_snps = {b: np.array(g["SNP"].values)
               for b, g in master.groupby("bin_id")}
all_snps = master["SNP"].values
del master; gc.collect()

# ---- Draw matched SNPs for THIS perm ----
print(f"  [4] draw perm {PERM}...", flush=True)
rng = np.random.RandomState(PERM)
drawn = []
for bin_id, n_needed in c0_bin_counts.items():
    pool = bin_to_snps.get(bin_id, np.array([]))
    if len(pool) >= n_needed:
        drawn.extend(rng.choice(pool, size=n_needed, replace=False).tolist())
    else:
        drawn.extend(pool.tolist())
        need = n_needed - len(pool)
        drawn.extend(rng.choice(all_snps, size=need, replace=False).tolist())
drawn_set = set(drawn)
del bin_to_snps, all_snps; gc.collect()

# ---- Write per-chr annot (read template fresh from disk, no cache) ----
print(f"  [5] write 22 annot files...", flush=True)
for chrom in range(1, 23):
    bf = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.annot.gz",
                     sep="\t", compression="gzip",
                     usecols=["CHR", "BP", "SNP", "CM"])
    bf["C0_perm"] = bf["SNP"].astype(str).isin(drawn_set).astype(int)
    bf.to_csv(perm_dir / f"cluster_perm.{chrom}.annot.gz",
              sep="\t", index=False, compression="gzip")
    del bf; gc.collect()

dt = (datetime.now() - t0).total_seconds()
print(f"  perm {PERM} DONE in {dt:.0f}s", flush=True)
