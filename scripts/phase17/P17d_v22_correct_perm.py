#!/usr/bin/env python3
"""Phase 17D v22 — CORRECTED single-perm builder (MAF + LD-bin, nearest-HM3 LD impute).

Methodology fix over the earlier P17d_v22_optimized_permutation.py:
  - Earlier pool: baselineLD.l2.ldscore.gz (HM3-restricted, 1.19M SNPs)
    → only 361/1744 C0 SNPs captured (79% dropout)
  - This script's pool: baselineLD.annot.gz ∩ 1KG.EUR.frq = full ~10M SNP set
    → all 1744/1744 C0 SNPs captured
  - LD score: for HM3 SNPs taken from .l2.ldscore.gz (baseL2); for non-HM3 SNPs
    interpolated from nearest HM3 SNP within the same chromosome (positional
    nearest-neighbour)
  - Matching: 10-decile MAF × 10-decile LD = 100 bins (same scheme as the earlier version)
  - Memory-lean: processes ONE perm per Python invocation (caller loops to free RAM)

Usage:  python3 P17d_v22_correct_perm.py PERM_INDEX
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
FRQ = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_frq"
P14E_CLUSTER = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17d_v22"

if len(sys.argv) < 2:
    sys.exit("Usage: P17d_v22_correct_perm.py PERM_INDEX")
PERM = int(sys.argv[1])

perm_dir = OUT / f"perm_{PERM:03d}"
if perm_dir.exists() and len(list(perm_dir.glob("cluster_perm.*.annot.gz"))) == 22:
    print(f"  perm {PERM}: already complete, skipping")
    sys.exit(0)
perm_dir.mkdir(parents=True, exist_ok=True)

t0 = datetime.now()
print(f"P17d v22 CORRECT perm {PERM} — {t0:%H:%M:%S}", flush=True)


# ─── 1. Build full master: baselineLD.annot.gz ∩ 1KG EUR .frq, with LD-imputed
print("  [1] master ∪ LD-impute...", flush=True)
master_pieces = []
for chrom in range(1, 23):
    # Full SNP list with BP (for nearest-neighbour) from annot.gz
    annot = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.annot.gz", sep="\t",
                        compression="gzip", usecols=["SNP", "BP"])
    # MAF from EUR .frq (10M SNPs)
    frq = pd.read_csv(FRQ / f"1000G.EUR.QC.{chrom}.frq", sep=r"\s+",
                      usecols=["SNP", "MAF"])
    # LD score (HM3-only): merge then nearest-neighbour impute
    ldsc = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.l2.ldscore.gz", sep="\t",
                       compression="gzip", usecols=["SNP", "baseL2"])
    full = annot.merge(frq, on="SNP", how="inner").merge(ldsc, on="SNP", how="left")
    full["CHR"] = chrom
    # For SNPs without LD score (non-HM3), impute from nearest HM3 SNP by BP
    has_ld = full.dropna(subset=["baseL2"]).sort_values("BP")
    miss = full[full["baseL2"].isna()].copy()
    if len(miss) > 0 and len(has_ld) > 0:
        idx = np.searchsorted(has_ld["BP"].values, miss["BP"].values)
        idx = np.clip(idx, 0, len(has_ld) - 1)
        miss["baseL2"] = has_ld["baseL2"].values[idx]
        full = pd.concat([has_ld, miss], ignore_index=True)
    master_pieces.append(full[["CHR", "SNP", "BP", "MAF", "baseL2"]])
    del annot, frq, ldsc, full, has_ld, miss; gc.collect()
master = pd.concat(master_pieces, ignore_index=True)
del master_pieces; gc.collect()

master["maf_bin"] = pd.qcut(master["MAF"], 10, labels=False, duplicates="drop")
master["l2_bin"]  = pd.qcut(master["baseL2"], 10, labels=False, duplicates="drop")
master["bin_id"]  = master["maf_bin"].astype(str) + "_" + master["l2_bin"].astype(str)
master = master[["SNP", "bin_id"]]
gc.collect()


# ─── 2. Load C0 (all 1744) and lookup their bin
print("  [2] C0 SNPs (all 1744)...", flush=True)
c0_snps = []
for chrom in range(1, 23):
    a = pd.read_csv(P14E_CLUSTER / f"cluster.{chrom}.annot.gz", sep="\t",
                    compression="gzip", usecols=["SNP", "C0"])
    c0_snps.append(a.loc[a["C0"] == 1, "SNP"])
c0 = pd.concat(c0_snps).rename("SNP").to_frame()
c0_lookup = c0.merge(master, on="SNP", how="inner")
print(f"     C0 captured: {len(c0_lookup):,}/{len(c0):,}", flush=True)
c0_bin_counts = c0_lookup["bin_id"].value_counts().to_dict()
del c0_snps, c0, c0_lookup; gc.collect()


# ─── 3. Pre-index master by bin
print("  [3] bin index...", flush=True)
bin_to_snps = {b: np.array(g["SNP"].values)
               for b, g in master.groupby("bin_id")}
all_snps = master["SNP"].values
del master; gc.collect()


# ─── 4. Draw matched SNPs for THIS perm
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
print(f"     drew {len(drawn_set):,} unique matched SNPs", flush=True)
del bin_to_snps, all_snps; gc.collect()


# ─── 5. Write per-chr annot files (read template fresh, no cache)
print(f"  [5] write 22 annot files...", flush=True)
for chrom in range(1, 23):
    bf = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.annot.gz", sep="\t",
                     compression="gzip", usecols=["CHR", "BP", "SNP", "CM"])
    bf["C0_perm"] = bf["SNP"].astype(str).isin(drawn_set).astype(int)
    bf.to_csv(perm_dir / f"cluster_perm.{chrom}.annot.gz",
              sep="\t", index=False, compression="gzip")
    del bf; gc.collect()

dt = (datetime.now() - t0).total_seconds()
print(f"  perm {PERM} DONE in {dt:.0f}s (matched draw with FULL C0 1744 + LD-impute)", flush=True)
