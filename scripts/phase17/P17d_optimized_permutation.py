#!/usr/bin/env python3
"""Phase 17D: OPTIMIZED empirical permutation null for C0 (Young) enrichment.

Strategy: Generate annotation files efficiently by only writing the 4 standard
columns (CHR/BP/SNP/CM) + a single C0_perm column. Pre-load baseline annot
once per chrom. Annotation generation should be ~3 sec/file.

For LD scores: parallel 4-way ldsc --l2 per chr.
For S-LDSC: parallel 4-way per permutation.

N_PERM = 50 (sufficient for null distribution given resource constraints).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P14E_BASELINE = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
P14E_FRQ = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_frq"
P14E_CLUSTER = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17d"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
np.random.seed(42)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17D OPTIMIZED: Empirical permutation null — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)
log(f"  N_PERM = {N_PERM}")


# ─── 1. HM3 MASTER WITH LD-BIN × MAF-BIN ───────────────────────────────────
log("\n[1] Building HM3 master with LD-bin × MAF-bin (one-pass)")
all_snps = []
all_base_annots = {}  # chr -> baseline annot DataFrame with CHR/BP/SNP/CM only

for chrom in range(1, 23):
    base_ld = pd.read_csv(P14E_BASELINE / f"baseline.{chrom}.l2.ldscore.gz",
                          sep="\t", compression="gzip", usecols=["CHR", "SNP", "BP", "baseL2"])
    frq = pd.read_csv(P14E_FRQ / f"1000G.EUR.QC.{chrom}.frq", delim_whitespace=True,
                       usecols=["SNP", "MAF"])
    merged = base_ld.merge(frq, on="SNP", how="inner")
    all_snps.append(merged)
    # Build minimal baseline annot DataFrame for this chr (CHR/BP/SNP/CM)
    bf = pd.read_csv(P14E_BASELINE / f"baseline.{chrom}.annot.gz",
                     sep="\t", compression="gzip", usecols=["CHR", "BP", "SNP", "CM"])
    all_base_annots[chrom] = bf
master = pd.concat(all_snps, ignore_index=True)
log(f"  HM3 master SNPs: {len(master):,}")

master["maf_bin"] = pd.qcut(master["MAF"], 10, labels=False, duplicates="drop")
master["l2_bin"]  = pd.qcut(master["baseL2"], 10, labels=False, duplicates="drop")
master["bin_id"]  = master["maf_bin"].astype(str) + "_" + master["l2_bin"].astype(str)
log(f"  Decile bin combinations: {master['bin_id'].nunique()}")


# ─── 2. LOAD PRIMARY C0 AND COMPUTE BIN DISTRIBUTION ──────────────────────
log("\n[2] Loading primary C0 cluster SNPs")
c0_snps = []
for chrom in range(1, 23):
    annot = pd.read_csv(P14E_CLUSTER / f"cluster.{chrom}.annot.gz", sep="\t",
                        compression="gzip", usecols=["CHR", "SNP", "BP", "C0"])
    c0_snps.append(annot[annot["C0"] == 1][["CHR", "SNP", "BP"]])
c0_df = pd.concat(c0_snps, ignore_index=True)
log(f"  Primary C0 SNPs: {len(c0_df):,}")

c0_merged = c0_df.merge(master[["SNP", "bin_id"]], on="SNP", how="inner")
log(f"  C0 SNPs in HM3 master: {len(c0_merged):,}")
c0_bin_counts = c0_merged["bin_id"].value_counts().to_dict()
log(f"  C0 distinct bins: {len(c0_bin_counts)}")


# ─── 3. PRE-INDEX MASTER BY BIN ────────────────────────────────────────────
log("\n[3] Pre-indexing master by bin")
bin_to_snps = {b: np.array(g["SNP"].values) for b, g in master.groupby("bin_id")}


# ─── 4. GENERATE PERMUTATION ANNOTATION FILES ──────────────────────────────
log(f"\n[4] Generating {N_PERM} permutation annot files ({N_PERM} × 22 chr = {N_PERM*22} files)")
start = datetime.now()

for p in range(N_PERM):
    perm_dir = OUT / f"perm_{p:03d}"
    perm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(p)  # reproducible per perm
    # Draw matched SNPs
    drawn = []
    for bin_id, n_needed in c0_bin_counts.items():
        pool = bin_to_snps.get(bin_id, np.array([]))
        if len(pool) >= n_needed:
            drawn.extend(rng.choice(pool, size=n_needed, replace=False).tolist())
        else:
            drawn.extend(pool.tolist())
            need = n_needed - len(pool)
            other = rng.choice(master["SNP"].values, size=need, replace=False)
            drawn.extend(other.tolist())
    drawn_set = set(drawn)

    # Write per-chr annot files (minimal 5-column format)
    for chrom in range(1, 23):
        bf = all_base_annots[chrom].copy()
        bf["C0_perm"] = bf["SNP"].astype(str).isin(drawn_set).astype(int)
        bf.to_csv(perm_dir / f"cluster_perm.{chrom}.annot.gz",
                  sep="\t", index=False, compression="gzip")
    if (p + 1) % 5 == 0:
        elapsed = (datetime.now() - start).total_seconds()
        rate = (p + 1) / elapsed
        eta = (N_PERM - (p + 1)) / rate
        log(f"  perm {p+1}/{N_PERM} done (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

log(f"\nDone. Annot files in {OUT}/perm_NNN/")
log(f"Total elapsed: {(datetime.now() - start).total_seconds():.0f}s")

with open(OUT / "P17d_optimized_log.md", "w") as f:
    f.write("# Phase 17D OPTIMIZED: Annotation build\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
