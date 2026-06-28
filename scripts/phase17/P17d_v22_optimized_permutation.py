#!/usr/bin/env python3
"""Phase 17D v22: OPTIMIZED empirical permutation null for C0 (Young) enrichment
under baseline-LD v2.2 (Gazal 2017; 97 annotations).

Differs from the v1.2 version (P17d_optimized_permutation.py) only in:
  - HM3 master is binned on baseline-LD v2.2 L2 scores (not v1.2 baseL2)
  - Annot template is taken from baselineLD.{chr}.annot.gz (not baseline.{chr}.annot.gz)
  - Output goes to results/phase17d_v22/ (not results/phase17d/)

Strategy: Generate annotation files efficiently by only writing the 4 standard
columns (CHR/BP/SNP/CM) + a single C0_perm column. Pre-load baseline annot
once per chrom. Annotation generation should be ~3 sec/file.

For LD scores: parallel 4-way ldsc --l2 per chr (handled by run_perms.sh).
For S-LDSC: parallel 4-way per permutation (handled by run_perms.sh).

N_PERM = 15 by default, matching the manuscript description (§E3).
Usage:
  python3 P17d_v22_optimized_permutation.py [N_PERM]
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
BASELINE_V22 = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"
P14E_FRQ = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_frq"
P14E_CLUSTER = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17d_v22"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 15
np.random.seed(42)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17D v22 OPTIMIZED: Empirical permutation null — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)
log(f"  Baseline-LD model: v2.2 (Gazal 2017; 97 annotations)")
log(f"  Baseline path:    {BASELINE_V22}")
log(f"  Cluster annot:    {P14E_CLUSTER}")
log(f"  Output:           {OUT}")
log(f"  N_PERM = {N_PERM}")

if not BASELINE_V22.exists():
    log(f"ERROR: baseline-LD v2.2 directory not found: {BASELINE_V22}")
    log("Download from Zenodo 10515792 first (see README_v22_perm.md).")
    sys.exit(1)


# ─── 1. HM3 MASTER WITH LD-BIN (v2.2 baseL2) × MAF-BIN ─────────────────────
log("\n[1] Building HM3 master with v2.2 L2-bin × MAF-bin")
all_snps = []
all_base_annots = {}  # chr -> baseline annot DataFrame with CHR/BP/SNP/CM only

for chrom in range(1, 23):
    # v2.2 L2 ldscore: total LD score column is named 'baseL2' in Gazal 2017 release
    base_ld = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.l2.ldscore.gz",
                          sep="\t", compression="gzip")
    # The total-LD-score column in Gazal 2017 baseline-LD v2.2 is 'baseL2'
    # (sanity check that it exists; if absent, take the first L2 column)
    if "baseL2" in base_ld.columns:
        l2_col = "baseL2"
    else:
        l2_cols = [c for c in base_ld.columns if c.endswith("L2")]
        if not l2_cols:
            log(f"ERROR: no L2 column in baselineLD.{chrom}.l2.ldscore.gz")
            sys.exit(1)
        l2_col = l2_cols[0]
        log(f"  WARN: 'baseL2' not present; using '{l2_col}'")
    base_ld = base_ld[["CHR", "SNP", "BP", l2_col]].rename(columns={l2_col: "baseL2_v22"})

    frq = pd.read_csv(P14E_FRQ / f"1000G.EUR.QC.{chrom}.frq",
                      delim_whitespace=True, usecols=["SNP", "MAF"])
    merged = base_ld.merge(frq, on="SNP", how="inner")
    all_snps.append(merged)

    # Minimal baseline annot template for this chr (CHR/BP/SNP/CM)
    bf = pd.read_csv(BASELINE_V22 / f"baselineLD.{chrom}.annot.gz",
                     sep="\t", compression="gzip", usecols=["CHR", "BP", "SNP", "CM"])
    all_base_annots[chrom] = bf

master = pd.concat(all_snps, ignore_index=True)
log(f"  HM3 master SNPs (v2.2 baseline): {len(master):,}")

master["maf_bin"] = pd.qcut(master["MAF"], 10, labels=False, duplicates="drop")
master["l2_bin"]  = pd.qcut(master["baseL2_v22"], 10, labels=False, duplicates="drop")
master["bin_id"]  = master["maf_bin"].astype(str) + "_" + master["l2_bin"].astype(str)
log(f"  Decile bin combinations: {master['bin_id'].nunique()}")


# ─── 2. LOAD PRIMARY C0 AND COMPUTE BIN DISTRIBUTION ──────────────────────
log("\n[2] Loading primary C0 cluster SNPs (no MAPT)")
c0_snps = []
for chrom in range(1, 23):
    annot = pd.read_csv(P14E_CLUSTER / f"cluster.{chrom}.annot.gz", sep="\t",
                        compression="gzip", usecols=["CHR", "SNP", "BP", "C0"])
    c0_snps.append(annot[annot["C0"] == 1][["CHR", "SNP", "BP"]])
c0_df = pd.concat(c0_snps, ignore_index=True)
log(f"  Primary C0 SNPs (cluster annot): {len(c0_df):,}")

c0_merged = c0_df.merge(master[["SNP", "bin_id"]], on="SNP", how="inner")
log(f"  C0 SNPs in v2.2 HM3 master: {len(c0_merged):,}")
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
    if (p + 1) % 5 == 0 or (p + 1) == N_PERM:
        elapsed = (datetime.now() - start).total_seconds()
        rate = (p + 1) / elapsed
        eta = (N_PERM - (p + 1)) / rate if (p + 1) < N_PERM else 0
        log(f"  perm {p+1}/{N_PERM} done (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

log(f"\nDone. Annot files in {OUT}/perm_NNN/")
log(f"Total elapsed: {(datetime.now() - start).total_seconds():.0f}s")

with open(OUT / "P17d_v22_optimized_log.md", "w") as f:
    f.write("# Phase 17D v22 OPTIMIZED: Annotation build\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
