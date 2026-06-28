#!/usr/bin/env python3
"""Phase 17D: Empirical permutation null for C0 (Young) cluster enrichment.

Robustness check #5 (permutation portion): C0 = 0.029% of HapMap3 SNPs is below
the Tashman 2021 0.5% threshold for which the LDSC block-jackknife is reliable.
Tashman 2021 recommends an empirical permutation null built from matched-LD-MAF
SNP draws to validate the observed enrichment.

This script:
  1. Constructs an LD-bin × MAF-bin lookup over the HapMap3 baseline-LD reference
     SNPs, using deciles of baseline L2 score and deciles of EUR allele frequency.
  2. Reads the primary C0 cluster annotation (1,742 SNPs across 22 chr) and
     computes their LD-bin × MAF-bin distribution.
  3. Generates `n_perm` permutation draws where each draw samples 1,742 control
     SNPs from the baseline reference, matched to the original C0 distribution
     in LD-bin × MAF-bin.
  4. Writes per-chrom single-annotation files for each permutation under
     phase17d/perm_NNN/cluster_perm.{chr}.annot.gz.

LD-score computation + S-LDSC execution handled by P17d_run_perm.sh in parallel.

Output:
  results/phase17d/perm_master.tsv.gz   — observed + null SNP master with bin
  results/phase17d/perm_NNN/            — 100 permutation directories
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import gzip
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P14E_BASELINE = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
P14E_FRQ = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_frq"
P14E_CLUSTER = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17d"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 100
N_C0_REF = 1742  # primary C0 cluster size in EUR LD reference
np.random.seed(42)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17D: Empirical permutation null — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)
log(f"  n_perm = {N_PERM}")

# ─── 1. BUILD HM3 SNP MASTER (CHR, BP, SNP, L2, MAF) ───────────────────────
log("\n[1] Building HM3 SNP master with baseline L2 and EUR MAF")
all_snps = []
for chrom in range(1, 23):
    base_ld = pd.read_csv(P14E_BASELINE / f"baseline.{chrom}.l2.ldscore.gz",
                          sep="\t", compression="gzip", usecols=["CHR", "SNP", "BP", "baseL2"])
    frq = pd.read_csv(P14E_FRQ / f"1000G.EUR.QC.{chrom}.frq", delim_whitespace=True,
                       usecols=["SNP", "MAF"])
    merged = base_ld.merge(frq, on="SNP", how="inner")
    all_snps.append(merged)
master = pd.concat(all_snps, ignore_index=True)
log(f"  Total HM3 SNPs with L2 and MAF: {len(master):,}")

# Decile binning over the master (10×10 = 100 bins)
master["maf_bin"] = pd.qcut(master["MAF"], 10, labels=False, duplicates="drop")
master["l2_bin"]  = pd.qcut(master["baseL2"], 10, labels=False, duplicates="drop")
master["bin_id"]  = master["maf_bin"].astype(str) + "_" + master["l2_bin"].astype(str)
log(f"  Decile-bin combinations covered: {master['bin_id'].nunique()}")

# ─── 2. LOAD PRIMARY C0 ANNOTATION AND COMPUTE ITS BIN DISTRIBUTION ────────
log("\n[2] Loading primary C0 cluster SNPs and computing bin distribution")
c0_snps = []
for chrom in range(1, 23):
    annot = pd.read_csv(P14E_CLUSTER / f"cluster.{chrom}.annot.gz", sep="\t",
                        compression="gzip", usecols=["CHR", "SNP", "BP", "C0"])
    c0_snps.append(annot[annot["C0"] == 1][["CHR", "SNP", "BP"]])
c0_df = pd.concat(c0_snps, ignore_index=True)
log(f"  Primary C0 SNPs: {len(c0_df):,}")

c0_merged = c0_df.merge(master[["SNP", "bin_id"]], on="SNP", how="inner")
log(f"  C0 SNPs matched to HM3 master: {len(c0_merged):,}")
c0_bin_dist = c0_merged["bin_id"].value_counts().to_dict()
log(f"  C0 distinct bins: {len(c0_bin_dist)}")

# Save observed C0 master for reference
c0_merged.to_csv(OUT / "c0_observed_bins.tsv", sep="\t", index=False)


# ─── 3. GENERATE PERMUTATION DRAWS ─────────────────────────────────────────
log(f"\n[3] Generating {N_PERM} matched-LD-MAF permutation draws")

# Group HM3 master by bin for efficient sampling
bin_groups = master.groupby("bin_id")["SNP"].apply(list).to_dict()
total_drawn = 0
for p in range(N_PERM):
    perm_dir = OUT / f"perm_{p:03d}"
    perm_dir.mkdir(parents=True, exist_ok=True)
    drawn = []
    for bin_id, n_needed in c0_bin_dist.items():
        pool = bin_groups.get(bin_id, [])
        if len(pool) >= n_needed:
            draw = np.random.choice(pool, size=n_needed, replace=False)
        else:
            draw = np.array(pool)
            need = n_needed - len(draw)
            # Top up from neighbouring bins (with replacement penalty avoided)
            other_pool = master["SNP"].sample(n=need, random_state=p, replace=False).values
            draw = np.concatenate([draw, other_pool])
        drawn.extend(draw.tolist())
    drawn = list(set(drawn))  # dedupe
    drawn_set = set(drawn)

    # Write per-chr annot files (single C0_perm column)
    for chrom in range(1, 23):
        base_ld = pd.read_csv(P14E_BASELINE / f"baseline.{chrom}.l2.ldscore.gz",
                              sep="\t", compression="gzip", usecols=["CHR", "SNP", "BP"])
        # Add CM column matching original annotation format
        base_annot = pd.read_csv(P14E_BASELINE / f"baseline.{chrom}.annot.gz",
                                 sep="\t", compression="gzip", usecols=["CHR", "BP", "SNP", "CM"])
        base_annot["C0_perm"] = base_annot["SNP"].astype(str).isin(drawn_set).astype(int)
        base_annot.to_csv(perm_dir / f"cluster_perm.{chrom}.annot.gz",
                          sep="\t", index=False, compression="gzip")
    total_drawn += len(drawn)
    if (p + 1) % 10 == 0:
        log(f"  perm {p+1}/{N_PERM} done (n_C0_perm avg = {total_drawn / (p+1):.0f})")

log(f"\nDone. Permutation directories under {OUT}/perm_NNN/")
log(f"Next: P17d_run_perm.sh computes LD scores + S-LDSC per perm.")

with open(OUT / "P17d_perm_build_log.md", "w") as f:
    f.write("# Phase 17D: Permutation annotation build\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
