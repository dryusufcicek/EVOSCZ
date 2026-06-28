#!/usr/bin/env python3
"""
Phase 14p5b — Merge per-arm Wohns ages, analyze cluster-stratified distribution
================================================================================
After P14p5a SLURM array completes 39 chr arms, merge all per-arm TSVs,
join with cluster assignments + GEVA age, and answer:

  Q1. How does Wohns age distribute across C0/C1/C2 clusters?
       - If C0 still youngest, C2 still oldest by Wohns → cluster framework robust to age estimator
       - If pattern reverses or flattens → GEVA-circularity confound real

  Q2. What fraction of v10 C2 ("Old") variants have Wohns age > 60 kyr (pre-OOA AMH)?
       - Tests whether C2 contains genuine deep-time pre-OOA variants
       - If yes → C2 partially captures Crow-relevant pre-AMH-speciation signal
       - If no → C2 is "old within post-OOA" only

  Q3. Cluster × Wohns age statistical comparison
       - Pairwise KS D + Mann-Whitney U
       - Pre-registered: C0 vs C2 KS D > 0.30 → cluster structure robust

  Q4. GEVA vs Wohns concordance per cluster
       - Per cluster, Spearman correlation
       - If correlation differs by cluster → age estimator-dependent biases

Output:
  results/phase14p5/P14p5b_wohns_age_per_variant.parquet
  results/phase14p5/P14p5b_cluster_wohns_distribution.tsv
  results/phase14p5/P14p5b_cluster_wohns_tests.tsv
  results/phase14p5/P14p5b_geva_wohns_calibration.tsv
  results/phase14p5/P14p5b_pre_OOA_partition.tsv
  results/phase14p5/P14p5b_NARRATIVE.md
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime
import glob
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
DATA = Path((_SCRATCH + "/v11_data/phase14p5"))
OUT  = BASE / "results/phase14p5"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*70)
log("Phase 14p5b — Merge Wohns ages + cluster-stratified analysis")
log("="*70)

# Merge per-arm TSVs
log("[1] Merging per-arm Wohns age TSVs")
arm_files = sorted(glob.glob(str(DATA / "P14p5a_chr*.tsv")))
log(f"  arm files: {len(arm_files)}")
parts = []
for f in arm_files:
    d = pd.read_csv(f, sep="\t")
    if len(d) > 0:
        parts.append(d)
df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
log(f"  total Wohns-aged variants: {len(df):,}")

# Add cluster + GEVA + maf
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = df.merge(vm[["rsid","chr","pos","age_median_yr","maf"]], on="rsid", how="left")
df = df.merge(clu, on="rsid", how="left")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
log(f"  with cluster: {df['cluster'].notna().sum()}")
log(f"  with GEVA age: {df['age_median_yr'].notna().sum()}")

df.to_parquet(OUT / "P14p5b_wohns_age_per_variant.parquet")
log(f"  saved: P14p5b_wohns_age_per_variant.parquet")

# ── Q1+Q3: Cluster × Wohns age distribution ────────────────────────
log("\n[2] Q1+Q3: Cluster-stratified Wohns age distribution")
dist_rows = []
clu_df = df.dropna(subset=["cluster"]).copy()
clu_df["cluster"] = clu_df["cluster"].astype(int)
for c in sorted(clu_df["cluster"].unique()):
    sub = clu_df[clu_df["cluster"]==c]["wohns_midpoint_age_yr"].values
    dist_rows.append({
        "cluster": f"C{int(c)}",
        "n": len(sub),
        "median_wohns_age_yr": float(np.median(sub)),
        "q01": float(np.quantile(sub,0.01)),
        "q25": float(np.quantile(sub,0.25)),
        "q75": float(np.quantile(sub,0.75)),
        "q99": float(np.quantile(sub,0.99)),
        "max": float(np.max(sub)),
        "pct_age_lt_60kyr":   float((sub < 60_000).mean()*100),
        "pct_age_60_to_200kyr": float(((sub >= 60_000) & (sub < 200_000)).mean()*100),
        "pct_age_ge_200kyr":  float((sub >= 200_000).mean()*100),
        "pct_age_ge_500kyr":  float((sub >= 500_000).mean()*100),
    })
dist = pd.DataFrame(dist_rows)
dist.to_csv(OUT / "P14p5b_cluster_wohns_distribution.tsv", sep="\t", index=False)
log(dist.to_string(index=False))

# Pairwise tests
log("\n[3] Pairwise cluster comparisons (KS + MWU)")
pair_rows = []
clusters = sorted(clu_df["cluster"].unique())
for i, c1 in enumerate(clusters):
    for c2 in clusters[i+1:]:
        x = clu_df[clu_df["cluster"]==c1]["wohns_midpoint_age_yr"].values
        y = clu_df[clu_df["cluster"]==c2]["wohns_midpoint_age_yr"].values
        ks_D, ks_p = stats.ks_2samp(x, y)
        mwu_U, mwu_P = stats.mannwhitneyu(x, y, alternative="two-sided")
        pair_rows.append({
            "contrast": f"C{int(c1)}_vs_C{int(c2)}",
            "n1": len(x), "n2": len(y),
            "median_1": float(np.median(x)),
            "median_2": float(np.median(y)),
            "median_diff": float(np.median(x) - np.median(y)),
            "KS_D": float(ks_D),
            "KS_P": float(ks_p),
            "MWU_P": float(mwu_P),
        })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT / "P14p5b_cluster_wohns_tests.tsv", sep="\t", index=False)
log(pair_df.to_string(index=False))

# ── Q4: GEVA vs Wohns calibration per cluster ──────────────────────
log("\n[4] Q4: GEVA vs Wohns concordance per cluster")
calib_rows = []
both_valid = clu_df.dropna(subset=["age_median_yr","wohns_midpoint_age_yr"])
both_valid = both_valid[
    (both_valid["age_median_yr"] > 0) & (both_valid["wohns_midpoint_age_yr"] > 0)
]
for c in sorted(both_valid["cluster"].unique()):
    sub = both_valid[both_valid["cluster"]==c]
    if len(sub) < 5: continue
    r, p = stats.pearsonr(np.log10(sub["age_median_yr"]+1), np.log10(sub["wohns_midpoint_age_yr"]+1))
    rho, prho = stats.spearmanr(sub["age_median_yr"], sub["wohns_midpoint_age_yr"])
    calib_rows.append({
        "cluster": f"C{int(c)}",
        "n": len(sub),
        "Pearson_log_r": float(r),
        "Pearson_P": float(p),
        "Spearman_rho": float(rho),
        "Spearman_P": float(prho),
        "median_GEVA_yr": float(sub["age_median_yr"].median()),
        "median_Wohns_yr": float(sub["wohns_midpoint_age_yr"].median()),
        "Wohns_to_GEVA_ratio": float(sub["wohns_midpoint_age_yr"].median()/sub["age_median_yr"].median()),
    })
# Also pooled
sub = both_valid
r, p = stats.pearsonr(np.log10(sub["age_median_yr"]+1), np.log10(sub["wohns_midpoint_age_yr"]+1))
rho, prho = stats.spearmanr(sub["age_median_yr"], sub["wohns_midpoint_age_yr"])
calib_rows.append({
    "cluster": "all",
    "n": len(sub),
    "Pearson_log_r": float(r), "Pearson_P": float(p),
    "Spearman_rho": float(rho), "Spearman_P": float(prho),
    "median_GEVA_yr": float(sub["age_median_yr"].median()),
    "median_Wohns_yr": float(sub["wohns_midpoint_age_yr"].median()),
    "Wohns_to_GEVA_ratio": float(sub["wohns_midpoint_age_yr"].median()/sub["age_median_yr"].median()),
})
calib = pd.DataFrame(calib_rows)
calib.to_csv(OUT / "P14p5b_geva_wohns_calibration.tsv", sep="\t", index=False)
log(calib.to_string(index=False))

# ── Q2: Pre-OOA AMH-shared partition ──────────────────────────────
log("\n[5] Q2: Pre-OOA partition (Wohns age >= 60 kyr → pre-OOA candidate)")
partition_rows = []
for c in [0, 1, 2]:
    sub = clu_df[clu_df["cluster"]==c]
    pre_OOA = sub[sub["wohns_midpoint_age_yr"] >= 60_000]
    deep    = sub[sub["wohns_midpoint_age_yr"] >= 200_000]
    very_deep = sub[sub["wohns_midpoint_age_yr"] >= 500_000]
    partition_rows.append({
        "cluster": f"C{c}",
        "n_total": len(sub),
        "n_pre_OOA": len(pre_OOA),
        "pct_pre_OOA": float(len(pre_OOA)/len(sub)*100) if len(sub) else np.nan,
        "n_deep_amh": len(deep),
        "pct_deep_amh": float(len(deep)/len(sub)*100) if len(sub) else np.nan,
        "n_very_deep": len(very_deep),
        "pct_very_deep": float(len(very_deep)/len(sub)*100) if len(sub) else np.nan,
    })
part = pd.DataFrame(partition_rows)
part.to_csv(OUT / "P14p5b_pre_OOA_partition.tsv", sep="\t", index=False)
log(part.to_string(index=False))

# ── Narrative ──────────────────────────────────────────────────────
with open(OUT / "P14p5b_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p5b — Wohns ages × cluster\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write("## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p5b complete.")
