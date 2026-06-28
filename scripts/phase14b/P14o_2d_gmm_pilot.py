#!/usr/bin/env python3
"""
Phase 14o: Option-A pilot — 2D GMM (log_age × |iHS|) clustering with
brain_spec REMOVED from cluster definition.

Rationale: brain_spec requires GTEx brain ∩ blood eQTL data, which restricts
GMM input to ~26% of credible-set variants (4,918 of 19,024 non-MAPT). By
clustering on age + |iHS| only, GMM coverage rises to ~85%, and the cluster
definition becomes purely evolutionary (no regulatory annotation circularity).
Brain-regulatory engagement of each cluster then becomes a downstream
characterization rather than a definitional input.

Output:
  results/phase14b/P14o_2d_assignments.tsv.gz  (rsid, cluster: 0=Young, 1+=NotYoung)
  results/phase14b/P14o_2d_pilot_log.md
  results/phase14b/P14o_2d_overlap_with_3d.tsv

This is a PILOT script — does NOT modify existing 3D infrastructure.
"""
import warnings
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14b"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14o: Option-A 2D GMM pilot — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

np.random.seed(42)

m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)
m_nm = m[~in_mapt].copy()

# 2D feature space: log_age × |iHS|
mix_data = m_nm[["log_age", "abs_ihs"]].dropna()
indices = mix_data.index
mix_arr = mix_data.values

log(f"\nTotal variants: {len(m):,}")
log(f"Non-MAPT: {len(m_nm):,}")
log(f"2D-fittable (log_age + |iHS| both present): {len(mix_arr):,}")
log(f"Coverage relative to non-MAPT credible-set: {100*len(mix_arr)/len(m_nm):.1f}%")
log(f"Coverage relative to total credible-set:    {100*len(mix_arr)/len(m):.1f}%")

scaler = StandardScaler()
mix_z = scaler.fit_transform(mix_arr)

# BIC scan for k
log("\n" + "=" * 72)
log("[1] BIC scan for k = 1..6 (2D feature space)")
log("=" * 72)

bic_results = []
for k in [1, 2, 3, 4, 5, 6]:
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(mix_z)
    bic = gmm.bic(mix_z)
    aic = gmm.aic(mix_z)
    bic_results.append({"k": k, "BIC": bic, "AIC": aic})
    log(f"  k={k}: BIC={bic:,.1f}, AIC={aic:,.1f}")

df_bic = pd.DataFrame(bic_results)
df_bic["delta_BIC_vs_k1"] = df_bic["BIC"] - df_bic.loc[0, "BIC"]
df_bic.to_csv(OUT / "P14o_2d_bic.tsv", sep="\t", index=False)

# Use k=3 for direct comparison with 3D primary
USE_K = 3
log(f"\nUsing k={USE_K} (matches 3D primary k for direct comparison)")

# Fit final GMM
gmm = GaussianMixture(n_components=USE_K, covariance_type="full",
                      random_state=42, n_init=10)
gmm.fit(mix_z)
labels = gmm.predict(mix_z)

# Sort cluster IDs by mean unscaled log_age (ascending) → cluster 0 = Young
unscaled_means = scaler.inverse_transform(gmm.means_)
sort_idx = np.argsort(unscaled_means[:, 0])
remap = {old: new for new, old in enumerate(sort_idx)}
labels_sorted = np.array([remap[l] for l in labels])

log("\n" + "=" * 72)
log(f"[2] 2D GMM cluster centroids (k={USE_K}, sorted by log_age):")
log("=" * 72)
for new_id, old_id in enumerate(sort_idx):
    n_in = int((labels_sorted == new_id).sum())
    age_centroid = unscaled_means[old_id, 0]
    ihs_centroid = unscaled_means[old_id, 1]
    log(f"  cluster {new_id}: n={n_in:5d}; mean log10_age={age_centroid:.3f}; |iHS|={ihs_centroid:.3f}")

# Save assignments
m_nm_local = m_nm.copy()
m_nm_local["cluster"] = np.nan
m_nm_local.loc[indices, "cluster"] = labels_sorted
out_df = m_nm_local[["rsid", "cluster"]].dropna(subset=["rsid"]).copy()
out_df["cluster"] = out_df["cluster"].astype("Int64")
out_path = OUT / "P14o_2d_assignments.tsv.gz"
out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
log(f"\nSaved {len(out_df):,} 2D assignments → {out_path.name}")

# Overlap with 3D primary
log("\n" + "=" * 72)
log("[3] Overlap between 2D Young cluster and 3D primary Young cluster")
log("=" * 72)

assign_3d_path = OUT / "P14b_v3_cluster_assignments.tsv.gz"
if assign_3d_path.exists():
    a3d = pd.read_csv(assign_3d_path, sep="\t")
    a3d = a3d[a3d["cluster"].notna()].copy()
    a3d["cluster"] = a3d["cluster"].astype(int)

    # Merge on rsid
    merged = out_df.merge(a3d, on="rsid", how="outer", suffixes=("_2d", "_3d"))
    merged = merged[merged["cluster_2d"].notna() & merged["cluster_3d"].notna()].copy()
    log(f"  Variants in both 2D and 3D GMM: {len(merged):,}")

    # Cross-tab
    log("\n  Cross-tabulation (rows = 2D cluster, cols = 3D cluster):")
    ct = pd.crosstab(merged["cluster_2d"], merged["cluster_3d"], margins=True)
    log("  " + ct.to_string().replace("\n", "\n  "))

    young_2d = set(out_df.loc[out_df["cluster"] == 0, "rsid"].astype(str))
    young_3d = set(a3d.loc[a3d["cluster"] == 0, "rsid"].astype(str))

    log(f"\n  Young cluster (cluster 0) overlap analysis:")
    log(f"    Young in 2D only:        {len(young_2d - young_3d):5d}")
    log(f"    Young in 3D only:        {len(young_3d - young_2d):5d}")
    log(f"    Young in BOTH:           {len(young_2d & young_3d):5d}")
    log(f"    Young in either:         {len(young_2d | young_3d):5d}")
    log(f"\n    Jaccard(Young 2D, Young 3D) = {len(young_2d & young_3d)/len(young_2d | young_3d):.3f}")
    log(f"    Recall: 3D Young captured by 2D Young = {len(young_2d & young_3d)/max(len(young_3d),1):.3f}")
    log(f"    Precision: 2D Young captured by 3D Young = {len(young_2d & young_3d)/max(len(young_2d),1):.3f}")
else:
    log(f"  ! 3D assignments not found at {assign_3d_path}")

with open(OUT / "P14o_2d_pilot_log.md", "w") as f:
    f.write("# Phase 14o: 2D GMM Option-A Pilot\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
