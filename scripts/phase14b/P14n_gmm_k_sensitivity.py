#!/usr/bin/env python3
"""
Phase 14n: GMM k-sensitivity check (k=2, k=3, k=4).

Tests whether the SCZ heritability concentration in the "Young" GMM cluster
is robust to the choice of k. Re-fits GMM with k=2 and k=4 (alongside the
primary k=3 baseline), identifies the Young cluster (lowest mean unscaled
log_age) at each k, and outputs cluster assignments suitable for downstream
annotation building (P14e_build_cluster_annot.py).

Output files:
  results/phase14b/P14n_assignments_k2.tsv.gz  (rsid, cluster: 0=Young, 1=NotYoung)
  results/phase14b/P14n_assignments_k4.tsv.gz  (rsid, cluster: 0=Young, 1/2/3=NotYoung)
  results/phase14b/P14n_k_sensitivity_log.md

Usage:
  python3 P14n_gmm_k_sensitivity.py
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

log(f"Phase 14n: GMM k-sensitivity — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

np.random.seed(42)

m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["b_logp"]  = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                          -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                          -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)

m_nm = m[~in_mapt].copy()
mix_data = m_nm[["log_age", "brain_spec", "abs_ihs"]].dropna()
indices = mix_data.index
mix_arr = mix_data.values

log(f"\nVariants for mixture (no MAPT, all 3 features): {len(mix_arr)}")

scaler = StandardScaler()
mix_z = scaler.fit_transform(mix_arr)

for K in [2, 3, 4]:
    log("\n" + "=" * 72)
    log(f"[k={K}] Fitting GMM with k={K} components")
    log("=" * 72)

    gmm = GaussianMixture(n_components=K, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(mix_z)
    cluster_labels = gmm.predict(mix_z)

    # Sort cluster IDs by mean unscaled log_age (ascending) so cluster 0 = Young
    unscaled_means = scaler.inverse_transform(gmm.means_)
    sort_idx = np.argsort(unscaled_means[:, 0])
    remap = {old: new for new, old in enumerate(sort_idx)}
    cluster_labels_sorted = np.array([remap[l] for l in cluster_labels])

    # Per-cluster summary
    log(f"\n  Sorted cluster centroids (by log_age, ascending):")
    for new_id, old_id in enumerate(sort_idx):
        n_in_cluster = int((cluster_labels_sorted == new_id).sum())
        mean_log_age = unscaled_means[old_id, 0]
        mean_brain_spec = unscaled_means[old_id, 1]
        mean_abs_ihs = unscaled_means[old_id, 2]
        log(f"    cluster {new_id}: n={n_in_cluster:5d}; "
            f"mean log10_age={mean_log_age:.3f}; brain_spec={mean_brain_spec:.3f}; "
            f"|iHS|={mean_abs_ihs:.3f}")

    # Save assignments
    m_nm_local = m_nm.copy()
    m_nm_local["cluster"] = np.nan
    m_nm_local.loc[indices, "cluster"] = cluster_labels_sorted

    out_df = m_nm_local[["rsid", "cluster"]].dropna(subset=["rsid"]).copy()
    out_df["cluster"] = out_df["cluster"].astype("Int64")
    out_path = OUT / f"P14n_assignments_k{K}.tsv.gz"
    out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    log(f"\n  Saved {len(out_df):,} assignments → {out_path.name}")

    n_young = int((out_df["cluster"] == 0).sum())
    log(f"  Young cluster (cluster 0): {n_young:,} variants")

# Save log
with open(OUT / "P14n_k_sensitivity_log.md", "w") as f:
    f.write("# Phase 14n: GMM k-sensitivity\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
