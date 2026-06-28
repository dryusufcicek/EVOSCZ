#!/usr/bin/env python3
"""
Phase 14p: Comprehensive GMM model-selection battery.

Computes for k=1..8 in BOTH 3D and 2D feature spaces:
  - BIC (Bayesian information criterion)
  - AIC (Akaike information criterion)
  - ICL (Integrated Completed Likelihood; BIC + entropy penalty)
  - Silhouette score (cluster cohesion + separation)
  - Calinski-Harabasz score (between/within variance ratio)
  - Davies-Bouldin score (avg cluster similarity, lower=better)
  - Log-likelihood

Plus stability analysis: 100 bootstrap re-fits at the chosen k, report
mean ARI (adjusted rand index) of cluster assignments vs reference.

Output: results/phase14b/P14p_model_selection.tsv + .md narrative
"""
import warnings
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering  # for silhouette compatibility
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
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

log(f"Phase 14p: Model selection battery — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)

np.random.seed(42)

# Load data
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


def compute_icl(gmm, X):
    """Integrated Completed Likelihood: BIC + entropy of posterior probs."""
    log_resp = gmm._estimate_log_prob_resp(X)[1]  # log posterior probabilities
    resp = np.exp(log_resp)
    # Entropy: -sum p log p
    entropy = -np.sum(resp * np.log(np.clip(resp, 1e-300, None)))
    bic = gmm.bic(X)
    icl = bic + 2 * entropy
    return icl, entropy


def run_battery(features_used, label, df_nm):
    log(f"\n{'='*78}\n[{label}] Feature space: {features_used}\n{'='*78}")
    mix_data = df_nm[features_used].dropna()
    mix_arr = mix_data.values
    log(f"  Variants with all features: {len(mix_arr):,}")
    log(f"  Coverage of non-MAPT credible-set: {100*len(mix_arr)/len(df_nm):.1f}%")

    scaler = StandardScaler()
    mix_z = scaler.fit_transform(mix_arr)

    rows = []
    for k in range(1, 9):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=10)
        gmm.fit(mix_z)
        labels = gmm.predict(mix_z)

        bic = gmm.bic(mix_z)
        aic = gmm.aic(mix_z)
        ll = gmm.score(mix_z) * len(mix_z)

        # ICL
        if k > 1:
            icl, entropy = compute_icl(gmm, mix_z)
        else:
            icl, entropy = bic, 0.0

        # Silhouette / CH / DB only defined for k >= 2
        if k >= 2:
            sil = silhouette_score(mix_z, labels, sample_size=min(5000, len(mix_z)),
                                    random_state=42)
            ch = calinski_harabasz_score(mix_z, labels)
            db = davies_bouldin_score(mix_z, labels)
        else:
            sil = ch = db = np.nan

        rows.append({
            "feature_space": label, "k": k,
            "BIC": bic, "AIC": aic, "ICL": icl, "log_likelihood": ll,
            "entropy": entropy, "silhouette": sil,
            "calinski_harabasz": ch, "davies_bouldin": db,
        })
        log(f"  k={k}: BIC={bic:,.0f} AIC={aic:,.0f} ICL={icl:,.0f} "
            f"sil={sil:.3f} CH={ch:,.0f} DB={db:.3f}" if k >= 2 else
            f"  k={k}: BIC={bic:,.0f} AIC={aic:,.0f} LL={ll:,.0f}")

    df = pd.DataFrame(rows)
    return df, mix_z, mix_data.index


def stability_analysis(mix_z, k, label, n_boot=100):
    """Bootstrap stability at chosen k; return mean ARI vs reference fit."""
    log(f"\n[{label}] Stability check — k={k}, {n_boot} bootstraps")

    # Reference fit on full data
    gmm_ref = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=10)
    gmm_ref.fit(mix_z)
    ref_labels = gmm_ref.predict(mix_z)

    n = len(mix_z)
    aris = []
    for boot_seed in range(n_boot):
        rng = np.random.RandomState(boot_seed)
        idx = rng.choice(n, n, replace=True)
        boot_z = mix_z[idx]
        gmm_boot = GaussianMixture(n_components=k, covariance_type="full",
                                    random_state=boot_seed, n_init=5)
        gmm_boot.fit(boot_z)
        # Predict on FULL original data using bootstrap-fitted model
        boot_pred = gmm_boot.predict(mix_z)
        ari = adjusted_rand_score(ref_labels, boot_pred)
        aris.append(ari)
    aris = np.array(aris)
    log(f"  ARI: mean={aris.mean():.4f}, std={aris.std():.4f}, "
        f"min={aris.min():.4f}, max={aris.max():.4f}")
    log(f"  ARI > 0.5: {int((aris > 0.5).sum())}/{n_boot} ({100*np.mean(aris>0.5):.0f}%)")
    log(f"  ARI > 0.8: {int((aris > 0.8).sum())}/{n_boot} ({100*np.mean(aris>0.8):.0f}%)")
    return aris


# Run for both 3D and 2D
df_3d, z_3d, _ = run_battery(["log_age", "brain_spec", "abs_ihs"], "3D", m_nm)
df_2d, z_2d, _ = run_battery(["log_age", "abs_ihs"], "2D", m_nm)

# Combine and save
df_all = pd.concat([df_3d, df_2d], ignore_index=True)
df_all.to_csv(OUT / "P14p_model_selection.tsv", sep="\t", index=False)
log(f"\n  Wrote {OUT / 'P14p_model_selection.tsv'}")

# Stability at k=3 for both
log("\n" + "="*78)
log("STABILITY ANALYSIS")
log("="*78)
ari_3d = stability_analysis(z_3d, 3, "3D")
ari_2d = stability_analysis(z_2d, 3, "2D")
np.savetxt(OUT / "P14p_ari_3d_k3.txt", ari_3d)
np.savetxt(OUT / "P14p_ari_2d_k3.txt", ari_2d)

# Best-k summary per metric
log("\n" + "="*78)
log("BEST k BY EACH METRIC (lower BIC/AIC/ICL/DB; higher sil/CH)")
log("="*78)
for label, df in [("3D", df_3d), ("2D", df_2d)]:
    log(f"\n{label}:")
    log(f"  Min BIC: k={int(df.loc[df['BIC'].idxmin(), 'k'])}")
    log(f"  Min AIC: k={int(df.loc[df['AIC'].idxmin(), 'k'])}")
    log(f"  Min ICL: k={int(df.loc[df['ICL'].idxmin(), 'k'])}")
    df_k2plus = df[df['k'] >= 2]
    log(f"  Max silhouette: k={int(df_k2plus.loc[df_k2plus['silhouette'].idxmax(), 'k'])}")
    log(f"  Max Calinski-Harabasz: k={int(df_k2plus.loc[df_k2plus['calinski_harabasz'].idxmax(), 'k'])}")
    log(f"  Min Davies-Bouldin: k={int(df_k2plus.loc[df_k2plus['davies_bouldin'].idxmin(), 'k'])}")

# Save log
with open(OUT / "P14p_model_selection_log.md", "w") as f:
    f.write("# Phase 14p: Comprehensive Model Selection\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
