#!/usr/bin/env python3
"""
Phase 14p extended: 1000 bootstrap stability analysis at k=3 for both 3D and 2D.
Reports mean/SD/min/max of ARI distribution + percentile CIs.
"""
import warnings
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14b"

print(f"Phase 14p extended: 1000-bootstrap stability — {datetime.now().strftime('%H:%M:%S')}")
np.random.seed(42)

m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["b_logp"] = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                        -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                         -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)
m_nm = m[~in_mapt].copy()


def stability_1000(features, label, n_boot=1000):
    print(f"\n[{label}] features={features}, n_boot={n_boot}")
    mix_data = m_nm[features].dropna()
    mix_arr = mix_data.values
    print(f"  n={len(mix_arr):,}")

    scaler = StandardScaler()
    mix_z = scaler.fit_transform(mix_arr)

    # Reference fit
    gmm_ref = GaussianMixture(n_components=3, covariance_type="full",
                              random_state=42, n_init=10)
    gmm_ref.fit(mix_z)
    ref_labels = gmm_ref.predict(mix_z)

    aris = np.zeros(n_boot)
    n = len(mix_z)
    for i in range(n_boot):
        rng = np.random.RandomState(i)
        idx = rng.choice(n, n, replace=True)
        boot_z = mix_z[idx]
        gmm_b = GaussianMixture(n_components=3, covariance_type="full",
                                 random_state=i, n_init=5)
        gmm_b.fit(boot_z)
        boot_pred = gmm_b.predict(mix_z)
        aris[i] = adjusted_rand_score(ref_labels, boot_pred)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_boot}: running mean ARI = {aris[:i+1].mean():.4f}", flush=True)

    print(f"\n  Final: mean={aris.mean():.4f} SD={aris.std():.4f}")
    print(f"  Range: [{aris.min():.4f}, {aris.max():.4f}]")
    print(f"  95% percentile CI: [{np.percentile(aris, 2.5):.4f}, {np.percentile(aris, 97.5):.4f}]")
    print(f"  Mean SE = {aris.std()/np.sqrt(n_boot):.5f}")
    print(f"  ARI > 0.5: {int((aris > 0.5).sum())}/{n_boot} ({100*np.mean(aris>0.5):.1f}%)")
    print(f"  ARI > 0.8: {int((aris > 0.8).sum())}/{n_boot} ({100*np.mean(aris>0.8):.1f}%)")
    print(f"  ARI > 0.9: {int((aris > 0.9).sum())}/{n_boot} ({100*np.mean(aris>0.9):.1f}%)")
    return aris

ari_3d = stability_1000(["log_age", "brain_spec", "abs_ihs"], "3D primary")
np.savetxt(OUT / "P14p_ari_3d_1000.txt", ari_3d)

ari_2d = stability_1000(["log_age", "abs_ihs"], "2D sensitivity")
np.savetxt(OUT / "P14p_ari_2d_1000.txt", ari_2d)

print(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")
