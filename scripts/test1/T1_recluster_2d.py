#!/usr/bin/env python3
"""
T1_recluster_2d.py — concern 4: which of the 3 GMM axes is load-bearing?
Re-cluster on 2 of {log_age, brain_spec, abs_ihs}, dropping one, on the SAME 4918
variants (3-feature dropna set) for comparability. Mut clock (justified for SCZ
variants). k=3, full cov, random_state=42, n_init=10, sort by first feature.

combo: age_brain (drop iHS) | ihs_brain (drop age) | age_ihs (drop brain = published "dissolves" sanity)
Outputs recluster/assign_2d_{combo}_k3.tsv.gz + per-cluster centroids (all 3 features) for interpretation.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

combo = sys.argv[1]
FEAT = {"age_brain": ["log_age", "brain_spec"],
        "ihs_brain": ["abs_ihs", "brain_spec"],
        "age_ihs":   ["log_age", "abs_ihs"]}[combo]
B = Path(_ROOT)
OUT = Path((_SCRATCH + "/test1_age_conditioning/recluster"))
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

m = pd.read_parquet(B / "results/phase11/variant_master_v3.parquet")
m["log_age"] = np.where((m["age_median_yr"].notna()) & (m["age_median_yr"] > 0), np.log10(m["age_median_yr"]), np.nan)
m["b_logp"] = np.where((m["gtex_brain_minp"].notna()) & (m["gtex_brain_minp"] > 0), -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where((m["gtex_blood_minp"].notna()) & (m["gtex_blood_minp"] > 0), -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs()
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)
m_nm = m[~in_mapt].copy()

mix3 = m_nm[["log_age", "brain_spec", "abs_ihs"]].dropna()   # SAME 4918 set as 3D
idx = mix3.index
mix = m_nm.loc[idx, FEAT]
scaler = StandardScaler().fit(mix.values)
z = scaler.transform(mix.values)
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10).fit(z)
lab = gmm.predict(z)
means = scaler.inverse_transform(gmm.means_)
order = np.argsort(means[:, 0])                 # sort by FIRST feature of the combo
remap = {old: new for new, old in enumerate(order)}
labs = np.array([remap[l] for l in lab])
out = m_nm.loc[idx, ["rsid"]].copy()
out["cluster"] = labs
out.to_csv(OUT / f"assign_2d_{combo}_k3.tsv.gz", sep="\t", index=False, compression="gzip")

full = m_nm.loc[idx, ["log_age", "brain_spec", "abs_ihs"]].copy()
full["c"] = labs
print(f"combo={combo} features={FEAT}: n={len(idx)} (same 4918 set)", flush=True)
for c in range(3):
    s = full[full.c == c]
    print(f"  C{c} (n={len(s)}): mean_age={10**s.log_age.mean():.0f}gen  brain_spec={s.brain_spec.mean():.3f}  |iHS|={s.abs_ihs.mean():.3f}", flush=True)
print("RECLUSTER2D_DONE", flush=True)
