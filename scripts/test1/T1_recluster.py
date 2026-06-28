#!/usr/bin/env python3
"""
T1_recluster.py — re-fit the 3D GMM with a chosen allele-age CLOCK, replicating
phase14b/P14b_v3_mixture_clusters.py EXACTLY except the age feature's clock.

  clock=mut : age_median_yr (Mut/Combined) = PUBLISHED -> MUST reproduce the
              published P14b_v3 clusters (built-in sanity concordance check).
  clock=jnt : AgeMedian_Jnt (same Combined>TGP>SGDP source selection) =
              the methodologically-preferred clock (U4a/U4a_filter/U5).

Only the age input changes; brain_spec, abs_ihs, MAPT exclusion, StandardScaler,
GMM(k, full cov, random_state=42, n_init=10), and youngest->C0 sort are identical.
Outputs recluster/assign_{clock}_k{2,3}.tsv.gz (rsid, cluster).
"""
import sys
import glob
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

clock = sys.argv[1] if len(sys.argv) > 1 else "mut"
B = Path(_ROOT)
OUT = Path((_SCRATCH + "/test1_age_conditioning/recluster"))
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

m = pd.read_parquet(B / "results/phase11/variant_master_v3.parquet")
if clock == "mut":
    age = pd.to_numeric(m["age_median_yr"], errors="coerce")
elif clock == "jnt":
    parts = [pd.read_csv(f, sep="\t", usecols=["rsid", "age_jnt_gen"])
             for f in sorted(glob.glob((_SCRATCH + "/test1_age_conditioning/age_tables/age_chr*.tsv")))]
    aj = pd.concat(parts, ignore_index=True).drop_duplicates("rsid")
    aj["age_jnt_gen"] = pd.to_numeric(aj["age_jnt_gen"], errors="coerce")
    m = m.merge(aj, on="rsid", how="left")
    age = m["age_jnt_gen"]
else:
    sys.exit("clock must be 'mut' or 'jnt'")

m["log_age"] = np.where(age.notna() & (age > 0), np.log10(age), np.nan)
m["b_logp"] = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                       -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                        -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs()
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)
m_nm = m[~in_mapt].copy()
mix = m_nm[["log_age", "brain_spec", "abs_ihs"]].dropna()
idx = mix.index
scaler = StandardScaler().fit(mix.values)
z = scaler.transform(mix.values)
print(f"clock={clock}: {len(mix)} variants for GMM (MAPT-excluded, 3 features)", flush=True)

for K in [2, 3]:
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=42, n_init=10).fit(z)
    lab = gmm.predict(z)
    means = scaler.inverse_transform(gmm.means_)
    order = np.argsort(means[:, 0])               # sort clusters by mean log_age (youngest first)
    remap = {old: new for new, old in enumerate(order)}
    labs = np.array([remap[l] for l in lab])
    out = m_nm.loc[idx, ["rsid"]].copy()
    out["cluster"] = labs
    out.to_csv(OUT / f"assign_{clock}_k{K}.tsv.gz", sep="\t", index=False, compression="gzip")
    sizes = pd.Series(labs).value_counts().sort_index().to_dict()
    ages = {c: round(float(10 ** np.mean(mix["log_age"].values[labs == c])), 1) for c in range(K)}
    print(f"  k={K}: sizes={sizes}  mean_age C0..C{K-1}={ages}", flush=True)

if clock == "mut":
    pub = pd.read_csv(B / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz",
                      sep="\t", usecols=["rsid", "cluster"])
    pub = pub[pub["cluster"].notna()].copy()
    pub["cluster"] = pub["cluster"].astype(float).astype(int)
    mine = pd.read_csv(OUT / "assign_mut_k3.tsv.gz", sep="\t")
    cmp = pub.merge(mine, on="rsid", suffixes=("_pub", "_mine"))
    conc = (cmp["cluster_pub"] == cmp["cluster_mine"].astype(int)).mean()
    print(f"  SANITY mut-k3 vs published P14b_v3: {len(cmp)} shared rsids, "
          f"concordance={conc*100:.2f}%  ({'FAITHFUL' if conc > 0.99 else 'REPLICATION DIVERGES — investigate'})",
          flush=True)
print("RECLUSTER_DONE", flush=True)
