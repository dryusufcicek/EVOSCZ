#!/usr/bin/env python3
"""Phase 17C: Tissue-power-matched brain_spec re-definition + GMM + S-LDSC.

Robustness check #7: GTEx v10 tissue sample sizes differ substantially between
Whole_Blood (N≈838) and individual brain tissues (N≈140-280). For an identical
effect-size β, the P-value is smaller in higher-N tissues because Z = β·sqrt(N).
The original brain_spec formula

    brain_spec = -log10(brain_minp) / [-log10(brain_minp) + -log10(blood_minp)]

is therefore systematically biased downward (blood signal artificially inflated).

This script:
  1. Reconstructs per-tissue Z-statistic from observed minP using two-sided
     standard normal inverse: |Z| = -Φ⁻¹(P/2); signed via slope sign.
  2. Scales Z to an equal-N reference (N_ref = 220, median GTEx v10 brain tissue):
     Z_eq = Z_observed × sqrt(N_ref / N_tissue_observed).
  3. Recomputes power-matched brain_spec_eq using equal-N P-values:
     P_eq = 2·Φ(-|Z_eq|).
     brain_spec_eq = -log10(P_eq_brain) / [-log10(P_eq_brain) + -log10(P_eq_blood)].
  4. Re-fits the primary 3D GMM (k=3) with brain_spec_eq replacing brain_spec.
     Other features (log_age, |iHS|) unchanged.
  5. Builds annotation files + computes LD scores + runs S-LDSC.

GTEx v10 sample sizes from gtexportal.org/home/datasets (v10 release notes).
N_ref = 220 chosen as median Brain N to keep brain tissue Z-statistic largely
unchanged and downscale Whole_Blood by sqrt(220/838) ≈ 0.51.

Output:
  results/phase17c/cluster_assignments_power_matched.tsv.gz
  results/phase17c/cluster_annot_power_matched/
  results/phase17c/h2_pgc3_eur_power_matched.results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
P14E_BASELINE = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
OUT = BASE / "results/phase17c"
OUT_ANNOT = OUT / "cluster_annot_power_matched"
for d in [OUT, OUT_ANNOT]:
    d.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17C: Tissue-power-matched brain_spec — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# ─── GTEx v10 sample sizes (donor N per tissue, official release notes) ───
GTEX_V10_N = {
    "Whole_Blood": 838,
    "Brain_Caudate_basal_ganglia": 246,
    "Brain_Cerebellar_Hemisphere": 220,
    "Brain_Cerebellum": 250,
    "Brain_Cortex": 276,
    "Brain_Frontal_Cortex_BA9": 219,
    "Brain_Hippocampus": 213,
    "Brain_Hypothalamus": 216,
    "Brain_Nucleus_accumbens_basal_ganglia": 246,
    "Brain_Putamen_basal_ganglia": 220,
    "Brain_Spinal_cord_cervical_c-1": 159,
    "Brain_Substantia_nigra": 144,
    "Brain_Amygdala": 169,
    "Brain_Anterior_cingulate_cortex_BA24": 191,
    "Spleen": 247,
}
N_REF = 220  # median brain tissue N

log(f"\nGTEx v10 sample sizes (donor N per tissue):")
for t, n in sorted(GTEX_V10_N.items(), key=lambda x: -x[1]):
    log(f"  {t:<45s} N={n}")
log(f"\nReference N_ref = {N_REF} (median brain tissue)")


# ─── 1. LOAD VARIANT MASTER AND COMPUTE Z-RECONSTRUCTED brain_spec_eq ─────
log("\n[1] Loading variant master")
m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int"] >= 43_000_000) & (m["pos_int"] <= 46_000_000)

# Original brain_spec for comparison
m["b_logp"]  = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                          -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                          -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec_orig"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])


# ─── 2. RECONSTRUCT Z FROM minP AND TISSUE, RESCALE TO N_REF ──────────────
log("\n[2] Reconstructing per-tissue Z and rescaling to N_ref")

def p_to_abs_z(p):
    """Two-sided P-value → |Z| via inverse normal."""
    p = np.clip(p, 1e-300, 1.0)
    return -stats.norm.ppf(p / 2.0)

# Brain side
have_brain = m["gtex_brain_minp"].notna() & m["gtex_brain_tissue"].notna()
m["z_brain_abs"] = np.nan
m.loc[have_brain, "z_brain_abs"] = p_to_abs_z(m.loc[have_brain, "gtex_brain_minp"].astype(float).values)
# Lookup N per row; unknown tissues mapped to N_REF (no scaling)
m["n_brain"] = m["gtex_brain_tissue"].map(GTEX_V10_N).fillna(N_REF)
# Rescale
m["z_brain_eq"] = m["z_brain_abs"] * np.sqrt(N_REF / m["n_brain"])
# Power-matched P
m["p_brain_eq"] = 2.0 * stats.norm.sf(np.abs(m["z_brain_eq"]))
m["p_brain_eq"] = m["p_brain_eq"].clip(lower=1e-300)

# Blood side
have_blood = m["gtex_blood_minp"].notna() & m["gtex_blood_tissue"].notna()
m["z_blood_abs"] = np.nan
m.loc[have_blood, "z_blood_abs"] = p_to_abs_z(m.loc[have_blood, "gtex_blood_minp"].astype(float).values)
m["n_blood"] = m["gtex_blood_tissue"].map(GTEX_V10_N).fillna(N_REF)
m["z_blood_eq"] = m["z_blood_abs"] * np.sqrt(N_REF / m["n_blood"])
m["p_blood_eq"] = 2.0 * stats.norm.sf(np.abs(m["z_blood_eq"]))
m["p_blood_eq"] = m["p_blood_eq"].clip(lower=1e-300)

# Power-matched brain_spec
m["b_logp_eq"]  = -np.log10(m["p_brain_eq"])
m["bl_logp_eq"] = -np.log10(m["p_blood_eq"])
m["brain_spec_eq"] = m["b_logp_eq"] / (m["b_logp_eq"] + m["bl_logp_eq"])

# Compare distributions
both_avail = m["brain_spec_orig"].notna() & m["brain_spec_eq"].notna()
log(f"\n  Both brain_spec_orig and brain_spec_eq computable: {both_avail.sum():,}")
log(f"  brain_spec_orig: mean={m.loc[both_avail,'brain_spec_orig'].mean():.4f}, "
    f"median={m.loc[both_avail,'brain_spec_orig'].median():.4f}, "
    f"std={m.loc[both_avail,'brain_spec_orig'].std():.4f}")
log(f"  brain_spec_eq:   mean={m.loc[both_avail,'brain_spec_eq'].mean():.4f}, "
    f"median={m.loc[both_avail,'brain_spec_eq'].median():.4f}, "
    f"std={m.loc[both_avail,'brain_spec_eq'].std():.4f}")

# Quantify the shift
log(f"\n  Correlation (orig vs eq): r = {m.loc[both_avail,['brain_spec_orig','brain_spec_eq']].corr().iat[0,1]:.4f}")
log(f"  Median shift (orig - eq): {(m.loc[both_avail,'brain_spec_orig'] - m.loc[both_avail,'brain_spec_eq']).median():+.4f}")
log(f"  Variants that flip 'brain-biased' (>0.5) status:")
flip_to_brain = both_avail & (m["brain_spec_orig"] <= 0.5) & (m["brain_spec_eq"] > 0.5)
flip_from_brain = both_avail & (m["brain_spec_orig"] > 0.5) & (m["brain_spec_eq"] <= 0.5)
log(f"    Original <=0.5 → Equal-N >0.5 (more brain-biased after correction): {flip_to_brain.sum():,}")
log(f"    Original >0.5 → Equal-N <=0.5: {flip_from_brain.sum():,}")


# ─── 3. RE-FIT 3D GMM WITH brain_spec_eq ───────────────────────────────────
log("\n[3] Re-fitting 3D GMM with brain_spec_eq (k=3, n_init=10, random_state=42)")

m_nm = m[~in_mapt].copy()
mix_data = m_nm[["log_age", "brain_spec_eq", "abs_ihs"]].dropna()
indices = mix_data.index
mix_arr = mix_data.values
log(f"  Non-MAPT with all 3 features (eq): {len(mix_arr):,}")

scaler = StandardScaler()
mix_z = scaler.fit_transform(mix_arr)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10)
gmm.fit(mix_z)
unscaled_means = scaler.inverse_transform(gmm.means_)
sort_idx = np.argsort(unscaled_means[:, 0])  # sort by log_age
remap = {old: new for new, old in enumerate(sort_idx)}

log(f"\n  GMM converged: {gmm.converged_}")
log(f"  Component centroids (sorted by log_age, unscaled):")
for new_idx, old_idx in enumerate(sort_idx):
    centroid = unscaled_means[old_idx]
    age_yr = 10 ** centroid[0]
    log(f"    C{new_idx}: log_age={centroid[0]:.3f} ({age_yr:.0f} yr), "
        f"brain_spec_eq={centroid[1]:.3f}, |iHS|={centroid[2]:.3f}")

preds_raw = gmm.predict(mix_z)
preds_sorted = np.array([remap[l] for l in preds_raw])
m_nm["cluster"] = pd.NA
m_nm.loc[indices, "cluster"] = preds_sorted

log(f"\n  Cluster sizes (eq):")
for c in range(3):
    n = int((m_nm["cluster"] == c).sum())
    log(f"    C{c}: {n:,}")


# ─── 4. SAVE ASSIGNMENTS + BUILD ANNOTATION FILES ──────────────────────────
log("\n[4] Saving assignments and building cluster annotation files")
to_save = m_nm[m_nm["cluster"].notna()].copy()
to_save["cluster"] = to_save["cluster"].astype(int)
to_save[["rsid", "credible_set_id", "chr", "pos", "gene_symbol",
         "age_median_yr", "brain_spec_orig", "brain_spec_eq", "abs_ihs", "cluster"]].to_csv(
    OUT / "cluster_assignments_power_matched.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"  Saved: {OUT / 'cluster_assignments_power_matched.tsv.gz'}")

rsid2cluster = dict(zip(to_save["rsid"].astype(str), to_save["cluster"]))

for chrom in range(1, 23):
    base_path = P14E_BASELINE / f"baseline.{chrom}.annot.gz"
    if not base_path.exists():
        continue
    base = pd.read_csv(base_path, sep="\t", compression="gzip",
                       usecols=["CHR", "BP", "SNP", "CM"])
    base["C0"] = 0
    base["C1"] = 0
    base["C2"] = 0
    snps = base["SNP"].astype(str).values
    for i, snp in enumerate(snps):
        c = rsid2cluster.get(snp)
        if c == 0:
            base.iat[i, base.columns.get_loc("C0")] = 1
        elif c == 1:
            base.iat[i, base.columns.get_loc("C1")] = 1
        elif c == 2:
            base.iat[i, base.columns.get_loc("C2")] = 1
    out_path = OUT_ANNOT / f"cluster.{chrom}.annot.gz"
    base.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n0 = int(base["C0"].sum())
    n1 = int(base["C1"].sum())
    n2 = int(base["C2"].sum())
    log(f"  chr{chrom}: SNPs={len(base):,}, C0={n0}, C1={n1}, C2={n2}")

with open(OUT / "P17c_power_matched_log.md", "w") as f:
    f.write("# Phase 17C: Tissue-power-matched brain_spec\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")

log(f"\nDone. Annotation files in {OUT_ANNOT}")
log(f"Next: compute LD scores per-chrom, then S-LDSC.")
