#!/usr/bin/env python3
"""
Phase 14b v3: Mixture Model with 17q21.31 Exclusion + Cluster Characterization
================================================================================
Following P14b_v2 finding that BIC strongly favors k≥2 components, this
script:
  1. Re-fits Gaussian mixture with 17q21.31 excluded — is multimodality real or
     MAPT structural artifact?
  2. If still k≥2 supported, assigns each variant to its most-probable cluster
  3. Characterizes each cluster:
     - Mean/median of each feature
     - Tissue distribution
     - Gene class (LoF intolerant, coding etc.)
     - Pathway hint (top genes per cluster)
  4. Per-cluster within-locus tests (does the primary brain_spec×age finding
     hold within each cluster?)

Output:
  results/phase14b/P14b_v3_mixture_no_mapt.tsv (BIC table)
  results/phase14b/P14b_v3_cluster_assignments.tsv.gz
  results/phase14b/P14b_v3_cluster_profiles.tsv
  results/phase14b/P14b_v3_cluster_genes.tsv
  results/phase14b/P14b_v3_NARRATIVE.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14b"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14b v3: Mixture Clusters — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
m["abs_akbari_s"] = m["akbari_s"].abs() if "akbari_s" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int_num"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int_num"] >= 43_000_000) & (m["pos_int_num"] <= 46_000_000)
log(f"\nTotal variants: {len(m)}")
log(f"17q21.31 (MAPT) variants: {in_mapt.sum()}")


# ─── 1. RE-FIT mixture with 17q21.31 EXCLUDED ─────────────────────────────
log("\n" + "=" * 72)
log("[1] Mixture model BIC — 17q21.31 EXCLUDED")
log("=" * 72)

m_nm = m[~in_mapt].copy()
mix_data = m_nm[["log_age", "brain_spec", "abs_ihs"]].dropna()
indices = mix_data.index
mix_arr = mix_data.values

log(f"\n  Variants for mixture (no MAPT, all 3 features): {len(mix_arr)}")

scaler = StandardScaler()
mix_z = scaler.fit_transform(mix_arr)

bic_results = []
for k in [1, 2, 3, 4, 5, 6]:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=10)
    gmm.fit(mix_z)
    bic = gmm.bic(mix_z)
    aic = gmm.aic(mix_z)
    ll = gmm.score(mix_z) * len(mix_z)
    bic_results.append({"k_components": k, "BIC": bic, "AIC": aic, "log_likelihood": ll})
    log(f"  k={k}: BIC={bic:.1f}, AIC={aic:.1f}, LL={ll:.1f}")

df_bic = pd.DataFrame(bic_results)
df_bic["delta_BIC_vs_k1"] = df_bic["BIC"] - df_bic.loc[0, "BIC"]
df_bic.to_csv(OUT / "P14b_v3_mixture_no_mapt.tsv", sep="\t", index=False)

best_k = int(df_bic.loc[df_bic["BIC"].idxmin(), "k_components"])
log(f"\n  Best k by BIC: {best_k}")
log(f"  ΔBIC k=2 vs k=1: {df_bic.loc[1,'BIC'] - df_bic.loc[0,'BIC']:+.1f}")
log(f"  ΔBIC k=3 vs k=1: {df_bic.loc[2,'BIC'] - df_bic.loc[0,'BIC']:+.1f}")

# Choose k=3 (parsimonious, big BIC drop) for cluster characterization
USE_K = 3
log(f"\n  Using k={USE_K} for cluster assignment (parsimonious, conservative)")


# ─── 2. CLUSTER ASSIGNMENT ────────────────────────────────────────────────
log("\n" + "=" * 72)
log(f"[2] Cluster assignment with k={USE_K} (variants with all 3 features)")
log("=" * 72)

gmm = GaussianMixture(n_components=USE_K, covariance_type="full", random_state=42, n_init=10)
gmm.fit(mix_z)
cluster_labels = gmm.predict(mix_z)
cluster_probs = gmm.predict_proba(mix_z)

# Sort cluster IDs by mean log_age (ascending) for interpretability
unscaled_means = scaler.inverse_transform(gmm.means_)  # back to original scale
sort_idx = np.argsort(unscaled_means[:, 0])  # sort by log_age (col 0)
remap = {old: new for new, old in enumerate(sort_idx)}
cluster_labels_sorted = np.array([remap[l] for l in cluster_labels])

# Attach to dataframe (FIX-E-B-1 verification: explicit length + index assertion).
# `mix_data = m_nm[...].dropna()` preserves m_nm's index ordering so
# `indices` is a Pandas Index where the i-th label corresponds to the i-th
# row of mix_arr → mix_z → cluster_labels → cluster_labels_sorted.
# `m_nm.loc[indices, "cluster"] = arr` does positional alignment of the i-th
# array element to the i-th label in `indices` (label-not-positional align,
# but since indices is a uniqu Pandas Index, the result is well-defined).
assert len(indices) == len(cluster_labels_sorted), (
    f"index/label length mismatch: {len(indices)} vs {len(cluster_labels_sorted)}"
)
m_nm["cluster"] = pd.NA
# Use a Series with the matching index to make the alignment unambiguous.
cluster_series = pd.Series(cluster_labels_sorted, index=indices, name="cluster")
m_nm.loc[indices, "cluster"] = cluster_series.values
# Sanity check: pick a few random indices and verify their cluster matches
# the prediction we'd get from re-feeding their feature vector to the GMM.
_sample_check = m_nm.loc[indices, ["log_age", "brain_spec", "abs_ihs"]].iloc[:5]
_sample_pred = gmm.predict(scaler.transform(_sample_check.values))
_sample_pred_sorted = np.array([remap[l] for l in _sample_pred])
_sample_assigned = m_nm.loc[_sample_check.index, "cluster"].astype(int).values
assert np.array_equal(_sample_pred_sorted, _sample_assigned), (
    "Cluster assignment misaligned with GMM prediction at sample rows"
)

log(f"\n  Cluster sizes:")
cluster_counts = m_nm["cluster"].value_counts().sort_index()
log(cluster_counts.to_string())

# Cluster characteristics from GMM means (un-scaled)
cluster_means = pd.DataFrame(unscaled_means[sort_idx], columns=["log_age", "brain_spec", "abs_ihs"])
cluster_means["age_yr"] = 10 ** cluster_means["log_age"]
cluster_means["cluster_id"] = range(USE_K)
log(f"\n  Cluster centroids (unscaled):")
log(cluster_means.to_string(index=False))


# ─── 3. PER-CLUSTER FEATURE PROFILE ───────────────────────────────────────
log("\n" + "=" * 72)
log("[3] Per-cluster feature profiles (median values, all features)")
log("=" * 72)

profile_features = [
    ("age_median_yr", "Age (yr)"),
    ("maf", "MAF"),
    ("beta", "GWAS β"),
    ("pip", "PIP"),
    ("cadd_phred", "CADD"),
    ("brain_spec", "Brain spec"),
    ("abs_ihs", "|iHS|"),
    ("sds", "SDS"),
    ("akbari_s", "Akbari S"),
    ("akbari_pi", "Akbari π"),
    ("loeuf", "LOEUF"),
    ("pLI", "pLI"),
    ("b_logp", "−log10 brain p"),
    ("bl_logp", "−log10 blood p"),
]

profile_rows = []
for col, label in profile_features:
    if col not in m_nm.columns: continue
    row = {"feature": label}
    for c in range(USE_K):
        sub = m_nm.loc[m_nm["cluster"] == c, col].dropna()
        row[f"C{c}_n"] = len(sub)
        row[f"C{c}_med"] = float(sub.median()) if len(sub) > 0 else np.nan
    profile_rows.append(row)

df_prof = pd.DataFrame(profile_rows)
df_prof.to_csv(OUT / "P14b_v3_cluster_profiles.tsv", sep="\t", index=False)

log(f"\n  Per-cluster median (all features):")
log(f"  {'Feature':<22s} " + " ".join(f"{'C'+str(c)+'(n)':>10s}" for c in range(USE_K)))
for _, r in df_prof.iterrows():
    vals = " ".join(f"{r[f'C{c}_med']:>10.4f}" for c in range(USE_K))
    ns = " ".join(f"({int(r[f'C{c}_n'])})" for c in range(USE_K))
    log(f"  {r['feature']:<22s} " + vals + "  " + ns)


# ─── 4. PER-CLUSTER GENE ENRICHMENT ────────────────────────────────────────
log("\n" + "=" * 72)
log("[4] Top genes per cluster")
log("=" * 72)

cluster_gene_top = []
for c in range(USE_K):
    sub = m_nm[m_nm["cluster"] == c]
    if len(sub) == 0: continue
    log(f"\n  Cluster {c} (n={len(sub)}, median age={sub['age_median_yr'].median():.0f} yr):")
    top_genes = sub["gene_symbol"].value_counts().head(15)
    for g, n in top_genes.items():
        cluster_gene_top.append({"cluster": c, "gene": g, "n_variants": n})
    log(top_genes.to_string())

df_genes = pd.DataFrame(cluster_gene_top)
df_genes.to_csv(OUT / "P14b_v3_cluster_genes.tsv", sep="\t", index=False)


# ─── 5. PER-CLUSTER PRIMARY TESTS ──────────────────────────────────────────
log("\n" + "=" * 72)
log("[5] Per-cluster primary tests — does brain_spec × age hold within each?")
log("=" * 72)

import sys as _sys
_sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation (code-review Faz C lib)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]

cluster_tests = []
for c in range(USE_K):
    sub = m_nm[m_nm["cluster"] == c].copy()
    for tname, x, y in [
        ("brain_spec × age", "log_age", "brain_spec"),
        ("|iHS| × age", "log_age", "abs_ihs"),
        ("Akbari π × age", "log_age", "akbari_pi"),
    ]:
        rho, p, n = maf_resid_rho(sub, x, y)
        cluster_tests.append({"cluster": c, "test": tname,
                                "n": n, "rho": rho if rho is not None else np.nan,
                                "p": p if p is not None else np.nan})
log(f"\n  Per-cluster primary tests:")
log(f"  {'Cluster':<8s} {'Test':<22s} {'n':>6s} {'rho':>9s} {'p':>10s}")
for r in cluster_tests:
    rho_str = f"{r['rho']:>+8.4f}" if pd.notna(r['rho']) else "    NA"
    p_str = f"{r['p']:>10.2e}" if pd.notna(r['p']) else "        NA"
    log(f"  {r['cluster']:<8} {r['test']:<22s} {r['n']:>6d} {rho_str} {p_str}")

df_ctests = pd.DataFrame(cluster_tests)
df_ctests.to_csv(OUT / "P14b_v3_cluster_tests.tsv", sep="\t", index=False)


# ─── 6. ASSIGNMENTS DUMP ──────────────────────────────────────────────────
m_nm[["rsid", "credible_set_id", "chr", "pos", "gene_symbol",
       "age_median_yr", "brain_spec", "abs_ihs", "cluster"]].to_csv(
    OUT / "P14b_v3_cluster_assignments.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"\n  Saved assignments: {OUT / 'P14b_v3_cluster_assignments.tsv.gz'}")


# ─── 7. INTERPRETATION ────────────────────────────────────────────────────
log("\n" + "=" * 72)
log("INTERPRETATION")
log("=" * 72)

dbic_2v1 = df_bic.loc[1,'BIC'] - df_bic.loc[0,'BIC']
dbic_3v1 = df_bic.loc[2,'BIC'] - df_bic.loc[0,'BIC']

if dbic_2v1 < -10 and dbic_3v1 < dbic_2v1 - 10:
    log(f"""
  Mixture model with 17q21.31 EXCLUDED:
    - k=1 strongly rejected (ΔBIC k=2 vs k=1: {dbic_2v1:+.0f})
    - k=3 strongly preferred over k=2 (ΔBIC k=3 vs k=1: {dbic_3v1:+.0f})
    - Multimodality is REAL biology, not MAPT structural artifact

  Manuscript framing should be:
    "PGC3 SCZ variants partition into ≥3 distinct evolutionary sub-populations
     (BIC-justified k=3, ΔBIC=−{abs(int(dbic_3v1))} vs k=1, after 17q21.31 exclusion).
     Cluster centroids span a range of (allele age, brain specificity, |iHS|)
     joint distributions that cannot be explained by a single-class model."
""")
elif dbic_2v1 < -10:
    log(f"""
  Mixture model with 17q21.31 EXCLUDED:
    - k=1 rejected (ΔBIC k=2 vs k=1: {dbic_2v1:+.0f})
    - But k=3 not strongly preferred over k=2
    - Bimodal (2-class) structure may be defensible
""")
else:
    log(f"""
  Mixture model with 17q21.31 EXCLUDED:
    - k=1 NOT rejected (ΔBIC k=2 vs k=1: {dbic_2v1:+.0f}, not <-10)
    - Multimodality previous detected was MAPT structural artifact
    - Continuous gradient is the correct framing
""")

with open(OUT / "P14b_v3_NARRATIVE.md", "w") as f:
    f.write("# Phase 14b v3: Mixture Clusters (no MAPT)\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nNarrative: {OUT / 'P14b_v3_NARRATIVE.md'}")
log("\nPhase 14b v3 complete.")
