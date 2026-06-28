#!/usr/bin/env python3
"""
Phase 14b: PIP, Effect Size, and Age Stratification
====================================================
User-suggested analyses:
  1. PIP-stratified: do high-PIP (causal candidate) variants concentrate the
     evolutionary signal vs low-PIP tag variants?
  2. Effect size stratified: do large-effect SCZ alleles have different
     evolutionary characteristics?
  3. Young vs Old (age quartile) profiles:
     - Tissue distribution (which GTEx tissue most often shows eQTL)
     - Gene class distribution (LoF intolerant? Coding?)
     - VEP impact distribution
     - Brain region preference

Output:
  results/phase14b/P14b_pip_stratification.tsv
  results/phase14b/P14b_effect_stratification.tsv
  results/phase14b/P14b_young_vs_old_profile.tsv
  results/phase14b/P14b_tissue_distribution.tsv
  results/phase14b/P14b_NARRATIVE.md
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14b"
OUT.mkdir(parents=True, exist_ok=True)

# Code-review Faz C corrected helper (within-locus partial rank correlation
# with optional within-locus MAF-rank residualisation)
sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14b: PIP / Effect / Age Stratification — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

np.random.seed(42)

m = pd.read_parquet(P11 / "variant_master_v4.parquet")
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
m["abs_beta"] = m["beta"].abs()


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation with within-locus MAF-rank
    residualisation (code-review Faz C corrected; delegates to
    _within_locus_lib.within_locus_partial_rank_correlation)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


# ─── 1. PIP STRATIFICATION ─────────────────────────────────────────────────
log("\n" + "=" * 72)
log("[1] PIP stratification — do high-PIP causal candidates concentrate effects?")
log("=" * 72)

log(f"\n  PIP distribution:")
log(f"    median: {m['pip'].median():.4f}")
log(f"    mean:   {m['pip'].mean():.4f}")
log(f"    n high (PIP>0.5): {(m['pip']>0.5).sum()}")
log(f"    n high (PIP>0.1): {(m['pip']>0.1).sum()}")
log(f"    n low  (PIP<0.01): {(m['pip']<0.01).sum()}")

# 4-bin PIP stratification
m["pip_bin"] = pd.cut(m["pip"], bins=[-0.001, 0.01, 0.1, 0.5, 1.0],
                       labels=["LOW(<0.01)", "MID(0.01-0.1)", "HIGH(0.1-0.5)", "VHIGH(>0.5)"])
log(f"\n  PIP bin distribution:")
for k, v in m["pip_bin"].value_counts(dropna=False).items():
    log(f"    {k}: {v}")

pip_results = []
for bin_label in ["LOW(<0.01)", "MID(0.01-0.1)", "HIGH(0.1-0.5)", "VHIGH(>0.5)"]:
    sub = m[m["pip_bin"] == bin_label].copy()
    sub_no_mapt = sub[~in_mapt[sub.index]]
    for tname, x, y in [
        ("|iHS| × age", "log_age", "abs_ihs"),
        ("Brain spec × age", "log_age", "brain_spec"),
        ("Akbari π × age", "log_age", "akbari_pi"),
    ]:
        rho, p, n = maf_resid_rho(sub_no_mapt, x, y)
        if rho is not None:
            pip_results.append({
                "pip_bin": bin_label,
                "test": tname,
                "n": n, "rho": rho, "p": p
            })

df_pip = pd.DataFrame(pip_results)
df_pip.to_csv(OUT / "P14b_pip_stratification.tsv", sep="\t", index=False)
log(f"\n  PIP-stratified primary tests (no 17q21.31):")
log(f"  {'PIP bin':<18s} {'Test':<22s} {'n':>6s} {'rho':>9s} {'p':>10s}")
for _, r in df_pip.iterrows():
    log(f"  {r['pip_bin']:<18s} {r['test']:<22s} {r['n']:>6d} {r['rho']:>+8.4f} {r['p']:>10.2e}")


# ─── 2. EFFECT SIZE STRATIFICATION ────────────────────────────────────────
log("\n" + "=" * 72)
log("[2] Effect size stratification (|β|) — large vs small SCZ effect alleles")
log("=" * 72)

log(f"\n  |β| distribution: median={m['abs_beta'].median():.4f}, p95={m['abs_beta'].quantile(0.95):.4f}")

# Quartile bins
q_thresh = m["abs_beta"].quantile([0.25, 0.5, 0.75]).tolist()
m["abs_beta_q"] = pd.cut(m["abs_beta"], bins=[-0.001] + q_thresh + [10],
                          labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"])

eff_results = []
for q in ["Q1_smallest", "Q2", "Q3", "Q4_largest"]:
    sub = m[m["abs_beta_q"] == q].copy()
    sub_no_mapt = sub[~in_mapt[sub.index]]
    for tname, x, y in [
        ("|iHS| × age", "log_age", "abs_ihs"),
        ("Brain spec × age", "log_age", "brain_spec"),
        ("Akbari π × age", "log_age", "akbari_pi"),
    ]:
        rho, p, n = maf_resid_rho(sub_no_mapt, x, y)
        if rho is not None:
            eff_results.append({
                "beta_quartile": q,
                "test": tname,
                "n": n, "rho": rho, "p": p
            })

df_eff = pd.DataFrame(eff_results)
df_eff.to_csv(OUT / "P14b_effect_stratification.tsv", sep="\t", index=False)
log(f"\n  Effect-size-stratified primary tests (no 17q21.31):")
log(f"  {'|β| Quartile':<14s} {'Test':<22s} {'n':>6s} {'rho':>9s} {'p':>10s}")
for _, r in df_eff.iterrows():
    log(f"  {r['beta_quartile']:<14s} {r['test']:<22s} {r['n']:>6d} {r['rho']:>+8.4f} {r['p']:>10.2e}")

# Median age per quartile
log(f"\n  Median age (yr) by |β| quartile:")
log(m.groupby("abs_beta_q")["age_median_yr"].agg(['median', 'count']).to_string())


# ─── 3. YOUNG vs OLD PROFILE ──────────────────────────────────────────────
log("\n" + "=" * 72)
log("[3] Young vs Old quartile (Q1 vs Q4) profiles")
log("=" * 72)

age_q = m["age_median_yr"].quantile([0.25, 0.5, 0.75]).tolist()
m["age_q"] = pd.cut(m["age_median_yr"],
                     bins=[0] + age_q + [1e10],
                     labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"])
log(f"\n  Age quartile thresholds: Q1<{age_q[0]:.0f}, Q2<{age_q[1]:.0f}, Q3<{age_q[2]:.0f}, Q4>{age_q[2]:.0f}")
log(f"  N per quartile:")
log(m["age_q"].value_counts().sort_index().to_string())

# Compare Q1 (young) vs Q4 (old) on multiple features
profile_rows = []
for feat, label in [
    ("maf", "MAF"),
    ("abs_beta", "|GWAS β|"),
    ("pip", "PIP"),
    ("cadd_phred", "CADD"),
    ("abs_ihs", "|iHS|"),
    ("sds", "SDS"),
    ("abs_akbari_s", "|Akbari S|"),
    ("akbari_pi", "Akbari π"),
    ("brain_spec", "Brain specificity"),
    ("b_logp", "−log10(brain min p)"),
    ("bl_logp", "−log10(blood min p)"),
    ("loeuf", "LOEUF"),
    ("pLI", "pLI"),
    ("atac_n_clusters", "ATAC clusters"),
    ("desert_tier", "Desert tier"),
]:
    if feat not in m.columns: continue
    young = m.loc[m["age_q"] == "Q1_youngest", feat].dropna()
    old = m.loc[m["age_q"] == "Q4_oldest", feat].dropna()
    if len(young) < 20 or len(old) < 20: continue
    u, p = stats.mannwhitneyu(young, old, alternative="two-sided")
    profile_rows.append({
        "feature": label,
        "n_young": len(young),
        "median_young": young.median(),
        "n_old": len(old),
        "median_old": old.median(),
        "delta_median": young.median() - old.median(),
        "MWU_p": p
    })

df_prof = pd.DataFrame(profile_rows)
df_prof["fdr_q"] = multipletests(df_prof["MWU_p"], method="fdr_bh")[1]
df_prof = df_prof.sort_values("MWU_p")
df_prof.to_csv(OUT / "P14b_young_vs_old_profile.tsv", sep="\t", index=False)

log(f"\n  Q1 (young) vs Q4 (old) feature comparisons (Mann-Whitney U):")
log(f"  {'Feature':<22s} {'n_yg':>5s} {'med_yg':>10s} {'n_old':>5s} {'med_old':>10s} {'p':>10s} {'q':>10s}")
for _, r in df_prof.iterrows():
    log(f"  {r['feature']:<22s} {int(r['n_young']):>5d} {r['median_young']:>10.4f} "
        f"{int(r['n_old']):>5d} {r['median_old']:>10.4f} {r['MWU_p']:>10.2e} {r['fdr_q']:>10.2e}")


# ─── 4. TISSUE DISTRIBUTION ────────────────────────────────────────────────
log("\n" + "=" * 72)
log("[4] Tissue distribution: where do young vs old SCZ variants show eQTL?")
log("=" * 72)

# Brain tissue distribution
log(f"\n  Most common brain eQTL tissue per age quartile:")
brain_tissue_dist = m.groupby("age_q")["gtex_brain_tissue"].value_counts().unstack(fill_value=0)
log("\n" + brain_tissue_dist.to_string())

# Save full distribution
brain_tissue_dist.to_csv(OUT / "P14b_tissue_distribution.tsv", sep="\t")

# VEP impact distribution
log(f"\n  VEP impact by age quartile:")
vep_dist = m.groupby("age_q")["vep_impact"].value_counts(normalize=True).unstack(fill_value=0) * 100
log("\n" + vep_dist.to_string())

# Constraint by quartile
log(f"\n  pLI > 0.9 (LoF-intolerant) frequency by age quartile:")
constraint_freq = m.groupby("age_q").apply(lambda g: (g["pLI"] > 0.9).sum() / g["pLI"].notna().sum() * 100).reset_index()
constraint_freq.columns = ["age_q", "pct_constrained"]
log(constraint_freq.to_string(index=False))


# ─── 5. PROPORTIONAL HAZARD: high-PIP variants more likely young or old? ──
log("\n" + "=" * 72)
log("[5] Are high-PIP causal candidates concentrated at specific age?")
log("=" * 72)

high_pip = m[m["pip"] > 0.5]
low_pip = m[m["pip"] < 0.01]
log(f"\n  high-PIP (>0.5): n={len(high_pip)}, median_age={high_pip['age_median_yr'].median():.0f}")
log(f"  low-PIP (<0.01): n={len(low_pip)}, median_age={low_pip['age_median_yr'].median():.0f}")
u, p = stats.mannwhitneyu(high_pip["age_median_yr"].dropna(),
                           low_pip["age_median_yr"].dropna(),
                           alternative="two-sided")
log(f"  MWU p={p:.3e}")

# Cross-tab: PIP bin × age quartile
log(f"\n  Cross-tab PIP bin × age quartile (counts):")
ct = pd.crosstab(m["pip_bin"], m["age_q"])
log("\n" + ct.to_string())

chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
log(f"\n  Chi-square independence: chi2={chi2:.2f}, dof={dof}, p={p_chi2:.3e}")


# Save log
with open(OUT / "P14b_NARRATIVE.md", "w") as f:
    f.write("# Phase 14b: PIP / Effect / Age Stratification\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nNarrative: {OUT / 'P14b_NARRATIVE.md'}")
log("\nPhase 14b complete.")
