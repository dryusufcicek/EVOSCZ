#!/usr/bin/env python3
"""
Phase 12: Variant-Level Statistical Analyses
=============================================
Uses results/phase11/variant_master.parquet (per-credible-set-variant table) to
run analyses at variant resolution rather than locus/gene aggregation.

Tests:
  1. Allele age × GTEx brain eQTL min-p (continuous correlation, MAF-controlled)
  2. Allele age × GTEx blood eQTL min-p (negative control)
  3. SDS × allele age (recent vs ancient selection)
  4. ATAC cluster overlap × age (cell-type regulatory variants)
  5. HAR overlap × age + pleiotropy
  6. Desert tier × eQTL (do desert variants regulate genes?)
  7. Brain vs blood eQTL specificity × age (variant-level)

Output:
  - results/phase12/P12_variant_results.tsv  (test summary)
  - results/phase12/P12_ANALYSIS_LOG.md
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


# ─── Load variant master ───────────────────────────────────────────────────
log(f"Phase 12: Variant-Level Analyses — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

m = pd.read_parquet(BASE / "results/phase11/variant_master.parquet")
log(f"\nLoaded variant master: {len(m)} rows × {len(m.columns)} cols")
log(f"Unique credible-set variants: {m['rsid'].nunique()}")

# Deduplicate to one row per rsid (keep highest PIP if multiple)
if "PIP" in m.columns or "pip" in m.columns:
    pip_col = "PIP" if "PIP" in m.columns else "pip"
    m = m.sort_values(pip_col, ascending=False).drop_duplicates("rsid", keep="first")
else:
    m = m.drop_duplicates("rsid", keep="first")
log(f"After dedup to one row per rsid: {len(m)}")

# Filter to variants with age
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
log(f"Variants with allele age: {len(m_age)}")


results = []
def add(test, n, statistic, p, note=""):
    results.append({"test": test, "n": n, "statistic": statistic, "p": p, "note": note})


# ─── Test 1: Age × Brain eQTL (continuous) ────────────────────────────────
log("\n" + "=" * 72)
log("[Test 1] Allele age × GTEx brain eQTL min-p (continuous)")
log("=" * 72)

if "gtex_brain_minp" in m_age.columns:
    sub = m_age[m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0)].copy()
    sub["neglog_brain_p"] = -np.log10(sub["gtex_brain_minp"])
    log(f"  n variants with brain eQTL + age: {len(sub)}")
    if len(sub) > 30:
        rho, p = stats.spearmanr(sub["log_age"], sub["neglog_brain_p"])
        log(f"  Spearman: rho={rho:.4f}, p={p:.4e}")
        add("Age × brain eQTL -log10p (Spearman)", len(sub), rho, p)

        # MAF-controlled partial correlation
        if "maf" in sub.columns and sub["maf"].notna().sum() > 30:
            maf_sub = sub[sub["maf"].notna()]
            from scipy.stats import linregress
            # Residuals approach
            r_age = stats.linregress(maf_sub["maf"], maf_sub["log_age"])
            res_age = maf_sub["log_age"] - (r_age.intercept + r_age.slope * maf_sub["maf"])
            r_eqtl = stats.linregress(maf_sub["maf"], maf_sub["neglog_brain_p"])
            res_eqtl = maf_sub["neglog_brain_p"] - (r_eqtl.intercept + r_eqtl.slope * maf_sub["maf"])
            rho2, p2 = stats.spearmanr(res_age, res_eqtl)
            log(f"  After MAF control (residuals): rho={rho2:.4f}, p={p2:.4e}")
            add("Age × brain eQTL (MAF-controlled)", len(maf_sub), rho2, p2)


# ─── Test 2: Age × Blood eQTL (negative control) ───────────────────────────
log("\n" + "=" * 72)
log("[Test 2] Allele age × GTEx blood eQTL min-p (negative control)")
log("=" * 72)

if "gtex_blood_minp" in m_age.columns:
    sub = m_age[m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0)].copy()
    sub["neglog_blood_p"] = -np.log10(sub["gtex_blood_minp"])
    log(f"  n variants with blood eQTL + age: {len(sub)}")
    if len(sub) > 30:
        rho, p = stats.spearmanr(sub["log_age"], sub["neglog_blood_p"])
        log(f"  Spearman: rho={rho:.4f}, p={p:.4e}")
        add("Age × blood eQTL -log10p (Spearman)", len(sub), rho, p)


# ─── Test 3: SDS × Age (recent vs ancient selection) ───────────────────────
log("\n" + "=" * 72)
log("[Test 3] SDS × Allele Age")
log("=" * 72)

sub = m_age[m_age["sds"].notna()].copy()
log(f"  n variants with SDS + age: {len(sub)}")
if len(sub) > 30:
    rho, p = stats.spearmanr(sub["log_age"], sub["sds"])
    log(f"  SDS vs log(age): Spearman rho={rho:.4f}, p={p:.4e}")
    log(f"    Negative rho expected: young variants → high recent SDS")
    add("Age × SDS (Spearman)", len(sub), rho, p)

    # Top 5% SDS = candidate recent sweep targets
    sds_thresh = sub["sds"].quantile(0.95)
    high_sds = sub[sub["sds"] > sds_thresh]
    low_sds = sub[sub["sds"] <= sds_thresh]
    if len(high_sds) > 5:
        u, pmw = stats.mannwhitneyu(high_sds["age_median_yr"], low_sds["age_median_yr"],
                                     alternative="less")
        log(f"  High-SDS (top 5%) median age: {high_sds['age_median_yr'].median():.0f}")
        log(f"  Other variants median age: {low_sds['age_median_yr'].median():.0f}")
        log(f"  MWU one-sided p={pmw:.4e}")
        add("High-SDS younger than rest", len(sub), u, pmw,
            f"high={len(high_sds)} low={len(low_sds)}")


# ─── Test 4: ATAC cluster overlap × Age ────────────────────────────────────
log("\n" + "=" * 72)
log("[Test 4] ATAC cell-type cluster overlap × Age")
log("=" * 72)

if "atac_n_clusters" in m_age.columns:
    sub = m_age[m_age["atac_n_clusters"].notna()].copy()
    rho, p = stats.spearmanr(sub["log_age"], sub["atac_n_clusters"])
    log(f"  n: {len(sub)}, Spearman age vs n_clusters: rho={rho:.4f}, p={p:.4e}")
    add("Age × ATAC n_clusters", len(sub), rho, p)

    # Per-cluster age comparison
    cluster_cols = [c for c in m_age.columns if c.startswith("atac_cluster")]
    if cluster_cols:
        log(f"\n  Per-cluster age comparison (variants in cluster vs not):")
        log(f"  {'Cluster':<15s} {'n_in':>6s} {'med_age_in':>12s} {'med_age_out':>12s} {'p':>10s}")
        cluster_results = []
        for cc in cluster_cols:
            in_grp = m_age.loc[m_age[cc] == 1, "age_median_yr"]
            out_grp = m_age.loc[m_age[cc] == 0, "age_median_yr"]
            if len(in_grp) < 10 or len(out_grp) < 10:
                continue
            u, p = stats.mannwhitneyu(in_grp, out_grp, alternative="two-sided")
            cluster_results.append((cc, len(in_grp), in_grp.median(), out_grp.median(), p))
        cluster_results.sort(key=lambda x: x[4])
        for cc, n, mi, mo, p in cluster_results[:10]:
            log(f"  {cc:<15s} {n:>6d} {mi:>12.0f} {mo:>12.0f} {p:>10.3e}")


# ─── Test 5: HAR overlap × Age & pleiotropy ────────────────────────────────
log("\n" + "=" * 72)
log("[Test 5] HAR overlap × Age")
log("=" * 72)

if "har_overlap" in m_age.columns:
    in_har = m_age.loc[m_age["har_overlap"] == 1, "age_median_yr"]
    out_har = m_age.loc[m_age["har_overlap"] == 0, "age_median_yr"]
    log(f"  HAR variants: {len(in_har)}, non-HAR: {len(out_har)}")
    if len(in_har) >= 5:
        u, p = stats.mannwhitneyu(in_har, out_har, alternative="two-sided")
        log(f"  Median age in HAR: {in_har.median():.0f}, non-HAR: {out_har.median():.0f}")
        log(f"  MWU p={p:.4e}")
        add("HAR overlap × age", len(m_age), u, p,
            f"in_har={len(in_har)}, out={len(out_har)}")


# ─── Test 6: Desert tier × eQTL ────────────────────────────────────────────
log("\n" + "=" * 72)
log("[Test 6] Introgression desert × Brain eQTL")
log("=" * 72)

if "desert_tier" in m_age.columns and "gtex_brain_minp" in m_age.columns:
    sub = m_age[m_age["gtex_brain_minp"].notna()].copy()
    sub["neglog_p"] = -np.log10(sub["gtex_brain_minp"].clip(lower=1e-300))
    in_des = sub.loc[sub["desert_tier"] > 0, "neglog_p"]
    out_des = sub.loc[sub["desert_tier"] == 0, "neglog_p"]
    log(f"  Desert variants with brain eQTL: {len(in_des)}, non-desert: {len(out_des)}")
    if len(in_des) >= 5 and len(out_des) >= 5:
        u, p = stats.mannwhitneyu(in_des, out_des, alternative="two-sided")
        log(f"  Median -log10 p in desert: {in_des.median():.2f}, non: {out_des.median():.2f}")
        log(f"  MWU p={p:.4e}")
        add("Desert × brain eQTL strength", len(sub), u, p)


# ─── Test 7: Brain vs Blood eQTL specificity × Age ─────────────────────────
log("\n" + "=" * 72)
log("[Test 7] Brain-vs-blood eQTL specificity × Age")
log("=" * 72)

if "gtex_brain_minp" in m_age.columns and "gtex_blood_minp" in m_age.columns:
    both = m_age[m_age["gtex_brain_minp"].notna() & m_age["gtex_blood_minp"].notna()].copy()
    log(f"  Variants with both brain + blood eQTL: {len(both)}")
    if len(both) > 30:
        both["b_score"] = -np.log10(both["gtex_brain_minp"].clip(lower=1e-300))
        both["bl_score"] = -np.log10(both["gtex_blood_minp"].clip(lower=1e-300))
        both["brain_specificity"] = both["b_score"] / (both["b_score"] + both["bl_score"])
        rho, p = stats.spearmanr(both["log_age"], both["brain_specificity"])
        log(f"  Spearman age × brain specificity: rho={rho:.4f}, p={p:.4e}")
        add("Brain specificity × age", len(both), rho, p)

    # Brain-only vs blood-only variants
    brain_only = m_age["gtex_brain_minp"].notna() & m_age["gtex_blood_minp"].isna()
    blood_only = m_age["gtex_blood_minp"].notna() & m_age["gtex_brain_minp"].isna()
    log(f"  Brain-only eQTL: {brain_only.sum()}, Blood-only eQTL: {blood_only.sum()}")
    if brain_only.sum() > 10 and blood_only.sum() > 10:
        ages_b = m_age.loc[brain_only, "age_median_yr"]
        ages_bl = m_age.loc[blood_only, "age_median_yr"]
        u, p = stats.mannwhitneyu(ages_b, ages_bl, alternative="two-sided")
        log(f"  Brain-only median age: {ages_b.median():.0f}, blood-only: {ages_bl.median():.0f}")
        log(f"  MWU p={p:.4e}")
        add("Brain-only vs blood-only eQTL × age", brain_only.sum() + blood_only.sum(), u, p)


# ─── Save results ──────────────────────────────────────────────────────────
log("\n" + "=" * 72)
log("[Final] Results Summary + BH-FDR")
log("=" * 72)

df = pd.DataFrame(results)
if len(df) > 0:
    from statsmodels.stats.multitest import multipletests
    rej, qvals, _, _ = multipletests(df["p"], method="fdr_bh")
    df["fdr_q"] = qvals
    df["sig_FDR05"] = rej

    log(f"\n  Test results ({len(df)} tests):")
    log(f"  {'test':<55s} {'n':>6s} {'p':>11s} {'q':>11s} {'sig':>4s}")
    log(f"  {'-'*55} {'-'*6} {'-'*11} {'-'*11} {'-'*4}")
    for _, r in df.iterrows():
        sig = "***" if r["sig_FDR05"] else ""
        log(f"  {r['test']:<55s} {r['n']:>6d} {r['p']:>11.3e} {r['fdr_q']:>11.3e} {sig:>4s}")

    df.to_csv(OUT / "P12_variant_results.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P12_variant_results.tsv'}")

# Save log
log_path = OUT / "P12_ANALYSIS_LOG.md"
with open(log_path, "w") as f:
    f.write("# Phase 12 Variant-Level Analysis Log\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log(f"  Log saved: {log_path}")
log("\nPhase 12 complete.")
