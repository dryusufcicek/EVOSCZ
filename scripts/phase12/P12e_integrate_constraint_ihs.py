#!/usr/bin/env python3
"""
Phase 12e: Integrate gnomAD Constraint + per-variant iHS into Variant Master
=============================================================================
Reads outputs from P11b (gnomad_constraint_genes.tsv) and P12d (P12d_ihs_per_variant.tsv)
and produces the final variant_master_v2.parquet with all annotations integrated.

Then runs:
  - Age × eQTL with constraint as covariate (does effect survive constraint control?)
  - iHS distribution + age correlation (true sweep test, replacing window-max nSL)

Output:
  - results/phase11/variant_master_v2.parquet
  - results/phase12/P12e_integrated_results.tsv
  - results/phase12/P12e_ANALYSIS_LOG.md
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
OUT11 = BASE / "results/phase11"
OUT12 = BASE / "results/phase12"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


log(f"Phase 12e: Integrate Constraint + iHS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)


# ─── Load v1 master + new annotations ──────────────────────────────────────
m = pd.read_parquet(OUT11 / "variant_master.parquet")
log(f"\nVariant master v1: {len(m)} rows")

# gnomAD constraint
constraint_path = OUT11 / "gnomad_constraint_genes.tsv"
if constraint_path.exists():
    constraint = pd.read_csv(constraint_path, sep="\t")
    log(f"gnomAD constraint loaded: {len(constraint)} genes")
    keep_cols = ["gene_symbol", "pLI", "oe_lof_upper", "mis_z", "lof_z"]
    constraint = constraint[[c for c in keep_cols if c in constraint.columns]]
    constraint = constraint.rename(columns={"oe_lof_upper": "loeuf"})
    m = m.merge(constraint, on="gene_symbol", how="left")
    n_pli = m["pLI"].notna().sum() if "pLI" in m.columns else 0
    log(f"  Variants with pLI: {n_pli}")
else:
    log(f"! gnomAD constraint file not found: {constraint_path}")

# Per-variant iHS
ihs_path = OUT12 / "P12d_ihs_per_variant.tsv"
if ihs_path.exists():
    ihs = pd.read_csv(ihs_path, sep="\t")
    log(f"iHS per variant: {len(ihs)} entries")
    ihs_keep = ihs[["rsid", "ihs_raw", "ihs_std"]]
    m = m.merge(ihs_keep, on="rsid", how="left")
    n_ihs = m["ihs_std"].notna().sum() if "ihs_std" in m.columns else 0
    log(f"  Variants with iHS: {n_ihs}")
else:
    log(f"! iHS file not found: {ihs_path}")


# ─── Save v2 master ────────────────────────────────────────────────────────
out_path = OUT11 / "variant_master_v2.parquet"
m.to_parquet(out_path, index=False, compression="snappy")
log(f"\nSaved: {out_path} ({len(m)} rows × {len(m.columns)} cols)")


# ─── Analyses ──────────────────────────────────────────────────────────────
m_uniq = m.drop_duplicates("rsid", keep="first")
m_age = m_uniq[m_uniq["age_median_yr"].notna() & (m_uniq["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)

results = []
def add(test, n, stat, p, note=""):
    results.append({"test": test, "n": n, "statistic": stat, "p": p, "note": note})


# ─── Test 1: Age × Brain eQTL with constraint covariate ───────────────────
log("\n" + "=" * 70)
log("[Test 1] Age × Brain eQTL with constraint (LOEUF) covariate")
log("=" * 70)

if "loeuf" in m_age.columns and "gtex_brain_minp" in m_age.columns:
    sub = m_age[m_age["gtex_brain_minp"].notna() & m_age["loeuf"].notna() &
                 (m_age["gtex_brain_minp"] > 0)].copy()
    sub["neg_p"] = -np.log10(sub["gtex_brain_minp"].clip(lower=1e-300))
    log(f"  n: {len(sub)}")
    if len(sub) > 30:
        # Spearman
        rho, p = stats.spearmanr(sub["log_age"], sub["neg_p"])
        log(f"  Raw Age × Brain eQTL: rho={rho:.4f}, p={p:.4e}")
        # Partial out LOEUF (lower = more constrained)
        from scipy.stats import linregress
        # Residualize neg_p against loeuf
        r1 = linregress(sub["loeuf"], sub["neg_p"])
        res_p = sub["neg_p"] - (r1.intercept + r1.slope * sub["loeuf"])
        # Residualize log_age against loeuf
        r2 = linregress(sub["loeuf"], sub["log_age"])
        res_age = sub["log_age"] - (r2.intercept + r2.slope * sub["loeuf"])
        # Spearman on residuals
        rho2, p2 = stats.spearmanr(res_age, res_p)
        log(f"  After LOEUF residual control: rho={rho2:.4f}, p={p2:.4e}")
        add("Age × brain eQTL (LOEUF-controlled)", len(sub), rho2, p2)


# ─── Test 2: Constraint × Age (do constrained genes have older variants?) ─
log("\n" + "=" * 70)
log("[Test 2] LOEUF × Age (constraint vs allele age)")
log("=" * 70)

if "loeuf" in m_age.columns:
    sub = m_age[m_age["loeuf"].notna()].copy()
    log(f"  n: {len(sub)}")
    if len(sub) > 30:
        rho, p = stats.spearmanr(sub["loeuf"], sub["log_age"])
        log(f"  LOEUF × log(age): rho={rho:.4f}, p={p:.4e}")
        log(f"  (negative rho expected: low LOEUF = constrained = ancient)")
        add("LOEUF × log(age)", len(sub), rho, p)

        # Top 20% constrained vs rest
        loeuf_q20 = sub["loeuf"].quantile(0.20)
        constrained = sub[sub["loeuf"] <= loeuf_q20]
        rest = sub[sub["loeuf"] > loeuf_q20]
        if len(constrained) > 20 and len(rest) > 20:
            u, pmw = stats.mannwhitneyu(constrained["age_median_yr"], rest["age_median_yr"],
                                         alternative="two-sided")
            log(f"  Top 20% constrained median age: {constrained['age_median_yr'].median():.0f}")
            log(f"  Rest median age: {rest['age_median_yr'].median():.0f}")
            log(f"  MWU p={pmw:.4e}")
            add("Constrained genes vs rest (age)", len(sub), u, pmw)


# ─── Test 3: iHS distribution + age correlation ────────────────────────────
log("\n" + "=" * 70)
log("[Test 3] iHS × Age — replace window-max nSL with per-variant iHS")
log("=" * 70)

if "ihs_std" in m_age.columns:
    sub = m_age[m_age["ihs_std"].notna()].copy()
    log(f"  n with valid iHS: {len(sub)}")
    if len(sub) > 30:
        # |iHS| × age (selection signature should appear in EXTREME |iHS|)
        sub["abs_ihs"] = sub["ihs_std"].abs()
        rho, p = stats.spearmanr(sub["abs_ihs"], sub["log_age"])
        log(f"  |iHS_std| × log(age): rho={rho:.4f}, p={p:.4e}")
        add("|iHS| × log(age)", len(sub), rho, p)

        # Variants with |iHS|>2 (95th percentile threshold)
        n_extreme = (sub["abs_ihs"] > 2).sum()
        log(f"  Variants with |iHS|>2: {n_extreme}/{len(sub)} ({n_extreme/len(sub)*100:.1f}%)")

        # Top 5% iHS — younger?
        thresh = sub["abs_ihs"].quantile(0.95)
        high = sub[sub["abs_ihs"] > thresh]
        low = sub[sub["abs_ihs"] <= thresh]
        if len(high) > 5 and len(low) > 30:
            u, pmw = stats.mannwhitneyu(high["age_median_yr"], low["age_median_yr"],
                                         alternative="less")
            log(f"  Top 5% |iHS| median age: {high['age_median_yr'].median():.0f}")
            log(f"  Rest median age: {low['age_median_yr'].median():.0f}")
            log(f"  MWU one-sided p={pmw:.4e}")
            add("High-iHS younger (top 5%)", len(sub), u, pmw)


# ─── Test 4: iHS × SDS (two recent-selection metrics, should agree) ────────
log("\n" + "=" * 70)
log("[Test 4] iHS × SDS — consistency between selection metrics")
log("=" * 70)

if "ihs_std" in m_age.columns and "sds" in m_age.columns:
    sub = m_age[m_age["ihs_std"].notna() & m_age["sds"].notna()].copy()
    log(f"  n with both iHS and SDS: {len(sub)}")
    if len(sub) > 30:
        # |iHS| vs |SDS|: both selection signatures
        rho, p = stats.spearmanr(sub["ihs_std"].abs(), sub["sds"].abs())
        log(f"  |iHS| × |SDS|: rho={rho:.4f}, p={p:.4e}")
        add("|iHS| × |SDS|", len(sub), rho, p)


# ─── FDR + save ────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
if len(df) > 0:
    from statsmodels.stats.multitest import multipletests
    rej, qvals, _, _ = multipletests(df["p"], method="fdr_bh")
    df["fdr_q"] = qvals
    df["sig_FDR05"] = rej

    log(f"\n  P12e Test Results ({len(df)} tests):")
    log(f"  {'test':<55s} {'n':>6s} {'p':>11s} {'q':>11s} {'sig':>4s}")
    for _, r in df.iterrows():
        sig = "***" if r["sig_FDR05"] else ""
        log(f"  {r['test']:<55s} {r['n']:>6d} {r['p']:>11.3e} {r['fdr_q']:>11.3e} {sig:>4s}")

    df.to_csv(OUT12 / "P12e_integrated_results.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT12 / 'P12e_integrated_results.tsv'}")

# Save log
log_path = OUT12 / "P12e_ANALYSIS_LOG.md"
with open(log_path, "w") as f:
    f.write("# Phase 12e Integration Log\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log(f"\n  Log saved: {log_path}")
log("Phase 12e complete.")
