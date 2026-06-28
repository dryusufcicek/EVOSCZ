#!/usr/bin/env python3
"""
Phase 13b: MAF-Residualized Within-Locus Analyses
==================================================
Robustness check: |iHS|, eQTL detection, GEVA age, LOEUF gene-rarity all
correlate with MAF. The within-locus correlations may reflect MAF structure
rather than biological signal.

Solution: Double residualization within each credible set:
  1. Center variable on credible-set mean (within-locus residual, Phase 12g)
  2. Then residualize against MAF (linear regression residuals)
  → Pure within-locus, MAF-independent signal

Tests included:
- Brain specificity × age (primary)
- |iHS| × age (primary)
- Brain eQTL × age
- Blood eQTL × age
- Brain eQTL × LOEUF (constrained gene confound check)

Output:
  - results/phase13/P13b_maf_residualized_results.tsv
  - results/phase13/P13b_NARRATIVE.md
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
OUT = BASE / "results/phase13"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 13b: MAF-Residualized Within-Locus — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# Load v2 master (includes constraint + iHS)
m = pd.read_parquet(BASE / "results/phase11/variant_master_v2.parquet")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0) & m["maf"].notna()].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)

m_age["b_logp"]  = np.where(m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0),
                              -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m_age["bl_logp"] = np.where(m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0),
                              -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m_age["brain_spec"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])
m_age["abs_ihs"] = m_age["ihs_std"].abs() if "ihs_std" in m_age.columns else np.nan


def double_residualize(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation with MAF-rank covariate
    (code-review Faz C corrected: rank-within-group + within-group MAF-rank
    residualization, not the prior pooled-linear approach which left
    locus×MAF interaction structure)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


def within_only(df, x_col, y_col, locus_col="credible_set_id"):
    """Within-locus partial rank correlation (no MAF). Code-review Faz C."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=None, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


# ─── Run paired tests: within-only vs within+MAF ───────────────────────────
results = []
tests = [
    ("Brain spec × age (all)", "log_age", "brain_spec", m_age),
    ("Brain spec × age, no 17q21.31", "log_age", "brain_spec", m_age[~in_mapt]),
    ("|iHS| × age (all)", "log_age", "abs_ihs", m_age),
    ("|iHS| × age, no 17q21.31", "log_age", "abs_ihs", m_age[~in_mapt]),
    ("Brain eQTL × age", "log_age", "b_logp", m_age),
    ("Blood eQTL × age", "log_age", "bl_logp", m_age),
    ("Brain eQTL × LOEUF (constraint)", "loeuf", "b_logp", m_age),
    ("Brain spec × LOEUF", "loeuf", "brain_spec", m_age),
    ("|iHS| × SDS (consistency)", "sds", "abs_ihs", m_age),
    ("SDS × age", "log_age", "sds", m_age),
]

log("\n  Test                                     n_w        rho_w    p_w        rho_maf    p_maf")
log("  " + "-" * 105)
for label, x, y, df in tests:
    rho_w, p_w, n_w = within_only(df, x, y)
    rho_m, p_m, n_m = double_residualize(df, x, y)
    if rho_w is not None and rho_m is not None:
        # pct change
        pct = (rho_m - rho_w) / abs(rho_w) * 100 if abs(rho_w) > 1e-10 else 0
        results.append({
            "test": label,
            "n_within": n_w, "rho_within": rho_w, "p_within": p_w,
            "n_maf": n_m, "rho_maf": rho_m, "p_maf": p_m,
            "rho_pct_change": pct,
        })
        log(f"  {label:<40s}  {n_w:>5d}  {rho_w:>+7.4f}  {p_w:.2e}  {rho_m:>+7.4f}  {p_m:.2e}  Δ={pct:+5.1f}%")

df_r = pd.DataFrame(results)
if len(df_r) > 0:
    df_r["fdr_q_maf"] = multipletests(df_r["p_maf"].fillna(1.0), method="fdr_bh")[1]
    df_r.to_csv(OUT / "P13b_maf_residualized_results.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P13b_maf_residualized_results.tsv'}")

# Save log
with open(OUT / "P13b_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 13b: MAF-Residualized Within-Locus\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13b complete.")
