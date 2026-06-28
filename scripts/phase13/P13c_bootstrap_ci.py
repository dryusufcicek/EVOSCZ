#!/usr/bin/env python3
"""
Phase 13c: Bootstrap 95% Confidence Intervals
==============================================
For primary findings, compute 1000-iteration bootstrap CI to provide
precision estimates beyond p-values.

Tests:
  - Brain spec × age (within-locus, MAF-residualized, no 17q21.31)
  - |iHS| × age (within-locus, MAF-residualized, no 17q21.31)
  - Both with full sample as well

Bootstrap method: per-locus block bootstrap (resample credible sets with
replacement, preserving within-locus structure).

Output:
  - results/phase13/P13c_bootstrap_ci.tsv
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
OUT = BASE / "results/phase13"

sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import (
    within_locus_partial_rank_correlation,
    per_locus_bootstrap_ci,
)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 13c: Bootstrap CI — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)

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


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation (Faz C lib)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None
    return r["rho"], r["n_pooled"]


def block_bootstrap_ci(df, x_col, y_col, n_iter=1000, locus_col="credible_set_id",
                        maf_col="maf", seed=42):
    """Per-locus block bootstrap that RECOMPUTES within-locus residualization
    inside each iteration (code-review Faz D fix-13c-1: prior code precomputed
    residuals once on full data, so CI did not reflect uncertainty from the
    residualization step). Delegates to per_locus_bootstrap_ci in
    _within_locus_lib."""
    return per_locus_bootstrap_ci(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5,
        n_iter=n_iter, seed=seed
    )


tests = [
    ("Brain spec × age, no 17q21.31", "log_age", "brain_spec", m_age[~in_mapt]),
    ("Brain spec × age (all)", "log_age", "brain_spec", m_age),
    ("|iHS| × age, no 17q21.31", "log_age", "abs_ihs", m_age[~in_mapt]),
    ("|iHS| × age (all)", "log_age", "abs_ihs", m_age),
]

results = []
for ti, (label, x, y, df) in enumerate(tests):
    sub = df[[x, y, "credible_set_id", "maf"]].dropna().copy()
    rho_obs, n = maf_resid_rho(sub, x, y)
    log(f"\n  {label}: observed rho = {rho_obs} (n={n})")
    log(f"    Running 1000 block bootstraps (residualize-inside-loop) ...")
    ci = block_bootstrap_ci(sub, x, y, n_iter=1000, seed=42 + ti)
    if ci is None:
        log(f"    !! bootstrap returned None")
        continue
    log(f"    point-est ρ = {ci['rho_point']:.4f}, p = {ci['p_point']:.3e}")
    log(f"    bootstrap mean = {ci['rho_boot_mean']:.4f}")
    log(f"    95% CI: [{ci['rho_ci95_lower']:.4f}, {ci['rho_ci95_upper']:.4f}]")
    results.append({
        "test": label,
        "n_pooled": ci["n_pooled"],
        "n_loci": ci["n_groups"],
        "rho_point": ci["rho_point"],
        "p_point": ci["p_point"],
        "boot_mean": ci["rho_boot_mean"],
        "ci_lower": ci["rho_ci95_lower"],
        "ci_upper": ci["rho_ci95_upper"],
        "n_iter_success": ci["n_iter_success"],
    })

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "P13c_bootstrap_ci.tsv", sep="\t", index=False)
log(f"\n  Saved: {OUT / 'P13c_bootstrap_ci.tsv'}")

with open(OUT / "P13c_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 13c: Bootstrap CI\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13c complete.")
