#!/usr/bin/env python3
"""
Phase 13d: Brain Specificity Metric Robustness
================================================
Robustness check: Brain specificity = -log10(p_brain) / [-log10(p_brain) + -log10(p_blood)]
is mathematically unstable when one p-value approaches 1.

Solution: Test multiple alternative formulations and check that all give
qualitatively the same age dissociation:

  M1. Original ratio (-log10(p_brain) / [-log10(p_brain) + -log10(p_blood)])
  M2. Log10 ratio: log10(brain_p) - log10(blood_p)  → positive = blood-stronger
       (Note this is a continuous ratio of strengths; sign is preserved)
  M3. Z-score difference: zscore(neglog_brain) - zscore(neglog_blood) [after global z]
  M4. Tissue dominant: ifelse(brain_minp < blood_minp, 1, -1) — categorical
  M5. ABS slope difference: |slope_brain| - |slope_blood|

All tested within-locus, MAF-residualized, with and without 17q21.31.

Output:
  - results/phase13/P13d_robustness_results.tsv
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
from _within_locus_lib import within_locus_partial_rank_correlation

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 13d: Brain Specificity Robustness — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)

m = pd.read_parquet(BASE / "results/phase11/variant_master_v2.parquet")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0) & m["maf"].notna()].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)


# Compute different brain-specificity metrics
mask = m_age["gtex_brain_minp"].notna() & m_age["gtex_blood_minp"].notna() & \
       (m_age["gtex_brain_minp"] > 0) & (m_age["gtex_blood_minp"] > 0)
m_age = m_age[mask].copy()
m_age["b_logp"] = -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300))
m_age["bl_logp"] = -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300))

# M1: original ratio
m_age["spec_M1_ratio"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])

# M2: log10 ratio (brain stronger = negative)
m_age["spec_M2_logratio"] = np.log10(m_age["gtex_brain_minp"]) - np.log10(m_age["gtex_blood_minp"])
# (negative = brain p smaller = brain stronger)

# M3: Z-score difference
b_mean, b_std = m_age["b_logp"].mean(), m_age["b_logp"].std()
bl_mean, bl_std = m_age["bl_logp"].mean(), m_age["bl_logp"].std()
m_age["spec_M3_zdiff"] = ((m_age["b_logp"] - b_mean) / b_std) - ((m_age["bl_logp"] - bl_mean) / bl_std)

# M4: categorical dominance
m_age["spec_M4_cat"] = np.where(m_age["gtex_brain_minp"] < m_age["gtex_blood_minp"], 1, -1)

# M5: absolute slope difference (need slopes available)
if "gtex_brain_slope" in m_age.columns and "gtex_blood_slope" in m_age.columns:
    m_age["spec_M5_slopediff"] = m_age["gtex_brain_slope"].abs() - m_age["gtex_blood_slope"].abs()

log(f"  Variants with both brain + blood eQTL: {len(m_age)}")


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation with MAF-rank covariate
    (code-review Faz C corrected; uses _within_locus_lib helper)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


# ─── Test all metrics ──────────────────────────────────────────────────────
metrics = ["spec_M1_ratio", "spec_M2_logratio", "spec_M3_zdiff", "spec_M4_cat"]
if "spec_M5_slopediff" in m_age.columns:
    metrics.append("spec_M5_slopediff")

# Expected sign per metric
# M1: more brain = more pos value → negative rho with age means older = less brain
# M2: more brain = more NEG value → positive rho with age means older = less brain
# M3: more brain = more pos → negative rho means older = less brain
# M4: brain dominant = +1 → negative rho means older = less brain dominant
# M5: brain stronger slope = positive → negative rho means older = brain weaker than blood

results = []
log(f"\n  {'Metric':<22s} {'Subset':<20s} {'n':>7s} {'rho':>9s} {'p':>10s}")
log("  " + "-" * 72)
for metric in metrics:
    if metric not in m_age.columns:
        continue
    # Excl-MAPT mask aligned to current m_age index
    mapt_mask = (m_age["chr_int"] == 17) & \
                (m_age["pos_int"] >= 43_000_000) & \
                (m_age["pos_int"] <= 46_000_000)
    for label, df in [("All variants", m_age), ("Excl 17q21.31", m_age[~mapt_mask])]:
        rho, p, n = maf_resid_rho(df, "log_age", metric)
        if rho is not None:
            results.append({
                "metric": metric,
                "subset": label,
                "n": n,
                "rho": rho,
                "p": p,
            })
            log(f"  {metric:<22s} {label:<20s} {n:>7d}  {rho:>+8.4f} {p:>10.2e}")

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "P13d_robustness_results.tsv", sep="\t", index=False)
log(f"\n  Saved: {OUT / 'P13d_robustness_results.tsv'}")

# Also: do all metrics give the same direction?
log("\n  Direction consistency check (Brain spec × age):")
log("  Expected: M1 negative, M2 positive, M3 negative, M4 negative, M5 negative")
log("  (i.e., older = less brain-specific in all formulations)")

with open(OUT / "P13d_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 13d: Brain Specificity Robustness\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13d complete.")
