#!/usr/bin/env python3
"""
Phase 12h: Final Integration — gnomAD constraint + per-variant iHS + within-locus
==================================================================================
Combines all Phase 11/12 outputs into final variant_master_v2.parquet, then runs
the corrected within-locus decomposition for ALL key tests, including new ones
involving constraint and iHS.

Output:
  - results/phase11/variant_master_v2.parquet (clean dedup + all annotations)
  - results/phase12/P12h_final_results.tsv
  - results/phase12/P12h_FINAL_NARRATIVE.md
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
P12 = BASE / "results/phase12"

sys.path.insert(0, str(P12.parent.parent / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 12h: Final Integration — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)

m = pd.read_parquet(P11 / "variant_master_clean.parquet")
log(f"Clean variant master: {len(m)} unique variants")

# Add gnomAD constraint
constraint_path = P11 / "gnomad_constraint_genes.tsv"
if constraint_path.exists():
    constraint = pd.read_csv(constraint_path, sep="\t")
    log(f"gnomAD constraint: {len(constraint)} genes ({constraint['pLI'].notna().sum()} with pLI)")
    keep = ["gene_symbol", "pLI", "oe_lof_upper", "mis_z", "lof_z"]
    keep = [c for c in keep if c in constraint.columns]
    constraint = constraint[keep].rename(columns={"oe_lof_upper": "loeuf"})
    m = m.merge(constraint, on="gene_symbol", how="left")
    log(f"  Variants with pLI: {m['pLI'].notna().sum() if 'pLI' in m.columns else 0}")

# Add per-variant iHS
ihs_path = P12 / "P12d_ihs_per_variant.tsv"
if ihs_path.exists():
    ihs = pd.read_csv(ihs_path, sep="\t")
    log(f"iHS per variant: {len(ihs)} entries")
    m = m.merge(ihs[["rsid", "ihs_raw", "ihs_std"]], on="rsid", how="left")
    log(f"  Variants with iHS: {m['ihs_std'].notna().sum() if 'ihs_std' in m.columns else 0}")

# Save final master
out = P11 / "variant_master_v2.parquet"
m.to_parquet(out, index=False, compression="snappy")
log(f"Saved: {out}")


# ─── Within-locus decomposition for all key tests ──────────────────────────
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)

# Brain specificity
m_age["b_logp"]  = np.where(m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0),
                              -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m_age["bl_logp"] = np.where(m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0),
                              -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m_age["brain_spec"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])
m_age["abs_ihs"] = m_age["ihs_std"].abs() if "ihs_std" in m_age.columns else np.nan


def within_locus_test(df, x, y, label):
    """Within-group partial rank correlation (code-review Faz C corrected)."""
    r = within_locus_partial_rank_correlation(
        df, x, y, "credible_set_id", maf_col=None, min_n=5
    )
    if r is None:
        return None
    return {
        "test": label, "n_residuals": r["n_pooled"], "n_loci": r["n_groups"],
        "rho": r["rho"], "p": r["p"],
    }


results = []
for test_name, x, y, df in [
    # Existing
    ("Brain spec × age (within)", "log_age", "brain_spec", m_age),
    ("Brain spec × age, no 17q21.31", "log_age", "brain_spec", m_age[~in_mapt]),
    ("Brain eQTL × age (within)", "log_age", "b_logp", m_age),
    ("Brain eQTL × age, no 17q21.31", "log_age", "b_logp", m_age[~in_mapt]),
    ("Blood eQTL × age (within)", "log_age", "bl_logp", m_age),
    # New with constraint + iHS
    ("LOEUF × age (within)", "log_age", "loeuf", m_age),
    ("Brain eQTL × LOEUF (within)", "loeuf", "b_logp", m_age),
    ("Brain spec × LOEUF (within)", "loeuf", "brain_spec", m_age),
    ("|iHS| × age (within)", "log_age", "abs_ihs", m_age),
    ("|iHS| × SDS (within)", "sds", "abs_ihs", m_age),
]:
    if x in df.columns and y in df.columns:
        r = within_locus_test(df, x, y, test_name)
        if r:
            results.append(r)
            log(f"  {test_name}: n={r['n_residuals']}, rho={r['rho']:.4f}, p={r['p']:.3e}")

if results:
    df_r = pd.DataFrame(results)
    df_r["fdr_q"] = multipletests(df_r["p"], method="fdr_bh")[1]
    df_r["sig"] = df_r["fdr_q"] < 0.05
    df_r.to_csv(P12 / "P12h_final_results.tsv", sep="\t", index=False)
    log(f"\nSaved: {P12 / 'P12h_final_results.tsv'}")

# Save log
with open(P12 / "P12h_FINAL_NARRATIVE.md", "w") as f:
    f.write("# Phase 12h: Final Integration & Within-Locus Tests\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log("\nPhase 12h complete.")
