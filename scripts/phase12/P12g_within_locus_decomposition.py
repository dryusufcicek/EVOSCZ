#!/usr/bin/env python3
"""
Phase 12g v2: Within-Locus vs Between-Locus Decomposition (code-review-corrected)
=================================================================================
Code-review Faz C correction: prior `decomp` did mean-centering on raw values
then computed Spearman on the pooled centered values. This is NOT a within-
locus rank correlation; the centering altered the rank structure across loci
and biased ρ. New protocol: rank within each locus, then pool the centered
ranks, then Pearson on pooled centered ranks (= within-group partial Spearman).
Implemented in `_within_locus_lib.within_locus_partial_rank_correlation`.

Output:
  - results/phase12/P12g_decomposition_results.tsv (raw / within / between)
  - results/phase12/P12g_NARRATIVE.md
"""
import sys
from pathlib import Path
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _within_locus_lib import within_locus_partial_rank_correlation

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


log(f"Phase 12g v2: Within/Between Decomposition — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)
log("Code-review Faz C: replaced mean-centred-Spearman with within-group")
log("rank residualization + pooled Pearson (= within-group partial Spearman).")
log("")

m = pd.read_parquet(BASE / "results/phase11/variant_master_clean.parquet")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)

m_age["b_logp"] = np.where(
    m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0),
    -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan
)
m_age["bl_logp"] = np.where(
    m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0),
    -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan
)
m_age["brain_spec"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])


def decomp(df, x_col, y_col, label, dataset="all"):
    """Decompose into raw / within-locus partial rank / between-locus."""
    sub = df[[x_col, y_col, "credible_set_id"]].dropna().copy()
    if len(sub) < 30:
        return None

    # Raw (pooled Spearman)
    rho_raw, p_raw = stats.spearmanr(sub[x_col], sub[y_col])

    # Within-locus partial rank correlation (correct protocol)
    w = within_locus_partial_rank_correlation(
        sub, x_col, y_col, "credible_set_id", maf_col=None, min_n=5
    )

    # Between-locus (locus medians, then Spearman across loci)
    cs_med = sub.groupby("credible_set_id").agg(
        x=(x_col, "median"), y=(y_col, "median"), n=(x_col, "count")
    )
    cs_med = cs_med[cs_med["n"] >= 5]
    if len(cs_med) >= 10:
        rho_b, p_b = stats.spearmanr(cs_med["x"], cs_med["y"])
    else:
        rho_b, p_b = (np.nan, np.nan)

    return {
        "test": label,
        "dataset": dataset,
        "n_raw": len(sub),
        "n_loci_within": (w["n_groups"] if w else None),
        "n_pooled_within": (w["n_pooled"] if w else None),
        "n_loci_between": len(cs_med),
        "rho_raw": rho_raw, "p_raw": p_raw,
        "rho_within": (w["rho"] if w else np.nan),
        "p_within": (w["p"] if w else np.nan),
        "rho_between": rho_b, "p_between": p_b,
    }


results = []

log("\n[Tests with 17q21.31 included]")
for test_name, x, y in [
    ("Brain eQTL × age", "log_age", "b_logp"),
    ("Blood eQTL × age", "log_age", "bl_logp"),
    ("Brain specificity × age", "log_age", "brain_spec"),
    ("Brain eQTL × SDS", "sds", "b_logp"),
]:
    r = decomp(m_age, x, y, test_name, "all")
    if r:
        results.append(r)
        log(f"  {test_name}:")
        log(f"    raw     n={r['n_raw']:>5d}, rho={r['rho_raw']:>7.4f}, p={r['p_raw']:.3e}")
        log(f"    within  n_loci={r['n_loci_within']}, "
            f"n_pooled={r['n_pooled_within']}, rho={r['rho_within']:>7.4f}, "
            f"p={r['p_within']:.3e}")
        log(f"    between n_loci={r['n_loci_between']:>3d}, "
            f"rho={r['rho_between']:>7.4f}, p={r['p_between']:.3e}")

log("\n[Tests EXCLUDING 17q21.31]")
for test_name, x, y in [
    ("Brain eQTL × age", "log_age", "b_logp"),
    ("Blood eQTL × age", "log_age", "bl_logp"),
    ("Brain specificity × age", "log_age", "brain_spec"),
]:
    r = decomp(m_age[~in_mapt], x, y, test_name + " (no 17q21.31)", "no_mapt")
    if r:
        results.append(r)
        log(f"  {test_name} (no 17q21.31):")
        log(f"    raw     n={r['n_raw']:>5d}, rho={r['rho_raw']:>7.4f}, p={r['p_raw']:.3e}")
        log(f"    within  n_loci={r['n_loci_within']}, "
            f"n_pooled={r['n_pooled_within']}, rho={r['rho_within']:>7.4f}, "
            f"p={r['p_within']:.3e}")
        log(f"    between n_loci={r['n_loci_between']:>3d}, "
            f"rho={r['rho_between']:>7.4f}, p={r['p_between']:.3e}")


df = pd.DataFrame(results)
df.to_csv(OUT / "P12g_decomposition_results.tsv", sep="\t", index=False)
log(f"\nSaved: {OUT / 'P12g_decomposition_results.tsv'}")

# Narrative — re-emit on rerun, refer to current numbers
spec_r = next((r for r in results if r["test"] == "Brain specificity × age" and r["dataset"] == "all"), None)
spec_no = next((r for r in results if r["test"] == "Brain specificity × age (no 17q21.31)"), None)
brain_r = next((r for r in results if r["test"] == "Brain eQTL × age" and r["dataset"] == "all"), None)
brain_no = next((r for r in results if r["test"] == "Brain eQTL × age (no 17q21.31)"), None)
with open(OUT / "P12g_NARRATIVE.md", "w") as f:
    f.write("# Phase 12g v2: Within/Between-Locus Decomposition (code-review-corrected)\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## Methodology\n\nWithin-group partial rank correlation: each variant's covariate is\n")
    f.write("converted to a within-locus rank (average method); ranks are then\n")
    f.write("centered on the locus mean rank and pooled across loci; Pearson\n")
    f.write("correlation on pooled centered ranks gives the within-locus partial\n")
    f.write("Spearman ρ. Between-locus is Spearman across locus medians.\n\n")
    if brain_r:
        f.write("## Brain eQTL × Age\n\n| Component | n | ρ | p |\n|---|---|---|---|\n")
        f.write(f"| Raw | {brain_r['n_raw']} | {brain_r['rho_raw']:.4f} | {brain_r['p_raw']:.2e} |\n")
        f.write(f"| Within-locus (rank) | {brain_r['n_pooled_within']} (n_loci={brain_r['n_loci_within']}) | {brain_r['rho_within']:.4f} | {brain_r['p_within']:.2e} |\n")
        f.write(f"| Between-locus | {brain_r['n_loci_between']} | {brain_r['rho_between']:.4f} | {brain_r['p_between']:.2e} |\n")
        if brain_no:
            f.write(f"| Within-locus excl 17q21.31 | {brain_no['n_pooled_within']} | {brain_no['rho_within']:.4f} | {brain_no['p_within']:.2e} |\n")
        f.write("\n")
    if spec_r:
        f.write("## Brain Specificity × Age (PRIMARY WITHIN-LOCUS FINDING)\n\n| Component | n | ρ | p |\n|---|---|---|---|\n")
        f.write(f"| Raw | {spec_r['n_raw']} | {spec_r['rho_raw']:.4f} | {spec_r['p_raw']:.2e} |\n")
        f.write(f"| Within-locus (rank) | {spec_r['n_pooled_within']} (n_loci={spec_r['n_loci_within']}) | {spec_r['rho_within']:.4f} | {spec_r['p_within']:.2e} |\n")
        f.write(f"| Between-locus | {spec_r['n_loci_between']} | {spec_r['rho_between']:.4f} | {spec_r['p_between']:.2e} |\n")
        if spec_no:
            f.write(f"| Within-locus excl 17q21.31 | {spec_no['n_pooled_within']} | {spec_no['rho_within']:.4f} | {spec_no['p_within']:.2e} |\n")
        f.write("\n")

log(f"Narrative saved: {OUT / 'P12g_NARRATIVE.md'}")
log("\nPhase 12g v2 complete.")
