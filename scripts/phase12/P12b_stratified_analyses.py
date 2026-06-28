#!/usr/bin/env python3
"""
Phase 12b: Stratified Analyses (17q21.31 exclusion + per-locus heatmap)
=======================================================================
Re-runs Phase 12 tests excluding the MAPT inversion (17q21.31) to isolate
genome-wide signal from this single dominant locus.

Also produces per-locus brain/blood eQTL specificity profile.

Output:
  - results/phase12/P12b_stratified_results.tsv
  - results/phase12/P12b_per_locus_specificity.tsv
  - results/phase12/P12b_ANALYSIS_LOG.md
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
OUT = BASE / "results/phase12"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


log(f"Phase 12b: 17q21.31 Stratification — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

m = pd.read_parquet(BASE / "results/phase11/variant_master.parquet")
m = m.drop_duplicates("rsid", keep="first")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")

# 17q21.31 region (MAPT inversion)
MAPT_REGION = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
              (m_age["pos_int"] <= 46_000_000)
n_mapt = MAPT_REGION.sum()
log(f"\nVariants in 17q21.31 (chr17:43-46Mb): {n_mapt} / {len(m_age)} ({n_mapt/len(m_age)*100:.1f}%)")


# ─── Section 1: Stratified eQTL × Age tests ────────────────────────────────
log("\n" + "=" * 72)
log("[1] Brain & Blood eQTL × Age — 17q21.31 stratification")
log("=" * 72)

results = []

def test_subset(df, brain_col, blood_col, label):
    """Returns dict of test results for this subset."""
    out = {"subset": label, "n_total": len(df)}

    # Brain
    bsub = df[df[brain_col].notna() & (df[brain_col] > 0)].copy()
    if len(bsub) > 30:
        bsub["neg_p"] = -np.log10(bsub[brain_col].clip(lower=1e-300))
        rho, p = stats.spearmanr(bsub["log_age"], bsub["neg_p"])
        out.update({"brain_n": len(bsub), "brain_rho": rho, "brain_p": p})
    else:
        out.update({"brain_n": len(bsub), "brain_rho": None, "brain_p": None})

    # Blood
    blsub = df[df[blood_col].notna() & (df[blood_col] > 0)].copy()
    if len(blsub) > 30:
        blsub["neg_p"] = -np.log10(blsub[blood_col].clip(lower=1e-300))
        rho, p = stats.spearmanr(blsub["log_age"], blsub["neg_p"])
        out.update({"blood_n": len(blsub), "blood_rho": rho, "blood_p": p})
    else:
        out.update({"blood_n": len(blsub), "blood_rho": None, "blood_p": None})

    # Brain specificity (when both available)
    both = df[df[brain_col].notna() & df[blood_col].notna() &
              (df[brain_col] > 0) & (df[blood_col] > 0)].copy()
    if len(both) > 30:
        both["b"]  = -np.log10(both[brain_col].clip(lower=1e-300))
        both["bl"] = -np.log10(both[blood_col].clip(lower=1e-300))
        both["spec"] = both["b"] / (both["b"] + both["bl"])
        rho, p = stats.spearmanr(both["log_age"], both["spec"])
        out.update({"spec_n": len(both), "spec_rho": rho, "spec_p": p})
    else:
        out.update({"spec_n": len(both), "spec_rho": None, "spec_p": None})

    return out


full = test_subset(m_age, "gtex_brain_minp", "gtex_blood_minp", "All variants (n=20,565)")
no_mapt = test_subset(m_age[~MAPT_REGION], "gtex_brain_minp", "gtex_blood_minp",
                       "Excluding 17q21.31")
only_mapt = test_subset(m_age[MAPT_REGION], "gtex_brain_minp", "gtex_blood_minp",
                         "Only 17q21.31")

# Pretty print
log(f"\n  {'Test':<35s} {'All':>15s} {'No 17q21.31':>15s} {'Only 17q21.31':>15s}")
log(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*15}")
for k in [("brain_rho", "brain_n", "Brain eQTL × age (rho)"),
          ("blood_rho", "blood_n", "Blood eQTL × age (rho)"),
          ("spec_rho", "spec_n", "Brain specificity × age (rho)")]:
    rho_k, n_k, label = k
    log(f"  {label:<35s} {full[rho_k]:>11.4f} (n={full[n_k]:5d}) "
        f"{no_mapt[rho_k]:>9.4f} (n={no_mapt[n_k]:5d}) "
        f"{only_mapt[rho_k]:>9.4f} (n={only_mapt[n_k]:5d})")
log("\n  P-values:")
for k in [("brain_p", "brain_n", "Brain eQTL × age"),
          ("blood_p", "blood_n", "Blood eQTL × age"),
          ("spec_p", "spec_n", "Brain specificity × age")]:
    pk, nk, label = k
    log(f"  {label:<35s} {full[pk]:>15.3e} {no_mapt[pk]:>15.3e} {only_mapt[pk]:>15.3e}")

# Save stratified
strat_rows = []
for d, lbl in [(full, "all"), (no_mapt, "no_mapt"), (only_mapt, "only_mapt")]:
    for tissue in ["brain", "blood"]:
        if d[f"{tissue}_rho"] is not None:
            strat_rows.append({
                "subset": lbl, "test": f"{tissue}_eqtl_x_age",
                "n": d[f"{tissue}_n"], "rho": d[f"{tissue}_rho"],
                "p": d[f"{tissue}_p"]
            })
    if d["spec_rho"] is not None:
        strat_rows.append({"subset": lbl, "test": "brain_specificity_x_age",
                           "n": d["spec_n"], "rho": d["spec_rho"], "p": d["spec_p"]})

strat_df = pd.DataFrame(strat_rows)
# FDR
ps = strat_df["p"].dropna().values
if len(ps) > 0:
    _, q, _, _ = multipletests(ps, method="fdr_bh")
    strat_df["fdr_q"] = np.nan
    strat_df.loc[strat_df["p"].notna(), "fdr_q"] = q
strat_df.to_csv(OUT / "P12b_stratified_results.tsv", sep="\t", index=False)
log(f"\n  Saved: {OUT / 'P12b_stratified_results.tsv'}")


# ─── Section 2: Per-locus brain/blood specificity heatmap ─────────────────
log("\n" + "=" * 72)
log("[2] Per-locus brain/blood eQTL specificity profile")
log("=" * 72)

# Aggregate per credible_set_id
both_eqtl = m_age[m_age["gtex_brain_minp"].notna() & m_age["gtex_blood_minp"].notna() &
                   (m_age["gtex_brain_minp"] > 0) & (m_age["gtex_blood_minp"] > 0)].copy()
both_eqtl["b"]  = -np.log10(both_eqtl["gtex_brain_minp"].clip(lower=1e-300))
both_eqtl["bl"] = -np.log10(both_eqtl["gtex_blood_minp"].clip(lower=1e-300))
both_eqtl["spec"] = both_eqtl["b"] / (both_eqtl["b"] + both_eqtl["bl"])

if "credible_set_id" in both_eqtl.columns:
    locus = both_eqtl.groupby("credible_set_id").agg({
        "spec": ["mean", "median", "std", "count"],
        "age_median_yr": "median",
        "log_age": "median",
        "b": "mean",
        "bl": "mean",
        "chr": "first",
        "pos": "median",
    })
    locus.columns = ["_".join(c) for c in locus.columns]
    locus = locus.reset_index()
    locus = locus.rename(columns={
        "spec_mean": "brain_specificity_mean",
        "spec_count": "n_variants_with_both",
        "age_median_yr_median": "median_age_yr",
        "chr_first": "chr",
    })
    locus = locus.sort_values("brain_specificity_mean", ascending=False)
    log(f"  Loci with ≥2 variants in both tissues: {(locus['n_variants_with_both']>=2).sum()}")
    log(f"\n  Top 10 brain-specific loci:")
    log(f"  {'CS_ID':<10s} {'chr':>4s} {'spec':>8s} {'n':>4s} {'age':>10s}")
    for _, r in locus.head(10).iterrows():
        log(f"  {r['credible_set_id']:<10s} {r['chr']:>4} {r['brain_specificity_mean']:>8.3f} "
            f"{int(r['n_variants_with_both']):>4d} {r['median_age_yr']:>10.0f}")

    log(f"\n  Top 10 blood-specific loci:")
    for _, r in locus.tail(10).iloc[::-1].iterrows():
        log(f"  {r['credible_set_id']:<10s} {r['chr']:>4} {r['brain_specificity_mean']:>8.3f} "
            f"{int(r['n_variants_with_both']):>4d} {r['median_age_yr']:>10.0f}")

    locus.to_csv(OUT / "P12b_per_locus_specificity.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P12b_per_locus_specificity.tsv'} ({len(locus)} loci)")


# ─── Section 3: SDS × age stratified ───────────────────────────────────────
log("\n" + "=" * 72)
log("[3] SDS × Age stratification")
log("=" * 72)

for label, mask in [("All", slice(None)), ("Excl 17q21.31", ~MAPT_REGION),
                    ("Only 17q21.31", MAPT_REGION)]:
    sub = m_age[mask] if not isinstance(mask, slice) else m_age
    sub = sub[sub["sds"].notna()]
    if len(sub) > 30:
        rho, p = stats.spearmanr(sub["log_age"], sub["sds"])
        log(f"  {label:<20s}: n={len(sub):>6d}, rho={rho:>8.4f}, p={p:>11.3e}")


# ─── Save log ──────────────────────────────────────────────────────────────
log_path = OUT / "P12b_ANALYSIS_LOG.md"
with open(log_path, "w") as f:
    f.write("# Phase 12b Stratified Analysis Log\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log(f"\nLog saved: {log_path}")
log("Phase 12b complete.")
