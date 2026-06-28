#!/usr/bin/env python3
"""
Phase 14p3d — EUR-MAF-stratified sensitivity for the pure-AFR DAF × cluster signal
====================================================================================
The CRITICAL sensitivity test for H4:

If C0 / C2 cluster identity is collinear with EUR MAF (i.e., C0 is mostly
low-MAF and C2 is mostly high-MAF in EUR, and the AFR DAF separation just
mirrors that), then **conditioning on EUR-MAF decile should collapse the
KS D**. If the signal SURVIVES decile stratification, cluster identity
carries pure-AFR information BEYOND EUR-MAF — which is the H4 prediction.

Approach:
  1. Compute KS D for C0_vs_C2 WITHIN each EUR-MAF decile (10 deciles
     defined on the PGC3 credible-set MAF distribution).
  2. Test consistency of the within-decile KS D:
     - Median within-decile KS D
     - Number of deciles with KS D > 0.20 (substantive within stratum)
     - Stouffer's combined Z across deciles
  3. Falsification:
     - G3a (collapse): median within-decile KS D < 0.15 → SUBSTANTIAL
       portion of the pooled signal is EUR-MAF-mediated; H4 weakened.
     - G3b (survival): median within-decile KS D ≥ 0.20 AND Stouffer Z
       > 4 → H4 supported beyond MAF collinearity.

Output:
  results/phase14p3/P14p3d_maf_stratified_KS.tsv
  results/phase14p3/P14p3d_stouffer_summary.tsv
  results/phase14p3/P14p3d_NARRATIVE.md
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
OUT = BASE / "results/phase14p3"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*70)
log("Phase 14p3d — EUR-MAF stratified sensitivity")
log("="*70)

df = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = df.merge(clu, on="rsid", how="inner").dropna(subset=["cluster","AF_AFR_derived","maf"])
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype(int)
log(f"  variants with cluster + DAF + EUR-MAF: {len(df):,}")

# Define MAF deciles on the full credible-set MAF distribution
maf_bins = np.quantile(df["maf"].values, np.linspace(0, 1, 11))
maf_bins[0]  = 0
maf_bins[-1] = 1
df["maf_decile"] = pd.cut(df["maf"], maf_bins, labels=False, include_lowest=True) + 1

log("\n[1] MAF decile boundaries:")
for i, (lo, hi) in enumerate(zip(maf_bins[:-1], maf_bins[1:])):
    log(f"  decile {i+1}: {lo:.4f}–{hi:.4f}")

log("\n[2] Within-decile KS D for C0_vs_C2, C0_vs_C1, C1_vs_C2")
rows = []
for d in range(1, 11):
    sub = df[df["maf_decile"] == d]
    if len(sub) < 50:
        log(f"  decile {d}: undersized (n={len(sub)}), skipping"); continue
    for (a,b) in [(0,2), (0,1), (1,2)]:
        A = sub[sub["cluster"] == a]["AF_AFR_derived"].values
        B = sub[sub["cluster"] == b]["AF_AFR_derived"].values
        if len(A) < 5 or len(B) < 5: continue
        ks_D, ks_p = stats.ks_2samp(A, B)
        mwu_U, mwu_p = stats.mannwhitneyu(A, B, alternative="two-sided")
        # z-stat for stouffer: convert two-sided p to z (signed by direction)
        sign = 1 if np.median(A) < np.median(B) else -1
        z = sign * stats.norm.isf(ks_p / 2)
        rows.append({
            "maf_decile": d,
            "maf_lo": float(maf_bins[d-1]), "maf_hi": float(maf_bins[d]),
            "contrast": f"C{a}_vs_C{b}",
            "n_a": len(A), "n_b": len(B),
            "median_a": float(np.median(A)),
            "median_b": float(np.median(B)),
            "median_diff": float(np.median(A) - np.median(B)),
            "KS_D": float(ks_D),
            "KS_P": float(ks_p),
            "MWU_P": float(mwu_p),
            "z_stouffer": float(z),
        })

strat = pd.DataFrame(rows)
strat.to_csv(OUT / "P14p3d_maf_stratified_KS.tsv", sep="\t", index=False)
log("\n  Within-decile KS D (printed):")
log(strat.to_string(index=False))

# Stouffer combined Z per contrast
log("\n[3] Stouffer's combined Z per contrast")
stouffer_rows = []
for contrast in strat["contrast"].unique():
    sub = strat[strat["contrast"] == contrast]
    z_sum = sub["z_stouffer"].sum() / np.sqrt(len(sub))
    p_stouffer = stats.norm.sf(abs(z_sum)) * 2
    median_ks = float(sub["KS_D"].median())
    n_sub_substantive = int((sub["KS_D"] > 0.20).sum())
    n_strata = len(sub)
    verdict = ("SURVIVES" if median_ks >= 0.20 and abs(z_sum) > 4
                else ("MIXED" if median_ks >= 0.10 else "COLLAPSES"))
    stouffer_rows.append({
        "contrast": contrast,
        "n_deciles_tested": n_strata,
        "median_within_decile_KS_D": median_ks,
        "n_deciles_substantive_D_gt_0_20": n_sub_substantive,
        "stouffer_Z": float(z_sum),
        "stouffer_P_two_sided": float(p_stouffer),
        "verdict_pre_registered": verdict,
    })

stouffer = pd.DataFrame(stouffer_rows)
stouffer.to_csv(OUT / "P14p3d_stouffer_summary.tsv", sep="\t", index=False)
log(stouffer.to_string(index=False))

# Pre-reg verdict
log("\n[4] Pre-registered falsification gates")
ksm_C0C2 = stouffer[stouffer["contrast"]=="C0_vs_C2"]["median_within_decile_KS_D"].iloc[0]
zs_C0C2  = stouffer[stouffer["contrast"]=="C0_vs_C2"]["stouffer_Z"].iloc[0]
log(f"  G3 (C0 vs C2 stratified):")
log(f"    median within-decile KS D = {ksm_C0C2:.3f}  (threshold 0.20)")
log(f"    Stouffer Z = {zs_C0C2:.2f}  (threshold 4.0)")

if ksm_C0C2 < 0.15:
    log("    G3a VERDICT: SIGNAL COLLAPSES under MAF stratification — H4 weakened")
elif ksm_C0C2 >= 0.20 and abs(zs_C0C2) > 4:
    log("    G3b VERDICT: H4 SURVIVES — cluster identity carries pure-AFR info "
        "beyond EUR-MAF")
else:
    log("    G3 VERDICT: PARTIAL — interpret with caution")

with open(OUT / "P14p3d_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3d — EUR-MAF stratified sensitivity\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p3d complete.")
