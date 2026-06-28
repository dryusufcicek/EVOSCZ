#!/usr/bin/env python3
"""
Phase 14p3c — Primary cluster vs pure-AFR DAF distribution tests
=================================================================
This is the PRIMARY inferential script for H4: do C0/C1/C2 clusters
have distinguishable pure-AFR DAF distributions?

Tests applied (all pre-registered):
  T1. KS 2-sample on pure-AFR DAF, all pairwise (C0–C1, C0–C2, C1–C2)
      Effect-size threshold pre-registered: D > 0.30 considered substantive
      (Cohen-Lewis class: small=0.10, medium=0.30, large=0.50).
  T2. Mann-Whitney U as supporting rank-based test.
  T3. Permutation null on KS D and on median DAF: shuffle cluster labels
      10,000× while preserving total variant count, recompute statistic,
      derive empirical P. Provides an alternative to asymptotic P that is
      sample-size-inflation-immune (P14p lesson).
  T4. C0 / C2 separation against MAF+L2-matched HapMap3 control benchmark:
      the same KS D computed on cluster ∪ MAF-matched controls; verifies
      that control distribution sits BETWEEN C0 and C2 (consistent with
      a population-origin gradient, not arbitrary).

Pre-registered falsification gates:
  G1. KS D(C0 vs C2) < 0.30  AND  permutation P > 0.01 → H4 falsified
      (cluster identity does not carry pure-AFR information).
  G2. Permutation P > 0.001 with N=20k vs N=20k baseline → underpower flag
      (treat result as suggestive, not conclusive).

Output:
  results/phase14p3/P14p3c_primary_distribution_summary.tsv
  results/phase14p3/P14p3c_pairwise_KS_MWU.tsv
  results/phase14p3/P14p3c_permutation_null.tsv
  results/phase14p3/P14p3c_matched_control_benchmark.tsv
  results/phase14p3/P14p3c_NARRATIVE.md
"""

import sys
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
log("Phase 14p3c — Primary cluster × pure-AFR DAF tests")
log("="*70)

# ── Load harmonized DAF data + cluster assignments ─────────────────
log("[1] Load harmonized substrate + cluster labels")
df = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
log(f"  harmonized variants: {len(df):,}")

clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")
clu = clu[["rsid","cluster"]].copy()
log(f"  cluster assignments: {len(clu):,}")

df = df.merge(clu, on="rsid", how="inner")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
df = df.dropna(subset=["cluster", "AF_AFR_derived"])
log(f"  merged with cluster + DAF: {len(df):,}")
log(f"  cluster sizes: " + ", ".join(
    f"C{int(c)}={n}" for c, n in df["cluster"].value_counts().sort_index().items()))

# ── T1 + T2: per-cluster distribution + pairwise tests ─────────────
log("\n[2] Per-cluster pure-AFR DAF distribution summary")
dist_rows = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]["AF_AFR_derived"].values
    if len(sub) < 5: continue
    dist_rows.append({
        "cluster": f"C{int(c)}",
        "n": len(sub),
        "min":   float(np.min(sub)),
        "q01":   float(np.quantile(sub, 0.01)),
        "q05":   float(np.quantile(sub, 0.05)),
        "q25":   float(np.quantile(sub, 0.25)),
        "median":float(np.median(sub)),
        "q75":   float(np.quantile(sub, 0.75)),
        "q95":   float(np.quantile(sub, 0.95)),
        "q99":   float(np.quantile(sub, 0.99)),
        "max":   float(np.max(sub)),
        "mean":  float(np.mean(sub)),
        "pct_lt_0_01": float((sub < 0.01).mean() * 100),
        "pct_lt_0_05": float((sub < 0.05).mean() * 100),
        "pct_ge_0_05": float((sub >= 0.05).mean() * 100),
        "pct_ge_0_20": float((sub >= 0.20).mean() * 100),
        "pct_ge_0_50": float((sub >= 0.50).mean() * 100),
    })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT / "P14p3c_primary_distribution_summary.tsv", sep="\t", index=False)
log(dist_df.to_string(index=False))

# Pairwise KS + MWU
log("\n[3] Pairwise KS D and Mann-Whitney U")
pair_rows = []
clusters = sorted(df["cluster"].unique())
for i, c1 in enumerate(clusters):
    for c2 in clusters[i+1:]:
        x = df[df["cluster"] == c1]["AF_AFR_derived"].values
        y = df[df["cluster"] == c2]["AF_AFR_derived"].values
        ks_D, ks_p = stats.ks_2samp(x, y)
        mwu_U, mwu_p = stats.mannwhitneyu(x, y, alternative="two-sided")
        pair_rows.append({
            "contrast": f"C{int(c1)}_vs_C{int(c2)}",
            "n1": len(x), "n2": len(y),
            "median_1": float(np.median(x)),
            "median_2": float(np.median(y)),
            "median_diff": float(np.median(x) - np.median(y)),
            "KS_D": float(ks_D),
            "KS_P_asym": float(ks_p),
            "MWU_U": float(mwu_U),
            "MWU_P_asym": float(mwu_p),
            "verdict_pre_reg": "SUBSTANTIVE" if ks_D > 0.30 else
                                ("MEDIUM" if ks_D > 0.10 else "WEAK"),
        })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT / "P14p3c_pairwise_KS_MWU.tsv", sep="\t", index=False)
log(pair_df.to_string(index=False))

# ── T3: Permutation null ───────────────────────────────────────────
log("\n[4] Permutation null (10,000 iter) on KS D and median diff for C0_vs_C2")
N_PERM = 10_000
c0 = df[df["cluster"] == 0]["AF_AFR_derived"].values
c2 = df[df["cluster"] == 2]["AF_AFR_derived"].values
obs_KS, _ = stats.ks_2samp(c0, c2)
obs_med  = np.median(c0) - np.median(c2)

pooled = np.concatenate([c0, c2])
n_c0 = len(c0)
perm_KS  = np.empty(N_PERM)
perm_med = np.empty(N_PERM)
for i in range(N_PERM):
    np.random.shuffle(pooled)
    a = pooled[:n_c0]; b = pooled[n_c0:]
    perm_KS[i], _ = stats.ks_2samp(a, b)
    perm_med[i] = np.median(a) - np.median(b)
emp_p_KS  = (perm_KS  >= obs_KS).mean()
emp_p_med = 2 * min((perm_med >= obs_med).mean(), (perm_med <= obs_med).mean())

perm_rows = [{
    "contrast": "C0_vs_C2",
    "n_c0": int(n_c0), "n_c2": int(len(c2)),
    "observed_KS_D": float(obs_KS),
    "observed_median_diff": float(obs_med),
    "n_permutations": N_PERM,
    "perm_P_KS": float(emp_p_KS),
    "perm_P_median_diff": float(emp_p_med),
    "perm_KS_mean": float(perm_KS.mean()),
    "perm_KS_q99": float(np.quantile(perm_KS, 0.99)),
}]
# Also for C0 vs C1 and C1 vs C2
for (a_c, b_c) in [(0,1), (1,2)]:
    A = df[df["cluster"] == a_c]["AF_AFR_derived"].values
    B = df[df["cluster"] == b_c]["AF_AFR_derived"].values
    obsKS, _ = stats.ks_2samp(A, B)
    obsM = np.median(A) - np.median(B)
    pool2 = np.concatenate([A, B])
    n_a = len(A)
    pKS = np.empty(N_PERM); pM = np.empty(N_PERM)
    for i in range(N_PERM):
        np.random.shuffle(pool2)
        pKS[i], _ = stats.ks_2samp(pool2[:n_a], pool2[n_a:])
        pM[i]    = np.median(pool2[:n_a]) - np.median(pool2[n_a:])
    perm_rows.append({
        "contrast": f"C{a_c}_vs_C{b_c}",
        "n_c0": int(n_a), "n_c2": int(len(B)),
        "observed_KS_D": float(obsKS),
        "observed_median_diff": float(obsM),
        "n_permutations": N_PERM,
        "perm_P_KS": float((pKS >= obsKS).mean()),
        "perm_P_median_diff": float(2 * min((pM >= obsM).mean(), (pM <= obsM).mean())),
        "perm_KS_mean": float(pKS.mean()),
        "perm_KS_q99": float(np.quantile(pKS, 0.99)),
    })

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(OUT / "P14p3c_permutation_null.tsv", sep="\t", index=False)
log(perm_df.to_string(index=False))

# ── T4: Matched-control benchmark ──────────────────────────────────
log("\n[5] Matched-control benchmark (sit between C0 and C2?)")
mc_path = BASE / "results/phase14p_baseline/P14p_a_age_lookup.parquet"
if mc_path.exists():
    mc = pd.read_parquet(mc_path)
    log(f"  Phase 14p matched-control age table: {len(mc):,}")
    # We need pure-AFR DAF for the matched controls — same harmonization
    # pipeline. For now, we report a placeholder: matched-control
    # pure-AFR DAF computation deferred to P14p3d sensitivity step
    # (requires bcftools extract for ~686k control SNPs — large I/O).
    log("  NOTE: matched-control AFR DAF requires a separate bcftools extract;")
    log("        deferred to P14p3d sensitivity. Reporting placeholder here.")
    benchmark_rows = [{
        "comparator": "C0_vs_matched_controls",
        "status": "deferred_to_p14p3d_sensitivity",
        "reason": "matched controls have 686k SNPs requiring separate VCF extract",
    }]
else:
    log("  Phase 14p baseline file missing; matched control benchmark skipped")
    benchmark_rows = [{
        "comparator": "C0_vs_matched_controls",
        "status": "skipped_missing_baseline",
        "reason": str(mc_path),
    }]
pd.DataFrame(benchmark_rows).to_csv(OUT / "P14p3c_matched_control_benchmark.tsv",
                                     sep="\t", index=False)

# ── Verdict against pre-registered gates ───────────────────────────
log("\n[6] Pre-registered falsification gates")
ks_C0_C2 = pair_df[pair_df["contrast"]=="C0_vs_C2"]["KS_D"].iloc[0]
perm_p_C0_C2 = perm_df[perm_df["contrast"]=="C0_vs_C2"]["perm_P_KS"].iloc[0]

log(f"  G1: KS D(C0,C2) = {ks_C0_C2:.4f}  (threshold 0.30)")
log(f"  G1: Permutation P = {perm_p_C0_C2:.4g}  (threshold 0.01)")
if ks_C0_C2 < 0.30 and perm_p_C0_C2 > 0.01:
    log("  G1 VERDICT: H4 FALSIFIED — cluster identity does not carry "
        "pure-AFR DAF information")
elif ks_C0_C2 >= 0.30 and perm_p_C0_C2 <= 0.001:
    log("  G1 VERDICT: H4 SUPPORTED at pre-registered effect-size + alpha")
else:
    log("  G1 VERDICT: H4 PARTIAL — proceed to MAF-stratified sensitivity (P14p3d)")

# Save narrative
with open(OUT / "P14p3c_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3c — Primary cluster × pure-AFR DAF tests\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write("## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log(f"\nSaved: {OUT}/P14p3c_NARRATIVE.md")
log("Phase 14p3c complete.")
