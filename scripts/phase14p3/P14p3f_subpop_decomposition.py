#!/usr/bin/env python3
"""
Phase 14p3f — Per-AFR-subpopulation independent replication (5 subpops × 3 contrasts)
======================================================================================
The CRITICAL independent-replication test: same KS D for C0_vs_C2 should
appear when computed independently in each of the 5 pure-AFR
subpopulations: ESN, GWD, LWK, MSL, YRI. If the signal is real, it should
NOT depend on any one subpop.

Pre-registered:
  - In ≥4/5 subpops, KS D(C0,C2) > 0.20 with P < 0.01 → REPLICATED.
  - Signal restricted to 1-2 subpops only → spurious / population-
    structure-driven; H4 reframed.

Output:
  results/phase14p3/P14p3f_subpop_replication.tsv
  results/phase14p3/P14p3f_NARRATIVE.md
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
LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*70); log("Phase 14p3f — Per-AFR-subpop replication"); log("="*70)

df = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = df.merge(clu, on="rsid", how="inner")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
df = df.dropna(subset=["cluster"])
log(f"  variants: {len(df):,}")

SUBPOPS = ["ESN", "GWD", "LWK", "MSL", "YRI"]
rows = []

for sub in SUBPOPS:
    col = f"AF_{sub}_pgc_a2"
    if col not in df.columns:
        log(f"  {sub}: column {col} missing; skipping")
        continue
    sub_df = df.dropna(subset=[col]).copy()
    # Approximate derived freq: same harmonization as pooled. We use
    # AF_{sub}_pgc_a2 = freq of PGC3 A2 in this subpop. To get DAF, we
    # use the same DAF_call from harmonization step (anc=ref → DAF=alt,
    # which in pgc_a2 convention may flip — but P14p3b stored
    # AF_AFR_derived using the pooled-AFR AF. We follow the same
    # ancestral logic for each subpop: it's coded in match_case.
    #   if match_case=A_direct: PGC A2=ALT → AF_{sub}_pgc_a2 = freq(ALT)
    #     if ancestral=REF → DAF = freq(ALT) = AF_{sub}_pgc_a2
    #     if ancestral=ALT → DAF = 1 - freq(ALT) = 1 - AF_{sub}_pgc_a2
    #   if match_case=B_flipped: PGC A2=REF → AF_{sub}_pgc_a2 = freq(REF) = 1 - freq(ALT)
    #     if ancestral=REF → DAF = freq(ALT) = 1 - AF_{sub}_pgc_a2
    #     if ancestral=ALT → DAF = 1 - freq(ALT) = AF_{sub}_pgc_a2
    def daf_for_subpop(row):
        af_a2 = row[col]
        call  = row["DAF_call"]
        case  = row["match_case"]
        if call == "ancestral_unknown_alt_proxy":
            # Use the same fallback as pooled: ALT-allele-frequency proxy
            return af_a2 if case == "A_direct" else 1 - af_a2
        if "anc=ref" in call:
            # DAF = alt freq
            return af_a2 if case == "A_direct" else 1 - af_a2
        elif "anc=alt" in call:
            # DAF = 1 - alt freq
            return (1 - af_a2) if case == "A_direct" else af_a2
        return np.nan

    sub_df["DAF_sub"] = sub_df.apply(daf_for_subpop, axis=1)
    sub_df = sub_df.dropna(subset=["DAF_sub"])
    log(f"\n  {sub}: n={len(sub_df):,} with derived AF")

    for (a, b) in [(0, 2), (0, 1), (1, 2)]:
        A = sub_df[sub_df["cluster"] == a]["DAF_sub"].values
        B = sub_df[sub_df["cluster"] == b]["DAF_sub"].values
        if len(A) < 5 or len(B) < 5:
            log(f"    C{a}_vs_C{b}: undersized (n={len(A)},{len(B)}); skip")
            continue
        D, P = stats.ks_2samp(A, B)
        mwu_U, mwu_P = stats.mannwhitneyu(A, B, alternative="two-sided")
        rows.append({
            "subpop": sub,
            "contrast": f"C{a}_vs_C{b}",
            "n_a": len(A), "n_b": len(B),
            "median_a": float(np.median(A)),
            "median_b": float(np.median(B)),
            "median_diff": float(np.median(A) - np.median(B)),
            "KS_D": float(D),
            "KS_P": float(P),
            "MWU_P": float(mwu_P),
        })

res = pd.DataFrame(rows)
res.to_csv(OUT / "P14p3f_subpop_replication.tsv", sep="\t", index=False)
log("\n" + res.to_string(index=False))

# Replication count: for C0_vs_C2, how many subpops pass KS D > 0.20 + P < 0.01?
c02 = res[res["contrast"] == "C0_vs_C2"]
replicated = ((c02["KS_D"] > 0.20) & (c02["KS_P"] < 0.01)).sum()
log(f"\n[Replication summary for C0_vs_C2]")
log(f"  Subpops tested: {len(c02)}")
log(f"  Subpops with KS D > 0.20 AND P < 0.01: {replicated}/{len(c02)}")
log(f"  Pre-registered threshold: ≥4/5 → REPLICATED")
if replicated >= 4:
    log("  VERDICT: REPLICATED across subpops — population-origin partition supported")
elif replicated >= 2:
    log("  VERDICT: PARTIAL — signal not driven by single subpop but not fully consistent")
else:
    log("  VERDICT: NOT REPLICATED — signal may be population-structure-driven")

with open(OUT / "P14p3f_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3f — Per-AFR-subpop replication\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p3f complete.")
