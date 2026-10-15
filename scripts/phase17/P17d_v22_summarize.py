#!/usr/bin/env python3
"""Phase 17D v22: Summarise the empirical permutation null distribution.

Reads results/phase17d_v22/perm_summary_v22.tsv (produced by run_perms.sh)
and computes null mean ± SD, null max, and observed-Z position relative
to the null. Writes a one-row summary table (null_summary_v22.tsv).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import sys

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase17d_v22"

SUMMARY = OUT / "perm_summary_v22.tsv"
if not SUMMARY.exists():
    print(f"ERROR: {SUMMARY} not found. Run P17d_v22_run_perms.sh first.")
    sys.exit(1)

# Observed C0 v22 conditional Z (from primary v22 PGC3 EUR SCZ partitioned LDSC)
OBSERVED_Z_V22 = 3.05
OBSERVED_ENRICH_V22 = 47.35

df = pd.read_csv(SUMMARY, sep="\t")
n = len(df)
mean_z   = df["Z"].mean()
sd_z     = df["Z"].std(ddof=1)
max_z    = df["Z"].max()
min_z    = df["Z"].min()
mean_enr = df["Enrichment"].mean()
sd_enr   = df["Enrichment"].std(ddof=1)
n_exceed = int((df["Z"] >= OBSERVED_Z_V22).sum())

print(f"Phase 17D v22 NULL SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print(f"N permutations    : {n}")
print(f"Null Z mean ± SD  : {mean_z:+.2f} ± {sd_z:.2f}")
print(f"Null Z range      : [{min_z:+.2f}, {max_z:+.2f}]")
print(f"Null Enrichment   : {mean_enr:+.2f}× ± {sd_enr:.2f}")
print(f"Observed C0 Z v22 : +{OBSERVED_Z_V22:.2f}")
print(f"Controls Z ≥ obs  : {n_exceed}/{n}")
print(f"Position          : observed Z is {(OBSERVED_Z_V22 - mean_z)/sd_z:.2f} SD above null mean")

summary = OUT / "null_summary_v22.tsv"
with open(summary, "w") as f:
    f.write("n_perm\tnull_Z_mean\tnull_Z_sd\tnull_Z_min\tnull_Z_max\tnull_enrichment_mean\tnull_enrichment_sd\t"
            "observed_Z\tn_controls_Z_ge_observed\tobserved_minus_null_mean_in_SD\tempirical_P\n")
    f.write(f"{n}\t{mean_z:.6f}\t{sd_z:.6f}\t{min_z:.6f}\t{max_z:.6f}\t{mean_enr:.6f}\t{sd_enr:.6f}\t"
            f"{OBSERVED_Z_V22:.2f}\t{n_exceed}\t{(OBSERVED_Z_V22 - mean_z) / sd_z:.6f}\t{(n_exceed + 1) / (n + 1):.6f}\n")

print(f"\nSummary table: {summary}")
