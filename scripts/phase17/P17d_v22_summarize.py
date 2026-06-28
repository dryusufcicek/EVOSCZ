#!/usr/bin/env python3
"""Phase 17D v22: Summarise the empirical permutation null distribution.

Reads results/phase17d_v22/perm_summary_v22.tsv (produced by run_perms.sh)
and computes null mean ± SD, null max, and observed-Z position relative
to the null. Writes a Markdown summary suitable for direct use in
SupplementaryMethods_NatGen_v22.md §E3 and Supp Table 20 update.
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

md = OUT / "null_summary_v22.md"
with open(md, "w") as f:
    f.write(f"# Supplementary Table 20 v22 — Empirical matched-LD-MAF permutation null\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Baseline-LD model: v2.2 (Gazal 2017; 97 annotations)\n")
    f.write(f"N permutations: {n}\n")
    f.write(f"Observed C0 conditional Z (primary v22 PGC3 EUR SCZ): **+{OBSERVED_Z_V22:.2f}**\n\n")
    f.write("## Null distribution\n\n")
    f.write(f"- Null mean Z ± SD: **{mean_z:+.2f} ± {sd_z:.2f}**\n")
    f.write(f"- Null Z range: [{min_z:+.2f}, {max_z:+.2f}]\n")
    f.write(f"- Null enrichment mean ± SD: **{mean_enr:+.2f}× ± {sd_enr:.2f}**\n")
    f.write(f"- Controls reaching observed Z ≥ +{OBSERVED_Z_V22:.2f}: **{n_exceed}/{n}**\n")
    f.write(f"- Position: observed Z is {(OBSERVED_Z_V22 - mean_z)/sd_z:.2f} SD above null mean\n\n")
    f.write("## Per-permutation table\n\n")
    f.write("| perm | Prop_SNPs | Prop_h² | Enrichment | s.e. | P | Z |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        f.write(f"| {int(r['perm'])} | {r['Prop_SNPs']:.6f} | {r['Prop_h2']:.5f} | "
                f"{r['Enrichment']:+.2f}× | {r['Enrichment_se']:.2f} | "
                f"{r['Enrichment_P']:.3f} | {r['Z']:+.2f} |\n")
    f.write("\n## Interpretation\n\n")
    if n_exceed == 0:
        f.write(f"The observed C0 conditional *Z* = +{OBSERVED_Z_V22:.2f} lies outside the "
                f"matched-LD-MAF empirical null distribution under baseline-LD v2.2 "
                f"({n_exceed}/{n} controls reached the observed Z; null mean {mean_z:+.2f} ± {sd_z:.2f}). "
                f"This confirms the v1.2-based proportional-scaling projection reported in the "
                f"prior pending state of this table.\n")
    else:
        f.write(f"WARNING: {n_exceed}/{n} control annotations reached or exceeded the observed Z. "
                f"Empirical *P* = {(n_exceed + 1)/(n + 1):.3f}. Review individual permutations.\n")

print(f"\nMarkdown summary: {md}")
print("Outputs written; use these to populate the permutation-null supplementary table.")
