#!/usr/bin/env python3
"""Phase 17D v22 (n=1000): Summarise the empirical matched-LD-MAF permutation null.

Reads the L2_1 (perm annotation) row from each of the 1000
results/phase17d_v22_n1000/perm_NNN/h2_perm_v22.results files,
builds perm_summary_v22_n1000.tsv (same columns as the n=15 table),
and computes the null distribution summary.

Extraction is IDENTICAL to the n=15 methodology:
  Z = L2_1 Coefficient_z-score (last column of the .results file).
Verified consistent: n=15 mean Z = +0.13 reproduced from these rows.

Pure python + math only (no pandas/scipy) so it runs under any python3.
"""

import math
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

BASE = (_ROOT + "/results/phase17d_v22_n1000")
N_PERM = 1000

# Observed C0 v22 primary conditional Z / enrichment (PGC3 EUR SCZ, baseline-LD v2.2).
# These are the established headline numbers, used as the fixed comparison point
# exactly as in the n=15 summarizer (P17d_v22_summarize.py).
OBSERVED_Z = 3.05
OBSERVED_ENRICH = 47.35

rows = []
missing = []
for p in range(N_PERM):
    f = os.path.join(BASE, f"perm_{p:03d}", "h2_perm_v22.results")
    if not os.path.exists(f):
        missing.append(p)
        continue
    l2_1 = None
    with open(f) as fh:
        for line in fh:
            if line.startswith("L2_1\t"):
                l2_1 = line.rstrip("\n").split("\t")
                break
    if l2_1 is None or len(l2_1) < 10:
        missing.append(p)
        continue
    # cols: 0 Category | 1 Prop_SNPs | 2 Prop_h2 | 3 Prop_h2_se | 4 Enrichment |
    #       5 Enr_se | 6 Enr_p | 7 Coef | 8 Coef_se | 9 Coef_z-score
    rows.append({
        "perm": p,
        "Prop_SNPs": float(l2_1[1]),
        "Prop_h2": float(l2_1[2]),
        "Prop_h2_se": float(l2_1[3]),
        "Enrichment": float(l2_1[4]),
        "Enrichment_se": float(l2_1[5]),
        "Enrichment_P": float(l2_1[6]),
        "Coef": float(l2_1[7]),
        "Coef_se": float(l2_1[8]),
        "Z": float(l2_1[9]),
    })

n = len(rows)
if n == 0:
    raise SystemExit("ERROR: no perm .results parsed.")

zs = [r["Z"] for r in rows]
enr = [r["Enrichment"] for r in rows]


def mean(x):
    return sum(x) / len(x)


def sd(x):  # sample SD, ddof=1 (matches n=15 pandas .std())
    m = mean(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


mean_z, sd_z = mean(zs), sd(zs)
zmin, zmax = min(zs), max(zs)
mean_e, sd_e = mean(enr), sd(enr)

n_exceed = sum(1 for z in zs if z >= OBSERVED_Z)
emp_p = (n_exceed + 1) / (n + 1)          # empirical P with +1 (cannot be 0)
pos_sd = (OBSERVED_Z - mean_z) / sd_z     # observed position in SD units
gauss_p = 0.5 * math.erfc(pos_sd / math.sqrt(2))  # one-sided parametric Gaussian P

# percentile of observed within null
n_below = sum(1 for z in zs if z < OBSERVED_Z)
pctile = 100.0 * n_below / n

# ── write per-perm tsv (same column order as the n=15 table) ────────────────
tsv = os.path.join(BASE, "perm_summary_v22_n1000.tsv")
with open(tsv, "w") as f:
    f.write("perm\tProp_SNPs\tProp_h2\tProp_h2_se\tEnrichment\tEnrichment_se\t"
            "Enrichment_P\tCoef\tCoef_se\tZ\n")
    for r in sorted(rows, key=lambda x: x["perm"]):
        f.write(f"{r['perm']}\t{r['Prop_SNPs']:.10g}\t{r['Prop_h2']:.10g}\t"
                f"{r['Prop_h2_se']:.10g}\t{r['Enrichment']:.10g}\t{r['Enrichment_se']:.10g}\t"
                f"{r['Enrichment_P']:.10g}\t{r['Coef']:.10g}\t{r['Coef_se']:.10g}\t{r['Z']:.10g}\n")

# ── write summary table ──────────────────────────────────────────────────────
md = os.path.join(BASE, "null_summary_v22_n1000.tsv")
with open(md, "w") as f:
    f.write("n_parsed\tn_perm_expected\tn_missing\tnull_Z_mean\tnull_Z_sd\tnull_Z_min\tnull_Z_max\t"
            "null_enrichment_mean\tnull_enrichment_sd\tobserved_Z\tobserved_enrichment\tn_controls_Z_ge_observed\t"
            "observed_percentile\tobserved_minus_null_mean_in_SD\tempirical_P\tgaussian_one_sided_P\n")
    f.write(f"{n}\t{N_PERM}\t{len(missing)}\t{mean_z:.6f}\t{sd_z:.6f}\t{zmin:.6f}\t{zmax:.6f}\t"
            f"{mean_e:.6f}\t{sd_e:.6f}\t{OBSERVED_Z:.2f}\t{OBSERVED_ENRICH:.2f}\t{n_exceed}\t"
            f"{pctile:.6f}\t{pos_sd:.6f}\t{emp_p:.6e}\t{gauss_p:.6e}\n")

# ── console ─────────────────────────────────────────────────────────────────
print(f"Phase 17D v22 NULL SUMMARY n=1000 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print(f"N parsed         : {n}/{N_PERM}  (missing {len(missing)}: {missing})")
print(f"Null Z mean ± SD : {mean_z:+.4f} ± {sd_z:.4f}")
print(f"Null Z range     : [{zmin:+.4f}, {zmax:+.4f}]")
print(f"Null enrichment  : {mean_e:+.3f}x +/- {sd_e:.3f}")
print(f"Observed C0 Z    : +{OBSERVED_Z:.2f}")
print(f"Controls Z>=obs  : {n_exceed}/{n}")
print(f"Percentile       : {pctile:.3f}%")
print(f"Position (SD)    : {pos_sd:.4f} SD above null mean")
print(f"Empirical P      : {emp_p:.4e}  (floor 1/{n+1})")
print(f"Gaussian P (1s)  : {gauss_p:.4e}")
print(f"\nTSV : {tsv}")
print(f"SUMMARY : {md}")
