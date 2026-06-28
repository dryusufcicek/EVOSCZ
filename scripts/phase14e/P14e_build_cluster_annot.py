#!/usr/bin/env python3
"""Phase 14e: Build per-chromosome cluster annotation files for partitioned LDSC.

Maps P14b cluster assignments (C0/C1/C2) to the EUR LDSC baseline SNP set
and writes 22 chr-level annot.gz files matching baseline schema.

Usage:  python3 P14e_build_cluster_annot.py <ANCESTRY:eur|eas>

Output: results/phase14e/cluster_annot/cluster.{chr}.annot.gz  (EUR)
        results/phase14h/cluster_annot_eas/cluster.{chr}.annot.gz  (EAS)
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])

ancestry = sys.argv[1] if len(sys.argv) > 1 else "eur"
if ancestry == "eur":
    BASELINE_DIR = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
    OUT_DIR = BASE / "results/phase14e/cluster_annot"
elif ancestry == "eas":
    BASELINE_DIR = BASE / "data/ldsc/sldsc_ref_eas/1000G_EAS_Phase3_baseline"
    OUT_DIR = BASE / "results/phase14h/cluster_annot_eas"
else:
    raise ValueError(f"unknown ancestry {ancestry}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSIGN = BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz"

print(f"P14e cluster annot build [{ancestry.upper()}] — "
      f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Assignments: {ASSIGN}")
print(f"  Baseline:    {BASELINE_DIR}")
print(f"  Output:      {OUT_DIR}")

assign = pd.read_csv(ASSIGN, sep="\t")
print(f"  Loaded {len(assign):,} cluster assignments")
# Drop variants without cluster assignment (NaN)
assign = assign[assign["cluster"].notna()].copy()
assign["cluster"] = assign["cluster"].astype(int)
print(f"  After dropping NaN clusters: {len(assign):,}")
print(f"  Cluster sizes: {assign['cluster'].value_counts().sort_index().to_dict()}")

# Build rsid → cluster lookup (cluster ids: 0, 1, 2)
rsid2cluster = dict(zip(assign["rsid"].astype(str), assign["cluster"]))

for chrom in range(1, 23):
    base_path = BASELINE_DIR / f"baseline.{chrom}.annot.gz"
    if not base_path.exists():
        print(f"  ! chr{chrom}: baseline annot missing — skipping")
        continue
    base = pd.read_csv(base_path, sep="\t", compression="gzip",
                        usecols=["CHR", "BP", "SNP", "CM"])
    base["C0"] = 0
    base["C1"] = 0
    base["C2"] = 0
    snps = base["SNP"].astype(str).values
    for i, snp in enumerate(snps):
        c = rsid2cluster.get(snp)
        if c == 0:
            base.iat[i, base.columns.get_loc("C0")] = 1
        elif c == 1:
            base.iat[i, base.columns.get_loc("C1")] = 1
        elif c == 2:
            base.iat[i, base.columns.get_loc("C2")] = 1
    out_path = OUT_DIR / f"cluster.{chrom}.annot.gz"
    base.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n0 = int(base["C0"].sum())
    n1 = int(base["C1"].sum())
    n2 = int(base["C2"].sum())
    print(f"  chr{chrom}: {len(base):,} SNPs (C0={n0}, C1={n1}, C2={n2})")

print(f"\nDone. Cluster annot files in {OUT_DIR}")
