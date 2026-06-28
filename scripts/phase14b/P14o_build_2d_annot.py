#!/usr/bin/env python3
"""Phase 14o: Build 2D cluster annotations (C0/C1/C2 binary) for LDSC."""
from datetime import datetime
from pathlib import Path
import os
import pandas as pd

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
ASSIGN = BASE / "results/phase14b/P14o_2d_assignments.tsv.gz"
BASELINE_DIR = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
OUT_DIR = BASE / "results/phase14b/cluster_annot_2d"
OUT_DIR.mkdir(exist_ok=True)

print(f"Phase 14o build_2d_annot — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
assign = pd.read_csv(ASSIGN, sep="\t")
assign = assign[assign["cluster"].notna()].copy()
assign["cluster"] = assign["cluster"].astype(int)
print(f"Loaded {len(assign):,} 2D cluster assignments")
print(f"  Cluster sizes: {assign['cluster'].value_counts().sort_index().to_dict()}")

rsid2cluster = dict(zip(assign["rsid"].astype(str), assign["cluster"]))

n_per_chr = []
for chrom in range(1, 23):
    base_path = BASELINE_DIR / f"baseline.{chrom}.annot.gz"
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
    n_per_chr.append((chrom, n0, n1, n2))

# Summary
df_n = pd.DataFrame(n_per_chr, columns=["chr", "C0", "C1", "C2"])
print(f"\nTotal 2D cluster annotation SNPs in baseline reference:")
print(f"  C0 (Young): {df_n['C0'].sum():,}")
print(f"  C1 (Mid):   {df_n['C1'].sum():,}")
print(f"  C2 (Old):   {df_n['C2'].sum():,}")
print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
