#!/usr/bin/env python3
"""Phase 17e: Build C0-only (single-annotation) cluster file for fair comparison with brain_spec.

For each chr, take baseline annot CHR/BP/SNP/CM + add a single C0_only column
(=1 if SNP is in primary Young cluster, 0 otherwise). Drop C1 and C2.

Output: results/phase17e/c0_only_annot/c0_only.{1..22}.annot.gz
"""
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
SRC = BASE / "results/phase14e/cluster_annot"
OUT = BASE / "results/phase17e/c0_only_annot"
OUT.mkdir(parents=True, exist_ok=True)

print(f"Phase 17e: Building C0-only single-annotation files — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

for chrom in range(1, 23):
    src_path = SRC / f"cluster.{chrom}.annot.gz"
    if not src_path.exists():
        print(f"  ! chr{chrom}: source missing — skip"); continue
    annot = pd.read_csv(src_path, sep="\t", compression="gzip",
                        usecols=["CHR", "BP", "SNP", "CM", "C0"])
    annot = annot.rename(columns={"C0": "C0_only"})
    out_path = OUT / f"c0_only.{chrom}.annot.gz"
    annot.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n_c0 = int(annot["C0_only"].sum())
    print(f"  chr{chrom}: {len(annot):,} SNPs, C0_only={n_c0}")

print(f"\nDone. Files in {OUT}")
