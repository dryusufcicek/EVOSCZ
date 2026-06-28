#!/usr/bin/env python3
"""
Phase 14o: Mask Price 2008 long-range LD regions from 2D cluster annotations
for sensitivity analysis.

Mirrors Phase 14g for the 3D primary, applied to 2D 'cluster_annot_2d'.
"""
from datetime import datetime
from pathlib import Path
import os
import pandas as pd

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
ANNOT_IN = BASE / "results/phase14b/cluster_annot_2d"
ANNOT_OUT = BASE / "results/phase14b/cluster_annot_2d_no_lrld"
ANNOT_OUT.mkdir(exist_ok=True)

# Price 2008 LR-LD regions (canonical-corrected, per Phase 14g)
LR_LD_REGIONS = [
    (1,  48000000,  52000000),
    (2,  86000000, 100500000),
    (2, 134500000, 138000000),
    (2, 183000000, 190000000),
    (3,  47500000,  50000000),
    (3,  83500000,  87000000),
    (3,  89000000,  97500000),
    (5,  45500000,  50500000),
    (5,  98000000, 100500000),
    (5, 129000000, 132000000),
    (6,  25500000,  33500000),  # HLA
    (6,  57000000,  64000000),
    (6, 140000000, 142500000),
    (7,  55000000,  66000000),
    (8,   8000000,  12000000),
    (8,  43000000,  50000000),
    (8, 112000000, 115000000),
    (10, 37000000,  43000000),
    (11, 46000000,  57000000),
    (11, 87500000,  90500000),
    (12, 33000000,  40000000),
    (12,109500000, 112000000),
    (20, 32000000,  34500000),
    (22, 18000000,  22000000),
]

print(f"Phase 14o LR-LD mask 2D — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
masked_total = {"C0": 0, "C1": 0, "C2": 0}
for chrom in range(1, 23):
    in_path = ANNOT_IN / f"cluster.{chrom}.annot.gz"
    if not in_path.exists():
        continue
    df = pd.read_csv(in_path, sep="\t", compression="gzip")
    chrom_regions = [(s, e) for c, s, e in LR_LD_REGIONS if c == chrom]
    in_lrld = pd.Series(False, index=df.index)
    for s, e in chrom_regions:
        in_lrld |= (df["BP"] >= s) & (df["BP"] <= e)
    n_masked = {col: int(((df[col] == 1) & in_lrld).sum()) for col in ["C0", "C1", "C2"]}
    df.loc[in_lrld, ["C0", "C1", "C2"]] = 0
    out_path = ANNOT_OUT / f"cluster.{chrom}.annot.gz"
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    for col in ["C0", "C1", "C2"]:
        masked_total[col] += n_masked[col]

print(f"\nLR-LD-masked variants:")
for col, n in masked_total.items():
    print(f"  {col}: {n} masked")
print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
