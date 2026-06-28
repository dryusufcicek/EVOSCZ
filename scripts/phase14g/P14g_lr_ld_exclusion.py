#!/usr/bin/env python3
"""
Phase 14g: Long-range LD region (LR-LD) exclusion sensitivity
==============================================================
González-Peñas et al. 2023 (Sci Rep) recommended sensitivity-testing partitioned
LDSC results by excluding the 24 long-range LD regions described in Price et al.
2008 *AJHG*. These regions can inflate variance explanations for selection-marker
annotations.

We re-run partitioned LDSC for SCZ and AD with our 3 cluster annotations after
masking variants in any of the 24 LR-LD regions. Compare cluster enrichments
to the unmasked Phase 14e/14f results.

Procedure:
  1. Load Price 2008 LR-LD region coordinates (Table 1 of that paper)
  2. Mask SNPs falling within any LR-LD region in our 3 cluster annotations
  3. Re-compute LD scores per chromosome with masked annotations
  4. Re-run --h2 partitioned for SCZ and AD
  5. Compare enrichment Δ between unmasked and masked
"""

import pandas as pd
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import os
import gzip

OUT_DIR = str(BASE / "results/phase14g")
os.makedirs(OUT_DIR, exist_ok=True)
ANNOT_DIR_IN = str(BASE / "results/phase14e/cluster_annot")
ANNOT_DIR_OUT = f"{OUT_DIR}/cluster_annot_no_lrld"
os.makedirs(ANNOT_DIR_OUT, exist_ok=True)

# Price 2008 long-range LD regions (build hg19/GRCh37)
# From Price AL et al. 2008 Am J Hum Genet 83:132-135, Table 1
# FIX-E-G-1: prior code had three coordinate discrepancies with the canonical
# Price 2008 Table 1 / PLINK reference list:
#   - chr5 5q11: 44500000-50500000 → 45500000-50500000 (1 Mb wider on left)
#   - chr6 HLA: 25000000-35000000 → 25500000-33500000 (over-mask of ~3 Mb)
#   - chr8 8p23: 7000000-13000000 → 8000000-12000000 (1 Mb wider on each side)
# These shrink the masked footprint to the published Price 2008 boundaries.
LR_LD_REGIONS = [
    (1,  48000000,  52000000, "1p13.3"),     # 1
    (2,  86000000, 100500000, "2p13"),        # 2
    (2, 134500000, 138000000, "2q13"),        # 3
    (2, 183000000, 190000000, "2q21"),        # 4
    (3,  47500000,  50000000, "3p21.31"),     # 5
    (3,  83500000,  87000000, "3p11"),        # 6
    (3,  89000000,  97500000, "3p11/q12"),    # 7
    (5,  45500000,  50500000, "5q11"),        # 8 — corrected from 44500000
    (5,  98000000, 100500000, "5q14"),        # 9
    (5, 129000000, 132000000, "5q23"),        # 10
    (5, 135500000, 138500000, "5q31"),        # 11
    (6,  25500000,  33500000, "6p21 (HLA)"),  # 12 — corrected from 25-35
    (6,  57000000,  64000000, "6p11"),        # 13
    (6, 140000000, 142500000, "6q23"),        # 14
    (7,  55000000,  66000000, "7p11"),        # 15
    (8,   8000000,  12000000, "8p23"),        # 16 — corrected from 7-13
    (8,  43000000,  50000000, "8p11"),        # 17
    (8, 112000000, 115000000, "8q23"),        # 18
    (10, 37000000,  43000000, "10p11"),       # 19
    (11, 46000000,  57000000, "11p11"),       # 20
    (11, 87500000,  90500000, "11q14"),       # 21
    (12, 33000000,  40000000, "12p11"),       # 22
    (12, 109500000, 112000000, "12q21"),      # 23
    (20, 32000000,  34500000, "20p12"),       # 24
]

print(f"Phase 14g: LR-LD exclusion — masking {len(LR_LD_REGIONS)} Price 2008 regions")

# For each annot file, set C0/C1/C2 = 0 for variants in LR-LD regions
def in_lr_ld(chrom, bp):
    for c, s, e, name in LR_LD_REGIONS:
        if int(chrom) == c and s <= int(bp) <= e:
            return True, name
    return False, None

import multiprocessing as mp

def mask_chrom(chrom):
    in_path = f"{ANNOT_DIR_IN}/cluster.{chrom}.annot.gz"
    out_path = f"{ANNOT_DIR_OUT}/cluster.{chrom}.annot.gz"
    df = pd.read_csv(in_path, sep="\t", compression="gzip")
    n_total = len(df)
    # Find LR-LD regions on this chrom
    relevant_regions = [(s, e) for c, s, e, _ in LR_LD_REGIONS if c == chrom]
    if relevant_regions:
        mask = pd.Series(False, index=df.index)
        for s, e in relevant_regions:
            mask |= (df["BP"] >= s) & (df["BP"] <= e)
        n_masked = mask.sum()
        # Set C0/C1/C2 = 0 for masked rows
        for col in ["C0", "C1", "C2"]:
            n_lost = ((df[col] == 1) & mask).sum()
            df.loc[mask, col] = 0
        print(f"chr{chrom}: total={n_total}, mask={n_masked} ({n_masked/n_total*100:.1f}%), "
              f"variants lost from clusters: C0={((df['C0']==1) & mask).sum()}", flush=True)
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n_c0 = (df["C0"] == 1).sum()
    n_c1 = (df["C1"] == 1).sum()
    n_c2 = (df["C2"] == 1).sum()
    return chrom, n_total, n_c0, n_c1, n_c2

if __name__ == "__main__":
    print("\nMasking annotations in parallel (8 workers)...")
    with mp.Pool(8) as p:
        results = p.map(mask_chrom, list(range(1, 23)))

    # Summary
    total_c0 = sum(r[2] for r in results)
    total_c1 = sum(r[3] for r in results)
    total_c2 = sum(r[4] for r in results)
    print(f"\nAfter masking:")
    print(f"  Total C0 variants in clusters: {total_c0} (was 1715)")
    print(f"  Total C1 variants in clusters: {total_c1} (was 1554)")
    print(f"  Total C2 variants in clusters: {total_c2} (was 2130)")

    print("\nDone. Run LDSC --l2 next on the masked annotations.")
