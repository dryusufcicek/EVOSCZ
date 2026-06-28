#!/usr/bin/env python3
"""Phase 17D v22 — Build per-chromosome regression-SNP lists from baseline-LD v2.2.

The S-LDSC --print-snps flag requires a pre-computed list of SNPs that will be
used as the regression substrate. Under baseline-LD v2.2 (Gazal 2017; 97
annotations) the regression-SNP set differs from the v1.2 set, so we regenerate
the snplist from the v2.2 annot files.

Writes to /tmp/baselineLD_v22_chrN_snplist.txt for N in 1..22.
Consumed by P17d_v22_run_perms.sh (--print-snps argument to ldsc.py --l2).
"""

import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import sys

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
BASELINE_V22 = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"
SNPLIST_DIR = Path("/tmp")

print(f"P17d v22 SNP-list build — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

if not BASELINE_V22.exists():
    print(f"ERROR: baseline-LD v2.2 directory not found: {BASELINE_V22}")
    print("Download from Zenodo 10515792 first (see README).")
    sys.exit(1)

total_snps = 0
for chrom in range(1, 23):
    annot_path = BASELINE_V22 / f"baselineLD.{chrom}.annot.gz"
    if not annot_path.exists():
        print(f"ERROR: missing {annot_path}")
        sys.exit(1)
    annot = pd.read_csv(annot_path, sep="\t", compression="gzip", usecols=["SNP"])
    out_path = SNPLIST_DIR / f"baselineLD_v22_chr{chrom}_snplist.txt"
    annot["SNP"].to_csv(out_path, index=False, header=False)
    total_snps += len(annot)
    print(f"  chr{chrom:>2}: {len(annot):>7,} SNPs -> {out_path}")

print(f"\nTotal v22 regression SNPs across 22 chr: {total_snps:,}")
print(f"Snp-lists ready at /tmp/baselineLD_v22_chr{{1..22}}_snplist.txt")
