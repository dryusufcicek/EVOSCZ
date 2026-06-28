#!/usr/bin/env python3
"""
Phase 14n: Build binary "Young" cluster annotations for k=2 and k=4
sensitivity analyses, suitable for partitioned LDSC.

Reads P14n_assignments_k{2,4}.tsv.gz, creates per-chromosome annot.gz files
with a single Young column (1 if cluster==0, 0 otherwise) for each k.

Output:
  results/phase14b/young_annot_k2/young.{1..22}.annot.gz
  results/phase14b/young_annot_k4/young.{1..22}.annot.gz
"""
from datetime import datetime
from pathlib import Path
import os

import pandas as pd

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
ASSIGN_DIR = BASE / "results/phase14b"
BASELINE_DIR = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"

print(f"Phase 14n build_young_annot — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

for K in [2, 4]:
    out_dir = ASSIGN_DIR / f"young_annot_k{K}"
    out_dir.mkdir(exist_ok=True)
    print(f"\n--- k={K} ---")
    print(f"Output dir: {out_dir}")

    assign_path = ASSIGN_DIR / f"P14n_assignments_k{K}.tsv.gz"
    assign = pd.read_csv(assign_path, sep="\t")
    assign = assign[assign["cluster"].notna()].copy()
    assign["cluster"] = assign["cluster"].astype(int)
    young_rsids = set(assign.loc[assign["cluster"] == 0, "rsid"].astype(str).tolist())
    print(f"Loaded {len(assign):,} assignments; Young rsids: {len(young_rsids):,}")

    n_young_total = 0
    for chrom in range(1, 23):
        base_path = BASELINE_DIR / f"baseline.{chrom}.annot.gz"
        if not base_path.exists():
            print(f"  ! chr{chrom}: baseline annot missing — skipping")
            continue
        base = pd.read_csv(base_path, sep="\t", compression="gzip",
                            usecols=["CHR", "BP", "SNP", "CM"])
        base["Young"] = base["SNP"].astype(str).isin(young_rsids).astype(int)
        n_young = int(base["Young"].sum())
        n_young_total += n_young
        out_path = out_dir / f"young.{chrom}.annot.gz"
        base.to_csv(out_path, sep="\t", index=False, compression="gzip")
        if chrom in (1, 6, 22):
            print(f"  chr{chrom}: {len(base):,} SNPs, Young={n_young}")

    print(f"  Total Young SNPs (k={K}) in baseline reference: {n_young_total:,}")

print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
