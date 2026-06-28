#!/usr/bin/env python3
"""Build EUR cluster annotations for 3D GMM at k=2 and k=4 (sensitivity)."""
from datetime import datetime
from pathlib import Path
import os
import pandas as pd

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
BASELINE_DIR = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
ASSIGN_DIR = BASE / "results/phase14b"

print(f"Build k=2/k=4 annotations — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

for K in [2, 4]:
    out_dir = ASSIGN_DIR / f"cluster_annot_3d_k{K}"
    out_dir.mkdir(exist_ok=True)
    print(f"\n--- k={K} ---")

    assign = pd.read_csv(ASSIGN_DIR / f"P14n_assignments_k{K}.tsv.gz", sep="\t")
    assign = assign[assign["cluster"].notna()].copy()
    assign["cluster"] = assign["cluster"].astype(int)
    rsid2cluster = dict(zip(assign["rsid"].astype(str), assign["cluster"]))

    # Output binary "Young" (1 if cluster=0, else 0) in column C0
    n_total = 0
    for chrom in range(1, 23):
        base_path = BASELINE_DIR / f"baseline.{chrom}.annot.gz"
        base = pd.read_csv(base_path, sep="\t", compression="gzip",
                            usecols=["CHR", "BP", "SNP", "CM"])
        base["C0"] = base["SNP"].astype(str).map(
            lambda r: 1 if rsid2cluster.get(r) == 0 else 0).astype(int)
        out_path = out_dir / f"cluster.{chrom}.annot.gz"
        base.to_csv(out_path, sep="\t", index=False, compression="gzip")
        n_total += int(base["C0"].sum())
    print(f"  Total Young (C0=1) at k={K}: {n_total}")
print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
