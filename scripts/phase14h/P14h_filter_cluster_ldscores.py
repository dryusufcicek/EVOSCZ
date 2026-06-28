#!/usr/bin/env python3
"""Phase 14h: Filter cluster .l2.ldscore.gz to the baseline regression-SNP set per chr.

Cluster LD scores were emitted across the full EAS plink panel; baseline ldscore files
contain only the standard regression-SNP subset. LDSC's concatenation requires identical
SNP columns across all --ref-ld-chr files. This script restricts cluster.{chr}.l2.ldscore.gz
to baseline.{chr}.l2.ldscore.gz SNPs (preserving baseline order). M and M_5_50 totals are
panel-wide (computed by ldsc.py --l2 from the bfile) and remain untouched.
"""
import pandas as pd
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import os
from datetime import datetime

CLUSTER_DIR = str(BASE / "results/phase14h/cluster_annot_eas")
BASELINE_DIR = str(BASE / "data/ldsc/sldsc_ref_eas/1000G_EAS_Phase3_baseline")

print(f"Phase 14h ldscore filter — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

for chr in range(1, 23):
    base_path = f"{BASELINE_DIR}/baseline.{chr}.l2.ldscore.gz"
    clu_path = f"{CLUSTER_DIR}/cluster.{chr}.l2.ldscore.gz"
    base_snps = pd.read_csv(base_path, sep="\t", usecols=["SNP"])
    clu = pd.read_csv(clu_path, sep="\t")
    n_clu_orig = len(clu)
    merged = base_snps.merge(clu, on="SNP", how="left")
    n_missing = merged["CHR"].isna().sum()
    if n_missing > 0:
        print(f"chr{chr}: {n_missing} baseline SNPs missing from cluster ldscore — ABORT")
        raise SystemExit(1)
    merged["CHR"] = merged["CHR"].astype(int)
    merged["BP"] = merged["BP"].astype(int)
    merged.to_csv(clu_path, sep="\t", index=False, compression="gzip")
    print(f"chr{chr}: {n_clu_orig} -> {len(merged)} SNPs filtered")

print("Done filtering all 22 chromosomes.")
