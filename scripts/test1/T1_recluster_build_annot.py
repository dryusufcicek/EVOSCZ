#!/usr/bin/env python3
"""Build one-hot cluster annotation from a recluster assign file (already sorted
youngest->C0), aligned row-for-row to baseline-LD v2.2.
Usage: T1_recluster_build_annot.py <assign_file.tsv.gz> <out_subdir> <k>"""
import sys
import pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

BLD = Path((_ROOT + "/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"))
assign_file, outsub, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = pd.read_csv(assign_file, sep="\t").drop_duplicates("rsid")
amap = dict(zip(a["rsid"], a["cluster"].astype(int)))
OUT = Path(f"{_SCRATCH}/test1_age_conditioning/recluster_annot/{outsub}")
OUT.mkdir(parents=True, exist_ok=True)
tot = 0
for c in range(1, 23):
    base = pd.read_csv(BLD / f"baselineLD.{c}.annot.gz", sep="\t", usecols=["CHR", "BP", "SNP", "CM"])
    cl = base["SNP"].map(amap)
    for i in range(k):
        base[f"C{i}"] = (cl == i).astype(int)
    base.to_csv(OUT / f"cluster.{c}.annot.gz", sep="\t", index=False, compression="gzip")
    tot += int(cl.notna().sum())
print(f"{outsub}: annot built (k={k}), {tot} clustered SNPs across 22 chr", flush=True)
