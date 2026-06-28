#!/usr/bin/env python3
"""
T1_verify_extraction.py — CHECKPOINT after J1.

Sanity: the genome-wide age extraction (age_tables/age_chr*.tsv, built by the
Combined>TGP>SGDP + AgeMedian_Mut recipe) must reproduce variant_master's
age_median_yr for the credible-set variants (the EXACT age axis the clustering
used). Locally this recipe matched 20,565/20,565 with max diff 0.0000; the
genome-wide code path must do the same on its credible-set subset. PASS only on
near-perfect exact match — else there is a genome-wide-parsing bug to fix.
"""
import glob
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
import numpy as np
import pandas as pd

W = Path((_SCRATCH + "/test1_age_conditioning"))
VM = (_ROOT + "/results/phase11/variant_master_v4.parquet")

parts = []
for f in sorted(glob.glob(str(W / "age_tables/age_chr*.tsv"))):
    parts.append(pd.read_csv(f, sep="\t", usecols=["rsid", "age_mut_gen"]))
gw = pd.concat(parts, ignore_index=True).drop_duplicates("rsid")
gw["age_mut_gen"] = pd.to_numeric(gw["age_mut_gen"], errors="coerce")
print(f"genome-wide extraction: {len(gw):,} unique rsids ({len(parts)} chr files)")

vm = pd.read_parquet(VM)[["rsid", "age_median_yr"]].dropna()
print(f"variant_master credible-set with age: {len(vm):,}")

cmp = vm.merge(gw, on="rsid", how="inner").dropna(subset=["age_median_yr", "age_mut_gen"])
cmp["diff"] = (cmp["age_median_yr"] - cmp["age_mut_gen"]).abs()
exact = int((cmp["diff"] < 0.01).sum())
n = len(cmp)
print(f"compared (shared credible-set rsids): {n:,}")
print(f"EXACT match (|diff|<0.01): {exact:,} ({exact/n*100:.3f}%)  | max abs diff: {cmp['diff'].max():.6f}")

if n >= 20000 and exact == n:
    print("VERIFY_PASS")
else:
    print(f"VERIFY_FAIL — {n-exact} mismatches (or too few compared: n={n})")
    mism = cmp[cmp["diff"] >= 0.01]
    if len(mism):
        print(mism.head(10).to_string(index=False))
