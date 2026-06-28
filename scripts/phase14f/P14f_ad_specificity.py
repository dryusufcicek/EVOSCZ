#!/usr/bin/env python3
"""
Phase 14f: Alzheimer's Disease GWAS specificity test
======================================================
Tests whether the 3 evolutionary clusters identified for PGC3 SCZ also carry
disproportionate Alzheimer's disease (AD) heritability, using Wightman 2021
GWAS summary statistics (n=1,126,563; UKB excluded sumstats).

Procedure:
  1. Munge Wightman AD sumstats to LDSC format (HM3-restricted, allele-matched)
  2. Run partitioned h² on AD with same 3 cluster annotations + baseline-LD v2.2
  3. Compare cluster enrichment between SCZ (Phase 14e) and AD

Interpretation:
  - If a cluster is SCZ-only enriched → SCZ-specific evolutionary signature
  - If a cluster is enriched for both → brain-disorder-generic property
"""

import pandas as pd
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import numpy as np
import os
from datetime import datetime

print(f"Phase 14f v2: AD GWAS specificity test — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)
print("Code-review Faz E: pre-filter strand-ambiguous palindromic A/T,C/G")
print("variants; add reverse-complement allele matching for non-palindromic")
print("strand flips (prior code dropped these silently).")
print()


def revcomp(allele):
    if not isinstance(allele, str) or len(allele) != 1:
        return allele
    return {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}.get(allele.upper(), allele)


def is_palindromic(a1, a2):
    if not (isinstance(a1, str) and isinstance(a2, str)
            and len(a1) == 1 and len(a2) == 1):
        return False
    pair = {a1.upper(), a2.upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}

# Step 1: Munge Wightman 2021 AD sumstats
AD_RAW = os.path.join(os.environ.get("AD_SUMSTATS_DIR", "data/ad_sumstats"), "PGCALZ2sumstatsExcluding23andMe.txt.gz")
HM3 = str(BASE / "data/ldsc/sldsc_ref/w_hm3.snplist")
OUT_DIR = str(BASE / "results/phase14f")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = f"{OUT_DIR}/Wightman_AD.sumstats.gz"

print("\n[1] Loading Wightman 2021 AD sumstats...")
df = pd.read_csv(AD_RAW, sep="\t")
df.columns = df.columns.str.strip()  # Wightman header has leading space on 'chr'
print(f"  AD: {len(df):,} variants. Columns: {list(df.columns)}")
print(f"  Sample N range: {df['N'].min():,}-{df['N'].max():,}")

# Sumstats has chr+pos but no rsID — we need to match to HM3 via chr:pos
# HM3 list has SNP rsID. We need to crosswalk via 1KG plink files.
print("\n[2] Building chr:pos → rsID lookup from baseline annotations...")
crosswalk = []
for chrom in range(1, 23):
    f = f"{BASE}/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline/baseline.{chrom}.annot.gz"
    if not os.path.exists(f): continue
    sub = pd.read_csv(f, sep="\t", compression="gzip", usecols=["CHR","BP","SNP"])
    crosswalk.append(sub)
crosswalk = pd.concat(crosswalk, ignore_index=True)
crosswalk.columns = ["chr", "pos", "rsid"]
print(f"  Crosswalk: {len(crosswalk):,} SNPs")

# Step 3: Restrict AD to HM3 SNPs and add rsID
print("\n[3] Restricting AD to HM3 + adding rsIDs...")
hm3 = pd.read_csv(HM3, sep="\t")
print(f"  HM3 list: {len(hm3):,} SNPs")

# Merge AD with crosswalk on chr+pos
df = df.rename(columns={"chr": "chr", "PosGRCh37": "pos"})
df["chr"] = df["chr"].astype(int)
df["pos"] = df["pos"].astype(int)
df_keyed = df.merge(crosswalk, on=["chr","pos"], how="inner")
print(f"  After chr:pos crosswalk: {len(df_keyed):,}")

# Restrict to HM3 SNPs and harmonize alleles
df_hm3 = df_keyed.merge(hm3, left_on="rsid", right_on="SNP", how="inner",
                          suffixes=("","_hm3"))
print(f"  After HM3 merge: {len(df_hm3):,}")

# Pre-filter strand-ambiguous palindromic variants (FIX-E-F-1)
df_hm3["palin"] = [is_palindromic(a, b)
                   for a, b in zip(df_hm3["testedAllele"], df_hm3["otherAllele"])]
n_palin = int(df_hm3["palin"].sum())
df_hm3 = df_hm3[~df_hm3["palin"]].drop(columns=["palin"]).reset_index(drop=True)
print(f"  Dropped strand-ambiguous palindromic: {n_palin:,}")


def harmonize(row):
    a1 = str(row["testedAllele"]).upper()
    a2 = str(row["otherAllele"]).upper()
    a1h = str(row["A1"]).upper()
    a2h = str(row["A2"]).upper()
    if a1 == a1h and a2 == a2h:
        return row["z"], 1
    if a1 == a2h and a2 == a1h:
        return -row["z"], 1
    # Reverse complement (non-palindromic only — already filtered)
    rc1, rc2 = revcomp(a1), revcomp(a2)
    if rc1 == a1h and rc2 == a2h:
        return row["z"], 1
    if rc1 == a2h and rc2 == a1h:
        return -row["z"], 1
    return None, 0

df_hm3["z_h"], df_hm3["match"] = zip(*df_hm3.apply(harmonize, axis=1))
df_hm3 = df_hm3[df_hm3["match"] == 1].dropna(subset=["z_h"]).reset_index(drop=True)
print(f"  After allele matching (incl rev-comp): {len(df_hm3):,}")

# Step 4: Save in LDSC format: SNP A1 A2 N Z
out = df_hm3[["rsid", "A1", "A2", "N", "z_h"]].copy()
out.columns = ["SNP", "A1", "A2", "N", "Z"]
out["N"] = out["N"].astype(int)
out.to_csv(OUT, sep="\t", index=False, compression="gzip")
print(f"\n  Saved: {OUT} ({len(out):,} variants)")
print(out.head())
