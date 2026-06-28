#!/usr/bin/env python3
"""
Phase 14h v2: EAS PGC3 SCZ sumstats munge to LDSC format (code-review-corrected)

Code-review Faz E corrections:
  [FIX-E-H-1] Strand-ambiguous (A/T, C/G) variants pre-filtered. Prior code
    only checked exact A1/A2 ↔ A1_h3/A2_h3 match in either order, dropping
    strand-flipped variants silently and accepting palindromic matches that
    cannot be resolved without frequency.
  [FIX-E-H-2] Reverse-complement allele matching for non-palindromic strand
    flips. After dropping palindromic, attempt rc(A1)/rc(A2) match before
    discarding as unmatched.
  [FIX-E-H-3] Effective sample size: N = 4 * NCAS * NCON / (NCAS + NCON)
    instead of NCAS + NCON. Required for LDSC h² scaling on case/control
    GWAS; the prior raw-total N misled LDSC's continuous-trait N handling.

Schema in: chr, ID, pos, A1, A2, FCAS, FCON, IMPINFO, BETA, SE, PVAL, NCAS, NCON, NEFF
Schema out (LDSC): SNP, A1, A2, N, Z
"""
import os
from pathlib import Path
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
from datetime import datetime

import numpy as np
import pandas as pd

print(f"Phase 14h v2 munge — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

EAS_RAW = os.path.join(os.environ.get("PGC_SUMSTATS_DIR", "data/pgc_sumstats"), "PGC3_SCZ_wave3.asian.autosome.public.v3.vcf.tsv.gz")
HM3 = str(BASE / "data/ldsc/sldsc_ref/w_hm3.snplist")
OUT_DIR = str(BASE / "results/phase14h")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = f"{OUT_DIR}/PGC3_SCZ_EAS.sumstats.gz"


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


print("[1] Loading EAS sumstats (skipping VCF-style ## headers)...")
df = pd.read_csv(EAS_RAW, sep="\t", comment="#")
df.columns = df.columns.str.strip()
print(f"  Total: {len(df):,} variants. Cols: {list(df.columns)}")

print("\n[2] Restricting to HM3 + matching alleles...")
hm3 = pd.read_csv(HM3, sep="\t")
print(f"  HM3: {len(hm3):,}")

df_h = df.merge(hm3, left_on="ID", right_on="SNP", how="inner", suffixes=("", "_h3"))
print(f"  After HM3 merge: {len(df_h):,}")

# FIX-E-H-1: pre-filter strand-ambiguous palindromic variants
df_h["palin"] = [is_palindromic(a1, a2) for a1, a2 in zip(df_h["A1"], df_h["A2"])]
n_palin = int(df_h["palin"].sum())
df_h = df_h[~df_h["palin"]].drop(columns=["palin"]).reset_index(drop=True)
print(f"  Dropped strand-ambiguous (A/T, C/G) palindromic: {n_palin:,}")
print(f"  After palindromic drop: {len(df_h):,}")


# Allele harmonization with rev-comp (FIX-E-H-2)
def harmonize(row):
    a1, a2 = str(row["A1"]).upper(), str(row["A2"]).upper()
    a1h, a2h = str(row["A1_h3"]).upper(), str(row["A2_h3"]).upper()
    if a1 == a1h and a2 == a2h:
        return row["BETA"], 1
    if a1 == a2h and a2 == a1h:
        return -row["BETA"], 1
    # Reverse complement (non-palindromic only — palindromic already dropped)
    rc1, rc2 = revcomp(a1), revcomp(a2)
    if rc1 == a1h and rc2 == a2h:
        return row["BETA"], 1
    if rc1 == a2h and rc2 == a1h:
        return -row["BETA"], 1
    return None, 0


df_h["beta_h"], df_h["match"] = zip(*df_h.apply(harmonize, axis=1))
df_h = df_h[df_h["match"] == 1].dropna(subset=["beta_h", "SE"]).reset_index(drop=True)
print(f"  After allele match (incl. rev-comp): {len(df_h):,}")

# Z and effective N (FIX-E-H-3)
df_h["Z"] = df_h["beta_h"] / df_h["SE"]
ncas = df_h["NCAS"].astype(float)
ncon = df_h["NCON"].astype(float)
df_h["N_eff"] = (4.0 * ncas * ncon / (ncas + ncon)).astype(int)
print(f"  Effective N: mean={df_h['N_eff'].mean():.0f}, "
      f"range {df_h['N_eff'].min()}-{df_h['N_eff'].max()}")
print(f"  Raw total N (NCAS+NCON, for reference): "
      f"mean={(ncas + ncon).mean():.0f}")

out = df_h[["ID", "A1_h3", "A2_h3", "N_eff", "Z"]].copy()
out.columns = ["SNP", "A1", "A2", "N", "Z"]
out.to_csv(OUT, sep="\t", index=False, compression="gzip")
print(f"\nSaved: {OUT} ({len(out):,} variants)")
print(f"  Mean Neff: {out['N'].mean():.0f}; range {out['N'].min()}-{out['N'].max()}")
