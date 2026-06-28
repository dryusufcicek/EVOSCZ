#!/usr/bin/env python3
"""
Phase 14m: GIANT-format negative-control sumstats munge for LDSC.

Adapter for Yengo 2018 GIANT height + BMI (and similar GIANT-format)
non-psychiatric sumstats: CHR POS SNP Tested_Allele Other_Allele
Freq_Tested_Allele_in_HRS BETA SE P N.

Use case: negative-control external traits to test whether the SCZ-derived
C0 cluster annotation captures specifically schizophrenia-architectural
variation or broader generic ancient-allele structure.

Schema in:  CHR POS SNP Tested_Allele Other_Allele Freq_Tested_Allele_in_HRS BETA SE P N
Schema out: SNP A1 A2 N Z

Usage:
  python3 P14m_giant_negctrl_munge.py --input <giant.txt.gz> --output-prefix <prefix>
"""
import argparse
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HM3 = str(BASE / "data/ldsc/sldsc_ref/w_hm3.snplist")


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


def munge_giant(input_path: str, output_prefix: str) -> None:
    print(f"Phase 14m GIANT munge — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_prefix}.sumstats.gz")

    df = pd.read_csv(input_path, sep=r'\s+', engine='python', compression='gzip')
    df.columns = df.columns.str.strip()
    print(f"  Total: {len(df):,}; Cols: {list(df.columns)}")

    # GIANT 2018 format
    expected = {"SNP", "Tested_Allele", "Other_Allele", "BETA", "SE", "N"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"FATAL: missing expected GIANT columns: {missing}")

    # Rename to LDSC-style A1/A2 (Tested = effect = A1 in LDSC convention)
    df = df.rename(columns={"Tested_Allele": "A1", "Other_Allele": "A2"})

    print("\n[1] HM3 restriction...")
    hm3 = pd.read_csv(HM3, sep="\t")
    df_h = df.merge(hm3, on="SNP", how="inner", suffixes=("", "_h3"))
    print(f"  HM3 merge: {len(df_h):,}")

    df_h["palin"] = [is_palindromic(a1, a2) for a1, a2 in zip(df_h["A1"], df_h["A2"])]
    n_palin = int(df_h["palin"].sum())
    df_h = df_h[~df_h["palin"]].drop(columns=["palin"]).reset_index(drop=True)
    print(f"\n[2] Dropped palindromic: {n_palin:,}; remaining {len(df_h):,}")

    print("\n[3] Allele harmonisation (BETA direct, sign-flip for reverse match)...")
    def harmonise(row):
        a1, a2 = str(row["A1"]).upper(), str(row["A2"]).upper()
        a1h, a2h = str(row["A1_h3"]).upper(), str(row["A2_h3"]).upper()
        try:
            beta = float(row["BETA"])
        except (ValueError, TypeError):
            return (np.nan, 0)
        if np.isnan(beta) or np.isinf(beta):
            return (np.nan, 0)
        if a1 == a1h and a2 == a2h:
            return (beta, 1)
        if a1 == a2h and a2 == a1h:
            return (-beta, 1)
        rc1, rc2 = revcomp(a1), revcomp(a2)
        if rc1 == a1h and rc2 == a2h:
            return (beta, 1)
        if rc1 == a2h and rc2 == a1h:
            return (-beta, 1)
        return (np.nan, 0)

    h = df_h.apply(harmonise, axis=1, result_type="expand")
    df_h["BETA_aligned"] = h[0]
    df_h["valid"] = h[1]
    n_unmatched = int((df_h["valid"] == 0).sum())
    df_h = df_h[df_h["valid"] == 1].reset_index(drop=True)
    print(f"  Dropped unmatched: {n_unmatched:,}; remaining {len(df_h):,}")

    print("\n[4] Computing Z = BETA_aligned / SE...")
    df_h["SE"] = pd.to_numeric(df_h["SE"], errors="coerce")
    df_h["Z"] = df_h["BETA_aligned"] / df_h["SE"]
    df_h = df_h[np.isfinite(df_h["Z"])].reset_index(drop=True)
    print(f"  Final: {len(df_h):,} SNPs")

    df_h["N"] = pd.to_numeric(df_h["N"], errors="coerce")
    df_h = df_h[df_h["N"].notna() & (df_h["N"] > 0)].reset_index(drop=True)

    print("\n[5] Writing LDSC sumstats...")
    out = df_h[["SNP", "A1", "A2", "N", "Z"]].copy()
    out_path = f"{output_prefix}.sumstats.gz"
    out.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  Wrote {len(out):,} rows to {out_path}")
    print(f"  Mean N = {out['N'].mean():,.0f}; Mean Z² = {(out['Z'] ** 2).mean():.4f}")
    print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIANT-format negctrl munge for LDSC")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    munge_giant(args.input, args.output_prefix)
