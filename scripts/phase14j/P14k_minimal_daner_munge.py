#!/usr/bin/env python3
"""
Phase 14k: Minimal-daner sumstats munge to LDSC format.

For sumstats with daner-like CHR/SNP/BP/A1/A2/INFO/OR/SE/P columns
but WITHOUT per-row Nca/Nco — sample size must be supplied via CLI.

Use case: ASD Grove 2019 (iPSYCH-PGC_ASD_Nov2017.gz) — 18,381 cases / 27,969 controls fixed.
LDSC N convention: N = 4 * Nca * Nco / (Nca + Nco) = effective sample size.

Schema in (minimal daner): CHR, SNP, BP, A1, A2, INFO, OR, SE, P
Schema out (LDSC): SNP, A1, A2, N, Z

Usage:
  python3 P14k_minimal_daner_munge.py \\
    --input <minimal-daner.gz> \\
    --output-prefix <prefix> \\
    --ncas <N_cases> --ncon <N_controls>
"""
import argparse
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import gzip
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


def munge_minimal(input_path: str, output_prefix: str, ncas: int, ncon: int) -> None:
    n_eff = 4 * ncas * ncon / (ncas + ncon)
    print(f"Phase 14k minimal-daner munge — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_prefix}.sumstats.gz")
    print(f"  Cohort: {ncas:,} cases / {ncon:,} controls")
    print(f"  Fixed N_eff = 4*Nca*Nco/(Nca+Nco) = {n_eff:,.1f}")

    # Auto-detect separator
    print("\n[1] Loading minimal-daner sumstats...")
    with gzip.open(input_path, 'rt') as _f:
        _first = _f.readline().rstrip('\n')
    if '\t' in _first:
        df = pd.read_csv(input_path, sep='\t', compression='gzip')
        print(f"  Separator: tab (C engine)")
    else:
        df = pd.read_csv(input_path, sep=r'\s+', engine='python', compression='gzip')
        print(f"  Separator: whitespace regex (Python engine)")
    df.columns = df.columns.str.strip()
    print(f"  Total: {len(df):,} variants. Cols: {list(df.columns)}")

    expected = {"CHR", "SNP", "BP", "A1", "A2", "OR", "SE", "P"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"FATAL: missing expected columns: {missing}")

    print("\n[2] Restricting to HapMap3...")
    hm3 = pd.read_csv(HM3, sep="\t")
    df_h = df.merge(hm3, on="SNP", how="inner", suffixes=("", "_h3"))
    print(f"  HM3 merge: {len(df_h):,}")

    df_h["palin"] = [is_palindromic(a1, a2) for a1, a2 in zip(df_h["A1"], df_h["A2"])]
    n_palin = int(df_h["palin"].sum())
    df_h = df_h[~df_h["palin"]].drop(columns=["palin"]).reset_index(drop=True)
    print(f"\n[3] Dropped palindromic: {n_palin:,}; remaining {len(df_h):,}")

    print("\n[4] Allele harmonisation (BETA = log(OR))...")
    def harmonise(row):
        a1, a2 = str(row["A1"]).upper(), str(row["A2"]).upper()
        a1h, a2h = str(row["A1_h3"]).upper(), str(row["A2_h3"]).upper()
        try:
            beta = np.log(float(row["OR"]))
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

    print("\n[5] Computing Z = BETA / SE...")
    df_h["SE"] = pd.to_numeric(df_h["SE"], errors="coerce")
    df_h["Z"] = df_h["BETA_aligned"] / df_h["SE"]
    df_h = df_h[np.isfinite(df_h["Z"])].reset_index(drop=True)
    print(f"  Final: {len(df_h):,} SNPs")

    # Fixed N for all rows
    df_h["N"] = n_eff

    print("\n[6] Writing LDSC sumstats...")
    out = df_h[["SNP", "A1", "A2", "N", "Z"]].copy()
    out_path = f"{output_prefix}.sumstats.gz"
    out.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  Wrote {len(out):,} rows to {out_path}")
    print(f"  N (fixed) = {n_eff:.1f}")
    print(f"  Mean Z² = {(out['Z'] ** 2).mean():.4f}")
    print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal-daner munge for LDSC (fixed N)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--ncas", type=int, required=True)
    parser.add_argument("--ncon", type=int, required=True)
    args = parser.parse_args()
    munge_minimal(args.input, args.output_prefix, args.ncas, args.ncon)
