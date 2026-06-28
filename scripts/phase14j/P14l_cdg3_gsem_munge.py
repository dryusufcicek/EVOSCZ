#!/usr/bin/env python3
"""
Phase 14l: CDG3 (Grotzinger 2025) Genomic-SEM-derived factor sumstats munge to LDSC format.

CDG3 factor sumstats are tsv with:
  SNP, CHR, BP, MAF, A1, A2, BETA, SE, P, Q_P

BETA is on log-linear scale (factor effect, not OR). Z = BETA / SE directly (no log transform).
Sample size (N) is the Genomic SEM "implied N" — supplied via CLI (e.g. F3 = 84,760).

LDSC compatibility verified: Grotzinger 2025 themselves apply S-LDSC enrichment to factor sumstats
(excitatory-neuron F2; oligodendrocyte F4). Factor Z-scores are LDSC-input-ready.

Schema in (CDG3 tsv): SNP, CHR, BP, MAF, A1, A2, BETA, SE, P, Q_P
Schema out (LDSC): SNP, A1, A2, N, Z

QSNP filter: NOT applied in main munge (raw factor sumstats). Sensitivity-only filter
deferred to a separate run with --qsnp-threshold flag if needed.

Usage:
  python3 P14l_cdg3_gsem_munge.py --input <cdg3_factor.tsv.gz> \\
    --output-prefix <prefix> --implied-n <N>
"""
import argparse
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import os
import gzip
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HM3 = str(BASE / "data/ldsc/sldsc_ref/w_hm3.snplist")


def revcomp(a):
    if not isinstance(a, str) or len(a) != 1:
        return a
    return {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}.get(a.upper(), a)


def is_palindromic(a1, a2):
    if not (isinstance(a1, str) and isinstance(a2, str)
            and len(a1) == 1 and len(a2) == 1):
        return False
    pair = {a1.upper(), a2.upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}


def munge_cdg3(input_path: str, output_prefix: str, implied_n: float) -> None:
    print(f"Phase 14l CDG3 GSEM munge — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_prefix}.sumstats.gz")
    print(f"  Implied N (Genomic SEM): {implied_n:,.0f}")

    print("\n[1] Loading CDG3 factor sumstats...")
    df = pd.read_csv(input_path, sep='\t', compression='gzip')
    df.columns = df.columns.str.strip()
    print(f"  Total: {len(df):,}; Cols: {list(df.columns)}")

    expected = {"SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"FATAL: missing expected columns: {missing}")

    print("\n[2] HapMap3 restriction...")
    hm3 = pd.read_csv(HM3, sep="\t")
    df_h = df.merge(hm3, on="SNP", how="inner", suffixes=("", "_h3"))
    print(f"  HM3 merge: {len(df_h):,}")

    df_h["palin"] = [is_palindromic(a1, a2) for a1, a2 in zip(df_h["A1"], df_h["A2"])]
    n_palin = int(df_h["palin"].sum())
    df_h = df_h[~df_h["palin"]].drop(columns=["palin"]).reset_index(drop=True)
    print(f"\n[3] Dropped palindromic: {n_palin:,}; remaining {len(df_h):,}")

    print("\n[4] Allele harmonisation (BETA direct, sign-flip for reverse match)...")
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

    print("\n[5] Computing Z = BETA_aligned / SE (BETA already log-scale, no transform)...")
    df_h["SE"] = pd.to_numeric(df_h["SE"], errors="coerce")
    df_h["Z"] = df_h["BETA_aligned"] / df_h["SE"]
    df_h = df_h[np.isfinite(df_h["Z"])].reset_index(drop=True)
    print(f"  Final: {len(df_h):,} SNPs")

    df_h["N"] = implied_n  # Fixed implied N from Genomic SEM

    print("\n[6] Writing LDSC sumstats...")
    out = df_h[["SNP", "A1", "A2", "N", "Z"]].copy()
    out_path = f"{output_prefix}.sumstats.gz"
    out.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  Wrote {len(out):,} rows to {out_path}")
    print(f"  N (fixed) = {implied_n:,.0f}")
    print(f"  Mean Z² = {(out['Z'] ** 2).mean():.4f}")
    print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDG3 GSEM factor sumstats munge for LDSC")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--implied-n", type=float, required=True,
                        help="Genomic SEM implied N (e.g. 84760 for F3, 1637337 for F4)")
    args = parser.parse_args()
    munge_cdg3(args.input, args.output_prefix, args.implied_n)
