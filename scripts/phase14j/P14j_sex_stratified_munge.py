#!/usr/bin/env python3
"""
Phase 14j: Sex-stratified PGC3 SCZ daner-format sumstats munge to LDSC format.

Adapts P14h_eas_sumstats_munge.py to handle daner-format files (PGC2 era):
  - Daner columns: CHR, SNP, BP, A1, A2, FRQ_A_<Nca>, FRQ_U_<Nco>, INFO, OR, SE, P,
    ngt, Direction, HetISqt, HetDf, HetPVa, Nca, Nco, Neff
  - PGC3-public columns (EAS reference): chr, ID, pos, A1, A2, FCAS, FCON, IMPINFO,
    BETA, SE, PVAL, NCAS, NCON

Key differences vs P14h_eas:
  [DANER-1] Daner reports OR (odds ratio); EAS reports BETA (log-OR). We compute
    BETA = log(OR) before Z-score = BETA / SE.
  [DANER-2] Daner provides per-row Nca, Nco, Neff columns (imputation-quality-
    adjusted). We use N_eff = 4 * Nca * Nco / (Nca + Nco) for LDSC consistency
    with manuscript convention (line 80) rather than the daner Neff column,
    which has different scaling.
  [DANER-3] No VCF-style ## headers; tab-delimited directly.

Code-review fixes (FIX-E-H-1, -2, -3) applied identically:
  [FIX-E-H-1] Strand-ambiguous (A/T, C/G) palindromic variants pre-filtered.
  [FIX-E-H-2] Reverse-complement allele matching for non-palindromic strand flips.
  [FIX-E-H-3] Effective sample size: N = 4 * Nca * Nco / (Nca + Nco).

Schema in (daner): CHR, SNP, BP, A1, A2, FRQ_A_*, FRQ_U_*, INFO, OR, SE, P, ..., Nca, Nco, Neff
Schema out (LDSC): SNP, A1, A2, N, Z

Usage:
  python3 P14j_sex_stratified_munge.py --input <daner.gz> --output-prefix <prefix>

Examples:
  python3 P14j_sex_stratified_munge.py \\
    --input ${PGC_SUMSTATS_DIR}/daner_PGC_SCZ_w3_75_0618a_eur_male.gz \\
    --output-prefix ${EVOSCZ_ROOT}/results/phase14j/PGC3_SCZ_EUR_male
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


def munge_daner(input_path: str, output_prefix: str) -> None:
    print(f"Phase 14j daner munge — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_prefix}.sumstats.gz")

    # [1] Load daner sumstats (no ## VCF headers).
    # Auto-detect separator: tab (PGC standard, C engine fast) vs space (BD2024 raw).
    # Tab-delimited files preferred — pre-process via `tr -s ' ' '\t'` if needed.
    print("\n[1] Loading daner sumstats...")
    with gzip.open(input_path, 'rt') as _f:
        _first = _f.readline().rstrip('\n')
    if '\t' in _first:
        df = pd.read_csv(input_path, sep='\t', compression='gzip')
        print(f"  Separator: tab (C engine)")
    else:
        df = pd.read_csv(input_path, sep=r'\s+', engine='python', compression='gzip')
        print(f"  Separator: whitespace regex (Python engine — slower)")
    df.columns = df.columns.str.strip()
    print(f"  Total: {len(df):,} variants")
    print(f"  Cols: {list(df.columns)}")

    # Verify expected daner columns present
    expected = {"CHR", "SNP", "BP", "A1", "A2", "INFO", "OR", "SE", "P", "Nca", "Nco"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"FATAL: missing expected daner columns: {missing}")

    # [2] HapMap3 SNP restriction
    print("\n[2] Restricting to HapMap3 SNPs...")
    if not os.path.exists(HM3):
        sys.exit(f"FATAL: HM3 snplist not found at {HM3}")
    hm3 = pd.read_csv(HM3, sep="\t")
    print(f"  HM3: {len(hm3):,}")

    df_h = df.merge(hm3, on="SNP", how="inner", suffixes=("", "_h3"))
    print(f"  After HM3 merge: {len(df_h):,}")

    # [3] FIX-E-H-1: Pre-filter strand-ambiguous palindromic variants
    df_h["palin"] = [is_palindromic(a1, a2) for a1, a2 in zip(df_h["A1"], df_h["A2"])]
    n_palin = int(df_h["palin"].sum())
    df_h = df_h[~df_h["palin"]].drop(columns=["palin"]).reset_index(drop=True)
    print(f"\n[3] Dropped strand-ambiguous (A/T, C/G) palindromic: {n_palin:,}")
    print(f"  Remaining: {len(df_h):,}")

    # [4] Allele harmonisation with reverse-complement (FIX-E-H-2)
    print("\n[4] Harmonising alleles vs HM3 reference (forward / flip / rev-comp / rev-comp-flip)...")

    def harmonise(row):
        """Returns (BETA_aligned, valid_flag).
        BETA_aligned: log(OR) sign-corrected to match HM3 A1.
        valid_flag: 1 if matched, 0 if drop.
        """
        a1, a2 = str(row["A1"]).upper(), str(row["A2"]).upper()
        a1h, a2h = str(row["A1_h3"]).upper(), str(row["A2_h3"]).upper()
        # DANER-1: BETA = log(OR)
        try:
            beta = np.log(float(row["OR"]))
        except (ValueError, TypeError):
            return (np.nan, 0)
        if np.isnan(beta) or np.isinf(beta):
            return (np.nan, 0)
        # Forward match
        if a1 == a1h and a2 == a2h:
            return (beta, 1)
        # Flip
        if a1 == a2h and a2 == a1h:
            return (-beta, 1)
        # Reverse complement (palindromics already dropped)
        rc1, rc2 = revcomp(a1), revcomp(a2)
        if rc1 == a1h and rc2 == a2h:
            return (beta, 1)
        if rc1 == a2h and rc2 == a1h:
            return (-beta, 1)
        return (np.nan, 0)

    harmonised = df_h.apply(harmonise, axis=1, result_type="expand")
    df_h["BETA_aligned"] = harmonised[0]
    df_h["valid"] = harmonised[1]
    n_unmatched = int((df_h["valid"] == 0).sum())
    df_h = df_h[df_h["valid"] == 1].reset_index(drop=True)
    print(f"  Dropped unmatched alleles: {n_unmatched:,}")
    print(f"  Remaining: {len(df_h):,}")

    # [5] Z-score = BETA_aligned / SE
    print("\n[5] Computing Z-score = BETA_aligned / SE...")
    df_h["SE"] = pd.to_numeric(df_h["SE"], errors="coerce")
    df_h["Z"] = df_h["BETA_aligned"] / df_h["SE"]
    n_invalid_z = int(df_h["Z"].isna().sum() + (~np.isfinite(df_h["Z"])).sum())
    df_h = df_h[np.isfinite(df_h["Z"])].reset_index(drop=True)
    print(f"  Dropped non-finite Z: {n_invalid_z:,}")
    print(f"  Remaining: {len(df_h):,}")

    # [6] FIX-E-H-3: Effective sample size N = 4 * Nca * Nco / (Nca + Nco)
    print("\n[6] Computing N_eff = 4 * Nca * Nco / (Nca + Nco) per row...")
    df_h["Nca"] = pd.to_numeric(df_h["Nca"], errors="coerce")
    df_h["Nco"] = pd.to_numeric(df_h["Nco"], errors="coerce")
    df_h["N"] = 4 * df_h["Nca"] * df_h["Nco"] / (df_h["Nca"] + df_h["Nco"])
    df_h = df_h[df_h["N"].notna() & (df_h["N"] > 0)].reset_index(drop=True)
    print(f"  N_eff range: [{df_h['N'].min():.1f}, {df_h['N'].max():.1f}], mean={df_h['N'].mean():.1f}")

    # [7] Output LDSC format: SNP A1 A2 N Z
    print("\n[7] Writing LDSC sumstats...")
    out = df_h[["SNP", "A1", "A2", "N", "Z"]].copy()
    out_path = f"{output_prefix}.sumstats.gz"
    out.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  Wrote {len(out):,} rows to {out_path}")
    print(f"  Mean N = {out['N'].mean():.1f}")
    print(f"  Mean Z² = {(out['Z'] ** 2).mean():.4f} (should be > 1 for polygenic signal)")

    print(f"\nDone — {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sex-stratified daner munge for LDSC")
    parser.add_argument("--input", required=True, help="Input daner-format .gz file")
    parser.add_argument("--output-prefix", required=True,
                        help="Output prefix (will append .sumstats.gz)")
    args = parser.parse_args()
    munge_daner(args.input, args.output_prefix)
