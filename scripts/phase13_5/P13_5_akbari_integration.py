#!/usr/bin/env python3
"""
Phase 13.5: Akbari et al. 2026 (Nature) Ancient DNA Selection Integration
==========================================================================
Integrates Akbari et al. 2026 ancient-DNA selection coefficients with our
20,766 PGC3 credible set variant master.

Akbari schema (GRCh37):
  CHROM, POS, REF, ALT, ANC, ID, RSID, AF, S, SE, X, P_X, POSTERIOR, FDR,
  CHI2_BE, FILTER

Tests added (within-locus + MAF residualized):
  T1. |iHS| × |Akbari S|     concordance of two independent selection metrics
  T2. |Akbari S| × age        ancient DNA replication of |iHS| × age
  T3. Brain spec × Akbari S   pleiotropy and selection direction
  T4. |Akbari S| × SDS        cross-method consistency

Output:
  results/phase11/variant_master_v3.parquet
  results/phase13_5/P13_5_akbari_results.tsv
  results/phase13_5/P13_5_NARRATIVE.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase13_5"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


log(f"Phase 13.5: Akbari 2026 Integration — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# Load PGC3 variant master
m = pd.read_parquet(P11 / "variant_master_v2.parquet")
log(f"\nPGC3 variant master: {len(m)} variants")

# Normalize chr to clean string ('1' not '1.0'); P11 v5 stores chr as numeric
# after FIX-A5 numeric coercion, so direct astype(str) gives '1.0' which
# does not match Akbari's '1'.
def _chr_clean(x):
    try:
        v = float(x)
        if pd.isna(v):
            return ""
        return str(int(v))
    except (ValueError, TypeError):
        return str(x)

m["chr_str"] = m["chr"].map(_chr_clean)
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce").astype("Int64")
m_keys_pre = m
m = m[m["chr_str"] != ""].copy()
log(f"  After chr_str cleanup: {len(m)} variants (dropped {len(m_keys_pre)-len(m)} with invalid chr)")
m_keys = m[m["pos_int"].notna()].copy()

pgc3_pos_by_chr = {}
for c, grp in m_keys.groupby("chr_str"):
    pgc3_pos_by_chr[c] = set(grp["pos_int"].tolist())
log(f"Unique chromosomes in PGC3: {sorted(pgc3_pos_by_chr.keys(), key=lambda x: int(x) if x.isdigit() else 99)}")


# Stream Akbari TSV with chunk filtering
akbari_path = BASE / "data/raw/annotations/akbari_2026/Selection_Summary_Statistics_01OCT2025.tsv.gz"
log(f"\nStreaming Akbari from {akbari_path}")

chunks = []
n_total = 0
n_kept = 0
chunk_size = 500_000

reader = pd.read_csv(akbari_path, sep='\t', comment='#', chunksize=chunk_size,
                      dtype={'CHROM': str, 'POS': 'Int64'})

for i, chunk in enumerate(reader):
    n_total += len(chunk)
    chunk["CHROM"] = chunk["CHROM"].astype(str)
    chunk = chunk[chunk["CHROM"].isin(pgc3_pos_by_chr.keys())]
    if len(chunk) == 0:
        continue
    # Vectorized position filter per chromosome
    keep_mask = pd.Series(False, index=chunk.index)
    for c in chunk["CHROM"].unique():
        positions = pgc3_pos_by_chr.get(c, set())
        if not positions: continue
        sub_idx = chunk[chunk["CHROM"] == c].index
        keep_mask.loc[sub_idx] = chunk.loc[sub_idx, "POS"].isin(positions)
    chunk = chunk[keep_mask]
    n_kept += len(chunk)
    chunks.append(chunk)
    if (i + 1) % 5 == 0:
        log(f"  Processed {n_total/1e6:.1f}M rows, kept {n_kept} hits")

log(f"\nTotal Akbari rows scanned: {n_total/1e6:.2f}M")
log(f"Hits at PGC3 positions: {n_kept}")

akbari_pgc = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
log(f"Akbari subset for PGC3: {len(akbari_pgc)} rows")


# Merge by chr+pos+REF+ALT (with allele orientation matching)
log(f"\n[2] Merging Akbari with PGC3 by chr+pos+ref+alt orientation")

akbari_pgc["chr_str"] = akbari_pgc["CHROM"].astype(str)
akbari_pgc["pos_int"] = pd.to_numeric(akbari_pgc["POS"], errors="coerce").astype("Int64")

# FIX-D-13_5-1: prior code matched on forward orientation only (REF|ALT vs
# effect|other in either order). Strand-flipped variants where PGC3 reports
# A/G but Akbari reports T/C (reverse-complement) were silently dropped,
# causing ascertainment bias. Add reverse-complement matching for non-
# palindromic variants. Palindromic A/T and C/G pairs cannot be resolved
# without frequency info — they remain unmatched here and are excluded from
# downstream sign-sensitive Akbari analyses.

def _rc(a):
    if not isinstance(a, str) or len(a) != 1:
        return a
    return {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}.get(a.upper(), a)


def _is_palin(a1, a2):
    if not (isinstance(a1, str) and isinstance(a2, str)
            and len(a1) == 1 and len(a2) == 1):
        return False
    pair = {a1.upper(), a2.upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}

m_keys["allele_pair_a"] = m_keys["effect_allele"].astype(str) + "|" + m_keys["other_allele"].astype(str)
m_keys["allele_pair_b"] = m_keys["other_allele"].astype(str) + "|" + m_keys["effect_allele"].astype(str)
m_keys["palindromic"] = [_is_palin(a, b) for a, b in zip(m_keys["effect_allele"], m_keys["other_allele"])]
# Reverse-complement pairs (skip for palindromic variants)
m_keys["allele_pair_a_rc"] = [
    _rc(a) + "|" + _rc(b) if not p else "____"
    for a, b, p in zip(m_keys["effect_allele"], m_keys["other_allele"], m_keys["palindromic"])
]
m_keys["allele_pair_b_rc"] = [
    _rc(b) + "|" + _rc(a) if not p else "____"
    for a, b, p in zip(m_keys["effect_allele"], m_keys["other_allele"], m_keys["palindromic"])
]
akbari_pgc["akbari_pair"] = akbari_pgc["REF"].astype(str) + "|" + akbari_pgc["ALT"].astype(str)

merged = m_keys[["rsid", "credible_set_id", "chr_str", "pos_int", "effect_allele",
                  "other_allele", "palindromic",
                  "allele_pair_a", "allele_pair_b",
                  "allele_pair_a_rc", "allele_pair_b_rc"]].merge(
    akbari_pgc[["chr_str", "pos_int", "REF", "ALT", "RSID", "AF", "S", "SE", "X",
                  "P_X", "POSTERIOR", "FDR", "CHI2_BE", "FILTER", "akbari_pair"]],
    on=["chr_str", "pos_int"], how="left"
)
log(f"  Merged rows (1+ Akbari hits per PGC3): {len(merged)}")
log(f"  PGC3 variants with any Akbari hit: {merged[merged['S'].notna()]['rsid'].nunique()}")

# Forward-orientation match (effect=ALT or effect=REF, no strand flip)
forward_a = (merged["allele_pair_a"] == merged["akbari_pair"])
forward_b = (merged["allele_pair_b"] == merged["akbari_pair"])
# Reverse-complement match for non-palindromic
rc_a = (merged["allele_pair_a_rc"] == merged["akbari_pair"])
rc_b = (merged["allele_pair_b_rc"] == merged["akbari_pair"])

matched = merged[forward_a | forward_b | rc_a | rc_b].copy()
matched["_orient"] = "unknown"
matched.loc[forward_a, "_orient"] = "fwd_a"  # effect_allele == ALT
matched.loc[forward_b & ~forward_a, "_orient"] = "fwd_b"  # effect_allele == REF
matched.loc[rc_a & ~forward_a & ~forward_b, "_orient"] = "rc_a"  # rc(effect) == ALT
matched.loc[rc_b & ~forward_a & ~forward_b & ~rc_a, "_orient"] = "rc_b"  # rc(effect) == REF
log(f"  Allele-matched rows (incl rev-comp): {len(matched)}")
log(f"  Orientation breakdown: {matched['_orient'].value_counts().to_dict()}")
n_palin_unmatched = (
    merged["palindromic"]
    & merged["S"].notna()
    & ~(forward_a | forward_b | rc_a | rc_b)
).sum()
log(f"  Palindromic variants without forward match (sign-ambiguous, dropped): "
    f"{n_palin_unmatched}")

# Sign convention: Akbari S is selection coefficient on ALT (Akbari schema).
# When effect_allele aligns with ALT (orient = fwd_a or rc_a) → S sign matches PGC3.
# When effect_allele aligns with REF (orient = fwd_b or rc_b) → S sign flipped.
# (The rc cases additionally interpret the *complement* of the original allele.)
align_alt = matched["_orient"].isin(["fwd_a", "rc_a"])
matched["s_for_effect_allele"] = np.where(align_alt, matched["S"], -matched["S"])
matched["x_for_effect_allele"] = np.where(align_alt, matched["X"], -matched["X"])

# Tie-breaker for duplicate (rsid, locus): primary FDR (lower = more significant),
# secondary POSTERIOR (HIGHER = more confident in directional selection). Use a
# negated POSTERIOR sort key with ascending=True so that ties on FDR resolve to
# the higher-POSTERIOR record. (Verified by audit 2026-05-03.)
matched["_neg_post"] = -matched["POSTERIOR"].fillna(-1)  # NaN treated as least confident
matched = matched.sort_values(
    ["rsid", "FDR", "_neg_post"], na_position="last"
).drop_duplicates("rsid", keep="first").drop(columns=["_neg_post"])
log(f"  Unique PGC3 variants with allele-matched Akbari: {len(matched)}")


# Build variant_master_v3
log(f"\n[3] Building variant_master_v3.parquet")

akbari_cols = matched[["rsid", "S", "SE", "X", "P_X", "POSTERIOR", "FDR", "CHI2_BE",
                         "FILTER", "AF", "s_for_effect_allele", "x_for_effect_allele"]].copy()
akbari_cols.columns = ["rsid"] + [f"akbari_{c}" for c in
                                     ["s", "se", "x", "p", "pi", "fdr", "chi2_be",
                                      "filter", "af_alt", "s_effect", "x_effect"]]

m_v3 = m.merge(akbari_cols, on="rsid", how="left")
log(f"  v3 master: {len(m_v3)} rows × {len(m_v3.columns)} cols")
log(f"  Akbari coverage: {m_v3['akbari_s'].notna().sum()}/{len(m_v3)} "
    f"({m_v3['akbari_s'].notna().mean()*100:.1f}%)")

m_v3.to_parquet(P11 / "variant_master_v3.parquet", index=False, compression="snappy")
log(f"  Saved: {P11 / 'variant_master_v3.parquet'}")


# Statistical tests
log(f"\n[4] Within-locus + MAF residualized tests")

m_age = m_v3[m_v3["age_median_yr"].notna() & (m_v3["age_median_yr"] > 0) &
              m_v3["maf"].notna() & m_v3["akbari_s"].notna()].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["abs_akbari_s"] = m_age["akbari_s"].abs()
m_age["abs_akbari_x"] = m_age["akbari_x"].abs()
m_age["abs_ihs"] = m_age["ihs_std"].abs() if "ihs_std" in m_age.columns else np.nan

m_age["b_logp"]  = np.where(m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0),
                              -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m_age["bl_logp"] = np.where(m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0),
                              -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m_age["brain_spec"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])

m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int_num"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int_num"] >= 43_000_000) & \
          (m_age["pos_int_num"] <= 46_000_000)


import sys as _sys
_sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation (code-review Faz C lib)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


tests = [
    ("|Akbari S| × age", "log_age", "abs_akbari_s", m_age),
    ("|Akbari S| × age, no 17q21.31", "log_age", "abs_akbari_s", m_age[~in_mapt]),
    ("|Akbari X| × age", "log_age", "abs_akbari_x", m_age),
    ("|iHS| × |Akbari S| consistency", "abs_ihs", "abs_akbari_s", m_age),
    ("|iHS| × |Akbari X|", "abs_ihs", "abs_akbari_x", m_age),
    ("Brain spec × Akbari S (signed)", "brain_spec", "akbari_s", m_age),
    ("Brain spec × |Akbari S|", "brain_spec", "abs_akbari_s", m_age),
    ("|Akbari S| × SDS consistency", "sds", "abs_akbari_s", m_age),
    ("Akbari posterior π × age", "log_age", "akbari_pi", m_age),
]

results = []
log("\n  Test                                                n     rho      p")
log("  " + "-" * 72)
for label, x, y, df in tests:
    rho, p, n = maf_resid_rho(df, x, y)
    if rho is not None:
        results.append({"test": label, "n": n, "rho": rho, "p": p})
        log(f"  {label:<50s}  {n:>5d}  {rho:>+7.4f}  {p:.3e}")

from statsmodels.stats.multitest import multipletests
df_r = pd.DataFrame(results)
if len(df_r) > 0:
    df_r["fdr_q"] = multipletests(df_r["p"].fillna(1.0), method="fdr_bh")[1]
    df_r["sig"] = df_r["fdr_q"] < 0.05
    df_r.to_csv(OUT / "P13_5_akbari_results.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P13_5_akbari_results.tsv'}")

with open(OUT / "P13_5_NARRATIVE.md", "w") as f:
    f.write("# Phase 13.5: Akbari 2026 Ancient DNA Integration\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13.5 complete.")
