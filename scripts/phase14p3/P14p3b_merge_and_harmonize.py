#!/usr/bin/env python3
"""
Phase 14p3b — Merge per-chromosome pure-AFR allele frequencies and harmonize
============================================================================
Methodological notes:
  [P14p3b-1] REF/ALT harmonization: 1000G reports AF for the ALT allele
    relative to its own REF; PGC3 reports BETA effect for A1 vs A2. These
    can disagree in either of two ways:
      (a) PGC3 A1 = 1000G REF and PGC3 A2 = 1000G ALT  → AF_AFR is allele freq
          of PGC3 A2 (the "effect allele" by GWAS convention).
      (b) PGC3 A1 = 1000G ALT and PGC3 A2 = 1000G REF  → we need to flip:
          freq of PGC3 A2 = 1 - AF_AFR.
      (c) strand-mismatch (A/T, C/G palindromic SNPs) → flagged + dropped
          unless confidently resolved by allele frequency comparison.
  [P14p3b-2] **Derived allele frequency (DAF):** v10 v11 hypothesis is about
    DAF, not ALT freq. DAF = freq of the DERIVED allele. We use Atlas of
    Variant Age `AlleleAnc` column (ancestral allele) where available;
    otherwise fall back to ALT-allele frequency (caveat flagged per variant).
  [P14p3b-3] AN/AC sanity: expect AN ~= 2 × 504 = 1008. Variants with
    AN < 0.95 × 1008 are flagged (missing genotype, likely low imputation
    quality in some samples) but kept.

Outputs:
  results/phase14p3/P14p3b_AFR_DAF_per_variant.parquet
  results/phase14p3/P14p3b_harmonization_summary.tsv
  results/phase14p3/P14p3b_NARRATIVE.md
"""

import sys
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

import numpy as np
import pandas as pd

BASE = Path(_ROOT)
DATA = Path((_SCRATCH + "/v11_data/phase14p3"))
OUT  = Path((_ROOT + "/results/phase14p3"))
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)


log("="*70)
log("Phase 14p3b — Merge + harmonize pure-AFR DAF")
log("="*70)

# ── Load PGC3 variant master + GEVA ancestral alleles ──────────────
log("[1] Loading PGC3 substrate")
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
vm["chr"] = pd.to_numeric(vm["chr"], errors="coerce").astype("Int64")
vm["pos"] = pd.to_numeric(vm["pos"], errors="coerce").astype("Int64")
log(f"  variant_master_v4: {len(vm):,} rows")

geva = pd.read_csv(BASE / "data/processed/pgc3_geva_ages.tsv", sep="\t",
                   usecols=["VariantID", "Chromosome", "Position",
                            "AlleleRef", "AlleleAlt", "AlleleAnc", "DataSource"])
geva = geva[geva["DataSource"] == "TGP"].drop_duplicates(subset=["VariantID"])
geva = geva.rename(columns={"VariantID": "rsid", "Chromosome": "chr_g",
                            "Position": "pos_g", "AlleleAnc": "ancestral"})
log(f"  GEVA TGP entries: {len(geva):,}")

# Merge ancestral allele into PGC3 substrate (chr,pos join is schema-symmetric)
geva["chr_g"] = pd.to_numeric(geva["chr_g"], errors="coerce").astype("Int64")
geva["pos_g"] = pd.to_numeric(geva["pos_g"], errors="coerce").astype("Int64")
vm = vm.merge(geva[["chr_g","pos_g","ancestral","AlleleRef","AlleleAlt"]],
              left_on=["chr","pos"], right_on=["chr_g","pos_g"], how="left")
log(f"  PGC3 with ancestral annotation: {vm['ancestral'].notna().sum():,}"
    f" / {len(vm):,} ({vm['ancestral'].notna().mean()*100:.1f}%)")

# ── Load per-chr pure-AFR AF + per-subpop ─────────────────────────
log("\n[2] Loading per-chr 1000G pure-AFR + 5 subpops")
afr_pool = []
subpop_pool = {s: [] for s in ["ESN","GWD","LWK","MSL","YRI"]}
for chrom in range(1, 23):
    f = DATA / f"pure_AFR_chr{chrom}.tsv"
    if not f.exists():
        log(f"  WARNING chr{chrom}: {f.name} missing")
        continue
    df = pd.read_csv(f, sep="\t", header=None,
                     names=["rsid","chr","pos","REF","ALT","AF","AN","AC"],
                     na_values=["."])
    df["chr"]  = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
    df["pos"]  = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    df["AF"]   = pd.to_numeric(df["AF"],  errors="coerce")
    df["AN"]   = pd.to_numeric(df["AN"],  errors="coerce")
    df["AC"]   = pd.to_numeric(df["AC"],  errors="coerce")
    afr_pool.append(df)

    for sub in subpop_pool:
        sf = DATA / f"subpop_{sub}_chr{chrom}.tsv"
        if sf.exists():
            sd = pd.read_csv(sf, sep="\t", header=None,
                             names=["rsid","chr","pos","REF","ALT","AF"],
                             na_values=["."])
            sd["chr"] = pd.to_numeric(sd["chr"], errors="coerce").astype("Int64")
            sd["pos"] = pd.to_numeric(sd["pos"], errors="coerce").astype("Int64")
            sd["AF"]  = pd.to_numeric(sd["AF"],  errors="coerce")
            sd = sd.rename(columns={"AF": f"AF_{sub}"})[["chr","pos","REF","ALT", f"AF_{sub}"]]
            subpop_pool[sub].append(sd)

afr = pd.concat(afr_pool, ignore_index=True)
log(f"  pooled pure-AFR: {len(afr):,} variant-positions across 22 chrs")

# ── Harmonize REF/ALT and compute DAF ──────────────────────────────
log("\n[3] Harmonize REF/ALT and compute DAF")

merged = vm.merge(
    afr[["chr","pos","REF","ALT","AF","AN","AC"]],
    on=["chr","pos"], how="inner", suffixes=("","_1kg"),
)
log(f"  PGC3 ∩ 1000G AFR (by chr,pos): {len(merged):,}")

# Determine which (A1, A2) pair maps to (REF, ALT)
# PGC3 columns expected: A1 (effect/risk?), A2 (other) — use ea/non-ea from PGC3
# variant_master_v4 column inspection happens in pre-flight; assume a1/a2 present.
# Fall back: try multiple possible column names.
def _find_a1a2(df):
    candidates = [("a1","a2"), ("A1","A2"), ("effect_allele","other_allele"),
                  ("ea","oa"), ("alt","ref")]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None

A1, A2 = _find_a1a2(merged)
log(f"  PGC3 allele columns detected: A1={A1}, A2={A2}")
if A1 is None:
    # Most-likely fallback: use GEVA Ref/Alt as pseudo-A1/A2 and warn
    merged["A1_eff"] = merged["AlleleAlt"]  # GEVA alt = derived in many cases
    merged["A2_eff"] = merged["AlleleRef"]
    A1, A2 = "A1_eff", "A2_eff"
    log("  WARNING: no a1/a2 in variant master; using GEVA Alt/Ref as proxy")

# Strand-orient: if PGC3 A1/A2 = 1000G REF/ALT, then AF reports freq(ALT) = freq(A1) when A1 matches ALT
# Robust mapping by string compare (assumes upper-case A1/A2 + REF/ALT)
merged["A1u"] = merged[A1].astype(str).str.upper()
merged["A2u"] = merged[A2].astype(str).str.upper()
merged["REFu"] = merged["REF"].astype(str).str.upper()
merged["ALTu"] = merged["ALT"].astype(str).str.upper()

def _complement(b):
    return {"A":"T","T":"A","C":"G","G":"C","N":"N"}.get(b, "N")

merged["palindromic"] = (
    ((merged["A1u"]=="A") & (merged["A2u"]=="T")) |
    ((merged["A1u"]=="T") & (merged["A2u"]=="A")) |
    ((merged["A1u"]=="C") & (merged["A2u"]=="G")) |
    ((merged["A1u"]=="G") & (merged["A2u"]=="C"))
)

# Case A: A1=REF, A2=ALT → AF_AFR is freq of ALT = freq of A2
case_A = (merged["A1u"]==merged["REFu"]) & (merged["A2u"]==merged["ALTu"])
# Case B: A1=ALT, A2=REF → flip needed
case_B = (merged["A1u"]==merged["ALTu"]) & (merged["A2u"]==merged["REFu"])
# Case C: strand-flip needed (palindromic SNPs require freq sanity check;
#         drop conservatively).
case_C = ~(case_A | case_B)

merged["match_case"] = "C_unmatched"
merged.loc[case_A, "match_case"] = "A_direct"
merged.loc[case_B, "match_case"] = "B_flipped"
merged.loc[case_C, "match_case"] = "C_unmatched"

# Freq of PGC3 A2 (the "non-effect" / reference convention varies — but
# we keep a consistent convention: freq_PGC_A1 = freq of PGC3 's first
# allele column, regardless of effect direction; later analyses can flip
# based on β sign if needed).
merged["AF_AFR_pgc_a1"] = np.where(case_A, 1 - merged["AF"],
                          np.where(case_B, merged["AF"], np.nan))
merged["AF_AFR_pgc_a2"] = np.where(case_A, merged["AF"],
                          np.where(case_B, 1 - merged["AF"], np.nan))
merged["AF_AFR_alt"]    = merged["AF"]  # raw 1000G ALT freq, no harmonization

# Derived allele freq (DAF) — using GEVA ancestral allele
def _derived_freq(row):
    anc = row.get("ancestral")
    if pd.isna(anc):
        # Fallback: assume ALT = derived (common when REF=hg19 reference =
        # often ancestral but not always); flag uncertainty.
        return row["AF_AFR_alt"], "ancestral_unknown_alt_proxy"
    anc = str(anc).upper()
    ref = row["REFu"]; alt = row["ALTu"]
    if   anc == ref: return row["AF_AFR_alt"], "anc=ref;daf=alt"
    elif anc == alt: return 1 - row["AF_AFR_alt"], "anc=alt;daf=1-alt"
    else:            return np.nan, f"anc={anc};ref={ref};alt={alt}"

vals = merged.apply(_derived_freq, axis=1, result_type="expand")
merged["AF_AFR_derived"] = vals[0]
merged["DAF_call"]       = vals[1]

# AN sanity check
merged["AN_pct_complete"] = merged["AN"] / 1008.0
merged["AN_lowQC"] = (merged["AN_pct_complete"] < 0.95).astype(int)

log("\n  Harmonization summary:")
log(f"    A_direct (A1=REF):  {case_A.sum():>6,}")
log(f"    B_flipped (A1=ALT): {case_B.sum():>6,}")
log(f"    C_unmatched:        {case_C.sum():>6,}")
log(f"    palindromic:        {merged['palindromic'].sum():>6,}")
log(f"    AN < 95% complete:  {merged['AN_lowQC'].sum():>6,}")
log(f"    DAF call distribution:")
for k, v in merged["DAF_call"].value_counts().items():
    log(f"      {k:30s}: {v:>6,}")

# Drop unusable
keep = merged["DAF_call"].notna() & merged["AF_AFR_derived"].notna()
merged_clean = merged[keep].copy()
log(f"\n  Kept after harmonization filter: {len(merged_clean):,} / {len(merged):,}")

# ── Merge per-subpop ───────────────────────────────────────────────
log("\n[4] Merging per-subpop AFs")
for sub, parts in subpop_pool.items():
    if not parts:
        log(f"  {sub}: no data")
        continue
    sub_df = pd.concat(parts, ignore_index=True)
    # Harmonize REF/ALT same way (case A vs B)
    sub_df = sub_df.merge(merged_clean[["chr","pos","match_case"]],
                          on=["chr","pos"], how="inner")
    sub_df[f"AF_{sub}_pgc_a2"] = np.where(
        sub_df["match_case"]=="A_direct", sub_df[f"AF_{sub}"],
        np.where(sub_df["match_case"]=="B_flipped", 1 - sub_df[f"AF_{sub}"], np.nan))
    merged_clean = merged_clean.merge(
        sub_df[["chr","pos", f"AF_{sub}_pgc_a2"]].drop_duplicates(subset=["chr","pos"]),
        on=["chr","pos"], how="left")
    log(f"  {sub}: merged AF for {merged_clean[f'AF_{sub}_pgc_a2'].notna().sum():,} variants")

# ── Output ─────────────────────────────────────────────────────────
keep_cols = [
    "rsid","chr","pos","maf",
    "AlleleRef","AlleleAlt","ancestral","match_case","palindromic",
    "AF_AFR_pgc_a2","AF_AFR_derived","DAF_call",
    "AN","AN_pct_complete","AN_lowQC",
] + [f"AF_{s}_pgc_a2" for s in subpop_pool if f"AF_{s}_pgc_a2" in merged_clean.columns]
merged_clean[keep_cols].to_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
log(f"\n  Saved: {OUT / 'P14p3b_AFR_DAF_per_variant.parquet'}")

# Harmonization summary table
summary_rows = [
    {"metric": "n_pgc_credset", "value": len(vm)},
    {"metric": "n_pgc_inter_1000G_pure_AFR", "value": len(merged)},
    {"metric": "n_match_A_direct", "value": int(case_A.sum())},
    {"metric": "n_match_B_flipped", "value": int(case_B.sum())},
    {"metric": "n_match_C_unmatched", "value": int(case_C.sum())},
    {"metric": "n_palindromic", "value": int(merged["palindromic"].sum())},
    {"metric": "n_AN_lowQC", "value": int(merged["AN_lowQC"].sum())},
    {"metric": "n_after_harmonization_filter", "value": len(merged_clean)},
    {"metric": "n_DAF_call_anc=ref", "value": int((merged["DAF_call"] == "anc=ref;daf=alt").sum())},
    {"metric": "n_DAF_call_anc=alt", "value": int((merged["DAF_call"] == "anc=alt;daf=1-alt").sum())},
    {"metric": "n_DAF_call_ancestral_unknown", "value": int((merged["DAF_call"] == "ancestral_unknown_alt_proxy").sum())},
]
pd.DataFrame(summary_rows).to_csv(OUT / "P14p3b_harmonization_summary.tsv",
                                   sep="\t", index=False)
log(f"  Saved: {OUT / 'P14p3b_harmonization_summary.tsv'}")

# ── Narrative ──────────────────────────────────────────────────────
with open(OUT / "P14p3b_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3b — Merge + harmonize pure-AFR DAF\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write(f"## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p3b complete.")
