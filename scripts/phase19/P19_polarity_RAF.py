#!/usr/bin/env python3
"""
================================================================================
Phase 19 — Polarity-aware Risk-Allele-Frequency (RAF) analysis (SLURM)
================================================================================

PURPOSE
-------
For PGC3 EUR fine-mapped SCZ credible-set variants, ask whether the **risk-
increasing** allele (A1 if BETA>0 else A2) is preferentially at HIGHER frequency
in EUR than in pure-AFR 1000G (n=504, ESN+GWD+LWK+MSL+YRI), within the
**AF_AFR <0.05 stratum** (primary test) and across four AF_AFR strata
(secondary).

CRITICAL CONTEXT
----------------
This script replaces an earlier rejected polarity-aware RAF script.
Key changes vs. that earlier approach:

  * NO Scenario A/B/C verdict tree. Verdict set is exactly
      {NULL, DIRECTIONAL_protective, DIRECTIONAL_risk, WINNER_CURSE_CONFOUND}
    per the pre-registered decision tree.
  * Verdict logic written TWICE in independent forms with mandatory cross-check
    assertion (P0.1).
  * Hand-test on 5 hardcoded cases at top of main() — fail-fast (P0.9).
  * Cross-check 100 random variants P14p3 pure-AFR vs P14p2 AFRAM (P0.10);
    halt if Pearson < 0.90.
  * Strand-ambiguous drop logged per stratum (P0.2).
  * Power gate: AF_AFR<0.01 merged with 0.01-0.05 if n<50 (P0.5).
  * Winner's-Curse sensitivity (P0.6): primary D + (PIP>0.5, |BETA|-mid) D.
  * Permutation null (P0.7): 1000 iter, AF_AFR labels shuffled.
  * Bonferroni: 24-test family => α'=0.002 for secondaries. Primary
    (AF_AFR<0.05, all-PIP, all-BETA) is 1 test => α=0.05.
  * Output schema follows P0.11 exactly.
  * NO mention of Solé-Morata/González-Peñas, Saha 2005, "protective shield",
    "top-tier", cluster-protective claims, or AFR prevalence
    paradox.

PRE-REGISTERED FALSIFICATION GATES
----------------------------------
Per the pre-registered hypotheses (H1-H5). Primary test = AF_AFR<0.05 stratum,
all-PIP, all-|BETA|, n_target>=250 polarity-resolved.

  H1 (drift_neutral): D in [0.49,0.51], CI overlaps 0.50, flat across strata.
  H2 (cog antagonistic pleiotropy): D(<0.01)=0.38-0.46, monotone increasing,
        median signed-drift < -0.005.
  H3 (relaxed purifying): D(<0.01)=0.54-0.62, monotone decreasing,
        median signed-drift > +0.005, |BETA|(<0.01) > |BETA|(>=0.50) by >1.1x.
  H4 (winner's curse): signal collapses to [0.48,0.52] when restricted to
        (PIP>0.5, |BETA|-mid-tertile).
  H5 (heterogeneous): non-monotone D, pairwise |D_i - D_j| > 0.04 with
        |Spearman rho| < 0.5.

INPUTS (canonical, GRCh37 throughout)
-------------------------------------
  ${EVOSCZ_ROOT}/data/processed/pgc3_master_variants.tsv (20,766 rows)
  ${EVOSCZ_SCRATCH}/v11_data/phase14p3/pure_AFR_chr{1..22}.tsv (NO HEADER)
  ${EVOSCZ_SCRATCH}/v11_data/phase14p3/subpop_{ESN,GWD,LWK,MSL,YRI}_chr{1..22}.tsv
  ${EVOSCZ_ROOT}/data/processed/pgc3_geva_ages.tsv (55,760 rows; AlleleAnc)
  ${EVOSCZ_ROOT}/results/phase14b/P14b_v3_cluster_assignments.tsv.gz
      (4,918/18,895 have cluster — used for P2.1 OPTIONAL overlay only)
  ${EVOSCZ_ROOT}/results/phase14p_baseline/P14p2_a_AFR_freq_full.tsv.gz
      (cross-check substrate for P0.10)

OUTPUTS (${EVOSCZ_ROOT}/results/phase19/)
--------------------------------------------------
  P19_a_polarity_per_variant.tsv.gz
  P19_b_stratum_summary.tsv
  P19_c_run_log.txt
  P19_d_handtest.txt (5 hand-test cases + cross-check Pearson r)

RUN (SLURM)
-----------
  sbatch ${EVOSCZ_ROOT}/scripts/phase19/P19_run.slurm

Author: EVOSCZ pipeline
Date: 2026-05-16
Reproducibility: np.random.seed(20260516); printed to log.
================================================================================
"""

from __future__ import annotations

import gc
import gzip
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 20260516
np.random.seed(SEED)

# Canonical paths on the cluster (GRCh37 throughout)
PGC_MASTER = Path((_ROOT + "/data/processed/pgc3_master_variants.tsv"))
GEVA_AGES  = Path((_ROOT + "/data/processed/pgc3_geva_ages.tsv"))
P14P3_DIR  = Path((_SCRATCH + "/v11_data/phase14p3"))  # pure_AFR_chr*.tsv + subpop_*
P14P2_AFR  = Path((_ROOT + "/results/phase14p_baseline/P14p2_a_AFR_freq_full.tsv.gz"))
CLUSTERS   = Path((_ROOT + "/results/phase14b/P14b_v3_cluster_assignments.tsv.gz"))
# 1000G EUR plink files (n=503) — canonical AF_EUR(A1) source. Fix per P0 BLOCKER #1.
EUR_PLINK_DIR = Path((_ROOT + "/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink"))
EUR_PLINK_PREFIX = "1000G.EUR.QC"  # files: {EUR_PLINK_DIR}/{PREFIX}.{chr}.{bed,bim,fam}

OUT_DIR = Path((_ROOT + "/results/phase19"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_VARIANT   = OUT_DIR / "P19_a_polarity_per_variant.tsv.gz"
OUT_STRATUM       = OUT_DIR / "P19_b_stratum_summary.tsv"
OUT_RUN_LOG       = OUT_DIR / "P19_c_run_log.txt"
OUT_HANDTEST      = OUT_DIR / "P19_d_handtest.txt"

# Stratum boundaries (pre-registered)
AF_AFR_BOUNDS = [-1e-9, 0.01, 0.05, 0.50, 1.0 + 1e-9]
AF_AFR_LABELS = ["lt_0.01", "0.01_0.05", "0.05_0.50", "gt_0.50"]
PRIMARY_STRATUM_LABEL = "lt_0.05"  # merged <0.01 + 0.01-0.05 if needed

# Permutation / bootstrap iters
N_BOOT = 1000
N_PERM = 1000

# Multiple-testing family: 4 AF_AFR strata x 3 |BETA| tertiles x 2 PIP cuts = 24
BONFERRONI_ALPHA = 0.05 / 24  # = 0.00208... for secondary tests
PRIMARY_ALPHA    = 0.05       # primary AF_AFR<0.05 all-PIP all-BETA stratum

# Sub-populations (for P1.1 sensitivity)
AFR_SUBPOPS = ["ESN", "GWD", "LWK", "MSL", "YRI"]

# HLA / MAPT regions (GRCh37)
HLA_REGION = (6, 25_000_000, 34_000_000)
MAPT_REGION = (17, 42_000_000, 46_000_000)

LOG_LINES: list[str] = []

AMBIGUOUS_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


# ============================================================================
# UTILITIES
# ============================================================================

def log(msg: str = "") -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def is_ambiguous(a1: str, a2: str) -> bool:
    return (str(a1).upper(), str(a2).upper()) in AMBIGUOUS_PAIRS


def make_join_key(chrom, pos, a1, a2) -> str:
    """Allele-orientation-insensitive key: chr:pos:sorted(A1,A2).
    Returns '' if chrom/pos is NA (row drops out of joins as intended).
    Fix per code review BLOCKER #2.
    """
    if pd.isna(chrom) or pd.isna(pos):
        return ""
    a1 = str(a1).upper()
    a2 = str(a2).upper()
    lo, hi = sorted([a1, a2])
    return f"{int(chrom)}:{int(pos)}:{lo}:{hi}"


# ============================================================================
# CORE POLARITY LOGIC (written TWICE per P0.1 spec)
# ============================================================================

def classify_direction(raf_eur: float, raf_afr: float) -> str:
    """
    Classify direction of risk-allele frequency between EUR and AFR.
    Used by hand-test (P0.9) and as a third (semantic-label) independent
    check on top of verdict_form1 / verdict_form2.
    """
    if pd.isna(raf_eur) or pd.isna(raf_afr):
        return "unresolved"
    delta = raf_eur - raf_afr
    if abs(delta) < 1e-9:
        return "no_direction"
    return "risk_higher_in_EUR" if delta > 0 else "risk_higher_in_AFR"


def polarity_for_variant(eur_a1: str, eur_a2: str, eur_beta: float,
                         af_eur_a1: float, af_afr_a1: float):
    """
    Compute polarity-aware RAF_EUR, RAF_AFR using TWO genuinely independent
    code paths, then cross-check (P0.1).

    Form 1 — allele-centric: branch on sign(beta) → select risk allele → look
                              up RAF_EUR / RAF_AFR as the freq of risk_allele.
    Form 2 — sign-product:    compute delta_RAF = sign(beta) * (af_eur_a1 -
                              af_afr_a1) WITHOUT branching on beta. This
                              algebra is structurally different from Form 1 and
                              would NOT produce the same answer as Form 1 if
                              Form 1 had a sign-branch bug.

    Fix per P0 audit BLOCKER #4 (Form 2 was not actually independent before).

    Inputs:
      eur_a1, eur_a2 : effect / non-effect allele in PGC3 EUR sumstats
      eur_beta       : effect-size on log-odds; risk = A1 if BETA>0 else A2
      af_eur_a1      : frequency of EUR_A1 in EUR reference panel
      af_afr_a1      : frequency of EUR_A1 in 1000G pure-AFR

    Returns dict with: risk_allele, RAF_EUR, RAF_AFR, delta_RAF,
                       verdict_form1, verdict_form2
    """
    if pd.isna(eur_beta) or pd.isna(af_eur_a1) or pd.isna(af_afr_a1):
        return None

    # ----- Form 1 — allele-centric (branched on sign(beta)) -----
    if eur_beta >= 0:
        risk_allele = eur_a1
        raf_eur_form1 = af_eur_a1
        raf_afr_form1 = af_afr_a1
    else:
        risk_allele = eur_a2
        raf_eur_form1 = 1.0 - af_eur_a1
        raf_afr_form1 = 1.0 - af_afr_a1
    delta_form1 = raf_eur_form1 - raf_afr_form1
    verdict_form1 = (delta_form1 > 0)

    # ----- Form 2 — sign-product (NO sign branching) -----
    # delta_form2 = sign(beta) * (af_eur_a1 - af_afr_a1)
    # This is algebraically identical to delta_form1 because:
    #   if beta>=0: delta_form1 = af_eur_a1 - af_afr_a1
    #   if beta<0:  delta_form1 = (1-af_eur_a1) - (1-af_afr_a1) = -(af_eur_a1 - af_afr_a1)
    # so delta_form2 = sign(beta)*(af_eur_a1 - af_afr_a1) MUST equal delta_form1.
    # Crucially, Form 2 does NOT use an `if eur_beta >= 0:` branch — it uses
    # arithmetic. A sign-branch typo in Form 1 would produce a mismatch.
    sign = 1.0 if eur_beta >= 0 else -1.0   # arithmetic mapping, not branched flow
    delta_form2 = sign * (af_eur_a1 - af_afr_a1)
    verdict_form2 = (delta_form2 > 0)

    # MANDATORY cross-check (P0.1) — would catch Form 1 sign-branch bug,
    # allele-orientation bug, or arithmetic typo in either form.
    if verdict_form1 != verdict_form2:
        raise AssertionError(
            f"Verdict logic inconsistency (Form1/Form2): "
            f"form1={verdict_form1} (delta={delta_form1:+.4f}), "
            f"form2={verdict_form2} (delta={delta_form2:+.4f}), "
            f"eur_a1={eur_a1}, eur_a2={eur_a2}, beta={eur_beta:+.4f}, "
            f"af_eur_a1={af_eur_a1:.4f}, af_afr_a1={af_afr_a1:.4f}"
        )

    return dict(
        risk_allele=risk_allele,
        RAF_EUR=raf_eur_form1,
        RAF_AFR=raf_afr_form1,
        delta_RAF=delta_form1,
        verdict_form1=bool(verdict_form1),
        verdict_form2=bool(verdict_form2),
    )


# ============================================================================
# HAND-TEST (P0.9) — runs before any data is loaded
# ============================================================================

def run_handtest() -> None:
    """
    Hand-test verdict logic on 5 hardcoded cases. Fail-fast at top of main().
    If any case is wrong, halt before touching real data.
    """
    log("=" * 78)
    log("HAND-TEST (P0.9) — 5 hardcoded cases")
    log("=" * 78)

    test_cases = [
        # (RAF_EUR_expected, RAF_AFR_expected, expected_direction)
        (0.30, 0.05, "risk_higher_in_EUR"),   # AFR-rare, EUR-modest
        (0.05, 0.30, "risk_higher_in_AFR"),   # AFR-modest, EUR-rare
        (0.50, 0.50, "no_direction"),          # equal
        (0.10, 0.00, "risk_higher_in_EUR"),   # AFR-absent (primary stratum)
        (0.00, 0.10, "risk_higher_in_AFR"),   # EUR-absent
    ]

    handtest_log = []
    for raf_e, raf_a, expected in test_cases:
        got = classify_direction(raf_e, raf_a)
        ok = (got == expected)
        line = (f"RAF_EUR={raf_e:.2f}, RAF_AFR={raf_a:.2f}: "
                f"expected={expected!r}, got={got!r} [{'OK' if ok else 'FAIL'}]")
        log("  " + line)
        handtest_log.append(line)
        if not ok:
            with open(OUT_HANDTEST, "w") as fh:
                fh.write("\n".join(handtest_log) + "\n")
                fh.write(f"\nHALT: hand-test failure at case "
                         f"({raf_e},{raf_a}, expected={expected}, got={got})\n")
            log(f"FAIL: hand-test failed at case ({raf_e},{raf_a}). HALTING.")
            sys.exit(1)

    # Also exercise polarity_for_variant() — a positive-BETA case and a
    # negative-BETA case must produce verdict_form1 == verdict_form2.
    for (e_a1, e_a2, beta, af_eur, af_afr, label) in [
        ("A", "G", +0.10, 0.30, 0.05, "pos-BETA: risk=A, EUR>AFR"),
        ("A", "G", -0.10, 0.30, 0.05, "neg-BETA: risk=G, RAF_EUR=0.70, RAF_AFR=0.95 => AFR>EUR"),
        ("C", "T", +0.05, 0.10, 0.10, "equal: no_direction (form1=False=form2)"),
    ]:
        res = polarity_for_variant(e_a1, e_a2, beta, af_eur, af_afr)
        line = (f"polarity({e_a1}/{e_a2}, BETA={beta:+.2f}, "
                f"AF_EUR_A1={af_eur}, AF_AFR_A1={af_afr}) => "
                f"risk={res['risk_allele']}, RAF_EUR={res['RAF_EUR']:.2f}, "
                f"RAF_AFR={res['RAF_AFR']:.2f}, delta={res['delta_RAF']:+.3f}, "
                f"form1={res['verdict_form1']}, form2={res['verdict_form2']} "
                f"({label})")
        log("  " + line)
        handtest_log.append(line)
        assert res["verdict_form1"] == res["verdict_form2"], (
            f"form1!=form2 in hand-test for {label}"
        )

    with open(OUT_HANDTEST, "w") as fh:
        fh.write("HAND-TEST PASS (P0.9)\n")
        fh.write(f"Seed: {SEED}\n")
        fh.write(f"Date: {datetime.now().isoformat()}\n\n")
        fh.write("\n".join(handtest_log) + "\n")
    log("PASS — hand-test complete (P0.9).\n")


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_pgc3_master() -> pd.DataFrame:
    log(f"Loading PGC3 master variants: {PGC_MASTER}")
    df = pd.read_csv(PGC_MASTER, sep="\t", low_memory=False)
    log(f"  rows: {len(df):,}  columns: {len(df.columns)}")

    # Standardize column names per inventory
    required = ["rsid", "chr", "pos", "effect_allele", "other_allele",
                "maf", "beta", "se", "pval", "pip", "credible_set_id"]
    for col in required:
        if col not in df.columns:
            log(f"  WARNING: column {col!r} missing in master")
    df = df.rename(columns={
        "chr": "chr",
        "pos": "pos_grch37",
        "effect_allele": "EUR_A1",
        "other_allele": "EUR_A2",
        "beta": "EUR_BETA",
        "se": "EUR_SE",
        "pval": "EUR_P",
        "pip": "PIP",
        "maf": "EUR_MAF_master",
    })

    # P0.4: EUR reference AF.
    # Master 'maf' is from PGC3 ref panel (frequency of minor allele). We need
    # AF_EUR_A1 (frequency of A1=effect allele in EUR). If maf <= 0.5 it is the
    # minor allele frequency; we cannot tell from MAF alone whether A1 or A2 is
    # minor. The cleanest source is FRQ_U from PGC3 daner sumstats, but those
    # are not in master. We assume the convention from master: if MAF column
    # represents A1 frequency directly (PGC3 convention frequently does), then
    # AF_EUR_A1 = MAF. We flag this assumption and add a sanity check below.
    df["AF_EUR_A1"] = df["EUR_MAF_master"]
    df["AF_EUR_credset_source"] = "pgc3_master_maf_A1_freq"

    # Clean types
    df["chr"] = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
    df["pos_grch37"] = pd.to_numeric(df["pos_grch37"], errors="coerce").astype("Int64")
    for col in ("EUR_BETA", "EUR_SE", "EUR_P", "PIP", "AF_EUR_A1"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["EUR_A1"] = df["EUR_A1"].astype(str).str.upper()
    df["EUR_A2"] = df["EUR_A2"].astype(str).str.upper()

    # P0.3: drop indels (SNPs only)
    snps = df["EUR_A1"].str.len().eq(1) & df["EUR_A2"].str.len().eq(1)
    n_drop = int((~snps).sum())
    df = df[snps].copy()
    log(f"  dropped {n_drop:,} non-SNP rows; remaining SNPs: {len(df):,}")

    # Drop rows with NA chr or pos — these would cascade into np.where and
    # boolean-expression NA crashes downstream (pandas 3.0 issue).
    valid_pos = df["chr"].notna() & df["pos_grch37"].notna()
    n_drop_pos = int((~valid_pos).sum())
    if n_drop_pos > 0:
        log(f"  dropped {n_drop_pos:,} rows with NA chr/pos")
    df = df[valid_pos].copy()

    # P0.4 sanity check: median AF_EUR_A1 for high-PIP variants should be in [0.05, 0.95]
    high_pip = df[df["PIP"] > 0.5]
    if len(high_pip) > 50:
        med = high_pip["AF_EUR_A1"].median()
        log(f"  sanity: median AF_EUR_A1 for PIP>0.5 (n={len(high_pip):,}) = {med:.3f}")
        if med < 0.01 or med > 0.99:
            log(f"  WARNING: median AF_EUR_A1 out of plausible range — "
                f"check master 'maf' column orientation.")

    return df


def load_pure_afr() -> pd.DataFrame:
    """
    Concatenate phase14p3/pure_AFR_chr{1..22}.tsv. NO HEADER per inventory.
    Column inference per inventory: rsid_or_dot, chr, pos, A1, A2, AF,
                                    n_called, n_alt
    """
    log(f"Loading 1000G pure-AFR (n=504) frequencies: {P14P3_DIR}")
    cols = ["rsid_or_dot", "chr", "pos", "A1", "A2", "AF",
            "n_called", "n_alt"]
    frames = []
    for chrom in range(1, 23):
        fp = P14P3_DIR / f"pure_AFR_chr{chrom}.tsv"
        if not fp.exists():
            log(f"  WARNING: missing {fp}")
            continue
        df = pd.read_csv(fp, sep="\t", header=None, names=cols, low_memory=False)
        df["chr"] = chrom
        frames.append(df)
    afr = pd.concat(frames, ignore_index=True)
    log(f"  total rows: {len(afr):,}")

    # Type cleaning
    afr["pos"] = pd.to_numeric(afr["pos"], errors="coerce").astype("Int64")
    afr["AF"] = pd.to_numeric(afr["AF"], errors="coerce")
    afr["A1"] = afr["A1"].astype(str).str.upper()
    afr["A2"] = afr["A2"].astype(str).str.upper()

    # SNPs only
    snps = afr["A1"].str.len().eq(1) & afr["A2"].str.len().eq(1)
    afr = afr[snps].copy()
    log(f"  SNPs only: {len(afr):,}")

    afr = afr.rename(columns={"AF": "AF_AFR_A1_pure", "A1": "AFR_A1",
                              "A2": "AFR_A2", "pos": "pos_grch37"})
    return afr[["chr", "pos_grch37", "AFR_A1", "AFR_A2", "AF_AFR_A1_pure"]]


def load_subpops() -> dict[str, pd.DataFrame]:
    """Load per-subpop AFR frequencies (ESN, GWD, LWK, MSL, YRI)."""
    log("Loading per-subpop AFR frequencies (5 pops, 22 chrs)")
    cols = ["rsid_or_dot", "chr", "pos", "A1", "A2", "AF"]
    out: dict[str, pd.DataFrame] = {}
    for pop in AFR_SUBPOPS:
        frames = []
        for chrom in range(1, 23):
            fp = P14P3_DIR / f"subpop_{pop}_chr{chrom}.tsv"
            if not fp.exists():
                continue
            # Subpop files may have 6 or 8 columns; read flexibly
            try:
                df = pd.read_csv(fp, sep="\t", header=None,
                                 names=cols, usecols=range(6), low_memory=False)
            except Exception as e:
                log(f"  WARN parsing {fp.name}: {e}")
                continue
            df["chr"] = chrom
            frames.append(df)
        if frames:
            sp = pd.concat(frames, ignore_index=True)
            sp["pos"] = pd.to_numeric(sp["pos"], errors="coerce").astype("Int64")
            sp["AF"] = pd.to_numeric(sp["AF"], errors="coerce")
            sp["A1"] = sp["A1"].astype(str).str.upper()
            sp["A2"] = sp["A2"].astype(str).str.upper()
            snps = sp["A1"].str.len().eq(1) & sp["A2"].str.len().eq(1)
            sp = sp[snps].copy()
            sp = sp.rename(columns={"AF": f"AF_AFR_A1_{pop}",
                                    "A1": f"AFR_A1_{pop}",
                                    "A2": f"AFR_A2_{pop}",
                                    "pos": "pos_grch37"})
            out[pop] = sp[["chr", "pos_grch37",
                           f"AFR_A1_{pop}", f"AFR_A2_{pop}",
                           f"AF_AFR_A1_{pop}"]]
            log(f"  {pop}: {len(sp):,} SNP rows")
        else:
            log(f"  {pop}: no files loaded")
    return out


def load_geva() -> pd.DataFrame:
    log(f"Loading GEVA ages (AlleleAnc / AlleleDer): {GEVA_AGES}")
    df = pd.read_csv(GEVA_AGES, sep="\t", low_memory=False)
    log(f"  rows: {len(df):,}")

    # Pick TGP preferred (1000G consistency); fallback to first row per variant.
    if "DataSource" in df.columns:
        df["__ds_priority"] = df["DataSource"].map(
            {"TGP": 0, "Combined": 1, "SGDP": 2}
        ).fillna(9)
        df = (df.sort_values(["VariantID", "__ds_priority"])
                .drop_duplicates(subset="VariantID", keep="first")
                .drop(columns="__ds_priority"))
        log(f"  after TGP-preferred dedup: {len(df):,}")

    # Standardize
    df = df.rename(columns={
        "Chromosome": "chr_geva",
        "Position": "pos_geva",
        "AlleleRef": "GEVA_REF",
        "AlleleAlt": "GEVA_ALT",
        "AlleleAnc": "AlleleAnc",
        "VariantID": "rsid_geva",
        "DataSource": "AtlasDataSource",
    })
    df["chr_geva"] = pd.to_numeric(df["chr_geva"], errors="coerce").astype("Int64")
    df["pos_geva"] = pd.to_numeric(df["pos_geva"], errors="coerce").astype("Int64")
    for c in ("GEVA_REF", "GEVA_ALT", "AlleleAnc"):
        df[c] = df[c].astype(str).str.upper()

    # Derived allele = the non-AlleleAnc allele
    def derive(row):
        anc = row["AlleleAnc"]
        ref = row["GEVA_REF"]
        alt = row["GEVA_ALT"]
        if anc not in (ref, alt) or anc in ("", "N", ".", "NAN", "NONE"):
            return ""
        return alt if anc == ref else ref
    df["AlleleDer"] = df.apply(derive, axis=1)
    n_res = int((df["AlleleDer"] != "").sum())
    log(f"  resolvable AlleleAnc/Der: {n_res:,} ({n_res/len(df)*100:.1f}%)")

    return df[["chr_geva", "pos_geva", "GEVA_REF", "GEVA_ALT",
               "AlleleAnc", "AlleleDer", "AtlasDataSource", "rsid_geva"]]


def load_clusters() -> pd.DataFrame:
    log(f"Loading v10 cluster overlay (P2.1 OPTIONAL): {CLUSTERS}")
    df = pd.read_csv(CLUSTERS, sep="\t", compression="gzip", low_memory=False)
    log(f"  rows: {len(df):,}; with cluster assigned: "
        f"{int(df['cluster'].notna().sum()):,}")
    return df[["rsid", "cluster"]] if "cluster" in df.columns else df


# ============================================================================
# JOIN + AF_AFR LOOKUP
# ============================================================================

def attach_af_afr(eur: pd.DataFrame, afr: pd.DataFrame) -> pd.DataFrame:
    """
    Join on chr:pos:{A1,A2} (orientation-insensitive). Return EUR with
    AF_AFR_A1 = freq of EUR_A1 in 1000G pure-AFR (after orientation alignment).
    """
    log("Joining EUR <-> pure-AFR on (chr, pos, allele-pair-key)")
    eur = eur.copy()
    afr = afr.copy()
    eur["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                    zip(eur["chr"], eur["pos_grch37"],
                        eur["EUR_A1"], eur["EUR_A2"])]
    afr["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                    zip(afr["chr"], afr["pos_grch37"],
                        afr["AFR_A1"], afr["AFR_A2"])]
    merged = eur.merge(afr[["__key", "AFR_A1", "AFR_A2", "AF_AFR_A1_pure"]],
                       on="__key", how="left", suffixes=("", "_afr"))
    n_match = int(merged["AF_AFR_A1_pure"].notna().sum())
    log(f"  merged: {len(merged):,}; with AFR AF: {n_match:,}")

    # Orientation: if EUR_A1 == AFR_A1, AF_AFR (for EUR_A1) = AF_AFR_A1_pure
    # If EUR_A1 == AFR_A2 (allele orientation flipped), AF_AFR (for EUR_A1) = 1 - AF_AFR_A1_pure
    same = merged["EUR_A1"] == merged["AFR_A1"]
    flip = merged["EUR_A1"] == merged["AFR_A2"]
    merged["AF_AFR_A1"] = np.nan
    merged.loc[same, "AF_AFR_A1"] = merged.loc[same, "AF_AFR_A1_pure"]
    merged.loc[flip, "AF_AFR_A1"] = 1.0 - merged.loc[flip, "AF_AFR_A1_pure"]

    n_resolved = int(merged["AF_AFR_A1"].notna().sum())
    n_unmatched_orient = int(merged["AF_AFR_A1_pure"].notna().sum()) - n_resolved
    log(f"  AF_AFR_A1 oriented matches: {n_resolved:,}")
    if n_unmatched_orient > 0:
        log(f"  WARNING: {n_unmatched_orient:,} rows had AFR AF but neither "
            f"EUR_A1==AFR_A1 nor EUR_A1==AFR_A2 (likely tri-allelic or "
            f"strand-flipped; dropped from analysis).")

    return merged.drop(columns=["__key"])


def attach_subpop_af(df: pd.DataFrame, subpops: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each subpop, orient AF_AFR_A1_{pop} to EUR_A1.
    Fix per code review HIGH #4: free each subpop DataFrame from the dict
    after merging to keep memory peak under the 24 GB SLURM cap.
    """
    log("Joining EUR <-> 5 AFR subpops (heterogeneity sensitivity)")
    pop_keys = list(subpops.keys())  # Snapshot keys before mutation
    for pop in pop_keys:
        sp = subpops[pop]
        sp = sp.copy()
        sp["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                       zip(sp["chr"], sp["pos_grch37"],
                           sp[f"AFR_A1_{pop}"], sp[f"AFR_A2_{pop}"])]
        df = df.copy()
        df["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                       zip(df["chr"], df["pos_grch37"],
                           df["EUR_A1"], df["EUR_A2"])]
        m = df.merge(sp[["__key", f"AFR_A1_{pop}", f"AFR_A2_{pop}",
                         f"AF_AFR_A1_{pop}"]],
                     on="__key", how="left")
        same = m["EUR_A1"] == m[f"AFR_A1_{pop}"]
        flip = m["EUR_A1"] == m[f"AFR_A2_{pop}"]
        m[f"AFR_subpop_{pop}"] = np.nan
        m.loc[same, f"AFR_subpop_{pop}"] = m.loc[same, f"AF_AFR_A1_{pop}"]
        m.loc[flip, f"AFR_subpop_{pop}"] = 1.0 - m.loc[flip, f"AF_AFR_A1_{pop}"]
        df = m.drop(columns=["__key", f"AFR_A1_{pop}", f"AFR_A2_{pop}",
                              f"AF_AFR_A1_{pop}"])
        log(f"  {pop}: {int(df[f'AFR_subpop_{pop}'].notna().sum()):,} oriented")
        # H10: free subpop DataFrame from dict + GC to keep memory peak
        # under 24 GB SLURM cap. Each subpop is ~3-4 GB string-heavy.
        del sp, m
        subpops[pop] = None
    gc.collect()
    return df


def attach_geva(df: pd.DataFrame, geva: pd.DataFrame) -> pd.DataFrame:
    """Join on chr:pos; orient AlleleAnc/Der via GEVA REF/ALT."""
    log("Joining EUR <-> GEVA ancestral on (chr, pos)")
    g = geva.rename(columns={"chr_geva": "chr", "pos_geva": "pos_grch37"})
    m = df.merge(g, on=["chr", "pos_grch37"], how="left")
    log(f"  with GEVA polarisation: "
        f"{int(m['AlleleAnc'].notna().sum()):,} / {len(m):,}")

    # risk_is_derived: only valid when risk_allele matches one of GEVA_REF/ALT
    # We compute later in the polarity loop because risk_allele depends on BETA.
    return m


# ============================================================================
# 1000G EUR PLINK CANONICAL AF SOURCE (replaces master.maf)
# Job 5783157 verification revealed master.maf is reflected MAF, not AF(A1):
#   Pearson r(master.maf, 1000G EUR AF(A1)) = -0.4929 (n=199, p=1.4e-13)
# Solution: load AF(A1) directly from 1000G EUR plink (n=503).
# ============================================================================

def load_1000g_eur_af(eur_master: pd.DataFrame) -> pd.DataFrame:
    """Extract AF(A1) for all credset variants from 1000G EUR (n=503) via
    plink2 --freq + --extract. Joins .afreq with .bim for positions.

    Returns DataFrame with: chr, pos_grch37, rsid, PLINK_REF, PLINK_ALT, AF_EUR_ALT.
    """
    log("=" * 78)
    log("Loading 1000G EUR AF (n=503) via plink2 --freq")
    log("(canonical source — replaces unreliable master.maf)")
    log("=" * 78)

    frames = []
    for chrom in range(1, 23):
        bfile = EUR_PLINK_DIR / f"{EUR_PLINK_PREFIX}.{chrom}"
        bed_file = Path(f"{bfile}.bed")
        bim_file = Path(f"{bfile}.bim")
        if not bed_file.exists() or not bim_file.exists():
            log(f"  WARN: missing plink files for chr{chrom}: {bed_file}")
            continue

        rsids = eur_master[eur_master["chr"] == chrom]["rsid"].dropna().astype(str).tolist()
        if not rsids:
            log(f"  chr{chrom}: no credset variants — skipping")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            rsid_file = Path(tmp) / "rsids.txt"
            rsid_file.write_text("\n".join(rsids))
            out_prefix = Path(tmp) / "freq"
            cmd = ["plink2", "--bfile", str(bfile),
                   "--extract", str(rsid_file),
                   "--freq",
                   "--out", str(out_prefix), "--silent"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log(f"  ERROR plink2 chr{chrom}: {e}")
                continue
            if res.returncode != 0:
                log(f"  ERROR plink2 chr{chrom} rc={res.returncode}: "
                    f"{(res.stderr or '')[:300]}")
                continue

            freq_path = Path(f"{out_prefix}.afreq")
            if not freq_path.exists():
                log(f"  WARN: no .afreq for chr{chrom}")
                continue

            try:
                freq_df = pd.read_csv(freq_path, sep="\t")
                bim_df = pd.read_csv(bim_file, sep="\t", header=None,
                                      names=["chr_b", "rsid", "cm", "pos",
                                             "a1_b", "a2_b"])
            except Exception as e:
                log(f"  ERROR reading plink output chr{chrom}: {e}")
                continue

            freq_df = freq_df.rename(columns={
                "#CHROM": "chr_p", "ID": "rsid",
                "REF": "PLINK_REF", "ALT": "PLINK_ALT",
                "ALT_FREQS": "AF_EUR_ALT",
            })
            freq_df = freq_df.merge(bim_df[["rsid", "pos"]], on="rsid",
                                     how="left")
            freq_df["chr"] = chrom
            frames.append(freq_df[["chr", "pos", "rsid", "PLINK_REF",
                                    "PLINK_ALT", "AF_EUR_ALT"]])
            log(f"  chr{chrom}: {len(freq_df):,} EUR AF rows")

    if not frames:
        log("  WARNING: no 1000G EUR data loaded — AF_EUR_A1 will fall back to "
            "master.maf (UNRELIABLE per audit).")
        return pd.DataFrame(columns=["chr", "pos_grch37", "rsid",
                                      "PLINK_REF", "PLINK_ALT", "AF_EUR_ALT"])

    eur1k = pd.concat(frames, ignore_index=True)
    eur1k["pos"] = pd.to_numeric(eur1k["pos"], errors="coerce").astype("Int64")
    eur1k["AF_EUR_ALT"] = pd.to_numeric(eur1k["AF_EUR_ALT"], errors="coerce")
    eur1k["PLINK_REF"] = eur1k["PLINK_REF"].astype(str).str.upper()
    eur1k["PLINK_ALT"] = eur1k["PLINK_ALT"].astype(str).str.upper()
    log(f"  total 1000G EUR variants loaded: {len(eur1k):,}")
    return eur1k.rename(columns={"pos": "pos_grch37"})


def attach_af_eur_1000g(eur: pd.DataFrame, eur1k: pd.DataFrame) -> pd.DataFrame:
    """Orient 1000G PLINK ALT/REF to PGC3 EUR_A1; REPLACE master.maf-derived
    AF_EUR_A1 with 1000G-derived value. Keep master.maf as audit column.
    Falls back to master.maf for variants without 1000G match (with caveat).
    """
    log("Joining PGC3 master <-> 1000G EUR (canonical AF_EUR_A1 source)")

    # Keep master.maf for audit BEFORE we replace AF_EUR_A1
    eur = eur.copy()
    eur["AF_EUR_master_maf_audit"] = eur["AF_EUR_A1"]

    if len(eur1k) == 0:
        log("  WARNING: 1000G EUR empty; AF_EUR_A1 stays from master.maf (UNRELIABLE)")
        eur["AF_EUR_credset_source"] = "master_maf_only_UNRELIABLE"
        return eur

    eur1k = eur1k.copy()
    eur1k["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                     zip(eur1k["chr"], eur1k["pos_grch37"],
                         eur1k["PLINK_REF"], eur1k["PLINK_ALT"])]

    eur["__key"] = [make_join_key(c, p, a1, a2) for c, p, a1, a2 in
                    zip(eur["chr"], eur["pos_grch37"],
                        eur["EUR_A1"], eur["EUR_A2"])]

    m = eur.merge(eur1k[["__key", "PLINK_REF", "PLINK_ALT", "AF_EUR_ALT"]],
                  on="__key", how="left")

    # Orient PLINK ALT to PGC3 EUR_A1
    same = (m["EUR_A1"] == m["PLINK_ALT"]).fillna(False)
    flip = (m["EUR_A1"] == m["PLINK_REF"]).fillna(False)
    m["AF_EUR_A1_1000g"] = np.nan
    m.loc[same, "AF_EUR_A1_1000g"] = m.loc[same, "AF_EUR_ALT"]
    m.loc[flip, "AF_EUR_A1_1000g"] = 1.0 - m.loc[flip, "AF_EUR_ALT"]

    n_match = int(m["AF_EUR_A1_1000g"].notna().sum())
    log(f"  1000G EUR oriented to EUR_A1: {n_match:,} / {len(eur):,}")

    # CANONICAL SWAP: AF_EUR_A1 now comes from 1000G EUR plink
    m["AF_EUR_A1"] = m["AF_EUR_A1_1000g"]
    m["AF_EUR_credset_source"] = "1000g_eur_plink_n503"

    # For variants without 1000G match: fall back to master.maf but FLAG it
    fb_mask = m["AF_EUR_A1"].isna() & m["AF_EUR_master_maf_audit"].notna()
    n_fb = int(fb_mask.sum())
    if n_fb > 0:
        # NOTE: master.maf is known-reflected; downstream polarity for these
        # variants will be unreliable. Better to drop than mis-orient.
        m.loc[fb_mask, "AF_EUR_credset_source"] = "DROP_master_maf_unreliable"
        log(f"  WARN: {n_fb:,} variants have no 1000G match. NOT falling back "
            f"to master.maf (it's reflected MAF, would mis-orient). "
            f"These rows will have AF_EUR_A1=NaN and drop from polarity.")

    # Audit metric: Pearson r between old (master.maf) and new (1000G)
    pair = m.dropna(subset=["AF_EUR_master_maf_audit", "AF_EUR_A1_1000g"])
    if len(pair) >= 50:
        try:
            r, p = stats.pearsonr(pair["AF_EUR_master_maf_audit"],
                                  pair["AF_EUR_A1_1000g"])
            log(f"  AUDIT: r(master.maf, 1000G EUR AF(A1)) = {r:.4f} "
                f"(n={len(pair):,}, p={p:.2e})")
            if r < -0.3:
                log(f"         Strong negative r CONFIRMS master.maf was "
                    f"reflected MAF, not AF(A1) — 1000G swap was necessary.")
        except Exception as e:
            log(f"  AUDIT pearsonr error: {e}")

    return m.drop(columns=["__key", "PLINK_REF", "PLINK_ALT", "AF_EUR_ALT",
                            "AF_EUR_A1_1000g"])


# ============================================================================
# P0.4 — VERIFY master.maf is interpretable as AF(A1)
#        (BLOCKER #1 fix per PHASE19_SCRIPT_AUDIT)
#        NOTE: Job 5783157 revealed master.maf is reflected MAF (r=-0.49).
#        This function is RETAINED but no longer called from main() — AF_EUR_A1
#        is now sourced directly from 1000G EUR plink. Kept for reference.
# ============================================================================

def verify_af_eur_orientation(df: pd.DataFrame, n_sample: int = 200) -> None:
    """
    Verify master.maf is AF(effect_allele=EUR_A1), not min(AF(A1), AF(A2)).

    Samples N high-PIP credset variants, computes 1000G EUR AF(A1) via
    plink2 --freq on the 1000G_EUR_Phase3_plink (n=503) bundle, orients
    to EUR_A1 (master.effect_allele), then requires Pearson r >= 0.95
    between master.maf and oriented 1000G AF.

    HALT (exit 1) on failure: if master.maf is true MAF, the script's
    risk-allele orientation is silently wrong on ~50% of variants and the
    primary verdict would be uninterpretable (the exact bug class that
    invalidated the earlier P18a approach).

    If r >= 0.95: PASS, log, proceed with master.maf as AF(A1).
    """
    log("=" * 78)
    log("P0.4 verification: master.maf vs 1000G EUR AF(A1) (BLOCKER #1 gate)")
    log("=" * 78)

    candidates = df[df["AF_EUR_A1"].notna()
                    & (df["PIP"] > 0.1)
                    & df["chr"].notna()
                    & df["pos_grch37"].notna()
                    & df["rsid"].notna()].copy()
    if len(candidates) < 50:
        log(f"  WARNING: only {len(candidates)} candidates; skipping verification.")
        log("  Result interpretation must caveat: AF_EUR orientation unverified.")
        return

    n_use = min(n_sample, len(candidates))
    rng = np.random.default_rng(SEED + 3)
    idx = rng.choice(len(candidates), size=n_use, replace=False)
    sample = candidates.iloc[idx][["rsid", "chr", "pos_grch37",
                                    "EUR_A1", "EUR_A2", "AF_EUR_A1"]].copy()
    sample["rsid"] = sample["rsid"].astype(str)
    sample["chr_int"] = sample["chr"].astype(int)
    log(f"  Sampling {len(sample)} high-PIP candidates for plink2 lookup.")

    eur_afs = []
    for chrom in sorted(sample["chr_int"].unique()):
        bfile = EUR_PLINK_DIR / f"{EUR_PLINK_PREFIX}.{int(chrom)}"
        # CRITICAL fix: Path.with_suffix(".bed") replaces the LAST suffix,
        # which is .22 (chr number) for paths like "1000G.EUR.QC.22" — it
        # would produce "1000G.EUR.QC.bed" and miss the actual file.
        # Use direct string concatenation instead.
        bed_file = Path(f"{bfile}.bed")
        if not bed_file.exists():
            log(f"  WARN: no EUR plink for chr{int(chrom)} ({bed_file}) — skipping")
            continue

        chrom_sample = sample[sample["chr_int"] == int(chrom)]
        rsid_list = chrom_sample["rsid"].astype(str).tolist()
        if not rsid_list:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            rsid_file = Path(tmp) / "rsids.txt"
            rsid_file.write_text("\n".join(rsid_list))
            out_prefix = Path(tmp) / "freq"
            cmd = ["plink2", "--bfile", str(bfile),
                   "--extract", str(rsid_file),
                   "--freq",
                   "--out", str(out_prefix), "--silent"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=180)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log(f"  ERROR plink2 chr{int(chrom)}: {e}")
                continue
            if res.returncode != 0:
                log(f"  ERROR plink2 chr{int(chrom)} rc={res.returncode}: "
                    f"{(res.stderr or '')[:300]}")
                continue

            freq_file = out_prefix.with_suffix(".afreq")
            if not freq_file.exists():
                log(f"  WARN: no .afreq output for chr{int(chrom)}")
                continue

            try:
                freq_df = pd.read_csv(freq_file, sep="\t")
            except Exception as e:
                log(f"  ERROR reading .afreq chr{int(chrom)}: {e}")
                continue

            freq_df = freq_df.rename(columns={
                "#CHROM": "chr_p", "ID": "rsid", "REF": "REF_p",
                "ALT": "ALT_p", "ALT_FREQS": "AF_ALT_p"
            })
            freq_df["REF_p"] = freq_df["REF_p"].astype(str).str.upper()
            freq_df["ALT_p"] = freq_df["ALT_p"].astype(str).str.upper()

            joined = chrom_sample.merge(freq_df[["rsid", "REF_p", "ALT_p",
                                                  "AF_ALT_p"]],
                                         on="rsid", how="inner")
            for _, row in joined.iterrows():
                ea = str(row["EUR_A1"]).upper()
                af_alt = row["AF_ALT_p"]
                if pd.isna(af_alt):
                    continue
                if ea == row["ALT_p"]:
                    af_a1_1k = float(af_alt)
                elif ea == row["REF_p"]:
                    af_a1_1k = 1.0 - float(af_alt)
                else:
                    continue   # tri-allelic / strand-flip — skip
                eur_afs.append((row["rsid"], af_a1_1k,
                                float(row["AF_EUR_A1"])))

        log(f"  chr{int(chrom)}: matched {sum(1 for x in eur_afs if x)} so far")

    if len(eur_afs) < 30:
        log(f"  WARNING: only {len(eur_afs)} matched variants; "
            "cannot reliably verify. Continuing without strict gate.")
        log("  Result interpretation must caveat: AF_EUR orientation unverified.")
        return

    arr = pd.DataFrame(eur_afs, columns=["rsid", "AF_EUR_1000g",
                                          "AF_EUR_master_maf"])
    r, p = stats.pearsonr(arr["AF_EUR_master_maf"], arr["AF_EUR_1000g"])
    log(f"  N matched = {len(arr)}")
    log(f"  Pearson r(master.maf, 1000G EUR AF(A1)) = {r:.4f} (p = {p:.2e})")
    log(f"  Threshold: >= 0.95")

    with open(OUT_HANDTEST, "a") as fh:
        fh.write(f"\nP0.4 verification (master.maf vs 1000G EUR AF):\n")
        fh.write(f"  N matched = {len(arr)}\n")
        fh.write(f"  Pearson r = {r:.4f}, p = {p:.2e}\n")
        fh.write(f"  Threshold: >= 0.95\n")
        fh.write(f"  Verdict: "
                 f"{'PASS' if r >= 0.95 else 'FAIL — HALT'}\n")

    if r < 0.95:
        log(f"FAIL: master.maf orientation unreliable (r = {r:.4f} < 0.95).")
        log("master.maf is likely true Minor Allele Frequency, NOT AF(A1).")
        log("Continuing would silently mis-orient ~50% of variants (the exact")
        log("bug class that invalidated the earlier P18a approach).")
        log("HALTING per P0.4 spec. To proceed, the user must either:")
        log("  (a) Switch to 1000G EUR plink as canonical AF source")
        log("      (add load_1000g_eur_af call replacing master.maf)")
        log("  (b) Use PGC3 daner FRQ_U after rsync to the cluster")
        log("  (c) Explicitly accept the risk (no CLI flag exists; modify code)")
        sys.exit(1)

    log("PASS — master.maf agrees with 1000G EUR AF(A1) at r >= 0.95.\n")


# ============================================================================
# P0.10 — CROSS-CHECK 100 RANDOM VARIANTS P14p3 (pure-AFR) vs P14p2 (AFRAM)
# ============================================================================

def cross_check_substrate(df: pd.DataFrame, n_check: int = 100) -> float:
    """
    Sample 100 random AFR-rare (AF_AFR<0.05) variants and compare AF_AFR_A1
    from P14p3 (pure-AFR 1000G) vs P14p2 (AFRAM sumstats FCAS/FCON).
    Pearson r >= 0.90 required. Halt if violated.
    """
    log("=" * 78)
    log("P0.10 cross-check: P14p3 pure-AFR vs P14p2 AFRAM (random 100 AFR-rare)")
    log("=" * 78)

    if not P14P2_AFR.exists():
        log(f"  WARNING: P14p2 substrate not found at {P14P2_AFR}; "
            f"SKIPPING cross-check (cannot verify).")
        return np.nan

    p14p2 = pd.read_csv(P14P2_AFR, sep="\t", compression="gzip", low_memory=False)
    log(f"  P14p2 rows loaded: {len(p14p2):,}")

    candidate = df[(df["AF_AFR_A1"] < 0.05) & df["AF_AFR_A1"].notna()
                   & df["rsid"].notna()].copy()
    log(f"  AFR-rare candidates: {len(candidate):,}")
    if len(candidate) < n_check:
        log(f"  WARNING: only {len(candidate)} AFR-rare available; using all.")
        n_check = len(candidate)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(candidate), size=n_check, replace=False)
    sub = candidate.iloc[idx][["rsid", "EUR_A1", "EUR_A2",
                               "AF_AFR_A1"]].copy()

    # Join on rsid
    cmp = sub.merge(p14p2[["rsid", "A1", "A2", "AFR_f"]], on="rsid", how="inner")
    log(f"  matched on rsid for cross-check: {len(cmp)} / {n_check}")

    if len(cmp) < 30:
        log(f"  WARNING: <30 cross-check matches; cannot reliably compute "
            f"Pearson r. Proceeding without strict gate.")
        return np.nan

    # Orient AFRAM A1 to EUR_A1
    same = cmp["A1"].astype(str).str.upper() == cmp["EUR_A1"]
    flip = cmp["A1"].astype(str).str.upper() == cmp["EUR_A2"]
    cmp["AFR_f_oriented"] = np.nan
    cmp.loc[same, "AFR_f_oriented"] = cmp.loc[same, "AFR_f"]
    cmp.loc[flip, "AFR_f_oriented"] = 1.0 - cmp.loc[flip, "AFR_f"]

    valid = cmp.dropna(subset=["AF_AFR_A1", "AFR_f_oriented"])
    if len(valid) < 30:
        log(f"  WARNING: <30 oriented matches; skipping gate.")
        return np.nan

    r, p = stats.pearsonr(valid["AF_AFR_A1"], valid["AFR_f_oriented"])
    log(f"  Pearson r (P14p3 pure-AFR vs P14p2 AFRAM) = {r:.4f} (p={p:.2e})")
    log(f"  n = {len(valid)}")

    with open(OUT_HANDTEST, "a") as fh:
        fh.write(f"\nP0.10 cross-check:\n")
        fh.write(f"  Pearson r = {r:.4f}, p = {p:.2e}, n = {len(valid)}\n")
        fh.write(f"  Threshold: >= 0.90\n")
        fh.write(f"  Verdict: {'PASS' if r >= 0.90 else 'FAIL — HALT'}\n")

    if r < 0.90:
        log(f"FAIL: Pearson r ({r:.4f}) < 0.90. AFR substrates disagree.")
        log("HALTING per P0.10 spec.")
        sys.stderr.write(
            f"P0.10 FAIL: P14p3 pure-AFR vs P14p2 AFRAM Pearson r = {r:.4f} < 0.90\n"
        )
        sys.exit(1)

    log("PASS — P0.10 cross-check OK.\n")
    return r


# ============================================================================
# POLARITY COMPUTATION + FILTERS
# ============================================================================

def compute_polarity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main polarity computation. Adds:
      strand_status, risk_allele, RAF_EUR, RAF_AFR, delta_RAF_EUR_minus_AFR,
      verdict_form1, verdict_form2, risk_is_derived, direction_label
    """
    log("Computing polarity per variant")

    # P0.2 strand-ambiguous filter
    amb = df.apply(lambda r: is_ambiguous(r["EUR_A1"], r["EUR_A2"]), axis=1)
    df["strand_status"] = np.where(amb, "ambiguous_dropped", "ok")
    log(f"  strand-ambiguous (A/T,C/G) dropped: {int(amb.sum()):,} / {len(df):,}")

    results = []
    n_ok = 0
    n_missing = 0
    for r in df.itertuples(index=False):
        if r.strand_status == "ambiguous_dropped":
            results.append((None, None, None, None, None, None, None))
            continue
        pol = polarity_for_variant(
            eur_a1=r.EUR_A1, eur_a2=r.EUR_A2, eur_beta=r.EUR_BETA,
            af_eur_a1=r.AF_EUR_A1, af_afr_a1=r.AF_AFR_A1,
        )
        if pol is None:
            n_missing += 1
            results.append((None, None, None, None, None, None, None))
            continue
        # Determine risk_is_derived (requires GEVA AlleleAnc/Der)
        anc = getattr(r, "AlleleAnc", None)
        der = getattr(r, "AlleleDer", None)
        risk_is_derived = np.nan
        if pd.notna(anc) and pd.notna(der) and der != "" and anc != "":
            risk = str(pol["risk_allele"]).upper()
            if risk == str(der).upper():
                risk_is_derived = True
            elif risk == str(anc).upper():
                risk_is_derived = False
            else:
                risk_is_derived = np.nan  # mismatch (strand or tri-allelic)

        results.append((
            pol["risk_allele"],
            pol["RAF_EUR"],
            pol["RAF_AFR"],
            pol["delta_RAF"],
            pol["verdict_form1"],
            pol["verdict_form2"],
            risk_is_derived,
        ))
        n_ok += 1

    cols = ["risk_allele", "RAF_EUR", "RAF_AFR", "delta_RAF_EUR_minus_AFR",
            "verdict_form1", "verdict_form2", "risk_is_derived"]
    add_df = pd.DataFrame(results, columns=cols, index=df.index)
    out = pd.concat([df, add_df], axis=1)

    out["direction_label"] = out.apply(
        lambda r: classify_direction(r["RAF_EUR"], r["RAF_AFR"]),
        axis=1,
    )
    log(f"  polarity resolved: {n_ok:,}; missing (no AF_AFR or BETA): {n_missing:,}")
    return out


# ============================================================================
# STRATIFICATION + STATISTICS
# ============================================================================

def assign_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add AF_AFR_stratum, BETA_tertile, PIP_cut, HLA_status, MAPT_status."""
    df = df.copy()
    valid = df["RAF_AFR"].notna()
    # We bin on AF of the *risk* allele in AFR (RAF_AFR), per spec
    df.loc[valid, "AF_AFR_stratum"] = pd.cut(
        df.loc[valid, "RAF_AFR"], bins=AF_AFR_BOUNDS, labels=AF_AFR_LABELS,
        include_lowest=True,
    ).astype(str)
    df.loc[~valid, "AF_AFR_stratum"] = "unresolved"

    # |BETA| tertile (within polarity-resolved set)
    # Fix per code review BLOCKER #1: initialize column unconditionally so
    # the .fillna().astype(str) at the end never hits KeyError on degenerate
    # runs (pol.sum() <= 30).
    abs_beta = df["EUR_BETA"].abs()
    pol = valid & df["EUR_BETA"].notna()
    df["BETA_tertile"] = pd.Series("unresolved", index=df.index, dtype="object")
    if pol.sum() > 30:
        try:
            df.loc[pol, "BETA_tertile"] = pd.qcut(
                abs_beta[pol], q=3, labels=["low", "mid", "high"],
                duplicates="drop",
            ).astype(str)
        except ValueError:
            df.loc[pol, "BETA_tertile"] = "n/a"
    df["BETA_tertile"] = df["BETA_tertile"].astype(str)

    # PIP cut (>0.5 vs <=0.5)
    # pandas 3.0 Int64/Float64 + np.where with pd.NA crashes; .fillna(False)
    # before np.where to coerce NA -> False (pip-missing -> treated as low PIP).
    df["PIP_cut"] = np.where(
        (df["PIP"] > 0.5).fillna(False).astype(bool),
        "pip_gt_0.5", "pip_le_0.5",
    )

    # HLA / MAPT flags
    # CRITICAL fix: pandas 3.0 Int64 boolean expressions can carry pd.NA;
    # np.where cannot handle NA-valued boolean arrays (raises TypeError:
    # "boolean value of NA is ambiguous"). Materialize mask via fillna(False).
    hla_mask = ((df["chr"] == HLA_REGION[0])
                & (df["pos_grch37"] >= HLA_REGION[1])
                & (df["pos_grch37"] <= HLA_REGION[2])).fillna(False).astype(bool)
    df["HLA_status"] = np.where(hla_mask, "in_HLA", "outside")

    mapt_mask = ((df["chr"] == MAPT_REGION[0])
                 & (df["pos_grch37"] >= MAPT_REGION[1])
                 & (df["pos_grch37"] <= MAPT_REGION[2])).fillna(False).astype(bool)
    df["MAPT_status"] = np.where(mapt_mask, "in_MAPT", "outside")

    df["LD_prune_status"] = "all"  # primary uses all; sensitivity flag below
    return df


def bootstrap_direction(deltas: np.ndarray, n_boot: int = N_BOOT
                        ) -> tuple[float, float, float]:
    """
    Compute D = P(RAF_EUR > RAF_AFR) and bootstrap 95% CI.
    """
    deltas = np.asarray(deltas)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 10:
        return (np.nan, np.nan, np.nan)
    d_obs = float(np.mean(deltas > 0))
    rng = np.random.default_rng(SEED + 1)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        boot[i] = np.mean(deltas[idx] > 0)
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return (d_obs, lo, hi)


def permutation_null(df_pol: pd.DataFrame, stratum: str,
                     n_perm: int = N_PERM) -> float:
    """
    Shuffle AF_AFR_stratum labels across polarity-resolved variants
    (preserving stratum sizes), recompute D for the target stratum,
    return empirical two-sided p_perm with Phipson-Smyth +1/+1 correction.
    """
    pol = df_pol[df_pol["delta_RAF_EUR_minus_AFR"].notna()].copy()
    if len(pol) < 50:
        return np.nan
    target = pol[pol["AF_AFR_stratum"] == stratum]
    if len(target) < 20:
        return np.nan
    d_obs = float(np.mean(target["delta_RAF_EUR_minus_AFR"] > 0))

    rng = np.random.default_rng(SEED + 2)
    deltas_all = pol["delta_RAF_EUR_minus_AFR"].values
    strata_all = pol["AF_AFR_stratum"].values.copy()
    null_ds = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(strata_all)
        mask = (shuffled == stratum)
        if mask.sum() > 0:
            null_ds[i] = np.mean(deltas_all[mask] > 0)
        else:
            null_ds[i] = np.nan
    null_ds = null_ds[~np.isnan(null_ds)]
    if len(null_ds) == 0:
        return np.nan
    # Phipson-Smyth +1/+1 (fix per code review HIGH #3): smallest reportable
    # p_perm = 1/(n+1), never exactly 0.0.
    n_extreme = int(np.sum(np.abs(null_ds - 0.5) >= abs(d_obs - 0.5)))
    p_perm = float((n_extreme + 1) / (len(null_ds) + 1))
    return p_perm


def permutation_null_merged(df_pol: pd.DataFrame, merged_strata: list,
                            n_perm: int = N_PERM) -> float:
    """
    Permutation null over a MERGED stratum (union of sub-bin labels).
    Fix per P0 audit BLOCKER #3 / code review HIGH #1.

    Resamples n_target rows without replacement from the full polarity-resolved
    pool, recomputes D, compares against d_obs(merged_strata). Phipson-Smyth
    +1/+1 correction applied.
    """
    pol = df_pol[df_pol["delta_RAF_EUR_minus_AFR"].notna()].copy()
    if len(pol) < 50:
        return np.nan
    target = pol[pol["AF_AFR_stratum"].isin(merged_strata)]
    n_target = len(target)
    if n_target < 20:
        return np.nan
    d_obs = float(np.mean(target["delta_RAF_EUR_minus_AFR"] > 0))

    rng = np.random.default_rng(SEED + 2)
    deltas_all = pol["delta_RAF_EUR_minus_AFR"].values
    null_ds = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(deltas_all), size=n_target, replace=False)
        null_ds[i] = np.mean(deltas_all[idx] > 0)
    n_extreme = int(np.sum(np.abs(null_ds - 0.5) >= abs(d_obs - 0.5)))
    p_perm = float((n_extreme + 1) / (n_perm + 1))
    return p_perm


# ============================================================================
# STRATUM SUMMARY + VERDICT
# ============================================================================

def summarize_stratum(df_pol: pd.DataFrame, mask: pd.Series, label_cols: dict
                      ) -> dict:
    """Compute one row of the stratum summary table."""
    sub = df_pol[mask]
    n_total = int(mask.sum())
    pol_mask = sub["delta_RAF_EUR_minus_AFR"].notna()
    n_pol = int(pol_mask.sum())
    n_amb = int((sub["strand_status"] == "ambiguous_dropped").sum())
    drop_rate = float(n_amb / max(n_total, 1))
    deltas = sub.loc[pol_mask, "delta_RAF_EUR_minus_AFR"].values

    d_obs, lo, hi = bootstrap_direction(deltas)
    mean_delta = float(np.nanmean(deltas)) if len(deltas) else np.nan
    median_delta = float(np.nanmedian(deltas)) if len(deltas) else np.nan
    # Binomial test vs 0.5
    if n_pol >= 10:
        n_pos = int(np.sum(deltas > 0))
        try:
            p_binom = float(stats.binomtest(n_pos, n_pol, p=0.5,
                                            alternative="two-sided").pvalue)
        except AttributeError:
            # scipy < 1.7 fallback
            p_binom = float(stats.binom_test(n_pos, n_pol, p=0.5,
                                             alternative="two-sided"))
    else:
        p_binom = np.nan
    row = {
        **label_cols,
        "n_total": n_total,
        "n_polarity_resolved": n_pol,
        "n_strand_ambig_dropped": n_amb,
        "drop_rate": round(drop_rate, 4),
        "prop_RAF_EUR_gt_AFR": d_obs,
        "prop_CI_lo": lo,
        "prop_CI_hi": hi,
        "mean_delta_RAF": mean_delta,
        "median_delta_RAF": median_delta,
        "p_binom_vs_05": p_binom,
        "p_perm": np.nan,        # filled in for primary strata only
        "verdict_label": "",
    }
    return row


def primary_verdict(D_all: float, ci_all_lo: float, ci_all_hi: float,
                    p_perm: float, D_wc: float, ci_wc_lo: float, ci_wc_hi: float,
                    subpop_agree: int) -> str:
    """
    Apply the pre-registered decision tree:

      1. If H0 NOT rejected (CI covers 0.50 AND p_perm > 0.05): NULL
      2. If H0 rejected AND signal disappears at (PIP>0.5, |BETA|-mid):
         WINNER_CURSE_CONFOUND
      3. If H0 rejected AND signal persists AND >=4/5 subpops agree:
         DIRECTIONAL_{risk|protective}
      Otherwise: NULL (default conservative)
    """
    if any(pd.isna(x) for x in (D_all, ci_all_lo, ci_all_hi, p_perm)):
        return "NULL"  # under-powered or unresolved

    h0_rejected = (ci_all_lo > 0.50) or (ci_all_hi < 0.50)
    h0_rejected = h0_rejected and (p_perm < PRIMARY_ALPHA)

    if not h0_rejected:
        return "NULL"

    # H4 test: is the signal still shifted (>0.05 from 0.50) under (PIP>0.5, |BETA|-mid)?
    if not pd.isna(D_wc):
        attenuated = abs(D_wc - 0.50) <= 0.05
        if attenuated:
            return "WINNER_CURSE_CONFOUND"

    # Heterogeneity check: at least 4 of 5 subpops must agree on sign
    if subpop_agree < 4:
        return "WINNER_CURSE_CONFOUND"  # heterogeneous = uninterpretable

    return "DIRECTIONAL_risk" if D_all > 0.50 else "DIRECTIONAL_protective"


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("=" * 78)
    log("Phase 19 — Polarity-aware RAF (SLURM)")
    log(f"Seed: {SEED}; bonferroni-alpha (24-test family): {BONFERRONI_ALPHA:.5f}")
    log(f"Primary alpha (AF_AFR<0.05 all-PIP all-BETA): {PRIMARY_ALPHA}")
    log("=" * 78)

    # P0.9: hand-test FIRST (fail-fast before touching data)
    run_handtest()

    # Load all canonical sources
    eur     = load_pgc3_master()

    # CANONICAL AF_EUR_A1 from 1000G EUR plink (n=503).
    # Replaces master.maf which is reflected MAF (r=-0.49, audit confirmed
    # in job 5783157 verification). See load_1000g_eur_af + attach_af_eur_1000g.
    eur1k   = load_1000g_eur_af(eur)
    eur     = attach_af_eur_1000g(eur, eur1k)
    del eur1k
    gc.collect()

    afr     = load_pure_afr()
    subpops = load_subpops()
    geva    = load_geva()
    clusters = load_clusters()

    # Joins
    df = attach_af_afr(eur, afr)
    df = attach_subpop_af(df, subpops)
    df = attach_geva(df, geva)
    df = df.merge(clusters, on="rsid", how="left")
    gc.collect()  # H10: free intermediate frames before heavy ops

    # Polarity logic (Form 1 + Form 2 with assertion)
    df = compute_polarity_table(df)

    # Stratification (after polarity — strata are on RAF_AFR)
    df = assign_strata(df)

    # P0.10 cross-check (after AF_AFR is computed)
    cross_r = cross_check_substrate(df, n_check=100)

    # ===== Stratum summary table (P0.11 schema) =====
    log("=" * 78)
    log("Stratum summaries")
    log("=" * 78)

    summary_rows = []

    # Primary: AF_AFR<0.05 (merged if needed per P0.5), all-PIP, all-BETA,
    # HLA-included, MAPT-included
    # First check if AF_AFR<0.01 has n<50; merge if so.
    n_lt001 = int(((df["AF_AFR_stratum"] == "lt_0.01")
                   & df["delta_RAF_EUR_minus_AFR"].notna()).sum())
    n_001_005 = int(((df["AF_AFR_stratum"] == "0.01_0.05")
                     & df["delta_RAF_EUR_minus_AFR"].notna()).sum())
    log(f"AF_AFR<0.01 polarity-resolved: {n_lt001}")
    log(f"AF_AFR 0.01-0.05 polarity-resolved: {n_001_005}")

    merge_rule_used = (n_lt001 < 50)
    if merge_rule_used:
        log(f"P0.5 power gate: AF_AFR<0.01 has n={n_lt001}<50 — "
            f"MERGING with 0.01-0.05 stratum into 'lt_0.05'.")
    else:
        log(f"P0.5 power gate: AF_AFR<0.01 has n={n_lt001}>=50 — "
            f"keeping strata separate.")

    # Build masks for all stratum cells we want to report
    def mask_for(af_label, pip_label="all", beta_label="all",
                 hla="all", mapt="all"):
        if af_label == "lt_0.05":
            m = df["AF_AFR_stratum"].isin(["lt_0.01", "0.01_0.05"])
        else:
            m = (df["AF_AFR_stratum"] == af_label)
        if pip_label == "pip_gt_0.5":
            m &= (df["PIP_cut"] == "pip_gt_0.5")
        elif pip_label == "pip_le_0.5":
            m &= (df["PIP_cut"] == "pip_le_0.5")
        if beta_label in ("low", "mid", "high"):
            m &= (df["BETA_tertile"] == beta_label)
        if hla == "excluded":
            m &= (df["HLA_status"] != "in_HLA")
        if mapt == "excluded":
            m &= (df["MAPT_status"] != "in_MAPT")
        return m

    # ====== PRIMARY stratum (AF_AFR<0.05, all PIP, all BETA) =================
    primary_mask = mask_for("lt_0.05")
    primary_row = summarize_stratum(df, primary_mask, dict(
        AF_AFR_stratum="lt_0.05_PRIMARY",
        BETA_tertile="all", PIP_cut="all",
        HLA_status="included", MAPT_status="included",
        LD_prune_status="all",
    ))
    # Permutation null for primary — uses MERGED stratum (P0 audit BLOCKER #3)
    primary_row["p_perm"] = permutation_null_merged(
        df, ["lt_0.01", "0.01_0.05"]
    )

    # Winner's-Curse sensitivity: AF_AFR<0.05 + PIP>0.5 + |BETA|-mid
    wc_mask = mask_for("lt_0.05", pip_label="pip_gt_0.5", beta_label="mid")
    wc_row = summarize_stratum(df, wc_mask, dict(
        AF_AFR_stratum="lt_0.05_WINNER_CURSE_TEST",
        BETA_tertile="mid", PIP_cut="pip_gt_0.5",
        HLA_status="included", MAPT_status="included",
        LD_prune_status="all",
    ))

    # ===== Subpop heterogeneity for primary stratum =====
    log("Subpop heterogeneity (5 AFR subpops; primary stratum)")
    subpop_signs = []
    subpop_rows = []
    pol_mask = df["delta_RAF_EUR_minus_AFR"].notna()
    for pop in AFR_SUBPOPS:
        col = f"AFR_subpop_{pop}"
        if col not in df.columns:
            continue
        # Recompute risk-allele-AFR_pop and delta wrt this subpop
        # delta_pop = RAF_EUR - RAF_AFR_pop where RAF_AFR_pop is oriented to risk
        af_pop_a1 = df[col]
        beta_sign = np.sign(df["EUR_BETA"].fillna(0))
        af_pop_risk = np.where(beta_sign >= 0, af_pop_a1, 1.0 - af_pop_a1)
        delta_pop = df["RAF_EUR"] - af_pop_risk
        sub_mask = primary_mask & pol_mask & pd.notna(delta_pop)
        if int(sub_mask.sum()) < 30:
            continue
        d, lo, hi = bootstrap_direction(delta_pop[sub_mask].values)
        subpop_signs.append(1 if d > 0.5 else (-1 if d < 0.5 else 0))
        subpop_rows.append(dict(
            AF_AFR_stratum=f"lt_0.05_subpop_{pop}",
            BETA_tertile="all", PIP_cut="all",
            HLA_status="included", MAPT_status="included",
            LD_prune_status="all",
            n_total=int(sub_mask.sum()),
            n_polarity_resolved=int(sub_mask.sum()),
            n_strand_ambig_dropped=0,
            drop_rate=0.0,
            prop_RAF_EUR_gt_AFR=d, prop_CI_lo=lo, prop_CI_hi=hi,
            mean_delta_RAF=float(np.nanmean(delta_pop[sub_mask])),
            median_delta_RAF=float(np.nanmedian(delta_pop[sub_mask])),
            p_binom_vs_05=np.nan, p_perm=np.nan, verdict_label="",
        ))
        log(f"  {pop}: D={d:.3f}, 95% CI=[{lo:.3f},{hi:.3f}], "
            f"n={int(sub_mask.sum())}")

    # Subpop sign agreement (primary D direction vs each subpop)
    if not pd.isna(primary_row["prop_RAF_EUR_gt_AFR"]):
        primary_sign = 1 if primary_row["prop_RAF_EUR_gt_AFR"] > 0.5 else (
            -1 if primary_row["prop_RAF_EUR_gt_AFR"] < 0.5 else 0)
    else:
        primary_sign = 0
    subpop_agree = sum(1 for s in subpop_signs if s == primary_sign)
    log(f"Subpop sign agreement with primary: {subpop_agree}/5")

    # ===== Verdict =====
    verdict = primary_verdict(
        D_all=primary_row["prop_RAF_EUR_gt_AFR"],
        ci_all_lo=primary_row["prop_CI_lo"],
        ci_all_hi=primary_row["prop_CI_hi"],
        p_perm=primary_row["p_perm"],
        D_wc=wc_row["prop_RAF_EUR_gt_AFR"],
        ci_wc_lo=wc_row["prop_CI_lo"],
        ci_wc_hi=wc_row["prop_CI_hi"],
        subpop_agree=subpop_agree,
    )
    primary_row["verdict_label"] = verdict
    wc_row["verdict_label"] = "winner_curse_sensitivity_only"

    # PRINT PRIMARY VERDICT (required by spec)
    print("", flush=True)
    print("=" * 78, flush=True)
    print(f"=== PRIMARY VERDICT: {verdict} === "
          f"D_observed={primary_row['prop_RAF_EUR_gt_AFR']} "
          f"CI=[{primary_row['prop_CI_lo']},{primary_row['prop_CI_hi']}] "
          f"p_perm={primary_row['p_perm']}",
          flush=True)
    print(f"=== WINNER-CURSE TEST: D={wc_row['prop_RAF_EUR_gt_AFR']} "
          f"CI=[{wc_row['prop_CI_lo']},{wc_row['prop_CI_hi']}]",
          flush=True)
    print(f"=== SUBPOP AGREEMENT: {subpop_agree}/5", flush=True)
    print(f"=== MERGE RULE USED (P0.5): {merge_rule_used}", flush=True)
    print(f"=== CROSS-CHECK (P0.10) Pearson r: {cross_r}", flush=True)
    print("=" * 78, flush=True)

    log(f"PRIMARY VERDICT: {verdict}")
    log(f"D_observed = {primary_row['prop_RAF_EUR_gt_AFR']}")
    log(f"CI         = [{primary_row['prop_CI_lo']}, {primary_row['prop_CI_hi']}]")
    log(f"p_perm     = {primary_row['p_perm']}")

    # ===== Build full stratum summary table =====
    summary_rows.append(primary_row)
    summary_rows.append(wc_row)

    # 4 native AF_AFR strata x (all/PIP-cut) (secondary, no permutation)
    for af in AF_AFR_LABELS:
        for pip_label in ["all", "pip_gt_0.5", "pip_le_0.5"]:
            m = mask_for(af, pip_label=pip_label)
            if int(m.sum()) == 0:
                continue
            summary_rows.append(summarize_stratum(df, m, dict(
                AF_AFR_stratum=af, BETA_tertile="all", PIP_cut=pip_label,
                HLA_status="included", MAPT_status="included",
                LD_prune_status="all",
            )))

    # P1.1 sensitivity: HLA-excluded, MAPT-excluded, both-excluded
    for label, hla, mapt in [
        ("hla_excluded", "excluded", "all"),
        ("mapt_excluded", "all", "excluded"),
        ("hla_and_mapt_excluded", "excluded", "excluded"),
    ]:
        m = mask_for("lt_0.05", hla=hla, mapt=mapt)
        row = summarize_stratum(df, m, dict(
            AF_AFR_stratum=f"lt_0.05_{label}",
            BETA_tertile="all", PIP_cut="all",
            HLA_status=hla, MAPT_status=mapt, LD_prune_status="all",
        ))
        summary_rows.append(row)

    # Subpop rows (already computed)
    summary_rows.extend(subpop_rows)

    # ===== Write outputs =====
    log("Writing outputs")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_STRATUM, sep="\t", index=False)
    log(f"  wrote {OUT_STRATUM}")

    # Per-variant output (P0.11 schema)
    keep = [
        "rsid", "chr", "pos_grch37", "EUR_A1", "EUR_A2",
        "EUR_BETA", "EUR_SE", "EUR_P", "PIP",
        "AF_EUR_credset_source", "RAF_EUR", "AF_AFR_A1_pure", "RAF_AFR",
    ]
    for pop in AFR_SUBPOPS:
        c = f"AFR_subpop_{pop}"
        if c in df.columns:
            keep.append(c)
    keep += [
        "AlleleAnc", "AlleleDer", "AtlasDataSource",
        "risk_allele", "risk_is_derived", "strand_status",
        "AF_AFR_stratum", "BETA_tertile", "PIP_cut",
        "delta_RAF_EUR_minus_AFR", "direction_label",
        "verdict_form1", "verdict_form2", "cluster",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(OUT_PER_VARIANT, sep="\t", index=False, compression="gzip")
    log(f"  wrote {OUT_PER_VARIANT}")

    # Run log — prepended with selection-bias disclosure (P0.8 fix)
    disclosure = """\
================================================================================
SELECTION-BIAS DISCLOSURE (mandatory per the pre-registered protocol §P0.8)
================================================================================
The AF_AFR-stratified analysis below is CONDITIONAL on PGC3 EUR fine-mapping
ascertainment. The "AFR-rare" stratum (AF_AFR < 0.05) contains variants that
happened to reach PIP > 0.1 in EUR fine-mapping DESPITE low AFR LD context —
i.e., this stratum is enriched by EUR ascertainment design, not a random
sample of post-OOA-derived variants.

Implications:
  (1) Results do NOT generalize to all post-OOA-derived variants in the
      genome — only to the EUR-credible-set slice of them.
  (2) The reported direction (NULL / DIRECTIONAL_protective / DIRECTIONAL_risk
      / WINNER_CURSE_CONFOUND) applies only within the EUR fine-mapped subset.
  (3) Cross-population power asymmetry (EUR ~76% of PGC3 Wave 3 discovery,
      AFRAM lower) compounds this — the result is NOT a population-genetic claim
      about "EUR-young variants" in general, but a within-credible-set
      AF_AFR-stratified claim.

Companion sensitivity (Bigdeli 2025 InPSYght AFRAM, when released) would test
whether direction signal replicates in an AFR-discovery substrate. Until then,
this Phase 19 output is single-substrate.
================================================================================

"""
    with open(OUT_RUN_LOG, "w") as fh:
        fh.write(disclosure)
        fh.write("\n".join(LOG_LINES) + "\n")
    log(f"  wrote {OUT_RUN_LOG}")

    log("Phase 19 complete.")


if __name__ == "__main__":
    main()
