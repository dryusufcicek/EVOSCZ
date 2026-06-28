#!/usr/bin/env python3
"""Phase 17B: Brain_spec-only single-annotation LDSC sensitivity.

Robustness check #1: Does brain_spec alone (without age or |iHS|) yield the same
heritability concentration as the 3D Young cluster? If yes, the "evolutionary
cluster" framing is over-stated; if no, the joint-feature definition carries
non-redundant load.

Two annotations tested:
  A) Top-1742 brain_spec — C0-size-matched binary annotation (brain_spec ≥ 0.553,
     1,742 highest-brain_spec variants among brain_spec-available non-MAPT set).
     Brain-biased SNPs only — no age or |iHS| feature.
  B) Continuous brain_spec — per-SNP brain_spec value [0,1] annotation; tests
     whether the brain-vs-blood specificity axis as a continuous variable
     explains heritability concentration.

Annotation files built for all 22 chromosomes; LD scores computed; S-LDSC
applied on PGC3 EUR schizophrenia with 53-baseline-LD.

Output:
  results/phase17b/brain_spec_top1742.tsv.gz
  results/phase17b/brain_spec_top1742_annot/
  results/phase17b/brain_spec_continuous_annot/
  results/phase17b/h2_pgc3_eur_brain_spec_top1742.results
  results/phase17b/h2_pgc3_eur_brain_spec_continuous.results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
P14E_BASELINE = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
OUT = BASE / "results/phase17b"
OUT_TOP_ANNOT = OUT / "brain_spec_top1742_annot"
OUT_CONT_ANNOT = OUT / "brain_spec_continuous_annot"
for d in [OUT, OUT_TOP_ANNOT, OUT_CONT_ANNOT]:
    d.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17B: Brain_spec-only annotations — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# ─── 1. RECOMPUTE FEATURES (identical to P14b) ─────────────────────────────
m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["b_logp"]  = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                          -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                          -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce")
in_mapt = (m["chr_int"] == 17) & (m["pos_int"] >= 43_000_000) & (m["pos_int"] <= 46_000_000)
m_nm = m[~in_mapt].copy()

# ─── 2. ANNOTATION A: TOP-1742 BRAIN_SPEC ──────────────────────────────────
log("\n[A] Top-1742 brain_spec annotation (C0-size-matched)")
bs_avail = m_nm["brain_spec"].notna()
top_n = 1742  # match C0 cluster size in primary
top_rsids = m_nm[bs_avail].nlargest(top_n, "brain_spec")["rsid"].astype(str).tolist()
log(f"  Top-{top_n} brain_spec rsids selected")
log(f"  brain_spec threshold: ≥ {m_nm.loc[m_nm['rsid'].isin(top_rsids), 'brain_spec'].min():.4f}")
log(f"  Median brain_spec: {m_nm.loc[m_nm['rsid'].isin(top_rsids), 'brain_spec'].median():.4f}")
top_set = set(top_rsids)

# Build annotation files per chr (single binary column: BRAIN_SPEC_HIGH)
log("\n  Building per-chrom binary annotation files…")
for chrom in range(1, 23):
    base_path = P14E_BASELINE / f"baseline.{chrom}.annot.gz"
    if not base_path.exists():
        continue
    base = pd.read_csv(base_path, sep="\t", compression="gzip",
                       usecols=["CHR", "BP", "SNP", "CM"])
    base["BRAIN_SPEC_HIGH"] = base["SNP"].astype(str).isin(top_set).astype(int)
    out_path = OUT_TOP_ANNOT / f"brain_spec_high.{chrom}.annot.gz"
    base.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n_high = int(base["BRAIN_SPEC_HIGH"].sum())
    log(f"  chr{chrom}: {n_high:>4d} brain_spec_high SNPs")

# ─── 3. ANNOTATION B: CONTINUOUS BRAIN_SPEC ────────────────────────────────
log("\n[B] Continuous brain_spec annotation (per-SNP value [0,1])")

# rsid → brain_spec value (only computed for variants with both brain and blood eQTL)
rsid2bs = dict(zip(m_nm.loc[bs_avail, "rsid"].astype(str),
                    m_nm.loc[bs_avail, "brain_spec"].astype(float)))
log(f"  brain_spec computable for {len(rsid2bs):,} non-MAPT variants")

for chrom in range(1, 23):
    base_path = P14E_BASELINE / f"baseline.{chrom}.annot.gz"
    if not base_path.exists():
        continue
    base = pd.read_csv(base_path, sep="\t", compression="gzip",
                       usecols=["CHR", "BP", "SNP", "CM"])
    base["BRAIN_SPEC"] = base["SNP"].astype(str).map(rsid2bs).fillna(0.0)
    out_path = OUT_CONT_ANNOT / f"brain_spec.{chrom}.annot.gz"
    base.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n_nonzero = int((base["BRAIN_SPEC"] > 0).sum())
    mean_bs = float(base.loc[base["BRAIN_SPEC"] > 0, "BRAIN_SPEC"].mean()) if n_nonzero > 0 else 0.0
    log(f"  chr{chrom}: {n_nonzero:>5d} SNPs with brain_spec ≠ 0 (mean among non-zero: {mean_bs:.3f})")

log(f"\nDone. Annot files in {OUT_TOP_ANNOT} and {OUT_CONT_ANNOT}")
log(f"Next: compute LD scores per-chrom, then S-LDSC.")

with open(OUT / "P17b_annot_build_log.md", "w") as f:
    f.write("# Phase 17B: Brain_spec-only annotation build\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
