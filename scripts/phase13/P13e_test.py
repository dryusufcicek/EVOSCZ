#!/usr/bin/env python3
"""
Phase 13e_test: Neighbor Controls — Annotate + Compare
========================================================
Annotate the 6,350 neighbor variants from P13e_neighbor_variants.tsv.gz
with GEVA age + GTEx brain/blood eQTL, then run within-locus residual
brain_spec × age test. Compare to PGC3 effect.

Output:
  - results/phase13/P13e_neighbor_test_results.tsv
  - results/phase13/P13e_NARRATIVE.md
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase13"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 13e_test: Neighbor Controls — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)

# Load neighbors
neighbors = pd.read_csv(OUT / "P13e_neighbor_variants.tsv.gz", sep="\t")
log(f"Neighbors loaded: {len(neighbors)}")
log(f"Chromosomes covered: {sorted(neighbors['chr'].unique())}")

# Annotate with GEVA age (load atlas chunk-by-chunk)
log("\n[1] Annotate with GEVA age")
geva_dir = BASE / "data/raw/annotations/allele_ages/geva/atlas_bulk"
geva_age_map = {}  # (chr, pos) → age
for chrom in sorted(neighbors["chr"].astype(str).unique(), key=lambda x: int(x)):
    f = geva_dir / f"atlas.chr{chrom}.csv.gz"
    if not f.exists():
        continue
    g = pd.read_csv(f, sep=",", skiprows=3, compression="gzip",
                    skipinitialspace=True,
                    usecols=["VariantID", "Chromosome", "Position", "AgeMedian_Mut", "DataSource"])
    g["_rank"] = g["DataSource"].map({"Combined": 0, "TGP": 1, "SGDP": 2}).fillna(99)
    g = g.sort_values(["Chromosome", "Position", "_rank"]).drop_duplicates(["Chromosome", "Position"], keep="first")
    for _, r in g.iterrows():
        geva_age_map[(str(r["Chromosome"]), int(r["Position"]))] = r["AgeMedian_Mut"]
    log(f"  chr{chrom}: {len(g)} GEVA entries")

neighbors["age_median_yr"] = neighbors.apply(
    lambda r: geva_age_map.get((str(r["chr"]), int(r["pos"])), np.nan), axis=1)
n_age = neighbors["age_median_yr"].notna().sum()
log(f"  Neighbors with age: {n_age}/{len(neighbors)} ({n_age/len(neighbors)*100:.1f}%)")


# LiftOver hg19→hg38
log("\n[2] LiftOver hg19→hg38")
from pyliftover import LiftOver
lo = LiftOver("hg19", "hg38")
def lift(chrom, pos):
    try:
        r = lo.convert_coordinate(f"chr{chrom}", int(pos))
        if r and len(r) > 0:
            return r[0][1]
    except: pass
    return None
neighbors["pos_hg38"] = [lift(c, p) for c, p in zip(neighbors["chr"], neighbors["pos"])]
log(f"  Lifted: {neighbors['pos_hg38'].notna().sum()}/{len(neighbors)}")


# GTEx lookup
log("\n[3] GTEx v10 brain + blood eQTL lookup")
sys.path.insert(0, str(BASE / "scripts/phase11"))
from lib_gtex_v10 import load_tissue, BRAIN_TISSUES, BLOOD_IMMUNE_TISSUES, list_available_parquet
available = list_available_parquet()

def collect_min_p(df_query, tissue_set, label):
    d = df_query[["rsid", "chr", "pos_hg38"]].copy()
    d["chr"] = d["chr"].astype(str)
    d = d[d["pos_hg38"].notna()]
    d["pos"] = d["pos_hg38"].astype(int)
    d = d.drop(columns=["pos_hg38"])
    rows = []
    for t in tissue_set:
        if t not in available: continue
        try:
            eqtl = load_tissue(t)[["chr", "pos", "gene_id", "pval_nominal"]]
            sub = eqtl.merge(d, on=["chr", "pos"], how="inner")
            rows.append(sub)
        except: pass
    if not rows: return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    idx = full.groupby("rsid")["pval_nominal"].idxmin()
    minp = full.loc[idx, ["rsid", "pval_nominal"]].copy()
    minp.columns = ["rsid", f"gtex_{label}_minp"]
    return minp

bp = collect_min_p(neighbors, BRAIN_TISSUES, "brain")
blp = collect_min_p(neighbors, BLOOD_IMMUNE_TISSUES, "blood")
neighbors = neighbors.merge(bp, on="rsid", how="left").merge(blp, on="rsid", how="left")
log(f"  Brain eQTL: {neighbors['gtex_brain_minp'].notna().sum()}")
log(f"  Blood eQTL: {neighbors['gtex_blood_minp'].notna().sum()}")


# Compute brain spec
mask = neighbors["gtex_brain_minp"].notna() & neighbors["gtex_blood_minp"].notna() & \
       (neighbors["gtex_brain_minp"]>0) & (neighbors["gtex_blood_minp"]>0)
neighbors["brain_spec"] = np.nan
if mask.sum() > 0:
    b = -np.log10(neighbors.loc[mask, "gtex_brain_minp"].clip(lower=1e-300))
    bl = -np.log10(neighbors.loc[mask, "gtex_blood_minp"].clip(lower=1e-300))
    neighbors.loc[mask, "brain_spec"] = b / (b + bl)

neighbors.to_csv(OUT / "P13e_neighbor_annotated.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"  Annotated saved: {OUT / 'P13e_neighbor_annotated.tsv.gz'}")


# ─── Within-locus + MAF residualized test ─────────────────────────────────
log("\n[4] Within-locus + MAF residualized test")

import sys as _sys
_sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation


def maf_resid_rho(df, x_col, y_col, locus_col, maf_col="maf"):
    """Within-locus partial rank correlation (code-review Faz C lib)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]

# Prepare data
n_age = neighbors[neighbors["age_median_yr"].notna() & (neighbors["age_median_yr"]>0)].copy()
n_age["log_age"] = np.log10(n_age["age_median_yr"])
log(f"\n  Neighbors usable for analysis: {len(n_age)}")

# Run brain_spec × age
rho_n, p_n, n_n = maf_resid_rho(n_age, "log_age", "brain_spec", "credible_set_id")
log(f"\n  Neighbor brain_spec × age (within-locus + MAF resid):")
log(f"    n={n_n}, rho={rho_n:.4f}, p={p_n:.3e}")

# PGC3 reference (for comparison)
m_pgc = pd.read_parquet(BASE / "results/phase11/variant_master_v2.parquet")
m_pgc = m_pgc[m_pgc["age_median_yr"].notna() & (m_pgc["age_median_yr"]>0) & m_pgc["maf"].notna()].copy()
m_pgc["log_age"] = np.log10(m_pgc["age_median_yr"])
m_pgc["b_logp"]  = np.where(m_pgc["gtex_brain_minp"].notna() & (m_pgc["gtex_brain_minp"]>0),
                              -np.log10(m_pgc["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m_pgc["bl_logp"] = np.where(m_pgc["gtex_blood_minp"].notna() & (m_pgc["gtex_blood_minp"]>0),
                              -np.log10(m_pgc["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m_pgc["brain_spec"] = m_pgc["b_logp"] / (m_pgc["b_logp"] + m_pgc["bl_logp"])
m_pgc["chr_int"] = pd.to_numeric(m_pgc["chr"], errors="coerce")

# Restrict PGC3 to same chromosomes covered by neighbors
covered_chrs = set(neighbors["chr"].astype(str).unique())
m_pgc_match = m_pgc[m_pgc["chr_int"].astype("Int64").astype(str).isin(covered_chrs)].copy()
log(f"  PGC3 restricted to neighbor-covered chrs: {len(m_pgc_match)}")

rho_p, p_p, n_p = maf_resid_rho(m_pgc_match, "log_age", "brain_spec", "credible_set_id")
log(f"\n  PGC3 (same chrs) brain_spec × age:")
log(f"    n={n_p}, rho={rho_p:.4f}, p={p_p:.3e}")

# Comparison
log(f"\n  COMPARISON:")
log(f"    PGC3 effect:     rho = {rho_p:>+7.4f} (n={n_p})")
log(f"    Neighbor effect: rho = {rho_n:>+7.4f} (n={n_n})")
log(f"    Difference:      Δrho = {rho_p - rho_n:>+7.4f}")
log(f"\n  Interpretation:")
log(f"    If Δrho substantially negative (more negative in PGC3) →")
log(f"      brain_spec × age dissociation is STRONGER in fine-mapped SCZ variants")
log(f"      (i.e., effect is SCZ-specific, not generic LD-block property)")

# Save
results = pd.DataFrame([
    {"group": "PGC3", "test": "brain_spec × age", "n": n_p, "rho": rho_p, "p": p_p},
    {"group": "Neighbor", "test": "brain_spec × age", "n": n_n, "rho": rho_n, "p": p_n},
])
results.to_csv(OUT / "P13e_neighbor_test_results.tsv", sep="\t", index=False)
log(f"\n  Saved: {OUT / 'P13e_neighbor_test_results.tsv'}")

with open(OUT / "P13e_NARRATIVE.md", "w") as f:
    f.write("# Phase 13e: Neighbor Control Comparison\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13e_test complete.")
