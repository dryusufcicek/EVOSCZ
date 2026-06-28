#!/usr/bin/env python3
"""
Phase 12f: SCZ-Immune Colocalization Case Study Deep-Dive (variant-level)
==========================================================================
For each of the 16 H4>0.8 SCZ-immune coloc loci, extract variant-level context:
  - All credible-set variants in the locus with H4>0.8
  - Per-variant: age, SDS, GTEx brain min-p, GTEx blood min-p, brain specificity,
    ATAC cluster overlap, desert tier, HAR overlap, gene
  - Per-locus summary: how heterogeneous is the locus internally?

Produces an annotated case-study table for manuscript Table 5.

Output:
  - results/phase12/P12f_immune_coloc_variant_table.tsv (all variants in 16 loci)
  - results/phase12/P12f_locus_summary.tsv (one row per locus)
  - results/phase12/P12f_NARRATIVE.md (vignette per locus)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 12f: SCZ-Immune Coloc Deep-Dive — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 70)

# Load Phase 10 H4>0.8 case studies
case = pd.read_csv(BASE / "results/phase10/P10_H4_case_studies.tsv", sep="\t")
log(f"H4>0.8 immune coloc loci: {len(case)}")

# Load full C4 for all variants per locus
c4 = pd.read_csv(BASE / "results/module_c/C4_opentargets_validated.tsv", sep="\t")
c4_h4 = c4[c4["H4"] >= 0.8] if "H4" in c4.columns else c4
log(f"C4 variants with H4>0.8: {len(c4_h4)}")

# Load variant master (variant-level annotations)
m = pd.read_parquet(BASE / "results/phase11/variant_master.parquet")
m_uniq = m.drop_duplicates("rsid", keep="first")
log(f"Variant master loaded: {len(m_uniq)} unique variants")


# ─── Section 1: Variant-level table for each locus ────────────────────────
log("\n[Section 1] Building variant-level context table")

# Get all variant-level data for H4>0.8 case studies
case_loci = case["credible_set_id"].dropna().unique()
log(f"  Loci to process: {len(case_loci)}")

# All variants in these loci (regardless of H4 status)
locus_variants = m_uniq[m_uniq["credible_set_id"].isin(case_loci)].copy()
log(f"  All credible set variants in case loci: {len(locus_variants)}")

# Add locus-level case context
locus_meta = case[["credible_set_id", "lead_rsid", "immune_trait", "max_H4",
                    "pleiotropy_type", "l2g_gene"]].drop_duplicates("credible_set_id")
locus_variants = locus_variants.merge(locus_meta, on="credible_set_id", how="left")

# Compute brain specificity if both eQTLs present
mask = locus_variants["gtex_brain_minp"].notna() & locus_variants["gtex_blood_minp"].notna() & \
       (locus_variants["gtex_brain_minp"] > 0) & (locus_variants["gtex_blood_minp"] > 0)
locus_variants["brain_specificity"] = np.nan
if mask.sum() > 0:
    b = -np.log10(locus_variants.loc[mask, "gtex_brain_minp"].clip(lower=1e-300))
    bl = -np.log10(locus_variants.loc[mask, "gtex_blood_minp"].clip(lower=1e-300))
    locus_variants.loc[mask, "brain_specificity"] = b / (b + bl)

# Select key columns
keep_cols = ["credible_set_id", "rsid", "chr", "pos", "lead_rsid", "immune_trait",
             "max_H4", "pleiotropy_type", "l2g_gene",
             "age_median_yr", "age_group" if "age_group" in locus_variants.columns else "age_median_yr",
             "sds",
             "gtex_brain_minp", "gtex_brain_gene", "gtex_brain_tissue",
             "gtex_blood_minp", "gtex_blood_gene", "gtex_blood_tissue",
             "brain_specificity",
             "atac_n_clusters", "har_overlap", "desert_tier",
             "smr_colocalised_gene"]
keep_cols = [c for c in keep_cols if c in locus_variants.columns]
out_table = locus_variants[keep_cols].copy()
out_table = out_table.sort_values(["credible_set_id", "max_H4"], ascending=[True, False])

out_table.to_csv(OUT / "P12f_immune_coloc_variant_table.tsv", sep="\t", index=False)
log(f"  Saved: {OUT / 'P12f_immune_coloc_variant_table.tsv'} ({len(out_table)} rows)")


# ─── Section 2: Per-locus summary ──────────────────────────────────────────
log("\n[Section 2] Per-locus summary")

summary_rows = []
for cs_id in case_loci:
    sub = locus_variants[locus_variants["credible_set_id"] == cs_id]
    if len(sub) == 0:
        continue
    case_row = case[case["credible_set_id"] == cs_id].iloc[0]
    summary_rows.append({
        "credible_set_id": cs_id,
        "lead_rsid": case_row["lead_rsid"],
        "immune_trait": case_row["immune_trait"],
        "max_H4": case_row["max_H4"],
        "l2g_gene": case_row["l2g_gene"],
        "median_age_yr": sub["age_median_yr"].median(),
        "age_group": case_row["age_group"],
        "n_variants_total": len(sub),
        "n_with_brain_eqtl": sub["gtex_brain_minp"].notna().sum(),
        "n_with_blood_eqtl": sub["gtex_blood_minp"].notna().sum(),
        "n_with_both": ((sub["gtex_brain_minp"].notna()) & (sub["gtex_blood_minp"].notna())).sum(),
        "median_brain_specificity": sub["brain_specificity"].median(),
        "min_brain_p": sub["gtex_brain_minp"].min(),
        "min_blood_p": sub["gtex_blood_minp"].min(),
        "n_in_atac_cluster": (sub["atac_n_clusters"] > 0).sum() if "atac_n_clusters" in sub.columns else 0,
        "n_in_desert": (sub["desert_tier"] > 0).sum(),
        "n_har_overlap": sub["har_overlap"].sum() if "har_overlap" in sub.columns else 0,
        "smr_genes_present": sub["smr_colocalised_gene"].sum() if "smr_colocalised_gene" in sub.columns else 0,
    })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("median_age_yr")
summary.to_csv(OUT / "P12f_locus_summary.tsv", sep="\t", index=False)
log(f"  Saved: {OUT / 'P12f_locus_summary.tsv'} ({len(summary)} loci)")

# Print summary table
log("\n  Locus summary:")
log(f"  {'CS_ID':<10s} {'gene':<12s} {'age':>10s} {'n_var':>5s} {'n_brain':>7s} {'n_blood':>7s} "
    f"{'spec':>6s} {'desert':>6s} {'trait':<30s}")
for _, r in summary.iterrows():
    spec_s = f"{r['median_brain_specificity']:.3f}" if pd.notna(r['median_brain_specificity']) else "NA"
    age_s = f"{r['median_age_yr']:.0f}" if pd.notna(r['median_age_yr']) else "NA"
    gene = str(r['l2g_gene'])[:11] if pd.notna(r['l2g_gene']) else "NA"
    trait = str(r['immune_trait'])[:30]
    log(f"  {r['credible_set_id']:<10s} {gene:<12s} {age_s:>10s} {r['n_variants_total']:>5d} "
        f"{r['n_with_brain_eqtl']:>7d} {r['n_with_blood_eqtl']:>7d} {spec_s:>6s} "
        f"{r['n_in_desert']:>6d} {trait:<30s}")


# ─── Section 3: Narrative vignette ─────────────────────────────────────────
log("\n[Section 3] Per-locus narrative vignette")

vignettes = []
vignettes.append("# Phase 12f — SCZ-Immune Colocalization Variant-Level Vignettes\n")
vignettes.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
vignettes.append("**Data:** 16 PGC3 credible sets with H4>0.8 SCZ-immune coloc, "
                 "annotated with variant-level evolution + regulatory features.\n")
vignettes.append("---\n")

for _, r in summary.iterrows():
    sub = locus_variants[locus_variants["credible_set_id"] == r["credible_set_id"]].copy()
    cs_id = r["credible_set_id"]
    gene = r['l2g_gene'] if pd.notna(r['l2g_gene']) else "—"
    trait = r['immune_trait']
    age = r['median_age_yr'] if pd.notna(r['median_age_yr']) else 0
    n_var = r['n_variants_total']
    H4 = r['max_H4']
    spec = r['median_brain_specificity']

    vignettes.append(f"## {cs_id} — {gene}: SCZ ↔ {trait}\n")
    vignettes.append(f"- **Lead variant:** {r['lead_rsid']}, max H4 = {H4:.3f}")
    vignettes.append(f"- **Median age:** {age:.0f} yr ({r['age_group']})")
    vignettes.append(f"- **Credible set size:** {n_var} variants total")
    vignettes.append(f"  - {r['n_with_brain_eqtl']} with GTEx brain eQTL")
    vignettes.append(f"  - {r['n_with_blood_eqtl']} with GTEx blood eQTL")
    vignettes.append(f"  - {r['n_with_both']} with both")
    if pd.notna(spec):
        if spec > 0.6:
            spec_label = "BRAIN-DOMINANT"
        elif spec < 0.4:
            spec_label = "BLOOD-DOMINANT"
        else:
            spec_label = "balanced"
        vignettes.append(f"  - Median brain specificity: {spec:.3f} ({spec_label})")
    if r['n_in_desert'] > 0:
        vignettes.append(f"- **Introgression desert:** {r['n_in_desert']} variants in tier 1/2/3")
    if r['n_har_overlap'] > 0:
        vignettes.append(f"- **HAR overlap:** {r['n_har_overlap']} variants")

    # Top 3 variants by H4 or brain eQTL strength
    top = sub.nsmallest(3, "gtex_brain_minp") if sub["gtex_brain_minp"].notna().any() else sub.head(3)
    vignettes.append("- **Top eQTL variants in locus:**")
    for _, v in top.iterrows():
        bp = f"{v['gtex_brain_minp']:.1e}" if pd.notna(v.get('gtex_brain_minp')) else "NA"
        bg = v.get('gtex_brain_gene', 'NA') or 'NA'
        bt = v.get('gtex_brain_tissue', 'NA') or 'NA'
        sds = f"{v['sds']:.2f}" if pd.notna(v.get('sds')) else "NA"
        v_age = f"{v['age_median_yr']:.0f}" if pd.notna(v.get('age_median_yr')) else "NA"
        vignettes.append(f"  - {v['rsid']}: age={v_age}, SDS={sds}, "
                          f"brain_p={bp} ({bg}, {bt[:25] if isinstance(bt,str) else 'NA'})")

    vignettes.append("")  # spacing

with open(OUT / "P12f_NARRATIVE.md", "w") as f:
    f.write("\n".join(vignettes))
log(f"  Saved: {OUT / 'P12f_NARRATIVE.md'}")


# ─── Section 4: Heterogeneity analysis ─────────────────────────────────────
log("\n[Section 4] Within-locus heterogeneity")

# How variable is brain specificity WITHIN each locus? If high heterogeneity,
# the H4 finding masks complex within-locus structure.
het = []
for cs_id in case_loci:
    sub = locus_variants[locus_variants["credible_set_id"] == cs_id]
    spec = sub["brain_specificity"].dropna()
    if len(spec) >= 3:
        het.append({
            "credible_set_id": cs_id,
            "n_with_spec": len(spec),
            "spec_mean": spec.mean(),
            "spec_std": spec.std(),
            "spec_range": spec.max() - spec.min(),
        })

if het:
    het_df = pd.DataFrame(het).sort_values("spec_std", ascending=False)
    log(f"\n  Loci with heterogeneous brain specificity (top variability):")
    for _, r in het_df.head(5).iterrows():
        log(f"    {r['credible_set_id']}: n={int(r['n_with_spec'])}, "
            f"mean={r['spec_mean']:.3f}, std={r['spec_std']:.3f}, range={r['spec_range']:.3f}")

# Save log
with open(OUT / "P12f_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 12f Analysis Log\n\n```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log(f"\nSaved log: {OUT / 'P12f_ANALYSIS_LOG.md'}")
log("Phase 12f complete.")
