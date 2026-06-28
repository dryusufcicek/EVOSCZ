#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Step U1: Extract PGC3 Credible Sets
=====================================================
Converts the PGC3 Supplementary Table 11 (FINEMAP 95% credible sets)
into the pipeline's master variant table format.

Input:  data/raw/pgc3_finemapping/ST11_credible_sets.xlsx
Output: data/processed/pgc3_master_variants.tsv
        data/processed/pgc3_region_definitions.tsv
        data/processed/pgc3_lead_snps.tsv

Scientific Rationale (Critical Thinking — Construct Validity):
    We use the FINEMAP 95% credible sets (ST11a), not just index SNPs,
    because index SNPs are the most statistically significant variant
    at each locus — NOT necessarily the causal variant. Credible sets
    contain the minimal set of variants with ≥95% probability of
    including the true causal variant (Bayesian fine-mapping).
    
    We create THREE output files for different analytical needs:
    - master_variants: ALL credible set SNPs (for enrichment analyses)
    - lead_snps: highest-PIP SNP per locus (for colocalization, point estimates)
    - region_definitions: FINEMAP region boundaries (for window-based analyses)

Coordinates: GRCh37/hg19 (as published in PGC3)
    NOTE: Some downstream analyses require hg38. Liftover will be
    performed in Step U4 using UCSC liftOver.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CREDIBLE_SETS = PROJECT_ROOT / "data/raw/pgc3_finemapping/ST11_credible_sets.xlsx"
INPUT_REGIONS = INPUT_CREDIBLE_SETS  # Same file, different sheet
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_credible_sets():
    """Extract and clean the FINEMAP 95% credible sets from ST11a."""
    
    print("=" * 60)
    print("EVOSCZ Step U1: Extracting PGC3 Credible Sets")
    print("=" * 60)
    
    # ─── Read ST11a: Full 95% credible sets ───
    print("\n[1/5] Reading ST11a (full 95% credible sets)...")
    df_full = pd.read_excel(
        INPUT_CREDIBLE_SETS,
        sheet_name="ST11a 95% Credible Sets",
        engine="openpyxl"
    )
    print(f"  Raw rows: {len(df_full):,}")
    print(f"  Columns: {list(df_full.columns)}")
    
    # ─── Standardize column names ───
    print("\n[2/5] Standardizing schema...")
    df = df_full.rename(columns={
        "index_snp": "index_snp",
        "expected_causals_k": "expected_k",
        "extended_gwas": "extended_gwas",
        "rsid": "rsid",
        "chromosome": "chr",
        "position": "pos",
        "GWAS_effect_allele": "effect_allele",
        "other_allele": "other_allele",
        "maf": "maf",
        "or": "odds_ratio",
        "se": "se",
        "pval": "pval",
        "finemap_posterior_probability": "pip",
        "gene_symbol": "gene_symbol",
        "gene_hgnc": "gene_hgnc_id",
        "gene_ensembl": "gene_ensembl",
        "transcript_feature": "transcript_feature",
        "canonical_transcript": "canonical_transcript",
        "ensembl_gene_classification": "gene_biotype",
        "impact": "vep_impact",
        "sift": "sift",
        "polyphen": "polyphen",
        "cadd": "cadd_phred"
    })

    # ─── Derive beta from odds ratio ───
    # PGC3 reports OR for case-control. Convert to log(OR) = beta
    df["beta"] = np.log(df["odds_ratio"])

    # ─── Assign credible set IDs ───
    # Each unique index_snp defines one credible set (locus)
    locus_map = {snp: f"CS_{i+1:03d}" for i, snp in enumerate(df["index_snp"].unique())}
    df["credible_set_id"] = df["index_snp"].map(locus_map)
    
    # ─── Compute derived allele frequency (DAF) ───
    # MAF in PGC3 is minor allele frequency. For evolutionary analyses,
    # we need DERIVED allele frequency. This requires ancestral allele
    # annotation (Step U4). For now, use MAF as a placeholder.
    # NOTE: This will be updated in Step U4 when we add ancestral alleles.
    df["daf_placeholder"] = df["maf"]

    # ─── Quality summary ───
    n_loci = df["credible_set_id"].nunique()
    n_variants = len(df)
    median_pip = df["pip"].median()
    variants_per_locus = df.groupby("credible_set_id").size()
    
    print(f"\n  Credible set loci: {n_loci}")
    print(f"  Total variants: {n_variants:,}")
    print(f"  Variants/locus: median={variants_per_locus.median():.0f}, "
          f"mean={variants_per_locus.mean():.1f}, "
          f"range=[{variants_per_locus.min()}, {variants_per_locus.max()}]")
    print(f"  Median PIP: {median_pip:.4f}")
    print(f"  Variants with PIP ≥ 0.5: {(df['pip'] >= 0.5).sum()}")
    print(f"  Variants with PIP ≥ 0.1: {(df['pip'] >= 0.1).sum()}")
    
    # ─── Select output columns ───
    out_cols = [
        "credible_set_id", "index_snp", "rsid", "chr", "pos",
        "effect_allele", "other_allele", "maf", "beta", "se", "pval", "pip",
        "odds_ratio", "expected_k", "extended_gwas",
        "gene_symbol", "gene_ensembl", "gene_biotype",
        "vep_impact", "sift", "polyphen", "cadd_phred"
    ]
    df_out = df[out_cols].copy()
    
    # ─── Save master variants ───
    print("\n[3/5] Saving master variant table...")
    master_path = OUTPUT_DIR / "pgc3_master_variants.tsv"
    df_out.to_csv(master_path, sep="\t", index=False)
    print(f"  → {master_path} ({len(df_out):,} rows)")
    
    # ─── Extract lead SNPs (highest PIP per locus) ───
    print("\n[4/5] Extracting lead SNPs (max PIP per locus)...")
    lead_idx = df_out.groupby("credible_set_id")["pip"].idxmax()
    df_leads = df_out.loc[lead_idx].copy()
    df_leads = df_leads.sort_values(["chr", "pos"]).reset_index(drop=True)
    
    lead_path = OUTPUT_DIR / "pgc3_lead_snps.tsv"
    df_leads.to_csv(lead_path, sep="\t", index=False)
    print(f"  → {lead_path} ({len(df_leads)} loci)")
    
    # ─── Extract region definitions ───
    print("\n[5/5] Extracting FINEMAP region definitions...")
    df_regions = pd.read_excel(
        INPUT_CREDIBLE_SETS,
        sheet_name="ST11e Region definitions",
        engine="openpyxl"
    )
    df_regions = df_regions.rename(columns={
        "index_snp": "index_snp",
        "chr": "chr",
        "start": "region_start",
        "stop": "region_end",
        "clump_indexes": "clump_indexes"
    })
    df_regions["credible_set_id"] = df_regions["index_snp"].map(locus_map)
    df_regions["region_size_kb"] = (df_regions["region_end"] - df_regions["region_start"]) / 1000
    
    region_path = OUTPUT_DIR / "pgc3_region_definitions.tsv"
    df_regions.to_csv(region_path, sep="\t", index=False)
    print(f"  → {region_path} ({len(df_regions)} regions)")
    print(f"  Median region size: {df_regions['region_size_kb'].median():.1f} kb")
    
    # ─── Summary statistics ───
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Credible set loci:     {n_loci}")
    print(f"  Total credible SNPs:   {n_variants:,}")
    chrs = sorted(df_out['chr'].unique(), key=lambda x: int(x) if str(x).isdigit() else 99)
    print(f"  Chromosomes covered:   {chrs}")
    print(f"  High-PIP (≥0.5):       {(df_out['pip'] >= 0.5).sum()} variants at "
          f"{df_out[df_out['pip'] >= 0.5]['credible_set_id'].nunique()} loci")
    print(f"  Single-SNP sets:       {(variants_per_locus == 1).sum()} loci")
    print(f"  Extended GWAS loci:    {df_out[df_out['extended_gwas'] == 'YES']['credible_set_id'].nunique()}")
    
    # Gene biotype distribution
    print(f"\n  VEP Impact distribution (top-PIP variants):")
    impact_dist = df_leads["vep_impact"].value_counts()
    for impact, count in impact_dist.items():
        print(f"    {impact}: {count}")
    
    return df_out, df_leads, df_regions


def extract_gene_prioritization():
    """Extract the PGC3 prioritized gene list from ST12."""
    
    print("\n\n" + "=" * 60)
    print("Extracting Gene Prioritization (ST12)")
    print("=" * 60)
    
    input_path = PROJECT_ROOT / "data/raw/pgc3_gene_priority/ST12_gene_prioritization.xlsx"
    
    # Full gene annotation table
    df_all = pd.read_excel(input_path, sheet_name="ST12 all criteria", engine="openpyxl")
    all_path = OUTPUT_DIR / "pgc3_gene_annotations_full.tsv"
    df_all.to_csv(all_path, sep="\t", index=False)
    print(f"  All gene annotations: {len(df_all)} genes → {all_path}")
    
    # Prioritized genes only
    df_prior = pd.read_excel(input_path, sheet_name="Prioritised", engine="openpyxl")
    prior_path = OUTPUT_DIR / "pgc3_prioritized_genes.tsv"
    df_prior.to_csv(prior_path, sep="\t", index=False)
    print(f"  Prioritized genes:    {len(df_prior)} genes → {prior_path}")
    
    return df_all, df_prior


if __name__ == "__main__":
    print("EVOSCZ Pipeline — Upstream Step U1")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input: {INPUT_CREDIBLE_SETS}")
    print()
    
    # Check input exists
    if not INPUT_CREDIBLE_SETS.exists():
        # Follow symlink
        real_path = INPUT_CREDIBLE_SETS.resolve()
        if not real_path.exists():
            print(f"ERROR: Input file not found: {INPUT_CREDIBLE_SETS}")
            print(f"  Resolved path: {real_path}")
            sys.exit(1)
    
    df_master, df_leads, df_regions = extract_credible_sets()
    df_genes_all, df_genes_prior = extract_gene_prioritization()
    
    print("\n\n✔ Step U1 complete. Files ready in data/processed/")
