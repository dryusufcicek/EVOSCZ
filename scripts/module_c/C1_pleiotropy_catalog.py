#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Module C: Pleiotropic Architecture
======================================================
Step C1: Extracting Antagonistic Pleiotropy using EBI GWAS Catalog.

Identifies exact variant-level colocalization between PGC3 credible 
set variants and immune/autoimmune traits.

Instead of downloading 15 GB of summary stats for cross-trait LDSC, 
we use the ontology-annotated EBI GWAS Catalog to scan for 
empirical pleiotropy at the variant level.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_TABLE = PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv"
GWAS_CATALOG = list((PROJECT_ROOT / "data/raw").glob("gwas-catalog-download-associations-alt-full.tsv"))
RESULTS_DIR  = PROJECT_ROOT / "results/module_c"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Traits of interest regex
IMMUNE_REGEX = re.compile(
    r'(crohn|ulcerative colitis|rheumatoid arthritis|multiple sclerosis|type 1 diabetes|psoriasis|lupus|inflammatory bowel|autoimmune|leukocyte|lymphocyte|eosinophil|neutrophil|monocyte|c-reactive protein|interleukin|immune)',
    re.IGNORECASE
)

def format_beta(x):
    """Normalize OR to Beta or extract Beta."""
    try:
        val = float(x)
        # If the value is tiny, it's a beta. If it's around 1+, it could be OR.
        # But wait, GWAS catalog 'OR or BETA' column is tricky. 
        # Often, if CI is missing or it says "increase", it's beta.
        # For simplicity, we just extract the sign of the effect size. 
        # If val > 0, risk allele increases risk. (Wait, OR > 1 means risk allele increases risk, beta > 0 means risk allele increases risk).
        # We just return the float value to evaluate later.
        return val
    except:
        return np.nan

def main():
    print("=" * 60)
    print("EVOSCZ Module C: Variant-Level Pleiotropy Extraction")
    print("=" * 60)

    if not GWAS_CATALOG:
        print("Error: GWAS Catalog TSV not found in data/raw.")
        return
    gwas_cat_path = GWAS_CATALOG[0]

    # 1. Load PGC3 Variants
    print("1. Loading PGC3 Credible Sets...")
    pgc3 = pd.read_csv(MASTER_TABLE, sep='\t')
    pgc3_rsids = set(pgc3['rsid'].dropna())
    print(f"   {len(pgc3_rsids)} unique rsIDs tracked.")

    # 2. Extract Pleiotropic loci from GWAS Catalog
    print(f"2. Scanning GWAS Catalog for colocalization ({gwas_cat_path.name})...")
    # Read chunked to save memory
    chunks = []
    
    # GWAS Catalog columns: mapped to standard names
    cols_to_keep = ['SNPS', 'DISEASE/TRAIT', 'MAPPED_TRAIT', 'P-VALUE', 'OR or BETA', 'RISK ALLELE FREQUENCY']
    
    for chunk in pd.read_csv(gwas_cat_path, sep='\t', chunksize=50000, low_memory=False):
        # Match any PGC3 variant directly
        # Some 'SNPS' column has multiple rsIDs separated by comma
        mask = chunk['SNPS'].astype(str).apply(lambda x: any(rs in pgc3_rsids for rs in x.split(', ') if rs.startswith('rs')))
        matched = chunk[mask].copy()
        
        if len(matched) > 0:
            # Add back exact matching PGC3 rsid
            matched['pgc3_rsid'] = matched['SNPS'].apply(
                lambda x: next((rs for rs in x.split(', ') if rs in pgc3_rsids), None)
            )
            chunks.append(matched)

    pleio_df = pd.concat(chunks, ignore_index=True)
    print(f"   Found {len(pleio_df)} associations involving PGC3 credible set variants across ALL traits.")

    # 3. Filter for Immune/Autoimmune Traits
    print("3. Filtering for immune and autoimmune phenotypic pleiotropy...")
    pleio_df['is_immune'] = pleio_df['MAPPED_TRAIT'].astype(str).str.contains(IMMUNE_REGEX) | \
                            pleio_df['DISEASE/TRAIT'].astype(str).str.contains(IMMUNE_REGEX)
    
    immune_pleio = pleio_df[pleio_df['is_immune']].copy()
    print(f"   Filtered to {len(immune_pleio)} immune/autoimmune associations.")

    out_file = RESULTS_DIR / "C1_pleiotropic_loci.tsv"
    immune_pleio.to_csv(out_file, sep='\t', index=False)
    
    # Merge with PGC3 data to determine antagonistic directionality
    print("4. Mapping Pleiotropic Directionality...")
    # Clean beta column
    immune_pleio['immune_effect_size'] = immune_pleio['OR or BETA'].apply(format_beta)
    
    merged = immune_pleio.merge(pgc3, left_on='pgc3_rsid', right_on='rsid', suffixes=('_immune', '_pgc3'))
    
    # Determine directionality: 
    # If the risk alleles match between SCZ and the immune trait, the effect is synergistic (if both increase risk)
    # The GWAS catalog reports 'STRONGEST SNP-RISK ALLELE' (e.g. rs123-A).
    # We must parse the risk allele.
    
    def extract_risk_allele(s):
        try:
            return str(s).split('-')[1].strip()
        except:
            return np.nan
            
    merged['immune_risk_allele'] = merged['STRONGEST SNP-RISK ALLELE'].apply(extract_risk_allele)
    
    # Assume antagonistic if the immune protective allele is the SCZ risk allele
    # This requires careful allele harmonization. For now, we output the raw mappings.
    final_out = RESULTS_DIR / "C2_antagonistic_candidates.tsv"
    
    summary_cols = ['credible_set_id', 'rsid', 'chr', 'pos', 'effect_allele', 'other_allele', 'beta', 
                    'immune_risk_allele', 'immune_effect_size', 'DISEASE/TRAIT', 'P-VALUE']
    merged[summary_cols].to_csv(final_out, sep='\t', index=False)
    
    print(f"   Extracted effect directions for {len(merged)} colocalizations.")
    print(f"   Saved to: {final_out}")
    print("\nTop 5 Immune Pleiotropic Loci:")
    print(merged[['credible_set_id', 'rsid', 'DISEASE/TRAIT', 'P-VALUE']].head(5))

    
if __name__ == "__main__":
    main()
