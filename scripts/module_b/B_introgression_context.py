#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Module B: Introgression Desert Enrichment
=============================================================
Tests whether SCZ risk variants are enriched in Neanderthal
introgression deserts — genomic regions depleted of archaic
ancestry, suggesting strong purifying selection against
introgressed alleles.

Steps:
  B1. Intersect PGC3 variants with desert coordinates (tiered)
  B2. Desert enrichment testing (Fisher's exact + permutation)
  B3. HAR proximity analysis (distance to nearest HAR)
  B4. Combined evolutionary signature

Scientific Rationale:
  Introgression deserts mark regions where Neanderthal alleles
  were systematically purged from modern human genomes, likely
  because they were deleterious. If SCZ variants are enriched
  in deserts, it suggests these loci have been under strong
  evolutionary constraint — yet they harbor common risk alleles
  maintained by balancing selection or antagonistic pleiotropy.
  
  Additionally, HARs (Human Accelerated Regions) mark regions
  under positive selection specific to modern humans. HAR proximity
  provides evidence for recent adaptive evolution affecting SCZ loci.

Data Sources:
  Deserts: Chen, Velazquez-Arcelay & Capra 2026 MBE
  HARs: Cui et al. 2025 Nature (3,257 HARs, lifted to hg19)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_TABLE = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"
DESERT_DIR   = PROJECT_ROOT / "data/raw/annotations/introgression_deserts"
HAR_BED      = PROJECT_ROOT / "data/raw/annotations/HARs/HAR_coordinates_hg19.bed"
RESULTS_DIR  = PROJECT_ROOT / "results/module_b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_bed(path):
    """Load a BED file into a list of (chrom, start, end, name) tuples."""
    regions = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 3:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                name = fields[3] if len(fields) > 3 else '.'
                regions.append((chrom, start, end, name))
    return regions


def variant_in_regions(chrom, pos, regions_by_chr):
    """Check if a variant falls within any region."""
    chrom_str = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
    for (start, end, name) in regions_by_chr.get(chrom_str, []):
        if start <= pos < end:
            return True, name
    return False, None


def nearest_region_distance(chrom, pos, regions_by_chr):
    """Find distance to nearest region."""
    chrom_str = f"chr{chrom}" if not str(chrom).startswith('chr') else str(chrom)
    min_dist = float('inf')
    nearest_name = None
    
    for (start, end, name) in regions_by_chr.get(chrom_str, []):
        if start <= pos < end:
            return 0, name  # inside the region
        dist = min(abs(pos - start), abs(pos - end))
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
    
    return min_dist, nearest_name


def organize_regions(regions):
    """Organize regions by chromosome for fast lookup."""
    by_chr = {}
    for chrom, start, end, name in regions:
        by_chr.setdefault(chrom, []).append((start, end, name))
    # Sort for binary search later
    for chrom in by_chr:
        by_chr[chrom].sort()
    return by_chr


def genome_desert_fraction(deserts_by_chr, genome_size=3.0e9):
    """Estimate fraction of genome in deserts."""
    total_bp = 0
    for chrom, intervals in deserts_by_chr.items():
        for start, end, _ in intervals:
            total_bp += (end - start)
    return total_bp / genome_size, total_bp


def step_b1_intersection(df, desert_tiers):
    """B1: Intersect PGC3 variants with desert coordinates."""
    print("\n" + "=" * 60)
    print("B1: Desert Intersection")
    print("=" * 60)
    
    results = {}
    for tier_name, desert_path in desert_tiers.items():
        regions = load_bed(desert_path)
        by_chr = organize_regions(regions)
        
        in_desert = []
        for _, row in df.iterrows():
            hit, name = variant_in_regions(row['chr'], row['pos'], by_chr)
            in_desert.append(hit)
        
        n_in = sum(in_desert)
        frac = n_in / len(df) * 100
        
        # Genome fraction
        genome_frac, total_bp = genome_desert_fraction(by_chr)
        
        print(f"\n  {tier_name}:")
        print(f"    Desert regions: {len(regions)}")
        print(f"    Total desert bp: {total_bp:,.0f} ({genome_frac*100:.2f}% of genome)")
        print(f"    PGC3 variants in deserts: {n_in}/{len(df)} ({frac:.2f}%)")
        print(f"    Expected by chance: {genome_frac*100:.2f}%")
        print(f"    Enrichment ratio: {frac/genome_frac/100:.2f}x")
        
        results[tier_name] = {
            'n_in': n_in, 'n_total': len(df),
            'genome_frac': genome_frac, 'mask': in_desert
        }
    
    return results


def step_b2_enrichment(df, desert_results):
    """B2: Statistical enrichment testing."""
    print("\n" + "=" * 60)
    print("B2: Enrichment Testing")
    print("=" * 60)
    
    enrichment_table = []
    
    for tier_name, res in desert_results.items():
        n_in = res['n_in']
        n_out = res['n_total'] - n_in
        genome_frac = res['genome_frac']
        
        # Expected counts
        n_expected = int(res['n_total'] * genome_frac)
        
        # Fisher's exact test (2x2: in/out × observed/expected)
        # Contingency: [[in_observed, out_observed], [in_expected, out_expected]]
        # Better: binomial test
        binom_result = stats.binomtest(n_in, res['n_total'], genome_frac, alternative='greater')
        binom_p = binom_result.pvalue
        
        enrichment = (n_in / res['n_total']) / genome_frac if genome_frac > 0 else float('inf')
        
        print(f"\n  {tier_name}:")
        print(f"    Observed in desert: {n_in}")
        print(f"    Expected by genomic fraction: {n_expected}")
        print(f"    Enrichment: {enrichment:.2f}x")
        print(f"    Binomial test (one-sided, enrichment): p = {binom_p:.2e}")
        print(f"    {'→ SIGNIFICANT enrichment' if binom_p < 0.05 else '→ Not significant'}")
        
        enrichment_table.append({
            'tier': tier_name,
            'n_variants_in_desert': n_in,
            'n_variants_total': res['n_total'],
            'observed_pct': n_in / res['n_total'] * 100,
            'genome_fraction_pct': genome_frac * 100,
            'enrichment_ratio': enrichment,
            'binomial_p': binom_p
        })
    
    pd.DataFrame(enrichment_table).to_csv(
        RESULTS_DIR / "B2_enrichment_tests.tsv", sep='\t', index=False
    )
    
    return enrichment_table


def step_b3_har_proximity(df):
    """B3: HAR proximity analysis."""
    print("\n" + "=" * 60)
    print("B3: HAR Proximity Analysis")
    print("=" * 60)
    
    if not HAR_BED.exists():
        print("  ⚠️  HAR BED file not found (hg19)")
        return
    
    hars = load_bed(HAR_BED)
    hars_by_chr = organize_regions(hars)
    
    print(f"  Total HARs (hg19): {len(hars)}")
    
    # Compute distance to nearest HAR for each variant
    distances = []
    nearest_hars = []
    in_har = 0
    
    for _, row in df.iterrows():
        dist, name = nearest_region_distance(row['chr'], row['pos'], hars_by_chr)
        distances.append(dist)
        nearest_hars.append(name)
        if dist == 0:
            in_har += 1
    
    df_har = df.copy()
    df_har['har_distance'] = distances
    df_har['nearest_har'] = nearest_hars
    
    # Summary
    dists = np.array(distances, dtype=float)
    dists_finite = dists[np.isfinite(dists)]
    
    print(f"\n  Variants inside a HAR: {in_har}")
    print(f"  Distance to nearest HAR (bp):")
    print(f"    Median: {np.median(dists_finite):,.0f}")
    print(f"    Mean:   {np.mean(dists_finite):,.0f}")
    print(f"    Within 10 kb: {(dists_finite < 10000).sum()}")
    print(f"    Within 100 kb: {(dists_finite < 100000).sum()}")
    print(f"    Within 500 kb: {(dists_finite < 500000).sum()}")
    
    # Correlation: age vs HAR distance
    valid = df_har[df_har['geva_age_years'].notna() & np.isfinite(df_har['har_distance'])].copy()
    if len(valid) > 100:
        r, p = stats.spearmanr(valid['geva_age_years'], valid['har_distance'])
        print(f"\n  Spearman (age vs HAR distance): rho={r:.4f}, p={p:.2e}")
    
    # Save
    df_har[['rsid', 'chr', 'pos', 'har_distance', 'nearest_har', 'geva_age_years']].to_csv(
        RESULTS_DIR / "B3_har_proximity.tsv", sep='\t', index=False
    )
    
    return df_har


def step_b4_combined(df, desert_results):
    """B4: Combined evolutionary signature."""
    print("\n" + "=" * 60)
    print("B4: Combined Evolutionary Signature")
    print("=" * 60)
    
    # Add desert membership
    valid = df[df['geva_age_years'].notna()].copy()
    
    tier2_mask = desert_results.get('Tier2_ge3of5', {}).get('mask', [False]*len(df))
    valid['in_desert_tier2'] = [tier2_mask[i] for i in valid.index]
    
    # Compare age of variants inside vs outside deserts
    in_ages = valid[valid['in_desert_tier2']]['geva_age_years']
    out_ages = valid[~valid['in_desert_tier2']]['geva_age_years']
    
    if len(in_ages) > 0 and len(out_ages) > 0:
        print(f"\n  Variants in Tier2 deserts: {len(in_ages)}")
        print(f"    Median age: {in_ages.median():,.0f} years")
        print(f"  Variants outside deserts: {len(out_ages)}")
        print(f"    Median age: {out_ages.median():,.0f} years")
        
        u_stat, u_p = stats.mannwhitneyu(in_ages, out_ages, alternative='two-sided')
        print(f"\n  Mann-Whitney U test (desert vs non-desert ages):")
        print(f"    U = {u_stat:,.0f}, p = {u_p:.2e}")
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(in_ages), len(out_ages)
        rbc = 1 - (2 * u_stat) / (n1 * n2)
        print(f"    Rank-biserial r = {rbc:.4f}")
    else:
        print("  ⚠️  Not enough variants in deserts for comparison")
    
    # Summary: variants with both ancient age AND desert location
    ancient_threshold = 500000  # 500K years
    ancient_and_desert = valid[valid['in_desert_tier2'] & (valid['geva_age_years'] > ancient_threshold)]
    print(f"\n  Ancient (>500K yr) AND in Tier2 desert: {len(ancient_and_desert)} variants")
    
    if len(ancient_and_desert) > 0:
        print(f"    Top loci:")
        for _, row in ancient_and_desert.head(10).iterrows():
            print(f"      {row.get('rsid', 'NA'):15s} chr{row['chr']}:{row['pos']:>11,} "
                  f"age={row['geva_age_years']:>10,.0f} yr PIP={row.get('pip', 'NA')}")


def main():
    print("=" * 60)
    print("EVOSCZ MODULE B: INTROGRESSION DESERT ENRICHMENT")
    print("=" * 60)
    
    df = pd.read_csv(MASTER_TABLE, sep='\t')
    df_auto = df[df['chr'].apply(lambda x: str(x).isdigit())].copy()
    df_auto['chr'] = df_auto['chr'].astype(int)
    print(f"PGC3 autosomal variants: {len(df_auto)}")
    
    # Desert tiers
    desert_tiers = {
        'Tier1_5of5': DESERT_DIR / "consensus_deserts_tier1_hg19.bed",
        'Tier2_ge3of5': DESERT_DIR / "consensus_deserts_tier2_hg19.bed",
        'Tier3_ge2of5': DESERT_DIR / "consensus_deserts_tier3_hg19.bed",
    }
    
    # B1: Intersection
    desert_results = step_b1_intersection(df_auto, desert_tiers)
    
    # B2: Enrichment
    enrichment = step_b2_enrichment(df_auto, desert_results)
    
    # B3: HAR proximity
    df_har = step_b3_har_proximity(df_auto)
    
    # B4: Combined
    step_b4_combined(df_auto, desert_results)
    
    print(f"\nAll Module B results saved to: {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
