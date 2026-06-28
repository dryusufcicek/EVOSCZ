#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Step U5: Build Master Annotation Table
=========================================================
Merges PGC3 credible set variants with:
  1. GEVA allele ages (matched by chr + pos + alleles)
  2. B-statistic (background selection, nearest interval)
  3. Ancestral allele state (from Ensembl FASTA)

Output: data/processed/pgc3_annotated_master.tsv

NOTE: Position-only matching for GEVA can produce duplicates
when the same position has multiple alt alleles in the GEVA atlas.
We resolve this by matching on ref+alt alleles, then keeping the
best-quality joint estimate.
"""

import pandas as pd
import numpy as np
import gzip
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
PGC3_MASTER = PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv"
GEVA_AGES   = PROJECT_ROOT / "data/processed/pgc3_geva_ages.tsv"
BSTAT_FILE  = PROJECT_ROOT / "data/raw/annotations/b_statistic/bkgd/data/hg19/bkgd_hg19.bed.gz"
ANCESTRAL_DIR = PROJECT_ROOT / "data/raw/annotations/ancestral_alleles/homo_sapiens_ancestor_GRCh37_e71"

# Output
OUTPUT = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"


def load_and_match_geva():
    """Match GEVA ages to PGC3 variants by chr + pos + alleles."""
    print("[1/3] Matching GEVA allele ages...")
    
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    geva = pd.read_csv(GEVA_AGES, sep='\t')
    
    # Clean GEVA columns
    geva['Position'] = pd.to_numeric(geva['Position'], errors='coerce').astype('Int64')
    geva['pgc3_chr'] = pd.to_numeric(geva['pgc3_chr'], errors='coerce').astype('Int64')
    
    # For numerical age columns
    age_cols = ['AgeMedian_Jnt', 'AgeMean_Jnt', 'AgeCI95Lower_Jnt', 'AgeCI95Upper_Jnt', 'QualScore_Jnt']
    for col in age_cols:
        if col in geva.columns:
            geva[col] = pd.to_numeric(geva[col], errors='coerce')
    
    # Strategy: match by chr + pos first, then try allele match
    pgc3_auto = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())].copy()
    pgc3_auto['chr'] = pgc3_auto['chr'].astype(int)
    
    # Build lookup: (chr, pos) → list of GEVA rows
    geva_lookup = defaultdict(list)
    for _, row in geva.iterrows():
        key = (row['pgc3_chr'], row['Position'])
        geva_lookup[key].append(row)
    
    # Match
    matched = []
    unmatched = 0
    for _, prow in pgc3_auto.iterrows():
        key = (prow['chr'], prow['pos'])
        candidates = geva_lookup.get(key, [])
        
        if not candidates:
            matched.append({
                'rsid': prow.get('rsid', ''),
                'geva_age_median': np.nan,
                'geva_age_mean': np.nan,
                'geva_age_ci_lower': np.nan,
                'geva_age_ci_upper': np.nan,
                'geva_quality': np.nan,
                'geva_match_type': 'no_match'
            })
            unmatched += 1
            continue
        
        # Try exact allele match (A1=ref, A2=alt or vice versa)
        best = None
        for c in candidates:
            ref_match = (str(c.get('AlleleRef', '')).upper() == str(prow.get('effect_allele', '')).upper() and 
                         str(c.get('AlleleAlt', '')).upper() == str(prow.get('other_allele', '')).upper())
            flip_match = (str(c.get('AlleleRef', '')).upper() == str(prow.get('other_allele', '')).upper() and 
                          str(c.get('AlleleAlt', '')).upper() == str(prow.get('effect_allele', '')).upper())
            if ref_match or flip_match:
                if best is None or c.get('QualScore_Jnt', 0) > best.get('QualScore_Jnt', 0):
                    best = c
        
        if best is not None:
            match_type = 'allele_exact'
        else:
            # Fallback: take highest quality score among candidates
            best = max(candidates, key=lambda c: c.get('QualScore_Jnt', 0) if pd.notna(c.get('QualScore_Jnt')) else 0)
            match_type = 'position_only'
        
        matched.append({
            'rsid': prow.get('rsid', ''),
            'geva_age_median': best.get('AgeMedian_Jnt'),
            'geva_age_mean': best.get('AgeMean_Jnt'),
            'geva_age_ci_lower': best.get('AgeCI95Lower_Jnt'),
            'geva_age_ci_upper': best.get('AgeCI95Upper_Jnt'),
            'geva_quality': best.get('QualScore_Jnt'),
            'geva_match_type': match_type
        })
    
    match_df = pd.DataFrame(matched)
    
    # Stats
    n_exact = (match_df['geva_match_type'] == 'allele_exact').sum()
    n_pos = (match_df['geva_match_type'] == 'position_only').sum()
    n_no = (match_df['geva_match_type'] == 'no_match').sum()
    
    print(f"  Allele-exact match: {n_exact}")
    print(f"  Position-only match: {n_pos}")
    print(f"  No match: {n_no}")
    print(f"  Coverage: {(n_exact + n_pos) / len(match_df) * 100:.1f}%")
    
    return match_df


def load_bstat_for_positions(pgc3):
    """Look up B-statistic values for PGC3 positions using interval search."""
    print("\n[2/3] Annotating B-statistic values...")
    
    pgc3_auto = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())].copy()
    pgc3_auto['chr'] = pgc3_auto['chr'].astype(int)
    
    # Load B-stat BED (chr, start, end, ., bstat)
    bstat_intervals = defaultdict(list)
    with gzip.open(BSTAT_FILE, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom = fields[0].replace('chr', '')
            try:
                chrom_int = int(chrom)
            except ValueError:
                continue
            start = int(fields[1])
            end = int(fields[2])
            bval = float(fields[4])
            bstat_intervals[chrom_int].append((start, end, bval))
    
    # Sort intervals
    for chrom in bstat_intervals:
        bstat_intervals[chrom].sort()
    
    print(f"  Loaded {sum(len(v) for v in bstat_intervals.values())} B-stat intervals")
    
    # Binary search for B-value at each position
    import bisect
    
    bstat_values = []
    for _, row in pgc3_auto.iterrows():
        chrom = row['chr']
        pos = row['pos']
        intervals = bstat_intervals.get(chrom, [])
        
        # Binary search: find interval containing pos
        idx = bisect.bisect_right(intervals, (pos,)) - 1
        if idx >= 0 and intervals[idx][0] <= pos < intervals[idx][1]:
            bstat_values.append(intervals[idx][2])
        else:
            bstat_values.append(np.nan)
    
    found = sum(1 for v in bstat_values if not np.isnan(v))
    print(f"  B-stat annotated: {found}/{len(bstat_values)} ({found/len(bstat_values)*100:.1f}%)")
    
    return bstat_values


def get_ancestral_allele(chrom, pos):
    """Get ancestral allele from Ensembl FASTA at given position."""
    chrom_str = str(chrom)
    fa_file = ANCESTRAL_DIR / f"homo_sapiens_ancestor_{chrom_str}.fa"
    
    if not fa_file.exists():
        return '.'
    
    # We'll handle this in batch mode below
    return '.'


def load_ancestral_alleles(pgc3):
    """Batch-extract ancestral alleles from Ensembl FASTA files."""
    print("\n[3/3] Extracting ancestral alleles from Ensembl FASTA...")
    
    pgc3_auto = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())].copy()
    pgc3_auto['chr'] = pgc3_auto['chr'].astype(int)
    
    if not ANCESTRAL_DIR.exists():
        print(f"  ⚠️  Ancestral allele directory not found: {ANCESTRAL_DIR}")
        return ['.'] * len(pgc3_auto)
    
    # Group positions by chromosome
    chr_positions = defaultdict(list)
    for idx, row in pgc3_auto.iterrows():
        chr_positions[row['chr']].append((idx, row['pos']))
    
    ancestral = {}
    for chrom in sorted(chr_positions.keys()):
        fa_file = ANCESTRAL_DIR / f"homo_sapiens_ancestor_{chrom}.fa"
        if not fa_file.exists():
            print(f"  chr{chrom}: FASTA not found")
            for idx, pos in chr_positions[chrom]:
                ancestral[idx] = '.'
            continue
        
        # Read FASTA (simple sequential read, positions are 1-based)
        # Build sequence in memory
        seq = []
        with open(fa_file) as f:
            for line in f:
                if line.startswith('>'):
                    continue
                seq.append(line.strip().upper())
        full_seq = ''.join(seq)
        
        found = 0
        for idx, pos in chr_positions[chrom]:
            if 0 < pos <= len(full_seq):
                aa = full_seq[pos - 1]  # 1-based to 0-based
                if aa in 'ACGT':
                    ancestral[idx] = aa
                    found += 1
                else:
                    ancestral[idx] = '.'  # ambiguous or gap
            else:
                ancestral[idx] = '.'
        
        print(f"  chr{chrom}: {found}/{len(chr_positions[chrom])} ancestral alleles found")
    
    # Return in original order
    result = [ancestral.get(idx, '.') for idx in pgc3_auto.index]
    total_found = sum(1 for v in result if v != '.')
    print(f"  Total: {total_found}/{len(result)} ({total_found/len(result)*100:.1f}%)")
    
    return result


def main():
    print("=" * 70)
    print("EVOSCZ Step U5: Building Master Annotation Table")
    print("=" * 70)
    print()
    
    # Load PGC3
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    pgc3_auto = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())].copy()
    pgc3_auto['chr'] = pgc3_auto['chr'].astype(int)
    print(f"PGC3 autosomal variants: {len(pgc3_auto)}\n")
    
    # 1. GEVA ages
    geva_df = load_and_match_geva()
    
    # 2. B-statistic
    bstat_values = load_bstat_for_positions(pgc3)
    
    # 3. Ancestral alleles
    aa_values = load_ancestral_alleles(pgc3)
    
    # Merge into master table
    print("\n[Merge] Building annotated master table...")
    result = pgc3_auto.copy()
    
    # Add GEVA
    result = result.merge(geva_df, on='rsid', how='left')
    
    # Add B-stat
    result['b_statistic'] = bstat_values
    
    # Add ancestral allele
    result['ancestral_allele'] = aa_values
    
    # Compute derived allele frequency (DAF)
    # If A1 == ancestral → A2 is derived → DAF = freq of A2 = 1 - freq_A1
    # If A2 == ancestral → A1 is derived → DAF = freq_A1
    def compute_daf(row):
        aa = str(row.get('ancestral_allele', '.')).upper()
        a1 = str(row.get('effect_allele', '')).upper()
        a2 = str(row.get('other_allele', '')).upper()
        freq = row.get('maf', np.nan)
        
        if aa == '.' or pd.isna(freq):
            return np.nan
        
        if aa == a1:
            return 1 - freq  # A2 is derived
        elif aa == a2:
            return freq  # A1 is derived
        else:
            return np.nan  # ancestral doesn't match either allele
    
    result['daf'] = result.apply(compute_daf, axis=1)
    
    # Convert age from generations to years
    result['geva_age_years'] = result['geva_age_median'] * 29
    
    # Save
    result.to_csv(OUTPUT, sep='\t', index=False)
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"MASTER ANNOTATION TABLE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output: {OUTPUT}")
    print(f"  Total variants: {len(result)}")
    print(f"  With GEVA age: {result['geva_age_median'].notna().sum()} ({result['geva_age_median'].notna().mean()*100:.1f}%)")
    print(f"  With B-stat: {result['b_statistic'].notna().sum()} ({result['b_statistic'].notna().mean()*100:.1f}%)")
    print(f"  With ancestral allele: {(result['ancestral_allele'] != '.').sum()} ({(result['ancestral_allele'] != '.').mean()*100:.1f}%)")
    print(f"  With DAF: {result['daf'].notna().sum()} ({result['daf'].notna().mean()*100:.1f}%)")
    
    if result['geva_age_years'].notna().any():
        ages = result['geva_age_years'].dropna()
        print(f"\n  Age distribution (years):")
        print(f"    Median: {ages.median():,.0f}")
        print(f"    Mean:   {ages.mean():,.0f}")
        print(f"    <29,000 yr (recent): {(ages < 29000).sum()} ({(ages < 29000).mean()*100:.1f}%)")
        print(f"    >290,000 yr (ancient): {(ages > 290000).sum()} ({(ages > 290000).mean()*100:.1f}%)")
    
    if result['daf'].notna().any():
        daf = result['daf'].dropna()
        print(f"\n  DAF distribution:")
        print(f"    Median: {daf.median():.3f}")
        print(f"    Mean:   {daf.mean():.3f}")
        print(f"    Low DAF (<0.1): {(daf < 0.1).sum()} ({(daf < 0.1).mean()*100:.1f}%)")
        print(f"    High DAF (>0.5): {(daf > 0.5).sum()} ({(daf > 0.5).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
