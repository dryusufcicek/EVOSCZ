#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Step U4a: Filter GEVA Allele Ages to PGC3 Variants
=====================================================================
GEVA bulk data already downloaded by user to:
  ~/Downloads/Human Genome Dating/atlas.chr{N}.csv.gz

This script:
  1. Reads PGC3 master variant table (20,766 variants)
  2. For each chromosome, streams through the GEVA atlas
  3. Extracts only rows matching PGC3 variant positions
  4. Saves filtered output (~50 MB instead of 4.8 GB)

GEVA Format (VERIFIED 2026-04-10):
  - 3 comment lines starting with '## '
  - CSV (comma + space separated)
  - Position = column index 2
  - Use _Jnt (joint) age estimates
  - Ages in generations (1 gen ≈ 29 years)
"""

import pandas as pd
import gzip
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGC3_MASTER = PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv"

# Symlinked GEVA data
GEVA_DIR = PROJECT_ROOT / "data/raw/annotations/allele_ages/geva/atlas_bulk"
OUTPUT_DIR = PROJECT_ROOT / "data/raw/annotations/allele_ages/geva"
MERGED_OUTPUT = PROJECT_ROOT / "data/processed/pgc3_geva_ages.tsv"


def get_pgc3_positions_by_chr():
    """Load PGC3 variants and group by chromosome."""
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    pgc3 = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())]
    pgc3['chr'] = pgc3['chr'].astype(int)
    
    chr_positions = {}
    for chrom, group in pgc3.groupby('chr'):
        chr_positions[chrom] = set(group['pos'].values)
    
    return chr_positions, pgc3


def filter_geva_chr(chrom, target_positions):
    """Filter GEVA atlas for one chromosome to PGC3 positions only."""
    gz_file = GEVA_DIR / f"atlas.chr{chrom}.csv.gz"
    filtered_file = OUTPUT_DIR / f"pgc3_ages_chr{chrom}.tsv"
    
    if filtered_file.exists():
        print(f"  chr{chrom}: Already filtered, loading cached")
        return pd.read_csv(filtered_file, sep='\t')
    
    if not gz_file.exists():
        print(f"  chr{chrom}: ❌ atlas file not found at {gz_file}")
        return pd.DataFrame()
    
    print(f"  chr{chrom}: Filtering ({len(target_positions)} targets)...", end=" ")
    matched_rows = []
    header = None
    total_lines = 0
    
    with gzip.open(gz_file, 'rt') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('##'):
                continue
            fields = [field.strip() for field in stripped.split(',')]
            if header is None:
                header = fields
                continue
            total_lines += 1
            try:
                pos = int(fields[2])  # Position = column 2
                if pos in target_positions:
                    matched_rows.append(fields)
            except (ValueError, IndexError):
                continue
    
    if matched_rows and header:
        df = pd.DataFrame(matched_rows, columns=header)
        df.to_csv(filtered_file, sep='\t', index=False)
        print(f"✅ {len(df)}/{len(target_positions)} found (scanned {total_lines:,} variants)")
    else:
        df = pd.DataFrame()
        print(f"⚠️  0 matches (scanned {total_lines:,})")
    
    return df


def main():
    print("=" * 70)
    print("EVOSCZ Step U4a: Filtering GEVA Allele Ages to PGC3 Variants")
    print("=" * 70)
    print(f"GEVA source: {GEVA_DIR}")
    print(f"PGC3 master: {PGC3_MASTER}")
    print()
    
    chr_positions, pgc3 = get_pgc3_positions_by_chr()
    total_targets = sum(len(v) for v in chr_positions.values())
    print(f"Target: {total_targets} PGC3 autosomal variants across {len(chr_positions)} chromosomes\n")
    
    all_ages = []
    total_found = 0
    for chrom in sorted(chr_positions.keys()):
        df = filter_geva_chr(chrom, chr_positions[chrom])
        if len(df) > 0:
            df.insert(0, 'pgc3_chr', chrom)
            all_ages.append(df)
            total_found += len(df)
    
    if all_ages:
        merged = pd.concat(all_ages, ignore_index=True)
        merged.to_csv(MERGED_OUTPUT, sep='\t', index=False)
        
        coverage = total_found / total_targets * 100
        print(f"\n{'=' * 70}")
        print(f"SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total PGC3 variants queried:  {total_targets}")
        print(f"  GEVA ages found:              {total_found}")
        print(f"  Coverage:                     {coverage:.1f}%")
        print(f"  Output: {MERGED_OUTPUT}")
        print(f"  Output size: {MERGED_OUTPUT.stat().st_size / 1e6:.1f} MB")
        
        # Quick stats on age distribution
        age_col = 'AgeMedian_Jnt'
        if age_col in merged.columns:
            ages = pd.to_numeric(merged[age_col], errors='coerce')
            print(f"\n  Age distribution (generations):")
            print(f"    Median: {ages.median():.0f}")
            print(f"    Mean:   {ages.mean():.0f}")
            print(f"    Min:    {ages.min():.0f}")
            print(f"    Max:    {ages.max():.0f}")
            print(f"    <1000 gen: {(ages < 1000).sum()} ({(ages < 1000).mean()*100:.1f}%)")
            print(f"    >10000 gen: {(ages > 10000).sum()} ({(ages > 10000).mean()*100:.1f}%)")
    else:
        print("\n❌ WARNING: No GEVA ages retrieved. Check file paths.")


if __name__ == "__main__":
    main()
