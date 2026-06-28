#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Step U4a: Acquire GEVA Allele Ages (Smart Strategy)
=====================================================================
Instead of downloading all 15 GB of GEVA data, we query allele ages
ONLY for PGC3 credible set variants (~20,766 variants).

Strategy:
  1. First try the GEVA bulk download API per-chromosome
  2. Extract only rows matching our PGC3 variant positions
  3. This reduces from ~15 GB → ~50 MB of data we actually need

GEVA bulk download format (per chr):
  https://human.genome.dating/bulk/atlas.chr{N}.csv.gz   ← VERIFIED 2026-04-10
  Note: dots between 'atlas' and 'chr', NOT underscores

File format (verified from chr22 download):
  - 3 comment lines starting with '## '
  - CSV (comma + space separated)
  - Columns: VariantID, Chromosome, Position, AlleleRef, AlleleAlt, AlleleAnc,
             DataSource, NumConcordant, NumDiscordant,
             AgeMode_Mut, AgeMean_Mut, AgeMedian_Mut, AgeCI95Lower_Mut, AgeCI95Upper_Mut, QualScore_Mut,
             AgeMode_Rec, AgeMean_Rec, AgeMedian_Rec, AgeCI95Lower_Rec, AgeCI95Upper_Rec, QualScore_Rec,
             AgeMode_Jnt, AgeMean_Jnt, AgeMedian_Jnt, AgeCI95Lower_Jnt, AgeCI95Upper_Jnt, QualScore_Jnt
  - Ages are in GENERATIONS (1 gen ≈ 29 years)
  - _Mut = mutation clock, _Rec = recombination clock, _Jnt = joint estimate (preferred)
  - Data source: TGP (1000 Genomes Project)
"""

import pandas as pd
import subprocess
import os
import sys
import gzip
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGC3_MASTER = PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data/raw/annotations/allele_ages/geva"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_OUTPUT = PROJECT_ROOT / "data/processed/pgc3_geva_ages.tsv"

# VERIFIED 2026-04-10: correct format uses dots, not underscores
GEVA_BASE_URL = "https://human.genome.dating/bulk/atlas.chr{chr}.csv.gz"


def get_pgc3_positions_by_chr():
    """Load PGC3 variants and group by chromosome."""
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    # Exclude X chromosome for GEVA (autosomal only)
    pgc3 = pgc3[pgc3['chr'].apply(lambda x: str(x).isdigit())]
    pgc3['chr'] = pgc3['chr'].astype(int)
    
    chr_positions = {}
    for chrom, group in pgc3.groupby('chr'):
        chr_positions[chrom] = set(group['pos'].values)
    
    return chr_positions, pgc3


def download_and_filter_geva_chr(chrom, target_positions):
    """
    Download GEVA atlas for one chromosome, extract only PGC3 positions.
    Streams the gzipped CSV to avoid storing the full file.
    """
    url = GEVA_BASE_URL.format(chr=chrom)
    local_gz = OUTPUT_DIR / f"atlas_chr{chrom}.csv.gz"
    filtered_file = OUTPUT_DIR / f"pgc3_ages_chr{chrom}.tsv"
    
    if filtered_file.exists():
        print(f"  chr{chrom}: Already filtered, skipping download")
        return pd.read_csv(filtered_file, sep='\t')
    
    # SAFETY CHECK: probe the URL before downloading
    # The GEVA bulk URL format is assumed — verify it returns actual data
    probe = subprocess.run(
        ['curl', '-sI', url], capture_output=True, text=True, timeout=30
    )
    content_type = ""
    for line in probe.stdout.split('\n'):
        if line.lower().startswith('content-type:'):
            content_type = line.strip()
            break
    
    if 'text/html' in content_type.lower() or '404' in probe.stdout[:20]:
        print(f"  chr{chrom}: ⚠️  URL returned '{content_type}' — NOT a CSV/gzip file")
        print(f"    URL: {url}")
        print(f"    The GEVA bulk download URL format may have changed.")
        print(f"    Please verify at: https://human.genome.dating/download/index")
        return pd.DataFrame()
    
    # Download
    print(f"  chr{chrom}: Downloading from GEVA ({len(target_positions)} targets)...")
    try:
        subprocess.run(
            ['curl', '-sL', '-o', str(local_gz), url],
            check=True, timeout=600
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  chr{chrom}: Download failed: {e}")
        return pd.DataFrame()
    
    # Stream-filter: read gzipped CSV, keep only matching positions
    # GEVA format: 3 comment lines (##), then CSV header, then data
    # Position is column index 2 (0=VariantID, 1=Chromosome, 2=Position)
    print(f"  chr{chrom}: Filtering to PGC3 positions...")
    matched_rows = []
    header = None
    
    with gzip.open(local_gz, 'rt') as f:
        for line in f:
            stripped = line.strip()
            # Skip comment lines
            if stripped.startswith('##'):
                continue
            # Parse CSV (comma + optional space separated)
            fields = [f.strip() for f in stripped.split(',')]
            if header is None:
                header = fields
                continue
            try:
                pos = int(fields[2])  # Position is 3rd column (index 2)
                if pos in target_positions:
                    matched_rows.append(fields)
            except (ValueError, IndexError):
                continue
    
    if matched_rows and header:
        df = pd.DataFrame(matched_rows, columns=header)
        df.to_csv(filtered_file, sep='\t', index=False)
        print(f"  chr{chrom}: Found {len(df)}/{len(target_positions)} variants with GEVA ages")
    else:
        df = pd.DataFrame()
        print(f"  chr{chrom}: No matches found")
    
    # Remove the large bulk download to save space
    if local_gz.exists():
        local_gz.unlink()
        print(f"  chr{chrom}: Cleaned up bulk file")
    
    return df


def main():
    print("=" * 60)
    print("EVOSCZ Step U4a: Acquiring GEVA Allele Ages")
    print("=" * 60)
    
    chr_positions, pgc3 = get_pgc3_positions_by_chr()
    total_targets = sum(len(v) for v in chr_positions.values())
    print(f"\nTarget: {total_targets} PGC3 autosomal variants across {len(chr_positions)} chromosomes\n")
    
    all_ages = []
    for chrom in sorted(chr_positions.keys()):
        df = download_and_filter_geva_chr(chrom, chr_positions[chrom])
        if len(df) > 0:
            df['chr'] = chrom
            all_ages.append(df)
    
    if all_ages:
        merged = pd.concat(all_ages, ignore_index=True)
        merged.to_csv(MERGED_OUTPUT, sep='\t', index=False)
        
        coverage = len(merged) / total_targets * 100
        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total PGC3 variants queried: {total_targets}")
        print(f"  GEVA ages found:             {len(merged)}")
        print(f"  Coverage:                    {coverage:.1f}%")
        print(f"  Output: {MERGED_OUTPUT}")
    else:
        print("\nWARNING: No GEVA ages retrieved. Check network connectivity.")


if __name__ == "__main__":
    main()
