#!/usr/bin/env python3
"""
EVOSCZ Upstream U3: Generate MAF + LD-Score Matched Control Variants
=====================================================================
Creates a matched control set from LDSC EUR reference SNPs for
enrichment testing (Module B desert enrichment, Module D selection).

Matching dimensions (2D):
  1. MAF (±2%) — controls frequency-dependent statistics (age, nSL, DAF)
  2. L2 LD score (±20% relative) — controls for LD architecture, partially
     captures B-statistic and gene density effects

For each PGC3 variant, we sample up to 100 matched controls from the
LDSC reference panel (~1.29M HapMap3 SNPs). Controls are excluded if
they share an rsID with any PGC3 credible set variant.

Limitations documented:
  - 2D matching (MAF + L2) is less comprehensive than the 6D matching
    in the original plan (which also included B-stat, TSS distance,
    GC content, recombination rate). Results should be interpreted
    with this caveat.
  - LDSC reference uses HapMap3 SNPs, which are biased toward common,
    well-characterized variants. Rare variant controls may be sparse.
  - No explicit LD pruning against PGC3 variants (would require VCF data).
    We exclude exact rsID matches only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGC3_MASTER = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"
LDSC_DIR = PROJECT_ROOT / "data/raw/annotations/ldsc_ref/eur_w_ld_chr"
OUTPUT = PROJECT_ROOT / "data/processed/matched_controls.tsv.gz"

N_CONTROLS = 100  # per PGC3 variant
MAF_WINDOW = 0.02  # ±2%
L2_WINDOW = 0.20   # ±20% relative
SEED = 42

# FIX-D-U3-1: prior code allowed unlimited control re-use across PGC3 variants
# in the same locus (same MAF/L2 stratum), producing pseudo-replicated controls
# that downstream aggregate tests (e.g. B2b Fisher) treated as independent —
# inflating power. New: cap each control SNP at REUSE_CAP appearances across
# the entire PGC3 matching, enforcing approximate stratum-level no-replacement.
REUSE_CAP = 5


def load_ldsc_reference():
    """Load all LDSC reference SNPs across 22 autosomes."""
    frames = []
    for chrom in range(1, 23):
        path = LDSC_DIR / f"{chrom}.l2.ldscore.gz"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping chr{chrom}")
            continue
        df = pd.read_csv(path, sep='\t', compression='gzip')
        frames.append(df)
    ref = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(ref):,} LDSC reference SNPs across {len(frames)} chromosomes")
    return ref


def main():
    print("=" * 60)
    print("EVOSCZ U3: Matched Control Generation (MAF + L2)")
    print("=" * 60)

    # Load PGC3 variants
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    print(f"PGC3 variants: {len(pgc3):,}")

    # Load LDSC reference
    print("Loading LDSC reference panel...")
    ref = load_ldsc_reference()

    # Exclude PGC3 variants from control pool
    pgc3_rsids = set(pgc3['rsid'].dropna().unique())
    ref_clean = ref[~ref['SNP'].isin(pgc3_rsids)].copy()
    n_excluded = len(ref) - len(ref_clean)
    print(f"  Excluded {n_excluded} PGC3 overlapping SNPs from control pool")
    print(f"  Control pool: {len(ref_clean):,} SNPs")

    # Get L2 values for PGC3 variants by positional matching
    # Merge PGC3 with LDSC ref on chr+position
    pgc3_l2 = pgc3.merge(
        ref[['CHR', 'BP', 'L2', 'SNP']].rename(columns={'CHR': 'chr', 'BP': 'pos', 'SNP': 'ldsc_snp'}),
        on=['chr', 'pos'], how='left'
    )
    # Also try rsID matching for those that didn't match by position
    pgc3_no_l2 = pgc3_l2[pgc3_l2['L2'].isna()]
    if len(pgc3_no_l2) > 0:
        rsid_match = pgc3_no_l2[['rsid']].merge(
            ref[['SNP', 'L2']].rename(columns={'SNP': 'rsid'}),
            on='rsid', how='left', suffixes=('', '_rsid')
        )
        pgc3_l2.loc[pgc3_l2['L2'].isna(), 'L2'] = rsid_match['L2'].values

    n_with_l2 = pgc3_l2['L2'].notna().sum()
    print(f"  PGC3 variants with L2: {n_with_l2:,}/{len(pgc3):,} ({n_with_l2/len(pgc3):.1%})")

    # For variants without L2, use the genome-wide median
    l2_median = ref_clean['L2'].median()
    pgc3_l2['L2'] = pgc3_l2['L2'].fillna(l2_median)

    # Generate matched controls
    print(f"\nGenerating {N_CONTROLS} matched controls per PGC3 variant...")
    rng = np.random.default_rng(SEED)

    # Pre-sort reference for faster lookup
    ref_arr = ref_clean[['SNP', 'CHR', 'BP', 'MAF', 'L2']].values
    ref_maf = ref_arr[:, 3].astype(float)
    ref_l2 = ref_arr[:, 4].astype(float)

    # FIX-D-U3-1: usage tracker for re-use cap
    usage = np.zeros(len(ref_arr), dtype=np.int32)

    controls_list = []
    n_matched = 0
    n_sparse = 0
    n_capped_out = 0

    for idx, row in pgc3_l2.iterrows():
        target_maf = row['maf']
        target_l2 = row['L2']

        if pd.isna(target_maf):
            continue

        # MAF window
        maf_lo = target_maf - MAF_WINDOW
        maf_hi = target_maf + MAF_WINDOW
        # L2 window (relative)
        l2_lo = target_l2 * (1 - L2_WINDOW)
        l2_hi = target_l2 * (1 + L2_WINDOW)

        # Strict bin: MAF + L2 + below re-use cap
        mask = (ref_maf >= maf_lo) & (ref_maf <= maf_hi) & \
               (ref_l2 >= l2_lo) & (ref_l2 <= l2_hi) & \
               (usage < REUSE_CAP)
        candidates_idx = np.where(mask)[0]

        # Stage 2: relax L2 if no candidates
        if len(candidates_idx) == 0:
            mask = (ref_maf >= maf_lo) & (ref_maf <= maf_hi) & (usage < REUSE_CAP)
            candidates_idx = np.where(mask)[0]
        # Stage 3: relax cap (last resort)
        if len(candidates_idx) == 0:
            mask = (ref_maf >= maf_lo) & (ref_maf <= maf_hi)
            candidates_idx = np.where(mask)[0]
            n_capped_out += 1
            if len(candidates_idx) == 0:
                continue

        # Sample without replacement within this PGC3 variant
        n_sample = min(N_CONTROLS, len(candidates_idx))
        if n_sample < N_CONTROLS:
            n_sparse += 1
        chosen = rng.choice(candidates_idx, size=n_sample, replace=False)
        # Increment usage
        for ci in chosen:
            usage[ci] += 1
            controls_list.append({
                'pgc3_rsid': row['rsid'],
                'pgc3_credset': row['credible_set_id'],
                'control_snp': ref_arr[ci, 0],
                'control_chr': int(ref_arr[ci, 1]),
                'control_pos': int(ref_arr[ci, 2]),
                'control_maf': float(ref_arr[ci, 3]),
                'control_l2': float(ref_arr[ci, 4]),
                'pgc3_maf': target_maf,
                'pgc3_l2': target_l2
            })
        n_matched += 1

        if (idx + 1) % 5000 == 0:
            print(f"  ... processed {idx+1}/{len(pgc3_l2)} variants")

    controls_df = pd.DataFrame(controls_list)
    print(f"\nMatched controls generated:")
    print(f"  PGC3 variants matched: {n_matched:,}")
    print(f"  Variants with < {N_CONTROLS} controls: {n_sparse:,}")
    print(f"  Variants needing re-use cap relaxation: {n_capped_out:,}")
    print(f"  Total control entries: {len(controls_df):,}")
    if len(controls_df):
        usage_dist = controls_df['control_snp'].value_counts()
        print(f"  Unique control SNPs: {len(usage_dist):,}")
        print(f"  Mean uses per unique control: {usage_dist.mean():.2f}, "
              f"max: {usage_dist.max()} (cap = {REUSE_CAP})")

    # Quality check: MAF and L2 distribution comparison
    print(f"\n--- Matching Quality ---")
    print(f"  PGC3 MAF:    mean={pgc3_l2['maf'].mean():.4f}, median={pgc3_l2['maf'].median():.4f}")
    print(f"  Control MAF: mean={controls_df['control_maf'].mean():.4f}, median={controls_df['control_maf'].median():.4f}")
    print(f"  PGC3 L2:     mean={pgc3_l2['L2'].mean():.2f}, median={pgc3_l2['L2'].median():.2f}")
    print(f"  Control L2:  mean={controls_df['control_l2'].mean():.2f}, median={controls_df['control_l2'].median():.2f}")

    # Save
    controls_df.to_csv(OUTPUT, sep='\t', index=False, compression='gzip')
    print(f"\nSaved to: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
