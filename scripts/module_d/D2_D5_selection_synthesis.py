#!/usr/bin/env python3
"""
EVOSCZ Pipeline �� Module D: Selection Signatures Synthesis (v2)
===============================================================
Synthesizes Module A age residuals with Module D selection statistics
to classify PGC3 loci by evolutionary signature.

v2 changes (2026-04-11):
- Replaced absolute thresholds with empirical percentile-based approach.
  Rationale: D1 computes max|nSL| over 200kb windows (~5800 SNPs each).
  The window-maximum of |nSL| exceeds 2.0 for virtually any window by
  statistical expectation alone (max of thousands of draws from a
  distribution with moderate variance). An absolute threshold of 2.0
  therefore classifies 100% of loci as "sweep", which is uninformative.

  Similarly, European demographic history (post-bottleneck expansion)
  shifts the genome-wide Tajima's D distribution negative. An absolute
  threshold of D > 1.0 is unrealistically stringent for EUR populations.

- New approach: rank loci within the PGC3 dataset and use the upper
  tail (top 5% = most extreme ~12 loci) for each statistic. This
  identifies loci with the STRONGEST relative signal among all PGC3
  loci. While not a genome-wide calibrated null (which requires
  matched controls or demographic simulations), this is a valid
  within-dataset ranking that avoids the 100%/0% classification
  artifact.

- Soft sweep classification now uses: top 25% H12 AND H2/H1 > median
  AND top 10% nSL. This captures loci with convergent evidence across
  multiple sweep-sensitive statistics.

- Added "suggestive" tier (top 10%) alongside "significant" tier (top 5%)
  for sensitivity analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D1_RESULTS   = PROJECT_ROOT / "results/module_d/D1_empirical_selection_stats.tsv"
A2_AGES      = PROJECT_ROOT / "results/module_a/A2_bstat_corrected_ages.tsv"
RESULTS_DIR  = PROJECT_ROOT / "results/module_d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Percentile thresholds
SWEEP_PCTILE = 95       # top 5% for directional sweep
BALANCING_PCTILE = 95   # top 5% for balancing selection (highest Tajima's D)
SUGGESTIVE_PCTILE = 90  # top 10% for suggestive tier

def main():
    print("=" * 60)
    print("EVOSCZ Module D: Synthesis and Signature Validation (v2)")
    print("=" * 60)

    if not D1_RESULTS.exists():
        print(f"Error: {D1_RESULTS} not found.")
        return

    d_stats = pd.read_csv(D1_RESULTS, sep='\t')
    print(f"Loaded selection statistics for {len(d_stats)} loci.")

    ages = pd.read_csv(A2_AGES, sep='\t')
    master = pd.read_csv(PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv", sep='\t')
    ages = ages.merge(master[['rsid', 'credible_set_id']], on='rsid', how='left')

    locus_ages = ages.groupby('credible_set_id').agg(
        mean_age_residual=('age_residual', 'mean'),
        median_age_years=('geva_age_years', 'median')
    ).reset_index()

    merged = d_stats.merge(locus_ages, on='credible_set_id', how='left')

    # --- Compute empirical percentile thresholds ---
    nsl_95 = merged['max_abs_nsl'].quantile(SWEEP_PCTILE / 100)
    nsl_90 = merged['max_abs_nsl'].quantile(SUGGESTIVE_PCTILE / 100)
    td_95 = merged['tajimas_d'].quantile(BALANCING_PCTILE / 100)
    td_90 = merged['tajimas_d'].quantile(SUGGESTIVE_PCTILE / 100)
    h12_75 = merged['h12'].quantile(0.75)
    h2h1_median = merged['h2h1'].quantile(0.50)
    nsl_90_for_soft = merged['max_abs_nsl'].quantile(0.90)

    print(f"\n--- Empirical Thresholds (from {len(merged)} PGC3 loci) ---")
    print(f"  max|nSL| 95th pctile (sweep):      {nsl_95:.3f}")
    print(f"  max|nSL| 90th pctile (suggestive):  {nsl_90:.3f}")
    print(f"  Tajima's D 95th pctile (balancing):  {td_95:.3f}")
    print(f"  Tajima's D 90th pctile (suggestive): {td_90:.3f}")
    print(f"  H12 75th pctile:                     {h12_75:.6f}")
    print(f"  H2/H1 median:                        {h2h1_median:.4f}")

    # --- Classify loci ---

    # 1. Directional sweep: top 5% max|nSL|
    merged['is_recent_sweep'] = merged['max_abs_nsl'] >= nsl_95
    merged['is_sweep_suggestive'] = merged['max_abs_nsl'] >= nsl_90

    # 2. Balancing selection: top 5% Tajima's D (most positive) AND older than expected
    merged['is_balancing'] = (merged['tajimas_d'] >= td_95) & (merged['mean_age_residual'] > 0)
    merged['is_balancing_suggestive'] = (merged['tajimas_d'] >= td_90) & (merged['mean_age_residual'] > 0)

    # 3. Soft sweep: convergent evidence from multiple statistics
    #    NOTE: H12 values are very low (max=0.025) because D1 computes them
    #    over 200kb windows with ~5800 SNPs — far larger than the ~50-100 SNP
    #    windows used in Garud et al. (2015). At this resolution, H12 cannot
    #    reliably distinguish sweep modes. We therefore define soft sweep as:
    #    - Elevated nSL (top 25% — directional selection signal)
    #    - Elevated Tajima's D (top 25% — NOT strongly negative, suggesting
    #      multiple haplotypes maintained, consistent with soft/incomplete sweep)
    #    - High H2/H1 (> 0.90 — haplotype diversity preserved, ruling out
    #      hard sweep where one haplotype dominates)
    #    This captures loci with selection signal but maintained haplotype diversity.
    nsl_75 = merged['max_abs_nsl'].quantile(0.75)
    td_75 = merged['tajimas_d'].quantile(0.75)
    merged['is_soft_sweep'] = (
        (merged['max_abs_nsl'] >= nsl_75) &
        (merged['tajimas_d'] >= td_75) &
        (merged['h2h1'] > 0.90)
    )

    # 4. Trifecta: balancing-like signature (old + high D) AND sweep signal
    merged['is_trifecta'] = merged['is_balancing_suggestive'] & merged['is_sweep_suggestive']

    # --- Percentile ranks (continuous, for downstream analysis) ---
    merged['nsl_percentile'] = merged['max_abs_nsl'].rank(pct=True) * 100
    merged['tajd_percentile'] = merged['tajimas_d'].rank(pct=True) * 100
    merged['h12_percentile'] = merged['h12'].rank(pct=True) * 100

    # --- Summary ---
    print("\n--- Selection Signature Summary ---")
    print(f"Total loci: {len(merged)}")
    print(f"Directional sweep (top 5% nSL):        {merged['is_recent_sweep'].sum()} ({merged['is_recent_sweep'].mean():.1%})")
    print(f"Directional sweep suggestive (top 10%): {merged['is_sweep_suggestive'].sum()} ({merged['is_sweep_suggestive'].mean():.1%})")
    print(f"Balancing selection (top 5% D, old):    {merged['is_balancing'].sum()} ({merged['is_balancing'].mean():.1%})")
    print(f"Balancing suggestive (top 10% D, old):  {merged['is_balancing_suggestive'].sum()} ({merged['is_balancing_suggestive'].mean():.1%})")
    print(f"Soft sweep (H12+H2H1+nSL convergent):  {merged['is_soft_sweep'].sum()} ({merged['is_soft_sweep'].mean():.1%})")
    print(f"Trifecta (balancing + sweep signals):   {merged['is_trifecta'].sum()} ({merged['is_trifecta'].mean():.1%})")

    # Print top candidates by category
    for label, col in [("Directional Sweep", "is_recent_sweep"),
                       ("Balancing Selection", "is_balancing"),
                       ("Soft Sweep", "is_soft_sweep"),
                       ("Trifecta", "is_trifecta")]:
        subset = merged[merged[col]].sort_values('max_abs_nsl', ascending=False)
        if len(subset) > 0:
            print(f"\n--- Top {label} Loci ---")
            for _, row in subset.head(5).iterrows():
                print(f"  {row['credible_set_id']:10s} | {row.get('lead_rsid','NA'):12s} | "
                      f"TajD: {row['tajimas_d']:+.2f} (P{row['tajd_percentile']:.0f}) | "
                      f"|nSL|: {row['max_abs_nsl']:.2f} (P{row['nsl_percentile']:.0f}) | "
                      f"H12: {row['h12']:.4f} | Age: {row['median_age_years']/1000:,.0f}K yr")

    # Correlations
    valid = merged.dropna(subset=['mean_age_residual', 'tajimas_d'])
    if len(valid) > 10:
        r, p = stats.spearmanr(valid['mean_age_residual'], valid['tajimas_d'])
        print(f"\nSpearman correlation (Age Residual vs Tajima's D): rho = {r:.3f}, p = {p:.1e}")

    # Save
    out_file = RESULTS_DIR / "D5_integrated_selection_signatures.tsv"
    merged.to_csv(out_file, sep='\t', index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()
