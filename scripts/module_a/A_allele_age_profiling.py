#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Module A: Allele Age Profiling
=================================================
Tests the central hypothesis: Are SCZ risk variant ages bimodally
distributed, suggesting maintenance by both recent mutation and
ancient balancing selection?

Steps:
  A1. Age distribution characterization
  A2. B-statistic corrected analysis (remove background selection confound)
  A3. Bimodality testing (Hartigan's dip test + Gaussian Mixture Model)
  A4. Functional annotation stratification (coding vs regulatory vs intergenic)
  A5. Age-PIP correlation (are higher-confidence variants older/younger?)

Scientific Rationale:
  If SCZ risk variants are maintained solely by mutation-selection balance,
  we expect a unimodal young age distribution. If balancing selection or
  antagonistic pleiotropy plays a role, we expect a bimodal distribution
  with a second mode at ancient ages (>200,000 years).

  CRITICAL CONFOUND: Background selection (B-statistic) can create
  the appearance of age structure. Variants in low-B regions appear
  younger due to reduced effective population size. We correct for this
  via B-statistic residualization.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_TABLE = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"
RESULTS_DIR = PROJECT_ROOT / "results/module_a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load annotated master table and prepare for analysis."""
    df = pd.read_csv(MASTER_TABLE, sep='\t')
    
    # Filter to variants with GEVA ages
    has_age = df['geva_age_median'].notna()
    has_bstat = df['b_statistic'].notna()
    
    print(f"Total variants: {len(df)}")
    print(f"With GEVA age: {has_age.sum()}")
    print(f"With B-statistic: {has_bstat.sum()}")
    print(f"With both: {(has_age & has_bstat).sum()}")
    
    return df


def step_a1_age_distribution(df):
    """A1: Characterize the GEVA age distribution of PGC3 variants."""
    print("\n" + "=" * 60)
    print("A1: Age Distribution Characterization")
    print("=" * 60)
    
    ages = df['geva_age_years'].dropna()
    ages_gen = df['geva_age_median'].dropna()
    
    # Basic statistics
    print(f"\n  N variants with age: {len(ages)}")
    print(f"  Median age: {ages.median():,.0f} years ({ages_gen.median():,.0f} generations)")
    print(f"  Mean age:   {ages.mean():,.0f} years ({ages_gen.mean():,.0f} generations)")
    print(f"  Std dev:    {ages.std():,.0f} years")
    print(f"  IQR:        {ages.quantile(0.25):,.0f} - {ages.quantile(0.75):,.0f} years")
    print(f"  Range:      {ages.min():,.0f} - {ages.max():,.0f} years")
    
    # Age strata
    strata = {
        'Very recent (<10,000 yr)': ages < 10000,
        'Recent (10K-50K yr)': (ages >= 10000) & (ages < 50000),
        'Intermediate (50K-200K yr)': (ages >= 50000) & (ages < 200000),
        'Pre-OOA (200K-500K yr)': (ages >= 200000) & (ages < 500000),
        'Ancient (500K-2M yr)': (ages >= 500000) & (ages < 2000000),
        'Very ancient (>2M yr)': ages >= 2000000,
    }
    
    print(f"\n  Age Strata:")
    for name, mask in strata.items():
        n = mask.sum()
        pct = n / len(ages) * 100
        print(f"    {name:35s}: {n:5d} ({pct:5.1f}%)")
    
    # Log-transform for downstream analysis
    log_ages = np.log10(ages + 1)
    
    # Normality test (should reject if bimodal)
    if len(log_ages) > 5000:
        # D'Agostino and Pearson's test (better for large N)
        stat, pval = stats.normaltest(log_ages)
        print(f"\n  Normality test (D'Agostino-Pearson) on log10(age):")
        print(f"    Statistic: {stat:.2f}")
        print(f"    P-value: {pval:.2e}")
        print(f"    {'→ REJECTS normality' if pval < 0.05 else '→ Cannot reject normality'}")
    
    # Skewness and kurtosis
    print(f"\n  Shape (log10-scale):")
    print(f"    Skewness: {log_ages.skew():.3f}")
    print(f"    Kurtosis: {log_ages.kurtosis():.3f}")
    
    # Save age distribution data
    age_dist = pd.DataFrame({
        'geva_age_years': ages,
        'log10_age': log_ages
    })
    age_dist.to_csv(RESULTS_DIR / "A1_age_distribution.tsv", sep='\t', index=False)
    
    return ages, log_ages


def step_a2_bstat_correction(df):
    """A2: Correct age distribution for background selection (B-statistic)."""
    print("\n" + "=" * 60)
    print("A2: B-statistic Correction")
    print("=" * 60)
    
    valid = df[df['geva_age_median'].notna() & df['b_statistic'].notna()].copy()
    valid['log_age'] = np.log10(valid['geva_age_years'] + 1)
    
    print(f"\n  Variants with both age and B-stat: {len(valid)}")
    
    # Correlation between B-stat and age
    r, p = stats.spearmanr(valid['b_statistic'], valid['log_age'])
    print(f"\n  Spearman correlation (B-stat vs log10-age):")
    print(f"    rho = {r:.4f}, p = {p:.2e}")
    print(f"    {'→ Significant confound — correction needed' if p < 0.05 else '→ No significant confound'}")
    
    # Residualize: regress log_age ~ b_statistic, use residuals
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(valid['b_statistic'], valid['log_age'], deg=1)
    predicted = np.polyval(coeffs, valid['b_statistic'])
    residuals = valid['log_age'] - predicted
    
    print(f"\n  Linear regression: log_age = {coeffs[0]:.4f} * B + {coeffs[1]:.4f}")
    print(f"  R² = {1 - np.var(residuals)/np.var(valid['log_age']):.4f}")
    
    valid['age_residual'] = residuals
    
    # Compare raw vs corrected distributions
    print(f"\n  Raw log10-age:      mean={valid['log_age'].mean():.3f}, std={valid['log_age'].std():.3f}")
    print(f"  B-stat residual:    mean={residuals.mean():.3f}, std={residuals.std():.3f}")
    
    # Save
    valid[['rsid', 'chr', 'pos', 'geva_age_years', 'b_statistic', 'log_age', 'age_residual']].to_csv(
        RESULTS_DIR / "A2_bstat_corrected_ages.tsv", sep='\t', index=False
    )
    
    return valid


def step_a3_bimodality(df_corrected, raw_log_ages):
    """A3: Test for bimodality using Hartigan's dip test and GMM."""
    print("\n" + "=" * 60)
    print("A3: Bimodality Testing")
    print("=" * 60)
    
    # --- Hartigan's Dip Test ---
    print("\n  [Hartigan's Dip Test]")
    
    try:
        import diptest
        has_diptest = True
    except ImportError:
        has_diptest = False
        print("    diptest not installed — using manual implementation")
    
    # Test on raw log ages
    if has_diptest:
        dip_stat, dip_p = diptest.diptest(raw_log_ages.values)
        print(f"    Raw log10-age: dip = {dip_stat:.4f}, p = {dip_p:.4f}")
        print(f"    {'→ Significant bimodality' if dip_p < 0.05 else '→ Cannot reject unimodality'}")
        
        # Test on B-stat corrected residuals
        residuals = df_corrected['age_residual'].values
        dip_stat_c, dip_p_c = diptest.diptest(residuals)
        print(f"    B-corrected:    dip = {dip_stat_c:.4f}, p = {dip_p_c:.4f}")
        print(f"    {'→ Significant bimodality (robust to B-stat)' if dip_p_c < 0.05 else '→ Bimodality not significant after correction'}")
    else:
        print("    Skipping dip test (install with: pip3 install diptest)")
        dip_p = None
        dip_p_c = None
    
    # --- Gaussian Mixture Model ---
    print("\n  [Gaussian Mixture Model (GMM)]")
    
    from sklearn.mixture import GaussianMixture
    
    # Fit 1-component vs 2-component GMM on B-corrected residuals
    X = df_corrected['age_residual'].values.reshape(-1, 1)
    
    results = {}
    for k in [1, 2, 3]:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        results[k] = {'bic': bic, 'aic': aic, 'gmm': gmm}
        print(f"    K={k}: BIC={bic:,.0f}, AIC={aic:,.0f}")
    
    # Best model by BIC
    best_k = min(results, key=lambda k: results[k]['bic'])
    print(f"\n    Best model (BIC): K={best_k}")
    
    # BIC evidence ratio
    delta_bic = results[1]['bic'] - results[2]['bic']
    print(f"    ΔBIC (K=1 vs K=2): {delta_bic:,.0f}")
    if delta_bic > 10:
        print(f"    → STRONG evidence for bimodality (ΔBIC > 10)")
    elif delta_bic > 6:
        print(f"    → Moderate evidence for bimodality (ΔBIC > 6)")
    elif delta_bic > 2:
        print(f"    → Weak evidence for bimodality (ΔBIC > 2)")
    else:
        print(f"    → No evidence for bimodality")
    
    # If K=2 is best, characterize the two components
    if best_k >= 2:
        gmm2 = results[2]['gmm']
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())
        weights = gmm2.weights_
        
        # Convert residual means back to approximate ages
        # residual ≈ 0 means "average age", positive = older, negative = younger
        print(f"\n    Two-component GMM parameters (residual space):")
        for i in range(2):
            # Back-transform: mean_log_age ≈ global_mean + residual
            global_mean = df_corrected['log_age'].mean()
            approx_log_age = global_mean + means[i]
            approx_age = 10**approx_log_age
            print(f"      Component {i+1}: weight={weights[i]:.3f}, "
                  f"mean_residual={means[i]:.3f} (≈{approx_age:,.0f} yr), "
                  f"std={stds[i]:.3f}")
        
        # Assign variants to components
        labels = gmm2.predict(X)
        df_corrected['gmm_component'] = labels
        
        component_sizes = pd.Series(labels).value_counts().sort_index()
        for comp, count in component_sizes.items():
            print(f"      Component {comp+1} contains {count} variants ({count/len(labels)*100:.1f}%)")
    
    # Save results
    summary = {
        'test': ['Hartigan_dip_raw', 'Hartigan_dip_Bcorrected', 
                 'GMM_BIC_K1', 'GMM_BIC_K2', 'GMM_BIC_K3', 'GMM_best_K', 'delta_BIC_1v2'],
        'value': [
            dip_p if dip_p is not None else 'NA',
            dip_p_c if dip_p_c is not None else 'NA',
            results[1]['bic'], results[2]['bic'], results[3]['bic'],
            best_k, delta_bic
        ]
    }
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "A3_bimodality_tests.tsv", sep='\t', index=False)
    
    if 'gmm_component' in df_corrected.columns:
        df_corrected.to_csv(RESULTS_DIR / "A3_variant_components.tsv", sep='\t', index=False)
    
    return results, df_corrected


def step_a4_functional_stratification(df):
    """A4: Stratify age distribution by functional annotation."""
    print("\n" + "=" * 60)
    print("A4: Functional Annotation Stratification")
    print("=" * 60)
    
    valid = df[df['geva_age_years'].notna()].copy()
    valid['log_age'] = np.log10(valid['geva_age_years'] + 1)
    
    # Stratify by VEP impact
    if 'vep_impact' in valid.columns:
        print("\n  [By VEP Impact]")
        for impact in ['HIGH', 'MODERATE', 'LOW', 'MODIFIER']:
            subset = valid[valid['vep_impact'] == impact]
            if len(subset) > 0:
                median_age = subset['geva_age_years'].median()
                print(f"    {impact:12s}: n={len(subset):5d}, median_age={median_age:>12,.0f} yr")
    
    # Stratify by gene biotype
    if 'gene_biotype' in valid.columns:
        print("\n  [By Gene Biotype (top 5)]")
        top_biotypes = valid['gene_biotype'].value_counts().head(5).index
        for biotype in top_biotypes:
            subset = valid[valid['gene_biotype'] == biotype]
            if len(subset) > 0:
                median_age = subset['geva_age_years'].median()
                print(f"    {str(biotype):25s}: n={len(subset):5d}, median_age={median_age:>12,.0f} yr")
    
    # Stratify by PIP quartile
    print("\n  [By PIP Quartile]")
    valid['pip_quartile'] = pd.qcut(valid['pip'], q=4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])
    for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
        subset = valid[valid['pip_quartile'] == q]
        median_age = subset['geva_age_years'].median()
        print(f"    {q:12s}: n={len(subset):5d}, median_age={median_age:>12,.0f} yr")
    
    # PIP-age correlation
    r, p = stats.spearmanr(valid['pip'], valid['log_age'])
    print(f"\n  Spearman (PIP vs log10-age): rho={r:.4f}, p={p:.2e}")
    
    # Stratify by DAF
    if 'daf' in valid.columns:
        print("\n  [By DAF Quartile]")
        daf_valid = valid[valid['daf'].notna()]
        daf_valid['daf_quartile'] = pd.qcut(daf_valid['daf'], q=4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'])
        for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
            subset = daf_valid[daf_valid['daf_quartile'] == q]
            median_age = subset['geva_age_years'].median()
            print(f"    {q:12s}: n={len(subset):5d}, median_age={median_age:>12,.0f} yr, median_daf={subset['daf'].median():.3f}")
        
        r_daf, p_daf = stats.spearmanr(daf_valid['daf'], daf_valid['log_age'])
        print(f"\n  Spearman (DAF vs log10-age): rho={r_daf:.4f}, p={p_daf:.2e}")
    
    # Save
    valid[['rsid', 'chr', 'pos', 'geva_age_years', 'log_age', 'pip', 'vep_impact', 
           'gene_biotype', 'daf', 'b_statistic']].to_csv(
        RESULTS_DIR / "A4_functional_stratification.tsv", sep='\t', index=False
    )


def step_a5_summary(df, gmm_results, df_corrected):
    """A5: Summary statistics and key findings."""
    print("\n" + "=" * 60)
    print("A5: MODULE A SUMMARY")
    print("=" * 60)
    
    ages = df['geva_age_years'].dropna()
    
    print(f"""
  EVOSCZ Module A — Allele Age Profiling Results
  ────────────────────────────────────────────────
  Total PGC3 credible set variants:     {len(df)}
  With GEVA allele age:                 {ages.notna().sum()} ({df['geva_age_median'].notna().mean()*100:.1f}%)
  With B-stat + age (for correction):   {len(df_corrected)}
  
  Age Distribution (years):
    Median:     {ages.median():>12,.0f}
    Mean:       {ages.mean():>12,.0f}
    IQR:        {ages.quantile(0.25):>12,.0f} — {ages.quantile(0.75):>12,.0f}
    
  Bimodality Evidence:
    GMM best K:           {min(gmm_results, key=lambda k: gmm_results[k]['bic'])}
    ΔBIC (K=1 vs K=2):    {gmm_results[1]['bic'] - gmm_results[2]['bic']:,.0f}
    
  Interpretation:
    Most SCZ risk variants are OLD (median ~558K years).
    This is consistent with ancient origin, not recent mutation.
    Whether the distribution is truly bimodal requires the dip test.
""")
    
    # Save summary
    with open(RESULTS_DIR / "A5_module_summary.txt", 'w') as f:
        f.write(f"EVOSCZ Module A Summary\n")
        f.write(f"Date: 2026-04-11\n")
        f.write(f"N variants: {len(df)}\n")
        f.write(f"N with age: {ages.notna().sum()}\n")
        f.write(f"Median age (years): {ages.median():.0f}\n")
        f.write(f"Mean age (years): {ages.mean():.0f}\n")
        f.write(f"GMM best K: {min(gmm_results, key=lambda k: gmm_results[k]['bic'])}\n")
        f.write(f"delta_BIC_1v2: {gmm_results[1]['bic'] - gmm_results[2]['bic']:.0f}\n")


def main():
    print("=" * 60)
    print("EVOSCZ MODULE A: ALLELE AGE PROFILING")
    print("=" * 60)
    
    df = load_data()
    
    # A1: Distribution characterization
    ages, log_ages = step_a1_age_distribution(df)
    
    # A2: B-statistic correction
    df_corrected = step_a2_bstat_correction(df)
    
    # A3: Bimodality testing
    gmm_results, df_corrected = step_a3_bimodality(df_corrected, log_ages)
    
    # A4: Functional stratification
    step_a4_functional_stratification(df)
    
    # A5: Summary
    step_a5_summary(df, gmm_results, df_corrected)
    
    print(f"\nAll Module A results saved to: {RESULTS_DIR}/")
    print(f"Files generated:")
    for f in sorted(RESULTS_DIR.glob("*.tsv")) + sorted(RESULTS_DIR.glob("*.txt")):
        print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
