#!/usr/bin/env python3
"""
EVOSCZ Phase 7c: Integration & Hypothesis Testing
===================================================
Merges P7a (L2G gene assignments) and P7b (eQTL tissue mapping) into
the P6 Master Analytic Table, then tests key evolutionary-genomic
hypotheses about SCZ risk loci.

New columns added to master table:
  - l2g_gene, l2g_score, l2g_ensembl, l2g_biotype
  - n_eqtl_total, n_brain_eqtl, n_immune_eqtl
  - primary_eqtl_tissue, primary_tissue_class
  - has_dlpfc_eqtl, has_immune_eqtl, has_brain_eqtl

Hypotheses tested:
  H1: Ancient variants (>500K yr) → immune eQTL enrichment (microglia)
  H2: Young sweep variants → neuronal/synaptic eQTL enrichment
  H3: Brain eQTL loci are older than immune eQTL loci
  H4: SCZ-immune colocalized loci (H4>0.8) overlap immune eQTL loci
  H5: DLPFC eQTL loci enriched among ancient variants
  H6: Selection signatures differ between brain-eQTL vs immune-eQTL loci
  H7: L2G-assigned genes overlap eQTL genes (convergent evidence)
  H8: Trifecta loci have distinct tissue profiles
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_IN = PROJECT_ROOT / "results/integration/P6_Master_Analytic_Table.tsv"
L2G_FILE = PROJECT_ROOT / "results/phase7/P7a_l2g_assignments.tsv"
EQTL_FILE = PROJECT_ROOT / "results/phase7/P7b_eqtl_tissue_map.tsv"
MASTER_OUT = PROJECT_ROOT / "results/integration/P7_Master_Analytic_Table.tsv"
REPORT_OUT = PROJECT_ROOT / "results/phase7/P7c_hypothesis_report.txt"
RESULTS_DIR = PROJECT_ROOT / "results/phase7"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    master = pd.read_csv(MASTER_IN, sep='\t')
    l2g = pd.read_csv(L2G_FILE, sep='\t')
    eqtl = pd.read_csv(EQTL_FILE, sep='\t')
    return master, l2g, eqtl


def prepare_l2g_per_locus(l2g):
    """Get best L2G gene per credible_set_id (highest score)."""
    l2g_sorted = l2g.sort_values('l2g_score', ascending=False)
    best = l2g_sorted.drop_duplicates(subset=['credible_set_id'], keep='first')
    return best[['credible_set_id', 'l2g_gene', 'l2g_ensembl',
                  'l2g_biotype', 'l2g_score']].copy()


def prepare_eqtl_per_locus(eqtl):
    """Aggregate eQTL tissue data per credible_set_id."""
    agg = []
    for csid, grp in eqtl.groupby('credible_set_id'):
        n_total = len(grp)
        n_brain = (grp['tissue_class'] == 'brain').sum()
        n_immune = (grp['tissue_class'] == 'immune').sum()

        # Primary tissue = most frequent biosample
        top_tissue = grp['biosample_name'].value_counts().index[0] if len(grp) > 0 else ''
        top_class = grp[grp['biosample_name'] == top_tissue]['tissue_class'].iloc[0] if len(grp) > 0 else ''

        # Specific tissue flags
        has_dlpfc = any('dorsolateral prefrontal' in str(b).lower() or
                        'dlpfc' in str(b).lower()
                        for b in grp['biosample_name'])
        has_immune = n_immune > 0
        has_brain = n_brain > 0

        # Unique eQTL genes
        eqtl_genes = ','.join(sorted(grp['eqtl_gene'].dropna().unique()))

        agg.append({
            'credible_set_id': csid,
            'n_eqtl_total': n_total,
            'n_brain_eqtl': n_brain,
            'n_immune_eqtl': n_immune,
            'primary_eqtl_tissue': top_tissue,
            'primary_tissue_class': top_class,
            'has_dlpfc_eqtl': has_dlpfc,
            'has_immune_eqtl': has_immune,
            'has_brain_eqtl': has_brain,
            'eqtl_genes': eqtl_genes
        })
    return pd.DataFrame(agg)


def merge_into_master(master, l2g_best, eqtl_agg):
    """Merge L2G and eQTL summary into master table at locus level."""
    merged = master.merge(l2g_best, on='credible_set_id', how='left', suffixes=('', '_l2g'))
    merged = merged.merge(eqtl_agg, on='credible_set_id', how='left')

    # Fill NaN for boolean columns
    for col in ['has_dlpfc_eqtl', 'has_immune_eqtl', 'has_brain_eqtl']:
        merged[col] = merged[col].fillna(False).astype(bool)
    for col in ['n_eqtl_total', 'n_brain_eqtl', 'n_immune_eqtl']:
        merged[col] = merged[col].fillna(0).astype(int)

    return merged


def get_locus_level(df):
    """One row per credible_set_id (highest PIP variant)."""
    return df.sort_values('pip', ascending=False).drop_duplicates(
        subset=['credible_set_id'], keep='first').copy()


def test_hypotheses(locus, report_lines):
    """Run all hypothesis tests at locus level."""

    def rprint(msg):
        print(msg)
        report_lines.append(msg)

    rprint("=" * 70)
    rprint("EVOSCZ Phase 7c: Hypothesis Testing Results")
    rprint("=" * 70)
    rprint(f"Loci total: {len(locus)}")
    rprint(f"Loci with L2G: {locus['l2g_gene'].notna().sum()}")
    rprint(f"Loci with eQTL: {(locus['n_eqtl_total'] > 0).sum()}")
    rprint(f"Loci with brain eQTL: {locus['has_brain_eqtl'].sum()}")
    rprint(f"Loci with immune eQTL: {locus['has_immune_eqtl'].sum()}")
    rprint(f"Loci with DLPFC eQTL: {locus['has_dlpfc_eqtl'].sum()}")

    # Age categories
    locus_with_age = locus[locus['geva_age_years'].notna()].copy()
    ancient = locus_with_age[locus_with_age['geva_age_years'] > 500000]
    young = locus_with_age[locus_with_age['geva_age_years'] <= 500000]

    rprint(f"\nLoci with GEVA age: {len(locus_with_age)}")
    rprint(f"  Ancient (>500K yr): {len(ancient)}")
    rprint(f"  Young (<=500K yr): {len(young)}")

    # ========== H1: Ancient → immune eQTL enrichment ==========
    rprint("\n" + "-" * 70)
    rprint("H1: Ancient variants enriched for immune eQTL")
    rprint("-" * 70)

    if len(ancient) > 0 and len(young) > 0:
        anc_immune = ancient['has_immune_eqtl'].sum()
        anc_total = len(ancient)
        yng_immune = young['has_immune_eqtl'].sum()
        yng_total = len(young)

        table_h1 = [[anc_immune, anc_total - anc_immune],
                     [yng_immune, yng_total - yng_immune]]
        or_h1, p_h1 = stats.fisher_exact(table_h1, alternative='greater')

        rprint(f"  Ancient with immune eQTL: {anc_immune}/{anc_total} ({anc_immune/anc_total*100:.1f}%)")
        rprint(f"  Young with immune eQTL:   {yng_immune}/{yng_total} ({yng_immune/yng_total*100:.1f}%)")
        rprint(f"  Fisher's exact (greater): OR={or_h1:.2f}, p={p_h1:.4f}")
        rprint(f"  Interpretation: {'SUPPORTED' if p_h1 < 0.05 else 'NOT SUPPORTED'} at alpha=0.05")

    # ========== H2: Young sweep → neuronal eQTL enrichment ==========
    rprint("\n" + "-" * 70)
    rprint("H2: Young sweep variants enriched for brain/neuronal eQTL")
    rprint("-" * 70)

    sweep_loci = locus[(locus['is_recent_sweep'] == True) | (locus['is_soft_sweep'] == True)]
    non_sweep = locus[(locus['is_recent_sweep'] == False) & (locus['is_soft_sweep'] == False)]

    if len(sweep_loci) > 0:
        sw_brain = sweep_loci['has_brain_eqtl'].sum()
        sw_total = len(sweep_loci)
        nsw_brain = non_sweep['has_brain_eqtl'].sum()
        nsw_total = len(non_sweep)

        table_h2 = [[sw_brain, sw_total - sw_brain],
                     [nsw_brain, nsw_total - nsw_brain]]
        or_h2, p_h2 = stats.fisher_exact(table_h2, alternative='greater')

        rprint(f"  Sweep loci with brain eQTL: {sw_brain}/{sw_total} ({sw_brain/sw_total*100:.1f}%)")
        rprint(f"  Non-sweep with brain eQTL:  {nsw_brain}/{nsw_total} ({nsw_brain/nsw_total*100:.1f}%)")
        rprint(f"  Fisher's exact (greater): OR={or_h2:.2f}, p={p_h2:.4f}")
        rprint(f"  Interpretation: {'SUPPORTED' if p_h2 < 0.05 else 'NOT SUPPORTED'} at alpha=0.05")
    else:
        rprint("  No sweep loci found - cannot test")

    # ========== H3: Brain eQTL loci older than immune eQTL loci ==========
    rprint("\n" + "-" * 70)
    rprint("H3: Brain eQTL loci are older than immune eQTL loci")
    rprint("-" * 70)

    brain_ages = locus_with_age[locus_with_age['has_brain_eqtl'] == True]['geva_age_years']
    immune_ages = locus_with_age[locus_with_age['has_immune_eqtl'] == True]['geva_age_years']

    if len(brain_ages) > 0 and len(immune_ages) > 0:
        u_stat, p_h3 = stats.mannwhitneyu(brain_ages, immune_ages, alternative='greater')
        rprint(f"  Brain eQTL loci (n={len(brain_ages)}):  median age = {brain_ages.median():,.0f} yr")
        rprint(f"  Immune eQTL loci (n={len(immune_ages)}): median age = {immune_ages.median():,.0f} yr")
        rprint(f"  Mann-Whitney U (brain > immune): U={u_stat:.0f}, p={p_h3:.4f}")
        rprint(f"  Interpretation: {'SUPPORTED' if p_h3 < 0.05 else 'NOT SUPPORTED'} at alpha=0.05")

        # Also test: exclusive categories
        brain_only = locus_with_age[(locus_with_age['has_brain_eqtl'] == True) &
                                    (locus_with_age['has_immune_eqtl'] == False)]['geva_age_years']
        immune_only = locus_with_age[(locus_with_age['has_immune_eqtl'] == True) &
                                     (locus_with_age['has_brain_eqtl'] == False)]['geva_age_years']
        if len(brain_only) > 0 and len(immune_only) > 0:
            u2, p_h3b = stats.mannwhitneyu(brain_only, immune_only, alternative='greater')
            rprint(f"  Brain-ONLY (n={len(brain_only)}): median = {brain_only.median():,.0f} yr")
            rprint(f"  Immune-ONLY (n={len(immune_only)}): median = {immune_only.median():,.0f} yr")
            rprint(f"  Mann-Whitney U (exclusive): U={u2:.0f}, p={p_h3b:.4f}")

    # ========== H4: SCZ-immune coloc (H4>0.8) overlaps immune eQTL ==========
    rprint("\n" + "-" * 70)
    rprint("H4: SCZ-immune colocalized loci overlap immune eQTL loci")
    rprint("-" * 70)

    coloc = locus[locus['OpenTargets_H4'] > 0.8]
    non_coloc = locus[locus['OpenTargets_H4'] <= 0.8]

    if len(coloc) > 0:
        col_immune = coloc['has_immune_eqtl'].sum()
        col_total = len(coloc)
        ncol_immune = non_coloc['has_immune_eqtl'].sum()
        ncol_total = len(non_coloc)

        table_h4 = [[col_immune, col_total - col_immune],
                     [ncol_immune, ncol_total - ncol_immune]]
        or_h4, p_h4 = stats.fisher_exact(table_h4, alternative='greater')

        rprint(f"  Colocalized (H4>0.8) with immune eQTL: {col_immune}/{col_total} ({col_immune/col_total*100:.1f}%)")
        rprint(f"  Non-colocalized with immune eQTL:      {ncol_immune}/{ncol_total} ({ncol_immune/ncol_total*100:.1f}%)")
        rprint(f"  Fisher's exact (greater): OR={or_h4:.2f}, p={p_h4:.4f}")
        rprint(f"  Interpretation: {'SUPPORTED' if p_h4 < 0.05 else 'NOT SUPPORTED'} at alpha=0.05")
    else:
        rprint("  No H4>0.8 colocalized loci found")

    # ========== H5: DLPFC eQTL enriched among ancient ==========
    rprint("\n" + "-" * 70)
    rprint("H5: DLPFC eQTL loci enriched among ancient variants")
    rprint("-" * 70)

    if len(ancient) > 0 and len(young) > 0:
        anc_dlpfc = ancient['has_dlpfc_eqtl'].sum()
        yng_dlpfc = young['has_dlpfc_eqtl'].sum()

        table_h5 = [[anc_dlpfc, len(ancient) - anc_dlpfc],
                     [yng_dlpfc, len(young) - yng_dlpfc]]
        or_h5, p_h5 = stats.fisher_exact(table_h5, alternative='greater')

        rprint(f"  Ancient with DLPFC eQTL: {anc_dlpfc}/{len(ancient)} ({anc_dlpfc/len(ancient)*100:.1f}%)")
        rprint(f"  Young with DLPFC eQTL:   {yng_dlpfc}/{len(young)} ({yng_dlpfc/len(young)*100:.1f}%)")
        rprint(f"  Fisher's exact (greater): OR={or_h5:.2f}, p={p_h5:.4f}")
        rprint(f"  Interpretation: {'SUPPORTED' if p_h5 < 0.05 else 'NOT SUPPORTED'} at alpha=0.05")

    # ========== H6: Selection differs between brain vs immune eQTL ==========
    rprint("\n" + "-" * 70)
    rprint("H6: Selection signatures differ: brain-eQTL vs immune-eQTL loci")
    rprint("-" * 70)

    brain_loci = locus[locus['has_brain_eqtl'] == True]
    immune_loci = locus[locus['has_immune_eqtl'] == True]

    for stat_col, stat_name in [('max_abs_nsl', '|nSL|'), ('tajimas_d', "Tajima's D")]:
        b_vals = brain_loci[stat_col].dropna()
        i_vals = immune_loci[stat_col].dropna()
        if len(b_vals) > 0 and len(i_vals) > 0:
            u, p = stats.mannwhitneyu(b_vals, i_vals, alternative='two-sided')
            rprint(f"  {stat_name}: brain median={b_vals.median():.3f}, immune median={i_vals.median():.3f}")
            rprint(f"    Mann-Whitney U={u:.0f}, p={p:.4f}")

    # Sweep enrichment in brain vs immune
    if len(brain_loci) > 0 and len(immune_loci) > 0:
        b_sweep = ((brain_loci['is_recent_sweep'] == True) | (brain_loci['is_soft_sweep'] == True)).sum()
        i_sweep = ((immune_loci['is_recent_sweep'] == True) | (immune_loci['is_soft_sweep'] == True)).sum()
        rprint(f"  Sweep loci in brain-eQTL: {b_sweep}/{len(brain_loci)} ({b_sweep/len(brain_loci)*100:.1f}%)")
        rprint(f"  Sweep loci in immune-eQTL: {i_sweep}/{len(immune_loci)} ({i_sweep/len(immune_loci)*100:.1f}%)")

    # ========== H7: L2G gene = eQTL gene convergence ==========
    rprint("\n" + "-" * 70)
    rprint("H7: L2G-assigned genes overlap with eQTL genes (convergent evidence)")
    rprint("-" * 70)

    both = locus[(locus['l2g_gene'].notna()) & (locus['eqtl_genes'].notna()) &
                 (locus['eqtl_genes'] != '')]
    if len(both) > 0:
        convergent = 0
        for _, row in both.iterrows():
            l2g_g = str(row['l2g_gene'])
            eqtl_g = str(row['eqtl_genes']).split(',')
            if l2g_g in eqtl_g:
                convergent += 1

        rprint(f"  Loci with both L2G and eQTL data: {len(both)}")
        rprint(f"  L2G gene found in eQTL gene list: {convergent}/{len(both)} ({convergent/len(both)*100:.1f}%)")
        rprint(f"  This represents convergent causal gene evidence from two independent methods")

    # ========== H8: Trifecta loci tissue profiles ==========
    rprint("\n" + "-" * 70)
    rprint("H8: Trifecta loci have distinct tissue profiles")
    rprint("-" * 70)

    trifecta = locus[locus['is_trifecta'] == True]
    if len(trifecta) > 0:
        rprint(f"  Trifecta loci: {len(trifecta)}")
        for _, row in trifecta.iterrows():
            rprint(f"    {row['credible_set_id']} ({row.get('lead_rsid', row.get('rsid', ''))}):")
            rprint(f"      L2G gene: {row.get('l2g_gene', 'N/A')} (score={row.get('l2g_score', 'N/A')})")
            rprint(f"      Brain eQTL: {row.get('n_brain_eqtl', 0)}, Immune eQTL: {row.get('n_immune_eqtl', 0)}")
            rprint(f"      Primary tissue: {row.get('primary_eqtl_tissue', 'N/A')}")
            rprint(f"      Age: {row.get('geva_age_years', 'N/A'):.0f} yr" if pd.notna(row.get('geva_age_years')) else "      Age: N/A")
            rprint(f"      nSL={row.get('max_abs_nsl', 'N/A')}, Tajima's D={row.get('tajimas_d', 'N/A')}")
    else:
        rprint("  No trifecta loci found")

    # ========== Summary table ==========
    rprint("\n" + "=" * 70)
    rprint("SUMMARY: Tissue × Evolutionary Class Cross-tabulation")
    rprint("=" * 70)

    for evo_class in locus['Evolutionary_Class'].unique():
        subset = locus[locus['Evolutionary_Class'] == evo_class]
        n = len(subset)
        n_brain = subset['has_brain_eqtl'].sum()
        n_immune = subset['has_immune_eqtl'].sum()
        n_dlpfc = subset['has_dlpfc_eqtl'].sum()
        rprint(f"  {evo_class:30s}: n={n:3d}, brain={n_brain:2d} ({n_brain/n*100:5.1f}%), "
               f"immune={n_immune:2d} ({n_immune/n*100:5.1f}%), DLPFC={n_dlpfc:2d}")


def main():
    print("=" * 70)
    print("EVOSCZ Phase 7c: Integration & Hypothesis Testing")
    print("=" * 70)

    # Load data
    master, l2g, eqtl = load_data()
    print(f"Master table: {len(master)} variants, {master['credible_set_id'].nunique()} loci")
    print(f"L2G predictions: {len(l2g)} entries, {l2g['credible_set_id'].nunique()} loci")
    print(f"eQTL tissue map: {len(eqtl)} entries, {eqtl['credible_set_id'].nunique()} loci")

    # Prepare per-locus summaries
    l2g_best = prepare_l2g_per_locus(l2g)
    eqtl_agg = prepare_eqtl_per_locus(eqtl)
    print(f"\nL2G best per locus: {len(l2g_best)}")
    print(f"eQTL aggregated per locus: {len(eqtl_agg)}")

    # Merge into master
    merged = merge_into_master(master, l2g_best, eqtl_agg)
    print(f"Merged table: {len(merged)} rows, {len(merged.columns)} columns")

    # Save merged master table
    merged.to_csv(MASTER_OUT, sep='\t', index=False)
    print(f"Saved: {MASTER_OUT}")

    # Hypothesis testing at locus level
    locus = get_locus_level(merged)
    print(f"Locus-level table: {len(locus)} loci\n")

    report_lines = []
    test_hypotheses(locus, report_lines)

    # Save report
    with open(REPORT_OUT, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved: {REPORT_OUT}")

    # Save locus-level summary
    locus_out = RESULTS_DIR / "P7c_locus_summary.tsv"
    locus_cols = ['credible_set_id', 'rsid', 'chr', 'pos', 'geva_age_years',
                  'Evolutionary_Class', 'l2g_gene', 'l2g_score',
                  'n_eqtl_total', 'n_brain_eqtl', 'n_immune_eqtl',
                  'has_dlpfc_eqtl', 'has_brain_eqtl', 'has_immune_eqtl',
                  'primary_eqtl_tissue', 'primary_tissue_class',
                  'max_abs_nsl', 'tajimas_d', 'OpenTargets_H4',
                  'is_recent_sweep', 'is_soft_sweep', 'is_balancing', 'is_trifecta']
    existing_cols = [c for c in locus_cols if c in locus.columns]
    locus[existing_cols].to_csv(locus_out, sep='\t', index=False)
    print(f"Locus summary saved: {locus_out}")


if __name__ == "__main__":
    main()
