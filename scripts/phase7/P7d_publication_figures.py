#!/usr/bin/env python3
"""
EVOSCZ Phase 7d: Updated Publication Figures with L2G & eQTL Data
==================================================================
Generates figures incorporating Phase 7 tissue mapping and L2G results.

Fig5: eQTL Tissue Distribution (brain vs immune vs other)
Fig6: Age vs Tissue Class (brain-only vs immune-only vs both)
Fig7: L2G-eQTL Convergence (Venn-like bar chart)
Fig8: Top L2G Genes with eQTL Tissue Annotation
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER = PROJECT_ROOT / "results/integration/P7_Master_Analytic_Table.tsv"
EQTL_FILE = PROJECT_ROOT / "results/phase7/P7b_eqtl_tissue_map.tsv"
FIGURES_DIR = PROJECT_ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def get_locus(df):
    return df.sort_values('pip', ascending=False).drop_duplicates(
        subset=['credible_set_id'], keep='first').copy()


def fig5_tissue_distribution(eqtl):
    """Fig 5: eQTL tissue distribution across SCZ loci."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Left: tissue class pie
    tc = eqtl['tissue_class'].value_counts()
    colors_map = {'brain': '#1f77b4', 'immune': '#d62728', 'other': '#cccccc', 'unknown': '#999999'}
    colors = [colors_map.get(c, '#cccccc') for c in tc.index]
    axes[0].pie(tc.values, labels=[f"{c}\n(n={n})" for c, n in zip(tc.index, tc.values)],
                colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    axes[0].set_title('eQTL Tissue Class Distribution\n(1,397 entries across 119 loci)',
                       fontsize=12, fontweight='bold')

    # Right: top individual tissues
    top_tissues = eqtl['biosample_name'].value_counts().head(15)
    tissue_colors = []
    for t in top_tissues.index:
        cls = eqtl[eqtl['biosample_name'] == t]['tissue_class'].mode().iloc[0]
        tissue_colors.append(colors_map.get(cls, '#cccccc'))

    axes[1].barh(range(len(top_tissues)), top_tissues.values, color=tissue_colors)
    axes[1].set_yticks(range(len(top_tissues)))
    axes[1].set_yticklabels(top_tissues.index, fontsize=9)
    axes[1].set_xlabel('Number of eQTL Entries', fontsize=11)
    axes[1].set_title('Top 15 eQTL Tissues at SCZ Loci', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()

    # Legend
    brain_patch = mpatches.Patch(color='#1f77b4', label='Brain')
    immune_patch = mpatches.Patch(color='#d62728', label='Immune')
    other_patch = mpatches.Patch(color='#cccccc', label='Other')
    axes[1].legend(handles=[brain_patch, immune_patch, other_patch], loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig5_eQTL_Tissue_Distribution.pdf')
    fig.savefig(FIGURES_DIR / 'Fig5_eQTL_Tissue_Distribution.png')
    plt.close(fig)
    print("  Fig5 saved")


def fig6_age_vs_tissue(locus):
    """Fig 6: Allele age stratified by eQTL tissue class."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    locus_age = locus[locus['geva_age_years'].notna()].copy()

    # Categorize loci by tissue
    def tissue_cat(row):
        b = row.get('has_brain_eqtl', False)
        i = row.get('has_immune_eqtl', False)
        if b and i:
            return 'Both Brain+Immune'
        if b:
            return 'Brain Only'
        if i:
            return 'Immune Only'
        return 'No eQTL'

    locus_age['tissue_cat'] = locus_age.apply(tissue_cat, axis=1)

    order = ['No eQTL', 'Brain Only', 'Immune Only', 'Both Brain+Immune']
    palette = {'No eQTL': '#cccccc', 'Brain Only': '#1f77b4',
               'Immune Only': '#d62728', 'Both Brain+Immune': '#9467bd'}

    data_for_plot = []
    for cat in order:
        subset = locus_age[locus_age['tissue_cat'] == cat]['geva_age_years']
        if len(subset) > 0:
            data_for_plot.append(subset.values)
        else:
            data_for_plot.append([])

    bp = ax.boxplot(data_for_plot, vert=True, patch_artist=True, widths=0.6)
    for i, cat in enumerate(order):
        bp['boxes'][i].set_facecolor(palette[cat])
        bp['boxes'][i].set_alpha(0.7)

    # Add individual points
    for i, cat in enumerate(order):
        subset = locus_age[locus_age['tissue_cat'] == cat]['geva_age_years']
        if len(subset) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(subset))
            ax.scatter(np.full(len(subset), i + 1) + jitter, subset,
                       c=palette[cat], alpha=0.4, s=20, zorder=3)

    # Counts
    counts = locus_age['tissue_cat'].value_counts()
    labels = [f"{cat}\n(n={counts.get(cat, 0)})" for cat in order]
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylabel('Allele Age (Years, GEVA)', fontsize=12)
    ax.set_title('SCZ Locus Allele Age by eQTL Tissue Category', fontsize=13, fontweight='bold')
    ax.axhline(500000, color='gray', linestyle='--', alpha=0.5, label='Ancient threshold (500K yr)')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig6_Age_vs_Tissue.pdf')
    fig.savefig(FIGURES_DIR / 'Fig6_Age_vs_Tissue.png')
    plt.close(fig)
    print("  Fig6 saved")


def fig7_l2g_eqtl_convergence(locus):
    """Fig 7: L2G vs eQTL gene convergence."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    has_l2g = locus['l2g_gene'].notna().sum()
    has_eqtl = (locus['n_eqtl_total'] > 0).sum()
    has_both = locus[(locus['l2g_gene'].notna()) & (locus['eqtl_genes'].notna()) &
                     (locus['eqtl_genes'] != '')]

    # Count convergent
    convergent = 0
    for _, row in has_both.iterrows():
        if str(row['l2g_gene']) in str(row['eqtl_genes']).split(','):
            convergent += 1

    categories = ['L2G only', 'eQTL only', 'Both\n(L2G gene IN eQTL)', 'Both\n(L2G gene NOT in eQTL)', 'Neither']
    l2g_only = has_l2g - len(has_both)
    eqtl_only = has_eqtl - len(has_both)
    both_convergent = convergent
    both_divergent = len(has_both) - convergent
    neither = len(locus) - has_l2g - eqtl_only

    values = [l2g_only, eqtl_only, both_convergent, both_divergent, neither]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#cccccc']

    bars = ax.bar(range(len(categories)), values, color=colors)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Number of Loci', fontsize=12)
    ax.set_title('L2G vs eQTL Gene Assignment Convergence\n(250 PGC3 SCZ Loci)',
                 fontsize=13, fontweight='bold')

    for i, v in enumerate(values):
        if v > 0:
            ax.text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig7_L2G_eQTL_Convergence.pdf')
    fig.savefig(FIGURES_DIR / 'Fig7_L2G_eQTL_Convergence.png')
    plt.close(fig)
    print("  Fig7 saved")


def fig8_top_l2g_genes(locus):
    """Fig 8: Top L2G genes annotated with tissue class."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    with_l2g = locus[locus['l2g_gene'].notna()].sort_values('l2g_score', ascending=False).head(20).copy()

    if len(with_l2g) == 0:
        plt.close(fig)
        return

    # Color by tissue
    def get_color(row):
        if row.get('has_brain_eqtl', False) and row.get('has_immune_eqtl', False):
            return '#9467bd'
        if row.get('has_brain_eqtl', False):
            return '#1f77b4'
        if row.get('has_immune_eqtl', False):
            return '#d62728'
        return '#cccccc'

    colors = with_l2g.apply(get_color, axis=1).values
    labels = [f"{row['l2g_gene']} ({row['credible_set_id']})" for _, row in with_l2g.iterrows()]

    ax.barh(range(len(with_l2g)), with_l2g['l2g_score'].values, color=colors)
    ax.set_yticks(range(len(with_l2g)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('L2G Score', fontsize=12)
    ax.set_title('Top 20 L2G Gene Assignments for SCZ Loci\n(colored by eQTL tissue class)',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    brain_patch = mpatches.Patch(color='#1f77b4', label='Brain eQTL')
    immune_patch = mpatches.Patch(color='#d62728', label='Immune eQTL')
    both_patch = mpatches.Patch(color='#9467bd', label='Brain + Immune')
    none_patch = mpatches.Patch(color='#cccccc', label='No eQTL data')
    ax.legend(handles=[brain_patch, immune_patch, both_patch, none_patch],
              loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig8_Top_L2G_Genes.pdf')
    fig.savefig(FIGURES_DIR / 'Fig8_Top_L2G_Genes.png')
    plt.close(fig)
    print("  Fig8 saved")


def main():
    print("EVOSCZ Phase 7d: Generating Updated Publication Figures")

    df = pd.read_csv(MASTER, sep='\t')
    eqtl = pd.read_csv(EQTL_FILE, sep='\t')
    locus = get_locus(df)
    print(f"Master: {len(df)} variants, {len(locus)} loci")
    print(f"eQTL: {len(eqtl)} entries")

    print("\nGenerating figures...")
    fig5_tissue_distribution(eqtl)
    fig6_age_vs_tissue(locus)
    fig7_l2g_eqtl_convergence(locus)
    fig8_top_l2g_genes(locus)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
