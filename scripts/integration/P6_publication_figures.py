#!/usr/bin/env python3
"""
EVOSCZ Phase 6: Publication Figures (v2)
=========================================
Updated to reflect v2 corrections:
- Percentile-based selection thresholds
- SCZ-specific OpenTargets H4 scores
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_MASTER = PROJECT_ROOT / "results/integration/P6_Master_Analytic_Table.tsv"
FIGURES_DIR  = PROJECT_ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def plot_age_vs_selection(df):
    """Fig 1: Age vs nSL with evolutionary classification."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    locus = df.sort_values('pip', ascending=False).drop_duplicates(subset=['credible_set_id']).copy()
    locus = locus.dropna(subset=['geva_age_years', 'max_abs_nsl'])

    # Color by evolutionary class
    def get_category(row):
        if row.get('is_trifecta', False):
            return 'Trifecta'
        if row.get('is_balancing', False):
            return 'Balancing'
        if row.get('is_soft_sweep', False):
            return 'Soft Sweep'
        if row.get('is_recent_sweep', False):
            return 'Directional Sweep'
        return 'Background'

    locus['selection_cat'] = locus.apply(get_category, axis=1)

    palette = {
        'Background': '#cccccc',
        'Directional Sweep': '#1f77b4',
        'Soft Sweep': '#ff7f0e',
        'Balancing': '#2ca02c',
        'Trifecta': '#d62728'
    }
    order = ['Background', 'Directional Sweep', 'Soft Sweep', 'Balancing', 'Trifecta']

    for cat in order:
        subset = locus[locus['selection_cat'] == cat]
        if len(subset) > 0:
            ax.scatter(subset['geva_age_years'], subset['max_abs_nsl'],
                       c=palette[cat], label=f"{cat} (n={len(subset)})",
                       alpha=0.7, s=60, edgecolor='white', linewidth=0.3, zorder=2 if cat == 'Background' else 3)

    # Highlight SCZ-immune colocalized
    pleio = locus[locus['OpenTargets_H4'] > 0.8]
    if not pleio.empty:
        ax.scatter(pleio['geva_age_years'], pleio['max_abs_nsl'],
                   c='gold', marker='*', s=200, edgecolor='black', linewidth=0.5,
                   label=f'SCZ-Immune H4>0.8 (n={len(pleio)})', zorder=4)

    # Percentile threshold lines
    nsl_95 = locus['max_abs_nsl'].quantile(0.95)
    ax.axhline(nsl_95, color='#1f77b4', linestyle='--', alpha=0.5, label=f'nSL 95th pctile ({nsl_95:.1f})')

    ax.set_xscale('log')
    ax.set_title('PGC3 Schizophrenia Risk Loci: Allele Age vs Selection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estimated Allele Age (Years, GEVA)', fontsize=12)
    ax.set_ylabel('max |nSL| (200kb window)', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig1_Age_vs_Selection.pdf')
    fig.savefig(FIGURES_DIR / 'Fig1_Age_vs_Selection.png')
    plt.close(fig)


def plot_pleiotropy_breakdown(df):
    """Fig 2: Pleiotropy directionality for SCZ-specific H4>0.8."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ot_valid = df[df['OpenTargets_H4'] > 0.8].drop_duplicates(subset=['rsid']).copy()

    def classify(val):
        v = str(val)
        if 'Antagonistic' in v:
            return 'Antagonistic\n(SCZ risk = Immune protective)'
        elif 'Synergistic' in v:
            return 'Synergistic\n(SCZ risk = Immune risk)'
        return 'SCZ-Immune Colocalized\n(direction unknown)'

    ot_valid['P_Class'] = ot_valid['pleiotropy_type'].apply(classify)
    counts = ot_valid['P_Class'].value_counts()

    colors = []
    for x in counts.index:
        if 'Antagonistic' in x:
            colors.append('#2ca02c')
        elif 'Synergistic' in x:
            colors.append('#d62728')
        else:
            colors.append('#7f7f7f')

    bars = ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=10)
    ax.set_title(f'SCZ-Immune Colocalization Directionality\n(n={len(ot_valid)} unique variants, Bayesian H4 > 0.8)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Variants', fontsize=12)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold', fontsize=12)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig2_Pleiotropy_Direction.pdf')
    fig.savefig(FIGURES_DIR / 'Fig2_Pleiotropy_Direction.png')
    plt.close(fig)


def plot_top_traits(df):
    """Fig 3: Top immune traits colocalized with SCZ."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ot_valid = df[df['OpenTargets_H4'] > 0.8].drop_duplicates(subset=['rsid']).copy()
    # Clean trait names
    ot_valid['trait_clean'] = ot_valid['OpenTargets_Trait'].str.strip()
    traits = ot_valid['trait_clean'].value_counts().head(12)

    sns.barplot(y=traits.index, x=traits.values, palette='viridis', ax=ax)
    ax.set_title('Top Immune Traits Colocalized with Schizophrenia\n(SCZ credible set → Immune, H4 > 0.8)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of SCZ Loci', fontsize=12)
    ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig3_Top_Immune_Traits.pdf')
    fig.savefig(FIGURES_DIR / 'Fig3_Top_Immune_Traits.png')
    plt.close(fig)


def plot_evolutionary_classes(df):
    """Fig 4: Evolutionary class distribution."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Simplify classes for visualization
    def simplify(cls):
        if 'Trifecta' in str(cls) or ('Balancing' in str(cls) and 'Sweep' in str(cls)):
            return 'Trifecta (Balancing + Sweep)'
        if 'Balancing' in str(cls):
            return 'Balancing Selection'
        if 'Directional-Sweep' in str(cls):
            return 'Directional Sweep'
        if 'Soft-Sweep' in str(cls):
            return 'Soft Sweep'
        if 'Pleiotropic' in str(cls) or 'Colocalized' in str(cls):
            return 'SCZ-Immune Pleiotropic'
        if 'Ancient' in str(cls):
            return 'Ancient (>500K yr)'
        return 'Neutral/Background'

    df['simple_class'] = df['Evolutionary_Class'].apply(simplify)
    counts = df['simple_class'].value_counts()

    colors_map = {
        'Ancient (>500K yr)': '#8c564b',
        'Neutral/Background': '#cccccc',
        'Directional Sweep': '#1f77b4',
        'Soft Sweep': '#ff7f0e',
        'Balancing Selection': '#2ca02c',
        'Trifecta (Balancing + Sweep)': '#d62728',
        'SCZ-Immune Pleiotropic': '#9467bd'
    }

    colors = [colors_map.get(c, '#cccccc') for c in counts.index]
    bars = ax.barh(range(len(counts)), counts.values, color=colors)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=10)
    ax.set_xlabel('Number of Variants', fontsize=12)
    ax.set_title('Evolutionary Classification of PGC3 SCZ Risk Variants', fontsize=14, fontweight='bold')

    for i, v in enumerate(counts.values):
        ax.text(v + 50, i, f'{v:,} ({v/len(df)*100:.1f}%)', va='center', fontsize=9)

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'Fig4_Evolutionary_Classes.pdf')
    fig.savefig(FIGURES_DIR / 'Fig4_Evolutionary_Classes.png')
    plt.close(fig)


def main():
    print("EVOSCZ Phase 6: Generating Publication Figures (v2)")
    if not INPUT_MASTER.exists():
        print(f"Error: {INPUT_MASTER} not found.")
        return

    df = pd.read_csv(INPUT_MASTER, sep='\t')
    print(f"Master table: {len(df)} variants")

    print("1. Fig1: Age vs Selection...")
    plot_age_vs_selection(df)

    print("2. Fig2: Pleiotropy Breakdown...")
    plot_pleiotropy_breakdown(df)

    print("3. Fig3: Top Immune Traits...")
    plot_top_traits(df)

    print("4. Fig4: Evolutionary Classes...")
    plot_evolutionary_classes(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
