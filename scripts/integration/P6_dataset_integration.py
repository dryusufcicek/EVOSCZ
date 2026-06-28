#!/usr/bin/env python3
"""
EVOSCZ Phase 6: Core Dataset Integration (v2)
==============================================
Merges all module outputs into a single master analytic table.

v2 changes (2026-04-11):
- Uses D5 v2 with percentile-based selection thresholds
  (replaces absolute thresholds that gave 100% sweep / 0% balancing)
- Uses C4 v2 with SCZ-specific OpenTargets colocalization
  (replaces unfiltered H4 that included non-SCZ colocalizations)
- Adds percentile ranks for continuous selection statistics
- Adds suggestive tiers for sensitivity analysis
- Improved Evolutionary_Class logic
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOD_A_MASTER = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"
MOD_A_AGES   = PROJECT_ROOT / "results/module_a/A2_bstat_corrected_ages.tsv"
MOD_D_SELECT = PROJECT_ROOT / "results/module_d/D5_integrated_selection_signatures.tsv"
MOD_C_PLEIO  = PROJECT_ROOT / "results/module_c/C4_opentargets_validated.tsv"

OUT_DIR      = PROJECT_ROOT / "results/integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASTER   = OUT_DIR / "P6_Master_Analytic_Table.tsv"


def main():
    print("=" * 60)
    print("EVOSCZ Phase 6: Core Dataset Integration (v2)")
    print("=" * 60)

    # 1. Load PGC3 base master (gene annotations + B-stat + GEVA ages)
    df_base = pd.read_csv(MOD_A_MASTER, sep='\t')
    print(f"Base master: {len(df_base)} variants")

    # 2. Merge age residuals from Module A
    df_ages = pd.read_csv(MOD_A_AGES, sep='\t')
    df_base = df_base.merge(df_ages[['rsid', 'age_residual', 'log_age']], on='rsid', how='left')

    # 3. Add Module D selection signatures (locus-level)
    df_sel = pd.read_csv(MOD_D_SELECT, sep='\t')
    sel_cols = ['credible_set_id', 'is_soft_sweep', 'is_recent_sweep', 'is_balancing',
                'is_trifecta', 'is_sweep_suggestive', 'is_balancing_suggestive',
                'max_abs_nsl', 'tajimas_d', 'h12',
                'nsl_percentile', 'tajd_percentile', 'h12_percentile']
    # Only use columns that exist
    sel_cols = [c for c in sel_cols if c in df_sel.columns]
    df_base = df_base.merge(df_sel[sel_cols], on='credible_set_id', how='left')

    # 4. Add Module C pleiotropy (variant-level, SCZ-filtered)
    df_c = pd.read_csv(MOD_C_PLEIO, sep='\t')
    c_cols = ['rsid', 'pleiotropy_type', 'OpenTargets_H4', 'OpenTargets_Trait',
              'DISEASE/TRAIT', 'OT_SCZ_Study', 'OT_N_SCZ_CredSets']
    c_cols = [c for c in c_cols if c in df_c.columns]
    df_master = df_base.merge(df_c[c_cols], on='rsid', how='left')

    # 5. Fill NA flags
    for col in ['is_soft_sweep', 'is_recent_sweep', 'is_balancing', 'is_trifecta',
                'is_sweep_suggestive', 'is_balancing_suggestive']:
        if col in df_master.columns:
            df_master[col] = df_master[col].fillna(False)
    df_master['OpenTargets_H4'] = df_master['OpenTargets_H4'].fillna(0)

    # 6. Compute Evolutionary Class
    def classify(row):
        classes = []

        # Age-based
        if pd.notnull(row.get('geva_age_years')) and row['geva_age_years'] > 500000:
            classes.append("Ancient")

        # Selection-based (use primary tier, top 5%)
        if row.get('is_recent_sweep', False):
            classes.append("Directional-Sweep")
        if row.get('is_soft_sweep', False):
            classes.append("Soft-Sweep")
        if row.get('is_balancing', False):
            classes.append("Balancing")

        # Pleiotropy-based (SCZ-specific H4)
        if row.get('OpenTargets_H4', 0) > 0.8:
            ptype = str(row.get('pleiotropy_type', ''))
            if 'Antagonistic' in ptype:
                classes.append("Antagonistic-Pleiotropic")
            elif 'Synergistic' in ptype:
                classes.append("Synergistic-Pleiotropic")
            else:
                classes.append("SCZ-Immune-Colocalized")

        return " + ".join(classes) if classes else "Neutral/Background"

    df_master['Evolutionary_Class'] = df_master.apply(classify, axis=1)

    # Save
    df_master.to_csv(OUT_MASTER, sep='\t', index=False)

    # Summary
    print(f"\nMaster table: {len(df_master)} variants, {len(df_master.columns)} columns")
    print(f"\n--- Evolutionary Class Distribution ---")
    class_counts = df_master['Evolutionary_Class'].value_counts()
    for cls, n in class_counts.items():
        print(f"  {cls:50s} {n:6d} ({n/len(df_master):5.1%})")

    # Key statistics
    n_ancient = df_master['geva_age_years'].gt(500000).sum()
    n_sweep = df_master['is_recent_sweep'].sum()
    n_soft = df_master['is_soft_sweep'].sum()
    n_balancing = df_master['is_balancing'].sum()
    n_trifecta = df_master['is_trifecta'].sum()
    n_h4 = (df_master['OpenTargets_H4'] > 0.8).sum()

    print(f"\n--- Summary Flags ---")
    print(f"Ancient (>500K yr):          {n_ancient}")
    print(f"Directional sweep (top 5%):  {n_sweep}")
    print(f"Soft sweep:                  {n_soft}")
    print(f"Balancing (top 5% D + old):  {n_balancing}")
    print(f"Trifecta:                    {n_trifecta}")
    print(f"SCZ-immune H4 > 0.8:         {n_h4}")
    print(f"\nSaved to: {OUT_MASTER}")


if __name__ == "__main__":
    main()
