#!/usr/bin/env python3
"""
EVOSCZ Module B2b: Control-Matched Desert Enrichment Testing
==============================================================
Re-tests Module B desert enrichment using MAF+L2 matched controls
instead of genome fraction as null.

This addresses the confound that PGC3 variants are non-randomly
distributed in the genome (enriched near genes, in regulatory regions)
and that gene-dense regions overlap introgression deserts independently
of any psychiatric-specific effect.

Compares: fraction of PGC3 variants in deserts vs. fraction of
          matched controls in deserts.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PGC3_MASTER = PROJECT_ROOT / "data/processed/pgc3_annotated_master.tsv"
CONTROLS = PROJECT_ROOT / "data/processed/matched_controls.tsv.gz"
DESERT_DIR = PROJECT_ROOT / "data/raw/annotations/introgression_deserts"
RESULTS_DIR = PROJECT_ROOT / "results/module_b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_bed(path):
    regions = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 3:
                chrom = fields[0].replace('chr', '')
                try:
                    regions.append((int(chrom), int(fields[1]), int(fields[2])))
                except ValueError:
                    continue
    return regions


def in_any_desert(chrom, pos, deserts_by_chr):
    for start, end in deserts_by_chr.get(chrom, []):
        if start <= pos < end:
            return True
    return False


def organize_deserts(regions):
    by_chr = {}
    for chrom, start, end in regions:
        by_chr.setdefault(chrom, []).append((start, end))
    for chrom in by_chr:
        by_chr[chrom].sort()
    return by_chr


def main():
    print("=" * 60)
    print("EVOSCZ B2b: Control-Matched Desert Enrichment")
    print("=" * 60)

    # Load PGC3 variants
    pgc3 = pd.read_csv(PGC3_MASTER, sep='\t')
    print(f"PGC3 variants: {len(pgc3):,}")

    # Load matched controls
    controls = pd.read_csv(CONTROLS, sep='\t', compression='gzip')
    print(f"Matched controls: {len(controls):,}")

    # Desert tiers
    desert_tiers = {}
    tier_files = {
        'Tier1': 'consensus_deserts_tier1_hg19.bed',
        'Tier2': 'consensus_deserts_tier2_hg19.bed',
        'Tier3': 'consensus_deserts_tier3_hg19.bed'
    }
    for tier, fname in tier_files.items():
        path = DESERT_DIR / fname
        if path.exists():
            desert_tiers[tier] = path
    print(f"Desert tiers found: {list(desert_tiers.keys())}")

    results_table = []

    for tier_name, desert_path in desert_tiers.items():
        print(f"\n--- {tier_name} ---")
        regions = load_bed(desert_path)
        deserts = organize_deserts(regions)

        # Count desert bp
        total_bp = sum(end - start for chr_regions in deserts.values() for start, end in chr_regions)
        genome_frac = total_bp / 3.0e9

        # PGC3 in deserts
        pgc3_in = pgc3.apply(lambda r: in_any_desert(r['chr'], r['pos'], deserts), axis=1)
        n_pgc3_in = pgc3_in.sum()
        pct_pgc3 = n_pgc3_in / len(pgc3) * 100

        # Controls in deserts
        ctrl_in = controls.apply(
            lambda r: in_any_desert(r['control_chr'], r['control_pos'], deserts), axis=1
        )
        n_ctrl_in = ctrl_in.sum()
        pct_ctrl = n_ctrl_in / len(controls) * 100

        # Fisher's exact test: PGC3 vs Controls
        table = [
            [n_pgc3_in, len(pgc3) - n_pgc3_in],
            [n_ctrl_in, len(controls) - n_ctrl_in]
        ]
        odds_ratio, fisher_p = stats.fisher_exact(table, alternative='greater')

        # Also: per-variant permutation approach
        # For each PGC3 variant, what fraction of its 100 controls are in desert?
        ctrl_per_variant = controls.copy()
        ctrl_per_variant['in_desert'] = ctrl_in.values
        ctrl_rates = ctrl_per_variant.groupby('pgc3_rsid')['in_desert'].mean()
        mean_ctrl_rate = ctrl_rates.mean()

        enrichment_vs_ctrl = (pct_pgc3 / pct_ctrl) if pct_ctrl > 0 else float('inf')
        enrichment_vs_genome = (pct_pgc3 / 100) / genome_frac if genome_frac > 0 else float('inf')

        print(f"  Desert regions: {len(regions)}, covering {genome_frac*100:.2f}% of genome")
        print(f"  PGC3 in desert:    {n_pgc3_in}/{len(pgc3)} ({pct_pgc3:.2f}%)")
        print(f"  Controls in desert: {n_ctrl_in}/{len(controls)} ({pct_ctrl:.2f}%)")
        print(f"  Enrichment vs genome fraction: {enrichment_vs_genome:.2f}x")
        print(f"  Enrichment vs MAF+L2 matched controls: {enrichment_vs_ctrl:.2f}x")
        print(f"  Fisher's exact (PGC3 vs controls): OR={odds_ratio:.2f}, p={fisher_p:.2e}")
        print(f"  Mean per-variant control desert rate: {mean_ctrl_rate:.4f}")

        results_table.append({
            'tier': tier_name,
            'n_pgc3_in_desert': n_pgc3_in,
            'n_pgc3_total': len(pgc3),
            'pct_pgc3': pct_pgc3,
            'n_ctrl_in_desert': n_ctrl_in,
            'n_ctrl_total': len(controls),
            'pct_ctrl': pct_ctrl,
            'genome_fraction_pct': genome_frac * 100,
            'enrichment_vs_genome': enrichment_vs_genome,
            'enrichment_vs_controls': enrichment_vs_ctrl,
            'fisher_OR': odds_ratio,
            'fisher_p': fisher_p
        })

    out = pd.DataFrame(results_table)
    outpath = RESULTS_DIR / "B2b_controlled_enrichment.tsv"
    out.to_csv(outpath, sep='\t', index=False)
    print(f"\nSaved to: {outpath}")


if __name__ == "__main__":
    main()
