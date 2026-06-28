#!/usr/bin/env python3
"""
EVOSCZ Phase 7b: eQTL Tissue/Cell-Type Mapping
================================================
For each PGC3 lead variant, queries ALL credible sets (not just SCZ)
to find eQTL/sceQTL/sqtl/tuqtl credible sets. These reveal which
tissues and cell types show gene expression regulation at SCZ loci.

Strategy: A PGC3 variant that is also an eQTL lead in dorsolateral
prefrontal cortex (DLPFC) establishes a direct regulatory link
between the SCZ risk allele and gene expression in a disease-relevant
tissue.

NOTE: OpenTargets returns max 50 credible sets per query. For variants
with >50 credible sets (like rs13107325 with 1957), we cannot capture
all eQTL tissues. We flag these for manual review.
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAD_SNPS = PROJECT_ROOT / "data/processed/pgc3_lead_snps.tsv"
P7A_CHECKPOINT = PROJECT_ROOT / "results/phase7/.P7a_checkpoint.json"
OUTPUT = PROJECT_ROOT / "results/phase7/P7b_eqtl_tissue_map.tsv"
CHECKPOINT = PROJECT_ROOT / "results/phase7/.P7b_checkpoint.json"
URL = "https://api.platform.opentargets.org/api/v4/graphql"

BRAIN_TERMS = [
    'brain', 'cortex', 'cerebellum', 'hippocampus', 'hypothalamus',
    'frontal', 'temporal', 'prefrontal', 'amygdala', 'striatum',
    'caudate', 'putamen', 'substantia nigra', 'nucleus accumbens',
    'neuron', 'astrocyte', 'microglia', 'oligodendrocyte',
    'neural', 'cerebral', 'spinal cord', 'neuroblastoma'
]
IMMUNE_TERMS = [
    'monocyte', 'macrophage', 'microglia', 'lymphocyte', 'T cell',
    'B cell', 'NK cell', 'neutrophil', 'dendritic', 'mast cell',
    'eosinophil', 'basophil', 'spleen', 'thymus', 'lymph',
    'bone marrow', 'immune', 'regulatory T', 'helper', 'follicular',
    'CD4', 'CD8', 'CD14', 'CD16', 'natural killer'
]


def classify_tissue(name):
    if not name:
        return 'unknown'
    nl = name.lower()
    if any(t in nl for t in BRAIN_TERMS):
        return 'brain'
    if any(t in nl for t in IMMUNE_TERMS):
        return 'immune'
    return 'other'


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT, 'w') as f:
        json.dump(data, f)


def query_eqtl_credsets(vid):
    """Get all eQTL/QTL credible sets for a variant."""
    q = """
    query {
      variant(variantId: "%s") {
        credibleSets(page: {index: 0, size: 50}) {
          count
          rows {
            study {
              id
              traitFromSource
              studyType
              biosample { biosampleId biosampleName }
              target { approvedSymbol id }
            }
          }
        }
      }
    }
    """ % vid
    try:
        res = requests.post(URL, json={'query': q}, timeout=20).json()
        cs = res.get('data', {}).get('variant', {}).get('credibleSets', {})
        total = cs.get('count', 0)
        rows = cs.get('rows', [])
        return total, rows
    except Exception:
        return 0, []


def main():
    print("=" * 60)
    print("EVOSCZ Phase 7b: eQTL Tissue/Cell-Type Mapping")
    print("=" * 60)

    leads = pd.read_csv(LEAD_SNPS, sep='\t')
    print(f"Lead loci: {len(leads)}")

    # Load P7a checkpoint to get variant IDs
    p7a = {}
    if P7A_CHECKPOINT.exists():
        with open(P7A_CHECKPOINT) as f:
            p7a = json.load(f)

    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} done")

    all_results = []
    n_with_eqtl = 0
    n_with_brain = 0
    n_truncated = 0

    for idx, row in leads.iterrows():
        rsid = row['rsid']
        csid = row['credible_set_id']

        if csid in checkpoint:
            all_results.extend(checkpoint[csid].get('results', []))
            if checkpoint[csid].get('has_eqtl'):
                n_with_eqtl += 1
            if checkpoint[csid].get('has_brain'):
                n_with_brain += 1
            if checkpoint[csid].get('truncated'):
                n_truncated += 1
            continue

        # Get variant ID from P7a
        vid = p7a.get(csid, {}).get('vid')
        if not vid:
            checkpoint[csid] = {'results': [], 'has_eqtl': False,
                                'has_brain': False, 'truncated': False}
            continue

        total, rows = query_eqtl_credsets(vid)
        truncated = total > 50

        eqtl_entries = []
        for r in rows:
            s = r.get('study', {})
            st = s.get('studyType', '')
            if st not in ('eqtl', 'sceqtl', 'sqtl', 'tuqtl', 'pqtl'):
                continue
            bs = s.get('biosample', {}) or {}
            tgt = s.get('target', {}) or {}
            bname = bs.get('biosampleName', '')
            entry = {
                'credible_set_id': csid,
                'lead_rsid': rsid,
                'eqtl_type': st,
                'eqtl_gene': tgt.get('approvedSymbol', ''),
                'eqtl_ensembl': tgt.get('id', ''),
                'biosample_name': bname,
                'biosample_id': bs.get('biosampleId', ''),
                'tissue_class': classify_tissue(bname),
                'total_credsets': total,
                'truncated': truncated
            }
            eqtl_entries.append(entry)

        has_eqtl = len(eqtl_entries) > 0
        has_brain = any(e['tissue_class'] == 'brain' for e in eqtl_entries)

        if has_eqtl:
            n_with_eqtl += 1
        if has_brain:
            n_with_brain += 1
        if truncated:
            n_truncated += 1

        all_results.extend(eqtl_entries)
        checkpoint[csid] = {
            'results': eqtl_entries,
            'has_eqtl': has_eqtl,
            'has_brain': has_brain,
            'truncated': truncated
        }

        if has_eqtl:
            brain_n = sum(1 for e in eqtl_entries if e['tissue_class'] == 'brain')
            immune_n = sum(1 for e in eqtl_entries if e['tissue_class'] == 'immune')
            genes = set(e['eqtl_gene'] for e in eqtl_entries if e['eqtl_gene'])
            trunc_flag = ' [TRUNCATED]' if truncated else ''
            print(f"  [{idx+1}/255] {csid} {rsid}: {len(eqtl_entries)} eQTLs "
                  f"({brain_n} brain, {immune_n} immune) genes={genes}{trunc_flag}")

        time.sleep(0.4)
        if (idx + 1) % 25 == 0:
            save_checkpoint(checkpoint)
            print(f"  ... checkpoint ({idx+1}/255)")

    save_checkpoint(checkpoint)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT, sep='\t', index=False)
    else:
        df = pd.DataFrame()

    # Summary
    print(f"\n{'='*60}")
    print(f"Phase 7b Summary")
    print(f"{'='*60}")
    print(f"Loci with any eQTL:   {n_with_eqtl}/255 ({n_with_eqtl/255*100:.1f}%)")
    print(f"Loci with brain eQTL: {n_with_brain}/255 ({n_with_brain/255*100:.1f}%)")
    print(f"Loci truncated (>50): {n_truncated}")
    print(f"Total eQTL entries:   {len(df)}")

    if not df.empty:
        print(f"\n--- Tissue Class Distribution ---")
        tc = df['tissue_class'].value_counts()
        for cls, n in tc.items():
            print(f"  {cls:10s}: {n:4d} ({n/len(df)*100:.1f}%)")

        print(f"\n--- Top Brain Tissues ---")
        brain = df[df['tissue_class'] == 'brain']
        if not brain.empty:
            for tissue, n in brain['biosample_name'].value_counts().head(10).items():
                n_loci = brain[brain['biosample_name'] == tissue]['credible_set_id'].nunique()
                print(f"  {tissue:45s}: {n:3d} entries, {n_loci:3d} unique loci")

        print(f"\n--- Top Immune Tissues ---")
        immune = df[df['tissue_class'] == 'immune']
        if not immune.empty:
            for tissue, n in immune['biosample_name'].value_counts().head(10).items():
                n_loci = immune[immune['biosample_name'] == tissue]['credible_set_id'].nunique()
                print(f"  {tissue:45s}: {n:3d} entries, {n_loci:3d} unique loci")

        print(f"\n--- Top eQTL Genes ---")
        for gene, n in df['eqtl_gene'].value_counts().head(15).items():
            n_loci = df[df['eqtl_gene'] == gene]['credible_set_id'].nunique()
            tissues = df[df['eqtl_gene'] == gene]['tissue_class'].value_counts().to_dict()
            print(f"  {gene:15s}: {n:3d} entries, {n_loci:2d} loci, tissues={tissues}")

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
