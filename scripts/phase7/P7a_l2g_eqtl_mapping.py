#!/usr/bin/env python3
"""
EVOSCZ Phase 7a: L2G Gene Assignment + eQTL Tissue Mapping
============================================================
For each PGC3 lead locus (255 loci), queries OpenTargets Platform v4:

1. L2G (Locus-to-Gene): Machine learning-based causal gene prediction.
   Uses chromatin interaction, eQTL, distance, and variant effect features
   to assign the most likely causal gene — far more accurate than
   "nearest gene" which ignores regulatory architecture.

2. eQTL Colocalisation: For each SCZ credible set, retrieves
   colocalisations with eQTL/sceQTL studies. The otherStudyLocus
   biosample field reveals which tissue/cell-type shows the eQTL
   signal, enabling cellular mapping of SCZ risk loci.

Output: Per-locus table with L2G gene, score, and eQTL tissue breakdown.
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAD_SNPS = PROJECT_ROOT / "data/processed/pgc3_lead_snps.tsv"
OUTPUT_L2G = PROJECT_ROOT / "results/phase7/P7a_l2g_assignments.tsv"
OUTPUT_EQTL = PROJECT_ROOT / "results/phase7/P7a_eqtl_tissues.tsv"
CHECKPOINT = PROJECT_ROOT / "results/phase7/.P7a_checkpoint.json"
URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Brain-related UBERON/CL terms for classification
BRAIN_TERMS = [
    'brain', 'cortex', 'cerebellum', 'hippocampus', 'hypothalamus',
    'frontal', 'temporal', 'prefrontal', 'amygdala', 'striatum',
    'caudate', 'putamen', 'substantia nigra', 'nucleus accumbens',
    'neuron', 'astrocyte', 'microglia', 'oligodendrocyte',
    'neural', 'cerebral', 'spinal cord'
]

IMMUNE_TERMS = [
    'monocyte', 'macrophage', 'microglia', 'lymphocyte', 'T cell',
    'B cell', 'NK cell', 'neutrophil', 'dendritic', 'mast cell',
    'eosinophil', 'basophil', 'blood', 'spleen', 'thymus',
    'lymph node', 'bone marrow', 'immune'
]


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT, 'w') as f:
        json.dump(data, f)


def get_variant_id(rsid):
    q = """query { search(queryString: "%s") { hits { id entity } } }""" % rsid
    try:
        res = requests.post(URL, json={'query': q}, timeout=15).json()
        for hit in res.get('data', {}).get('search', {}).get('hits', []):
            if hit['entity'] == 'variant':
                return hit['id']
    except Exception:
        pass
    return None


def query_locus(vid):
    """
    Single query to get:
    1. SCZ credible sets with L2G predictions
    2. SCZ credible sets' colocalisations with eQTL studies
    """
    q = """
    query getLocus($variantId: String!) {
      variant(variantId: $variantId) {
        credibleSets(page: {index: 0, size: 50}) {
          rows {
            studyLocusId
            study {
              id
              traitFromSource
              studyType
            }
            l2GPredictions(page: {index: 0, size: 10}) {
              rows {
                score
                target {
                  id
                  approvedSymbol
                  biotype
                }
              }
            }
            colocalisation(page: {index: 0, size: 100}) {
              rows {
                h4
                otherStudyLocus {
                  study {
                    id
                    traitFromSource
                    studyType
                    biosample {
                      biosampleId
                      biosampleName
                    }
                    target {
                      approvedSymbol
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    try:
        res = requests.post(URL, json={'query': q, 'variables': {'variantId': vid}},
                            timeout=30).json()
        return res.get('data', {}).get('variant', {}).get('credibleSets', {}).get('rows', [])
    except Exception as e:
        print(f"  Error: {e}")
        return []


def classify_tissue(biosample_name):
    """Classify a biosample as brain, immune, or other."""
    if not biosample_name:
        return 'unknown'
    name_lower = biosample_name.lower()
    if any(t in name_lower for t in BRAIN_TERMS):
        return 'brain'
    if any(t in name_lower for t in IMMUNE_TERMS):
        return 'immune'
    return 'other'


def process_locus(cs_rows):
    """Extract L2G and eQTL data from SCZ credible sets."""
    l2g_results = []
    eqtl_results = []

    for row in cs_rows:
        study = row.get('study', {})
        trait = (study.get('traitFromSource', '') or '').lower()
        study_type = study.get('studyType', '')

        # Only process SCZ GWAS credible sets
        if 'schizophreni' not in trait or study_type != 'gwas':
            continue

        scz_study_id = study.get('id', '')
        slid = row.get('studyLocusId', '')

        # 1. L2G predictions
        l2g_rows = row.get('l2GPredictions', {}).get('rows', [])
        for l2g in l2g_rows:
            target = l2g.get('target', {})
            l2g_results.append({
                'scz_study': scz_study_id,
                'l2g_gene': target.get('approvedSymbol', ''),
                'l2g_ensembl': target.get('id', ''),
                'l2g_biotype': target.get('biotype', ''),
                'l2g_score': l2g.get('score', 0)
            })

        # 2. eQTL colocalisations
        coloc_rows = row.get('colocalisation', {}).get('rows', [])
        for coloc in coloc_rows:
            h4 = coloc.get('h4', 0) or 0
            other = coloc.get('otherStudyLocus', {}).get('study', {})
            other_type = other.get('studyType', '')

            # Only eQTL / sceQTL / sqtl / tuqtl colocalisations
            if other_type not in ('eqtl', 'sceqtl', 'sqtl', 'tuqtl', 'pqtl'):
                continue

            biosample = other.get('biosample', {}) or {}
            target_gene = other.get('target', {})

            eqtl_results.append({
                'scz_study': scz_study_id,
                'eqtl_study': other.get('id', ''),
                'eqtl_type': other_type,
                'eqtl_trait': other.get('traitFromSource', ''),
                'eqtl_gene': target_gene.get('approvedSymbol', '') if target_gene else '',
                'biosample_id': biosample.get('biosampleId', ''),
                'biosample_name': biosample.get('biosampleName', ''),
                'tissue_class': classify_tissue(biosample.get('biosampleName', '')),
                'coloc_h4': h4
            })

    return l2g_results, eqtl_results


def main():
    print("=" * 60)
    print("EVOSCZ Phase 7a: L2G + eQTL Tissue Mapping")
    print("=" * 60)

    leads = pd.read_csv(LEAD_SNPS, sep='\t')
    print(f"Lead loci: {len(leads)}")

    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} loci done")

    all_l2g = []
    all_eqtl = []
    n_with_scz = 0
    n_with_l2g = 0
    n_with_eqtl = 0

    for idx, row in leads.iterrows():
        rsid = row['rsid']
        csid = row['credible_set_id']

        if csid in checkpoint:
            # Restore from checkpoint
            cached = checkpoint[csid]
            all_l2g.extend(cached.get('l2g', []))
            all_eqtl.extend(cached.get('eqtl', []))
            if cached.get('has_scz'):
                n_with_scz += 1
            if cached.get('has_l2g'):
                n_with_l2g += 1
            if cached.get('has_eqtl'):
                n_with_eqtl += 1
            continue

        # Get OpenTargets variant ID
        vid = get_variant_id(rsid)
        if not vid:
            checkpoint[csid] = {'l2g': [], 'eqtl': [], 'has_scz': False,
                                'has_l2g': False, 'has_eqtl': False, 'vid': None}
            time.sleep(0.3)
            if (idx + 1) % 20 == 0:
                save_checkpoint(checkpoint)
            continue

        # Query OpenTargets
        cs_rows = query_locus(vid)
        l2g, eqtl = process_locus(cs_rows)

        # Add locus identifiers
        for item in l2g:
            item['credible_set_id'] = csid
            item['lead_rsid'] = rsid
        for item in eqtl:
            item['credible_set_id'] = csid
            item['lead_rsid'] = rsid

        has_scz = len(l2g) > 0 or len(eqtl) > 0
        has_l2g = len(l2g) > 0
        has_eqtl = len(eqtl) > 0

        if has_scz:
            n_with_scz += 1
        if has_l2g:
            n_with_l2g += 1
        if has_eqtl:
            n_with_eqtl += 1

        all_l2g.extend(l2g)
        all_eqtl.extend(eqtl)

        checkpoint[csid] = {
            'l2g': l2g, 'eqtl': eqtl,
            'has_scz': has_scz, 'has_l2g': has_l2g, 'has_eqtl': has_eqtl,
            'vid': vid
        }

        # Progress
        top_gene = l2g[0]['l2g_gene'] if l2g else '-'
        top_score = f"{l2g[0]['l2g_score']:.3f}" if l2g else '-'
        n_eqtl_tissues = len(set(e['biosample_name'] for e in eqtl if e['biosample_name']))
        brain_eqtl = sum(1 for e in eqtl if e['tissue_class'] == 'brain')

        if has_l2g or has_eqtl:
            print(f"  [{idx+1}/255] {csid} {rsid}: L2G={top_gene}({top_score}) "
                  f"eQTL={len(eqtl)} tissues ({brain_eqtl} brain)")

        time.sleep(0.5)
        if (idx + 1) % 20 == 0:
            save_checkpoint(checkpoint)
            print(f"  ... checkpoint ({idx+1}/255)")

    save_checkpoint(checkpoint)

    # Save L2G results
    if all_l2g:
        df_l2g = pd.DataFrame(all_l2g)
        # Keep best L2G per locus (highest score across SCZ studies)
        df_l2g_best = df_l2g.sort_values('l2g_score', ascending=False).drop_duplicates(
            subset=['credible_set_id', 'l2g_gene'])
        df_l2g_best.to_csv(OUTPUT_L2G, sep='\t', index=False)
    else:
        df_l2g_best = pd.DataFrame()

    # Save eQTL results
    if all_eqtl:
        df_eqtl = pd.DataFrame(all_eqtl)
        df_eqtl.to_csv(OUTPUT_EQTL, sep='\t', index=False)
    else:
        df_eqtl = pd.DataFrame()

    # Summary
    print(f"\n{'='*60}")
    print(f"Phase 7a Summary")
    print(f"{'='*60}")
    print(f"Total loci queried:        255")
    print(f"Loci with SCZ credible sets: {n_with_scz}")
    print(f"Loci with L2G predictions:   {n_with_l2g}")
    print(f"Loci with eQTL coloc:        {n_with_eqtl}")

    if not df_l2g_best.empty:
        print(f"\n--- L2G Gene Assignments ---")
        print(f"Total L2G predictions: {len(df_l2g_best)}")
        top = df_l2g_best.sort_values('l2g_score', ascending=False).head(10)
        for _, r in top.iterrows():
            print(f"  {r['credible_set_id']:10s} {r['lead_rsid']:15s} -> "
                  f"{r['l2g_gene']:15s} (score={r['l2g_score']:.3f}, {r['l2g_biotype']})")

    if not df_eqtl.empty:
        print(f"\n--- eQTL Tissue Distribution ---")
        print(f"Total eQTL colocalisations: {len(df_eqtl)}")
        print(f"  H4 > 0.8: {(df_eqtl['coloc_h4'] > 0.8).sum()}")

        tissue_counts = df_eqtl['tissue_class'].value_counts()
        print(f"\n  Tissue class breakdown:")
        for cls, n in tissue_counts.items():
            print(f"    {cls:10s}: {n:4d} ({n/len(df_eqtl)*100:.1f}%)")

        brain_eqtl = df_eqtl[df_eqtl['tissue_class'] == 'brain']
        if not brain_eqtl.empty:
            print(f"\n  Top brain tissues:")
            brain_tissues = brain_eqtl['biosample_name'].value_counts().head(10)
            for tissue, n in brain_tissues.items():
                mean_h4 = brain_eqtl[brain_eqtl['biosample_name'] == tissue]['coloc_h4'].mean()
                print(f"    {tissue:40s}: {n:3d} colocs (mean H4={mean_h4:.3f})")

        immune_eqtl = df_eqtl[df_eqtl['tissue_class'] == 'immune']
        if not immune_eqtl.empty:
            print(f"\n  Top immune tissues:")
            immune_tissues = immune_eqtl['biosample_name'].value_counts().head(10)
            for tissue, n in immune_tissues.items():
                mean_h4 = immune_eqtl[immune_eqtl['biosample_name'] == tissue]['coloc_h4'].mean()
                print(f"    {tissue:40s}: {n:3d} colocs (mean H4={mean_h4:.3f})")

    print(f"\nSaved: {OUTPUT_L2G}")
    print(f"Saved: {OUTPUT_EQTL}")


if __name__ == "__main__":
    main()
