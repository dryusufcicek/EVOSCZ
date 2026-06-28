#!/usr/bin/env python3
"""
EVOSCZ Module C3: OpenTargets Bayesian Colocalization Validation (v2)
=====================================================================
For each pleiotropic PGC3 variant, queries the OpenTargets Platform
GraphQL API (v4) to retrieve Bayesian colocalization H4 scores
specifically between SCHIZOPHRENIA credible sets and IMMUNE trait
credible sets.

v2 changes (2026-04-11):
- Fixed critical bug: v1 retrieved ALL credible sets for a variant
  regardless of study/trait, so an H4 score could reflect e.g.
  eosinophil-neutrophil colocalization rather than SCZ-immune.
- Now filters parent credible sets to only those from schizophrenia
  GWAS studies (traitFromSource contains "schizophreni").
- Uses correct API field: study.id (not studyId, which was removed).
- Also retrieves L2G predictions for SCZ credible sets (for Phase 7).
- Added checkpoint/resume to avoid re-querying completed variants.
"""

import pandas as pd
import requests
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "results/module_c/C3_final_pleiotropy_direction.tsv"
OUTPUT_FILE = PROJECT_ROOT / "results/module_c/C4_opentargets_validated.tsv"
CHECKPOINT_FILE = PROJECT_ROOT / "results/module_c/.C3_checkpoint.json"
URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Immune/autoimmune trait keywords for filtering colocalization partners
IMMUNE_KEYWORDS = [
    'eosinophil', 'neutrophil', 'lymphocyte', 'leukocyte', 'white blood cell',
    'crohn', 'ulcerative colitis', 'rheumatoid', 'multiple sclerosis',
    'type 1 diabetes', 'psoriasis', 'c-reactive', 'inflammatory',
    'monocyte', 'basophil', 'platelet', 'lupus', 'celiac',
    'autoimmune', 'ankylosing', 'asthma', 'allerg'
]


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)


def get_variant_id(rsid):
    """Search OpenTargets for variant ID by rsID."""
    q = """
    query searchVariant($queryString: String!) {
      search(queryString: $queryString) {
        hits { id entity }
      }
    }
    """
    try:
        res = requests.post(URL, json={'query': q, 'variables': {"queryString": rsid}},
                            timeout=15).json()
        for hit in res.get('data', {}).get('search', {}).get('hits', []):
            if hit['entity'] == 'variant':
                return hit['id']
    except Exception as e:
        print(f"  Warning: search failed for {rsid}: {e}")
    return None


def get_scz_immune_coloc(vid):
    """
    Get colocalization H4 scores specifically between SCZ and immune traits.

    Strategy:
    1. Fetch all credible sets for the variant
    2. Filter to credible sets from schizophrenia studies only
    3. For each SCZ credible set, check colocalizations with immune traits
    4. Return the highest SCZ-immune H4 score
    """
    q = """
    query getColoc($variantId: String!) {
      variant(variantId: $variantId) {
        credibleSets(page: {index: 0, size: 100}) {
          rows {
            studyLocusId
            study { id traitFromSource }
            colocalisation(page: {index: 0, size: 100}) {
              rows {
                h4
                otherStudyLocus {
                  study { id traitFromSource }
                }
              }
            }
          }
        }
      }
    }
    """
    best_h4 = 0.0
    best_immune_trait = ""
    best_scz_study = ""
    n_scz_credsets = 0
    n_immune_colocs = 0

    try:
        res = requests.post(URL, json={'query': q, 'variables': {'variantId': vid}},
                            timeout=30).json()
        cs_rows = res.get('data', {}).get('variant', {}).get('credibleSets', {}).get('rows', [])

        for row in cs_rows:
            # FILTER: only process credible sets from schizophrenia studies
            parent_trait = (row.get('study', {}).get('traitFromSource', '') or '').lower()
            if 'schizophreni' not in parent_trait:
                continue

            n_scz_credsets += 1
            parent_study_id = row.get('study', {}).get('id', '')

            # Check colocalisations with immune traits
            coloc_rows = row.get('colocalisation', {}).get('rows', [])
            for coloc in coloc_rows:
                h4 = coloc.get('h4', 0) or 0
                other_study = coloc.get('otherStudyLocus', {}).get('study', {})
                other_trait = (other_study.get('traitFromSource', '') or '').lower()

                # Skip SCZ-SCZ colocalizations
                if 'schizophreni' in other_trait:
                    continue

                # Check if the other trait is immune/autoimmune
                is_immune = any(kw in other_trait for kw in IMMUNE_KEYWORDS)
                if is_immune:
                    n_immune_colocs += 1
                    if h4 > best_h4:
                        best_h4 = h4
                        best_immune_trait = other_study.get('traitFromSource', '')
                        best_scz_study = parent_study_id

    except Exception as e:
        print(f"  Warning: coloc query failed for {vid}: {e}")

    return {
        'h4': best_h4,
        'trait': best_immune_trait,
        'scz_study': best_scz_study,
        'n_scz_credsets': n_scz_credsets,
        'n_immune_colocs': n_immune_colocs
    }


def main():
    print("=" * 60)
    print("EVOSCZ Module C3: OpenTargets Colocalization (v2 — SCZ-filtered)")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, sep='\t')
    print(f"Input: {len(df)} pleiotropic loci from C2.")

    unique_rsids = df['rsid'].unique()
    print(f"Unique variants to query: {len(unique_rsids)}")

    # Load checkpoint for resume capability
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} variants already processed.")

    validations = {}
    for i, rsid in enumerate(unique_rsids):
        # Skip if already checkpointed
        if rsid in checkpoint:
            validations[rsid] = checkpoint[rsid]
            continue

        # Step 1: Get variant ID
        vid = get_variant_id(rsid)
        if not vid:
            result = {'h4': 0.0, 'trait': '', 'scz_study': '', 'vid': None,
                      'n_scz_credsets': 0, 'n_immune_colocs': 0}
            validations[rsid] = result
            checkpoint[rsid] = result
            if (i + 1) % 10 == 0:
                save_checkpoint(checkpoint)
            time.sleep(0.3)
            continue

        # Step 2: Get SCZ-specific immune colocalization
        result = get_scz_immune_coloc(vid)
        result['vid'] = vid
        validations[rsid] = result
        checkpoint[rsid] = result

        if result['h4'] > 0:
            print(f"  [{i+1}/{len(unique_rsids)}] {rsid} -> SCZ-immune H4: {result['h4']:.3f} "
                  f"({result['trait']}) [SCZ study: {result['scz_study']}]")
        elif result['n_scz_credsets'] == 0:
            print(f"  [{i+1}/{len(unique_rsids)}] {rsid} -> No SCZ credible sets found")

        # Checkpoint every 10 variants
        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint)
            print(f"  ... checkpoint saved ({i+1}/{len(unique_rsids)})")

        time.sleep(0.5)

    # Save final checkpoint
    save_checkpoint(checkpoint)

    # Map results back to dataframe
    df['OT_Variant_ID'] = df['rsid'].map(lambda x: validations.get(x, {}).get('vid'))
    df['OpenTargets_H4'] = df['rsid'].map(lambda x: validations.get(x, {}).get('h4', 0))
    df['OpenTargets_Trait'] = df['rsid'].map(lambda x: validations.get(x, {}).get('trait', ''))
    df['OT_SCZ_Study'] = df['rsid'].map(lambda x: validations.get(x, {}).get('scz_study', ''))
    df['OT_N_SCZ_CredSets'] = df['rsid'].map(lambda x: validations.get(x, {}).get('n_scz_credsets', 0))
    df['OT_N_Immune_Colocs'] = df['rsid'].map(lambda x: validations.get(x, {}).get('n_immune_colocs', 0))

    df.to_csv(OUTPUT_FILE, sep='\t', index=False)

    # Summary
    validated = df[df['OpenTargets_H4'] > 0.8].drop_duplicates(subset=['rsid'])
    no_scz = df[df['OT_N_SCZ_CredSets'] == 0].drop_duplicates(subset=['rsid'])
    has_scz_no_immune = df[(df['OT_N_SCZ_CredSets'] > 0) & (df['OpenTargets_H4'] == 0)].drop_duplicates(subset=['rsid'])

    print(f"\n{'='*60}")
    print(f"Results Summary (SCZ-specific colocalization)")
    print(f"{'='*60}")
    print(f"Total unique variants queried:           {len(unique_rsids)}")
    print(f"Variants with SCZ credible sets:         {len(unique_rsids) - len(no_scz)}")
    print(f"Variants WITHOUT SCZ credible sets:      {len(no_scz)}")
    print(f"SCZ-immune H4 > 0.8 (validated):         {len(validated)}")
    print(f"SCZ credible set found but no immune H4: {len(has_scz_no_immune)}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
