"""
API Clients — fetch variant/gene annotations without downloading full datasets
==============================================================================
Modules:
- OpenTargets Platform v4 GraphQL  : credible sets, coloc, L2G via genetics integration
- eQTL Catalogue v2 REST           : per-variant eQTL associations across ~150 datasets
- gnomAD GraphQL                   : per-gene constraint (pLI, LOEUF, mis_z) + per-variant AF
- Ensembl VEP REST                 : AlphaMissense + consequence per variant
- FinnGen REST                     : per-region phenotype assoc (immune disease coloc)

Coordinate notes:
- OpenTargets v4 uses GRCh38; our master is GRCh37 → use rsID lookup via search.
- gnomAD v4 uses GRCh38 for variant query; gene-level (constraint) is build-agnostic.
- eQTL Catalogue uses rsID directly.
"""

import json
import time
import requests
from typing import Optional, List, Dict, Any
from functools import lru_cache

# ─── Endpoints ────────────────────────────────────────────────────────────
OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
EQTL_API = "https://www.ebi.ac.uk/eqtl/api/v2"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
ENSEMBL_VEP = "https://rest.ensembl.org/vep/human/id"
FINNGEN_API = "https://results.finngen.fi/api"

DEFAULT_TIMEOUT = 60
RETRY_DELAY = 2.0


# ─── HTTP helpers ─────────────────────────────────────────────────────────
def _post_graphql(url, query, variables=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.post(url, json={"query": query, "variables": variables or {}},
                              timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                j = r.json()
                if "errors" not in j or "data" in j:
                    return j
            elif r.status_code == 429:
                # Rate limited — back off harder
                time.sleep(10 * (attempt + 1))
                continue
        except Exception:
            pass
        time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def _get_rest(url, params=None, retries=3, headers=None):
    h = headers or {"Accept": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, timeout=DEFAULT_TIMEOUT, headers=h)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ─── OpenTargets Platform v4 ──────────────────────────────────────────────
@lru_cache(maxsize=20000)
def ot_rsid_to_variantid(rsid: str) -> Optional[str]:
    """
    Resolve rsID → OT variant id (GRCh38 chrom_pos_ref_alt).
    Returns None if no variant found.
    """
    q = f'''{{ search(queryString: "{rsid}") {{
      hits {{ id name entity }}
    }} }}'''
    j = _post_graphql(OT_API, q)
    if not j or not j.get("data"):
        return None
    hits = j["data"].get("search", {}).get("hits", [])
    for h in hits:
        if h.get("entity") == "variant":
            return h.get("id")
    return None


@lru_cache(maxsize=10000)
def ot_variant_credible_sets(variant_id: str) -> Optional[List[Dict]]:
    """
    Return credible sets containing this GRCh38 variant + their colocalisations.
    """
    q = f'''{{ variant(variantId: "{variant_id}") {{
      id rsIds chromosome position
      credibleSets(page: {{index:0, size:50}}) {{
        count
        rows {{
          studyLocusId
          study {{ id traitFromSource studyType nSamples }}
          pValueMantissa pValueExponent
          isTransQtl
          locus(variantIds:["{variant_id}"]) {{
            rows {{ posteriorProbability pValueMantissa pValueExponent }}
          }}
          l2GPredictions(page: {{index:0, size:5}}) {{
            rows {{ score target {{ id approvedSymbol biotype }} }}
          }}
          colocalisation {{
            count
            rows {{
              otherStudyLocus {{
                studyLocusId
                study {{ id traitFromSource studyType }}
              }}
              h4 h3 clpp numberColocalisingVariants
              colocalisationMethod
            }}
          }}
        }}
      }}
    }} }}'''
    j = _post_graphql(OT_API, q)
    if not j or not j.get("data") or not j["data"].get("variant"):
        return None
    return j["data"]["variant"].get("credibleSets", {}).get("rows", [])


def ot_variant_full(rsid: str) -> Optional[Dict]:
    """
    One-call wrapper: rsID → all credible sets + L2G + coloc info.
    Returns dict with variant_id, credible_sets, summary stats.
    """
    vid = ot_rsid_to_variantid(rsid)
    if not vid:
        return None
    credsets = ot_variant_credible_sets(vid)
    if credsets is None:
        return {"variant_id": vid, "rsid": rsid, "credible_sets": [], "max_h4": None}

    # Compute summary
    max_h4 = None
    immune_h4_max = None
    h4_count = 0
    l2g_top_genes = []
    immune_keywords = ["immun", "inflam", "white blood", "monocyte", "neutrophil",
                       "lymphocyte", "platelet", "ulcerative", "crohn", "thyroid",
                       "lupus", "psoriasis", "rheumatoid", "asthma", "eczema"]
    for cs in credsets:
        for c in cs.get("colocalisation", {}).get("rows", []) or []:
            h4 = c.get("h4")
            if h4 is None:
                continue
            if h4 > 0.8:
                h4_count += 1
            if max_h4 is None or h4 > max_h4:
                max_h4 = h4
            other = c.get("otherStudyLocus", {})
            trait = (other.get("study", {}).get("traitFromSource") or "").lower()
            if any(k in trait for k in immune_keywords):
                if immune_h4_max is None or h4 > immune_h4_max:
                    immune_h4_max = h4
        for l2g in cs.get("l2GPredictions", {}).get("rows", []) or []:
            sym = l2g.get("target", {}).get("approvedSymbol")
            score = l2g.get("score")
            if sym:
                l2g_top_genes.append((sym, score))

    # Top L2G gene by score
    l2g_top_genes.sort(key=lambda x: -(x[1] or 0))
    top_l2g_gene = l2g_top_genes[0][0] if l2g_top_genes else None
    top_l2g_score = l2g_top_genes[0][1] if l2g_top_genes else None

    return {
        "rsid": rsid,
        "variant_id": vid,
        "n_credible_sets": len(credsets),
        "max_h4": max_h4,
        "immune_h4_max": immune_h4_max,
        "h4_above_08_count": h4_count,
        "top_l2g_gene": top_l2g_gene,
        "top_l2g_score": top_l2g_score,
        "credible_sets": credsets,
    }


# ─── eQTL Catalogue v2 ────────────────────────────────────────────────────
def eqtl_catalogue_for_rsid(rsid: str, p_threshold: float = 1e-4,
                            size: int = 200) -> Optional[List[Dict]]:
    """
    Fetch all significant eQTL associations for a variant across all R7 datasets.
    Returns: list of {dataset_id, gene_id, tissue, beta, pvalue, ...}
    """
    url = f"{EQTL_API}/associations"
    params = {"variant": rsid, "size": size, "p_upper": p_threshold}
    return _get_rest(url, params=params)


@lru_cache(maxsize=1)
def eqtl_catalogue_datasets() -> Optional[List[Dict]]:
    """Cache the dataset registry (~200 entries)."""
    return _get_rest(f"{EQTL_API}/datasets", params={"size": 500})


def eqtl_summary_for_rsid(rsid: str, p_threshold: float = 1e-4) -> Dict:
    """
    Variant-level eQTL summary: counts by tissue category + min p per category.
    """
    assocs = eqtl_catalogue_for_rsid(rsid, p_threshold)
    if not assocs:
        return {"rsid": rsid, "n_eqtl": 0}
    datasets = eqtl_catalogue_datasets()
    if not datasets:
        return {"rsid": rsid, "n_eqtl": len(assocs)}
    ds_map = {d["dataset_id"]: d for d in datasets}
    brain, blood, immune, other = [], [], [], []
    for a in assocs:
        d = ds_map.get(a.get("dataset_id"))
        if not d:
            continue
        tissue = (d.get("tissue_label") or "").lower()
        if "brain" in tissue or "neuron" in tissue or "cortex" in tissue or "psychencode" in tissue:
            brain.append(a)
        elif "blood" in tissue or "monocyte" in tissue or "lymph" in tissue or "macrophage" in tissue or "t_cell" in tissue or "b_cell" in tissue:
            blood.append(a)
        elif any(k in tissue for k in ["immun", "spleen", "thymus"]):
            immune.append(a)
        else:
            other.append(a)

    def min_p(lst):
        ps = [a.get("pvalue") for a in lst if a.get("pvalue")]
        return min(ps) if ps else None

    return {
        "rsid": rsid,
        "n_eqtl": len(assocs),
        "n_brain_eqtl": len(brain),
        "n_blood_eqtl": len(blood),
        "n_immune_eqtl": len(immune),
        "min_p_brain": min_p(brain),
        "min_p_blood": min_p(blood),
        "min_p_immune": min_p(immune),
        "associations": assocs,
    }


# ─── gnomAD ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=10000)
def gnomad_gene_constraint(gene_symbol: str) -> Optional[Dict]:
    """
    gnomAD v4 constraint scores for a gene (pLI, LOEUF, mis_z, etc.).
    """
    q = """query Constraint($s: String!) {
      gene(gene_symbol: $s, reference_genome: GRCh38) {
        gene_id symbol
        gnomad_constraint {
          exp_lof obs_lof oe_lof oe_lof_lower oe_lof_upper
          pLI lof_z mis_z syn_z
          obs_mis exp_mis oe_mis
        }
      }
    }"""
    j = _post_graphql(GNOMAD_API, q, {"s": gene_symbol})
    if j and j.get("data") and j["data"].get("gene"):
        return j["data"]["gene"].get("gnomad_constraint")
    return None


@lru_cache(maxsize=20000)
def gnomad_variant(variant_id: str) -> Optional[Dict]:
    """
    gnomAD v4 variant info (GRCh38, dash-separated id like '1-2440958-A-G').
    """
    q = """query Var($vid: String!) {
      variant(variantId: $vid, dataset: gnomad_r4) {
        variant_id rsids
        exome { ac an af populations { id af } }
        genome { ac an af populations { id af } }
        in_silico_predictors { id value flags }
      }
    }"""
    j = _post_graphql(GNOMAD_API, q, {"vid": variant_id})
    if j and j.get("data"):
        return j["data"].get("variant")
    return None


# ─── Ensembl VEP / AlphaMissense ──────────────────────────────────────────
def ensembl_vep_rsid(rsid: str) -> Optional[List[Dict]]:
    """
    Run VEP for an rsID with AlphaMissense + CADD plugins.
    """
    url = f"{ENSEMBL_VEP}/{rsid}"
    params = {"AlphaMissense": "1", "CADD": "1"}
    return _get_rest(url, params=params)


# ─── FinnGen ──────────────────────────────────────────────────────────────
def finngen_phenotypes() -> Optional[List[Dict]]:
    return _get_rest(f"{FINNGEN_API}/phenos")


def finngen_region(phenotype: str, chrom, start, end) -> Optional[Dict]:
    chrom_str = str(chrom).replace("chr", "")
    url = f"{FINNGEN_API}/region/{phenotype}/{chrom_str}:{start}-{end}"
    return _get_rest(url)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- API Client Self-Test ---\n")

    print("[OpenTargets] rs6688934 full lookup:")
    r = ot_variant_full("rs6688934")
    if r:
        print(f"  variant_id: {r['variant_id']}")
        print(f"  n_credible_sets: {r['n_credible_sets']}")
        print(f"  max_h4: {r['max_h4']}")
        print(f"  immune_h4_max: {r['immune_h4_max']}")
        print(f"  h4>0.8 count: {r['h4_above_08_count']}")
        print(f"  top L2G: {r['top_l2g_gene']} (score={r['top_l2g_score']})")
    else:
        print("  (no data)")

    print("\n[gnomAD] FURIN constraint:")
    c = gnomad_gene_constraint("FURIN")
    if c:
        print(f"  pLI={c.get('pLI'):.3f}, LOEUF={c.get('oe_lof_upper'):.3f}, mis_z={c.get('mis_z'):.3f}")

    print("\n[eQTL Catalogue] rs6688934 summary:")
    e = eqtl_summary_for_rsid("rs6688934")
    print(f"  n_eqtl={e['n_eqtl']}, brain={e.get('n_brain_eqtl',0)}, "
          f"blood={e.get('n_blood_eqtl',0)}, immune={e.get('n_immune_eqtl',0)}")
    if e.get("min_p_brain"):
        print(f"  min_p_brain={e['min_p_brain']:.2e}")
