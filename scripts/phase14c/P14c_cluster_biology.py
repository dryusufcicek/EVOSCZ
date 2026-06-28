#!/usr/bin/env python3
"""
Phase 14c: Per-Cluster Biological Coherence
=============================================
For the 3 evolutionary clusters identified in Phase 14b v3, characterize:
  1. Pathway enrichment (GO BP, REACTOME, KEGG, MSigDB Hallmark) via Enrichr
  2. Brain region distribution (GTEx tissue × cluster cross-tab)
  3. Functional gene class (constraint, VEP impact, gene biotype)
  4. STRING protein-protein interaction network density per cluster

Goal: Are the 3 clusters biologically coherent (each a distinct mechanism)
or statistical artifact?

Output:
  results/phase14c/P14c_cluster_pathway_enrichment.tsv
  results/phase14c/P14c_brain_region_distribution.tsv
  results/phase14c/P14c_cluster_string_network.tsv
  results/phase14c/P14c_NARRATIVE.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase14c"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14c: Per-Cluster Biology — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# Load cluster assignments
ca = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")
log(f"\nCluster assignments: {len(ca)} variants, {ca['cluster'].notna().sum()} with cluster")
log(ca["cluster"].value_counts().sort_index().to_string())


# Load full master for additional features
m = pd.read_parquet(BASE / "results/phase11/variant_master_v3.parquet")
ca = ca.merge(m[["rsid", "vep_impact", "gene_biotype", "gtex_brain_tissue",
                  "loeuf", "pLI", "smr_gene_supported"]], on="rsid", how="left")


# Get per-cluster gene lists (unique genes per cluster, weighted by n_variants)
log("\n" + "=" * 72)
log("[1] Building per-cluster gene lists for enrichment")
log("=" * 72)

cluster_genes = {}
for c in [0, 1, 2]:
    sub = ca[ca["cluster"] == c].copy()
    # Unique genes, ranked by number of variants (top genes most likely "hub" genes)
    gene_counts = sub["gene_symbol"].value_counts()
    # Filter out null/no-gene
    valid_genes = gene_counts[~gene_counts.index.isin(["-", "NA", ""])]
    valid_genes = valid_genes[valid_genes.index.notna()]
    cluster_genes[c] = valid_genes.head(200).index.tolist()  # Top 200 genes for enrichment
    log(f"  Cluster {c}: {len(valid_genes)} unique genes; top 200 used for enrichment")
    log(f"    Top 10: {valid_genes.head(10).index.tolist()}")


# ─── 2. PATHWAY ENRICHMENT (Enrichr via gseapy) ────────────────────────────
log("\n" + "=" * 72)
log("[2] Pathway enrichment (Enrichr API: GO_BP, KEGG, Reactome, MSigDB)")
log("=" * 72)

import gseapy as gp

GENESET_LIBS = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

all_enrich = []
for c in [0, 1, 2]:
    log(f"\n  Cluster {c} ({len(cluster_genes[c])} genes):")
    for lib in GENESET_LIBS:
        try:
            enr = gp.enrichr(
                gene_list=cluster_genes[c],
                gene_sets=lib,
                organism="human",
                outdir=None,
                cutoff=0.05,
            )
            if enr.results is None or len(enr.results) == 0:
                log(f"    {lib}: no enriched terms")
                continue
            top = enr.results.head(5).copy()
            top["cluster"] = c
            top["library"] = lib
            all_enrich.append(top[["cluster", "library", "Term", "P-value",
                                     "Adjusted P-value", "Overlap", "Genes"]])
            log(f"    {lib} top 3:")
            for _, r in top.head(3).iterrows():
                term_short = r["Term"][:60]
                log(f"      {term_short}: q={r['Adjusted P-value']:.2e} ({r['Overlap']})")
        except Exception as e:
            log(f"    {lib}: error {e}")

if all_enrich:
    df_enrich = pd.concat(all_enrich, ignore_index=True)
    df_enrich.to_csv(OUT / "P14c_cluster_pathway_enrichment.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P14c_cluster_pathway_enrichment.tsv'} ({len(df_enrich)} rows)")


# ─── 3. BRAIN REGION DISTRIBUTION ─────────────────────────────────────────
log("\n" + "=" * 72)
log("[3] Brain region (GTEx tissue with min p) × cluster cross-tab")
log("=" * 72)

ct = pd.crosstab(ca["gtex_brain_tissue"], ca["cluster"])
ct_pct = ct.div(ct.sum(axis=0), axis=1) * 100  # percentage of cluster

log(f"\n  Brain region distribution (% within each cluster):")
log(ct_pct.round(1).to_string())

# Test which tissues differ between clusters
log(f"\n  Chi-square test brain tissue × cluster: ")
ct_no_zero = ct.loc[(ct.sum(axis=1) >= 5)]
chi2, p, dof, expected = stats.chi2_contingency(ct_no_zero)
log(f"    chi2={chi2:.1f}, dof={dof}, p={p:.3e}")

ct.to_csv(OUT / "P14c_brain_region_distribution.tsv", sep="\t")


# ─── 4. FUNCTIONAL CLASS PROFILE ──────────────────────────────────────────
log("\n" + "=" * 72)
log("[4] Functional class per cluster (VEP impact, gene biotype)")
log("=" * 72)

log("\n  VEP impact distribution (% within each cluster):")
vep_ct = pd.crosstab(ca["vep_impact"], ca["cluster"], normalize="columns") * 100
top_vep = vep_ct.sort_values(by=0, ascending=False).head(10)
log(top_vep.round(2).to_string())

log("\n  Gene biotype distribution (% within each cluster):")
bio_ct = pd.crosstab(ca["gene_biotype"], ca["cluster"], normalize="columns") * 100
log(bio_ct.round(2).to_string())

log("\n  Constraint metrics by cluster (median):")
constraint_summary = ca.groupby("cluster").agg(
    n=("rsid", "count"),
    med_loeuf=("loeuf", "median"),
    med_pLI=("pLI", "median"),
    pct_constrained=("pLI", lambda v: (v > 0.9).sum() / max(v.notna().sum(), 1) * 100),
    pct_smr_gene=("smr_gene_supported", lambda v: (v == 1).sum() / max(v.notna().sum(), 1) * 100),
)
log(constraint_summary.to_string())


# ─── 5. STRING NETWORK ANALYSIS ────────────────────────────────────────────
log("\n" + "=" * 72)
log("[5] STRING protein-protein interaction network per cluster")
log("=" * 72)

import requests

STRING_API = "https://string-db.org/api"
SPECIES = 9606  # human

def get_string_network_stats(genes, species=9606, score_cutoff=400):
    """Get PPI network stats from STRING for a gene list."""
    if len(genes) > 1000:
        genes = genes[:1000]  # STRING limit
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": score_cutoff,
        "caller_identity": "EVOSCZ_phase14c",
    }
    try:
        # Get network
        r = requests.get(f"{STRING_API}/tsv/network", params=params, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.read_csv(pd.io.common.StringIO(r.text), sep="\t")
        return {
            "n_genes_input": len(genes),
            "n_edges": len(df),
            "n_genes_with_edge": len(set(df["preferredName_A"].tolist() +
                                            df["preferredName_B"].tolist()))
                                 if len(df) > 0 else 0,
            "mean_score": df["score"].mean() if len(df) > 0 else 0,
        }
    except Exception as e:
        return None


string_results = []
for c in [0, 1, 2]:
    log(f"\n  Cluster {c} STRING query (top 200 genes)...")
    stats_c = get_string_network_stats(cluster_genes[c], score_cutoff=400)
    if stats_c:
        stats_c["cluster"] = c
        # Compute density
        n = stats_c["n_genes_with_edge"]
        e = stats_c["n_edges"]
        max_e = n * (n - 1) / 2 if n > 1 else 1
        stats_c["density"] = e / max_e if max_e > 0 else 0
        string_results.append(stats_c)
        log(f"    n_genes_input={stats_c['n_genes_input']}, n_edges={stats_c['n_edges']}, "
            f"n_with_edge={stats_c['n_genes_with_edge']}, density={stats_c['density']:.4f}")

if string_results:
    df_string = pd.DataFrame(string_results)
    df_string.to_csv(OUT / "P14c_cluster_string_network.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P14c_cluster_string_network.tsv'}")

    # Test if any cluster has higher network density (more biologically coherent?)
    log(f"\n  Cluster network density comparison:")
    for r in string_results:
        log(f"    Cluster {r['cluster']}: density={r['density']:.4f} ({r['n_edges']} edges)")


# Save log
with open(OUT / "P14c_NARRATIVE.md", "w") as f:
    f.write("# Phase 14c: Per-Cluster Biology\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nNarrative: {OUT / 'P14c_NARRATIVE.md'}")
log("\nPhase 14c complete.")
