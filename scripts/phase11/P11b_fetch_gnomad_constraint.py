#!/usr/bin/env python3
"""
Phase 11b: gnomAD Constraint Fetcher
=====================================
For all unique gene symbols in the variant master, fetch gnomAD v4 constraint
scores via the gnomAD GraphQL API. Caches results to disk.

Output: results/phase11/gnomad_constraint_genes.tsv
"""

import sys
import time
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from lib_api_clients import gnomad_gene_constraint

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase11"
OUT.mkdir(parents=True, exist_ok=True)

print(f"Phase 11b: gnomAD Constraint Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# Load variant master to get unique gene symbols
master_path = OUT / "variant_master.parquet"
if not master_path.exists():
    print(f"  variant_master.parquet not found — using gene_atlas as fallback")
    master = pd.read_csv(BASE / "results/phase9/P9b_gene_atlas_expanded_SMR.tsv", sep="\t")
    genes = master["gene_symbol"].dropna().unique()
else:
    master = pd.read_parquet(master_path)
    genes = master["gene_symbol"].dropna().unique()

print(f"  Unique genes to query: {len(genes)}")

# Fetch with retry + caching
out_path = OUT / "gnomad_constraint_genes.tsv"
cache = {}
if out_path.exists():
    cached = pd.read_csv(out_path, sep="\t")
    cache = {r["gene_symbol"]: r.to_dict() for _, r in cached.iterrows()}
    print(f"  Resuming from {len(cache)} cached entries")

results = []
n_new = 0
for i, gene in enumerate(genes, 1):
    if gene in cache:
        results.append(cache[gene])
        continue
    c = gnomad_gene_constraint(gene)
    if c is None:
        c = {"gene_symbol": gene, "pLI": None, "oe_lof_upper": None,
             "mis_z": None, "lof_z": None, "syn_z": None}
    else:
        c["gene_symbol"] = gene
    results.append(c)
    n_new += 1
    if i % 25 == 0:
        print(f"  [{i}/{len(genes)}] queried (new: {n_new})")
        # Save partial
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(out_path, sep="\t", index=False)
    time.sleep(0.7)  # gnomAD rate limit ~1.5 req/s

df = pd.DataFrame(results)
df.to_csv(out_path, sep="\t", index=False)
print(f"\nSaved: {out_path}")
print(f"Genes with constraint: {df['pLI'].notna().sum()}/{len(df)}")
