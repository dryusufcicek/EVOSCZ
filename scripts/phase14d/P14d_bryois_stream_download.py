#!/usr/bin/env python3
"""
Phase 14d: Bryois 2022 Single-Cell Brain eQTL — Stream Download + Filter
=========================================================================
Downloads each Bryois chr file, filters to PGC3 rsIDs, accumulates filtered
rows, deletes raw chunk. Disk-friendly streaming approach.

Schema:
  snp_pos.txt.gz: SNP rsID → chr:pos_hg38 / chr:pos_hg19 / alleles / MAF
  CellType.{chr}.gz: gene_ensembl SNP_rsID dist_to_tss p_value beta (no header)

8 cell types × 22 chr = 176 cell-type files + 22 pb (pseudobulk) + 1 snp_pos
Total: 4.73 GB compressed.

For each PGC3 rsID, get min p value per cell type (across genes).

Output:
  data/processed/bryois_2022/Bryois_PGC3_minp_per_celltype.tsv.gz
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import json
from datetime import datetime
import os

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
RAW_DIR = BASE / "data/raw/annotations/bryois_2022"
PROC_DIR = BASE / "data/processed/bryois_2022"
PROC_DIR.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14d: Bryois Streaming — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)


# Get PGC3 rsIDs to filter on
m = pd.read_parquet(BASE / "results/phase11/variant_master_v3.parquet")
pgc3_rsids = set(m["rsid"].dropna().tolist())
log(f"PGC3 rsIDs: {len(pgc3_rsids)}")


# Get Zenodo file list with download URLs
with open("/tmp/zenodo_bryois.json") as f:
    zen = json.load(f)
files = zen.get("files", [])

# Build URL map
file_urls = {f["key"]: f["links"]["self"] for f in files}
log(f"Zenodo files: {len(file_urls)}")

# Cell types
CELL_TYPES = ["Excitatory.neurons", "Inhibitory.neurons", "Astrocytes",
               "Oligodendrocytes", "OPCs...COPs", "Microglia",
               "Endothelial.cells", "Pericytes"]
log(f"Cell types: {len(CELL_TYPES)}")


def download_file(url, dest_path):
    """Download with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == 2: raise
    return False


# Process each (cell_type, chr) pair
all_filtered = []
n_processed = 0

for ct in CELL_TYPES:
    for chrom in range(1, 23):
        fname = f"{ct}.{chrom}.gz"
        if fname not in file_urls:
            continue
        url = file_urls[fname]
        local = RAW_DIR / fname
        try:
            # Download
            download_file(url, local)
            # Stream-filter
            kept = []
            with gzip.open(local, "rt") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    rsid = parts[1]
                    if rsid in pgc3_rsids:
                        kept.append({
                            "cell_type": ct,
                            "chr": chrom,
                            "gene": parts[0],
                            "rsid": rsid,
                            "dist_tss": int(parts[2]) if parts[2].lstrip("-").isdigit() else None,
                            "p_value": float(parts[3]),
                            "beta": float(parts[4]),
                        })
            all_filtered.extend(kept)
            # Delete raw to save disk
            local.unlink()
            n_processed += 1
            if n_processed % 22 == 0 or len(kept) > 0:
                log(f"  [{ct} chr{chrom}] {len(kept)} PGC3 hits "
                    f"(total kept: {len(all_filtered)}, files done: {n_processed}/176)")
        except Exception as e:
            log(f"  ! {fname}: {e}")
            if local.exists(): local.unlink()

log(f"\nFinal: {n_processed} files processed, {len(all_filtered)} PGC3 hits")

# Save full hits
df_hits = pd.DataFrame(all_filtered)
df_hits.to_csv(PROC_DIR / "Bryois_PGC3_all_hits.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"Saved all hits: {PROC_DIR / 'Bryois_PGC3_all_hits.tsv.gz'}")

# Aggregate per (rsid, cell_type): min p, best gene
df_minp = df_hits.loc[df_hits.groupby(["rsid", "cell_type"])["p_value"].idxmin()].copy()
log(f"\nUnique (rsid × cell_type) min p rows: {len(df_minp)}")
log(f"PGC3 variants with any Bryois hit: {df_minp['rsid'].nunique()}")

# Pivot to wide: each PGC3 rsid x 8 cell types
pivot_p = df_minp.pivot_table(index="rsid", columns="cell_type", values="p_value", aggfunc="min")
pivot_p.columns = [f"bryois_{c.replace('.','_').replace('...','_')}_minp" for c in pivot_p.columns]
pivot_p = pivot_p.reset_index()
log(f"Wide table: {pivot_p.shape}")
pivot_p.to_csv(PROC_DIR / "Bryois_PGC3_minp_per_celltype.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"Saved wide: {PROC_DIR / 'Bryois_PGC3_minp_per_celltype.tsv.gz'}")

with open(PROC_DIR / "P14d_DOWNLOAD_LOG.md", "w") as f:
    f.write("# Phase 14d Bryois Download Log\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 14d streaming complete.")
