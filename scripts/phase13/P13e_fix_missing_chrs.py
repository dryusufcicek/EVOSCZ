#!/usr/bin/env python3
"""
Phase 13e fix: Sequentially extract chr1-6 (chr11 has no index, skip).
Append to existing neighbor TSV.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase13"

sys.path.insert(0, str(BASE / "scripts/phase11"))
from lib_tabix_remote import LOCAL_VCF_DIR, VCF_TEMPLATE, _is_complete
import pysam

WINDOW = 500_000

print(f"Phase 13e fix — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

m = pd.read_parquet(BASE / "results/phase11/variant_master_v2.parquet")
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce")

cs_regions = m.groupby("credible_set_id").agg({
    "chr_int": "first", "pos_int": ["min", "max"]
}).reset_index()
cs_regions.columns = ["credible_set_id", "chr", "pos_min", "pos_max"]
cs_regions = cs_regions.dropna(subset=["chr"])
cs_regions["chr"] = cs_regions["chr"].astype(int).astype(str)
cs_regions["window_start"] = (cs_regions["pos_min"] - WINDOW).clip(lower=1).astype(int)
cs_regions["window_end"] = (cs_regions["pos_max"] + WINDOW).astype(int)

pgc3_pos_by_chr = {}
for chrom_int, grp in m.groupby("chr_int"):
    if pd.isna(chrom_int): continue
    pgc3_pos_by_chr[str(int(chrom_int))] = set(grp["pos_int"].dropna().astype(int).tolist())

eur = (BASE / "data/raw/1kgp/eur_samples.txt").read_text().strip().split("\n")

# Only chr1-6 (skip chr11 - no index)
target_chrs = ["1", "2", "3", "4", "5", "6"]
all_neighbors = []

for chrom in target_chrs:
    fname = VCF_TEMPLATE.format(chr=chrom)
    if not _is_complete(LOCAL_VCF_DIR / fname):
        print(f"  chr{chrom}: VCF missing"); continue
    pgc3_positions = pgc3_pos_by_chr.get(chrom, set())
    print(f"  chr{chrom}: PGC3 pos={len(pgc3_positions)}")
    vcf = pysam.VariantFile(str(LOCAL_VCF_DIR / fname))
    sample_names = list(vcf.header.samples)
    sample_idx = [sample_names.index(s) for s in eur if s in sample_names]
    chrom_regions = cs_regions[cs_regions["chr"] == chrom].to_dict("records")
    chr_count = 0
    for region in chrom_regions:
        cs_id = region["credible_set_id"]
        start = max(0, int(region["window_start"]))
        end = int(region["window_end"])
        for rec in vcf.fetch(chrom, start, end):
            if rec.pos in pgc3_positions: continue
            if len(rec.alts or []) != 1: continue
            if len(rec.ref) > 1 or len(rec.alts[0]) > 1: continue
            ac, an = 0, 0
            for idx in sample_idx:
                sn = sample_names[idx]
                gt = rec.samples[sn]["GT"]
                if gt is None or any(g is None for g in gt): continue
                ac += sum(gt); an += len(gt)
            if an == 0: continue
            af = ac / an
            maf = min(af, 1 - af)
            if maf < 0.01: continue
            all_neighbors.append({
                "credible_set_id": cs_id,
                "rsid": f"chr{chrom}_{rec.pos}",
                "chr": chrom,
                "pos": rec.pos,
                "maf": maf,
            })
            chr_count += 1
    vcf.close()
    print(f"  chr{chrom}: {chr_count} neighbors", flush=True)

new_neighbors = pd.DataFrame(all_neighbors)
print(f"\nTotal new neighbors: {len(new_neighbors)}")

# Sample 50/CS for each new chr
np.random.seed(42)
sampled_new = new_neighbors.groupby("credible_set_id").apply(
    lambda g: g.sample(n=min(len(g), 50), random_state=42)
).reset_index(drop=True)
print(f"Sampled (50/CS): {len(sampled_new)}")

# Combine with existing
existing = pd.read_csv(OUT / "P13e_neighbor_variants.tsv.gz", sep="\t")
print(f"Existing neighbors: {len(existing)}")

combined = pd.concat([existing, sampled_new], ignore_index=True)
print(f"Combined: {len(combined)} ({combined['credible_set_id'].nunique()} unique CS)")

combined.to_csv(OUT / "P13e_neighbor_variants.tsv.gz", sep="\t", index=False, compression="gzip")
print(f"Saved combined: {OUT / 'P13e_neighbor_variants.tsv.gz'}")
print("Phase 13e fix complete.")
