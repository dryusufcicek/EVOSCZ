#!/usr/bin/env python3
"""
Phase 13e (PARALLEL): Within-Locus Neighbor Variants
=====================================================
Multiprocessing version: extract neighbor variants per chromosome in parallel.
Uses 8 worker processes for 22 autosomal chromosomes.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase13"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts/phase11"))
from lib_tabix_remote import LOCAL_VCF_DIR, VCF_TEMPLATE, _is_complete

WINDOW = 500_000


def process_chrom(args):
    """1000G VCFs lack rsIDs (ID column = '.'). Use (chr, pos) tuples for
    matching PGC3 variants. Neighbor identifier = chr_pos string."""
    chrom, regions, pgc3_positions, eur = args
    import pysam
    fname = VCF_TEMPLATE.format(chr=chrom)
    if not _is_complete(LOCAL_VCF_DIR / fname):
        return chrom, []
    vcf = pysam.VariantFile(str(LOCAL_VCF_DIR / fname))
    sample_names = list(vcf.header.samples)
    sample_idx = [sample_names.index(s) for s in eur if s in sample_names]
    out = []
    for region in regions:
        cs_id = region["credible_set_id"]
        start = max(0, int(region["window_start"]))
        end = int(region["window_end"])
        try:
            for rec in vcf.fetch(chrom, start, end):
                if rec.pos in pgc3_positions:
                    continue  # exclude PGC3 fine-mapped (by pos match)
                if len(rec.alts or []) != 1:
                    continue
                if len(rec.ref) > 1 or len(rec.alts[0]) > 1:
                    continue
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
                out.append({
                    "credible_set_id": cs_id,
                    "rsid": f"chr{chrom}_{rec.pos}",  # synthetic rsID
                    "chr": chrom,
                    "pos": rec.pos,
                    "maf": maf,
                })
        except Exception:
            pass
    vcf.close()
    return chrom, out


if __name__ == "__main__":
    print(f"Phase 13e parallel — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

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
    cs_regions = cs_regions[cs_regions["chr"].astype(str).str.match(r"^\d+$")]
    print(f"Credible sets: {len(cs_regions)}")

    # 1000G VCFs lack rsIDs. Build per-chr pos sets to filter PGC3 variants.
    pgc3_pos_by_chr = {}
    for chrom, grp in m.groupby("chr_int"):
        if pd.isna(chrom): continue
        pgc3_pos_by_chr[str(int(chrom))] = set(grp["pos_int"].dropna().astype(int).tolist())
    eur = (BASE / "data/raw/1kgp/eur_samples.txt").read_text().strip().split("\n")
    print(f"PGC3 chrs: {len(pgc3_pos_by_chr)}, EUR samples: {len(eur)}")

    # Group regions per chromosome
    tasks = []
    for chrom in sorted(cs_regions["chr"].unique(), key=lambda x: int(x)):
        regions = cs_regions[cs_regions["chr"] == chrom].to_dict("records")
        pgc3_positions = pgc3_pos_by_chr.get(chrom, set())
        tasks.append((chrom, regions, pgc3_positions, eur))
    print(f"Chromosomes to process: {len(tasks)}")

    print(f"\nLaunching parallel pool (8 workers)...")
    all_neighbors = []
    with Pool(processes=8) as pool:
        for chrom, results in pool.imap_unordered(process_chrom, tasks):
            print(f"  chr{chrom}: {len(results)} neighbors", flush=True)
            all_neighbors.extend(results)

    neighbors = pd.DataFrame(all_neighbors)
    print(f"\nTotal: {len(neighbors)} neighbor variants, {neighbors['rsid'].nunique()} unique rsIDs")

    np.random.seed(42)
    sampled = neighbors.groupby("credible_set_id").apply(
        lambda g: g.sample(n=min(len(g), 50), random_state=42)
    ).reset_index(drop=True)
    print(f"Sampled: {len(sampled)} neighbor variants ({len(sampled)/sampled['credible_set_id'].nunique():.1f} per CS)")

    out_path = OUT / "P13e_neighbor_variants.tsv.gz"
    sampled.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"Saved: {out_path}")
    print("\nP13e parallel complete.")
