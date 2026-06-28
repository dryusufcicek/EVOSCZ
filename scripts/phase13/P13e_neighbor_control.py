#!/usr/bin/env python3
"""
Phase 13e: Within-Locus Neighbor Variants as Negative Control
==============================================================
Methodological note: "Is the within-locus age × eQTL relationship specific to fine-mapped
SCZ risk variants, or does it apply to ANY common variant in the same LD block?"

Solution: For each PGC3 credible set, extract NON-credible-set variants in the
same 500kb window. These share LD, demography, and genomic context but are NOT
fine-mapped GWAS risk variants. Compare within-locus rho.

If PGC3 effect ≈ neighbor effect → not SCZ-specific (just LD-block property)
If PGC3 effect > neighbor effect → SCZ-specific signal beyond LD context

Reuses existing 1000G VCFs + GTEx v10 + GEVA atlas.

Output:
  - results/phase13/P13e_neighbor_control.tsv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase13"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 13e: Within-Locus Neighbor Control — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# ─── Load PGC3 master ──────────────────────────────────────────────────────
m = pd.read_parquet(BASE / "results/phase11/variant_master_v2.parquet")
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce")

# ─── Define windows around each credible set ──────────────────────────────
WINDOW = 500_000  # ±500kb
cs_regions = m.groupby("credible_set_id").agg({
    "chr_int": "first",
    "pos_int": ["min", "max"],
}).reset_index()
cs_regions.columns = ["credible_set_id", "chr", "pos_min", "pos_max"]
cs_regions = cs_regions.dropna(subset=["chr"])
cs_regions["chr"] = cs_regions["chr"].astype(int).astype(str)
cs_regions["window_start"] = (cs_regions["pos_min"] - WINDOW).clip(lower=1).astype(int)
cs_regions["window_end"] = (cs_regions["pos_max"] + WINDOW).astype(int)

# Restrict to autosomal
cs_regions = cs_regions[cs_regions["chr"].astype(str).str.match(r"^\d+$")]
log(f"Credible sets to process: {len(cs_regions)}")

# FIX-D-13e-1: prior code excluded PGC3 variants by `rec.id in pgc3_rsids`,
# but 1000G VCF rec.id is often '.' (no rsID assigned) — so PGC3 variants
# slipped into the neighbor set, biasing the comparison toward null. Use
# (chr, pos) match instead, which is universally available.
pgc3_chrpos = set(
    (str(int(c)), int(p))
    for c, p in zip(m["chr_int"], m["pos_int"])
    if pd.notna(c) and pd.notna(p)
)
log(f"PGC3 credible-set (chr,pos) keys: {len(pgc3_chrpos)}")


# ─── Use 1000G VCFs to enumerate neighbor SNPs ──────────────────────────────
sys.path.insert(0, str(BASE / "scripts/phase11"))
from lib_tabix_remote import LOCAL_VCF_DIR, VCF_TEMPLATE, _is_complete
import pysam

log("\n[1] Extract neighbor variants from 1000G EUR")

eur = (BASE / "data/raw/1kgp/eur_samples.txt").read_text().strip().split("\n")

neighbor_rsids = []  # List of (cs_id, chr, pos, rsid, maf)
n_chrs_done = 0

for chrom in sorted(cs_regions["chr"].unique(), key=lambda x: int(x)):
    fname = VCF_TEMPLATE.format(chr=chrom)
    if not _is_complete(LOCAL_VCF_DIR / fname):
        continue
    vcf = pysam.VariantFile(str(LOCAL_VCF_DIR / fname))
    sample_names = list(vcf.header.samples)
    sample_idx = [sample_names.index(s) for s in eur if s in sample_names]
    chrom_regions = cs_regions[cs_regions["chr"] == chrom]
    for _, region in chrom_regions.iterrows():
        cs_id = region["credible_set_id"]
        start, end = region["window_start"], region["window_end"]
        try:
            for rec in vcf.fetch(chrom, start, end):
                # FIX-D-13e-1: position-based PGC3 exclusion (rsID may be '.')
                if (str(chrom), int(rec.pos)) in pgc3_chrpos:
                    continue
                if len(rec.alts or []) != 1:
                    continue
                if len(rec.ref) > 1 or len(rec.alts[0]) > 1:
                    continue
                # Compute EUR MAF
                ac, an = 0, 0
                for idx in sample_idx:
                    sn = sample_names[idx]
                    gt = rec.samples[sn]["GT"]
                    if gt is None or any(g is None for g in gt): continue
                    ac += sum(gt); an += len(gt)
                if an == 0: continue
                af = ac / an
                maf = min(af, 1 - af)
                if maf < 0.01: continue  # match common variant range
                neighbor_rsids.append({
                    "credible_set_id": cs_id,
                    "rsid": rec.id if rec.id else f"chr{chrom}_{rec.pos}_{rec.ref}_{rec.alts[0]}",
                    "chr": chrom,
                    "pos": rec.pos,
                    "ref": rec.ref,
                    "alt": rec.alts[0],
                    "maf": maf,
                })
        except Exception:
            pass
    vcf.close()
    n_chrs_done += 1
    log(f"  chr{chrom} done — total neighbors so far: {len(neighbor_rsids)} ({n_chrs_done}/22)")

neighbors = pd.DataFrame(neighbor_rsids)
log(f"\nTotal neighbor variants: {len(neighbors)}")
log(f"Unique rsids: {neighbors['rsid'].nunique()}")

# Sample 1 random neighbor per CS to match PGC3 sample sizes
np.random.seed(42)
neighbor_sample = neighbors.groupby("credible_set_id").apply(
    lambda g: g.sample(n=min(len(g), 50), random_state=42)
).reset_index(drop=True)
log(f"Sampled {len(neighbor_sample)} neighbor variants ({neighbor_sample.groupby('credible_set_id').size().mean():.1f} per CS)")

neighbor_sample.to_csv(OUT / "P13e_neighbor_variants.tsv.gz", sep="\t", index=False, compression="gzip")
log(f"Saved: {OUT / 'P13e_neighbor_variants.tsv.gz'}")

# Save log
with open(OUT / "P13e_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 13e: Neighbor Control Variants\n\n```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log("\nPhase 13e (extraction) complete. Run P13e_test.py next to annotate + test.")
