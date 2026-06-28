#!/usr/bin/env python3
"""
Phase 12d: Per-Variant iHS for Credible Set Variants (v2 — Voight-2006-correct)
================================================================================
Computes |iHS| (Voight 2006) for each PGC3 credible-set variant using
scikit-allel on EUR-subset 1000G Phase 3 phased haplotypes, with the four
code-review-driven corrections:

  [FIX-B1] Phase enforcement. Prior code took the diploid GT tuple regardless
           of `rec.samples[s].phased`. 1000G Phase 3 is supposed to be fully
           phased, but we now verify explicitly per record and drop any
           variant where >0.5% of EUR genotypes are unphased.
  [FIX-B2] Drop missing instead of coding -1. Prior code passed `[-1, -1]`
           for missing genotypes; allel.ihs treats this as a third allele,
           corrupting EHH computation. New code drops samples-by-variant
           with missingness > 5% (variant-level) and drops variants whose
           remaining missingness fraction is > 0%; the resulting H is a
           clean (n_var, n_haps) 0/1 array.
  [FIX-B3] Ancestral-allele polarization (Voight 2006 convention).
           scikit-allel's iHS sign refers to allele 0 vs allele 1 via
           score = log(iHH at allele 1 / iHH at allele 0). Prior code passed
           raw 0/1 codes, leaving sign biologically meaningless. New code
           reads the Ensembl GRCh37 ancestral-allele FASTA per chromosome,
           drops variants with low-confidence (lowercase or N) ancestral,
           and flips genotypes where ancestral == ALT so that 0 = ancestral,
           1 = derived. After polarization, POSITIVE iHS = longer haplotypes
           around the DERIVED allele (recent positive sweep on derived);
           NEGATIVE iHS = longer haplotypes around the ANCESTRAL allele.
           Downstream analyses use |iHS| (absolute value), so the sign
           convention does not affect numerical results, but the Voight 2006
           interpretation requires this polarization to be biologically
           meaningful.
  [FIX-B4] Genome-wide DAF-binned standardization (Voight 2006 protocol).
           Prior code z-scored within each ±500 kb credible-set window,
           inflating |iHS| in low-variance windows and removing cross-locus
           comparability. New code computes raw iHS for every passing
           biallelic SNP genome-wide, concatenates, and applies
           `allel.standardize_by_allele_count(ihs_raw, daf_count, n_bins=50)`
           — the genome-wide-MAF-binned standardization that scikit-allel
           ports from the Voight 2006 protocol. PGC3 credible-set variants
           are then looked up from this canonical table.

  Also: `include_edges=False` (allel default; was True before — kept the
  uncomputable boundary variants).

Output:
  - results/phase12/P12d_ihs_per_variant.tsv (one row per credible-set variant
    with non-NaN polarized iHS)
  - results/phase12/P12d_genomewide_ihs.tsv.gz (full per-chr iHS table, kept
    for downstream sensitivity / matched-control / neighbor-control analyses)
  - results/phase12/P12d_ANALYSIS_LOG.md
"""

import sys
import gzip
from pathlib import Path
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pysam
import allel
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"
OUT.mkdir(parents=True, exist_ok=True)
MAP_DIR = BASE / "data/raw/annotations/recombination_maps"
ANC_DIR = BASE / "data/raw/annotations/ancestral_alleles/homo_sapiens_ancestor_GRCh37_e71"
EUR_FILE = BASE / "data/raw/1kgp/eur_samples.txt"
VCF_DIR = BASE / "data/raw/1kgp/vcf"
VCF_TEMPLATE = "ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


# ─── Ancestral-allele FASTA loader ────────────────────────────────────────
def load_ancestral_chr(chrom: str) -> str:
    """Return the ancestral-allele sequence for a chromosome as a single
    uppercase-or-lowercase string. Lowercase letters indicate low-confidence
    ancestral; we treat only uppercase as polarizable. 'N' / '.' / '-' = unknown.

    Coordinates: 1-based. seq[pos-1] gives ancestral allele at 1-based pos.
    """
    fa = ANC_DIR / f"homo_sapiens_ancestor_{chrom}.fa"
    if not fa.exists():
        return None
    seq_chars = []
    with open(fa, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq_chars.append(line.strip())
    return "".join(seq_chars)


def ancestral_at(seq: str, pos_1based: int) -> str:
    """Return ancestral allele at 1-based position. '' if unknown / out-of-range."""
    if seq is None:
        return ""
    idx = pos_1based - 1
    if idx < 0 or idx >= len(seq):
        return ""
    c = seq[idx]
    return c if c in {"A", "C", "G", "T"} else ""  # uppercase only = high-confidence


# ─── Genetic map loader / interpolator ─────────────────────────────────────
def load_map(chrom: str) -> pd.DataFrame:
    f = MAP_DIR / f"plink.chr{chrom}.GRCh37.map"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t", header=None, names=["chrom", "rsid", "cM", "bp"])
    return df.sort_values("bp").reset_index(drop=True)


def cm_for_positions(map_df: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, map_df["bp"].values, map_df["cM"].values)


# ─── Per-chromosome polarized haplotype matrix ─────────────────────────────
def extract_polarized_haps(chrom: str, eur_samples_set: set) -> tuple:
    """Stream a chromosome VCF, build polarized H matrix.

    Returns (H, positions, rsids, refs, alts, ancestrals, dac, n_drops_dict)
    where:
      - H: int8 (n_var, n_haps) with 0=ancestral, 1=derived (after polarization)
      - dac: derived allele count per variant
      - n_drops_dict: counts of variants dropped by reason
    """
    fname = VCF_TEMPLATE.format(chr=chrom)
    vcf_path = VCF_DIR / fname
    if not vcf_path.exists():
        log(f"  chr{chrom}: VCF not found at {vcf_path}")
        return None

    vcf = pysam.VariantFile(str(vcf_path))
    sample_names = list(vcf.header.samples)
    sample_idx = [i for i, s in enumerate(sample_names) if s in eur_samples_set]
    if not sample_idx:
        log(f"  chr{chrom}: no EUR samples found in VCF")
        vcf.close()
        return None

    log(f"  chr{chrom}: VCF open, {len(sample_idx)} EUR samples")
    anc_seq = load_ancestral_chr(chrom)
    if anc_seq is None:
        log(f"  chr{chrom}: ancestral FASTA missing — cannot polarize")
        vcf.close()
        return None

    haps_rows = []
    pos_list = []
    rsid_list = []
    ref_list = []
    alt_list = []
    anc_list = []
    dac_list = []
    n_dropped = {
        "non_biallelic": 0, "indel": 0, "ancestral_unknown": 0, "ancestral_other": 0,
        "unphased": 0, "missing": 0, "monomorphic": 0,
    }

    for rec in vcf.fetch(chrom):
        # Biallelic SNPs only
        if not rec.alts or len(rec.alts) != 1:
            n_dropped["non_biallelic"] += 1
            continue
        ref = rec.ref
        alt = rec.alts[0]
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            n_dropped["indel"] += 1
            continue

        # Ancestral allele (high-confidence uppercase only)
        anc = ancestral_at(anc_seq, rec.pos)
        if anc == "":
            n_dropped["ancestral_unknown"] += 1
            continue
        if anc not in {ref, alt}:
            # Ancestral allele is neither REF nor ALT — drop (could be a tri-allelic
            # site where 1000G reports two alleles but ancestral is the third)
            n_dropped["ancestral_other"] += 1
            continue

        # Build haplotype row (2 alleles per sample) with phase enforcement
        row = np.empty(2 * len(sample_idx), dtype=np.int8)
        unphased_seen = False
        any_missing = False
        for k, idx in enumerate(sample_idx):
            sname = sample_names[idx]
            samp = rec.samples[sname]
            gt = samp.get("GT")
            if gt is None or len(gt) != 2:
                any_missing = True
                row[2*k] = -1
                row[2*k+1] = -1
                continue
            if any(g is None for g in gt):
                any_missing = True
                row[2*k] = -1
                row[2*k+1] = -1
                continue
            if not samp.phased:
                unphased_seen = True
                break
            row[2*k] = gt[0]
            row[2*k+1] = gt[1]
        if unphased_seen:
            n_dropped["unphased"] += 1
            continue
        if any_missing:
            # Drop if missingness > 0% for the variant (we already have 503 EUR
            # and 1000G Phase 3 is dense; tolerance for a single missing per
            # variant is too lenient for valid iHS. Strict drop.)
            n_dropped["missing"] += 1
            continue

        # Polarize: if ancestral == alt, swap 0/1 so that 0=ancestral, 1=derived
        if anc == alt:
            row = 1 - row
            # After flip, the original ALT (which was 1 before) is now 0,
            # representing the ancestral allele. Conceptual REF/ALT are
            # logged for traceability but H is now polarized.

        dac = int(row.sum())
        n_haps = len(row)
        # Skip monomorphic (DAF = 0 or 1)
        if dac == 0 or dac == n_haps:
            n_dropped["monomorphic"] += 1
            continue

        haps_rows.append(row)
        pos_list.append(rec.pos)
        rsid_list.append(rec.id if rec.id else "")
        ref_list.append(ref)
        alt_list.append(alt)
        anc_list.append(anc)
        dac_list.append(dac)

    vcf.close()
    if not haps_rows:
        return None
    H = np.array(haps_rows, dtype=np.int8)
    return (
        H,
        np.array(pos_list, dtype=np.int64),
        np.array(rsid_list, dtype=object),
        np.array(ref_list, dtype=object),
        np.array(alt_list, dtype=object),
        np.array(anc_list, dtype=object),
        np.array(dac_list, dtype=np.int32),
        n_dropped,
    )


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    log(f"Phase 12d v2: Voight-2006 |iHS| — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 72)
    log("Voight-2006 protocol: phase enforcement + ancestral polarization +")
    log("genome-wide DAF-binned standardization. Output is canonical |iHS|.")
    log("")

    # PGC3 credible-set variants
    master = pd.read_csv(BASE / "data/processed/pgc3_master_variants.tsv", sep="\t")
    log(f"PGC3 master variants: {len(master)}")
    pgc3_lookup = {}
    for _, r in master.iterrows():
        try:
            chrom = str(int(r["chr"]))  # autosomal int
        except (ValueError, TypeError):
            continue
        pgc3_lookup.setdefault(chrom, set()).add(int(r["pos"]))
    log(f"PGC3 autosomal positions registered: "
        f"{sum(len(s) for s in pgc3_lookup.values())} across "
        f"{len(pgc3_lookup)} chromosomes")

    # EUR samples
    eur_samples = set(EUR_FILE.read_text().strip().split("\n"))
    log(f"EUR samples requested: {len(eur_samples)}")

    # Process each chromosome, compute raw iHS, accumulate genome-wide
    all_chroms = []
    for chrom in [str(c) for c in range(1, 23)]:
        log(f"\n[chr{chrom}] extract & polarize haplotypes")
        result = extract_polarized_haps(chrom, eur_samples)
        if result is None:
            log(f"  chr{chrom}: skipped (extract returned None)")
            continue
        H, positions, rsids, refs, alts, ancs, dacs, drops = result
        log(f"  chr{chrom}: {H.shape[0]} polarized biallelic SNPs, "
            f"{H.shape[1]} haplotypes")
        log(f"  chr{chrom}: drops = {drops}")

        # Genetic-map cM positions
        gmap = load_map(chrom)
        if gmap is None:
            log(f"  chr{chrom}: genetic map missing")
            continue
        map_pos_cm = cm_for_positions(gmap, positions)

        # iHS (raw, signed Voight-iHS = ln(iHHa / iHHd) since 0=ancestral after polarization)
        log(f"  chr{chrom}: computing iHS on {H.shape[0]} variants ...")
        # allel.ihs default: include_edges=False; min_ehh=0.05; min_maf=0.05
        ihs_raw = allel.ihs(H, positions, map_pos=map_pos_cm,
                             min_ehh=0.05, min_maf=0.05, include_edges=False,
                             use_threads=True)
        n_valid = (~np.isnan(ihs_raw)).sum()
        log(f"  chr{chrom}: iHS computed; non-NaN: {n_valid}/{len(ihs_raw)}")

        chr_df = pd.DataFrame({
            "chr": chrom,
            "pos": positions,
            "rsid": rsids,
            "ref": refs,
            "alt": alts,
            "ancestral": ancs,
            "dac": dacs,
            "ihs_raw": ihs_raw,
        })
        all_chroms.append(chr_df)

    if not all_chroms:
        log("\n!! No chromosomes processed; aborting.")
        return

    # ── Genome-wide DAF-binned standardization (Voight 2006) ──
    log("\n[Genome-wide DAF-binned standardization (n_bins=50)]")
    full = pd.concat(all_chroms, ignore_index=True)
    log(f"  Total polarized biallelic SNPs: {len(full):,}")
    log(f"  Non-NaN raw iHS: {full['ihs_raw'].notna().sum():,}")
    valid_mask = full["ihs_raw"].notna()
    valid = full[valid_mask].copy()
    ihs_std_v, bins = allel.standardize_by_allele_count(
        valid["ihs_raw"].values.astype(np.float64),
        valid["dac"].values.astype(np.int64),
        n_bins=50, diagnostics=False
    )
    full.loc[valid_mask, "ihs_std"] = ihs_std_v
    log(f"  Standardized: {full['ihs_std'].notna().sum():,} variants")
    log(f"  ihs_std mean={full['ihs_std'].mean():.3f}, "
        f"sd={full['ihs_std'].std():.3f}, "
        f"|iHS|>2: {(full['ihs_std'].abs()>2).sum():,}")

    # Save genome-wide table (compressed)
    gw_path = OUT / "P12d_genomewide_ihs.tsv.gz"
    full.to_csv(gw_path, sep="\t", index=False, compression="gzip")
    log(f"\nSaved genome-wide table: {gw_path}")

    # ── Lookup PGC3 credible-set variants ──
    log("\n[PGC3 credible-set variant lookup]")
    full["chr"] = full["chr"].astype(str)
    full["pos"] = full["pos"].astype(int)
    master["chr"] = master["chr"].astype(str)
    master["pos"] = master["pos"].astype(int)
    merged = master[["credible_set_id", "rsid", "chr", "pos"]].merge(
        full[["chr", "pos", "ref", "alt", "ancestral", "dac", "ihs_raw", "ihs_std"]],
        on=["chr", "pos"], how="left"
    )
    log(f"  PGC3 variants matched in 1000G EUR polarized iHS table: "
        f"{merged['ihs_raw'].notna().sum()}/{len(merged)}")

    out_path = OUT / "P12d_ihs_per_variant.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    log(f"Saved per-variant iHS: {out_path}")

    if merged["ihs_raw"].notna().any():
        v = merged[merged["ihs_raw"].notna()]
        log(f"\nPGC3 iHS summary (non-NaN n={len(v)}):")
        log(f"  raw : mean={v['ihs_raw'].mean():.3f}, "
            f"sd={v['ihs_raw'].std():.3f}, "
            f"|iHS|>2 (raw): {(v['ihs_raw'].abs()>2).sum()}")
        log(f"  std : mean={v['ihs_std'].mean():.3f}, "
            f"sd={v['ihs_std'].std():.3f}, "
            f"|iHS|>2 (std): {(v['ihs_std'].abs()>2).sum()}")

    # Save log
    log_path = OUT / "P12d_ANALYSIS_LOG.md"
    with open(log_path, "w") as f:
        f.write("# Phase 12d Per-Variant iHS Log (v2 — Voight-2006-correct)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("```\n")
        for line in LOG:
            f.write(line + "\n")
        f.write("```\n")
    log(f"\nLog: {log_path}")
    log("\nPhase 12d v2 complete.")


if __name__ == "__main__":
    main()
