#!/usr/bin/env python3
"""
Phase 12d v2 — single-chromosome iHS worker (callable per-chr in parallel).

Usage:  python3 P12d_ihs_per_chr.py <CHR>

Reads 1000G EUR phased VCF for the given chromosome, polarizes by Ensembl
GRCh37 ancestral FASTA, computes raw signed Voight iHS via scikit-allel
(`include_edges=False`), and writes per-chr table:

  results/phase12/per_chr/P12d_ihs_chr{CHR}.tsv.gz

with columns: chr, pos, rsid, ref, alt, ancestral, dac, ihs_raw

Genome-wide DAF-binned standardization is performed by the aggregator
script `P12d_aggregate.py`. This script does NOT standardize.

All four code-review-driven corrections (B1–B4) are applied here. See the
docstring of P12d_ihs_credible_sets.py for the protocol details.
"""
import sys
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
OUT_DIR = BASE / "results/phase12/per_chr"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR = BASE / "data/raw/annotations/recombination_maps"
ANC_DIR = BASE / "data/raw/annotations/ancestral_alleles/homo_sapiens_ancestor_GRCh37_e71"
EUR_FILE = BASE / "data/raw/1kgp/eur_samples.txt"
VCF_DIR = BASE / "data/raw/1kgp/vcf"
VCF_TEMPLATE = "ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"


def load_ancestral_chr(chrom: str) -> str:
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
    if seq is None:
        return ""
    idx = pos_1based - 1
    if idx < 0 or idx >= len(seq):
        return ""
    c = seq[idx]
    return c if c in {"A", "C", "G", "T"} else ""  # uppercase only


def load_map(chrom: str) -> pd.DataFrame:
    f = MAP_DIR / f"plink.chr{chrom}.GRCh37.map"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep="\t", header=None, names=["chrom", "rsid", "cM", "bp"])
    return df.sort_values("bp").reset_index(drop=True)


def cm_for_positions(map_df: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, map_df["bp"].values, map_df["cM"].values)


def extract_polarized_haps(chrom, eur_samples_set):
    fname = VCF_TEMPLATE.format(chr=chrom)
    vcf_path = VCF_DIR / fname
    if not vcf_path.exists():
        print(f"chr{chrom}: VCF not found at {vcf_path}", flush=True)
        return None

    vcf = pysam.VariantFile(str(vcf_path))
    sample_names = list(vcf.header.samples)
    sample_idx = [i for i, s in enumerate(sample_names) if s in eur_samples_set]
    if not sample_idx:
        vcf.close()
        return None

    print(f"chr{chrom}: VCF open, {len(sample_idx)} EUR samples", flush=True)
    anc_seq = load_ancestral_chr(chrom)
    if anc_seq is None:
        print(f"chr{chrom}: ancestral FASTA missing", flush=True)
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
    n_seen = 0
    for rec in vcf.fetch(chrom):
        n_seen += 1
        if not rec.alts or len(rec.alts) != 1:
            n_dropped["non_biallelic"] += 1
            continue
        ref = rec.ref
        alt = rec.alts[0]
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            n_dropped["indel"] += 1
            continue
        anc = ancestral_at(anc_seq, rec.pos)
        if anc == "":
            n_dropped["ancestral_unknown"] += 1
            continue
        if anc not in {ref, alt}:
            n_dropped["ancestral_other"] += 1
            continue

        row = np.empty(2 * len(sample_idx), dtype=np.int8)
        unphased_seen = False
        any_missing = False
        for k, idx in enumerate(sample_idx):
            sname = sample_names[idx]
            samp = rec.samples[sname]
            gt = samp.get("GT")
            if gt is None or len(gt) != 2:
                any_missing = True
                break
            if any(g is None for g in gt):
                any_missing = True
                break
            if not samp.phased:
                unphased_seen = True
                break
            row[2*k] = gt[0]
            row[2*k+1] = gt[1]
        if unphased_seen:
            n_dropped["unphased"] += 1
            continue
        if any_missing:
            n_dropped["missing"] += 1
            continue

        if anc == alt:
            row = 1 - row  # polarize so 0=ancestral, 1=derived

        dac = int(row.sum())
        n_haps = len(row)
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
    print(f"chr{chrom}: streamed {n_seen} records; kept {len(haps_rows)} polarized SNPs", flush=True)
    print(f"chr{chrom}: drops = {n_dropped}", flush=True)
    if not haps_rows:
        return None
    return (
        np.array(haps_rows, dtype=np.int8),
        np.array(pos_list, dtype=np.int64),
        np.array(rsid_list, dtype=object),
        np.array(ref_list, dtype=object),
        np.array(alt_list, dtype=object),
        np.array(anc_list, dtype=object),
        np.array(dac_list, dtype=np.int32),
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: P12d_ihs_per_chr.py <CHR>", file=sys.stderr)
        sys.exit(2)
    chrom = sys.argv[1]
    out_path = OUT_DIR / f"P12d_ihs_chr{chrom}.tsv.gz"

    print(f"[chr{chrom}] start at {datetime.now().strftime('%H:%M:%S')}", flush=True)
    eur = set(EUR_FILE.read_text().strip().split("\n"))

    res = extract_polarized_haps(chrom, eur)
    if res is None:
        print(f"chr{chrom}: no data — exiting", flush=True)
        sys.exit(1)
    H, positions, rsids, refs, alts, ancs, dacs = res

    gmap = load_map(chrom)
    if gmap is None:
        print(f"chr{chrom}: genetic map missing — exiting", flush=True)
        sys.exit(1)
    map_pos_cm = cm_for_positions(gmap, positions)

    print(f"[chr{chrom}] computing iHS on {H.shape[0]:,} variants ...", flush=True)
    ihs_raw = allel.ihs(
        H, positions, map_pos=map_pos_cm,
        min_ehh=0.05, min_maf=0.05, include_edges=False,
        use_threads=True
    )
    n_valid = int((~np.isnan(ihs_raw)).sum())
    print(f"[chr{chrom}] iHS computed; non-NaN: {n_valid:,}/{len(ihs_raw):,}",
          flush=True)

    # Use efficient construction (avoid pandas DataFrame from object-array dict
    # which is slow on 1M+ rows). Concatenate via columns.
    df = pd.DataFrame({
        "chr": np.full(len(positions), chrom, dtype=object),
        "pos": positions,
        "rsid": rsids,
        "ref": refs,
        "alt": alts,
        "ancestral": ancs,
        "dac": dacs,
        "ihs_raw": ihs_raw,
    })
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"[chr{chrom}] saved {out_path} ({len(df):,} rows) at "
          f"{datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
