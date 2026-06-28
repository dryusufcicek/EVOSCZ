"""
Tabix Remote Helper — query 1000G VCFs without full download
=============================================================
Wraps pysam.VariantFile around remote 1000G FTP URLs.
Fallbacks to local files when present (data/raw/1kgp/vcf/).
"""

import os
import pysam
from pathlib import Path
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])

LOCAL_VCF_DIR = Path(str(BASE / "data/raw/1kgp/vcf"))
REMOTE_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
VCF_TEMPLATE = "ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"


def _is_complete(path: Path) -> bool:
    """Quick gunzip integrity check."""
    if not path.exists():
        return False
    try:
        import subprocess
        r = subprocess.run(["gunzip", "-t", str(path)], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def get_vcf_handle(chrom):
    """
    Return pysam.VariantFile handle for chromosome `chrom`.
    Uses local file if complete, otherwise streams from 1000G FTP.
    """
    chrom_str = str(chrom).replace("chr", "")
    fname = VCF_TEMPLATE.format(chr=chrom_str)
    local = LOCAL_VCF_DIR / fname
    if _is_complete(local):
        return pysam.VariantFile(str(local)), "local"
    return pysam.VariantFile(f"{REMOTE_BASE}/{fname}"), "remote"


def fetch_region(chrom, start, end):
    """
    Generator yielding pysam.VariantRecord objects in [start, end] (1-based, inclusive end).
    Automatically uses local or remote source.
    """
    vcf, src = get_vcf_handle(chrom)
    chrom_str = str(chrom).replace("chr", "")
    for rec in vcf.fetch(chrom_str, start - 1, end):
        yield rec
    # Don't close — let caller manage if reusing


def fetch_variant(chrom, pos, rsid=None):
    """
    Fetch a single variant by position (and optional rsID confirm).
    Returns (record, source) or (None, source) if not found.
    """
    vcf, src = get_vcf_handle(chrom)
    chrom_str = str(chrom).replace("chr", "")
    for rec in vcf.fetch(chrom_str, pos - 1, pos):
        if rec.pos == pos:
            if rsid is None or rec.id == rsid:
                return rec, src
    return None, src


def extract_region_to_vcf(chrom, start, end, output_path, samples=None):
    """
    Extract a genomic region to a local VCF file (for selscan input).
    samples: optional list of sample IDs to subset (e.g. EUR samples).
    """
    vcf, src = get_vcf_handle(chrom)
    chrom_str = str(chrom).replace("chr", "")
    if samples is not None:
        vcf.subset_samples(samples)
    out = pysam.VariantFile(str(output_path), "wz", header=vcf.header)
    n = 0
    for rec in vcf.fetch(chrom_str, start - 1, end):
        out.write(rec)
        n += 1
    out.close()
    pysam.tabix_index(str(output_path), preset="vcf", force=True)
    return n, src


def load_eur_samples():
    """Load EUR sample IDs from data/raw/1kgp/eur_samples.txt."""
    p = Path(str(BASE / "data/raw/1kgp/eur_samples.txt"))
    return p.read_text().strip().split("\n")


if __name__ == "__main__":
    # Self-test
    print("--- Tabix Remote Helper Self-Test ---")
    eur = load_eur_samples()
    print(f"EUR samples: {len(eur)}")
    rec, src = fetch_variant("1", 2372397, "rs6688934")
    if rec:
        print(f"  rs6688934 found at {rec.contig}:{rec.pos} (source={src})")
        print(f"  REF={rec.ref}, ALT={rec.alts}, n_samples={len(rec.samples)}")
    else:
        print(f"  rs6688934 NOT FOUND (source={src})")
    n = sum(1 for _ in fetch_region("22", 16050000, 16100000))
    print(f"  chr22:16050000-16100000 has {n} variants")
