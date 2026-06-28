#!/usr/bin/env python3
"""
Phase 12d v2 — aggregate per-chr iHS into genome-wide standardized table.

Reads results/phase12/per_chr/P12d_ihs_chr{1..22}.tsv.gz, concatenates,
applies `allel.standardize_by_allele_count(ihs_raw, dac, n_bins=50)` for the
canonical Voight 2006 genome-wide DAF-binned standardization, then looks up
PGC3 credible-set variants and writes the canonical per-variant table.

Outputs:
  - results/phase12/P12d_genomewide_ihs.tsv.gz (genome-wide table with
    raw + standardized iHS for all polarized biallelic 1000G EUR SNPs)
  - results/phase12/P12d_ihs_per_variant.tsv (PGC3 credible-set variants
    only, with raw + standardized iHS for downstream Phase 12+ analyses)
  - results/phase12/P12d_ANALYSIS_LOG.md
"""
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd
import allel

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"
PER_CHR_DIR = OUT / "per_chr"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


def main():
    log(f"Phase 12d v2 — aggregate at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    parts = []
    for c in range(1, 23):
        p = PER_CHR_DIR / f"P12d_ihs_chr{c}.tsv.gz"
        if not p.exists():
            log(f"  ! chr{c}: per-chr file missing — skipping")
            continue
        df = pd.read_csv(p, sep="\t")
        log(f"  chr{c}: {len(df):,} rows, non-NaN ihs: {df['ihs_raw'].notna().sum():,}")
        parts.append(df)
    if not parts:
        log("No per-chr files found; aborting.")
        return
    full = pd.concat(parts, ignore_index=True)
    log(f"\nGenome-wide concatenated: {len(full):,} polarized biallelic SNPs")
    log(f"  Non-NaN raw ihs: {full['ihs_raw'].notna().sum():,}")

    # Standardize by allele count, n_bins=50, only on finite entries.
    # Critical: drop inf as well as NaN — a few inf values poison bin mean/std
    # computation, propagating NaN to all standardized scores. (Discovered via
    # warnings emitted by scipy.binned_statistic when first attempted with
    # only NaN filtering.)
    log("\nApplying allel.standardize_by_allele_count (n_bins=50; finite-only)...")
    raw = full["ihs_raw"].values
    valid_mask = np.isfinite(raw)
    n_inf = int((~valid_mask & ~np.isnan(raw)).sum())
    log(f"  Dropped non-finite (NaN+inf): {(~valid_mask).sum():,} (of which inf: {n_inf})")
    valid = full[valid_mask].copy()
    ihs_std_v, _ = allel.standardize_by_allele_count(
        valid["ihs_raw"].values.astype(np.float64),
        valid["dac"].values.astype(np.int64),
        n_bins=50, diagnostics=False,
    )
    full["ihs_std"] = np.nan
    full.loc[valid_mask, "ihs_std"] = ihs_std_v
    log(f"  Standardized iHS: mean={full['ihs_std'].mean():.4f}, "
        f"sd={full['ihs_std'].std():.4f}")
    log(f"  |iHS_std|>2 (genome-wide): {(full['ihs_std'].abs() > 2).sum():,}")

    gw_path = OUT / "P12d_genomewide_ihs.tsv.gz"
    full.to_csv(gw_path, sep="\t", index=False, compression="gzip")
    log(f"\nSaved genome-wide table: {gw_path}")

    # ── Lookup PGC3 credible-set variants ──
    log("\n[PGC3 credible-set variant lookup]")
    master = pd.read_csv(BASE / "data/processed/pgc3_master_variants.tsv", sep="\t")
    full["chr"] = full["chr"].astype(str)
    full["pos"] = full["pos"].astype(int)
    master["chr"] = master["chr"].astype(str)
    master["pos"] = master["pos"].astype(int)
    merged = master[["credible_set_id", "rsid", "chr", "pos"]].merge(
        full[["chr", "pos", "ref", "alt", "ancestral", "dac", "ihs_raw", "ihs_std"]],
        on=["chr", "pos"], how="left"
    )
    log(f"  PGC3 variants matched in genome-wide iHS table: "
        f"{merged['ihs_raw'].notna().sum()}/{len(merged)}")
    out_path = OUT / "P12d_ihs_per_variant.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    log(f"Saved PGC3 per-variant iHS: {out_path}")

    if merged["ihs_raw"].notna().any():
        v = merged[merged["ihs_raw"].notna()]
        log(f"\nPGC3 iHS summary (n={len(v)}):")
        log(f"  raw : mean={v['ihs_raw'].mean():.3f}, sd={v['ihs_raw'].std():.3f}, "
            f"|raw|>2: {(v['ihs_raw'].abs()>2).sum()}")
        log(f"  std : mean={v['ihs_std'].mean():.3f}, sd={v['ihs_std'].std():.3f}, "
            f"|std|>2: {(v['ihs_std'].abs()>2).sum()}")

    log_path = OUT / "P12d_ANALYSIS_LOG.md"
    with open(log_path, "w") as f:
        f.write("# Phase 12d Per-Variant iHS Log (v2 — Voight-2006-correct, parallel)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("```\n")
        for line in LOG:
            f.write(line + "\n")
        f.write("```\n")
    log(f"\nLog saved: {log_path}")
    log("Phase 12d aggregate complete.")


if __name__ == "__main__":
    main()
