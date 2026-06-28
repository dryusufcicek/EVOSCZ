#!/usr/bin/env python3
"""
================================================================================
Phase 19r — Bigdeli 2026 AFR-discovery REPLICATION of Phase 19 polarity finding
================================================================================

GOAL
----
Test whether the 32 EUR-bottleneck-amplified SCZ credible-set lead variants
(Phase 19 primary stratum) replicate in Bigdeli 2026 (Nature) AFR-discovery
sumstats. Reports:

1. Direction concordance: do EUR risk alleles also have positive β in AFR?
2. β correlation: Pearson r(EUR_β, AFR_β) on overlapping variants
3. Effect size attenuation: AFR β magnitude vs EUR β (winner's curse test)
4. Gene-level cross-check: AFR-discovery top genes vs EUR-amplified gene list

INPUT (manual download required by user)
----------------------------------------
- data/bigdeli2026/media-2.xlsx OR similar (under $EVOSCZ_ROOT)
  Source: https://www.medrxiv.org/content/10.1101/2024.08.27.24312631v1.supplementary-material
  Specifically Supplement 2 with AFR meta-analysis SNP-level results.

SCRIPT auto-detects xlsx files in the bigdeli2026/ directory and tries each.

OUTPUT
------
- results/phase19_local/P19r_bigdeli_replication.tsv
- results/phase19_local/P19r_bigdeli_summary.md

USAGE
-----
  python3 scripts/phase19/P19r_bigdeli_replication.py

Author: 2026-05-16
================================================================================
"""

import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np

# Paths
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P19_OUTPUT = BASE / "results/phase19_local/P19_a_polarity_per_variant.tsv.gz"
MASTER = BASE / "data/processed/pgc3_master_variants.tsv"
BIGDELI_DIR = BASE / "data/bigdeli2026"
OUT_DIR = BASE / "results/phase19_local"
OUT_TSV = OUT_DIR / "P19r_bigdeli_replication.tsv"
OUT_MD = OUT_DIR / "P19r_bigdeli_summary.md"


def log(msg=""):
    print(f"[P19r] {msg}", flush=True)


def find_bigdeli_xlsx():
    """Auto-detect a real .xlsx file in BIGDELI_DIR (not HTML page)."""
    if not BIGDELI_DIR.exists():
        log(f"ERROR: directory {BIGDELI_DIR} does not exist.")
        log("Please create it and place Bigdeli supplementary .xlsx file(s) there.")
        sys.exit(1)

    candidates = sorted(BIGDELI_DIR.glob("*.xlsx")) + sorted(BIGDELI_DIR.glob("*.xls"))
    if not candidates:
        log(f"ERROR: no .xlsx files found in {BIGDELI_DIR}")
        log("Please manually download the Bigdeli 2024 medrxiv supplement (media-2.xlsx)")
        log("from https://www.medrxiv.org/content/10.1101/2024.08.27.24312631v1.supplementary-material")
        log(f"and place it in {BIGDELI_DIR}/")
        sys.exit(1)

    # Test which file is a real Excel (binary not HTML)
    valid = []
    for fp in candidates:
        with open(fp, "rb") as f:
            head = f.read(8)
        # XLSX zip header: PK\x03\x04 / XLS: \xD0\xCF
        if head.startswith(b"PK\x03\x04") or head.startswith(b"\xD0\xCF\x11\xE0"):
            valid.append(fp)
            log(f"Valid Excel: {fp.name} ({fp.stat().st_size:,} bytes)")
        else:
            log(f"Skipping (not real Excel): {fp.name}")

    if not valid:
        log("ERROR: all candidate .xlsx files appear to be HTML pages (failed downloads).")
        log("Please verify the supplementary download — must be a real .xlsx binary.")
        sys.exit(1)

    return valid


def read_all_sheets(xlsx_path):
    """Read all sheets, return dict of {sheet_name: DataFrame}."""
    log(f"Reading {xlsx_path.name}...")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    out = {}
    for name, df in sheets.items():
        log(f"  sheet '{name}': {df.shape}")
        out[name] = df
    return out


def find_afr_sumstats_sheet(all_sheets):
    """
    Heuristic: find the sheet that contains per-SNP results with columns like
    rsid/SNP, chr, pos, A1/A2, BETA/OR, P. Look for the AFR-discovery
    suggestive table (~10K-50K rows).
    """
    log("Searching for AFR sumstats sheet (heuristic)...")
    best = None
    best_score = -1

    for name, df in all_sheets.items():
        # Try first row as header
        if df.shape[0] < 10:
            continue
        # Score by row count + presence of typical columns in any of first 5 rows
        score = df.shape[0]
        # bonus if column names suggest GWAS sumstats
        for ridx in range(min(5, df.shape[0])):
            row_vals = " ".join(str(x).lower() for x in df.iloc[ridx].fillna(""))
            if "snp" in row_vals or "rsid" in row_vals:
                score += 500
            if "chr" in row_vals and ("pos" in row_vals or "bp" in row_vals):
                score += 500
            if "beta" in row_vals or "logor" in row_vals or "log_or" in row_vals:
                score += 500
            if "afr" in row_vals or "african" in row_vals:
                score += 1000
        if score > best_score:
            best_score = score
            best = (name, df)

    if best is None:
        log("ERROR: could not auto-detect AFR sumstats sheet")
        return None, None

    name, df = best
    log(f"Best candidate: '{name}' with shape {df.shape}, score {best_score}")
    return name, df


def parse_sheet(df):
    """Try to parse the sheet: find header row, then extract data."""
    # Try each of the first 5 rows as header
    for header_idx in range(5):
        try:
            cols = [str(x).strip() for x in df.iloc[header_idx].fillna(f"col{header_idx}_")]
            data = df.iloc[header_idx + 1:].copy()
            data.columns = cols
            # Check if we have a column that looks like rsid (rsX numeric)
            for c in cols:
                col_data = data[c].astype(str).head(50)
                rsid_match = col_data.str.match(r"^rs\d+$", na=False).sum()
                if rsid_match > 5:
                    log(f"Found rsid column '{c}' (header row {header_idx})")
                    return data, c
        except Exception as e:
            log(f"  header {header_idx} failed: {e}")
            continue
    return None, None


def lookup_eur_amplified_in_afr(eur_leads, afr_data, rsid_col):
    """
    Given 32 EUR-amplified lead variants and Bigdeli AFR sumstats,
    look up direction + effect-size concordance.
    """
    log(f"Looking up {len(eur_leads)} EUR-amplified leads in AFR sumstats ({len(afr_data):,} rows)")

    # Standardize rsid column
    afr_data["__rsid"] = afr_data[rsid_col].astype(str).str.strip().str.lower()
    eur_leads = eur_leads.copy()
    eur_leads["__rsid"] = eur_leads["rsid"].astype(str).str.strip().str.lower()

    # Find direction-related columns in AFR sumstats
    cols_lower = {c.lower(): c for c in afr_data.columns}
    beta_col = next((cols_lower[k] for k in cols_lower if "beta" in k.lower() and "se" not in k.lower()), None)
    or_col = next((cols_lower[k] for k in cols_lower if ("logor" in k.lower() or "or" == k.lower())), None)
    a1_col = next((cols_lower[k] for k in cols_lower if k.lower() in ("a1", "tested", "effect", "ea")), None)
    a2_col = next((cols_lower[k] for k in cols_lower if k.lower() in ("a2", "other", "noneffect", "nea")), None)
    p_col = next((cols_lower[k] for k in cols_lower if k.lower() in ("p", "pval", "p-value", "pvalue")), None)

    log(f"AFR columns detected: beta={beta_col}, OR={or_col}, A1={a1_col}, A2={a2_col}, P={p_col}")

    merged = eur_leads.merge(
        afr_data,
        left_on="__rsid",
        right_on="__rsid",
        how="left",
        suffixes=("_eur", "_afr"),
    )

    n_match = merged[rsid_col].notna().sum()
    log(f"Matched {n_match} of {len(eur_leads)} EUR-amplified leads in AFR sumstats")

    return merged, dict(beta=beta_col, OR=or_col, a1=a1_col, a2=a2_col, p=p_col)


def compute_concordance(merged, afr_cols):
    """For matched variants, compute direction concordance and β correlation."""
    matched = merged[merged.get(afr_cols["a1"], pd.Series([np.nan]*len(merged))).notna() if afr_cols["a1"] else merged[afr_cols["beta"] or "rsid"].notna()].copy()

    n_matched = len(matched)
    if n_matched == 0:
        log("WARNING: no matched variants for concordance analysis")
        return {}

    # Get AFR β (or convert from OR)
    if afr_cols["beta"]:
        matched["AFR_BETA"] = pd.to_numeric(matched[afr_cols["beta"]], errors="coerce")
    elif afr_cols["OR"]:
        afr_or = pd.to_numeric(matched[afr_cols["OR"]], errors="coerce")
        matched["AFR_BETA"] = np.log(afr_or)
    else:
        log("WARNING: no β or OR column found for AFR")
        return {}

    # Orient AFR β to EUR_A1 (PGC3 effect allele)
    if afr_cols["a1"]:
        matched["AFR_A1"] = matched[afr_cols["a1"]].astype(str).str.upper()
        same = matched["EUR_A1"].astype(str).str.upper() == matched["AFR_A1"]
        matched["AFR_BETA_oriented"] = np.where(
            same, matched["AFR_BETA"], -matched["AFR_BETA"]
        )
    else:
        matched["AFR_BETA_oriented"] = matched["AFR_BETA"]
        log("WARNING: no AFR A1 column — assuming alleles aligned")

    # Concordance: does sign(EUR_β) == sign(AFR_β_oriented)?
    valid = matched.dropna(subset=["EUR_BETA", "AFR_BETA_oriented"])
    if len(valid) == 0:
        return dict(n_lookup=n_matched, n_oriented=0, concordance=np.nan)

    concord = (np.sign(valid["EUR_BETA"]) == np.sign(valid["AFR_BETA_oriented"])).mean()

    # β correlation
    from scipy import stats as spstats
    try:
        r, p = spstats.pearsonr(valid["EUR_BETA"], valid["AFR_BETA_oriented"])
    except Exception:
        r, p = np.nan, np.nan

    return dict(
        n_lookup=n_matched,
        n_oriented=len(valid),
        concordance=concord,
        beta_pearson_r=r,
        beta_pearson_p=p,
        median_eur_beta_abs=valid["EUR_BETA"].abs().median(),
        median_afr_beta_abs=valid["AFR_BETA_oriented"].abs().median(),
        attenuation_ratio=valid["AFR_BETA_oriented"].abs().median() / valid["EUR_BETA"].abs().median() if valid["EUR_BETA"].abs().median() > 0 else np.nan,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 78)
    log("Phase 19r — Bigdeli 2026 AFR-discovery replication")
    log("=" * 78)

    # Load Phase 19 per-variant output + master
    log("Loading Phase 19 per-variant output...")
    p19 = pd.read_csv(P19_OUTPUT, sep="\t")
    master = pd.read_csv(MASTER, sep="\t")
    m = p19.merge(master[["rsid", "credible_set_id"]], on="rsid", how="left")
    pol = m[m["delta_RAF_EUR_minus_AFR"].notna()].copy()
    all_leads = pol.sort_values("PIP", ascending=False).groupby("credible_set_id").head(1)
    eur_leads_32 = all_leads[all_leads["AF_AFR_stratum"].isin(["lt_0.01", "0.01_0.05"])].copy()
    log(f"Phase 19 EUR-amplified leads: {len(eur_leads_32)}")
    log(f"  rsids: {', '.join(eur_leads_32['rsid'].astype(str).head(5).tolist())}...")

    # Find and read Bigdeli xlsx
    bigdeli_files = find_bigdeli_xlsx()
    log(f"Bigdeli candidate files: {[f.name for f in bigdeli_files]}")

    # Try each file
    all_sheets = {}
    for fp in bigdeli_files:
        try:
            sheets = read_all_sheets(fp)
            for name, df in sheets.items():
                all_sheets[f"{fp.stem}::{name}"] = df
        except Exception as e:
            log(f"  ERROR reading {fp.name}: {e}")

    if not all_sheets:
        log("ERROR: no sheets could be read")
        sys.exit(1)

    log(f"\nTotal sheets across all files: {len(all_sheets)}")

    # Find AFR sumstats sheet
    sheet_name, afr_sheet = find_afr_sumstats_sheet(all_sheets)
    if sheet_name is None:
        log("Could not auto-detect AFR sumstats sheet.")
        log("Available sheets:")
        for n, df in all_sheets.items():
            log(f"  {n}: {df.shape}")
        sys.exit(1)

    # Parse sheet
    parsed, rsid_col = parse_sheet(afr_sheet)
    if parsed is None:
        log("ERROR: could not parse AFR sheet (no rsid column found)")
        log("Sheet head:")
        log(afr_sheet.head(10).to_string())
        sys.exit(1)

    log(f"Parsed AFR sheet: {len(parsed):,} rows, {len(parsed.columns)} columns")
    log(f"Columns: {list(parsed.columns)[:15]}")
    log(f"First 3 rows:\n{parsed.head(3).to_string()}")

    # Lookup
    merged, afr_cols = lookup_eur_amplified_in_afr(eur_leads_32, parsed, rsid_col)
    n_match = merged.dropna(subset=["EUR_BETA"]).shape[0]
    if afr_cols["a1"]:
        n_match = merged[merged[afr_cols["a1"]].notna()].shape[0]
    log(f"Direction concordance lookup: {n_match} of 32 EUR-amplified leads found in AFR sumstats")

    # Concordance
    metrics = compute_concordance(merged, afr_cols)
    log(f"\n=== REPLICATION METRICS ===")
    for k, v in metrics.items():
        log(f"  {k}: {v}")

    # Write output
    merged.to_csv(OUT_TSV, sep="\t", index=False)
    log(f"\nWrote per-variant detail: {OUT_TSV}")

    # Markdown summary
    with open(OUT_MD, "w") as fh:
        fh.write("# Phase 19r — Bigdeli 2026 AFR-discovery replication\n\n")
        fh.write(f"**Date:** 2026-05-16\n\n")
        fh.write(f"## Bigdeli source\n")
        fh.write(f"Files used: {[f.name for f in bigdeli_files]}\n")
        fh.write(f"Sheet selected: `{sheet_name}` ({len(parsed):,} rows)\n\n")
        fh.write(f"## Lookup\n")
        fh.write(f"- Phase 19 EUR-amplified leads: 32\n")
        fh.write(f"- Matched in Bigdeli AFR sumstats: {n_match}\n\n")
        fh.write(f"## Metrics\n")
        for k, v in metrics.items():
            fh.write(f"- **{k}**: {v}\n")
        fh.write("\n")
        fh.write(f"## Tier decision criterion\n")
        if "concordance" in metrics and not np.isnan(metrics["concordance"]):
            c = metrics["concordance"]
            if c >= 0.80:
                fh.write(f"Concordance {c:.2%} ≥ 80% → **Mol Psychiatry primary target**\n")
            elif c >= 0.65:
                fh.write(f"Concordance {c:.2%} in [0.65, 0.80) → **AJHG primary target** (more conservative)\n")
            else:
                fh.write(f"Concordance {c:.2%} < 65% → **Revise narrative**; H4 winner's curse may dominate\n")
    log(f"Wrote summary: {OUT_MD}")
    log("\nPhase 19r complete.")


if __name__ == "__main__":
    main()
