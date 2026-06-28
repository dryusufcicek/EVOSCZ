#!/usr/bin/env python3
"""
Phase 14p2 — AFR allele frequency lookup for PGC3 EUR credible-set.

For each of the 20,637 PGC3 EUR fine-mapped credible-set variants, retrieve
allele frequency in the PGC3 African American sample (5,998 cases + 3,826
controls; total N=9,824). Compute combined f_AFR and classify each variant
into AFR ancestry-context partitions:

  AFR_polymorphic       : f_AFR >= 0.01  → candidate pre-OOA AMH-shared
  AFR_rare              : 0 < f_AFR < 0.01
  AFR_monomorphic       : f_AFR == 0 or absent → likely post-OOA EUR-derived
  absent_from_AFRAM     : variant not in AFRAM sumstats panel

This is the foundation for H_AFR pre-OOA AMH-shared partition test
(Phase 14q-rev in PRIOR_ART document).

Inputs:
  - PGC3 EUR credible-set: results/phase11/variant_master_v4.parquet
  - PGC3 AFRAM sumstats: ~/Downloads/Scz_PGC_GWAS/PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz
  - Cluster assignments: results/phase14b/P14b_v3_cluster_assignments.tsv.gz

Outputs (results/phase14p_baseline/):
  - P14p2_a_AFR_freq_lookup.tsv.gz       — per-rsid AFR allele frequency
  - P14p2_b_partition_summary.tsv         — overall + per-cluster partition counts
  - P14p2_NARRATIVE.md                    — narrative log
"""

import gzip
from pathlib import Path
import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
AFRAM = Path(os.environ.get("PGC_SUMSTATS_DIR", "data/pgc_sumstats")) / "PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz"
OUT = BASE / "results/phase14p_baseline"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []

# Per Trubetskoy 2022 + PHASE14_RESOURCE_EVALUATION.md
N_AFR_CASES = 5998
N_AFR_CONTROLS = 3826
N_AFR_TOTAL = N_AFR_CASES + N_AFR_CONTROLS


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.append(line)


log("=" * 72)
log("Phase 14p2 — AFR allele frequency lookup for PGC3 EUR credible-set")
log("=" * 72)

# Load credible-set rsids
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
credset_rsids = set(vm["rsid"].dropna().astype(str).unique())
log(f"\n[1] Credible-set rsids: {len(credset_rsids):,}")

# Load cluster assignments for downstream merge
clu = pd.read_csv(
    BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t"
)
log(f"  cluster assignments: {len(clu):,}")

# Stream AFRAM sumstats
log(f"\n[2] Streaming AFRAM sumstats: {AFRAM.name}")
log(f"    (skip ## lines, header is first non-## line)")
log(f"    AFR N: {N_AFR_CASES} cases + {N_AFR_CONTROLS} controls = {N_AFR_TOTAL}")

hits = []
n_data_lines = 0
n_matched = 0

with gzip.open(AFRAM, "rt") as f:
    header_seen = False
    col_idx = {}
    for line in f:
        if line.startswith("##"):
            continue
        if not header_seen:
            cols = line.rstrip("\n").split("\t")
            for i, c in enumerate(cols):
                col_idx[c] = i
            header_seen = True
            log(f"    header parsed: {cols}")
            continue
        n_data_lines += 1
        if n_data_lines % 1_000_000 == 0:
            log(f"    scanned {n_data_lines:,} rows, matched {n_matched:,}")
        parts = line.rstrip("\n").split("\t")
        try:
            rsid = parts[col_idx["ID"]]
        except IndexError:
            continue
        if rsid not in credset_rsids:
            continue
        try:
            chr_ = parts[col_idx["CHROM"]]
            pos = parts[col_idx["POS"]]
            a1 = parts[col_idx["A1"]]
            a2 = parts[col_idx["A2"]]
            fcas = float(parts[col_idx["FCAS"]])
            fcon = float(parts[col_idx["FCON"]])
            impinfo = float(parts[col_idx["IMPINFO"]])
            beta = float(parts[col_idx["BETA"]])
            se = float(parts[col_idx["SE"]])
            pval = float(parts[col_idx["PVAL"]])
        except (ValueError, IndexError):
            continue
        f_afr = (fcas * N_AFR_CASES + fcon * N_AFR_CONTROLS) / N_AFR_TOTAL
        hits.append({
            "rsid": rsid,
            "chr": chr_,
            "pos_grch37": int(pos),
            "A1": a1, "A2": a2,
            "AFR_FCAS": fcas, "AFR_FCON": fcon, "AFR_f": f_afr,
            "AFR_IMPINFO": impinfo,
            "AFR_BETA": beta, "AFR_SE": se, "AFR_P": pval,
        })
        n_matched += 1

log(f"\n  AFRAM scan complete: {n_data_lines:,} data rows scanned")
log(f"  matched in credible-set: {n_matched:,}/{len(credset_rsids):,} "
    f"({n_matched/len(credset_rsids)*100:.2f}%)")

afr_df = pd.DataFrame(hits)
afr_df.to_csv(OUT / "P14p2_a_AFR_freq_lookup.tsv.gz", sep="\t", index=False,
              compression="gzip")
log(f"  saved → P14p2_a_AFR_freq_lookup.tsv.gz")

# Classify
def classify(row):
    f = row["AFR_f"]
    if pd.isna(f):
        return "absent_from_AFRAM"
    if f >= 0.01:
        return "AFR_polymorphic"
    if f > 0:
        return "AFR_rare"
    return "AFR_monomorphic"

# Merge with all credible-set rsids (including ones absent from AFRAM)
all_credset = pd.DataFrame({"rsid": sorted(credset_rsids)})
merged = all_credset.merge(afr_df, on="rsid", how="left")
merged["AFR_partition"] = merged.apply(classify, axis=1)

# Add cluster assignment
merged = merged.merge(clu[["rsid", "cluster"]], on="rsid", how="left")

# Save full table
merged.to_csv(OUT / "P14p2_a_AFR_freq_full.tsv.gz", sep="\t", index=False,
              compression="gzip")
log(f"  saved → P14p2_a_AFR_freq_full.tsv.gz (full credible-set merge)")

# Partition summary
log(f"\n[3] AFR partition summary")
overall = merged["AFR_partition"].value_counts()
log("  Overall (all credible-set):")
for k, v in overall.items():
    log(f"    {k:>22s}: {v:>6,} ({v/len(merged)*100:.1f}%)")

# Per-cluster summary
log("\n  Per-cluster:")
cluster_summary = []
for c in [0.0, 1.0, 2.0]:
    sub = merged[merged["cluster"] == c]
    if len(sub) == 0:
        continue
    vc = sub["AFR_partition"].value_counts()
    log(f"  Cluster C{int(c)} (n={len(sub):,}):")
    for k in ["AFR_polymorphic", "AFR_rare", "AFR_monomorphic", "absent_from_AFRAM"]:
        v = vc.get(k, 0)
        log(f"    {k:>22s}: {v:>6,} ({v/len(sub)*100:.1f}%)")
        cluster_summary.append({
            "cluster": f"C{int(c)}",
            "AFR_partition": k,
            "count": int(v),
            "pct_of_cluster": float(v / len(sub) * 100),
            "n_cluster": len(sub),
        })

overall_rows = []
for k in ["AFR_polymorphic", "AFR_rare", "AFR_monomorphic", "absent_from_AFRAM"]:
    v = overall.get(k, 0)
    overall_rows.append({
        "cluster": "all",
        "AFR_partition": k,
        "count": int(v),
        "pct_of_cluster": float(v / len(merged) * 100),
        "n_cluster": len(merged),
    })

summary_df = pd.DataFrame(overall_rows + cluster_summary)
summary_df.to_csv(OUT / "P14p2_b_partition_summary.tsv", sep="\t", index=False)
log(f"\n  saved → P14p2_b_partition_summary.tsv")

# AFR allele frequency distribution per cluster
log(f"\n[4] AFR allele frequency distribution per cluster")
freq_rows = []
for c in [0.0, 1.0, 2.0]:
    sub = merged[(merged["cluster"] == c) & merged["AFR_f"].notna()]
    if len(sub) == 0:
        continue
    f = sub["AFR_f"].values
    row = {
        "cluster": f"C{int(c)}",
        "n_with_AFR_freq": len(sub),
        "AFR_f_median": float(np.median(f)),
        "AFR_f_mean": float(np.mean(f)),
        "AFR_f_q01": float(np.quantile(f, 0.01)),
        "AFR_f_q99": float(np.quantile(f, 0.99)),
        "pct_AFR_polymorphic_ge1pct": float((f >= 0.01).mean() * 100),
        "pct_AFR_polymorphic_ge5pct": float((f >= 0.05).mean() * 100),
    }
    freq_rows.append(row)
    log(f"    C{int(c)}: n={row['n_with_AFR_freq']:,}, "
        f"median f_AFR={row['AFR_f_median']:.3f}, "
        f"pct >=1%={row['pct_AFR_polymorphic_ge1pct']:.1f}%, "
        f"pct >=5%={row['pct_AFR_polymorphic_ge5pct']:.1f}%")

pd.DataFrame(freq_rows).to_csv(OUT / "P14p2_c_AFR_freq_per_cluster.tsv",
                                 sep="\t", index=False)
log(f"  saved → P14p2_c_AFR_freq_per_cluster.tsv")

# Narrative
narr_path = OUT / "P14p2_NARRATIVE.md"
narr = []
narr.append(f"# Phase 14p2 — AFR Allele Frequency Lookup for PGC3 EUR Credible-Set\n\n")
narr.append(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n")
narr.append(f"**Purpose:** Foundation for H_AFR pre-OOA AMH-shared partition test.\n\n")

narr.append(f"## Inputs\n")
narr.append(f"- PGC3 EUR credible-set: `variant_master_v4.parquet` ({len(credset_rsids):,} rsids)\n")
narr.append(f"- PGC3 AFRAM sumstats: `PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz`\n")
narr.append(f"  (N = {N_AFR_CASES} cases + {N_AFR_CONTROLS} controls = {N_AFR_TOTAL})\n\n")

narr.append(f"## Match rate\n")
narr.append(f"- AFRAM sumstats rows scanned: {n_data_lines:,}\n")
narr.append(f"- Credible-set rsids matched in AFRAM: {n_matched:,}/{len(credset_rsids):,} "
            f"({n_matched/len(credset_rsids)*100:.2f}%)\n\n")

narr.append(f"## AFR partition classification\n\n")
narr.append(f"**Overall (all {len(merged):,} credible-set variants):**\n\n")
narr.append(f"| Partition | Count | % |\n|---|---|---|\n")
for k in ["AFR_polymorphic", "AFR_rare", "AFR_monomorphic", "absent_from_AFRAM"]:
    v = overall.get(k, 0)
    narr.append(f"| {k} | {v:,} | {v/len(merged)*100:.2f}% |\n")

narr.append(f"\n**Per-cluster:**\n\n")
narr.append(f"| Cluster | n | AFR_polymorphic | AFR_rare | AFR_monomorphic | absent_from_AFRAM |\n")
narr.append(f"|---|---|---|---|---|---|\n")
for c in [0.0, 1.0, 2.0]:
    sub = merged[merged["cluster"] == c]
    if len(sub) == 0:
        continue
    vc = sub["AFR_partition"].value_counts()
    parts = " | ".join(
        f"{vc.get(k, 0):,} ({vc.get(k, 0)/len(sub)*100:.1f}%)"
        for k in ["AFR_polymorphic", "AFR_rare", "AFR_monomorphic", "absent_from_AFRAM"]
    )
    narr.append(f"| C{int(c)} | {len(sub):,} | {parts} |\n")

narr.append(f"\n## AFR allele frequency distribution per cluster\n\n")
narr.append(f"| Cluster | n | median f_AFR | mean f_AFR | q01 | q99 | %≥1% | %≥5% |\n|---|---|---|---|---|---|---|---|\n")
for row in freq_rows:
    narr.append(f"| {row['cluster']} | {row['n_with_AFR_freq']:,} | "
                f"{row['AFR_f_median']:.3f} | {row['AFR_f_mean']:.3f} | "
                f"{row['AFR_f_q01']:.3f} | {row['AFR_f_q99']:.3f} | "
                f"{row['pct_AFR_polymorphic_ge1pct']:.1f}% | "
                f"{row['pct_AFR_polymorphic_ge5pct']:.1f}% |\n")

narr.append(f"\n## Interpretation guide\n\n")
narr.append(f"- **AFR_polymorphic (f_AFR ≥ 1%)**: variant segregates in AFR ancestry. ")
narr.append(f"Candidate for pre-OOA AMH-shared if deep TMRCA (test in next step with Wohns 2022).\n")
narr.append(f"- **AFR_rare (0 < f_AFR < 1%)**: present but uncommon in AFR. ")
narr.append(f"Could be drift-affected or recently-derived ancestral.\n")
narr.append(f"- **AFR_monomorphic (f_AFR == 0)**: derived in non-AFR lineage post-OOA; ")
narr.append(f"strong candidate for EUR-private post-OOA-derived.\n")
narr.append(f"- **absent_from_AFRAM**: not in AFRAM sumstats panel (low imputation quality, ")
narr.append(f"reference mismatch, or genuinely absent).\n\n")

narr.append(f"## Next step\n\n")
narr.append(f"Cross-reference AFR_polymorphic subset with Wohns 2022 unified tree sequence "
            f"to derive deep-time-valid TMRCA. Variants with both (AFR-polymorphic) AND "
            f"(Wohns AFR-context TMRCA > 60 kyr) → genuine pre-OOA AMH-shared partition.\n\n")

narr.append(f"## Files\n")
narr.append(f"- `P14p2_a_AFR_freq_lookup.tsv.gz`     — per-rsid AFR allele frequency (AFRAM hits only)\n")
narr.append(f"- `P14p2_a_AFR_freq_full.tsv.gz`       — full credible-set + AFR partition + cluster\n")
narr.append(f"- `P14p2_b_partition_summary.tsv`      — partition counts per cluster + overall\n")
narr.append(f"- `P14p2_c_AFR_freq_per_cluster.tsv`   — AFR freq distribution per cluster\n\n")

narr.append(f"## Run log\n\n```\n")
narr.extend(line + "\n" for line in LOG)
narr.append("```\n")

with open(narr_path, "w") as f:
    f.writelines(narr)
log(f"\nWrote: {narr_path}")
log("\nPhase 14p2 complete.")
