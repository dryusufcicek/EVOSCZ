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
  - P14p2_run.log                         — run log
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
log_path = OUT / "P14p2_run.log"
with open(log_path, "w") as f:
    f.write("\n".join(LOG) + "\n")
log(f"\nWrote run log: {log_path}")
log("\nPhase 14p2 complete.")
