#!/usr/bin/env python3
"""
Phase 14p6b — Merge AFR-TMRCA, partition variants, compute heritability proxy
==============================================================================
After P14p6a SLURM array completes 39 chr arms:

  1. Concatenate per-arm TSVs → single parquet
  2. Merge with: variant_master (effect_allele, beta, se, pval, maf, akbari_s,
     credible_set_id, pip), cluster assignments (C0/C1/C2 from v10)
  3. Define AFR-context partitions (data-driven thresholds, NOT pre-OOA labels):
       - AFR_absent          (n_AFR_carriers == 0)
       - AFR_low_TMRCA       (TMRCA < 50 kyr)
       - AFR_mid_TMRCA       (50–200 kyr)
       - AFR_high_TMRCA      (200–500 kyr)
       - AFR_deep_TMRCA      (≥ 500 kyr)
  4. Heritability proxy per partition (without LDSC):
       (a) mean(chi-squared) = mean((BETA/SE)^2) — relative h² signal per
           variant set; higher = more h² concentrated
       (b) max(-log10 P) within partition
       (c) sum(BETA^2 / SE^2) — total signal in the partition
     Note: This is a SUMSTATS-based h² proxy, not formal s-LDSC partitioned
     heritability.  The latter requires LDSC (not installed in the HPC cluster v11 env);
     defer to a follow-up if cleaner h² inference is needed.
  5. Cross-tabulate AFR partition × v10 GMM cluster (descriptive)

Output:
  results/phase14p6/P14p6b_AFR_TMRCA_per_variant.parquet
  results/phase14p6/P14p6b_AFR_partition_assignments.tsv
  results/phase14p6/P14p6b_h2_proxy_per_partition_EUR.tsv
  results/phase14p6/P14p6b_h2_proxy_per_partition_AFRAM.tsv
  results/phase14p6/P14p6b_AFR_partition_x_v10_cluster.tsv
  results/phase14p6/P14p6b_NARRATIVE.md
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime
import glob, gzip, json
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
DATA = Path((_SCRATCH + "/v11_data/phase14p6"))
OUT  = BASE / "results/phase14p6"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*72)
log("Phase 14p6b — Merge + partition + heritability proxy")
log("="*72)

# ── 1. Merge per-arm TSVs ──────────────────────────────────────────
log("[1] Merging per-arm AFR_TMRCA TSVs")
arm_files = sorted(glob.glob(str(DATA / "P14p6a_chr*.tsv")))
log(f"  arm files: {len(arm_files)}")
parts = []
for f in arm_files:
    try:
        if Path(f).stat().st_size < 50:  # empty or header-only file
            log(f"  skip empty/tiny file: {Path(f).name}")
            continue
        d = pd.read_csv(f, sep="\t")
        if len(d):
            parts.append(d)
    except (pd.errors.EmptyDataError, ValueError) as e:
        log(f"  skip unparseable file: {Path(f).name} ({e})")
df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
log(f"  total AFR-context-aged variants: {len(df):,}")

# ── 2. Merge with variant master + cluster ─────────────────────────
log("\n[2] Merging with variant_master + cluster + Akbari S")
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
# Don't double-include chr/pos (P14p6a already has them)
keep = ["rsid", "effect_allele", "other_allele",
        "beta", "se", "pval", "maf", "akbari_s", "akbari_s_effect",
        "credible_set_id", "pip"]
keep = [c for c in keep if c in vm.columns]
df = df.merge(vm[keep], on="rsid", how="left")

clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid", "cluster"]]
df = df.merge(clu, on="rsid", how="left")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
log(f"  with cluster assignments: {df['cluster'].notna().sum()}")

# Compute χ² from beta/se when available (using genome-wide significance proxies)
if "beta" in df.columns and "se" in df.columns:
    df["chi2"] = (df["beta"] / df["se"]) ** 2
    df["neglog10p"] = -np.log10(df["pval"].clip(lower=1e-300))
else:
    df["chi2"] = np.nan
    df["neglog10p"] = np.nan
log(f"  with χ² computed: {df['chi2'].notna().sum()}")

# ── 3. AFR-context partitions ──────────────────────────────────────
log("\n[3] AFR-context partition definition (TMRCA bins)")

def partition(row):
    if row["AFR_absent"]:
        return "AFR_absent"
    t = row["AFR_TMRCA_yr"]
    if pd.isna(t) or t < 0:
        return "AFR_singleton_or_undef"
    if t < 50_000:        return "AFR_TMRCA_lt_50kyr"
    if t < 200_000:       return "AFR_TMRCA_50_200kyr"
    if t < 500_000:       return "AFR_TMRCA_200_500kyr"
    return                       "AFR_TMRCA_ge_500kyr"

df["AFR_partition"] = df.apply(partition, axis=1)

partition_counts = df["AFR_partition"].value_counts().sort_index()
log(f"  Partition counts:")
for k, v in partition_counts.items():
    log(f"    {k:>30s}: {v:>6,} ({v/len(df)*100:.1f}%)")

# Save
df.to_parquet(OUT / "P14p6b_AFR_TMRCA_per_variant.parquet")
df[["rsid","chr","pos_hg38","AFR_DAF_in_tree","AFR_TMRCA_yr","AFR_absent","AFR_partition","cluster"]].to_csv(
    OUT / "P14p6b_AFR_partition_assignments.tsv", sep="\t", index=False
)
log(f"\n  Saved parquet + partition assignments")

# ── 4. Heritability proxy per partition (EUR sumstats from variant_master beta/se) ──
log("\n[4] Heritability proxy per partition (sumstats-based: mean χ² + others)")
log("    Note: not formal s-LDSC partitioned h²; this is a SUMSTATS-aggregate proxy.")
log("    Variant-level sumstats here = PGC3 EUR (variant_master_v4 beta/se is from PGC3 EUR fine-mapped credible-set).")

h2_rows = []
for part_name, sub in df.groupby("AFR_partition"):
    chi2 = sub["chi2"].dropna()
    if len(chi2) == 0: continue
    h2_rows.append({
        "partition": part_name,
        "n_variants": int(len(sub)),
        "n_chi2_available": int(len(chi2)),
        "mean_chi2": float(chi2.mean()),
        "median_chi2": float(chi2.median()),
        "q95_chi2": float(chi2.quantile(0.95)),
        "sum_chi2": float(chi2.sum()),
        "mean_neglog10P": float(sub["neglog10p"].mean()),
        "median_neglog10P": float(sub["neglog10p"].median()),
        "mean_maf": float(sub["maf"].mean()),
        "median_AFR_TMRCA_yr": float(sub["AFR_TMRCA_yr"].median()) if sub["AFR_TMRCA_yr"].notna().any() else np.nan,
    })

# Add genome-wide baseline (all credible-set variants)
all_chi2 = df["chi2"].dropna()
h2_rows.append({
    "partition": "ALL_credible_set (baseline)",
    "n_variants": int(len(df)),
    "n_chi2_available": int(len(all_chi2)),
    "mean_chi2": float(all_chi2.mean()),
    "median_chi2": float(all_chi2.median()),
    "q95_chi2": float(all_chi2.quantile(0.95)),
    "sum_chi2": float(all_chi2.sum()),
    "mean_neglog10P": float(df["neglog10p"].mean()),
    "median_neglog10P": float(df["neglog10p"].median()),
    "mean_maf": float(df["maf"].mean()),
    "median_AFR_TMRCA_yr": np.nan,
})

h2_df = pd.DataFrame(h2_rows)
# Sort: ALL first, then partition order
ord_map = {"ALL_credible_set (baseline)": 0, "AFR_absent": 1, "AFR_singleton_or_undef": 2,
           "AFR_TMRCA_lt_50kyr": 3, "AFR_TMRCA_50_200kyr": 4,
           "AFR_TMRCA_200_500kyr": 5, "AFR_TMRCA_ge_500kyr": 6}
h2_df["_sort"] = h2_df["partition"].map(ord_map).fillna(99)
h2_df = h2_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
h2_df.to_csv(OUT / "P14p6b_h2_proxy_per_partition_EUR.tsv", sep="\t", index=False)
log("  EUR sumstats h² proxy table:")
log(h2_df.to_string(index=False))

# ── 4b. AFRAM heritability proxy ───────────────────────────────────
log("\n[4b] AFRAM h² proxy (PGC3 wave3 African American sumstats)")
afram_path = (os.environ.get("PGC_SUMSTATS_DIR", _ROOT+"/data") + "/Scz_PGC_GWAS/PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz")
# Fallback search
if not Path(afram_path).exists():
    afram_candidates = glob.glob((os.environ.get("PGC_SUMSTATS_DIR", _ROOT+"/data") + "/**/PGC3_SCZ_wave3.afram.autosome*.gz"), recursive=True)
    afram_candidates += glob.glob(os.path.expanduser("~/**/PGC3_SCZ_wave3.afram.autosome*.gz"), recursive=True)
    if afram_candidates:
        afram_path = afram_candidates[0]
log(f"  AFRAM sumstats: {afram_path}")
afram_rsids_path = OUT / "_afram_lookup_cache.tsv"

if Path(afram_path).exists() and not afram_rsids_path.exists():
    log("  Streaming AFRAM (only PGC3 credible-set rsids)...")
    keep_rsids = set(df["rsid"].astype(str))
    rows = []
    with gzip.open(afram_path, "rt") as fh:
        header_seen = False
        idx = {}
        n = 0
        for line in fh:
            if line.startswith("##"): continue
            if not header_seen:
                cols = line.rstrip("\n").split("\t")
                idx = {c: i for i, c in enumerate(cols)}
                header_seen = True
                continue
            parts_l = line.rstrip("\n").split("\t")
            rsid = parts_l[idx.get("ID", 1)]
            if rsid not in keep_rsids: continue
            try:
                b = float(parts_l[idx["BETA"]])
                s = float(parts_l[idx["SE"]])
                p = float(parts_l[idx["PVAL"]])
            except (ValueError, KeyError):
                continue
            rows.append({"rsid": rsid, "afram_beta": b, "afram_se": s, "afram_pval": p})
            n += 1
    af_df = pd.DataFrame(rows)
    af_df.to_csv(afram_rsids_path, sep="\t", index=False)
    log(f"  AFRAM matches for credible-set rsids: {len(af_df):,}")
else:
    af_df = pd.read_csv(afram_rsids_path, sep="\t") if afram_rsids_path.exists() else pd.DataFrame()

if len(af_df) > 0:
    af_df["afram_chi2"] = (af_df["afram_beta"] / af_df["afram_se"]) ** 2
    df_af = df.merge(af_df, on="rsid", how="left")
    log(f"  variants with AFRAM χ² available: {df_af['afram_chi2'].notna().sum():,}")

    h2af_rows = []
    for part_name, sub in df_af.groupby("AFR_partition"):
        c = sub["afram_chi2"].dropna()
        if len(c) == 0: continue
        h2af_rows.append({
            "partition": part_name,
            "n_variants": int(len(sub)),
            "n_afram_chi2_available": int(len(c)),
            "mean_afram_chi2": float(c.mean()),
            "median_afram_chi2": float(c.median()),
            "q95_afram_chi2": float(c.quantile(0.95)),
            "sum_afram_chi2": float(c.sum()),
        })
    h2af_rows.append({
        "partition": "ALL_credible_set (baseline)",
        "n_variants": int(len(df_af)),
        "n_afram_chi2_available": int(df_af["afram_chi2"].notna().sum()),
        "mean_afram_chi2": float(df_af["afram_chi2"].mean()),
        "median_afram_chi2": float(df_af["afram_chi2"].median()),
        "q95_afram_chi2": float(df_af["afram_chi2"].quantile(0.95)),
        "sum_afram_chi2": float(df_af["afram_chi2"].sum()),
    })
    h2af = pd.DataFrame(h2af_rows)
    h2af["_sort"] = h2af["partition"].map(ord_map).fillna(99)
    h2af = h2af.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    h2af.to_csv(OUT / "P14p6b_h2_proxy_per_partition_AFRAM.tsv", sep="\t", index=False)
    log("  AFRAM sumstats h² proxy table:")
    log(h2af.to_string(index=False))
else:
    log("  AFRAM sumstats not accessible — skipped")

# ── 5. Cross-tabulate AFR partition × v10 GMM cluster ──────────────
log("\n[5] AFR partition × v10 cluster cross-tabulation")
ct_rows = []
sub_with_cluster = df.dropna(subset=["cluster"]).copy()
sub_with_cluster["cluster"] = sub_with_cluster["cluster"].astype(int)
for c in sorted(sub_with_cluster["cluster"].unique()):
    for p in sub_with_cluster["AFR_partition"].dropna().unique():
        cnt = int(((sub_with_cluster["cluster"] == c) & (sub_with_cluster["AFR_partition"] == p)).sum())
        ct_rows.append({"cluster": f"C{int(c)}", "AFR_partition": p, "count": cnt})

ct = pd.DataFrame(ct_rows).pivot_table(index="cluster", columns="AFR_partition",
                                         values="count", fill_value=0)
ct.to_csv(OUT / "P14p6b_AFR_partition_x_v10_cluster.tsv", sep="\t")
log("\n  Cross-tab (rows = v10 cluster, cols = AFR partition):")
log(ct.to_string())

# Pct-by-cluster
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
log("\n  Same, as %s within cluster:")
log(ct_pct.round(1).to_string())

# ── Narrative ──────────────────────────────────────────────────────
with open(OUT / "P14p6b_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p6b — AFR-TMRCA merge, partition, h² proxy\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write("## Method (proper)\n\n")
    f.write("- AFR-context TMRCA = MRCA of AFR sample descendants of the allele-matched mutation(s) in Wohns 2022 unified tree\n")
    f.write("- AFR_absent: variant has zero AFR-clade carriers (likely post-OOA EUR-derived)\n")
    f.write("- Partitions defined by TMRCA threshold bins (data-driven, no AMH/Crow labels)\n")
    f.write("- h² PROXY: sumstats-based mean χ² per partition (not formal s-LDSC)\n\n")
    f.write("## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")

log("\nPhase 14p6b complete.")
