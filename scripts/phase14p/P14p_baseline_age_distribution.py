#!/usr/bin/env python3
"""
Phase 14p — Baseline age-distribution comparison.

Tests whether PGC3 SCZ credible-set ages are systematically younger/older
than MAF+L2-matched HapMap3 control variants drawn from the same demographic
substrate.

This closes Audit Issue 8: "GWAS preferentially discovers common variants,
and common variants are typically old. The age distribution of PGC3 variants
reflects MAF ascertainment bias, not necessarily evolutionary pressure on
SCZ-relevant alleles. The meaningful question is: are SCZ variants OLDER
than MAF-matched controls? This is not tested."

Approach:
  1. PGC3 ages: from Atlas of Variant Age (AgeMedian_Mut, DataSource='TGP'),
     same convention as Phase 13a.
  2. Control ages: same Atlas convention, joined to matched_controls.tsv.gz
     via (chr, pos).
  3. Compute pooled and MAF-decile-stratified distribution comparisons.
  4. Per-cluster (C0/C1/C2) comparison against MAF-stratified controls.

Output (results/phase14p_baseline/):
  - P14p_a_age_lookup.parquet        — per-rsid (PGC3+control) age table
  - P14p_b_pooled_tests.tsv          — pooled distribution tests
  - P14p_c_maf_stratified_tests.tsv  — per MAF-decile distribution tests
  - P14p_d_cluster_stratified.tsv    — per cluster vs MAF-matched controls
  - P14p_e_density_pgc_vs_control.pdf
  - P14p_NARRATIVE.md
"""

import sys
import gzip
from pathlib import Path
import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
ATLAS  = Path(os.environ.get("ATLAS_DIR", "data/atlas_variant_age"))
OUT    = BASE / "results/phase14p_baseline"
OUT.mkdir(parents=True, exist_ok=True)
LOG    = []


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ─────────────────────────────────────────────────────────────────────
# Step 1: Load matched controls + variant master
# ─────────────────────────────────────────────────────────────────────
log("=" * 72)
log("Phase 14p — Baseline age-distribution comparison")
log("=" * 72)

log("\n[1] Loading matched controls + variant master + cluster assignments")

def _norm_chr(s):
    """Normalize chromosome to plain string '1'..'22' regardless of float/int input."""
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return str(s)


controls = pd.read_csv(BASE / "data/processed/matched_controls.tsv.gz", sep="\t")
controls["control_chr"] = controls["control_chr"].map(_norm_chr)
log(f"  matched controls: {len(controls):,} mappings; "
    f"{controls['control_snp'].nunique():,} unique control SNPs; "
    f"{controls['pgc3_rsid'].nunique():,} unique PGC3 rsids")

vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
vm["chr"] = vm["chr"].map(_norm_chr)
log(f"  variant_master_v4: {len(vm):,} rows × {vm.shape[1]} columns")

clu = pd.read_csv(
    BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t"
)
log(f"  cluster assignments: {len(clu):,}")
log(f"  cluster sizes: " + ", ".join(
    f"C{int(k)}={v}" for k, v in clu["cluster"].value_counts().sort_index().items()
))


# ─────────────────────────────────────────────────────────────────────
# Step 2: Stream Atlas TGP ages for PGC3 + controls, per chromosome
# ─────────────────────────────────────────────────────────────────────
log("\n[2] Streaming Atlas of Variant Age (TGP source) per chromosome")
log("    Using AgeMedian_Mut, DataSource='TGP' (consistent with Phase 13a)")

# Build per-chr (chr, pos) sets we need to look up
pgc3_positions = vm[["rsid", "chr", "pos"]].copy()
pgc3_positions["pos"] = pd.to_numeric(pgc3_positions["pos"], errors="coerce").astype("Int64")
pgc3_positions = pgc3_positions.dropna(subset=["pos"])
pgc3_positions["pos"] = pgc3_positions["pos"].astype(int)
pgc3_positions["source"] = "pgc3"

ctrl_positions = controls[["control_snp", "control_chr", "control_pos"]].rename(
    columns={"control_snp": "rsid", "control_chr": "chr", "control_pos": "pos"}
)
ctrl_positions["pos"] = ctrl_positions["pos"].astype(int)
ctrl_positions["source"] = "control"

needed = pd.concat([pgc3_positions, ctrl_positions], ignore_index=True).drop_duplicates(
    subset=["chr", "pos"]
)
log(f"  unique (chr,pos) needing GEVA lookup: {len(needed):,}")

age_lookup = {}  # (chr, pos) → AgeMedian_Mut (TGP)

for chrom in [str(i) for i in range(1, 23)]:
    gz = ATLAS / f"atlas.chr{chrom}.csv.gz"
    if not gz.exists():
        log(f"  chr{chrom}: file missing, skipping")
        continue

    chr_targets = needed[needed["chr"] == chrom]
    target_positions = set(chr_targets["pos"].values)
    if not target_positions:
        continue

    matched = 0
    scanned = 0
    with gzip.open(gz, "rt") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("##") or not line:
                continue
            if header is None:
                header = [c.strip() for c in line.split(",")]
                try:
                    pos_idx   = header.index("Position")
                    age_idx   = header.index("AgeMedian_Mut")
                    src_idx   = header.index("DataSource")
                    chr_idx   = header.index("Chromosome")
                except ValueError:
                    log(f"  chr{chrom}: unexpected header, skipping")
                    break
                continue
            scanned += 1
            cols = line.split(",")
            if len(cols) <= max(pos_idx, age_idx, src_idx):
                continue
            src = cols[src_idx].strip()
            if src != "TGP":
                continue
            try:
                pos = int(cols[pos_idx].strip())
            except ValueError:
                continue
            if pos not in target_positions:
                continue
            try:
                age = float(cols[age_idx].strip())
            except ValueError:
                continue
            age_lookup[(chrom, pos)] = age
            matched += 1
    log(f"  chr{chrom}: {scanned:,} TGP scanned, {matched:,} matched "
        f"({matched/len(target_positions)*100:.1f}% of {len(target_positions):,} targets)")

log(f"\n  age_lookup size: {len(age_lookup):,}")


# ─────────────────────────────────────────────────────────────────────
# Step 3: Annotate PGC3 + controls with looked-up Atlas ages
# ─────────────────────────────────────────────────────────────────────
log("\n[3] Annotating PGC3 and controls with looked-up Atlas TGP ages")

def lookup_age(row):
    return age_lookup.get((row["chr"], row["pos"]))

pgc3_positions["age_tgp"] = pgc3_positions.apply(lookup_age, axis=1)
ctrl_positions["age_tgp"] = ctrl_positions.apply(lookup_age, axis=1)

n_pgc_aged = pgc3_positions["age_tgp"].notna().sum()
n_ctl_aged = ctrl_positions["age_tgp"].notna().sum()
log(f"  PGC3   with Atlas TGP age: {n_pgc_aged:,}/{len(pgc3_positions):,} "
    f"({n_pgc_aged/len(pgc3_positions)*100:.1f}%)")
log(f"  Control with Atlas TGP age: {n_ctl_aged:,}/{len(ctrl_positions):,} "
    f"({n_ctl_aged/len(ctrl_positions)*100:.1f}%)")

pgc3_aged = pgc3_positions.dropna(subset=["age_tgp"]).copy()
ctrl_aged = ctrl_positions.dropna(subset=["age_tgp"]).copy()

# Persist age lookup for downstream reuse
combined = pd.concat([pgc3_aged, ctrl_aged], ignore_index=True)
combined.to_parquet(OUT / "P14p_a_age_lookup.parquet")
log(f"  saved → P14p_a_age_lookup.parquet ({len(combined):,} rows)")


# ─────────────────────────────────────────────────────────────────────
# Step 4: Pooled distribution comparison
# ─────────────────────────────────────────────────────────────────────
log("\n[4] Pooled distribution comparison")

scz_ages = pgc3_aged["age_tgp"].values
ctl_ages = ctrl_aged["age_tgp"].values

def dist_summary(ages, label):
    return {
        "group": label,
        "n":       len(ages),
        "min":     float(np.min(ages)),
        "q01":     float(np.quantile(ages, 0.01)),
        "q25":     float(np.quantile(ages, 0.25)),
        "median":  float(np.median(ages)),
        "mean":    float(np.mean(ages)),
        "q75":     float(np.quantile(ages, 0.75)),
        "q95":     float(np.quantile(ages, 0.95)),
        "q99":     float(np.quantile(ages, 0.99)),
        "q999":    float(np.quantile(ages, 0.999)),
        "max":     float(np.max(ages)),
        "pct_lt_5k":   float((ages < 5_000).mean() * 100),
        "pct_lt_10k":  float((ages < 10_000).mean() * 100),
        "pct_lt_20k":  float((ages < 20_000).mean() * 100),
        "pct_lt_50k":  float((ages < 50_000).mean() * 100),
        "pct_lt_100k": float((ages < 100_000).mean() * 100),
    }

summary_rows = [
    dist_summary(scz_ages, "PGC3_credible_set"),
    dist_summary(ctl_ages, "matched_controls_pooled"),
]

# Statistical tests
mwu_stat, mwu_p = stats.mannwhitneyu(scz_ages, ctl_ages, alternative="two-sided")
ks_stat, ks_p   = stats.ks_2samp(scz_ages, ctl_ages)

# Permutation: one control sampled per PGC3 rsid, 10k iterations,
# test diff in median + pct_lt_50k
log(f"  running 10,000-iteration permutation test...")
ctrl_by_pgc = ctrl_aged.merge(
    controls[["control_snp", "pgc3_rsid"]], left_on="rsid", right_on="control_snp",
    how="left",
).dropna(subset=["pgc3_rsid"])
log(f"    control_aged with pgc3 mapping: {len(ctrl_by_pgc):,}")

obs_median_diff = np.median(scz_ages) - np.median(ctl_ages)
obs_pct50_diff  = (scz_ages < 50_000).mean() - (ctl_ages < 50_000).mean()

perm_median_diffs = np.empty(10_000)
perm_pct50_diffs  = np.empty(10_000)
groups = ctrl_by_pgc.groupby("pgc3_rsid")["age_tgp"].apply(np.array)
log(f"    {len(groups):,} pgc3 rsids with control groups")

for i in range(10_000):
    rng = np.random.default_rng(i)
    sampled = np.array([
        rng.choice(arr) for arr in groups.values
    ])
    perm_median_diffs[i] = np.median(scz_ages) - np.median(sampled)
    perm_pct50_diffs[i]  = (scz_ages < 50_000).mean() - (sampled < 50_000).mean()

perm_p_median = 2 * min(
    (perm_median_diffs >= obs_median_diff).mean(),
    (perm_median_diffs <= obs_median_diff).mean(),
)
perm_p_pct50  = 2 * min(
    (perm_pct50_diffs >= obs_pct50_diff).mean(),
    (perm_pct50_diffs <= obs_pct50_diff).mean(),
)

pooled_tests = pd.DataFrame([{
    "test": "MannWhitneyU",
    "n_pgc": len(scz_ages),
    "n_ctl": len(ctl_ages),
    "statistic": mwu_stat,
    "p_value": mwu_p,
    "obs_median_diff_yr": float(obs_median_diff),
}, {
    "test": "KolmogorovSmirnov",
    "n_pgc": len(scz_ages),
    "n_ctl": len(ctl_ages),
    "statistic": ks_stat,
    "p_value": ks_p,
    "obs_median_diff_yr": float(obs_median_diff),
}, {
    "test": "Permutation_one_control_per_pgc_median",
    "n_pgc": len(scz_ages),
    "n_ctl": len(groups),
    "statistic": float(obs_median_diff),
    "p_value": float(perm_p_median),
    "obs_median_diff_yr": float(obs_median_diff),
}, {
    "test": "Permutation_one_control_per_pgc_pct_lt_50k",
    "n_pgc": len(scz_ages),
    "n_ctl": len(groups),
    "statistic": float(obs_pct50_diff * 100),
    "p_value": float(perm_p_pct50),
    "obs_median_diff_yr": float(obs_median_diff),
}])
pd.DataFrame(summary_rows).to_csv(OUT / "P14p_b_dist_summary.tsv", sep="\t", index=False)
pooled_tests.to_csv(OUT / "P14p_b_pooled_tests.tsv", sep="\t", index=False)
log(f"  saved → P14p_b_dist_summary.tsv + P14p_b_pooled_tests.tsv")
log("  Distribution summary:")
for row in summary_rows:
    log(f"    {row['group']:>30s}  n={row['n']:>9,}  median={row['median']:>9,.0f}  "
        f"pct<50k={row['pct_lt_50k']:.2f}%")
log(f"  MWU P = {mwu_p:.3e}; KS D = {ks_stat:.3f}, P = {ks_p:.3e}")
log(f"  Permutation P (median diff) = {perm_p_median:.4f}")
log(f"  Permutation P (pct<50k diff) = {perm_p_pct50:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Step 5: MAF-decile stratified comparison
# ─────────────────────────────────────────────────────────────────────
log("\n[5] MAF-decile-stratified comparison")

vm_maf = vm[["rsid", "maf"]].copy()
pgc3_aged_maf = pgc3_aged.merge(vm_maf, on="rsid", how="left").dropna(subset=["maf"])
ctrl_aged_maf = ctrl_aged.merge(
    controls[["control_snp", "control_maf"]].rename(
        columns={"control_snp": "rsid", "control_maf": "maf"}
    ),
    on="rsid", how="left",
).dropna(subset=["maf"]).drop_duplicates(subset=["rsid"])

# Decile by PGC3 MAF distribution (same bins applied to both)
maf_bins = np.quantile(pgc3_aged_maf["maf"], np.linspace(0, 1, 11))
maf_bins[0]  = 0
maf_bins[-1] = 1

pgc3_aged_maf["maf_decile"] = pd.cut(pgc3_aged_maf["maf"], maf_bins, labels=False)
ctrl_aged_maf["maf_decile"] = pd.cut(ctrl_aged_maf["maf"], maf_bins, labels=False)

strat_rows = []
for d in range(10):
    pgc = pgc3_aged_maf[pgc3_aged_maf["maf_decile"] == d]["age_tgp"].values
    ctl = ctrl_aged_maf[ctrl_aged_maf["maf_decile"] == d]["age_tgp"].values
    if len(pgc) < 20 or len(ctl) < 20:
        continue
    mwu_d, mwu_pd = stats.mannwhitneyu(pgc, ctl, alternative="two-sided")
    strat_rows.append({
        "maf_decile": d + 1,
        "maf_lo": float(maf_bins[d]),
        "maf_hi": float(maf_bins[d + 1]),
        "n_pgc": len(pgc),
        "n_ctl": len(ctl),
        "pgc_median_age": float(np.median(pgc)),
        "ctl_median_age": float(np.median(ctl)),
        "pgc_pct_lt_50k": float((pgc < 50_000).mean() * 100),
        "ctl_pct_lt_50k": float((ctl < 50_000).mean() * 100),
        "pgc_minus_ctl_median": float(np.median(pgc) - np.median(ctl)),
        "MWU_P": float(mwu_pd),
    })

maf_df = pd.DataFrame(strat_rows)
maf_df.to_csv(OUT / "P14p_c_maf_stratified_tests.tsv", sep="\t", index=False)
log(f"  saved → P14p_c_maf_stratified_tests.tsv ({len(maf_df)} deciles)")
for _, r in maf_df.iterrows():
    log(f"    MAF decile {int(r['maf_decile']):>2d} "
        f"({r['maf_lo']:.3f}–{r['maf_hi']:.3f})  "
        f"PGC med={r['pgc_median_age']:>7,.0f}  "
        f"CTL med={r['ctl_median_age']:>7,.0f}  "
        f"Δ={r['pgc_minus_ctl_median']:>+8,.0f}  "
        f"P={r['MWU_P']:.2e}")


# ─────────────────────────────────────────────────────────────────────
# Step 6: Per-cluster comparison
# ─────────────────────────────────────────────────────────────────────
log("\n[6] Per-cluster (C0/C1/C2) vs MAF-matched controls")

clu_only = clu[["rsid", "cluster"]].copy()
pgc3_aged_clu = pgc3_aged.merge(clu_only, on="rsid", how="left").merge(
    vm_maf, on="rsid", how="left"
).dropna(subset=["cluster", "maf"])
pgc3_aged_clu["cluster"] = pgc3_aged_clu["cluster"].astype(int)

cluster_rows = []
for c in [0, 1, 2]:
    sub = pgc3_aged_clu[pgc3_aged_clu["cluster"] == c]
    n_sub = len(sub)
    sub_median = float(np.median(sub["age_tgp"]))
    sub_pct50  = float((sub["age_tgp"] < 50_000).mean() * 100)

    # MAF-stratified control comparison: for each PGC3 variant in this cluster,
    # take its matched controls (~99 each) and compute aggregate age stats
    cluster_pgc_rsids = sub["rsid"].values
    cluster_controls = controls[controls["pgc3_rsid"].isin(cluster_pgc_rsids)]
    cluster_ctl_aged = ctrl_aged_maf[
        ctrl_aged_maf["rsid"].isin(cluster_controls["control_snp"].unique())
    ]
    ctl_median = float(np.median(cluster_ctl_aged["age_tgp"]))
    ctl_pct50  = float((cluster_ctl_aged["age_tgp"] < 50_000).mean() * 100)

    mwu_c, mwu_cp = stats.mannwhitneyu(
        sub["age_tgp"].values,
        cluster_ctl_aged["age_tgp"].values,
        alternative="two-sided",
    )

    cluster_rows.append({
        "cluster": f"C{c}",
        "n_pgc_cluster": n_sub,
        "pgc_median_age": sub_median,
        "pgc_pct_lt_50k": sub_pct50,
        "n_cluster_matched_controls": len(cluster_ctl_aged),
        "ctl_median_age": ctl_median,
        "ctl_pct_lt_50k": ctl_pct50,
        "pgc_minus_ctl_median": sub_median - ctl_median,
        "MWU_P": float(mwu_cp),
    })

clu_df = pd.DataFrame(cluster_rows)
clu_df.to_csv(OUT / "P14p_d_cluster_stratified.tsv", sep="\t", index=False)
log(f"  saved → P14p_d_cluster_stratified.tsv")
for _, r in clu_df.iterrows():
    log(f"    {r['cluster']}  PGC n={int(r['n_pgc_cluster']):>5,}  med={r['pgc_median_age']:>7,.0f}  "
        f"pct<50k={r['pgc_pct_lt_50k']:.2f}%  | "
        f"CTL n={int(r['n_cluster_matched_controls']):>7,}  med={r['ctl_median_age']:>7,.0f}  "
        f"pct<50k={r['ctl_pct_lt_50k']:.2f}%  | Δ={r['pgc_minus_ctl_median']:>+8,.0f}  P={r['MWU_P']:.2e}")


# ─────────────────────────────────────────────────────────────────────
# Step 7: Density plot
# ─────────────────────────────────────────────────────────────────────
log("\n[7] Density plot (PGC3 vs control)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Panel A: log10 age density
ax = axes[0]
log_scz = np.log10(scz_ages)
log_ctl = np.log10(ctl_ages)
ax.hist(log_ctl, bins=80, density=True, alpha=0.45, color="gray",
        label=f"Matched controls (n={len(ctl_ages):,})")
ax.hist(log_scz, bins=80, density=True, alpha=0.55, color="#2b5d8a",
        label=f"PGC3 credible-set (n={len(scz_ages):,})")
ax.axvline(np.log10(50_000), color="red", linestyle="--", alpha=0.5, lw=1)
ax.text(np.log10(50_000) + 0.04, ax.get_ylim()[1] * 0.9, "50 kyr",
        color="red", fontsize=8)
ax.set_xlabel("log10 Atlas TGP age (yr)")
ax.set_ylabel("density")
ax.set_title("a  Age distribution: PGC3 vs MAF+LD-matched controls")
ax.legend(fontsize=8)

# Panel B: cumulative
ax = axes[1]
sorted_scz = np.sort(scz_ages)
sorted_ctl = np.sort(ctl_ages)
ax.plot(sorted_ctl, np.linspace(0, 1, len(sorted_ctl)), color="gray", lw=2,
        label="Matched controls")
ax.plot(sorted_scz, np.linspace(0, 1, len(sorted_scz)), color="#2b5d8a", lw=2,
        label="PGC3 credible-set")
ax.axvline(50_000, color="red", linestyle="--", alpha=0.5, lw=1)
ax.set_xscale("log")
ax.set_xlabel("Atlas TGP age (yr, log scale)")
ax.set_ylabel("cumulative fraction")
ax.set_title("b  Empirical CDFs")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "P14p_e_density_pgc_vs_control.pdf", bbox_inches="tight")
plt.savefig(OUT / "P14p_e_density_pgc_vs_control.png", bbox_inches="tight", dpi=300)
log(f"  saved → P14p_e_density_pgc_vs_control.pdf|png")


# ─────────────────────────────────────────────────────────────────────
# Step 8: Narrative log
# ─────────────────────────────────────────────────────────────────────
narrative_path = OUT / "P14p_NARRATIVE.md"
narr = []
narr.append(f"# Phase 14p — Baseline Age-Distribution Comparison\n")
narr.append(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n")
narr.append(f"**Closes:** AUDIT_REPORT.md Issue 8 (genome-wide age baseline)\n")
narr.append(f"**Question:** Are PGC3 credible-set variants older/younger than MAF+L2-matched HapMap3 controls?\n\n")

narr.append(f"## Inputs\n")
narr.append(f"- PGC3 credible-set: `results/phase11/variant_master_v4.parquet` (n={len(vm):,})\n")
narr.append(f"- Matched controls: `data/processed/matched_controls.tsv.gz` "
            f"({controls['pgc3_rsid'].nunique():,} PGC3 → {controls['control_snp'].nunique():,} unique controls; "
            f"{len(controls):,} mappings; mean {len(controls)/controls['pgc3_rsid'].nunique():.1f} controls/PGC3)\n")
narr.append(f"- GEVA Atlas of Variant Age: `$ATLAS_DIR/atlas.chr*.csv.gz` "
            f"(AgeMedian_Mut, DataSource='TGP')\n")
narr.append(f"- Cluster assignments: `results/phase14b/P14b_v3_cluster_assignments.tsv.gz` "
            f"(C0/C1/C2 from Phase 14b GMM)\n\n")

narr.append(f"## Coverage\n")
narr.append(f"- PGC3 with Atlas TGP age: {n_pgc_aged:,}/{len(pgc3_positions):,} "
            f"({n_pgc_aged/len(pgc3_positions)*100:.1f}%)\n")
narr.append(f"- Control with Atlas TGP age: {n_ctl_aged:,}/{len(ctrl_positions):,} "
            f"({n_ctl_aged/len(ctrl_positions)*100:.1f}%)\n\n")

narr.append(f"## Pooled distribution\n\n")
narr.append(f"| Group | n | min | q01 | median | q99 | max | pct<50kyr |\n")
narr.append(f"|---|---|---|---|---|---|---|---|\n")
for row in summary_rows:
    narr.append(f"| {row['group']} | {row['n']:,} | {row['min']:,.0f} | "
                f"{row['q01']:,.0f} | {row['median']:,.0f} | {row['q99']:,.0f} | "
                f"{row['max']:,.0f} | {row['pct_lt_50k']:.2f}% |\n")
narr.append(f"\n**Mann-Whitney U:** P = {mwu_p:.3e}\n")
narr.append(f"**Kolmogorov-Smirnov:** D = {ks_stat:.4f}, P = {ks_p:.3e}\n")
narr.append(f"**Permutation (one control per PGC3, 10k iter, median diff):** P = {perm_p_median:.4f}\n")
narr.append(f"**Permutation (one control per PGC3, 10k iter, pct<50k diff):** P = {perm_p_pct50:.4f}\n\n")

narr.append(f"## MAF-decile stratified\n\n")
narr.append(f"| Decile | MAF range | n_PGC | n_CTL | PGC median | CTL median | Δ | MWU P |\n")
narr.append(f"|---|---|---|---|---|---|---|---|\n")
for _, r in maf_df.iterrows():
    narr.append(f"| {int(r['maf_decile'])} | {r['maf_lo']:.3f}–{r['maf_hi']:.3f} | "
                f"{int(r['n_pgc']):,} | {int(r['n_ctl']):,} | "
                f"{r['pgc_median_age']:,.0f} | {r['ctl_median_age']:,.0f} | "
                f"{r['pgc_minus_ctl_median']:+,.0f} | {r['MWU_P']:.2e} |\n")

narr.append(f"\n## Per-cluster vs MAF-matched controls\n\n")
narr.append(f"| Cluster | n_PGC | PGC med | PGC %<50k | n_CTL | CTL med | CTL %<50k | Δ med | MWU P |\n")
narr.append(f"|---|---|---|---|---|---|---|---|---|\n")
for _, r in clu_df.iterrows():
    narr.append(f"| {r['cluster']} | {int(r['n_pgc_cluster']):,} | "
                f"{r['pgc_median_age']:,.0f} | {r['pgc_pct_lt_50k']:.2f}% | "
                f"{int(r['n_cluster_matched_controls']):,} | "
                f"{r['ctl_median_age']:,.0f} | {r['ctl_pct_lt_50k']:.2f}% | "
                f"{r['pgc_minus_ctl_median']:+,.0f} | {r['MWU_P']:.2e} |\n")

narr.append(f"\n## Interpretation\n\n")
narr.append(f"- If pooled PGC3 median ≈ control median AND PGC3 %<50kyr ≈ control %<50kyr → ")
narr.append(f"the 'narrow temporal window' claim in v10 is **ascertainment-driven**, not SCZ-specific.\n")
narr.append(f"- If PGC3 is significantly younger after MAF stratification → SCZ-specific signal beyond ascertainment.\n")
narr.append(f"- Per-cluster shifts (C0 younger than its MAF-matched controls?) test cluster-level claim.\n\n")

narr.append(f"## Files\n")
narr.append(f"- `P14p_a_age_lookup.parquet`           — per-rsid Atlas TGP age\n")
narr.append(f"- `P14p_b_dist_summary.tsv`             — pooled distribution summary\n")
narr.append(f"- `P14p_b_pooled_tests.tsv`             — pooled MWU/KS/permutation tests\n")
narr.append(f"- `P14p_c_maf_stratified_tests.tsv`     — per-MAF-decile tests\n")
narr.append(f"- `P14p_d_cluster_stratified.tsv`       — per-cluster vs MAF-matched controls\n")
narr.append(f"- `P14p_e_density_pgc_vs_control.pdf`   — density + ECDF figure\n\n")

narr.append(f"## Run log\n\n```\n")
narr.extend(line + "\n" for line in LOG)
narr.append("```\n")

with open(narrative_path, "w") as f:
    f.writelines(narr)
log(f"\nWrote narrative: {narrative_path}")
log("\nPhase 14p complete.")
