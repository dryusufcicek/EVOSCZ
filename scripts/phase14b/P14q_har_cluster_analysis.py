#!/usr/bin/env python3
"""
Phase 14q: HAR (Human Accelerated Regions) × cluster cross-tabulation.

Tests Crow's hypothesis at variant level: are C0 (Young + brain-regulatory)
variants closer to HARs than C1 (Mid) and C2 (Old) variants?

HARs (Cui 2025 *Nature*) are sequences fast-evolving in the human lineage
relative to chimpanzee, primarily acting as neuronal enhancers. If C0 captures
"recent human-specific evolution of brain regulators", C0 variants should be
proximally enriched near HARs.

Tests run:
  1. Distance distribution comparison (C0 vs C1 vs C2): Mann-Whitney U
  2. Proximity cutoff enrichment (within X kb): Fisher's exact
  3. Direct overlap (var inside HAR interval): chi-square

Data sources:
  - Cluster assignments: results/phase14b/P14b_v3_cluster_assignments.tsv.gz
  - HAR proximity (precomputed): results/module_b/B3_har_proximity.tsv
  - Variant master (har_overlap binary): results/phase11/variant_master_v3.parquet
"""
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase14b"

print(f"Phase 14q: HAR × cluster — {datetime.now().strftime('%H:%M:%S')}")
print("=" * 72)

# Load cluster assignments
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")
clu = clu[clu["cluster"].notna()].copy()
clu["cluster"] = clu["cluster"].astype(int)
print(f"\nCluster assignments: {len(clu):,}")
print(clu["cluster"].value_counts().sort_index())

# Load HAR proximity
har = pd.read_csv(BASE / "results/module_b/B3_har_proximity.tsv", sep="\t")
print(f"\nHAR proximity rows: {len(har):,}")
print(f"Columns: {list(har.columns)}")
print(f"har_distance summary (bp):")
print(har["har_distance"].describe())

# Load variant master for har_overlap binary + chr/pos
m = pd.read_parquet(BASE / "results/phase11/variant_master_v3.parquet")
m = m[["rsid", "chr", "pos", "har_overlap"]].copy()
print(f"\nMaster rows: {len(m):,}")
print(f"Direct HAR overlap (har_overlap==1): {(m['har_overlap']==1).sum():,}")

# Merge cluster + HAR proximity + master
df = clu.merge(har[["rsid", "har_distance", "nearest_har"]], on="rsid", how="left")
df = df.merge(m[["rsid", "har_overlap"]], on="rsid", how="left")
print(f"\nMerged: {len(df):,} variants with cluster + HAR data")

n_have_distance = df["har_distance"].notna().sum()
print(f"Have HAR distance: {n_have_distance:,}")

# ============================================================
# Test 1: Distance distribution by cluster (Mann-Whitney)
# ============================================================
print("\n" + "=" * 72)
print("[1] Distance to nearest HAR — distribution by cluster")
print("=" * 72)

for cluster_id, name in [(0, "Young"), (1, "Mid"), (2, "Old")]:
    sub = df[df["cluster"] == cluster_id]
    dist = sub["har_distance"].dropna()
    if len(dist) > 0:
        print(f"  C{cluster_id} ({name}, n={len(dist):,}):")
        print(f"    median = {dist.median():,.0f} bp")
        print(f"    mean   = {dist.mean():,.0f} bp")
        print(f"    25th%  = {dist.quantile(0.25):,.0f} bp")
        print(f"    75th%  = {dist.quantile(0.75):,.0f} bp")

# Pairwise Mann-Whitney
print("\n  Pairwise Mann-Whitney U tests (one-tailed: smaller distance = closer to HAR):")
c0 = df.loc[df["cluster"] == 0, "har_distance"].dropna()
c1 = df.loc[df["cluster"] == 1, "har_distance"].dropna()
c2 = df.loc[df["cluster"] == 2, "har_distance"].dropna()

for (a, na, b, nb) in [(c0, "C0 (Young)", c1, "C1 (Mid)"),
                         (c0, "C0 (Young)", c2, "C2 (Old)"),
                         (c1, "C1 (Mid)", c2, "C2 (Old)")]:
    u_stat, p = stats.mannwhitneyu(a, b, alternative="less")
    print(f"    {na} closer than {nb}: U={u_stat:.0f}, P={p:.3e}")

# ============================================================
# Test 2: Proximity cutoff enrichment (Fisher's exact)
# ============================================================
print("\n" + "=" * 72)
print("[2] Within-X-kb HAR enrichment by cluster (Fisher's exact)")
print("=" * 72)

for cutoff_kb in [10, 50, 100, 250, 500]:
    cutoff_bp = cutoff_kb * 1000
    print(f"\n  Cutoff: within {cutoff_kb} kb of nearest HAR")
    for cluster_id, name in [(0, "Young"), (1, "Mid"), (2, "Old")]:
        sub = df[df["cluster"] == cluster_id]
        n_total = sub["har_distance"].notna().sum()
        n_within = (sub["har_distance"] <= cutoff_bp).sum()
        pct = 100 * n_within / max(n_total, 1)
        print(f"    C{cluster_id} ({name}): {n_within:,}/{n_total:,} ({pct:.2f}%)")

    # Fisher: C0 vs (C1+C2 combined)
    c0_yes = ((df["cluster"] == 0) & (df["har_distance"] <= cutoff_bp)).sum()
    c0_no  = ((df["cluster"] == 0) & (df["har_distance"] >  cutoff_bp)).sum()
    rest_yes = ((df["cluster"] != 0) & (df["har_distance"] <= cutoff_bp)).sum()
    rest_no  = ((df["cluster"] != 0) & (df["har_distance"] >  cutoff_bp)).sum()

    table = [[c0_yes, c0_no], [rest_yes, rest_no]]
    odds, p = stats.fisher_exact(table, alternative="greater")
    print(f"    Fisher C0 vs (C1+C2): OR={odds:.3f}, P={p:.3e}")

# ============================================================
# Test 3: Permutation test (random cluster relabeling preserves total Ns)
# ============================================================
print("\n" + "=" * 72)
print("[3] Permutation test: median HAR distance for C0 vs random cluster of same size")
print("=" * 72)

obs_median = df.loc[df["cluster"] == 0, "har_distance"].median()
n_c0 = (df["cluster"] == 0).sum()
all_dist = df["har_distance"].dropna()

n_perm = 10000
rng = np.random.RandomState(42)
perm_medians = np.zeros(n_perm)
for i in range(n_perm):
    sample = rng.choice(all_dist.values, size=int(n_c0), replace=False)
    perm_medians[i] = np.median(sample)

p_perm = (perm_medians <= obs_median).mean()
print(f"  Observed C0 median distance: {obs_median:,.0f} bp")
print(f"  Permutation null median (10k draws, mean): {perm_medians.mean():,.0f} bp")
print(f"  Permutation p (one-tailed, C0 closer): P = {p_perm:.4f}")
print(f"  C0 percentile in null: {(perm_medians < obs_median).mean()*100:.2f}%")

# ============================================================
# Save summary table
# ============================================================
summary_rows = []
for cluster_id, name in [(0, "Young"), (1, "Mid"), (2, "Old")]:
    sub = df[df["cluster"] == cluster_id]
    dist = sub["har_distance"].dropna()
    summary_rows.append({
        "cluster": f"C{cluster_id}_{name}",
        "n_variants": int(len(sub)),
        "n_with_har_distance": int(len(dist)),
        "median_distance_bp": float(dist.median()),
        "mean_distance_bp": float(dist.mean()),
        "frac_within_100kb": float((dist <= 100000).mean()),
        "frac_within_500kb": float((dist <= 500000).mean()),
        "n_direct_overlap": int(sub["har_overlap"].fillna(0).sum()),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "P14q_har_cluster_summary.tsv", sep="\t", index=False, float_format="%.4g")
print(f"\nSummary table saved: {OUT}/P14q_har_cluster_summary.tsv")
print(summary.to_string(index=False))

print(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")
