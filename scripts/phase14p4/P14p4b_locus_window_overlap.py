#!/usr/bin/env python3
"""
Phase 14p4b — Locus-window overlap (Akbari ± Xkb) per cluster
==============================================================
Beyond exact rsid/pos overlap (Phase 14p4a), test cluster enrichment
for **locus-level proximity** to Akbari 452. A credible-set variant
counts as Akbari-overlapping if any Akbari 452 variant is within
window W (kb) on the same chromosome.

Three windows tested: ±5 kb (strict locus), ±50 kb (gene-level),
±250 kb (LD-block-level).

This captures the case where the SCZ credible-set variant is not the
Akbari index but is in the same selection-relevant locus.

Output:
  results/phase14p4/P14p4b_window_overlap_per_cluster.tsv
  results/phase14p4/P14p4b_NARRATIVE.md
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
AUX  = Path((_SCRATCH + "/v11_data/aux"))
OUT  = BASE / "results/phase14p4"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*70); log("Phase 14p4b — Window overlap per cluster"); log("="*70)

vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = vm.merge(clu, on="rsid", how="inner")
df["chr"] = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
df = df.dropna(subset=["chr","pos","cluster"]).copy()
df["cluster"] = df["cluster"].astype(int)
log(f"  PGC3 ∩ cluster: {len(df):,}")

ak = pd.read_csv(AUX / "akbari_347_loci.tsv", sep="\t")
ak["chr"] = pd.to_numeric(ak["chr"], errors="coerce").astype("Int64")
ak["pos"] = pd.to_numeric(ak["pos"], errors="coerce").astype("Int64")
ak = ak.dropna(subset=["chr","pos"]).copy()
log(f"  Akbari 452: {len(ak):,}")

# Build per-chr sorted Akbari positions for fast window search
ak_by_chr = {c: np.sort(g["pos"].astype(int).values)
             for c, g in ak.groupby("chr")}

def nearest_akbari_distance(chrom, pos):
    if chrom not in ak_by_chr: return np.inf
    arr = ak_by_chr[chrom]
    idx = np.searchsorted(arr, pos)
    candidates = []
    if idx > 0: candidates.append(arr[idx-1])
    if idx < len(arr): candidates.append(arr[idx])
    if not candidates: return np.inf
    return min(abs(int(pos) - int(x)) for x in candidates)

log("\n[1] Compute nearest-Akbari distance per credible-set variant")
df["dist_to_akbari_kb"] = [
    nearest_akbari_distance(int(c), int(p))/1000
    for c, p in zip(df["chr"].astype(int), df["pos"].astype(int))
]
log(f"  Distance distribution (kb):")
log(f"  median={df['dist_to_akbari_kb'].median():.1f}, "
    f"q25={df['dist_to_akbari_kb'].quantile(0.25):.1f}, "
    f"q75={df['dist_to_akbari_kb'].quantile(0.75):.1f}")

rows = []
for win_kb in [5, 50, 250]:
    df[f"in_akbari_{win_kb}kb"] = (df["dist_to_akbari_kb"] <= win_kb).astype(int)
    for c in [0, 1, 2]:
        in_c = (df["cluster"] == c)
        not_c = (df["cluster"].isin([x for x in [0,1,2] if x != c]))
        a = int(((in_c)  & (df[f"in_akbari_{win_kb}kb"] == 1)).sum())
        b = int(((in_c)  & (df[f"in_akbari_{win_kb}kb"] == 0)).sum())
        cc = int(((not_c) & (df[f"in_akbari_{win_kb}kb"] == 1)).sum())
        d = int(((not_c) & (df[f"in_akbari_{win_kb}kb"] == 0)).sum())
        odds, p = stats.fisher_exact([[a, b], [cc, d]])
        rows.append({
            "window_kb": win_kb,
            "cluster": f"C{c}",
            "n_cluster": int(in_c.sum()),
            "n_window": a,
            "pct_window": float(a / in_c.sum() * 100) if in_c.sum() else np.nan,
            "fisher_OR": float(odds),
            "fisher_P": float(p),
        })
out = pd.DataFrame(rows)
out.to_csv(OUT / "P14p4b_window_overlap_per_cluster.tsv", sep="\t", index=False)
log("\n" + out.to_string(index=False))

with open(OUT / "P14p4b_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p4b — Window overlap per cluster\n\n## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p4b complete.")
