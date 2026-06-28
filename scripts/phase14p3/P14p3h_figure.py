#!/usr/bin/env python3
"""
Phase 14p3h — Multi-panel figure for H4 pure-AFR DAF × cluster identity
=========================================================================
Panel (a): pure-AFR DAF density per cluster (3 colored densities + rug)
Panel (b): empirical CDF (ECDF) per cluster — visualizes KS D directly
Panel (c): per-AFR-subpop median DAF per cluster (heatmap-style barchart)
Panel (d): EUR-MAF decile stratified C0 vs C2 KS D + 95% bootstrap CI
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(_ROOT)
OUT  = BASE / "results/phase14p3"

# Inputs
df  = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = df.merge(clu, on="rsid").dropna(subset=["AF_AFR_derived","cluster"])
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype(int)

pair   = pd.read_csv(OUT / "P14p3c_pairwise_KS_MWU.tsv", sep="\t")
strat  = pd.read_csv(OUT / "P14p3d_maf_stratified_KS.tsv", sep="\t") if (OUT / "P14p3d_maf_stratified_KS.tsv").exists() else None
subres = pd.read_csv(OUT / "P14p3f_subpop_replication.tsv", sep="\t") if (OUT / "P14p3f_subpop_replication.tsv").exists() else None

palette = {0: "#2b8a5d", 1: "#c89a3a", 2: "#a83333"}
labels  = {0: "C0 Young (n=%d)" % (df["cluster"]==0).sum(),
           1: "C1 Mid (n=%d)"   % (df["cluster"]==1).sum(),
           2: "C2 Old (n=%d)"   % (df["cluster"]==2).sum()}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

# (a) Density
ax = axes[0,0]
for c in [2, 1, 0]:
    vals = df[df["cluster"]==c]["AF_AFR_derived"].values
    if len(vals) < 5: continue
    ax.hist(vals, bins=40, density=True, alpha=0.55,
            color=palette[c], label=labels[c])
ax.set_xlabel("Pure-AFR DAF (1000G ESN+GWD+LWK+MSL+YRI)")
ax.set_ylabel("density")
ax.set_title("a  Pure-AFR DAF density by cluster")
ax.legend(fontsize=8, loc="upper center")

# (b) ECDF
ax = axes[0,1]
for c in [0, 1, 2]:
    vals = np.sort(df[df["cluster"]==c]["AF_AFR_derived"].values)
    if len(vals) < 5: continue
    ax.plot(vals, np.linspace(0, 1, len(vals)), color=palette[c],
            lw=2, label=labels[c])
ax.set_xlabel("Pure-AFR DAF")
ax.set_ylabel("Cumulative fraction")
ax.set_title("b  ECDFs (KS = max vertical gap)")
# annotate KS D
if "C0_vs_C2" in pair["contrast"].values:
    row = pair[pair["contrast"]=="C0_vs_C2"].iloc[0]
    ax.annotate(f"KS D(C0,C2) = {row['KS_D']:.3f}\nP = {row['KS_P_asym']:.1e}",
                xy=(0.55, 0.20), fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.85))
ax.legend(fontsize=8, loc="lower right")

# (c) Per-subpop median DAF per cluster
ax = axes[1,0]
if subres is not None and len(subres) > 0:
    # Long → wide: rows=subpop, cols=cluster median for cluster A in C0_vs_C2 (gives median_a for C0 in each subpop)
    c02 = subres[subres["contrast"]=="C0_vs_C2"].copy()
    c12 = subres[subres["contrast"]=="C1_vs_C2"].copy()
    medians = pd.DataFrame({
        "subpop": c02["subpop"].values,
        "C0_median": c02["median_a"].values,
        "C1_median": c12["median_a"].values,
        "C2_median": c02["median_b"].values,
    })
    x = np.arange(len(medians))
    w = 0.27
    ax.bar(x - w, medians["C0_median"], w, color=palette[0], label="C0")
    ax.bar(x,     medians["C1_median"], w, color=palette[1], label="C1")
    ax.bar(x + w, medians["C2_median"], w, color=palette[2], label="C2")
    ax.set_xticks(x); ax.set_xticklabels(medians["subpop"], rotation=0, fontsize=9)
    ax.set_ylabel("median DAF")
    ax.set_title("c  Per-AFR-subpop median DAF per cluster")
    ax.legend(fontsize=8, loc="upper left")

# (d) Stratified KS D per MAF decile
ax = axes[1,1]
if strat is not None:
    c02 = strat[strat["contrast"]=="C0_vs_C2"]
    ax.plot(c02["maf_decile"], c02["KS_D"], marker="o", color="#2b5d8a",
            lw=2, label="C0 vs C2 KS D")
    ax.axhline(0.20, color="gray", linestyle="--", lw=1, alpha=0.7,
               label="pre-reg threshold 0.20")
    ax.set_xlabel("EUR-MAF decile")
    ax.set_ylabel("Within-decile KS D")
    ax.set_title("d  EUR-MAF stratified KS D (consistency check)")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)

plt.tight_layout()
fig_path = OUT / "P14p3h_fig_pure_AFR_DAF_by_cluster"
plt.savefig(str(fig_path) + ".pdf", bbox_inches="tight")
plt.savefig(str(fig_path) + ".png", bbox_inches="tight", dpi=300)
print(f"Saved: {fig_path}.pdf|png")
