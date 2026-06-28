#!/usr/bin/env python3
"""
Phase 14p6c — Evolutionary dynamics + multi-panel figure
==========================================================
Joint distributions of AFR-context age, AFR DAF, Akbari S, and v10 cluster:

Panel (a) AFR_TMRCA_yr × AFR_DAF (scatter, colored by v10 cluster) — frequency vs age
Panel (b) AFR_TMRCA distribution per AFR partition (violin)
Panel (c) AFR_TMRCA × |Akbari S| (scatter) — selection magnitude × AFR age
Panel (d) Mean χ² per AFR partition (heritability proxy bar chart)
Panel (e) AFR partition × v10 cluster percentage heatmap
Panel (f) AFR_DAF histogram per partition (does AFR_TMRCA capture different
          frequency strata than v10 GEVA-cluster?)

Output:
  results/phase14p6/P14p6c_evolutionary_dynamics.pdf|png
  results/phase14p6/P14p6c_NARRATIVE.md
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
OUT  = BASE / "results/phase14p6"

df = pd.read_parquet(OUT / "P14p6b_AFR_TMRCA_per_variant.parquet")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
df["log10_AFR_TMRCA"] = np.where(
    df["AFR_TMRCA_yr"].notna() & (df["AFR_TMRCA_yr"] > 0),
    np.log10(df["AFR_TMRCA_yr"]),
    np.nan,
)
df["abs_akbari_s"] = df["akbari_s"].abs() if "akbari_s" in df.columns else np.nan

partition_order = [
    "AFR_absent", "AFR_singleton_or_undef",
    "AFR_TMRCA_lt_50kyr", "AFR_TMRCA_50_200kyr",
    "AFR_TMRCA_200_500kyr", "AFR_TMRCA_ge_500kyr",
]
partition_palette = {
    "AFR_absent": "#e63946",
    "AFR_singleton_or_undef": "#fb8500",
    "AFR_TMRCA_lt_50kyr": "#fdc132",
    "AFR_TMRCA_50_200kyr": "#52b788",
    "AFR_TMRCA_200_500kyr": "#1a759f",
    "AFR_TMRCA_ge_500kyr": "#3a0ca3",
}
cluster_palette = {0: "#2b8a5d", 1: "#c89a3a", 2: "#a83333"}

fig = plt.figure(figsize=(15, 10), dpi=300)
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

# (a) AFR_TMRCA × AFR_DAF
ax = fig.add_subplot(gs[0, 0])
sub = df.dropna(subset=["AFR_TMRCA_yr", "AFR_DAF_in_tree", "cluster"]).copy()
sub = sub[sub["AFR_TMRCA_yr"] > 0]
sub["cluster"] = sub["cluster"].astype(int)
for c in [2, 1, 0]:
    s2 = sub[sub["cluster"] == c]
    ax.scatter(s2["log10_AFR_TMRCA"], s2["AFR_DAF_in_tree"],
               color=cluster_palette[c], alpha=0.35, s=8,
               label=f"C{c} (n={len(s2)})")
ax.set_xlabel("log10 AFR_TMRCA (yr)")
ax.set_ylabel("AFR DAF (tree-based)")
ax.set_title("a  AFR DAF × AFR_TMRCA by v10 cluster")
ax.legend(fontsize=8, markerscale=2)

# (b) AFR_TMRCA distribution per partition (violin)
ax = fig.add_subplot(gs[0, 1])
present = df[df["AFR_TMRCA_yr"].notna() & (df["AFR_TMRCA_yr"] > 0)]
data_list, labels = [], []
for p in partition_order:
    if p in ("AFR_absent",): continue
    sub = present[present["AFR_partition"] == p]["log10_AFR_TMRCA"].values
    if len(sub):
        data_list.append(sub); labels.append(p.replace("AFR_TMRCA_", ""))
if data_list:
    parts = ax.violinplot(data_list, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(partition_palette[partition_order[i+1] if i+1 < len(partition_order) else partition_order[-1]])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
ax.set_ylabel("log10 AFR_TMRCA (yr)")
ax.set_title("b  TMRCA per AFR partition")

# (c) AFR_TMRCA × |Akbari S|
ax = fig.add_subplot(gs[0, 2])
sub = df.dropna(subset=["AFR_TMRCA_yr", "abs_akbari_s", "cluster"]).copy()
sub = sub[sub["AFR_TMRCA_yr"] > 0]
sub["cluster"] = sub["cluster"].astype(int)
for c in [2, 1, 0]:
    s2 = sub[sub["cluster"] == c]
    ax.scatter(s2["log10_AFR_TMRCA"], s2["abs_akbari_s"],
               color=cluster_palette[c], alpha=0.35, s=8,
               label=f"C{c}")
ax.set_xlabel("log10 AFR_TMRCA (yr)")
ax.set_ylabel("|Akbari S|")
ax.set_title("c  Akbari S magnitude × AFR_TMRCA")
ax.legend(fontsize=8, markerscale=2)

# (d) mean χ² per partition
ax = fig.add_subplot(gs[1, 0])
h2 = pd.read_csv(OUT / "P14p6b_h2_proxy_per_partition_EUR.tsv", sep="\t")
h2 = h2[h2["partition"] != "ALL_credible_set (baseline)"]
h2["color"] = h2["partition"].map(partition_palette).fillna("#888")
ax.bar(range(len(h2)), h2["mean_chi2"], color=h2["color"])
ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
ax.set_xticks(range(len(h2)))
ax.set_xticklabels([p.replace("AFR_TMRCA_", "").replace("AFR_", "") for p in h2["partition"]],
                    rotation=20, fontsize=8)
ax.set_ylabel("mean χ² (h² proxy)")
ax.set_title("d  Heritability proxy per partition (EUR)")
for i, r in h2.reset_index(drop=True).iterrows():
    ax.text(i, r["mean_chi2"]+0.2, f"n={int(r['n_variants'])}",
            ha="center", fontsize=7)

# (e) AFR_partition × v10 cluster percentage heatmap
ax = fig.add_subplot(gs[1, 1])
ct_file = OUT / "P14p6b_AFR_partition_x_v10_cluster.tsv"
if ct_file.exists():
    ct = pd.read_csv(ct_file, sep="\t", index_col=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    im = ax.imshow(pct.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pct.columns)))
    ax.set_xticklabels(pct.columns, rotation=30, fontsize=7, ha="right")
    ax.set_yticks(range(len(pct.index))); ax.set_yticklabels(pct.index)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct.iloc[i,j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if pct.iloc[i,j] > 40 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% in cluster")
    ax.set_title("e  AFR partition × v10 cluster (%)")

# (f) AFR_DAF histogram per partition (only AFR-present partitions)
ax = fig.add_subplot(gs[1, 2])
for p in partition_order:
    if p == "AFR_absent": continue
    sub = df[df["AFR_partition"] == p]
    if len(sub) < 5: continue
    ax.hist(sub["AFR_DAF_in_tree"].dropna(), bins=30, density=True, alpha=0.5,
            color=partition_palette[p], label=p.replace("AFR_TMRCA_", ""))
ax.set_xlabel("AFR DAF (tree-based)")
ax.set_ylabel("density")
ax.set_title("f  AFR DAF distribution per partition")
ax.legend(fontsize=7)

plt.savefig(OUT / "P14p6c_evolutionary_dynamics.pdf", bbox_inches="tight")
plt.savefig(OUT / "P14p6c_evolutionary_dynamics.png", bbox_inches="tight", dpi=300)
print(f"Saved figures: {OUT / 'P14p6c_evolutionary_dynamics.pdf|png'}")

# Narrative
with open(OUT / "P14p6c_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p6c — Evolutionary dynamics figure\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write("## Panels\n\n")
    f.write("- (a) AFR_DAF × AFR_TMRCA scatter, v10 cluster color\n")
    f.write("- (b) AFR_TMRCA distribution per AFR partition\n")
    f.write("- (c) |Akbari S| × AFR_TMRCA scatter by cluster\n")
    f.write("- (d) mean χ² per AFR partition (h² proxy)\n")
    f.write("- (e) AFR partition × v10 cluster cross-tab (%)\n")
    f.write("- (f) AFR DAF distribution per AFR partition\n")

print("Phase 14p6c complete.")
