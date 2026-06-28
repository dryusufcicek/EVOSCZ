#!/usr/bin/env python3
"""
Phase 14p4c — Multi-panel figure for H2: cluster × Akbari selection
====================================================================
Panel (a): |Akbari S| density per cluster (3 colored densities)
Panel (b): %-in-Akbari-452 per cluster (barchart with permutation null)
Panel (c): SCZ-risk vs Akbari-selected concordance (per-cluster)
Panel (d): Window overlap (Fisher OR) at 5/50/250 kb per cluster
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(_ROOT)
OUT  = BASE / "results/phase14p4"

vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
df = vm.merge(clu, on="rsid", how="inner").dropna(subset=["akbari_s"])
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype(int)
df["abs_S"] = df["akbari_s"].abs()

dist = pd.read_csv(OUT / "P14p4a_cluster_S_distribution.tsv", sep="\t")
direc = pd.read_csv(OUT / "P14p4a_direction_of_effect.tsv", sep="\t")
fish = pd.read_csv(OUT / "P14p4a_fisher_overlap.tsv", sep="\t")
win  = pd.read_csv(OUT / "P14p4b_window_overlap_per_cluster.tsv", sep="\t") if (OUT / "P14p4b_window_overlap_per_cluster.tsv").exists() else None

palette = {0: "#2b8a5d", 1: "#c89a3a", 2: "#a83333"}
labels = {0: "C0 Young", 1: "C1 Mid", 2: "C2 Old"}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

# (a) |S| density per cluster (log scale on x for visibility)
ax = axes[0, 0]
for c in [2, 1, 0]:
    vals = df[df["cluster"] == c]["abs_S"].values
    vals = vals[vals > 1e-6]
    ax.hist(np.log10(vals), bins=50, density=True, alpha=0.55,
            color=palette[c], label=labels[c])
ax.set_xlabel("log10 |Akbari s|")
ax.set_ylabel("density")
ax.set_title("a  |Akbari S| density by cluster")
ax.legend(fontsize=8)

# (b) %-in-Akbari-452 per cluster
ax = axes[0, 1]
cd = dist.copy()
cd["pct_akbari"] = cd["pct_in_akbari452"]
ax.bar([labels[int(c[1])] for c in cd["cluster"]],
       cd["pct_akbari"],
       color=[palette[int(c[1])] for c in cd["cluster"]])
for i, r in cd.iterrows():
    ax.text(i, r["pct_akbari"] + 0.1,
            f"{int(r['n_in_akbari452'])}/{int(r['n'])}",
            ha="center", fontsize=8)
ax.set_ylabel("% credible-set ∩ Akbari 452")
ax.set_title("b  Cluster ∩ Akbari 452 overlap")

# (c) Direction-of-effect concordance per cluster
ax = axes[1, 0]
dd = direc[direc["cluster"] != "all_overlap"]
ax.bar([labels[int(c[1])] for c in dd["cluster"]],
       dd["pct_concordance"],
       color=[palette[int(c[1])] for c in dd["cluster"]])
ax.axhline(50, color="gray", linestyle="--", lw=1, alpha=0.7,
           label="null = 50%")
for i, r in dd.iterrows():
    ax.text(i, r["pct_concordance"] + 1,
            f"{int(r['n_SCZrisk_eq_selectedFOR'])}/{int(r['n_overlap_valid'])}",
            ha="center", fontsize=8)
ax.set_ylabel("% SCZ-risk = Akbari-selected-FOR")
ax.set_title("c  Direction of effect (pleiotropy direction)")
ax.legend(fontsize=8)
ax.set_ylim(0, 100)

# (d) Window overlap Fisher OR
ax = axes[1, 1]
if win is not None:
    for c in [0, 1, 2]:
        sub = win[win["cluster"] == f"C{c}"]
        ax.plot(sub["window_kb"], sub["fisher_OR"], marker="o",
                color=palette[c], lw=2, label=labels[c])
    ax.axhline(1.0, color="gray", linestyle="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Window around Akbari (kb)")
    ax.set_ylabel("Fisher OR vs other clusters")
    ax.set_title("d  Window overlap Fisher OR")
    ax.legend(fontsize=8)

plt.tight_layout()
out_p = OUT / "P14p4c_fig_akbari_cluster"
plt.savefig(str(out_p) + ".pdf", bbox_inches="tight")
plt.savefig(str(out_p) + ".png", bbox_inches="tight", dpi=300)
print(f"Saved: {out_p}.pdf|png")
