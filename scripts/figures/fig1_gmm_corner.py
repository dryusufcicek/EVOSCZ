#!/usr/bin/env python3
"""Fig 1 (v25_3 reframe) — GMM clustering of fine-mapped PGC3 SCZ credible-set
variants. Four self-contained panels, each with full x- and y-axes:
  a  log10 allele age (yr)        vs  brain-blood specificity   (scatter)
  b  log10 allele age (yr)        vs  |iHS|                      (scatter)
  c  brain-blood specificity      vs  |iHS|                      (scatter)
  d  marginal density of log10 allele age (per cluster), cluster medians marked
Clusters C0 Young / C1 Mid / C2 Old.

DATA-CORRECTNESS: age_median_yr holds GENERATIONS; converted to years (x28.1)
before log10, so the (yr) axes are truthful (medians ~113/358/508 kyr).
White markers = empirical cluster means. Zero fabrication.
"""
from __future__ import annotations
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

GEN_TIME = 28.1
SRC = (BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz")
OUT = Path(os.environ.get("EVOSCZ_FIG_OUT") or (BASE / "figures"))
AGE = r"log$_{10}$ allele age (yr)"
BS = "Brain–blood specificity"
IHS = "|iHS|"
COL = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
NAME = {0: "Young", 1: "Mid", 2: "Old"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})

def main():
    df = pd.read_csv(SRC, sep="\t")
    df = df[df["cluster"].notna()].copy(); df["cluster"] = df["cluster"].astype(int)
    df["log10_age"] = np.log10(pd.to_numeric(df["age_median_yr"], errors="coerce") * GEN_TIME)
    df["brain_spec"] = pd.to_numeric(df["brain_spec"], errors="coerce")
    df["abs_ihs"] = pd.to_numeric(df["abs_ihs"], errors="coerce")
    df = df.dropna(subset=["log10_age", "brain_spec", "abs_ihs"])
    counts = df.groupby("cluster").size().to_dict()
    med = (df.groupby("cluster")["log10_age"].median().apply(lambda v: 10**v)).to_dict()
    print("n:", counts, "total", len(df), "| median age yr:", {k: int(round(v)) for k, v in med.items()})

    rng = {f: (df[f].min(), df[f].max()) for f in ("log10_age", "brain_spec", "abs_ihs")}
    def lim(f, p=0.05):
        lo, hi = rng[f]; d = (hi - lo) * p; return lo - d, hi + d

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), dpi=300)
    (a, b), (c, d) = axes

    def scat(ax, xf, yf, xl, yl):
        xlo, xhi = lim(xf); ylo, yhi = lim(yf)
        xx, yy = np.mgrid[xlo:xhi:90j, ylo:yhi:90j]
        pos = np.vstack([xx.ravel(), yy.ravel()])
        # light scatter for texture, then per-cluster 2-D density contours so the
        # cluster cores + separation read instead of a saturated point mush
        for k in (2, 0, 1):
            s = df[df["cluster"] == k]
            ax.scatter(s[xf], s[yf], s=2.2, alpha=0.09, c=COL[k], edgecolors="none",
                       rasterized=True, zorder=1)
        for k in (0, 1, 2):
            s = df[df["cluster"] == k]
            zz = gaussian_kde(np.vstack([s[xf].values, s[yf].values]))(pos).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=[zz.max() * l for l in (0.3, 0.6)],
                       colors=[COL[k]], linewidths=0.9, zorder=3)
        cen = df.groupby("cluster")[[xf, yf]].mean()
        for k, row in cen.iterrows():
            ax.scatter(row[xf], row[yf], s=46, facecolors="white", edgecolors=COL[k],
                       linewidths=1.4, zorder=6)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_xlabel(xl); ax.set_ylabel(yl)

    scat(a, "log10_age", "brain_spec", AGE, BS)
    scat(b, "log10_age", "abs_ihs", AGE, IHS)
    scat(c, "brain_spec", "abs_ihs", BS, IHS)

    # d: marginal density of log10 age, per cluster, with medians
    lo, hi = lim("log10_age", 0.02); grid = np.linspace(lo, hi, 256); ymax = 0
    for k in (0, 1, 2):
        y = gaussian_kde(df.loc[df["cluster"] == k, "log10_age"].values)(grid)
        d.fill_between(grid, y, color=COL[k], alpha=0.30, lw=0); d.plot(grid, y, color=COL[k], lw=1.0)
        ymax = max(ymax, y.max())
    d.set_xlim(lo, hi); d.set_ylim(0, ymax * 1.5)
    d.set_xlabel(AGE); d.set_ylabel("Density")
    # dashed median lines on the curves; values in a staggered corner block
    # (kept above the peaks so nothing collides with the densities)
    for k in (0, 1, 2):
        d.axvline(np.log10(med[k]), color=COL[k], lw=0.8, ls=(0, (3, 2)), alpha=0.9, zorder=2)
    for i, k in enumerate((0, 1, 2)):
        d.text(0.035, 0.96 - i * 0.075, f"C{k} median  {med[k]/1000:.0f} kyr",
               transform=d.transAxes, ha="left", va="top", fontsize=5.9, color=COL[k], zorder=8)
    d.text(0.035, 0.96 - 3 * 0.075, "k = 3 (ΔBIC = −3,554)", transform=d.transAxes,
           ha="left", va="top", fontsize=5.9, color="0.35", zorder=8)

    for ax, lab in zip([a, b, c, d], "abcd"):
        ax.text(-0.20, 1.04, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    handles = [Patch(facecolor=COL[k], edgecolor="black", linewidth=0.4,
                     label=f"C{k} {NAME[k]} (n = {counts[k]:,})") for k in (0, 1, 2)]
    handles.append(Line2D([], [], marker="o", ls="None", mfc="white", mec="black",
                          mew=0.9, ms=6, label="Cluster mean"))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.8, handletextpad=0.5)

    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=2.0, h_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "Fig1_gmm_cluster_projections"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", base)

if __name__ == "__main__":
    main()
