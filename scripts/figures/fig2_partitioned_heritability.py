#!/usr/bin/env python3
"""Fig 2 (v25_3 reframe) — Cluster-level partitioned heritability (S-LDSC,
baseline-LD v2.2) as a grouped FOREST / dot-and-whisker plot (publication-quality
house style). Twelve rows in four GWAS sub-groups (PGC3 EUR SCZ; EUR SCZ
LR-LD-masked; PGC3 EAS SCZ; Alzheimer's disease); within each sub-group the three
variant clusters (C0 Young / C1 Mid / C2 Old). One shared fold-enrichment x-axis
makes the cluster- and disorder-specificity readable at a glance: C0 lies far to
the right (filled, significant) in every schizophrenia panel but collapses to the
null in Alzheimer's disease.

Encoding: point = fold enrichment; horizontal whisker = ±1 s.e.; FILLED marker =
significant conditional-coefficient (one-tailed P < 0.05), OPEN marker = not
significant; marker colour = cluster; black asterisks = significance tier. Values
are read off the axis (no in-plot value labels). The collapsed EAS C1 estimate
(s.e. ~96, ~56 SNPs) is drawn as an open grey point whose uncertainty exceeds the
panel (dashed arrowed line); explained in the caption.

Data = raw S-LDSC outputs in results/phase18_v22/*.results (primary source,
identical to Main Table 2). Zero fabrication.
"""
from __future__ import annotations
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

RES = (BASE / "results/phase18_v22")
OUT = Path(os.environ.get("EVOSCZ_FIG_OUT") or (BASE / "figures"))
# order + (a)-(d) labels match the manuscript text references
# (Fig 2a EUR, 2b AD, 2c Price LR-LD, 2d EAS)
FACETS = [
    ("PGC3_SCZ_EUR_v22.results", "a  PGC3 EUR schizophrenia"),
    ("Wightman_AD_v22.results", "b  Alzheimer's disease"),
    ("SCZ_EUR_LRLD_v22.results", "c  EUR schizophrenia (LR-LD masked)"),
    ("PGC3_SCZ_EAS_v22.results", "d  PGC3 EAS schizophrenia"),
]
COL = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
NAME = {0: "Young", 1: "Mid", 2: "Old"}
COLLAPSE_SE = 50.0  # s.e. above this = uninformative (EAS C1)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.3, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})


def load(fn):
    d = pd.read_csv(RES / fn, sep="\t")
    cl = d[d["Category"].str.match(r"C\dL2_1")].copy()
    cl["k"] = cl["Category"].str.extract(r"C(\d)").astype(int)
    return cl.set_index("k").sort_index()


def stars(z):
    p = stats.norm.sf(z)
    return "***" if p <= 1e-3 else "**" if p <= 1e-2 else "*" if p <= 5e-2 else ""


def main():
    # ---- gather verified values --------------------------------------------
    data = {}  # (group_index, k) -> (enrichment, s.e., z)
    for gi, (fn, _) in enumerate(FACETS):
        d = load(fn)
        for k in (0, 1, 2):
            data[(gi, k)] = (float(d.loc[k, "Enrichment"]),
                             float(d.loc[k, "Enrichment_std_error"]),
                             float(d.loc[k, "Coefficient_z-score"]))

    # ---- vertical layout (top group highest y) -----------------------------
    yhead, ypos = {}, {}
    y = 0.0
    for gi in range(4):
        yhead[gi] = y
        y -= 0.95
        for k in (0, 1, 2):
            ypos[(gi, k)] = y
            y -= 1.0
        y -= 0.85
    ybot, ytop = y + 0.45, yhead[0] + 0.95

    # ---- shared x-range (exclude collapsed estimate) -----------------------
    fin = [(en, se) for (en, se, z) in data.values() if se < COLLAPSE_SE]
    xR = max(en + se for en, se in fin) + 8.0
    xL = min(en - se for en, se in fin) - 4.0

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    # faint alternating group bands (aid binding far-left label to its point)
    for gi in (1, 3):
        ax.axhspan(ypos[(gi, 2)] - 0.55, yhead[gi] + 0.55,
                   color="0.945", lw=0, zorder=0)
    ax.axvline(1.0, ls="--", color="0.55", lw=0.8, zorder=1)  # no enrichment

    yt = ax.get_yaxis_transform()  # x = axes fraction, y = data
    for gi, (fn, header) in enumerate(FACETS):
        ax.text(-0.165, yhead[gi], header, transform=yt, fontweight="bold",
                fontsize=7, va="center", ha="left", clip_on=False, zorder=6)
        for k in (0, 1, 2):
            en, se, z = data[(gi, k)]
            yy = ypos[(gi, k)]
            if se >= COLLAPSE_SE:  # uninformative: open grey point, off-scale bar
                ax.plot([xL + 2, xR - 2], [yy, yy], color="0.65", lw=0.7,
                        ls=(0, (2, 2)), zorder=1)
                ax.plot(xL + 2, yy, marker="<", color="0.65", ms=4, zorder=2)
                ax.plot(xR - 2, yy, marker=">", color="0.65", ms=4, zorder=2)
                ax.plot(en, yy, marker="o", mfc="white", mec="0.5", mew=1.0,
                        ms=6.5, zorder=4)
            else:
                sig = stats.norm.sf(z) < 0.05
                ax.errorbar(en, yy, xerr=se, fmt="none", ecolor=COL[k],
                            elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=2)
                ax.plot(en, yy, marker="o", ms=6.8, mew=1.2, mec=COL[k],
                        mfc=COL[k] if sig else "white", zorder=4)
                s = stars(z)
                if s:
                    ax.text(en + se + 1.6, yy, s, va="center", ha="left",
                            fontsize=8, zorder=5)

    # ---- left cluster labels (coloured) ------------------------------------
    rows = [(gi, k) for gi in range(4) for k in (0, 1, 2)]
    ax.set_yticks([ypos[r] for r in rows])
    ax.set_yticklabels([f"C{k} {NAME[k]}" for (_, k) in rows])
    for tl, (_, k) in zip(ax.get_yticklabels(), rows):
        tl.set_color(COL[k])
    ax.tick_params(axis="y", length=0, pad=2)

    ax.set_xlim(xL, xR); ax.set_ylim(ybot, ytop)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Heritability enrichment (fold)")

    handles = [
        Line2D([], [], marker="o", ls="None", mfc="0.35", mec="0.35", ms=6.8,
               label="Significant (one-tailed $P$ < 0.05)"),
        Line2D([], [], marker="o", ls="None", mfc="white", mec="0.35", mew=1.2,
               ms=6.8, label="Not significant"),
        Line2D([], [], ls="--", color="0.55", lw=0.8, label="No enrichment (1-fold)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.3,
              handletextpad=0.5, borderaxespad=0.8, labelspacing=0.5)

    fig.subplots_adjust(left=0.175, right=0.975, top=0.985, bottom=0.085)

    # ---- provenance print --------------------------------------------------
    for gi, (fn, h) in enumerate(FACETS):
        print(h, {f"C{k}": (round(data[(gi, k)][0], 1), round(data[(gi, k)][1], 1),
                  round(data[(gi, k)][2], 2), stars(data[(gi, k)][2]) or "ns")
                  for k in (0, 1, 2)})
    print("xlim", round(xL, 1), round(xR, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "Fig2_partitioned_heritability"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", base)


if __name__ == "__main__":
    main()
