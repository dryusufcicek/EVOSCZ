#!/usr/bin/env python3
"""Fig 3 (v25_3 reframe) — two-panel cell-type figure (publication-quality house
style), eight Bryois 2022 brain cell types on a shared y-axis ordered by panel-b
correlation.

  a  Cluster cis-eQTL coverage — proportion of each cluster's variants with >=1
     significant single-cell cis-eQTL, as a dumbbell/connected-dot plot
     (C0 Young / C1 Mid / C2 Old). Coverage is uniformly high (78-98%); the point
     is the small C0-C2 difference (0.1-3.1 pp) vs the consistent C1 deficit, so a
     dot plot (not a 0- or truncated-baseline bar) is used. Three-way chi-square P
     in the caption (all P < 6e-6; microglia 1.4e-16).
  b  Within-locus allele-age x per-cell-type cis-eQTL-strength partial rank
     correlation (MAPT-excluded), as a diverging lollipop about rho = 0. Filled =
     significant (BH-FDR q < 0.05), open = not significant. The sign reverses:
     oligodendrocytes/astrocytes negative (older variants -> weaker eQTLs) vs
     microglia/neurons positive.

Data: results/phase14d/P14d_cluster_celltype_enrichment_CORRECTED.tsv (= Supp 7a,
matches manuscript) and results/phase14d/P14d_per_celltype_age_tests.tsv (rho_no_mapt
= manuscript values). Zero fabrication.
"""
from __future__ import annotations
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

COVSRC = (BASE / "results/phase14d/P14d_cluster_celltype_enrichment_CORRECTED.tsv")
RHOSRC = (BASE / "results/phase14d/P14d_per_celltype_age_tests.tsv")
OUT = Path(os.environ.get("EVOSCZ_FIG_OUT") or (BASE / "figures"))
COL = {"C0": "#0072B2", "C1": "#E69F00", "C2": "#009E73"}
NAME = {"C0": "Young", "C1": "Mid", "C2": "Old"}
DOT = "#3a3a3a"  # neutral marker for panel b
DISPLAY = {
    "Astrocytes": "Astrocytes", "Endothelial_cells": "Endothelial cells",
    "Excitatory_neurons": "Excitatory neurons", "Inhibitory_neurons": "Inhibitory neurons",
    "Microglia": "Microglia", "OPCs___COPs": "OPCs/COPs",
    "Oligodendrocytes": "Oligodendrocytes", "Pericytes": "Pericytes",
}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 7,
    "legend.fontsize": 6.2, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})


def main():
    cov = pd.read_csv(COVSRC, sep="\t")
    rho = pd.read_csv(RHOSRC, sep="\t")
    rho = rho[["cell_type", "rho_no_mapt", "p_no_mapt", "fdr_q", "n_no_mapt"]].rename(
        columns={"fdr_q": "rho_fdr_q"})
    df = cov.merge(rho, on="cell_type")
    df = df.sort_values("rho_no_mapt").reset_index(drop=True)  # most negative first
    n = len(df)
    df["y"] = [n - 1 - i for i in range(n)]  # most negative rho on top

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 3.8), sharey=True,
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.11}, dpi=300)

    # ---- panel a: coverage dumbbell --------------------------------------
    for _, r in df.iterrows():
        y = r["y"]; vals = [r.C0_pct, r.C1_pct, r.C2_pct]
        axA.plot([min(vals), max(vals)], [y, y], color="0.72", lw=1.4, zorder=1,
                 solid_capstyle="round")
        for c in ("C2", "C0", "C1"):
            axA.plot(r[f"{c}_pct"], y, "o", ms=6.4, mfc=COL[c], mec="white",
                     mew=0.8, zorder=3)
    axA.set_xlim(75, 100); axA.set_xticks([75, 80, 85, 90, 95, 100])
    axA.set_xlabel("Cluster variants with $\\geq$1\nsignificant cis-eQTL (%)")

    # ---- panel b: age x eQTL-strength partial rho (diverging lollipop) -----
    axB.axvline(0, color="0.55", lw=0.8, zorder=1)
    for _, r in df.iterrows():
        y = r["y"]; rr = r["rho_no_mapt"]; sig = r["rho_fdr_q"] < 0.05
        axB.plot([0, rr], [y, y], color="0.72", lw=1.4, zorder=1, solid_capstyle="round")
        axB.plot(rr, y, "o", ms=6.4, mfc=DOT if sig else "white", mec=DOT,
                 mew=1.1, zorder=3)
    axB.set_xlim(-0.145, 0.115); axB.set_xticks([-0.1, 0.0, 0.1])
    axB.set_xlabel("Within-locus partial $\\rho$\n(allele age $\\times$ cis-eQTL strength)")

    # ---- shared y labels (cell types) ------------------------------------
    axA.set_yticks(range(n)); axA.set_ylim(-0.7, n - 0.3)
    lab = [DISPLAY[df.loc[df["y"] == j, "cell_type"].iloc[0]] for j in range(n)]
    axA.set_yticklabels(lab)
    axA.tick_params(axis="y", length=0, pad=3)
    axB.tick_params(axis="y", labelleft=False, length=0)

    # ---- panel letters ----------------------------------------------------
    axA.text(-0.46, 1.02, "a", transform=axA.transAxes, fontsize=10, fontweight="bold",
             va="bottom", ha="left")
    axB.text(-0.10, 1.02, "b", transform=axB.transAxes, fontsize=10, fontweight="bold",
             va="bottom", ha="left")

    # ---- legends ----------------------------------------------------------
    hA = [Line2D([], [], marker="o", ls="None", mfc=COL[c], mec="white", mew=0.8,
                 ms=6.4, label=f"{c} {NAME[c]}") for c in ("C0", "C1", "C2")]
    axA.legend(handles=hA, frameon=False, loc="lower left", fontsize=6.2,
               handletextpad=0.3, borderaxespad=0.4, labelspacing=0.35)
    hB = [Line2D([], [], marker="o", ls="None", mfc=DOT, mec=DOT, ms=6.4,
                 label="Significant (FDR $q$ < 0.05)"),
          Line2D([], [], marker="o", ls="None", mfc="white", mec=DOT, mew=1.1,
                 ms=6.4, label="Not significant")]
    axB.legend(handles=hB, frameon=False, loc="lower left", fontsize=6.2,
               handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)

    fig.subplots_adjust(left=0.125, right=0.99, top=0.93, bottom=0.18)

    # ---- provenance -------------------------------------------------------
    for _, r in df.sort_values("y", ascending=False).iterrows():
        print(f"{DISPLAY[r['cell_type']]:20s} cov C0/C1/C2={r.C0_pct:.1f}/{r.C1_pct:.1f}/"
              f"{r.C2_pct:.1f}  rho_noMAPT={r.rho_no_mapt:+.3f} fdrq={r.rho_fdr_q:.1e} "
              f"{'sig' if r.rho_fdr_q < 0.05 else 'ns'}")
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "Fig3_celltype_eqtl_coverage"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", base)


if __name__ == "__main__":
    main()
