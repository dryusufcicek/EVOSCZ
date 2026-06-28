#!/usr/bin/env python3
"""Fig 5 (v25_3 reframe) — Deep-time and cross-ancestry corroboration of the
variant subsets in African genealogies (publication-quality). Redesigned as a
2-panel figure (multi-agent design pass):

  a  RIDGELINE of African-lineage coalescent ages: a top "All credible-set"
     ridge (96.9% present in African lineages, 76% > 200 kyr) and the three
     ordered clusters C0 Young / C1 Mid / C2 Old below, showing the rightward
     age shift C0 < C1 < C2 (medians 311 / 535 / 617 kyr) on one shared
     log-age axis. Stroke-led ridges, direct-labelled, no legend.
  b  Per-variant association strength (mean chi-square +/- s.e.m.) in PGC3
     African-American schizophrenia across four African-lineage age partitions,
     as a lollipop anchored at the null (chi-square = 1); the recent <50 kyr set
     drops below the null while older partitions sit at the credible-set baseline
     (~1.23). Lowest partition highlighted.

UNITS: ages used in YEARS (AFR_TMRCA_yr; 28.1 yr/gen already applied). Zero
fabrication: panel-a ages are the per-variant Wohns African TMRCA; panel-b
chi-square is the genuine per-variant PGC3 African-American Wald chi-square
((beta/se)^2) and the per-partition means reproduce P14p6b_h2_proxy_per_partition_AFRAM.
Fig 5 uses the African-lineage extraction (P14p6) throughout (user decision).
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

RES = (BASE / "results")
OUT = Path(os.environ.get("EVOSCZ_FIG_OUT") or (BASE / "figures"))
PARQ = RES / "phase14p6/P14p6b_AFR_TMRCA_per_variant.parquet"
ASSIGN = RES / "phase14p6/P14p6b_AFR_partition_assignments.tsv"
CACHE = RES / "phase14p6/_afram_lookup_cache.tsv"
H2 = RES / "phase14p6/P14p6b_h2_proxy_per_partition_AFRAM.tsv"
COL = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
GREY = "#808080"
RED = "#c0392b"
PARTS = [("AFR_TMRCA_lt_50kyr", "<50"), ("AFR_TMRCA_50_200kyr", "50–200"),
         ("AFR_TMRCA_200_500kyr", "200–500"), ("AFR_TMRCA_ge_500kyr", "≥500")]
AGE_TICKS = [5e4, 1e5, 2e5, 5e5, 1e6]
AGE_LABELS = ["50", "100", "200", "500", "1000"]
L200 = np.log10(2e5)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.6, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})


def main():
    d = pd.read_parquet(PARQ)
    pres = d[(~d["AFR_absent"]) & (d["AFR_TMRCA_yr"] > 0) & d["AFR_TMRCA_yr"].notna()].copy()
    pres["la"] = np.log10(pres["AFR_TMRCA_yr"])
    pct_present = 100 * (~d["AFR_absent"]).mean()
    pct_gt200 = 100 * (d["AFR_TMRCA_yr"] >= 2e5).mean()
    cl = pres[pres["cluster"].notna()].copy(); cl["cluster"] = cl["cluster"].astype(int)

    rows = [("All variants", pres["la"].values, GREY, None, len(pres))]
    nm = {0: "C0 Young", 1: "C1 Mid", 2: "C2 Old"}
    for k in (0, 1, 2):
        v = cl.loc[cl["cluster"] == k, "la"].values
        rows.append((nm[k], v, COL[k], float(np.median(10 ** v)), len(v)))
    print("present %.1f%%  >200kyr %.1f%%" % (pct_present, pct_gt200),
          "medians", {r[0]: (None if r[3] is None else round(r[3] / 1e3)) for r in rows})

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.28}, dpi=300)

    # ---- panel a: ridgeline ----------------------------------------------
    grid = np.linspace(pres["la"].min(), pres["la"].max(), 320)
    nr = len(rows); Hgt = 0.92
    for i, (lab, data, color, med, n) in enumerate(rows):
        base = nr - 1 - i  # All on top
        k = gaussian_kde(data)(grid); k = k / k.max() * Hgt
        z = 2 + i
        axa.fill_between(grid, base, base + k, color=color, alpha=0.16, lw=0, zorder=z)
        axa.plot(grid, base + k, color=color, lw=1.3, zorder=z)
        if med is not None:
            x = np.log10(med)
            axa.plot([x, x], [base, base + Hgt * 0.9], color=color, lw=1.0,
                     ls=(0, (2, 1.5)), zorder=z + 0.3)
    axa.axvline(L200, color=RED, ls="--", lw=0.8, zorder=10)
    axa.text(L200 - 0.03, nr - 1 + Hgt + 0.10, "200 kyr", color=RED, fontsize=6.0,
             ha="right", va="top")
    axa.text(grid[0] + 0.04, nr - 1 + Hgt - 0.02, "96.9% present in African lineages\n76% > 200 kyr",
             fontsize=6.2, ha="left", va="top", color="0.25")
    ylabels = [f"{r[0]}\n(n = {r[4]:,})" if r[3] is None
               else f"{r[0]}\n{r[3]/1e3:.0f} kyr · n = {r[4]:,}" for r in rows]
    axa.set_yticks([nr - 1 - i + Hgt * 0.34 for i in range(nr)])
    axa.set_yticklabels(ylabels)
    for tl, r in zip(axa.get_yticklabels(), rows):
        tl.set_color(r[2] if r[3] is not None else "0.4")
    axa.tick_params(axis="y", length=0, pad=3, labelsize=6.2)
    axa.set_ylim(-0.18, nr - 1 + Hgt + 0.45)
    axa.set_xticks(np.log10(AGE_TICKS)); axa.set_xticklabels(AGE_LABELS)
    axa.tick_params(axis="x", labelsize=5.9)
    axa.set_xlabel("African-lineage coalescent age (kyr)")

    # ---- panel b: AFRAM mean chi2 lollipop anchored at null --------------
    cache = pd.read_csv(CACHE, sep="\t")
    cache["chi2"] = (cache["afram_beta"] / cache["afram_se"]) ** 2
    asg = pd.read_csv(ASSIGN, sep="\t")[["rsid", "AFR_partition"]]
    cc = cache.merge(asg, on="rsid", how="inner")
    base = pd.read_csv(H2, sep="\t").set_index("partition").loc[
        "ALL_credible_set (baseline)", "mean_afram_chi2"]
    h2 = pd.read_csv(H2, sep="\t").set_index("partition")
    axb.axvspan(base - 0.0, base, color="none")  # placeholder (keeps order)
    axb.axvline(base, color="0.72", ls=":", lw=0.9, zorder=1)
    axb.axvline(1.0, color="0.45", lw=0.9, zorder=1)
    ys = list(range(len(PARTS)))[::-1]  # recent (<50) on top
    for (pkey, lab), y in zip(PARTS, ys):
        g = cc.loc[cc["AFR_partition"] == pkey, "chi2"].dropna()
        m, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
        assert abs(m - h2.loc[pkey, "mean_afram_chi2"]) < 0.02
        c = RED if pkey == "AFR_TMRCA_lt_50kyr" else "#3a3a3a"
        axb.plot([1.0, m], [y, y], color=c, lw=1.6, zorder=2, solid_capstyle="round")
        axb.errorbar(m, y, xerr=se, fmt="o", ms=7, mfc=c, mec="white", mew=0.8,
                     ecolor=c, elinewidth=1.0, capsize=2.6, zorder=3)
        print(f"b {pkey:22s} mean={m:.3f} se={se:.3f} n={len(g)}")
    axb.set_yticks(ys); axb.set_yticklabels([lab for _, lab in PARTS])
    axb.tick_params(axis="y", length=0, pad=2)
    axb.set_ylim(-0.6, len(PARTS) - 0.4)
    axb.set_xlim(0.5, 1.42); axb.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    axb.set_ylabel("African-lineage age partition (kyr)")
    axb.set_xlabel(r"Mean $\chi^2$, PGC3 African-American SCZ")
    axb.text(1.0, len(PARTS) - 0.46, "null", color="0.45", fontsize=5.6, ha="center", va="bottom")
    axb.text(base, len(PARTS) - 0.46, "baseline", color="0.6", fontsize=5.6, ha="center", va="bottom")

    for ax, L, dx in zip([axa, axb], "ab", (-0.07, -0.20)):
        ax.text(dx, 1.03, L, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="right")

    fig.subplots_adjust(left=0.165, right=0.975, top=0.93, bottom=0.135)
    OUT.mkdir(parents=True, exist_ok=True)
    base_path = OUT / "Fig5_afr_deeptime"
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base_path}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", base_path)


if __name__ == "__main__":
    main()
