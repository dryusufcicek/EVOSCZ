#!/usr/bin/env python3
"""Fig 4 (v25_3 reframe) — Within-locus partial rank correlations between
regulatory specificity, selection metrics and allele age (publication-quality).

  a  Brain-blood specificity x log10 allele age, pooled within-locus rank
     residuals (MAF-rank residualised, no MAPT). LOWESS trend + a REAL
     1,000-iteration per-locus block-bootstrap distribution inset (the bootstrap
     was regenerated from the saved pipeline with the original seed; the 95% CI
     [-0.242, +0.021] reproduces results/phase13/P13c exactly and crosses zero).
  b  Voight-corrected |iHS| x log10 allele age (within-locus rank residuals).
  c  |Akbari S| x Field 2016 SDS, within-locus partial rank residuals
     (cross-method selection consistency).
  d  Brain-blood specificity x age within-locus partial rho, stratified by GWAS
     |beta| quartile (the spec-age signal concentrates in Q2 and Q4).

UNITS: panels a-c plot WITHIN-LOCUS RANK RESIDUALS, which are invariant to the
generations->years (x28.1) rescaling, so no allele-age unit conversion applies;
panel d is a stratified rho summary. Zero fabrication: every rho/P/n matches the
saved phase outputs and the manuscript; the bootstrap inset is the genuine
distribution (results/phase13/P13c_bootstrap_samples_brainspec_age.txt).
"""
from __future__ import annotations
from pathlib import Path
import os
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * n / (i + 1)); q[idx] = prev
    return q

sys.path.insert(0, str(BASE / "scripts/phase12"))
M2 = BASE / "results/phase11/variant_master_v2.parquet"   # panels a, b
M4 = BASE / "results/phase11/variant_master_v4.parquet"   # panel c (akbari)
EFFSTRAT = BASE / "results/phase14b/P14b_effect_stratification.tsv"  # panel d
BOOT = BASE / "results/phase13/P13c_bootstrap_samples_brainspec_age.txt"
P13B = BASE / "results/phase13/P13b_maf_residualized_results.tsv"
P13_5 = BASE / "results/phase13_5/P13_5_akbari_results.tsv"
OUT = Path(os.environ.get("EVOSCZ_FIG_OUT") or (BASE / "figures"))

PT = "#6f6f6f"     # scatter points (neutral)
LW = "#111111"     # LOWESS / structure
DOT = "#3a3a3a"    # panel d markers
MAPT = (17, 43_000_000, 46_000_000)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.2, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})


def lowess(x, y, frac):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _lo
        s = _lo(y, x, frac=frac, return_sorted=True)
        return s[:, 0], s[:, 1]
    except Exception:
        i = np.argsort(x)
        return x[i], pd.Series(y[i]).rolling(max(50, len(y) // 20), center=True,
                                             min_periods=10).mean().values


def residual_cloud(df, xcol, ycol, locus="credible_set_id", maf="maf", min_n=5):
    """Within-locus MAF-rank-residualised pooled rank residuals (rx, ry).
    Replicates scripts/phase12/_within_locus_lib so pearson(rx,ry) == canonical rho."""
    sub = df[[xcol, ycol, locus, maf]].dropna().copy()
    RX, RY = [], []
    for _, g in sub.groupby(locus):
        if len(g) < min_n or g[xcol].nunique() < 2 or g[ycol].nunique() < 2:
            continue
        xr = stats.rankdata(g[xcol].values); yr = stats.rankdata(g[ycol].values)
        if g[maf].nunique() >= 2:
            mr = stats.rankdata(g[maf].values); mrc = mr - mr.mean()
            den = float((mrc ** 2).sum())
            if den > 0:
                bx = float(((xr - xr.mean()) * mrc).sum()) / den
                by = float(((yr - yr.mean()) * mrc).sum()) / den
                xres = (xr - xr.mean()) - bx * mrc; yres = (yr - yr.mean()) - by * mrc
            else:
                xres = xr - xr.mean(); yres = yr - yr.mean()
        else:
            xres = xr - xr.mean(); yres = yr - yr.mean()
        RX.extend(xres.tolist()); RY.extend(yres.tolist())
    return np.array(RX), np.array(RY)


def fmt_p(p):
    if p == 0 or not np.isfinite(p):
        return r"$P < 10^{-300}$"
    if p >= 0.01:
        return f"$P$ = {p:.2g}"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    if m >= 9.95:
        m /= 10.0; e += 1
    return rf"$P$ = {m:.1f}$\times$10$^{{{e}}}$"


def prep_m2():
    m = pd.read_parquet(M2)
    m = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0) & m["maf"].notna()].copy()
    m["log_age"] = np.log10(m["age_median_yr"])
    bl = -np.log10(m["gtex_brain_minp"].clip(lower=1e-300))
    bd = -np.log10(m["gtex_blood_minp"].clip(lower=1e-300))
    m["brain_spec"] = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0)
                               & m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                               bl / (bl + bd), np.nan)
    m["abs_ihs"] = m["ihs_std"].abs()
    ci = pd.to_numeric(m["chr"], errors="coerce"); pi = pd.to_numeric(m["pos"], errors="coerce")
    m["mapt"] = (ci == MAPT[0]) & (pi >= MAPT[1]) & (pi <= MAPT[2])
    return m


def scatter_panel(ax, rx, ry, xlab, ylab, rho, p, n, frac, extra=None):
    # density view spans the central 99% of residuals; sparse-tail cells still
    # render (log bins) so outliers stay visible without dwarfing the core
    xl = float(np.percentile(np.abs(rx), 99.5)) * 1.04
    yl = float(np.percentile(np.abs(ry), 99.5)) * 1.04
    ax.axhline(0, color="0.82", lw=0.6, zorder=0)
    ax.axvline(0, color="0.82", lw=0.6, zorder=0)
    # 2-D density (hexbin, log count) makes the mass + concentration visible —
    # an alpha point cloud was a featureless blob
    ax.hexbin(rx, ry, gridsize=40, cmap="Greys", bins="log", mincnt=1,
              extent=(-xl, xl, -yl, yl), zorder=1)
    lo, hi = np.percentile(rx, [2.5, 97.5])          # LOWESS on central mass (reliable; no tail artefact)
    m = (rx >= lo) & (rx <= hi)
    sx, sy = lowess(rx[m], ry[m], frac)
    ax.plot(sx, sy, color="black", lw=1.7, zorder=3,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    txt = rf"$\rho$ = {rho:+.3f}" + "\n" + fmt_p(p) + f"\n$n$ = {n:,}"
    if extra:
        txt += "\n" + extra
    ax.text(0.035, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.4))


def main():
    m2 = prep_m2()
    m4 = pd.read_parquet(M4)
    m4 = m4[m4["age_median_yr"].notna() & (m4["age_median_yr"] > 0) & m4["maf"].notna()].copy()
    m4["abs_akbari_s"] = m4["akbari_s"].abs()
    p13b = pd.read_csv(P13B, sep="\t").set_index("test")
    p135 = pd.read_csv(P13_5, sep="\t").set_index("test")
    eff = pd.read_csv(EFFSTRAT, sep="\t")
    boot = np.loadtxt(BOOT)

    # canonical (manuscript) stats
    ca = dict(rho=p13b.loc["Brain spec × age, no 17q21.31", "rho_maf"],
              p=p13b.loc["Brain spec × age, no 17q21.31", "p_maf"],
              n=int(p13b.loc["Brain spec × age, no 17q21.31", "n_maf"]))
    cb = dict(rho=p13b.loc["|iHS| × age, no 17q21.31", "rho_maf"],
              p=p13b.loc["|iHS| × age, no 17q21.31", "p_maf"],
              n=int(p13b.loc["|iHS| × age, no 17q21.31", "n_maf"]))
    cc = dict(rho=p135.loc["|Akbari S| × SDS consistency", "rho"],
              p=p135.loc["|Akbari S| × SDS consistency", "p"],
              n=int(p135.loc["|Akbari S| × SDS consistency", "n"]))

    fig = plt.figure(figsize=(7.2, 6.1), dpi=300)
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.40,
                          left=0.10, right=0.975, top=0.965, bottom=0.085)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- panel a ----------------------------------------------------------
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rx, ry = residual_cloud(m2[~m2["mapt"]], "log_age", "brain_spec")
    scatter_panel(axa, rx, ry,
                  r"log$_{10}$ allele age (rank residual)",
                  "Brain–blood specificity\n(rank residual)",
                  ca["rho"], ca["p"], ca["n"], frac=0.55,
                  extra=f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    print("a cloud pearson", round(stats.pearsonr(rx, ry)[0], 4), "canonical", round(ca["rho"], 4))
    ins = axa.inset_axes([0.64, 0.60, 0.34, 0.33])  # framed mini-panel, sparse top-right
    ins.set_facecolor("white"); ins.patch.set_alpha(1.0); ins.set_zorder(6)
    ins.hist(boot, bins=18, color="#d9d9d9", edgecolor="#9a9a9a", linewidth=0.3)
    ins.axvline(0, color="0.55", lw=0.7, ls=":")                     # null
    ins.axvline(ca["rho"], color="#c0392b", lw=1.5)                  # observed ρ
    ins.set_yticks([]); ins.set_xticks([-0.2, 0.0])
    ins.tick_params(axis="x", labelsize=5.5, length=2, pad=1)
    for s in ("top", "right", "left", "bottom"):
        ins.spines[s].set_visible(True); ins.spines[s].set_linewidth(0.5)
        ins.spines[s].set_color("0.6")
    ins.set_title(r"block-bootstrap $\rho$ ($n$=1,000)", fontsize=5.5, pad=2)

    # ---- panel b ----------------------------------------------------------
    rx, ry = residual_cloud(m2[~m2["mapt"]], "log_age", "abs_ihs")
    scatter_panel(axb, rx, ry,
                  r"log$_{10}$ allele age (rank residual)",
                  "Voight-corrected |iHS|\n(rank residual)",
                  cb["rho"], cb["p"], cb["n"], frac=0.55)
    print("b cloud pearson", round(stats.pearsonr(rx, ry)[0], 4), "canonical", round(cb["rho"], 4))

    # ---- panel c ----------------------------------------------------------
    rx, ry = residual_cloud(m4, "abs_akbari_s", "sds")
    scatter_panel(axc, rx, ry,
                  "|Akbari S| (rank residual)",
                  "Field 2016 SDS\n(rank residual)",
                  cc["rho"], cc["p"], cc["n"], frac=0.40)
    print("c cloud pearson", round(stats.pearsonr(rx, ry)[0], 4), "canonical", round(cc["rho"], 4))

    # ---- panel d: brain-spec x age rho by |beta| quartile -----------------
    bs = eff[eff["test"] == "Brain spec × age"].copy()
    order = ["Q1_smallest", "Q2", "Q3", "Q4_largest"]
    lab = {"Q1_smallest": "Q1\n(smallest |β|)", "Q2": "Q2", "Q3": "Q3",
           "Q4_largest": "Q4\n(largest |β|)"}
    bs = bs.set_index("beta_quartile").loc[order].reset_index()
    yy = {q: len(order) - 1 - i for i, q in enumerate(order)}
    bs["q"] = bh_fdr(bs["p"].values)  # BH-FDR, harmonised with Fig 3b significance glyph
    axd.axvline(0, color="0.55", lw=0.8, zorder=1)
    for _, r in bs.iterrows():
        y = yy[r["beta_quartile"]]; rr = r["rho"]; sig = r["q"] < 0.05
        axd.plot([0, rr], [y, y], color="0.72", lw=1.5, zorder=1, solid_capstyle="round")
        axd.plot(rr, y, "o", ms=7.0, mfc=DOT if sig else "white", mec=DOT, mew=1.2, zorder=3)
        print(f"d {r['beta_quartile']:12s} rho={rr:+.3f} p={r['p']:.1e} q={r['q']:.1e} n={int(r['n'])} {'sig' if sig else 'ns'}")
    axd.set_yticks(list(yy.values()))
    axd.set_yticklabels([lab[q] for q in order])
    axd.set_ylim(-0.6, len(order) - 0.4)
    axd.set_xlim(-0.38, 0.10); axd.set_xticks([-0.3, -0.2, -0.1, 0.0])
    axd.set_xlabel("Brain–blood specificity × age\nwithin-locus partial $\\rho$")
    axd.spines["left"].set_visible(False); axd.tick_params(axis="y", length=0)
    axd.legend(handles=[
        Line2D([], [], marker="o", ls="None", mfc=DOT, mec=DOT, ms=7, label="Significant (FDR $q$ < 0.05)"),
        Line2D([], [], marker="o", ls="None", mfc="white", mec=DOT, mew=1.2, ms=7, label="Not significant")],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        fontsize=6.0, handletextpad=0.3, columnspacing=1.3)

    for ax, L in zip([axa, axb, axc, axd], "abcd"):
        ax.text(-0.20, 1.04, L, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "Fig4_within_locus_partial_rank"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", base)


if __name__ == "__main__":
    main()
