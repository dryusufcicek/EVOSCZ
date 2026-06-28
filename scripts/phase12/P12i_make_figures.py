#!/usr/bin/env python3
"""
Phase 12i: Manuscript Figures
==============================
Produces the primary figures for the EVOSCZ manuscript:
  Figure 1: Brain specificity × age, within-locus (PRIMARY FINDING)
  Figure 2: Per-tissue forest plot (rho ± 95% CI, full vs. no MAPT)
  Figure 3: Within-locus vs raw variant-pooled comparison (Simpson's paradox)
  Figure 4: 16 H4>0.8 immune coloc lokus brain specificity profile

Output:
  - results/phase12/figures/Fig1_brain_spec_age_within_locus.png
  - results/phase12/figures/Fig2_per_tissue_forest.png
  - results/phase12/figures/Fig3_within_vs_raw.png
  - results/phase12/figures/Fig4_immune_loci_specificity.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
P12 = BASE / "results/phase12"
FIG = P12 / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print(f"Phase 12i: Figures — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

m = pd.read_parquet(P11 / "variant_master_clean.parquet")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int"] >= 43_000_000) & \
          (m_age["pos_int"] <= 46_000_000)

m_age["b_logp"]  = np.where(m_age["gtex_brain_minp"].notna() & (m_age["gtex_brain_minp"] > 0),
                              -np.log10(m_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m_age["bl_logp"] = np.where(m_age["gtex_blood_minp"].notna() & (m_age["gtex_blood_minp"] > 0),
                              -np.log10(m_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m_age["brain_spec"] = m_age["b_logp"] / (m_age["b_logp"] + m_age["bl_logp"])

# ─── Figure 1: Brain specificity × age within-locus (primary) ─────────────
print("\nFigure 1: Brain specificity × age (within-locus residuals)")
sub = m_age[m_age["brain_spec"].notna() & (~in_mapt)].copy()
# Compute within-locus residuals
within_data = []
for cs_id, grp in sub.groupby("credible_set_id"):
    if len(grp) < 5: continue
    age_resid = grp["log_age"] - grp["log_age"].mean()
    spec_resid = grp["brain_spec"] - grp["brain_spec"].mean()
    for a, s in zip(age_resid, spec_resid):
        within_data.append((a, s))
df_w = pd.DataFrame(within_data, columns=["age_resid", "spec_resid"])
rho, p = stats.spearmanr(df_w["age_resid"], df_w["spec_resid"])
print(f"  n={len(df_w)}, rho={rho:.4f}, p={p:.3e}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.hexbin(df_w["age_resid"], df_w["spec_resid"], gridsize=40, cmap="Blues",
          mincnt=1, alpha=0.7)
# Trend line via Spearman-equivalent rank regression visualization
from scipy.stats import linregress
r = linregress(df_w["age_resid"], df_w["spec_resid"])
xx = np.array([df_w["age_resid"].min(), df_w["age_resid"].max()])
ax.plot(xx, r.intercept + r.slope * xx, "r-", lw=2, label=f"Linear fit (slope={r.slope:.3f})")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("log(age) residual (within credible set)")
ax.set_ylabel("Brain specificity residual")
ax.set_title(f"Within-locus age × brain-specificity (excl. 17q21.31)\n"
             f"Spearman ρ = {rho:.3f}, p = {p:.1e}, n = {len(df_w):,} variants")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "Fig1_brain_spec_age_within_locus.png", dpi=150)
plt.close()
print(f"  Saved: {FIG / 'Fig1_brain_spec_age_within_locus.png'}")


# ─── Figure 2: Per-tissue forest plot ─────────────────────────────────────
print("\nFigure 2: Per-tissue forest plot")
pt = pd.read_csv(P12 / "P12c_per_tissue_results.tsv", sep="\t")
pt = pt.sort_values("rho_full", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(pt))
ax.scatter(pt["rho_full"], y_pos + 0.15, s=50, c="steelblue", label="All variants", marker="o")
ax.scatter(pt["rho_no_mapt"], y_pos - 0.15, s=50, c="darkorange",
           label="Excl. 17q21.31", marker="s")
ax.axvline(0, color="black", lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([t.replace("_", " ").replace("Brain ", "") for t in pt["tissue"]],
                    fontsize=9)
ax.set_xlabel("Spearman ρ (allele age × −log10 eQTL p)")
ax.set_title("Per-tissue age × eQTL effect strength")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "Fig2_per_tissue_forest.png", dpi=150)
plt.close()
print(f"  Saved: {FIG / 'Fig2_per_tissue_forest.png'}")


# ─── Figure 3: Within vs raw comparison (Simpson's paradox) ────────────────
print("\nFigure 3: Within vs raw")
results = pd.read_csv(P12 / "P12_consolidated_findings.tsv", sep="\t")

fig, ax = plt.subplots(figsize=(8, 5))
# Pair the raw with within for each test type
test_pairs = [
    ("Brain spec × age (raw)", "Brain spec × age (within)"),
    ("Brain eQTL × age (raw)", "Brain eQTL × age (within)"),
    ("Blood eQTL × age (raw)", "Blood eQTL × age (within)"),
    ("SDS × age (raw)", "SDS × age (within)"),
]
labels = ["Brain specificity", "Brain eQTL", "Blood eQTL", "SDS"]
raw_rhos = []
within_rhos = []
for raw_name, within_name in test_pairs:
    raw = results[results["test"] == raw_name]
    within = results[results["test"] == within_name]
    if len(raw) and len(within):
        raw_rhos.append(raw["rho"].iloc[0])
        within_rhos.append(within["rho"].iloc[0])
    else:
        raw_rhos.append(np.nan)
        within_rhos.append(np.nan)

x = np.arange(len(labels))
ax.bar(x - 0.2, raw_rhos, 0.4, label="Variant-pooled (raw)", color="lightcoral")
ax.bar(x + 0.2, within_rhos, 0.4, label="Within-locus residual", color="steelblue")
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Spearman ρ vs allele age")
ax.set_title("Variant-pooled vs within-locus residual correlations\n"
             "(Simpson's paradox: raw correlations inflated by between-locus structure)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "Fig3_within_vs_raw.png", dpi=150)
plt.close()
print(f"  Saved: {FIG / 'Fig3_within_vs_raw.png'}")


# ─── Figure 4: 16 immune coloc loci specificity profile ────────────────────
print("\nFigure 4: 16 immune coloc loci")
spec_df = pd.read_csv(P12 / "P12f_locus_summary.tsv", sep="\t")
spec_df = spec_df.dropna(subset=["median_brain_specificity"]).sort_values("median_brain_specificity")

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(spec_df))
colors = ["red" if s < 0.4 else "blue" if s > 0.6 else "gray"
          for s in spec_df["median_brain_specificity"]]
ax.barh(y_pos, spec_df["median_brain_specificity"], color=colors, alpha=0.7)
ax.axvline(0.5, color="black", lw=0.5, ls="--", label="Equal brain/blood")
ax.axvline(0.4, color="gray", lw=0.5, ls=":", alpha=0.5)
ax.axvline(0.6, color="gray", lw=0.5, ls=":", alpha=0.5)
labels = [f"{r['credible_set_id']}: {str(r['l2g_gene'])[:10] if pd.notna(r['l2g_gene']) else '?'} | "
          f"{str(r['immune_trait'])[:25]}" for _, r in spec_df.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Median brain specificity (within locus)")
ax.set_title("16 PGC3 SCZ–immune H4>0.8 case studies\n"
             "blue=brain-dominant, red=blood-dominant, gray=balanced")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(FIG / "Fig4_immune_loci_specificity.png", dpi=150)
plt.close()
print(f"  Saved: {FIG / 'Fig4_immune_loci_specificity.png'}")

print("\nPhase 12i complete.")
