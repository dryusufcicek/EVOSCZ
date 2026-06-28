#!/usr/bin/env python3
"""
Phase 14d (analysis): Bryois Cell-Type eQTL Integration + Tests
=================================================================
Integrates Bryois 2022 cell-type-specific eQTL min p values with
variant_master_v3, then runs:

  1. variant_master_v4.parquet (8 cell type min_p columns)
  2. Per-cell-type eQTL × age within-locus + MAF residualized
  3. Cell-type-specificity score per variant
  4. Per-cluster (Phase 14b v3) × cell-type enrichment
  5. Cell-type × evolutionary class — biological coherence

Output:
  results/phase11/variant_master_v4.parquet
  results/phase14d/P14d_per_celltype_age_tests.tsv
  results/phase14d/P14d_cluster_celltype_enrichment.tsv
  results/phase14d/P14d_NARRATIVE.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.stats import linregress
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14d"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14d: Bryois Cell-Type Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

np.random.seed(42)

# Load Bryois wide table
bryois = pd.read_csv(BASE / "data/processed/bryois_2022/Bryois_PGC3_minp_per_celltype.tsv.gz", sep="\t")
log(f"\nBryois wide table: {bryois.shape}")
log(f"Columns: {list(bryois.columns)}")

# Load variant master v3
m_v3 = pd.read_parquet(P11 / "variant_master_v3.parquet")  # P14d input is v3 (BEFORE Bryois join — produces v4)
log(f"v3 master: {len(m_v3)} variants, {len(m_v3.columns)} cols")


# Merge Bryois into v3 → v4
m_v4 = m_v3.merge(bryois, on="rsid", how="left")
log(f"v4 master: {len(m_v4)} variants, {len(m_v4.columns)} cols")

# Identify cell type columns
ct_cols = [c for c in m_v4.columns if c.startswith("bryois_")]
log(f"Cell type min p columns: {ct_cols}")
for c in ct_cols:
    n = m_v4[c].notna().sum()
    log(f"  {c}: {n}/{len(m_v4)} ({n/len(m_v4)*100:.1f}%)")

m_v4.to_parquet(P11 / "variant_master_v4.parquet", index=False, compression="snappy")
log(f"\nSaved: {P11 / 'variant_master_v4.parquet'}")


# ─── Statistical analyses ──────────────────────────────────────────────────
log("\n" + "=" * 72)
log("[Analysis 1] Per-cell-type eQTL strength × age (within-locus + MAF resid)")
log("=" * 72)

m_age = m_v4[m_v4["age_median_yr"].notna() & (m_v4["age_median_yr"] > 0) &
              m_v4["maf"].notna()].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr_int"] = pd.to_numeric(m_age["chr"], errors="coerce")
m_age["pos_int_num"] = pd.to_numeric(m_age["pos"], errors="coerce")
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int_num"] >= 43_000_000) & \
          (m_age["pos_int_num"] <= 46_000_000)

# −log10 transform
for c in ct_cols:
    m_age[f"{c}_logp"] = np.where(m_age[c].notna() & (m_age[c] > 0),
                                    -np.log10(m_age[c].clip(lower=1e-300)), np.nan)


import sys as _sys
_sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation


def maf_resid_rho(df, x_col, y_col, locus_col="credible_set_id", maf_col="maf"):
    """Within-locus partial rank correlation (code-review Faz C lib)."""
    r = within_locus_partial_rank_correlation(
        df, x_col, y_col, locus_col, maf_col=maf_col, min_n=5
    )
    if r is None:
        return None, None, 0
    return r["rho"], r["p"], r["n_pooled"]


ct_results = []
log(f"\n  {'Cell type':<35s} {'n':>7s} {'rho':>9s} {'p':>11s} {'rho_no_MAPT':>12s} {'p_no_MAPT':>11s}")
log("  " + "-" * 95)
for c in ct_cols:
    ct_short = c.replace("bryois_", "").replace("_minp", "")
    rho_full, p_full, n_full = maf_resid_rho(m_age, "log_age", f"{c}_logp")
    rho_nm, p_nm, n_nm = maf_resid_rho(m_age[~in_mapt], "log_age", f"{c}_logp")
    if rho_full is not None and rho_nm is not None:
        ct_results.append({
            "cell_type": ct_short,
            "n_full": n_full, "rho_full": rho_full, "p_full": p_full,
            "n_no_mapt": n_nm, "rho_no_mapt": rho_nm, "p_no_mapt": p_nm,
        })
        log(f"  {ct_short:<35s} {n_full:>7d} {rho_full:>+8.4f} {p_full:>11.2e} "
            f"{rho_nm:>+12.4f} {p_nm:>11.2e}")

df_ct = pd.DataFrame(ct_results)
df_ct["fdr_q"] = multipletests(df_ct["p_no_mapt"].fillna(1.0), method="fdr_bh")[1]
df_ct.to_csv(OUT / "P14d_per_celltype_age_tests.tsv", sep="\t", index=False)


# ─── Analysis 2: Cell-type specificity (max p / mean p) ────────────────────
log("\n" + "=" * 72)
log("[Analysis 2] Cell-type specificity score per variant × age")
log("=" * 72)

# For variants with eQTLs in 3+ cell types, compute Gini coefficient or just
# (max -log10p / sum -log10p) — analog of brain spec
ct_logp_cols = [f"{c}_logp" for c in ct_cols]
m_age["ct_n_with_eqtl"] = m_age[ct_logp_cols].notna().sum(axis=1)
m_age["ct_max_logp"] = m_age[ct_logp_cols].max(axis=1)
m_age["ct_sum_logp"] = m_age[ct_logp_cols].sum(axis=1, min_count=1)
m_age["ct_specificity"] = m_age["ct_max_logp"] / m_age["ct_sum_logp"]

log(f"  Variants with ≥1 cell-type eQTL: {(m_age['ct_n_with_eqtl'] >= 1).sum()}")
log(f"  Variants with ≥3 cell-type eQTLs: {(m_age['ct_n_with_eqtl'] >= 3).sum()}")

mask_3plus = m_age["ct_n_with_eqtl"] >= 3
sub = m_age[mask_3plus].copy()
log(f"\n  Cell-type specificity tests (3+ cell types):")
for label, df in [("All", sub), ("No 17q21.31", sub[~in_mapt[sub.index]])]:
    rho, p, n = maf_resid_rho(df, "log_age", "ct_specificity")
    log(f"    {label}: n={n}, rho={rho:+.4f}, p={p:.2e}")


# ─── Analysis 3: Per-cluster cell-type preference ──────────────────────────
log("\n" + "=" * 72)
log("[Analysis 3] Per-evolutionary-cluster cell-type enrichment")
log("=" * 72)

# Load cluster assignments
ca = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")
m_age = m_age.merge(ca[["rsid", "cluster"]], on="rsid", how="left")
m_age = m_age.reset_index(drop=True)
# Recompute in_mapt after reset_index
in_mapt = (m_age["chr_int"] == 17) & (m_age["pos_int_num"] >= 43_000_000) & \
          (m_age["pos_int_num"] <= 46_000_000)

# For each cluster × cell type: what fraction of cluster variants have eQTL?
log(f"\n  Cluster size with cell-type info:")
for c in [0, 1, 2]:
    n_total = (m_age["cluster"] == c).sum()
    log(f"    C{c}: {n_total} variants")

log(f"\n  % of cluster variants with eQTL in each cell type (excl 17q21.31):")
log(f"  {'Cell type':<30s} {'C0 (n=)':>10s} {'C1':>10s} {'C2':>10s} {'chi2':>8s} {'p':>10s}")
log("  " + "-" * 80)

cluster_ct_rows = []
m_nm = m_age[~in_mapt]
for c in ct_cols:
    ct_short = c.replace("bryois_", "").replace("_minp", "")
    pct_per_cluster = []
    n_with_eqtl = m_nm[c].notna()
    for cl in [0, 1, 2]:
        in_cl = m_nm["cluster"] == cl
        n_total = in_cl.sum()
        n_eqtl = (in_cl & n_with_eqtl).sum()
        pct = n_eqtl / n_total * 100 if n_total > 0 else 0
        pct_per_cluster.append((n_eqtl, n_total, pct))
    # Chi-square
    contingency = np.array([[ne, nt - ne] for ne, nt, _ in pct_per_cluster])
    try:
        chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    except Exception:
        chi2, p_chi = np.nan, np.nan
    cluster_ct_rows.append({
        "cell_type": ct_short,
        "C0_pct": pct_per_cluster[0][2], "C1_pct": pct_per_cluster[1][2],
        "C2_pct": pct_per_cluster[2][2], "chi2": chi2, "p_chi2": p_chi
    })
    log(f"  {ct_short:<30s} "
        f"{pct_per_cluster[0][2]:>9.1f}% {pct_per_cluster[1][2]:>9.1f}% "
        f"{pct_per_cluster[2][2]:>9.1f}%  {chi2:>7.1f} {p_chi:>10.2e}")

df_clusters = pd.DataFrame(cluster_ct_rows)
df_clusters["fdr_q"] = multipletests(df_clusters["p_chi2"].fillna(1.0), method="fdr_bh")[1]
df_clusters.to_csv(OUT / "P14d_cluster_celltype_enrichment.tsv", sep="\t", index=False)


# ─── Save log ─────────────────────────────────────────────────────────────
with open(OUT / "P14d_NARRATIVE.md", "w") as f:
    f.write("# Phase 14d: Bryois Cell-Type Analysis\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nNarrative: {OUT / 'P14d_NARRATIVE.md'}")
log("\nPhase 14d complete.")
