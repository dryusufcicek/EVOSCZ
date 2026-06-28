"""
Within-locus rank-based correlation utilities (code-review Faz C fix).

Replaces the prior "mean-centred Spearman" pattern (which was NOT a valid
within-group rank correlation) with proper within-group ranking followed by
pooled Pearson correlation of within-group rank residuals — equivalent to a
fixed-effects rank model with optional MAF-rank covariate.

Two correctness issues addressed:
  1. Mean-centring values then computing Spearman on the pooled centred values
     altered the rank structure across loci. Correct: rank within each locus
     first, then center those within-group ranks, then pool, then Pearson.
  2. MAF residualization on within-locus residuals via a pooled linear regression
     left residual MAF-by-locus interaction structure. Correct: residualize the
     within-locus rank of x and y against the within-locus rank of MAF inside
     each locus, then pool.

Also includes a per-locus block-bootstrap CI helper that recomputes the
within-locus residualization inside each iteration (so CI properly reflects
uncertainty in the residualization step).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average-method ranks for a 1-D array (NaN-free)."""
    return stats.rankdata(x, method="average")


def within_locus_partial_rank_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    maf_col: str | None = None,
    min_n: int = 5,
) -> dict | None:
    """Within-group partial rank correlation (Spearman with locus fixed effect
    and optional MAF-rank covariate), pooled.

    Procedure:
      1. For each group (e.g. credible set), if it has >= min_n variants:
         a. Rank x, y, and MAF within the group (average method).
         b. If maf_col given, residualize x_rank ~ maf_rank and y_rank ~ maf_rank
            within the group via OLS; else center on group mean rank.
         c. Save within-group residualized ranks.
      2. Pool across groups.
      3. Pearson correlation on pooled residuals (= within-group partial Spearman).
      4. Asymptotic Pearson p-value (anti-conservative for non-iid data; use
         per_locus_bootstrap_ci for proper CI).

    Returns dict {rho, p, n_pooled, n_groups} or None if insufficient data.
    """
    cols = [x_col, y_col, group_col] + ([maf_col] if maf_col else [])
    sub = df[cols].dropna().copy()
    if len(sub) < min_n:
        return None
    pooled_x = []
    pooled_y = []
    n_groups = 0
    for _, grp in sub.groupby(group_col):
        if len(grp) < min_n:
            continue
        if grp[x_col].nunique() < 2 or grp[y_col].nunique() < 2:
            # Zero variance within group → cannot rank-correlate; skip
            continue
        x_rank = _ranks(grp[x_col].values)
        y_rank = _ranks(grp[y_col].values)
        if maf_col is not None and grp[maf_col].nunique() >= 2:
            maf_rank = _ranks(grp[maf_col].values)
            # OLS residualization: slope = cov(x_r, m_r)/var(m_r)
            mr_c = maf_rank - maf_rank.mean()
            denom = float((mr_c**2).sum())
            if denom > 0:
                bx = float(((x_rank - x_rank.mean()) * mr_c).sum()) / denom
                by = float(((y_rank - y_rank.mean()) * mr_c).sum()) / denom
                x_res = (x_rank - x_rank.mean()) - bx * mr_c
                y_res = (y_rank - y_rank.mean()) - by * mr_c
            else:
                x_res = x_rank - x_rank.mean()
                y_res = y_rank - y_rank.mean()
        else:
            x_res = x_rank - x_rank.mean()
            y_res = y_rank - y_rank.mean()
        pooled_x.extend(x_res.tolist())
        pooled_y.extend(y_res.tolist())
        n_groups += 1
    if len(pooled_x) < 30 or n_groups < 2:
        return None
    rho, p = stats.pearsonr(pooled_x, pooled_y)
    return {
        "rho": float(rho),
        "p": float(p),
        "n_pooled": len(pooled_x),
        "n_groups": n_groups,
    }


def per_locus_bootstrap_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    maf_col: str | None = None,
    min_n: int = 5,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict | None:
    """Per-locus (cluster) block bootstrap with proper within-iteration
    residualization. Returns 95% percentile CI plus point-estimate from the
    full data.
    """
    full = within_locus_partial_rank_correlation(df, x_col, y_col, group_col,
                                                  maf_col=maf_col, min_n=min_n)
    if full is None:
        return None
    rng = np.random.default_rng(seed)
    cols = [x_col, y_col, group_col] + ([maf_col] if maf_col else [])
    sub = df[cols].dropna().copy()
    groups = sub[group_col].unique()
    if len(groups) < 5:
        return None
    rhos = []
    for _ in range(n_iter):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        # Concatenate per-bootstrap-group; preserve grouping by giving each
        # sampled instance a unique synthetic group id (so duplicate locus picks
        # are treated as separate within-group blocks).
        parts = []
        for j, g in enumerate(sampled):
            grp = sub[sub[group_col] == g].copy()
            grp[group_col] = f"{g}__bs{j}"
            parts.append(grp)
        boot = pd.concat(parts, ignore_index=True)
        r = within_locus_partial_rank_correlation(
            boot, x_col, y_col, group_col, maf_col=maf_col, min_n=min_n
        )
        if r is not None:
            rhos.append(r["rho"])
    if not rhos:
        return None
    arr = np.array(rhos)
    return {
        "rho_point": full["rho"],
        "p_point": full["p"],
        "n_pooled": full["n_pooled"],
        "n_groups": full["n_groups"],
        "rho_boot_mean": float(arr.mean()),
        "rho_ci95_lower": float(np.percentile(arr, 2.5)),
        "rho_ci95_upper": float(np.percentile(arr, 97.5)),
        "n_iter_success": int(len(rhos)),
    }
