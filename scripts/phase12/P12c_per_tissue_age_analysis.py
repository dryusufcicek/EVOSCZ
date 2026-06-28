#!/usr/bin/env python3
"""
Phase 12c: Per-Tissue Age × eQTL Analysis
==========================================
Rather than aggregating to "brain" vs "blood", run age × eQTL min-p Spearman
for each individual GTEx tissue. Reveals which tissues show the age signal.

Output:
  - results/phase12/P12c_per_tissue_results.tsv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from datetime import datetime
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, str(Path(__file__).parent.parent / "phase11"))
from lib_gtex_v10 import load_tissue, list_available_parquet, GTEX_DIR

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase12"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


log(f"Phase 12c: Per-Tissue Age × eQTL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

# Load variant master
m = pd.read_parquet(BASE / "results/phase11/variant_master.parquet")
m = m.drop_duplicates("rsid", keep="first")
m_age = m[m["age_median_yr"].notna() & (m["age_median_yr"] > 0)].copy()
m_age["log_age"] = np.log10(m_age["age_median_yr"])
m_age["chr"] = m_age["chr"].astype(str)
m_age["pos_int"] = pd.to_numeric(m_age["pos"], errors="coerce")
m_age["pos_hg38"] = pd.to_numeric(m_age["pos_hg38"], errors="coerce")

# 17q21.31 mask
mapt_mask = (m_age["chr"] == "17") & (m_age["pos_int"] >= 43_000_000) & \
            (m_age["pos_int"] <= 46_000_000)

# Get all tissues
tissues = sorted(list_available_parquet())
log(f"Locally available tissues: {len(tissues)}")

# Build per-tissue keys for join
m38 = m_age[m_age["pos_hg38"].notna()][["rsid", "chr", "pos_hg38", "log_age", "age_median_yr"]].copy()
m38["pos"] = m38["pos_hg38"].astype(int)
m38["chr"] = m38["chr"].astype(str)

results = []
for t in tissues:
    eqtl = load_tissue(t)
    eqtl["chr"] = eqtl["chr"].astype(str)
    eqtl["pos"] = eqtl["pos"].astype(int)
    # Per variant min p in this tissue
    minp = eqtl.groupby(["chr", "pos"])["pval_nominal"].min().reset_index()
    merged = m38.merge(minp, on=["chr", "pos"], how="inner")
    if len(merged) < 50:
        log(f"  {t}: only {len(merged)} hits — skip")
        continue
    merged["neg_p"] = -np.log10(merged["pval_nominal"].clip(lower=1e-300))

    # Full
    rho, p = stats.spearmanr(merged["log_age"], merged["neg_p"])
    n = len(merged)

    # Excluding 17q21.31
    in_mapt = merged.merge(m_age[mapt_mask][["rsid"]].drop_duplicates(),
                            on="rsid", how="inner")
    no_mapt = merged[~merged["rsid"].isin(in_mapt["rsid"])]
    if len(no_mapt) > 30:
        rho_nm, p_nm = stats.spearmanr(no_mapt["log_age"], no_mapt["neg_p"])
        n_nm = len(no_mapt)
    else:
        rho_nm = None; p_nm = None; n_nm = len(no_mapt)

    results.append({
        "tissue": t,
        "n_full": n, "rho_full": rho, "p_full": p,
        "n_no_mapt": n_nm, "rho_no_mapt": rho_nm, "p_no_mapt": p_nm,
    })
    rho_nm_s = f"{rho_nm:.4f}" if rho_nm is not None else "NA"
    p_nm_s   = f"{p_nm:.2e}"   if p_nm  is not None else "NA"
    log(f"  {t:<45s}: n={n:>5d}, rho={rho:>7.4f} (p={p:.2e})  |  "
        f"no_mapt: rho={rho_nm_s} (p={p_nm_s})")

df = pd.DataFrame(results)

# FDR-correct full and no_mapt separately
if len(df) > 0:
    df["fdr_q_full"] = multipletests(df["p_full"], method="fdr_bh")[1]
    valid_nm = df["p_no_mapt"].notna()
    if valid_nm.sum() > 0:
        df.loc[valid_nm, "fdr_q_no_mapt"] = multipletests(df.loc[valid_nm, "p_no_mapt"],
                                                            method="fdr_bh")[1]

df = df.sort_values("rho_full", ascending=False)

log("\n" + "=" * 72)
log("Per-tissue summary (sorted by rho_full)")
log("=" * 72)
log(f"  {'Tissue':<45s} {'n':>6s} {'rho_full':>10s} {'q_full':>10s} {'rho_noM':>10s}")
for _, r in df.iterrows():
    qf = f"{r['fdr_q_full']:.2e}" if pd.notna(r['fdr_q_full']) else "NA"
    rnm = f"{r['rho_no_mapt']:.4f}" if pd.notna(r['rho_no_mapt']) else "NA"
    sig = "***" if pd.notna(r['fdr_q_full']) and r['fdr_q_full'] < 0.05 else ""
    log(f"  {r['tissue']:<45s} {r['n_full']:>6d} {r['rho_full']:>10.4f} {qf:>10s} {rnm:>10s}  {sig}")

df.to_csv(OUT / "P12c_per_tissue_results.tsv", sep="\t", index=False)
log(f"\nSaved: {OUT / 'P12c_per_tissue_results.tsv'}")

# Save log
with open(OUT / "P12c_ANALYSIS_LOG.md", "w") as f:
    f.write("# Phase 12c Per-Tissue Analysis Log\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log("\nPhase 12c complete.")
