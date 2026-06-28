#!/usr/bin/env python3
"""
Phase 14p3e — HLA exclusion + LD-aware (max-PIP per credible-set) sensitivity
==============================================================================
Two complementary sensitivity tests:

  (E1) HLA EXCLUSION: drop chr6:25–34 Mb (extended MHC, hg19) and re-run
       the C0_vs_C2 KS test. If KS D collapses → finding is HLA-driven
       (well-known balanced-selection story; reduces novelty).

  (E2) LD-AWARE: take only the max-PIP variant per credible-set
       (independent variants only). Avoids LD-block-non-independence
       confound where multiple correlated variants inflate KS apparent
       effect. If KS D collapses → finding is LD-block-driven, not
       independent-variant-driven.

Pre-registered:
  - HLA exclusion: KS D should retain ≥ 80% of full-substrate value.
    Drop > 20% → HLA-driven (NOT NULL but reframe needed).
  - LD pruning: KS D should retain ≥ 70% of full-substrate value
    (some shrinkage expected from n reduction).

Outputs:
  results/phase14p3/P14p3e_HLA_LD_sensitivity.tsv
  results/phase14p3/P14p3e_NARRATIVE.md
"""

from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
OUT = BASE / "results/phase14p3"
LOG = []

def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)


log("="*70)
log("Phase 14p3e — HLA + LD sensitivity")
log("="*70)

# Load substrate
df_dav = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
# Need PIP and credible_set_id to pick max-PIP per CS
keep_cols = ["rsid"]
for c in ["pip", "PIP", "credible_set_id", "cs_id", "credset"]:
    if c in vm.columns: keep_cols.append(c)
vm_keep = vm[keep_cols].drop_duplicates(subset=["rsid"])

df = df_dav.merge(clu, on="rsid", how="inner").merge(vm_keep, on="rsid", how="left")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
df = df.dropna(subset=["cluster","AF_AFR_derived"])
log(f"  variants: {len(df):,}")

# Identify PIP column
pip_col = next((c for c in ["pip","PIP"] if c in df.columns), None)
cs_col  = next((c for c in ["credible_set_id","cs_id","credset"] if c in df.columns), None)
log(f"  PIP column: {pip_col}; CS column: {cs_col}")

# Reference: full-substrate KS D for C0 vs C2
def ks_C0_C2(d):
    a = d[d["cluster"]==0]["AF_AFR_derived"].values
    b = d[d["cluster"]==2]["AF_AFR_derived"].values
    if len(a) < 5 or len(b) < 5: return np.nan, np.nan, len(a), len(b)
    D, P = stats.ks_2samp(a, b)
    return float(D), float(P), len(a), len(b)

D_full, P_full, n0_full, n2_full = ks_C0_C2(df)
log(f"\n[1] Full substrate (reference):")
log(f"    C0_vs_C2 KS D = {D_full:.4f}, P = {P_full:.3g}, n0={n0_full}, n2={n2_full}")

rows = [{
    "sensitivity": "full_substrate (reference)",
    "n_c0": n0_full, "n_c2": n2_full,
    "KS_D": D_full, "KS_P": P_full,
    "pct_retained_vs_full": 100.0,
}]

# ── E1: HLA exclusion ──────────────────────────────────────────────
log("\n[2] HLA exclusion (chr6: 25–34 Mb)")
hla_mask = (df["chr"] == 6) & (df["pos"].between(25_000_000, 34_000_000))
log(f"  HLA region variants: {hla_mask.sum():,}")
df_no_hla = df[~hla_mask].copy()
D_nh, P_nh, n0_nh, n2_nh = ks_C0_C2(df_no_hla)
pct_nh = (D_nh / D_full * 100) if D_full and D_full > 0 else np.nan
log(f"  After HLA exclusion: C0_vs_C2 KS D = {D_nh:.4f}, P = {P_nh:.3g}, "
    f"n0={n0_nh}, n2={n2_nh}  → {pct_nh:.1f}% of full KS retained")
rows.append({
    "sensitivity": "HLA_excluded (chr6:25-34Mb)",
    "n_c0": n0_nh, "n_c2": n2_nh,
    "KS_D": D_nh, "KS_P": P_nh,
    "pct_retained_vs_full": float(pct_nh),
})

# Also exclude broader MAPT-region inversion as added robustness
mapt_mask = (df["chr"] == 17) & (df["pos"].between(43_000_000, 46_000_000))
log(f"\n[3] MAPT 17q21.31 exclusion sensitivity (mapt variants={mapt_mask.sum()})")
df_no_mapt = df[~mapt_mask].copy()
D_nm, P_nm, n0_nm, n2_nm = ks_C0_C2(df_no_mapt)
rows.append({
    "sensitivity": "MAPT_excluded (chr17:43-46Mb)",
    "n_c0": n0_nm, "n_c2": n2_nm,
    "KS_D": D_nm, "KS_P": P_nm,
    "pct_retained_vs_full": float(D_nm / D_full * 100) if D_full > 0 else np.nan,
})

# Joint HLA+MAPT exclusion
df_no_hla_mapt = df[~hla_mask & ~mapt_mask].copy()
D_nhm, P_nhm, n0_nhm, n2_nhm = ks_C0_C2(df_no_hla_mapt)
rows.append({
    "sensitivity": "HLA_AND_MAPT_excluded",
    "n_c0": n0_nhm, "n_c2": n2_nhm,
    "KS_D": D_nhm, "KS_P": P_nhm,
    "pct_retained_vs_full": float(D_nhm / D_full * 100) if D_full > 0 else np.nan,
})

# ── E2: LD-aware (max-PIP per CS) ──────────────────────────────────
log("\n[4] LD-aware: max-PIP per credible-set")
if pip_col is None or cs_col is None:
    log("  PIP or CS column missing; LD-aware sensitivity SKIPPED")
    rows.append({
        "sensitivity": "max_PIP_per_CS (LD-aware)",
        "n_c0": np.nan, "n_c2": np.nan,
        "KS_D": np.nan, "KS_P": np.nan,
        "pct_retained_vs_full": np.nan,
    })
else:
    df_ld = df.sort_values(pip_col, ascending=False).drop_duplicates(subset=[cs_col], keep="first")
    log(f"  After LD-pruning to max-PIP per CS: {len(df_ld):,}")
    D_ld, P_ld, n0_ld, n2_ld = ks_C0_C2(df_ld)
    pct_ld = (D_ld / D_full * 100) if D_full > 0 else np.nan
    log(f"  C0_vs_C2 KS D = {D_ld:.4f}, P = {P_ld:.3g}, n0={n0_ld}, n2={n2_ld}  "
        f"→ {pct_ld:.1f}% retained")
    rows.append({
        "sensitivity": "max_PIP_per_CS (LD-aware)",
        "n_c0": n0_ld, "n_c2": n2_ld,
        "KS_D": D_ld, "KS_P": P_ld,
        "pct_retained_vs_full": float(pct_ld),
    })

# Save
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT / "P14p3e_HLA_LD_sensitivity.tsv", sep="\t", index=False)
log("\n" + df_out.to_string(index=False))

# Pre-registered verdict
log("\n[5] Pre-registered sensitivity gates")
hla_pct = rows[1]["pct_retained_vs_full"]
ld_row  = next((r for r in rows if "max_PIP" in r["sensitivity"]), None)
ld_pct  = ld_row["pct_retained_vs_full"] if ld_row else np.nan

log(f"  HLA exclusion retention: {hla_pct:.1f}%  (threshold ≥80%)")
log(f"  LD-pruning retention:    {ld_pct:.1f}%  (threshold ≥70%)") if not np.isnan(ld_pct) else log("  LD-pruning: skipped")

# Save narrative
with open(OUT / "P14p3e_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3e — HLA + LD sensitivity\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p3e complete.")
