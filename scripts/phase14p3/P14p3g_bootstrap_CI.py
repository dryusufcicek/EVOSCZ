#!/usr/bin/env python3
"""
Phase 14p3g — Block-bootstrap CI for KS D and median difference
================================================================
Block-bootstrap by credible-set locus (LD block proxy). Resamples
locus-level groups with replacement; recomputes KS D and median DAF
difference per resample. Reports 95% bootstrap CI.

Block bootstrap rationale: within-credible-set variants are NOT
independent (LD). Resampling at variant level would understate
uncertainty. Locus-level resampling preserves within-CS correlation
structure.

Pre-registered:
  - Lower 95% CI for KS D(C0,C2) must remain > 0.20 to corroborate
    primary effect.
  - 1,000 bootstrap iterations.

Output:
  results/phase14p3/P14p3g_bootstrap_CI.tsv
  results/phase14p3/P14p3g_NARRATIVE.md
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

log("="*70); log("Phase 14p3g — Block-bootstrap CI"); log("="*70)

df = pd.read_parquet(OUT / "P14p3b_AFR_DAF_per_variant.parquet")
clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz", sep="\t")[["rsid","cluster"]]
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")

# Determine CS column for block bootstrap
cs_col = next((c for c in ["credible_set_id","cs_id","credset"] if c in vm.columns), None)
if cs_col is None:
    log(f"  WARNING: no credible-set ID column found; using rsid groupwise (no block)")
    vm["_cs_dummy"] = vm["rsid"]
    cs_col = "_cs_dummy"
df_cs = vm[["rsid", cs_col]].drop_duplicates(subset=["rsid"])
df = df.merge(clu, on="rsid", how="inner").merge(df_cs, on="rsid", how="left")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
df = df.dropna(subset=["cluster","AF_AFR_derived",cs_col])
log(f"  variants: {len(df):,}; unique CS: {df[cs_col].nunique():,}")

# Group variants by CS
cs_groups = {cs: g for cs, g in df.groupby(cs_col)}
cs_ids = np.array(list(cs_groups.keys()))
log(f"  block-bootstrap unit: credible-set ({len(cs_ids):,} blocks)")

N_BOOT = 1000
log(f"\n[1] Running {N_BOOT}-iteration block bootstrap...")

ks_boot = np.empty(N_BOOT)
med_boot = np.empty(N_BOOT)
for i in range(N_BOOT):
    # Sample CS IDs with replacement
    rng = np.random.default_rng(i)
    sample_cs = rng.choice(cs_ids, size=len(cs_ids), replace=True)
    # Concatenate variants from sampled CS
    boot_df = pd.concat([cs_groups[c] for c in sample_cs], ignore_index=True)
    a = boot_df[boot_df["cluster"] == 0]["AF_AFR_derived"].values
    b = boot_df[boot_df["cluster"] == 2]["AF_AFR_derived"].values
    if len(a) < 5 or len(b) < 5:
        ks_boot[i] = np.nan; med_boot[i] = np.nan; continue
    D, _ = stats.ks_2samp(a, b)
    ks_boot[i] = D
    med_boot[i] = np.median(a) - np.median(b)

ks_boot = ks_boot[~np.isnan(ks_boot)]
med_boot = med_boot[~np.isnan(med_boot)]
log(f"  successful bootstrap samples: {len(ks_boot)}/{N_BOOT}")

# Compute observed
a = df[df["cluster"] == 0]["AF_AFR_derived"].values
b = df[df["cluster"] == 2]["AF_AFR_derived"].values
D_obs, _ = stats.ks_2samp(a, b)
med_obs = np.median(a) - np.median(b)

results = [{
    "statistic": "KS_D_C0_vs_C2",
    "observed": float(D_obs),
    "boot_mean": float(np.mean(ks_boot)),
    "boot_sd": float(np.std(ks_boot)),
    "boot_CI_2.5": float(np.quantile(ks_boot, 0.025)),
    "boot_CI_97.5": float(np.quantile(ks_boot, 0.975)),
}, {
    "statistic": "median_diff_C0_minus_C2",
    "observed": float(med_obs),
    "boot_mean": float(np.mean(med_boot)),
    "boot_sd": float(np.std(med_boot)),
    "boot_CI_2.5": float(np.quantile(med_boot, 0.025)),
    "boot_CI_97.5": float(np.quantile(med_boot, 0.975)),
}]

res = pd.DataFrame(results)
res.to_csv(OUT / "P14p3g_bootstrap_CI.tsv", sep="\t", index=False)
log("\n" + res.to_string(index=False))

# Pre-reg verdict
ks_lo = res.iloc[0]["boot_CI_2.5"]
log(f"\n[2] Pre-registered: KS D lower 2.5% > 0.20?")
log(f"   KS D 95% CI = [{ks_lo:.4f}, {res.iloc[0]['boot_CI_97.5']:.4f}]")
if ks_lo > 0.20:
    log("   VERDICT: H4 SURVIVES bootstrap CI floor")
elif ks_lo > 0.10:
    log("   VERDICT: H4 PARTIAL — lower CI substantive but below 0.20")
else:
    log("   VERDICT: H4 weakened by bootstrap CI")

with open(OUT / "P14p3g_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p3g — Block-bootstrap CI\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")
log("\nPhase 14p3g complete.")
