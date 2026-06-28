#!/usr/bin/env python3
"""Phase 17A: MAPT-included partitioned LDSC sensitivity.

Robustness check #6: 17q21.31 MAPT inversion (1,742 credible-set variants in
CS_224) was excluded from primary GMM clustering. We ask: does the
Young-cluster heritability concentration persist when MAPT is included?

Strategy (predict-only, preserves primary cluster definition):
  1. Re-fit the primary 3D GMM (k=3, log_age × brain_spec × abs_ihs, no MAPT)
     using the same hyper-parameters as P14b_v3_mixture_clusters.py
     (random_state=42, n_init=10, full covariance) and the same StandardScaler.
  2. Predict cluster labels for the 1,568 MAPT variants with all 3 features
     using the SAME GMM model and SAME scaler (no model re-fit). This treats
     MAPT variants as held-out predictions, not as new training points.
  3. Combine non-MAPT cluster assignments (from P14b_v3) with new MAPT
     predictions → cluster_assignments_mapt_included.tsv.gz.
  4. Build new cluster annotation files (per-chromosome) that include MAPT
     variants in C0/C1/C2 on chr17.

Step 5-6 (LD score recomputation on chr17 + S-LDSC) handled in shell wrapper.

Output:
  results/phase17a/cluster_assignments_mapt_included.tsv.gz
  results/phase17a/P17a_mapt_assignment_log.md
  results/phase17a/cluster_annot_mapt/cluster.{1..22}.annot.gz
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
P14B = BASE / "results/phase14b"
P14E_BASELINE = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"
OUT = BASE / "results/phase17a"
OUT_ANNOT = OUT / "cluster_annot_mapt"
OUT.mkdir(parents=True, exist_ok=True)
OUT_ANNOT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 17A: MAPT-included cluster assignment — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)

np.random.seed(42)

# ─── 1. LOAD MASTER VARIANT TABLE + RECOMPUTE FEATURES (identical to P14b) ──
log("\n[1] Loading variant master + recomputing features (identical to P14b)")
m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["b_logp"]  = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                          -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                          -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["chr_int"] = pd.to_numeric(m["chr"], errors="coerce")
m["pos_int"] = pd.to_numeric(m["pos"], errors="coerce")

in_mapt = (m["chr_int"] == 17) & (m["pos_int"] >= 43_000_000) & (m["pos_int"] <= 46_000_000)
log(f"  Total variants: {len(m):,}")
log(f"  MAPT (17q21.31) variants: {in_mapt.sum():,}")

# Non-MAPT training set (identical to P14b_v3)
m_nm = m[~in_mapt].copy()
mix_data_nm = m_nm[["log_age", "brain_spec", "abs_ihs"]].dropna()
log(f"  Non-MAPT with all 3 features (training): {len(mix_data_nm):,}")

# MAPT held-out set
m_mapt = m[in_mapt].copy()
mix_data_mapt = m_mapt[["log_age", "brain_spec", "abs_ihs"]].dropna()
log(f"  MAPT with all 3 features (held-out): {len(mix_data_mapt):,}")


# ─── 2. RE-FIT PRIMARY GMM (identical setup to P14b_v3) ────────────────────
log("\n[2] Re-fitting primary GMM (k=3, n_init=10, random_state=42)")
scaler = StandardScaler()
mix_z_nm = scaler.fit_transform(mix_data_nm.values)
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=10)
gmm.fit(mix_z_nm)

# Sort components by mean log_age (identical to P14b)
unscaled_means = scaler.inverse_transform(gmm.means_)
sort_idx = np.argsort(unscaled_means[:, 0])
remap = {old: new for new, old in enumerate(sort_idx)}

log(f"  GMM converged: {gmm.converged_}")
log(f"  Component centroids (sorted by log_age, unscaled):")
for new_idx, old_idx in enumerate(sort_idx):
    centroid = unscaled_means[old_idx]
    age_yr = 10 ** centroid[0]
    log(f"    C{new_idx}: log_age={centroid[0]:.3f} ({age_yr:.0f} yr), "
        f"brain_spec={centroid[1]:.3f}, |iHS|={centroid[2]:.3f}")

# Sanity check: predict non-MAPT and verify match with P14b_v3 assignments
nm_preds_raw = gmm.predict(mix_z_nm)
nm_preds_sorted = np.array([remap[l] for l in nm_preds_raw])
existing_assign = pd.read_csv(P14B / "P14b_v3_cluster_assignments.tsv.gz", sep="\t")
existing_assign = existing_assign[existing_assign["cluster"].notna()].copy()
existing_assign["cluster"] = existing_assign["cluster"].astype(int)
existing_lookup = dict(zip(existing_assign["rsid"].astype(str), existing_assign["cluster"]))

n_compared = 0
n_match = 0
nm_rsids = m_nm.loc[mix_data_nm.index, "rsid"].astype(str).values
for rsid, pred in zip(nm_rsids, nm_preds_sorted):
    if rsid in existing_lookup:
        n_compared += 1
        if existing_lookup[rsid] == pred:
            n_match += 1
log(f"\n  Reproducibility check vs P14b_v3 assignments:")
log(f"    Compared: {n_compared:,} / Match: {n_match:,} ({100*n_match/n_compared:.2f}%)")
if n_match / n_compared < 0.999:
    log(f"    !!! Reproducibility failure — investigate.")
    raise RuntimeError("GMM predictions differ from P14b_v3 assignments")
else:
    log(f"    GMM identical to P14b_v3 reference.")


# ─── 3. PREDICT MAPT CLUSTER LABELS (held-out predictions) ─────────────────
log("\n[3] Predicting cluster labels for MAPT variants (held-out)")
mix_z_mapt = scaler.transform(mix_data_mapt.values)
mapt_preds_raw = gmm.predict(mix_z_mapt)
mapt_preds_sorted = np.array([remap[l] for l in mapt_preds_raw])

mapt_assign = m_mapt.loc[mix_data_mapt.index, ["rsid", "credible_set_id", "chr", "pos", "gene_symbol",
                                                  "age_median_yr", "brain_spec", "abs_ihs"]].copy()
mapt_assign["cluster"] = mapt_preds_sorted

log(f"  MAPT cluster distribution:")
for c in range(3):
    n = int((mapt_preds_sorted == c).sum())
    log(f"    C{c}: {n:,}")


# ─── 4. COMBINE NON-MAPT + MAPT ASSIGNMENTS ────────────────────────────────
log("\n[4] Combining non-MAPT + MAPT cluster assignments")
nm_assign = existing_assign[["rsid", "credible_set_id", "chr", "pos", "gene_symbol",
                              "age_median_yr", "brain_spec", "abs_ihs", "cluster"]].copy()
nm_assign["mapt_origin"] = False
mapt_assign["mapt_origin"] = True

# Ensure column types align (chr/pos sometimes mixed)
for col in ["chr", "pos"]:
    nm_assign[col] = pd.to_numeric(nm_assign[col], errors="coerce")
    mapt_assign[col] = pd.to_numeric(mapt_assign[col], errors="coerce")

combined = pd.concat([nm_assign, mapt_assign], ignore_index=True)
combined["cluster"] = combined["cluster"].astype(int)
log(f"  Combined assignments: {len(combined):,}")
log(f"    of which MAPT: {combined['mapt_origin'].sum():,}")
log(f"  Combined cluster sizes:")
for c in range(3):
    nm_n = ((combined['cluster'] == c) & (~combined['mapt_origin'])).sum()
    mapt_n = ((combined['cluster'] == c) & (combined['mapt_origin'])).sum()
    log(f"    C{c}: total={(combined['cluster']==c).sum():,}  (non-MAPT {nm_n:,} + MAPT {mapt_n:,})")

combined.to_csv(OUT / "cluster_assignments_mapt_included.tsv.gz",
                sep="\t", index=False, compression="gzip")
log(f"\n  Saved: {OUT / 'cluster_assignments_mapt_included.tsv.gz'}")


# ─── 5. BUILD MAPT-INCLUDED CLUSTER ANNOTATION FILES ───────────────────────
log("\n[5] Building per-chromosome MAPT-included annotation files")
rsid2cluster = dict(zip(combined["rsid"].astype(str), combined["cluster"]))

for chrom in range(1, 23):
    base_path = P14E_BASELINE / f"baseline.{chrom}.annot.gz"
    if not base_path.exists():
        log(f"  ! chr{chrom}: baseline annot missing — skipping")
        continue
    base = pd.read_csv(base_path, sep="\t", compression="gzip",
                       usecols=["CHR", "BP", "SNP", "CM"])
    base["C0"] = 0
    base["C1"] = 0
    base["C2"] = 0
    snps = base["SNP"].astype(str).values
    for i, snp in enumerate(snps):
        c = rsid2cluster.get(snp)
        if c == 0:
            base.iat[i, base.columns.get_loc("C0")] = 1
        elif c == 1:
            base.iat[i, base.columns.get_loc("C1")] = 1
        elif c == 2:
            base.iat[i, base.columns.get_loc("C2")] = 1
    out_path = OUT_ANNOT / f"cluster.{chrom}.annot.gz"
    base.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n0 = int(base["C0"].sum())
    n1 = int(base["C1"].sum())
    n2 = int(base["C2"].sum())
    log(f"  chr{chrom}: SNPs={len(base):,}, C0={n0}, C1={n1}, C2={n2}")


# ─── 6. SAVE LOG ───────────────────────────────────────────────────────────
log(f"\nDone. Cluster annot files in {OUT_ANNOT}")
log(f"Next: compute LD scores per-chrom, then run partitioned LDSC.")

with open(OUT / "P17a_mapt_assignment_log.md", "w") as f:
    f.write("# Phase 17A: MAPT-included cluster assignment\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")

log(f"\nLog saved: {OUT / 'P17a_mapt_assignment_log.md'}")
