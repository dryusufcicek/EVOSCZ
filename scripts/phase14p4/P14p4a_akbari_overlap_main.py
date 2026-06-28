#!/usr/bin/env python3
"""
Phase 14p4a — Cluster × Akbari overlap, |S| distribution, direction-of-effect
==============================================================================
Primary tests for H2:

T1. **|Akbari S| distribution per cluster** — for each credible-set variant,
    Akbari method has already given a per-variant selection coefficient s
    (variant_master_v4.akbari_s). Test:
      - mean |s| in C0 vs (C1 ∪ C2) baseline
      - permutation null: shuffle cluster labels 10,000×
      - pre-registered: |S| ratio C0/C2 > 1.5 with permutation P < 0.01

T2. **Fisher overlap with Akbari 452 (extracted from MOESM1)** —
      - 4-way contingency: cluster × (in_akbari_significant Y/N)
      - per-cluster Fisher OR vs other clusters pooled
      - per-cluster Fisher OR vs MAF-matched HapMap3 controls (deferred —
        needs control's Akbari S which is not in workspace; sensitivity flag)

T3. **Direction-of-effect (pleiotropy direction)** —
    For (PGC3 credible-set ∩ Akbari 452) variants:
      - PGC3 effect: BETA on effect_allele (A1)
      - Akbari effect: s on DERIVED allele (Anc → derived)
    Test: is SCZ-risk allele the same as Akbari-selected-FOR allele?
      - concordance < 50% → non-antagonistic pleiotropy
        (Solé-Morata 2023 direction, extended to cluster-resolved level)
      - concordance > 65% → antagonistic pleiotropy (SCZ-risk hitchhiking
        on positive selection)
      - 50-65% → balanced / inconclusive
    Binomial test against null=0.5.

T4. **HLA region sensitivity (extended MHC chr6:25-34 Mb)** — Akbari 452
    has 77 variants in chr6 (17%); HLA exclusion is critical. Re-run all
    of T1-T3 with HLA-excluded substrate; report retention.

Output:
  results/phase14p4/P14p4a_cluster_S_distribution.tsv
  results/phase14p4/P14p4a_cluster_S_permutation.tsv
  results/phase14p4/P14p4a_fisher_overlap.tsv
  results/phase14p4/P14p4a_direction_of_effect.tsv
  results/phase14p4/P14p4a_HLA_sensitivity.tsv
  results/phase14p4/P14p4a_NARRATIVE.md
"""

import sys
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

BASE = Path(_ROOT)
AUX  = Path((_SCRATCH + "/v11_data/aux"))
OUT  = BASE / "results/phase14p4"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    msg = f"[{datetime.now():%H:%M:%S}] {m}"
    LOG.append(msg); print(msg, flush=True)

log("="*70)
log("Phase 14p4a — Cluster × Akbari overlap, |S|, direction-of-effect")
log("="*70)

# ── Load substrate ─────────────────────────────────────────────────
log("[1] Loading substrate")
vm = pd.read_parquet(BASE / "results/phase11/variant_master_v4.parquet")
log(f"  variant_master_v4: {len(vm):,} rows × {vm.shape[1]} cols")

clu = pd.read_csv(BASE / "results/phase14b/P14b_v3_cluster_assignments.tsv.gz",
                   sep="\t")[["rsid","cluster"]]
df = vm.merge(clu, on="rsid", how="inner")
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
log(f"  merged with cluster: {len(df):,}")
log(f"  cluster sizes: " + ", ".join(
    f"C{int(c)}={n}" for c, n in df["cluster"].value_counts().sort_index().items()))

# Akbari 452 set (extracted from MOESM1 PDF; nominal "347 independent loci")
ak452 = pd.read_csv(AUX / "akbari_347_loci.tsv", sep="\t")
log(f"  Akbari 452 selected variants: {len(ak452):,} (from MOESM1)")
ak_rsids = set(ak452["rsid"].astype(str))
ak452["chr"] = pd.to_numeric(ak452["chr"], errors="coerce").astype("Int64")
ak452["pos"] = pd.to_numeric(ak452["pos"], errors="coerce").astype("Int64")

# Build (chr,pos) lookup for Akbari 452 for position-based overlap
ak452_pos = set(zip(ak452["chr"].astype(int), ak452["pos"].astype(int)))

# Drop variants lacking akbari_s OR cluster (no genome-wide Akbari result or no GMM cluster)
df = df.dropna(subset=["akbari_s", "cluster"]).copy()
df["abs_S"] = df["akbari_s"].abs()
df["cluster"] = df["cluster"].astype(int)
log(f"  variants with akbari_s available: {len(df):,}")

# Tag membership in Akbari 452 (rsid + chr,pos backup)
df["in_akbari452_rsid"] = df["rsid"].astype(str).isin(ak_rsids).astype(int)
df["chr_int"] = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
df["pos_int"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
df["in_akbari452_pos"] = [
    ((c, p) in ak452_pos)
    for c, p in zip(df["chr_int"].astype("Int64"), df["pos_int"].astype("Int64"))
]
df["in_akbari452"] = ((df["in_akbari452_rsid"] | df["in_akbari452_pos"].astype(int))).astype(int)
log(f"  PGC3 credible-set ∩ Akbari 452: "
    f"rsid-match={df['in_akbari452_rsid'].sum()}, "
    f"pos-match={df['in_akbari452_pos'].sum().astype(int)}, "
    f"union={df['in_akbari452'].sum()}")


# ─────────────────────────────────────────────────────────────────
# T1. |Akbari S| distribution per cluster
# ─────────────────────────────────────────────────────────────────
log("\n[2] T1 — |Akbari S| distribution per cluster")
rows = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    s = sub["abs_S"].values
    rows.append({
        "cluster": f"C{int(c)}",
        "n": len(sub),
        "mean_abs_S": float(s.mean()),
        "median_abs_S": float(np.median(s)),
        "q95_abs_S": float(np.quantile(s, 0.95)),
        "q99_abs_S": float(np.quantile(s, 0.99)),
        "pct_S_gt_0_005": float((s > 0.005).mean() * 100),
        "pct_S_gt_0_01":  float((s > 0.01).mean() * 100),
        "n_in_akbari452": int(sub["in_akbari452"].sum()),
        "pct_in_akbari452": float(sub["in_akbari452"].mean() * 100),
    })
dist_df = pd.DataFrame(rows)
dist_df.to_csv(OUT / "P14p4a_cluster_S_distribution.tsv", sep="\t", index=False)
log(dist_df.to_string(index=False))

# Pairwise S comparisons
log("\n[3] Pairwise |S| comparisons + 10k permutation null")
perm_rows = []
N_PERM = 10_000
clusters = sorted(df["cluster"].unique())
for i, c1 in enumerate(clusters):
    for c2 in clusters[i+1:]:
        x = df[df["cluster"] == c1]["abs_S"].values
        y = df[df["cluster"] == c2]["abs_S"].values
        obs_ratio = x.mean() / y.mean() if y.mean() > 0 else np.nan
        obs_diff  = x.mean() - y.mean()
        mwu_U, mwu_P = stats.mannwhitneyu(x, y, alternative="two-sided")
        # Permutation: shuffle labels of pooled (x,y)
        pooled = np.concatenate([x, y])
        n_x = len(x)
        perm = np.empty(N_PERM)
        for k in range(N_PERM):
            np.random.shuffle(pooled)
            a = pooled[:n_x]; b = pooled[n_x:]
            perm[k] = a.mean() - b.mean()
        perm_P = 2 * min((perm >= obs_diff).mean(), (perm <= obs_diff).mean())
        perm_rows.append({
            "contrast": f"C{int(c1)}_vs_C{int(c2)}",
            "n1": len(x), "n2": len(y),
            "mean_abs_S_1": float(x.mean()),
            "mean_abs_S_2": float(y.mean()),
            "ratio_1_to_2": float(obs_ratio),
            "MWU_P": float(mwu_P),
            "perm_P_diff_mean_abs_S": float(perm_P),
            "perm_diff_q975": float(np.quantile(perm, 0.975)),
            "perm_diff_q025": float(np.quantile(perm, 0.025)),
        })
perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(OUT / "P14p4a_cluster_S_permutation.tsv", sep="\t", index=False)
log(perm_df.to_string(index=False))

# C0 vs (C1+C2) combined — primary pre-registered test
c0_S = df[df["cluster"] == 0]["abs_S"].values
rest_S = df[df["cluster"].isin([1, 2])]["abs_S"].values
obs_ratio_c0_rest = c0_S.mean() / rest_S.mean()
log(f"\n  PRIMARY: C0 mean|S| / rest mean|S| = {obs_ratio_c0_rest:.3f}  "
    f"(pre-reg threshold > 1.5)")


# ─────────────────────────────────────────────────────────────────
# T2. Fisher overlap with Akbari 452
# ─────────────────────────────────────────────────────────────────
log("\n[4] T2 — Fisher overlap with Akbari 452")
fish_rows = []
for c in clusters:
    in_c = (df["cluster"] == c)
    not_c = (df["cluster"].isin([x for x in clusters if x != c]))
    a = int(((in_c)  & (df["in_akbari452"] == 1)).sum())
    b = int(((in_c)  & (df["in_akbari452"] == 0)).sum())
    cc = int(((not_c) & (df["in_akbari452"] == 1)).sum())
    d = int(((not_c) & (df["in_akbari452"] == 0)).sum())
    odds, p = stats.fisher_exact([[a, b], [cc, d]])
    fish_rows.append({
        "cluster": f"C{int(c)}",
        "n_in_cluster": int(in_c.sum()),
        "n_akbari_in_cluster": a,
        "pct_akbari_in_cluster": float(a / in_c.sum() * 100) if in_c.sum() else np.nan,
        "n_akbari_outside_cluster": cc,
        "pct_akbari_outside_cluster": float(cc / not_c.sum() * 100) if not_c.sum() else np.nan,
        "fisher_OR_vs_others": float(odds),
        "fisher_P": float(p),
    })
fish_df = pd.DataFrame(fish_rows)
fish_df.to_csv(OUT / "P14p4a_fisher_overlap.tsv", sep="\t", index=False)
log(fish_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# T3. Direction of effect — SCZ risk vs Akbari selected-FOR
# ─────────────────────────────────────────────────────────────────
log("\n[5] T3 — SCZ-risk allele vs Akbari-selected allele direction")

# Merge PGC3 credible-set ∩ Akbari 452 with full alleles
ak_for_join = ak452[["rsid","ref","alt","anc","s"]].rename(
    columns={"ref":"ak_ref","alt":"ak_alt","anc":"ak_anc","s":"ak_s"})
ovr = df[df["in_akbari452"] == 1][
    ["rsid","cluster","effect_allele","other_allele","beta","akbari_s"]
].merge(ak_for_join, on="rsid", how="inner")
log(f"  Overlap variants with PGC3 effect + Akbari direction: {len(ovr):,}")

# Determine for each variant: is SCZ-risk allele == Akbari-selected-FOR allele?
def direction(row):
    """Return concordance: SCZ-risk allele == Akbari-derived (positive-s) allele.

    Logic:
      - PGC3: 'effect_allele' is A1. beta > 0 → A1 increases SCZ.
              → SCZ-risk-allele = effect_allele (if beta>0) or other_allele (if beta<0)
      - Akbari: ancestral=ak_anc; derived=other of (ak_ref, ak_alt)
              s>0 → derived selected FOR
              → selected-FOR-allele = derived (if s>0) or ancestral (if s<0)
    Returns concordance: 1 if SCZ-risk-allele == selected-FOR-allele
    """
    ea = str(row["effect_allele"]).upper()
    oa = str(row["other_allele"]).upper()
    beta = row["beta"]
    ak_anc = str(row["ak_anc"]).upper()
    ak_ref = str(row["ak_ref"]).upper()
    ak_alt = str(row["ak_alt"]).upper()
    s = row["ak_s"]

    if pd.isna(beta) or pd.isna(s):
        return np.nan, "missing"

    # SCZ risk allele
    if beta > 0:
        risk_allele = ea
    elif beta < 0:
        risk_allele = oa
    else:
        return np.nan, "zero_beta"

    # Akbari selected-FOR allele
    # Derived allele = whichever of (ak_ref, ak_alt) is NOT ak_anc
    if ak_ref == ak_anc:
        derived = ak_alt
    elif ak_alt == ak_anc:
        derived = ak_ref
    else:
        return np.nan, "anc_mismatch"

    if s > 0:
        selected_for = derived
    elif s < 0:
        selected_for = ak_anc
    else:
        return np.nan, "zero_s"

    # Resolve allele match — check both direct + strand-complement
    comp = {"A":"T","T":"A","C":"G","G":"C"}
    risk_comp = comp.get(risk_allele, "N")
    if risk_allele == selected_for or risk_comp == selected_for:
        return 1, "concordant_SCZrisk_eq_selectedFOR"
    elif risk_allele == comp.get(selected_for, "X"):
        # Strand-ambiguous
        return np.nan, "strand_ambiguous"
    else:
        return 0, "discordant_SCZrisk_ne_selectedFOR"

if len(ovr) > 0:
    result = ovr.apply(direction, axis=1, result_type="expand")
    result.columns = ["concord", "note"]
    ovr = pd.concat([ovr, result], axis=1)
    valid = ovr.dropna(subset=["concord"])
else:
    log("  NOTE: zero variant-level overlap → direction-of-effect undefined; "
        "see P14p4b for locus-window analysis (proximity-based).")
    ovr["concord"] = np.nan
    ovr["note"] = "no_overlap"
    valid = ovr
log(f"  Valid (resolvable) overlap variants: {len(valid):,} of {len(ovr):,}")

# Per-cluster + overall direction
dir_rows = []
for c in [0, 1, 2, "all"]:
    sub = valid if c == "all" else valid[valid["cluster"] == c]
    n_total = len(sub)
    if n_total == 0:
        continue
    n_conc = int(sub["concord"].sum())
    pct_conc = n_conc / n_total
    # Binomial test against null=0.5
    binom_p = stats.binomtest(n_conc, n_total, p=0.5).pvalue
    interp = ("non-antagonistic_pleiotropy" if pct_conc < 0.50 and binom_p < 0.05 else
              "balanced" if 0.45 <= pct_conc <= 0.65 else
              "antagonistic_pleiotropy" if pct_conc > 0.65 and binom_p < 0.05 else
              "indeterminate")
    dir_rows.append({
        "cluster": f"C{c}" if c != "all" else "all_overlap",
        "n_overlap_valid": n_total,
        "n_SCZrisk_eq_selectedFOR": n_conc,
        "pct_concordance": float(pct_conc * 100),
        "binom_P_vs_50pct": float(binom_p),
        "interpretation": interp,
    })

dir_df = pd.DataFrame(dir_rows)
dir_df.to_csv(OUT / "P14p4a_direction_of_effect.tsv", sep="\t", index=False)
log(dir_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# T4. HLA exclusion sensitivity
# ─────────────────────────────────────────────────────────────────
log("\n[6] T4 — HLA exclusion sensitivity (chr6:25-34Mb)")
df["in_hla"] = ((df["chr_int"] == 6) &
                df["pos_int"].between(25_000_000, 34_000_000)).astype(int)
log(f"  HLA region credible-set variants: {df['in_hla'].sum():,}")
log(f"  HLA region Akbari overlap: "
    f"{((df['in_hla']==1) & (df['in_akbari452']==1)).sum()}")

hla_rows = []
for label, mask in [("full", slice(None)), ("no_HLA", df["in_hla"] == 0)]:
    if isinstance(mask, slice):
        sub = df
    else:
        sub = df[mask]
    c0_S2 = sub[sub["cluster"] == 0]["abs_S"].values
    rest_S2 = sub[sub["cluster"].isin([1, 2])]["abs_S"].values
    ratio = c0_S2.mean() / rest_S2.mean() if rest_S2.mean() > 0 else np.nan
    # Akbari overlap rate per cluster (after HLA exclusion)
    c0_ovr = float(sub[sub["cluster"] == 0]["in_akbari452"].mean() * 100)
    c2_ovr = float(sub[sub["cluster"] == 2]["in_akbari452"].mean() * 100)
    hla_rows.append({
        "sensitivity": label,
        "n_substrate": len(sub),
        "n_akbari452_substrate": int(sub["in_akbari452"].sum()),
        "mean_abs_S_C0": float(c0_S2.mean()),
        "mean_abs_S_rest": float(rest_S2.mean()),
        "ratio_C0_to_rest": float(ratio),
        "pct_overlap_C0": c0_ovr,
        "pct_overlap_C2": c2_ovr,
    })
hla_df = pd.DataFrame(hla_rows)
hla_df.to_csv(OUT / "P14p4a_HLA_sensitivity.tsv", sep="\t", index=False)
log(hla_df.to_string(index=False))

# Pre-reg verdict
log("\n[7] Pre-registered verdict summary")
# T1 primary gate
log(f"  T1: |S| ratio C0/rest = {obs_ratio_c0_rest:.3f}  (threshold > 1.5)")
# T2 primary gate
c0_fish_OR = fish_df[fish_df["cluster"]=="C0"]["fisher_OR_vs_others"].iloc[0]
c0_fish_P  = fish_df[fish_df["cluster"]=="C0"]["fisher_P"].iloc[0]
log(f"  T2: C0 Akbari Fisher OR = {c0_fish_OR:.3f}, P = {c0_fish_P:.3g}  (threshold OR>2, P<0.01)")
# T3 primary gate
overall_dir = dir_df[dir_df["cluster"]=="all_overlap"].iloc[0]
log(f"  T3: SCZ-risk = Akbari-selected concordance = "
    f"{overall_dir['pct_concordance']:.1f}%, P = {overall_dir['binom_P_vs_50pct']:.3g}")
log(f"      → {overall_dir['interpretation']}")
# T4 HLA
hla_full = hla_df[hla_df["sensitivity"]=="full"].iloc[0]
hla_no   = hla_df[hla_df["sensitivity"]=="no_HLA"].iloc[0]
retention = hla_no["ratio_C0_to_rest"] / hla_full["ratio_C0_to_rest"] * 100
log(f"  T4: HLA-excluded |S| ratio retention = {retention:.1f}%  (threshold ≥80%)")


# ── Narrative ──────────────────────────────────────────────────────
with open(OUT / "P14p4a_NARRATIVE.md", "w") as f:
    f.write(f"# Phase 14p4a — Cluster × Akbari 2026 selection overlap\n\n")
    f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write("## Pre-registered tests\n\n")
    f.write("- T1 |S| distribution per cluster + perm null\n")
    f.write("- T2 Fisher overlap with Akbari 452\n")
    f.write("- T3 Direction-of-effect (pleiotropy direction)\n")
    f.write("- T4 HLA exclusion sensitivity\n\n")
    f.write("## Run log\n\n```\n")
    f.write("\n".join(LOG))
    f.write("\n```\n")

log(f"\nSaved: {OUT}/P14p4a_NARRATIVE.md")
log("Phase 14p4a complete.")
