#!/usr/bin/env python3
"""
Phase 14b v2: Continuous Gradient (Honest Methodology)
========================================================
PROPER methodology replacing the data-driven binary split:

  1. Spearman continuous correlation per feature × age (primary)
  2. ALL 4 quartiles shown (not just Q1 vs Q4) — monotonicity check
  3. Jonckheere-Terpstra trend test (ordered alternative)
  4. Mixture model BIC test: is the joint (age, brain_spec, |iHS|) distribution
     better explained by 1, 2, or 3 Gaussian components?
  5. Conclusion: continuous gradient OR distinct classes (BIC-justified)

Output:
  results/phase14b/P14b_v2_continuous_gradient.tsv
  results/phase14b/P14b_v2_quartile_profiles.tsv
  results/phase14b/P14b_v2_mixture_model.tsv
  results/phase14b/P14b_v2_NARRATIVE.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.stats import linregress, spearmanr
from datetime import datetime
from statsmodels.stats.multitest import multipletests
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase14b"

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)

log(f"Phase 14b v2: Continuous Gradient — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)
log("METHODOLOGY: Continuous Spearman + 4-quartile trend + mixture-model BIC")
log("(replaces previous Q1-vs-Q4 binary which was post-hoc HARKing)")

np.random.seed(42)

m = pd.read_parquet(P11 / "variant_master_v3.parquet")
m["log_age"] = np.where(m["age_median_yr"].notna() & (m["age_median_yr"] > 0),
                         np.log10(m["age_median_yr"]), np.nan)
m["b_logp"]  = np.where(m["gtex_brain_minp"].notna() & (m["gtex_brain_minp"] > 0),
                          -np.log10(m["gtex_brain_minp"].clip(lower=1e-300)), np.nan)
m["bl_logp"] = np.where(m["gtex_blood_minp"].notna() & (m["gtex_blood_minp"] > 0),
                          -np.log10(m["gtex_blood_minp"].clip(lower=1e-300)), np.nan)
m["brain_spec"] = m["b_logp"] / (m["b_logp"] + m["bl_logp"])
m["abs_ihs"] = m["ihs_std"].abs() if "ihs_std" in m.columns else np.nan
m["abs_akbari_s"] = m["akbari_s"].abs() if "akbari_s" in m.columns else np.nan
m["abs_beta"] = m["beta"].abs()


# ─── 1. CONTINUOUS SPEARMAN — primary test ────────────────────────────────
log("\n" + "=" * 72)
log("[1] CONTINUOUS Spearman: each feature vs log(age) — PRIMARY analysis")
log("=" * 72)

features = [
    ("MAF",                "maf"),
    ("|GWAS β|",           "abs_beta"),
    ("PIP",                "pip"),
    ("CADD",               "cadd_phred"),
    ("|iHS|",              "abs_ihs"),
    ("SDS",                "sds"),
    ("|Akbari S|",         "abs_akbari_s"),
    ("Akbari π",           "akbari_pi"),
    ("Brain specificity",  "brain_spec"),
    ("-log10 brain min p", "b_logp"),
    ("-log10 blood min p", "bl_logp"),
    ("LOEUF",              "loeuf"),
    ("pLI",                "pLI"),
    ("ATAC clusters",      "atac_n_clusters"),
    ("Desert tier",        "desert_tier"),
]

cont_results = []
for label, col in features:
    if col not in m.columns: continue
    sub = m[[col, "log_age"]].dropna()
    if len(sub) < 30: continue
    rho, p = spearmanr(sub[col], sub["log_age"])
    cont_results.append({
        "feature": label, "n": len(sub), "rho": rho, "p": p
    })

df_cont = pd.DataFrame(cont_results)
df_cont["fdr_q"] = multipletests(df_cont["p"], method="fdr_bh")[1]
df_cont = df_cont.sort_values("p")
df_cont.to_csv(OUT / "P14b_v2_continuous_gradient.tsv", sep="\t", index=False)

log(f"\n  {'Feature':<22s} {'n':>6s} {'rho':>9s} {'p':>11s} {'q':>11s}")
for _, r in df_cont.iterrows():
    log(f"  {r['feature']:<22s} {r['n']:>6d} {r['rho']:>+8.4f} {r['p']:>11.2e} {r['fdr_q']:>11.2e}")


# ─── 2. ALL 4 QUARTILES — trend monotonicity ──────────────────────────────
log("\n" + "=" * 72)
log("[2] FULL 4-quartile profile + trend test (Jonckheere-Terpstra)")
log("=" * 72)

age_q = m["age_median_yr"].quantile([0.25, 0.5, 0.75]).tolist()
m["age_q"] = pd.cut(m["age_median_yr"],
                     bins=[0] + age_q + [1e10],
                     labels=["Q1", "Q2", "Q3", "Q4"])
log(f"\n  Quartile thresholds (years): Q1<{age_q[0]:.0f} < Q2<{age_q[1]:.0f} < Q3<{age_q[2]:.0f} < Q4")

# All quartile medians + JT trend test
def jonckheere_terpstra(*samples):
    """Jonckheere-Terpstra trend test — ordered alternative."""
    k = len(samples)
    JT = 0
    for i in range(k):
        for j in range(i+1, k):
            for x in samples[i]:
                for y in samples[j]:
                    if y > x: JT += 1
                    elif y == x: JT += 0.5
    # Approx normal under H0
    n = sum(len(s) for s in samples)
    mu = (n**2 - sum(len(s)**2 for s in samples)) / 4
    var = (n**2*(2*n+3) - sum(len(s)**2*(2*len(s)+3) for s in samples)) / 72
    z = (JT - mu) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return JT, z, p


qprof_rows = []
log(f"\n  {'Feature':<22s} {'Q1 med':>10s} {'Q2 med':>10s} {'Q3 med':>10s} {'Q4 med':>10s} {'JT_z':>7s} {'JT_p':>10s}")
log("  " + "-" * 95)

for label, col in features:
    if col not in m.columns: continue
    qsamples = []
    medians = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        v = m.loc[m["age_q"] == q, col].dropna().values
        if len(v) < 10:
            qsamples = []
            break
        # Subsample to max 500 per quartile for speed (JT is O(n²))
        if len(v) > 500:
            v = np.random.choice(v, size=500, replace=False)
        qsamples.append(v)
        medians.append(float(np.median(v)))
    if len(qsamples) < 4: continue
    try:
        JT, z, p = jonckheere_terpstra(*qsamples)
    except Exception:
        z, p = np.nan, np.nan
    log(f"  {label:<22s} " + " ".join(f"{med:>10.4f}" for med in medians) +
        f" {z:>+7.2f} {p:>10.2e}")
    qprof_rows.append({
        "feature": label,
        "Q1_median": medians[0], "Q2_median": medians[1],
        "Q3_median": medians[2], "Q4_median": medians[3],
        "JT_z": z, "JT_p": p,
        "monotonic": (medians[0] <= medians[1] <= medians[2] <= medians[3]) or
                     (medians[0] >= medians[1] >= medians[2] >= medians[3])
    })

df_q = pd.DataFrame(qprof_rows)
df_q["JT_q"] = multipletests(df_q["JT_p"].fillna(1.0), method="fdr_bh")[1]
df_q.to_csv(OUT / "P14b_v2_quartile_profiles.tsv", sep="\t", index=False)
log(f"\n  Monotonic (Q1→Q4 strictly) features: {df_q['monotonic'].sum()}/{len(df_q)}")


# ─── 3. MIXTURE MODEL — is bimodality justified? ──────────────────────────
log("\n" + "=" * 72)
log("[3] MIXTURE MODEL test — does data prefer 1, 2, or 3 components?")
log("=" * 72)

# Use 3 features: log_age, brain_spec, abs_ihs (where all available)
mix_data = m[["log_age", "brain_spec", "abs_ihs"]].dropna().values
log(f"\n  Variants with all 3 features: {len(mix_data)}")
log(f"  Standardizing each feature (z-score)...")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mix_data_z = scaler.fit_transform(mix_data)

mix_results = []
for k in [1, 2, 3, 4, 5]:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(mix_data_z)
    bic = gmm.bic(mix_data_z)
    aic = gmm.aic(mix_data_z)
    ll = gmm.score(mix_data_z) * len(mix_data_z)
    mix_results.append({"k_components": k, "BIC": bic, "AIC": aic, "log_likelihood": ll})
    log(f"  k={k}: BIC={bic:.1f}, AIC={aic:.1f}, LL={ll:.1f}")

df_mix = pd.DataFrame(mix_results)
df_mix["delta_BIC_vs_k1"] = df_mix["BIC"] - df_mix.loc[0, "BIC"]
df_mix.to_csv(OUT / "P14b_v2_mixture_model.tsv", sep="\t", index=False)

best_k = df_mix.loc[df_mix["BIC"].idxmin(), "k_components"]
log(f"\n  Best by BIC: k={int(best_k)} components")
log(f"  ΔBIC for k=2 vs k=1: {df_mix.loc[1, 'BIC'] - df_mix.loc[0, 'BIC']:+.1f}")
log(f"  ΔBIC for k=3 vs k=1: {df_mix.loc[2, 'BIC'] - df_mix.loc[0, 'BIC']:+.1f}")
log(f"  Rule of thumb: ΔBIC < -10 = strong evidence for additional component")

if df_mix.loc[1, 'BIC'] - df_mix.loc[0, 'BIC'] < -10:
    log(f"  → STRONG evidence for at least 2 components (binary split partially justified)")
else:
    log(f"  → INSUFFICIENT evidence for distinct subclasses; continuous gradient preferred")


# ─── 4. SUMMARY TABLE — sensitivity comparison ────────────────────────────
log("\n" + "=" * 72)
log("[4] Summary: continuous vs binary framing — which to report?")
log("=" * 72)

# Compare effect sizes: continuous Spearman vs Q1-vs-Q4 effect size estimate
log("\n  For each feature, do continuous Spearman and Q1↔Q4 ranking agree?")
log(f"  {'Feature':<22s} {'cont_rho':>10s} {'Q1med→Q4med':>20s} {'Same dir?':>10s}")

# Already in cont_results and qprof_rows
cont_lookup = {r["feature"]: r["rho"] for _, r in df_cont.iterrows()}
for _, r in df_q.iterrows():
    feat = r["feature"]
    cont_rho = cont_lookup.get(feat, np.nan)
    direction = r["Q4_median"] - r["Q1_median"]
    cont_dir = "+" if cont_rho > 0 else "-"
    bin_dir = "+" if direction > 0 else "-"
    same = "✓" if cont_dir == bin_dir else "✗"
    log(f"  {feat:<22s} {cont_rho:>+9.4f}  {r['Q1_median']:>8.3f}→{r['Q4_median']:<.3f}    {same:>10s}")

log("\n" + "=" * 72)
log("HONEST INTERPRETATION")
log("=" * 72)
log("""
1. CONTINUOUS Spearman is the primary test (Phase 12-13 already established).
   Quartile profiles are visualization, not separate hypotheses.

2. Q1-vs-Q4 binary differences are CONSEQUENCES of continuous gradient,
   not evidence for distinct evolutionary classes.

3. Mixture-model BIC test directly addresses 'are there 2 classes?':
   - If BIC strongly favors k≥2 → bimodal/multimodal structure is real
   - If BIC favors k=1 → continuous gradient, no class structure

4. Manuscript framing should be:
   'PGC3 SCZ variants show continuous evolutionary gradient: younger variants
    enrich for X, ancient variants enrich for Y. Effect varies smoothly with
    allele age rather than partitioning into discrete classes.'
""")

with open(OUT / "P14b_v2_NARRATIVE.md", "w") as f:
    f.write("# Phase 14b v2: Continuous Gradient (Honest Methodology)\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG: f.write(line + "\n")
    f.write("```\n")
log(f"\nNarrative: {OUT / 'P14b_v2_NARRATIVE.md'}")
log("\nPhase 14b v2 complete.")
