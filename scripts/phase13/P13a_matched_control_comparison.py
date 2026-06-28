#!/usr/bin/env python3
"""
Phase 13a v2: PGC3 vs Matched-Control Comparison (code-review-corrected)
========================================================================
Code-review Faz D corrections:
  [FIX-D-13a-1] Pooled (not paired) test → true paired Wilcoxon. Prior code
    sampled 1 random control per PGC3 variant and computed pooled within-locus
    Spearman for each arm, then took the unpaired difference. New protocol:
    for each credible-set / matched-locus pair, compute within-locus partial
    rank correlation for PGC3 variants (`credible_set_id`) and for the
    matched controls assigned to that credible set (`pgc3_credset`).
    Then run a Wilcoxon signed-rank test on the paired per-locus Δρ values.
  [FIX-D-13a-2] GEVA lookup uses (chr, pos), not rsID alone. HapMap matched
    controls' rsIDs may not match GEVA atlas variant IDs; chr/pos join is
    schema-symmetric.
  [FIX-D-13a-3] Within-locus residualization uses _within_locus_lib's rank-
    based partial Spearman (consistent with Faz C corrections in P12g/h, P13b/d).

Output:
  - results/phase13/P13a_matched_control_comparison.tsv (per-test paired stats)
  - results/phase13/P13a_per_locus_rho.tsv (per-locus PGC3 vs control ρ)
  - results/phase13/P13a_ANALYSIS_LOG.md
"""

import sys
from pathlib import Path
import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
P11 = BASE / "results/phase11"
OUT = BASE / "results/phase13"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts/phase12"))
from _within_locus_lib import within_locus_partial_rank_correlation

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


def per_locus_partial_rank(df, x, y, locus_col, min_n=5, maf_col=None):
    """Compute per-locus partial rank correlation. Returns DataFrame
    locus_col, rho, n."""
    rows = []
    sub = df[[x, y, locus_col] + ([maf_col] if maf_col else [])].dropna().copy()
    for g, grp in sub.groupby(locus_col):
        if len(grp) < min_n:
            continue
        if grp[x].nunique() < 2 or grp[y].nunique() < 2:
            continue
        # use the lib in single-group mode (rank within, center, Pearson — but
        # for a single group, this is identical to in-group Spearman)
        x_r = stats.rankdata(grp[x].values, method="average")
        y_r = stats.rankdata(grp[y].values, method="average")
        if maf_col is not None and grp[maf_col].nunique() >= 2:
            m_r = stats.rankdata(grp[maf_col].values, method="average")
            mc = m_r - m_r.mean()
            denom = float((mc**2).sum())
            if denom > 0:
                bx = ((x_r - x_r.mean()) * mc).sum() / denom
                by = ((y_r - y_r.mean()) * mc).sum() / denom
                xr = (x_r - x_r.mean()) - bx * mc
                yr = (y_r - y_r.mean()) - by * mc
            else:
                xr = x_r - x_r.mean()
                yr = y_r - y_r.mean()
        else:
            xr = x_r - x_r.mean()
            yr = y_r - y_r.mean()
        if xr.std() == 0 or yr.std() == 0:
            continue
        rho, _ = stats.pearsonr(xr, yr)
        rows.append({locus_col: g, "rho": float(rho), "n": int(len(grp))})
    return pd.DataFrame(rows)


def main():
    log(f"Phase 13a v2: PGC3 vs Matched Controls (paired) — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("=" * 72)

    # ── Load PGC3 master + matched controls ──
    pgc3 = pd.read_parquet(P11 / "variant_master_v2.parquet")
    log(f"PGC3 variants: {len(pgc3)}")
    controls = pd.read_csv(BASE / "data/processed/matched_controls.tsv.gz", sep="\t")
    log(f"Matched control mappings: {len(controls):,}")
    log(f"Unique control SNPs: {controls['control_snp'].nunique():,}")
    log(f"Mean controls per PGC3 variant: "
        f"{len(controls) / controls['pgc3_rsid'].nunique():.1f}")

    # ── Annotate controls with GEVA age via (chr, pos) — FIX-D-13a-2 ──
    log("\n[Annotate controls with GEVA age via (chr, pos) join]")
    geva_dir = BASE / "data/raw/annotations/allele_ages/geva/atlas_bulk"
    control_chrs = sorted(controls["control_chr"].astype(str).unique(),
                          key=lambda x: int(x) if str(x).isdigit() else 99)
    geva_by_chrpos = {}
    for chrom in control_chrs:
        f = geva_dir / f"atlas.chr{chrom}.csv.gz"
        if not f.exists():
            log(f"  GEVA chr{chrom} missing")
            continue
        try:
            g = pd.read_csv(f, comment="#", compression="gzip", skipinitialspace=True,
                             usecols=["VariantID", "Chromosome", "Position",
                                      "AgeMedian_Mut", "DataSource"])
        except Exception:
            g = pd.read_csv(f, sep=",", skiprows=3, compression="gzip",
                             skipinitialspace=True,
                             usecols=["VariantID", "Chromosome", "Position",
                                      "AgeMedian_Mut", "DataSource"])
        g["_rank"] = g["DataSource"].map({"Combined": 0, "TGP": 1, "SGDP": 2}).fillna(99)
        g = g.sort_values(["Chromosome", "Position", "_rank"]).drop_duplicates(
            subset=["Chromosome", "Position"], keep="first"
        )
        for _, r in g.iterrows():
            geva_by_chrpos[(str(r["Chromosome"]), int(r["Position"]))] = r["AgeMedian_Mut"]
        log(f"  chr{chrom}: {len(g):,} GEVA entries (chr,pos keyed)")

    controls["age_median_yr"] = [
        geva_by_chrpos.get((str(c), int(p)))
        for c, p in zip(controls["control_chr"], controls["control_pos"])
    ]
    n_age = controls["age_median_yr"].notna().sum()
    log(f"  Controls with GEVA age: {n_age:,}/{len(controls):,} "
        f"({n_age/len(controls)*100:.1f}%)")

    # ── LiftOver + GTEx eQTL annotation for controls ──
    log("\n[LiftOver + GTEx v10 eQTL for matched controls]")
    from pyliftover import LiftOver
    lo = LiftOver("hg19", "hg38")

    def lift(c, p):
        try:
            r = lo.convert_coordinate(f"chr{c}", int(p))
            if r and len(r) > 0:
                return r[0][1]
        except Exception:
            pass
        return None
    controls["pos_hg38"] = [lift(c, p) for c, p in
                              zip(controls["control_chr"], controls["control_pos"])]
    log(f"  Lifted: {controls['pos_hg38'].notna().sum():,}/{len(controls):,}")

    sys.path.insert(0, str(BASE / "scripts/phase11"))
    from lib_gtex_v10 import load_tissue, BRAIN_TISSUES, BLOOD_IMMUNE_TISSUES, list_available_parquet
    available = list_available_parquet()

    def collect_min_p(df_q, tissue_set, label):
        d = df_q[["control_snp", "control_chr", "pos_hg38"]].copy()
        d.columns = ["rsid", "chr", "pos"]
        d["chr"] = d["chr"].astype(str)
        d = d[d["pos"].notna()]
        d["pos"] = d["pos"].astype(int)
        rows = []
        for t in tissue_set:
            if t not in available:
                continue
            try:
                eqtl = load_tissue(t)[["chr", "pos", "gene_id", "pval_nominal"]]
                sub = eqtl.merge(d, on=["chr", "pos"], how="inner")
                rows.append(sub)
            except Exception:
                pass
        if not rows:
            return pd.DataFrame()
        full = pd.concat(rows, ignore_index=True)
        idx = full.groupby("rsid")["pval_nominal"].idxmin()
        minp = full.loc[idx, ["rsid", "pval_nominal"]].copy()
        minp.columns = ["control_snp", f"gtex_{label}_minp"]
        return minp

    brain_minp = collect_min_p(controls, BRAIN_TISSUES, "brain")
    blood_minp = collect_min_p(controls, BLOOD_IMMUNE_TISSUES, "blood")
    if len(brain_minp):
        controls = controls.merge(brain_minp, on="control_snp", how="left")
    if len(blood_minp):
        controls = controls.merge(blood_minp, on="control_snp", how="left")
    n_b = controls.get("gtex_brain_minp", pd.Series(dtype=float)).notna().sum()
    n_bl = controls.get("gtex_blood_minp", pd.Series(dtype=float)).notna().sum()
    log(f"  Controls with brain eQTL: {n_b:,}")
    log(f"  Controls with blood eQTL: {n_bl:,}")

    # ── Build PGC3 + control test frames ──
    pgc3_age = pgc3[pgc3["age_median_yr"].notna() & (pgc3["age_median_yr"] > 0)].copy()
    pgc3_age["log_age"] = np.log10(pgc3_age["age_median_yr"])
    pgc3_age["b_logp"] = np.where(
        pgc3_age["gtex_brain_minp"].notna() & (pgc3_age["gtex_brain_minp"] > 0),
        -np.log10(pgc3_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan
    )
    pgc3_age["bl_logp"] = np.where(
        pgc3_age["gtex_blood_minp"].notna() & (pgc3_age["gtex_blood_minp"] > 0),
        -np.log10(pgc3_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan
    )
    pgc3_age["brain_spec"] = pgc3_age["b_logp"] / (pgc3_age["b_logp"] + pgc3_age["bl_logp"])

    ctrl_age = controls[controls["age_median_yr"].notna()
                          & (controls["age_median_yr"] > 0)].copy()
    ctrl_age["log_age"] = np.log10(ctrl_age["age_median_yr"])
    if "gtex_brain_minp" in ctrl_age.columns:
        ctrl_age["b_logp"] = np.where(
            ctrl_age["gtex_brain_minp"].notna() & (ctrl_age["gtex_brain_minp"] > 0),
            -np.log10(ctrl_age["gtex_brain_minp"].clip(lower=1e-300)), np.nan
        )
    else:
        ctrl_age["b_logp"] = np.nan
    if "gtex_blood_minp" in ctrl_age.columns:
        ctrl_age["bl_logp"] = np.where(
            ctrl_age["gtex_blood_minp"].notna() & (ctrl_age["gtex_blood_minp"] > 0),
            -np.log10(ctrl_age["gtex_blood_minp"].clip(lower=1e-300)), np.nan
        )
    else:
        ctrl_age["bl_logp"] = np.nan
    ctrl_age["brain_spec"] = ctrl_age["b_logp"] / (ctrl_age["b_logp"] + ctrl_age["bl_logp"])
    log(f"\n  PGC3:    n_age={len(pgc3_age)}, n_loci={pgc3_age['credible_set_id'].nunique()}")
    log(f"  Control: n_age={len(ctrl_age)}, n_loci={ctrl_age['pgc3_credset'].nunique()}")

    # ── Per-locus rho + paired Wilcoxon ──
    log("\n[Per-locus partial rank correlation, paired Wilcoxon]")
    paired_results = []
    per_locus_outputs = []
    for label, x, y in [
        ("Brain spec × age", "log_age", "brain_spec"),
        ("Brain eQTL × age", "log_age", "b_logp"),
        ("Blood eQTL × age", "log_age", "bl_logp"),
    ]:
        # Per-locus rho
        pgc_pl = per_locus_partial_rank(pgc3_age, x, y, "credible_set_id", min_n=5)
        ctl_pl = per_locus_partial_rank(ctrl_age, x, y, "pgc3_credset", min_n=5)
        if pgc_pl.empty or ctl_pl.empty:
            log(f"  {label}: insufficient data")
            continue
        # Pair on locus id
        pair = pgc_pl.rename(
            columns={"credible_set_id": "locus", "rho": "rho_pgc", "n": "n_pgc"}
        ).merge(
            ctl_pl.rename(columns={"pgc3_credset": "locus", "rho": "rho_ctl", "n": "n_ctl"}),
            on="locus", how="inner"
        )
        if len(pair) < 5:
            log(f"  {label}: only {len(pair)} paired loci — skipping")
            continue
        pair["d_rho"] = pair["rho_pgc"] - pair["rho_ctl"]
        pair["test"] = label
        per_locus_outputs.append(pair)
        try:
            stat, pval = stats.wilcoxon(pair["rho_pgc"], pair["rho_ctl"], zero_method="wilcox")
        except Exception as e:
            log(f"  {label}: wilcoxon failed: {e}")
            continue
        # Pooled rho (within-locus partial rank, for reference)
        pgc_all = within_locus_partial_rank_correlation(
            pgc3_age, x, y, "credible_set_id", maf_col=None, min_n=5
        )
        ctl_all = within_locus_partial_rank_correlation(
            ctrl_age, x, y, "pgc3_credset", maf_col=None, min_n=5
        )
        rho_pgc_p = pgc_all["rho"] if pgc_all else np.nan
        rho_ctl_p = ctl_all["rho"] if ctl_all else np.nan
        paired_results.append({
            "test": label,
            "n_loci_paired": len(pair),
            "median_rho_pgc": float(pair["rho_pgc"].median()),
            "median_rho_ctl": float(pair["rho_ctl"].median()),
            "median_d_rho": float(pair["d_rho"].median()),
            "wilcoxon_W": float(stat), "wilcoxon_p": float(pval),
            "pooled_rho_pgc": rho_pgc_p, "pooled_rho_ctl": rho_ctl_p,
            "pooled_d_rho": (rho_pgc_p - rho_ctl_p) if (
                pgc_all and ctl_all
            ) else np.nan,
        })
        log(f"  {label}:")
        log(f"    n_loci_paired={len(pair)}, "
            f"median ρ_PGC3={pair['rho_pgc'].median():.4f}, "
            f"median ρ_CTL={pair['rho_ctl'].median():.4f}, "
            f"median Δρ={pair['d_rho'].median():+.4f}, "
            f"Wilcoxon p={pval:.3e}")
        log(f"    pooled ρ_PGC3={rho_pgc_p}, pooled ρ_CTL={rho_ctl_p}")

    df_pair = pd.DataFrame(paired_results)
    df_pair.to_csv(OUT / "P13a_matched_control_comparison.tsv", sep="\t", index=False)
    log(f"\n  Saved: {OUT / 'P13a_matched_control_comparison.tsv'}")
    if per_locus_outputs:
        per_locus_full = pd.concat(per_locus_outputs, ignore_index=True)
        per_locus_full.to_csv(OUT / "P13a_per_locus_rho.tsv", sep="\t", index=False)
        log(f"  Saved per-locus pairs: {OUT / 'P13a_per_locus_rho.tsv'}")

    with open(OUT / "P13a_ANALYSIS_LOG.md", "w") as f:
        f.write("# Phase 13a v2: Matched Control Paired Comparison\n\n```\n")
        for line in LOG:
            f.write(line + "\n")
        f.write("```\n")
    log("\nPhase 13a v2 complete.")


if __name__ == "__main__":
    main()
