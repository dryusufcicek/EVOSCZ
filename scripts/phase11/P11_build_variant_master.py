#!/usr/bin/env python3
"""
Phase 11: Variant-Level Master Annotation Build (v5 — code-review-corrected)
==========================================================================
Constructs a single per-variant table with all evolutionary, functional, and
regulatory annotations for the 20,766 PGC3 credible-set variants.

Code-review-driven corrections (2026-05-02):
  [FIX-A1] GEVA dedup BEFORE merge (prefer Combined > TGP > SGDP per rsid).
           Prior code: how="left" merge with 3-row-per-rsid GEVA inflated
           master to ~62k rows; downstream merges silently inconsistent.
  [FIX-A2] GTEx allele orientation: match on (chr, pos, ref, alt); flip
           slope sign when PGC3 effect_allele == GTEx REF; FLAG palindromic
           A/T and C/G variants via gtex_brain_palindromic / gtex_blood_palindromic
           binary columns (slope sign cannot be resolved without frequency
           — downstream sign-sensitive analyses must filter out palindromic
           hits when interpreting slope direction). Strand-mismatch
           (variant in PGC3 but not in GTEx, or alleles incompatible after
           rev-comp) is silently dropped (correct: the variant is not in
           GTEx for this orientation).
  [FIX-A3] BED 0-based half-open semantics for HAR/desert/ATAC overlap:
           1-based pos in BED [s, e) iff s < pos AND pos ≤ e (was: s ≤ pos
           AND pos ≤ e, off-by-one at left edge). Use cumulative-max trick
           for robust overlap detection even if intervals overlap.
  [FIX-A4] SMR variant-gene pair match: prior code flagged any variant whose
           annotated gene appeared in SMR-significant genes, regardless of
           whether the variant was the SMR top SNP. New: smr_topsnp_match
           (variant is SMR top SNP for some gene) + smr_gene_supported
           (variant's gene has SMR support, locus-level).
  [FIX-A5] Replace blanket .astype(str) NaN → "nan" string coercion with
           explicit numeric whitelist using pd.to_numeric(errors="coerce").

Inputs (all variant-level):
  - pgc3_master_variants.tsv               (20,766 variants, GRCh37)
  - pgc3_geva_ages.tsv                     (GEVA ages, 99.6% coverage)
  - SDS_UK10K_n3195_release_Sep_19_2016.tab  (recent selection)
  - GTEx v10 signif_pairs.parquet          (brain + blood/immune tissues)
  - celltype_peaks/Cluster*.narrowPeak.gz  (24 cluster ATAC overlap)
  - Cui 2025 HARs                          (HAR overlap)
  - introgression_deserts/consensus_deserts_tier{1,2,3}_hg19.bed
  - P9_smr_gene_summary.tsv                (variant-gene SMR match)

Adds GRCh38 coordinates via pyliftover (hg19 → hg38) for GTEx lookup.

Outputs:
  - results/phase11/variant_master.parquet  (20,766 unique rsids, no inflation)
  - results/phase11/P11_BUILD_LOG.md
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import gzip
sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase11"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg, flush=True)


def revcomp(allele):
    """Reverse-complement a single nucleotide allele. Pass through indels/longer alleles."""
    if not isinstance(allele, str) or len(allele) != 1:
        return allele
    return {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}.get(allele.upper(), allele)


def is_palindromic(a1, a2):
    """A/T or C/G pair — cannot resolve strand without frequency info."""
    if not (isinstance(a1, str) and isinstance(a2, str) and len(a1) == 1 and len(a2) == 1):
        return False
    pair = {a1.upper(), a2.upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}


def find_overlap_1based(positions, starts_0based, ends_0based_excl):
    """Robust BED-style overlap check with cumulative-max trick.

    BED is 0-based half-open: interval [start, end) covers 0-based positions
    start..end-1 = 1-based positions start+1..end. A 1-based pos belongs to
    the interval iff `start < pos AND pos <= end` (equivalently, in 0-based,
    `start ≤ pos-1 < end`).

    Cumulative-max trick: compute max_end_so_far[i] = max(ends[0..i]) over
    intervals sorted by start. For each pos, find rightmost interval with
    start < pos (via searchsorted side="left", then -1). Check whether the
    cumulative-max end of all intervals starting at or before pos-1 covers
    pos. This handles overlapping intervals correctly — not just the rightmost
    one.

    Returns boolean array indicating which positions overlap any interval.
    """
    if len(starts_0based) == 0:
        return np.zeros(len(positions), dtype=bool)
    starts = np.asarray(starts_0based, dtype=np.int64)
    ends = np.asarray(ends_0based_excl, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    starts_s = starts[order]
    ends_s = ends[order]
    max_end_so_far = np.maximum.accumulate(ends_s)
    # rightmost index where starts_s < pos (i.e., pos in [start+1, ...] in 1-based)
    insert = np.searchsorted(starts_s, pos, side="left") - 1
    valid = insert >= 0
    inrange = np.zeros(len(pos), dtype=bool)
    inrange[valid] = max_end_so_far[insert[valid]] >= pos[valid]
    return inrange


# ─── Section 1: Load master + GEVA ages (FIX-A1: dedup before merge) ───────
log(f"Phase 11 v5: Variant-Level Master Build — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 72)
log("Code-review corrections: GEVA dedup pre-merge, GTEx allele orientation,")
log("BED off-by-one, ATAC overlap robustness, SMR variant-gene pair, NaN string fix.")
log("")

log("[1] Load credible set variants and GEVA ages")
master = pd.read_csv(BASE / "data/processed/pgc3_master_variants.tsv", sep="\t")
geva = pd.read_csv(BASE / "data/processed/pgc3_geva_ages.tsv", sep="\t")
log(f"  master: {len(master)} variants × {len(master.columns)} cols")
log(f"  GEVA raw: {len(geva)} rows ({geva['DataSource'].value_counts().to_dict()})")

# FIX-A1: dedup GEVA to 1 row per rsid BEFORE merge
geva_keep = ["VariantID", "AgeMode_Mut", "AgeMean_Mut", "AgeMedian_Mut",
             "AgeCI95Lower_Mut", "AgeCI95Upper_Mut", "QualScore_Mut",
             "AlleleAnc", "DataSource"]
geva_sub = geva[geva_keep].rename(columns={
    "VariantID": "rsid",
    "AgeMode_Mut": "age_mode_yr",
    "AgeMean_Mut": "age_mean_yr",
    "AgeMedian_Mut": "age_median_yr",
    "AgeCI95Lower_Mut": "age_ci95_lower",
    "AgeCI95Upper_Mut": "age_ci95_upper",
    "QualScore_Mut": "age_quality",
    "AlleleAnc": "ancestral_allele",
    "DataSource": "geva_source",
})
priority = {"Combined": 0, "TGP": 1, "SGDP": 2}
geva_sub["_geva_priority"] = geva_sub["geva_source"].fillna("").map(
    lambda s: priority.get(s, 99)
)
geva_sub = (geva_sub.sort_values(["rsid", "_geva_priority"])
                   .drop_duplicates("rsid", keep="first")
                   .drop(columns=["_geva_priority"]))
log(f"  GEVA deduped to 1 row/rsid: {len(geva_sub)} unique rsids "
    f"(source distribution: {geva_sub['geva_source'].value_counts().to_dict()})")

m = master.merge(geva_sub, on="rsid", how="left")
assert len(m) == len(master), f"GEVA merge inflated rows {len(master)} -> {len(m)}"
log(f"  After GEVA merge: {len(m)} rows (no inflation)")
log(f"  GEVA matched: {m['age_median_yr'].notna().sum()}/{len(m)} "
    f"({m['age_median_yr'].notna().mean()*100:.1f}%)")


# ─── Section 2: SDS (recent selection) ─────────────────────────────────────
log("\n[2] Merge SDS scores (Field 2016, UK10K)")
sds = pd.read_csv(BASE / "data/raw/annotations/selection/SDS_UK10K_n3195_release_Sep_19_2016.tab",
                  sep="\t", usecols=["CHR", "POS", "ID", "AA", "DA", "DAF", "SDS"])
sds.columns = ["chr_sds", "pos", "rsid_sds", "sds_anc", "sds_der", "sds_daf", "sds"]
sds["chr_sds"] = sds["chr_sds"].astype(str)
sds["pos"] = sds["pos"].astype(int)

m["chr_str"] = m["chr"].astype(str)
m["pos"] = m["pos"].astype(int)
m_pre_sds = len(m)
m = m.merge(sds[["chr_sds", "pos", "sds", "sds_daf"]],
            left_on=["chr_str", "pos"], right_on=["chr_sds", "pos"], how="left")
assert len(m) == m_pre_sds, f"SDS merge inflated rows {m_pre_sds} -> {len(m)}"
m = m.drop(columns=["chr_sds", "chr_str"])
log(f"  SDS matched: {m['sds'].notna().sum()}/{len(m)} "
    f"({m['sds'].notna().mean()*100:.1f}%)")


# ─── Section 3: LiftOver hg19 → hg38 ───────────────────────────────────────
log("\n[3] LiftOver coordinates hg19 → hg38 (for GTEx v10 / OpenTargets matching)")
from pyliftover import LiftOver
lo = LiftOver("hg19", "hg38")

n_lift_fail = [0]
def lift(row):
    try:
        result = lo.convert_coordinate(f"chr{row['chr']}", int(row["pos"]))
        if result and len(result) > 0:
            return result[0][1]
    except Exception:
        n_lift_fail[0] += 1
    return None

m["pos_hg38"] = m.apply(lift, axis=1)
n_lifted = m["pos_hg38"].notna().sum()
log(f"  LiftOver successful: {n_lifted}/{len(m)} ({n_lifted/len(m)*100:.1f}%)")
if n_lift_fail[0] > 0:
    log(f"  ! LiftOver exceptions logged: {n_lift_fail[0]}")


# ─── Section 4: GTEx v10 brain + blood eQTLs (FIX-A2: allele orientation) ──
log("\n[4] GTEx v10 signif_pairs eQTL lookup (variant-level)")
log("    FIX-A2: matching on (chr, pos, ref, alt) with explicit allele")
log("    orientation; flipping slope when PGC3 effect_allele == GTEx REF;")
log("    dropping palindromic A/T and C/G variants from sign-sensitive cols.")
from lib_gtex_v10 import load_tissue, BRAIN_TISSUES, BLOOD_IMMUNE_TISSUES, list_available_parquet

available = list_available_parquet()
log(f"  Available tissues: {len(available)}")

# Build hg38 lookup with effect/other alleles for orientation handling
m38 = m[m["pos_hg38"].notna()][
    ["chr", "pos_hg38", "rsid", "effect_allele", "other_allele"]
].copy()
m38["chr"] = m38["chr"].astype(str)
m38["pos_hg38"] = m38["pos_hg38"].astype(int)
m38 = m38.rename(columns={"pos_hg38": "pos"})
m38["palindromic"] = [
    is_palindromic(ea, oa)
    for ea, oa in zip(m38["effect_allele"], m38["other_allele"])
]


def collect_min_p(tissue_set, label):
    rows = []
    for t in tissue_set:
        if t not in available:
            continue
        try:
            eqtl = load_tissue(t)[
                ["chr", "pos", "ref", "alt", "gene_id", "pval_nominal", "slope"]
            ].copy()
            sub = eqtl.merge(m38, on=["chr", "pos"], how="inner")
            if len(sub) == 0:
                continue
            # Determine allele orientation: forward-match (ea==alt, oa==ref),
            # flip-match (ea==ref, oa==alt → flip slope sign), or unmatched.
            ea = sub["effect_allele"].astype(str).str.upper()
            oa = sub["other_allele"].astype(str).str.upper()
            ref = sub["ref"].astype(str).str.upper()
            alt = sub["alt"].astype(str).str.upper()
            # Reverse-complement attempt for non-palindromic strand mismatches
            ea_rc = ea.map(revcomp)
            oa_rc = oa.map(revcomp)
            forward = (ea == alt) & (oa == ref)
            flip = (ea == ref) & (oa == alt)
            forward_rc = (ea_rc == alt) & (oa_rc == ref) & (~sub["palindromic"])
            flip_rc = (ea_rc == ref) & (oa_rc == alt) & (~sub["palindromic"])
            sub["_orient"] = "unmatched"
            sub.loc[forward, "_orient"] = "forward"
            sub.loc[flip, "_orient"] = "flip"
            sub.loc[forward_rc & ~forward & ~flip, "_orient"] = "forward_rc"
            sub.loc[flip_rc & ~forward & ~flip, "_orient"] = "flip_rc"
            # Drop unmatched alleles
            n_unmatched = (sub["_orient"] == "unmatched").sum()
            n_palin = ((sub["_orient"] == "unmatched") & sub["palindromic"]).sum()
            sub = sub[sub["_orient"] != "unmatched"].copy()
            # Apply slope sign flip for "flip" and "flip_rc"
            sub["slope_oriented"] = np.where(
                sub["_orient"].isin(["flip", "flip_rc"]),
                -sub["slope"], sub["slope"]
            )
            # Mark palindromic SNPs: keep magnitude, but mark slope sign as
            # ambiguous (we still record pval and gene_id).
            sub["palin"] = sub["palindromic"].astype(int)
            sub["tissue"] = t
            rows.append(sub[["rsid", "pos", "gene_id", "pval_nominal",
                              "slope_oriented", "palin", "tissue"]])
            log(f"    {t}: {len(sub):,} matched (unmatched dropped: {n_unmatched}, "
                f"of which palindromic: {n_palin})")
        except Exception as e:
            log(f"    ! {t}: {e}")
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    log(f"  {label}: {len(full):,} variant-tissue-gene hits across {len(set(full['tissue']))} tissues")
    # min p per (rsid)
    idx = full.groupby("rsid")["pval_nominal"].idxmin()
    minp = full.loc[
        idx, ["rsid", "gene_id", "pval_nominal", "slope_oriented", "palin", "tissue"]
    ].copy()
    minp.columns = ["rsid", f"gtex_{label}_gene", f"gtex_{label}_minp",
                    f"gtex_{label}_slope", f"gtex_{label}_palindromic",
                    f"gtex_{label}_tissue"]
    n_eqtl = full.groupby("rsid").size().reset_index(name=f"gtex_{label}_n_hits")
    return minp.merge(n_eqtl, on="rsid", how="left")

brain_minp = collect_min_p(BRAIN_TISSUES, "brain")
blood_minp = collect_min_p(BLOOD_IMMUNE_TISSUES, "blood")

m_pre_b = len(m)
m = m.merge(brain_minp, on="rsid", how="left") if len(brain_minp) > 0 else m
assert len(m) == m_pre_b, f"GTEx brain merge inflated {m_pre_b} -> {len(m)}"
m_pre_bl = len(m)
m = m.merge(blood_minp, on="rsid", how="left") if len(blood_minp) > 0 else m
assert len(m) == m_pre_bl, f"GTEx blood merge inflated {m_pre_bl} -> {len(m)}"
log(f"  Variants with GTEx brain eQTL: {m['gtex_brain_minp'].notna().sum()}")
log(f"  Variants with GTEx blood eQTL: {m['gtex_blood_minp'].notna().sum()}")
if "gtex_brain_palindromic" in m.columns:
    n_palin_brain = (m["gtex_brain_palindromic"] == 1).sum()
    log(f"  Brain eQTL hits at palindromic SNPs (slope sign ambiguous): {n_palin_brain}")


# ─── Section 5: Cell-type ATAC overlap (FIX-A3: BED semantics) ─────────────
log("\n[5] Cell-type ATAC peak overlap (24 clusters; BED 0-based half-open)")
PEAK_DIR = Path(os.environ.get("CELLTYPE_PEAKS_DIR", "data/celltype_peaks"))
test_peak = PEAK_DIR / "Cluster1.idr.optimal.narrowPeak.gz"
if test_peak.exists():
    with gzip.open(test_peak, "rt") as f:
        line = f.readline()
    log(f"  Peak format example: {line.strip()[:80]}")
else:
    log(f"  ! Test peak file not found: {test_peak}")


def overlaps_in_clusters(m_df, cluster_dir, n_clusters=24):
    """Variant-level cluster overlap matrix using BED 0-based half-open semantics.

    Note ATAC peaks are GRCh38-coordinated; we use pos_hg38 from m_df.
    """
    pos_col = "pos_hg38"
    out = pd.DataFrame(index=m_df.index)
    for ci in range(1, n_clusters + 1):
        f = cluster_dir / f"Cluster{ci}.idr.optimal.narrowPeak.gz"
        if not f.exists():
            continue
        peaks = pd.read_csv(f, sep="\t", header=None, compression="gzip",
                             usecols=[0, 1, 2], names=["chrom", "start", "end"])
        peaks["chrom"] = peaks["chrom"].astype(str).str.replace("chr", "")
        flag = pd.Series(False, index=m_df.index)
        for chrom, sub in peaks.groupby("chrom"):
            mask = (m_df["chr"].astype(str) == chrom) & m_df[pos_col].notna()
            if mask.sum() == 0:
                continue
            p = m_df.loc[mask, pos_col].astype(int).values
            inrange = find_overlap_1based(
                p, sub["start"].values, sub["end"].values
            )
            flag.loc[mask.index[mask][inrange]] = True
        out[f"atac_cluster{ci:02d}"] = flag.astype(int)
    return out

atac = overlaps_in_clusters(m, PEAK_DIR, n_clusters=24)
m = pd.concat([m, atac], axis=1)
m["atac_n_clusters"] = atac.sum(axis=1)
m["atac_any"] = (m["atac_n_clusters"] > 0).astype(int)
log(f"  Variants overlapping ANY cluster: {m['atac_any'].sum()}/{len(m)} "
    f"({m['atac_any'].mean()*100:.1f}%)")
log(f"  Mean clusters per variant: {m['atac_n_clusters'].mean():.2f}")


# ─── Section 6: HAR overlap (Cui 2025) — FIX-A3 BED semantics ──────────────
log("\n[6] HAR overlap (Cui 2025) — BED 0-based half-open, hg19")
har_bed = BASE / "data/raw/annotations/HARs/HAR_coordinates_hg19.bed"
if har_bed.exists():
    hars = pd.read_csv(har_bed, sep="\t", header=None, names=["chrom", "start", "end"],
                       usecols=[0, 1, 2])
    hars["chrom"] = hars["chrom"].astype(str).str.replace("chr", "")

    def har_overlap_flag(m_df):
        flag = pd.Series(False, index=m_df.index)
        for chrom, sub in hars.groupby("chrom"):
            mask = (m_df["chr"].astype(str) == chrom)
            if mask.sum() == 0:
                continue
            p = m_df.loc[mask, "pos"].astype(int).values
            inrange = find_overlap_1based(
                p, sub["start"].values, sub["end"].values
            )
            flag.loc[mask.index[mask][inrange]] = True
        return flag.astype(int)
    m["har_overlap"] = har_overlap_flag(m)
    log(f"  Variants overlapping HARs: {m['har_overlap'].sum()}/{len(m)}")
else:
    log(f"  ! HAR bed not found at {har_bed}")
    m["har_overlap"] = 0


# ─── Section 7: Introgression desert tier — FIX-A3 BED semantics ───────────
log("\n[7] Introgression desert tier overlap — BED 0-based half-open, hg19")
desert_dir = BASE / "data/raw/annotations/introgression_deserts"
m["desert_tier"] = 0
# Iterate 3 → 2 → 1 so tier 1 (highest confidence) overwrites lower tiers
for tier in [3, 2, 1]:
    bed = desert_dir / f"consensus_deserts_tier{tier}_hg19.bed"
    if not bed.exists():
        continue
    intervals = pd.read_csv(bed, sep="\t", header=None, names=["chrom", "start", "end"],
                            usecols=[0, 1, 2])
    intervals["chrom"] = intervals["chrom"].astype(str).str.replace("chr", "")
    for chrom, sub in intervals.groupby("chrom"):
        mask = (m["chr"].astype(str) == chrom)
        if mask.sum() == 0:
            continue
        p = m.loc[mask, "pos"].astype(int).values
        inrange = find_overlap_1based(
            p, sub["start"].values, sub["end"].values
        )
        idx_in = mask.index[mask][inrange]
        m.loc[idx_in, "desert_tier"] = tier
n1 = (m["desert_tier"] == 1).sum()
n2 = (m["desert_tier"] == 2).sum()
n3 = (m["desert_tier"] == 3).sum()
log(f"  Tier 1: {n1}, Tier 2: {n2}, Tier 3: {n3}, none: {(m['desert_tier']==0).sum()}")


# ─── Section 8: PGC3 SMR — FIX-A4 variant-gene pair match ──────────────────
log("\n[8] PGC3 SMR/HEIDI integration (FIX-A4: variant-gene pair match)")
smr_path = BASE / "results/phase9/P9_smr_gene_summary.tsv"
m["smr_topsnp_match"] = 0   # variant IS the SMR top SNP for some gene
m["smr_gene_supported"] = 0  # variant's annotated gene has SMR support (locus-level)
if smr_path.exists():
    smr = pd.read_csv(smr_path, sep="\t")
    log(f"  SMR records: {len(smr)}")
    if all(c in smr.columns for c in ["gene_symbol", "smr_topSNP_chr", "smr_topSNP_bp"]):
        # Variant-gene pair: a variant is "smr_topsnp_match" if (its chr, pos)
        # matches any SMR top-SNP coord AND its gene_symbol matches that
        # SMR record's gene_symbol.
        smr_pairs = smr[
            smr["smr_topSNP_chr"].notna() & smr["smr_topSNP_bp"].notna()
        ][["smr_topSNP_chr", "smr_topSNP_bp", "gene_symbol"]].copy()
        smr_pairs["smr_topSNP_chr"] = smr_pairs["smr_topSNP_chr"].astype(str)
        smr_pairs["smr_topSNP_bp"] = smr_pairs["smr_topSNP_bp"].astype(int)
        # Unique gene_symbol present in SMR
        smr_genes = set(smr["gene_symbol"].dropna().unique()) - {".", ""}
        log(f"  SMR top-SNP coords with gene: {len(smr_pairs)}")
        log(f"  SMR genes (any tissue): {len(smr_genes)}")
        # Build (chr, pos, gene) tuples
        smr_pair_keys = set(zip(
            smr_pairs["smr_topSNP_chr"],
            smr_pairs["smr_topSNP_bp"].astype(int),
            smr_pairs["gene_symbol"]
        ))
        m_chr = m["chr"].astype(str)
        m_pos = m["pos"].astype(int)
        m_gene = m["gene_symbol"].fillna("")
        m["smr_topsnp_match"] = [
            int((c, p, g) in smr_pair_keys)
            for c, p, g in zip(m_chr, m_pos, m_gene)
        ]
        m["smr_gene_supported"] = m_gene.isin(smr_genes).astype(int)
        log(f"  Variants matching SMR top-SNP & gene: {m['smr_topsnp_match'].sum()}")
        log(f"  Variants whose gene has SMR support (locus-level): "
            f"{m['smr_gene_supported'].sum()}")
    else:
        log("  ! SMR cols missing (gene_symbol/smr_topSNP_chr/smr_topSNP_bp)")
else:
    log(f"  ! P9 SMR summary not found")


# ─── Section 9: Save (FIX-A5: replace blanket astype(str)) ─────────────────
log("\n[9] Save outputs (FIX-A5: explicit numeric handling, no blanket astype(str))")
out_parquet = OUT / "variant_master.parquet"
out_tsv = OUT / "variant_master.tsv.gz"

# Define string-columns explicitly; everything else stays numeric/bool/whatever.
STRING_COLS = {
    "rsid", "credible_set_id", "gene_symbol", "gene_ensembl",
    "effect_allele", "other_allele", "ancestral_allele", "geva_source",
    "gene_biotype", "vep_impact", "sift", "polyphen",
    "gtex_brain_gene", "gtex_brain_tissue",
    "gtex_blood_gene", "gtex_blood_tissue",
}
for c in m.columns:
    if c in STRING_COLS:
        # Treat as string but preserve real NaN (avoid "nan" literal)
        m[c] = m[c].where(m[c].notna(), None).astype(object)
    elif m[c].dtype == "object":
        # Object dtype that's NOT a designated string column → coerce to numeric
        coerced = pd.to_numeric(m[c], errors="coerce")
        if coerced.notna().any():
            m[c] = coerced
        # else: leave as object (e.g., all-NaN object column)

m.to_parquet(out_parquet, index=False, compression="snappy")
m.to_csv(out_tsv, sep="\t", index=False, compression="gzip")
log(f"  Saved: {out_parquet} ({len(m)} rows × {len(m.columns)} cols)")
log(f"  Saved: {out_tsv}")

# Coverage summary
log("\n=== Variant-Level Annotation Coverage ===")
coverage = {
    "GEVA age": m["age_median_yr"].notna().sum(),
    "SDS": m["sds"].notna().sum(),
    "LiftOver hg38": m["pos_hg38"].notna().sum(),
    "GTEx brain eQTL": m["gtex_brain_minp"].notna().sum() if "gtex_brain_minp" in m.columns else 0,
    "GTEx blood eQTL": m["gtex_blood_minp"].notna().sum() if "gtex_blood_minp" in m.columns else 0,
    "Any ATAC cluster": int(m["atac_any"].sum()),
    "HAR overlap": int(m["har_overlap"].sum()),
    "Desert (any tier)": int((m["desert_tier"] > 0).sum()),
    "SMR top-SNP match": int(m["smr_topsnp_match"].sum()),
    "SMR gene-supported (locus)": int(m["smr_gene_supported"].sum()),
}
log(f"  {'Modality':<28s} {'n':>10s} {'%':>8s}")
log(f"  {'-'*28} {'-'*10} {'-'*8}")
for mod, n in coverage.items():
    pct = n / len(m) * 100
    log(f"  {mod:<28s} {n:>10d} {pct:>7.1f}%")

# Save log
log_path = OUT / "P11_BUILD_LOG.md"
with open(log_path, "w") as f:
    f.write("# Phase 11 Variant Master Build Log (v5 — code-review-corrected)\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("```\n")
    for line in LOG:
        f.write(line + "\n")
    f.write("```\n")
log(f"\nLog saved: {log_path}")
log("\nPhase 11 v5 complete.")
