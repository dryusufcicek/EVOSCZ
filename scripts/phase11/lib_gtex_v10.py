"""
GTEx v10 signif_pairs Helper
=============================
Loads per-tissue signif_pairs.parquet files and creates a per-variant
lookup matrix across all available tissues.

Data location: $GTEX_V10_DIR/GTEx_Analysis_v10_eQTL_updated/
Currently extracted: 13 brain tissues. Run extract_all_tissues() to expand.
"""

import pandas as pd
import numpy as np
import tarfile
import os
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache

GTEX_DIR = Path(os.environ.get("GTEX_V10_DIR", "data/gtex_v10")) / "GTEx_Analysis_v10_eQTL_updated"
GTEX_TAR = Path(os.environ.get("GTEX_V10_DIR", "data/gtex_v10")) / "GTEx_Analysis_v10_eQTL.tar"

BRAIN_TISSUES = [
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
]

BLOOD_IMMUNE_TISSUES = [
    "Whole_Blood", "Cells_EBV-transformed_lymphocytes",
    "Spleen",
]

# Tissues that are IN the tar but might not be extracted yet
ALL_TISSUES = []  # Set after listing tar


def list_available_parquet() -> List[str]:
    """Return list of locally-extracted *.signif_pairs.parquet files."""
    return sorted([p.stem.replace(".v10.eQTLs.signif_pairs", "")
                   for p in GTEX_DIR.glob("*.signif_pairs.parquet")])


def extract_brain_blood_tissues():
    """Extract brain + blood/immune tissues from tar (idempotent)."""
    targets = []
    for t in BRAIN_TISSUES + BLOOD_IMMUNE_TISSUES:
        f = f"GTEx_Analysis_v10_eQTL_updated/{t}.v10.eQTLs.signif_pairs.parquet"
        targets.append(f)

    needed = []
    for t in targets:
        local = GTEX_DIR / Path(t).name
        if not local.exists():
            needed.append(t)

    if not needed:
        return list_available_parquet()

    print(f"Extracting {len(needed)} parquet files from tar...")
    with tarfile.open(GTEX_TAR, "r") as tf:
        for member_name in needed:
            try:
                tf.extract(member_name, path=GTEX_DIR.parent)
                print(f"  + {Path(member_name).name}")
            except Exception as e:
                print(f"  ! {member_name}: {e}")
    return list_available_parquet()


@lru_cache(maxsize=64)
def load_tissue(tissue: str) -> pd.DataFrame:
    """Load one tissue's signif_pairs parquet (cached)."""
    p = GTEX_DIR / f"{tissue}.v10.eQTLs.signif_pairs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Tissue file missing: {p}")
    df = pd.read_parquet(p)
    # variant_id format: chr1_611161_A_G_b38 → split into components
    parts = df["variant_id"].str.replace("_b38", "", regex=False).str.split("_", expand=True)
    df["chr"] = parts[0].str.replace("chr", "")
    df["pos"] = pd.to_numeric(parts[1], errors="coerce")
    df["ref"] = parts[2]
    df["alt"] = parts[3]
    df["tissue"] = tissue
    return df


def variants_in_tissue(rsid_or_chrpos: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """
    Lookup credible-set variants in a single tissue's eQTL data.
    Input df must have columns: chr, pos (GRCh38) OR variant_id_b38.
    """
    eqtl = load_tissue(tissue)
    on = ["chr", "pos"]
    return eqtl.merge(rsid_or_chrpos[on].drop_duplicates(), on=on, how="inner")


def per_variant_min_p(rsid_or_chrpos: pd.DataFrame,
                      tissues: List[str],
                      group_label: str = "tissue") -> pd.DataFrame:
    """
    For each input variant, return the min nominal p across the given tissues +
    the gene_id and slope of that minimum. Useful for 'min_p_brain' etc.

    Input df must have: chr (no chr prefix), pos (GRCh38 int).
    """
    rows = []
    for t in tissues:
        try:
            sub = variants_in_tissue(rsid_or_chrpos, t)
        except FileNotFoundError:
            continue
        if len(sub) == 0:
            continue
        rows.append(sub[["chr", "pos", "gene_id", "pval_nominal", "slope", "tissue"]])
    if not rows:
        return pd.DataFrame(columns=["chr", "pos", f"min_p_{group_label}",
                                      f"min_p_{group_label}_gene",
                                      f"n_eqtl_{group_label}"])
    full = pd.concat(rows, ignore_index=True)
    # min p per (chr, pos)
    idx = full.groupby(["chr", "pos"])["pval_nominal"].idxmin()
    minp = full.loc[idx, ["chr", "pos", "gene_id", "pval_nominal", "slope", "tissue"]]
    minp = minp.rename(columns={
        "pval_nominal": f"min_p_{group_label}",
        "gene_id": f"min_p_{group_label}_gene",
        "slope": f"min_p_{group_label}_slope",
        "tissue": f"min_p_{group_label}_tissue",
    })
    n_eqtl = full.groupby(["chr", "pos"]).size().reset_index(name=f"n_eqtl_{group_label}")
    minp = minp.merge(n_eqtl, on=["chr", "pos"], how="left")
    return minp


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- GTEx v10 Helper Self-Test ---\n")
    available = list_available_parquet()
    print(f"Locally available tissues: {len(available)}")
    print(f"  Brain: {sum(1 for t in available if 'Brain' in t)}")
    print(f"  Blood/Immune: {sum(1 for t in available if t in BLOOD_IMMUNE_TISSUES)}")
    print(f"  Other: {sum(1 for t in available if 'Brain' not in t and t not in BLOOD_IMMUNE_TISSUES)}")

    # Load Frontal Cortex BA9 and check the rs6688934 region
    df = load_tissue("Brain_Frontal_Cortex_BA9")
    print(f"\nBrain_Frontal_Cortex_BA9: {len(df):,} signif pairs")

    # Check chr1:2440958 (rs6688934 GRCh38)
    hit = df[(df["chr"] == "1") & (df["pos"] == 2440958)]
    print(f"  rs6688934 (chr1:2440958) hits: {len(hit)}")
    for _, h in hit.head(3).iterrows():
        print(f"    gene={h['gene_id']}, p={h['pval_nominal']:.2e}, slope={h['slope']:.3f}")
