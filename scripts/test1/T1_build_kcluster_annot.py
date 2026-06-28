#!/usr/bin/env python3
"""
T1_build_kcluster_annot.py — rebuild MULTI-cluster one-hot annotations for k=2 and
k=4 GMM solutions (the existing phase14b k2/k4 annots are C0-only, so a between-
cluster contrast was impossible). Clusters are RE-LABELLED by mean allele age so
C0 = youngest (matching the k=3 convention). Aligned row-for-row to baseline-LD v2.2.

Source: results/phase14b/P14n_assignments_k{k}.tsv.gz (rsid, cluster; blank=unclustered).
Output: kcluster_annot/k{k}/cluster.{chr}.annot.gz  cols CHR BP SNP CM C0..C{k-1}
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

B = Path(_ROOT)
BLD = B / "data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"
ASSIGN = B / "results/phase14b"
VM = B / "results/phase11/variant_master_v4.parquet"
OUT = Path((_SCRATCH + "/test1_age_conditioning/kcluster_annot"))


def build(k):
    a = pd.read_csv(ASSIGN / f"P14n_assignments_k{k}.tsv.gz", sep="\t", dtype={"cluster": str})
    a["cluster"] = a["cluster"].astype(str).str.strip()
    a = a[(a["cluster"] != "") & (a["cluster"].str.lower() != "nan")].copy()
    a["cluster"] = a["cluster"].astype(float).astype(int)
    assert a["cluster"].nunique() == k, f"k{k}: expected {k} clusters, got {a['cluster'].nunique()}"
    # relabel by mean age (youngest -> C0), matching the k=3 convention
    vm = pd.read_parquet(VM)[["rsid", "age_median_yr"]].drop_duplicates("rsid", keep="first")
    a = a.merge(vm, on="rsid", how="left")
    order = a.groupby("cluster")["age_median_yr"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}   # youngest cluster -> 0
    a["c"] = a["cluster"].map(remap)
    sizes = a["c"].value_counts().sort_index().to_dict()
    ages = a.groupby("c")["age_median_yr"].mean().round(0).to_dict()
    print(f"k={k}: sorted youngest->oldest C0..C{k-1} sizes={sizes} mean_age(gen)={ages}", flush=True)
    amap = dict(zip(a["rsid"], a["c"]))

    od = OUT / f"k{k}"; od.mkdir(parents=True, exist_ok=True)
    for chrom in range(1, 23):
        base = pd.read_csv(BLD / f"baselineLD.{chrom}.annot.gz", sep="\t",
                           usecols=["CHR", "BP", "SNP", "CM"])
        cl = base["SNP"].map(amap)              # NaN if SNP not a clustered credible-set variant
        for i in range(k):
            base[f"C{i}"] = (cl == i).astype(int)
        n_assigned = int((cl.notna()).sum())
        base.to_csv(od / f"cluster.{chrom}.annot.gz", sep="\t", index=False, compression="gzip")
        if chrom == 1:
            print(f"  chr1: {len(base):,} SNPs, {n_assigned} clustered; "
                  f"per-cluster on chr1: {[int(base[f'C{i}'].sum()) for i in range(k)]}", flush=True)
    print(f"k={k}: annot built -> {od}", flush=True)


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or [2, 4]
    for k in ks:
        build(k)
    print("KCLUSTER_BUILD_DONE", flush=True)
