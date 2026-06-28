#!/usr/bin/env python3
"""
T1_subset_build.py — decisive concern-4 test: does C0 enrich BEYOND the whole
brain-eQTL credible-set subset (the GMM input)?

Builds a 2-column annotation aligned to baseline-LD v2.2 (reusing the published
phase14e cluster annot SNP rows):
  subset = C0|C1|C2  (the whole eQTL-complete GMM-input subset, 4918 variants)
  C0     = the Young cluster

S-LDSC [baseline + subset + C0] then gives:
  - subset coefficient -> is the eQTL-credible-set substrate enriched beyond baseline?
  - C0 coefficient (conditional on subset) -> does YOUNG add beyond the eQTL substrate?
"""
import pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

CL = Path((_ROOT + "/results/phase14e/cluster_annot"))
OUT = Path((_SCRATCH + "/test1_age_conditioning/recluster_annot/subset_c0"))
OUT.mkdir(parents=True, exist_ok=True)
ns = nc = 0
for c in range(1, 23):
    a = pd.read_csv(CL / f"cluster.{c}.annot.gz", sep="\t")   # CHR BP SNP CM C0 C1 C2
    a["subset"] = ((a["C0"] + a["C1"] + a["C2"]) > 0).astype(int)
    a[["CHR", "BP", "SNP", "CM", "subset", "C0"]].to_csv(
        OUT / f"cluster.{c}.annot.gz", sep="\t", index=False, compression="gzip")
    ns += int(a["subset"].sum()); nc += int(a["C0"].sum())
print(f"subset_c0 built: subset={ns} SNPs, C0={nc} SNPs across 22 chr", flush=True)
print("SUBSET_BUILD_DONE", flush=True)
