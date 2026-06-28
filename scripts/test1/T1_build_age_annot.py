#!/usr/bin/env python3
"""
T1_build_age_annot.py (v2 — incorporates methodology bug-review fixes)

Builds standardized continuous-age AND decile S-LDSC annotations, aligned
row-for-row to baseline-LD v2.2, for conditioning the C0 cluster on allele age.

Outputs (per chr, S-LDSC full-annot format CHR BP SNP CM <cols>):
  age_mut.{c}.annot.gz      : age_mut_z, age_present_mut          (linear Mut — M1)
  age_mut_dec.{c}.annot.gz  : age_mut_dec1..age_mut_dec10         (DECILES Mut — M1d, PRIMARY; flexible age control, Fix-2)
  age_jnt.{c}.annot.gz      : age_jnt_z, age_present_jnt          (linear Jnt — M2 robustness)

Age recipe matches the clustering (Combined>TGP>SGDP, AgeMedian_Mut, generations;
verified 20,565/20,565). Standardization & decile edges use a GLOBAL pool over all
22 chr (SNPs with a GEVA age). Missing age -> z=0, deciles all 0, present=0.

Notes:
  - Linear file keeps age_present (separates mean-imputed-missing from at-mean).
  - Decile file does NOT keep age_present: missing <=> all-10-deciles==0 (so the
    indicator would be collinear). The 'missing' stratum is captured by base=1 & all deciles 0.
  - log10 uses .mask(<=0) so non-positive ages never reach log10 (no warnings / -W error safety).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

BASE = Path(_ROOT)
BLD = BASE / "data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2"
AGET = Path((_SCRATCH + "/test1_age_conditioning/age_tables"))
OUT = Path((_SCRATCH + "/test1_age_conditioning/age_annot"))
OUT.mkdir(parents=True, exist_ok=True)
NDEC = 10


def safe_log10(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log10(s.mask(s <= 0))   # non-positive -> NaN, never fed to log10


def build():
    frames, lmut_all, ljnt_all = {}, [], []
    for c in range(1, 23):
        base = pd.read_csv(BLD / f"baselineLD.{c}.annot.gz", sep="\t",
                           usecols=["CHR", "BP", "SNP", "CM"])
        age = pd.read_csv(AGET / f"age_chr{c}.tsv", sep="\t").drop_duplicates("rsid", keep="first")
        age["log_mut"] = safe_log10(age["age_mut_gen"])
        age["log_jnt"] = safe_log10(age["age_jnt_gen"])
        m = base.merge(age[["rsid", "log_mut", "log_jnt"]], left_on="SNP",
                       right_on="rsid", how="left").drop(columns=["rsid"])
        assert len(m) == len(base), f"chr{c}: merge changed row count {len(base)}->{len(m)}"
        frames[c] = m
        lmut_all.append(m["log_mut"].dropna().values)
        ljnt_all.append(m["log_jnt"].dropna().values)

    lmut = np.concatenate(lmut_all); ljnt = np.concatenate(ljnt_all)
    mean_m, sd_m = float(lmut.mean()), float(lmut.std(ddof=0))
    mean_j, sd_j = float(ljnt.mean()), float(ljnt.std(ddof=0))
    assert sd_m > 0 and sd_j > 0, "zero sd"
    # global decile edges (9 internal quantiles -> 10 bins) on Mut log-age
    edges = np.quantile(lmut, np.linspace(0.1, 0.9, NDEC - 1))
    print(f"[global] log_mut mean={mean_m:.5f} sd={sd_m:.5f} n={lmut.size:,}", flush=True)
    print(f"[global] log_jnt mean={mean_j:.5f} sd={sd_j:.5f} n={ljnt.size:,}", flush=True)
    print(f"[global] Mut decile edges (log10 gen): {np.round(edges,4).tolist()}", flush=True)

    total = pm = pj = 0
    dec_counts = np.zeros(NDEC, dtype=int)
    for c in range(1, 23):
        m = frames[c]
        present_m = m["log_mut"].notna()
        present_j = m["log_jnt"].notna()
        # linear
        m["age_mut_z"] = ((m["log_mut"] - mean_m) / sd_m).fillna(0.0)
        m["age_jnt_z"] = ((m["log_jnt"] - mean_j) / sd_j).fillna(0.0)
        m["age_present_mut"] = present_m.astype(int)
        m["age_present_jnt"] = present_j.astype(int)
        # deciles (Mut): np.digitize -> 0..NDEC-1; one-hot; missing -> all 0
        binidx = np.digitize(m["log_mut"].values, edges)   # 0..9 for present; NaN->9 but masked next
        dec = np.zeros((len(m), NDEC), dtype=int)
        pm_idx = np.where(present_m.values)[0]
        dec[pm_idx, binidx[pm_idx]] = 1
        dec_cols = [f"age_mut_dec{i+1}" for i in range(NDEC)]
        for i, col in enumerate(dec_cols):
            m[col] = dec[:, i]
        dec_counts += dec.sum(axis=0)
        empty_dec = [i + 1 for i in range(NDEC) if dec[:, i].sum() == 0]
        if empty_dec:
            print(f"chr{c}: WARNING empty decile cols {empty_dec} (degenerate column "
                  f"can break --overlap-annot in M1d)", flush=True)

        m[["CHR", "BP", "SNP", "CM", "age_mut_z", "age_present_mut"]].to_csv(
            OUT / f"age_mut.{c}.annot.gz", sep="\t", index=False, compression="gzip")
        m[["CHR", "BP", "SNP", "CM", "age_jnt_z", "age_present_jnt"]].to_csv(
            OUT / f"age_jnt.{c}.annot.gz", sep="\t", index=False, compression="gzip")
        m[["CHR", "BP", "SNP", "CM"] + dec_cols].to_csv(
            OUT / f"age_mut_dec.{c}.annot.gz", sep="\t", index=False, compression="gzip")

        total += len(m); pm += int(present_m.sum()); pj += int(present_j.sum())
        print(f"chr{c}: {len(m):,} SNPs | mut present {int(present_m.sum()):,} "
              f"| jnt present {int(present_j.sum()):,}", flush=True)

    print(f"[coverage] total ref SNPs={total:,} | Mut present={pm:,} ({pm/total*100:.2f}%) "
          f"| Jnt present={pj:,} ({pj/total*100:.2f}%)", flush=True)
    print(f"[deciles] per-bin SNP counts (should be ~equal): {dec_counts.tolist()}", flush=True)
    with open(OUT / "standardization_params.txt", "w") as f:
        f.write(f"log_mut mean={mean_m} sd={sd_m} n={lmut.size}\n")
        f.write(f"log_jnt mean={mean_j} sd={sd_j} n={ljnt.size}\n")
        f.write(f"mut_decile_edges_log10gen={edges.tolist()}\n")
        f.write(f"total_ref_snps={total} present_mut={pm} present_jnt={pj}\n")
        f.write(f"decile_counts={dec_counts.tolist()}\n")
    print("DONE_OK", flush=True)


if __name__ == "__main__":
    build()
