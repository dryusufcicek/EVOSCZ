#!/usr/bin/env python3
"""
T1_cluster_contrast.py — formal between-cluster coefficient contrast:
a FORMAL between-cluster coefficient contrast from the joint model M0
(baseline-LD v2.2 + C0/C1/C2). "Significant vs non-significant is not a
significant difference" (Gelman & Stern) — so we test tau_Ci - tau_Cj directly
with a PAIRED block-jackknife SE (same 200 blocks, accounts for covariance).

Uses the existing M0.results (tau) + M0.part_delete (per-block tau delete-values,
already self-verified to reproduce .results tau_SE). No new model run needed.
"""
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
import numpy as np

RES = Path((_SCRATCH + "/test1_age_conditioning/results"))
M0 = "M0_baseline_cluster"
CLUST = {"C0": "C0L2_1", "C1": "C1L2_1", "C2": "C2L2_1"}


def load():
    cats, tau, tause = [], {}, {}
    with open(RES / f"{M0}.results") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            cats.append(p[0])
            tau[p[0]] = float(p[ci["Coefficient"]])
            tause[p[0]] = float(p[ci["Coefficient_std_error"]])
    arr = np.loadtxt(RES / f"{M0}.part_delete")
    return cats, tau, tause, arr


def jk_se(col, n):
    return np.sqrt((n - 1) / n * np.sum((col - col.mean()) ** 2))


def main():
    cats, tau, tause, arr = load()
    n = arr.shape[0]
    idx = {k: cats.index(v) for k, v in CLUST.items()}
    # self-check each cluster's jackknife SE vs .results
    from math import erfc, sqrt
    tau_rows = []
    for k, v in CLUST.items():
        col = arr[:, idx[k]]
        tau_rows.append((k, tau[v], tause[v], jk_se(col, n), tau[v] / tause[v]))
    pair_rows = []
    for a, b in [("C0", "C2"), ("C0", "C1"), ("C1", "C2")]:
        delta = arr[:, idx[a]] - arr[:, idx[b]]
        d = tau[CLUST[a]] - tau[CLUST[b]]
        se = jk_se(delta, n)
        z = d / se if se > 0 else float("nan")
        pair_rows.append((f"{a}-{b}", d, se, z, erfc(abs(z) / sqrt(2))))
    with open(RES / "T1_cluster_contrast_tau.tsv", "w") as f:
        f.write("cluster\ttau\ttau_SE_results\ttau_SE_jackknife\tZ\tn_blocks\n")
        for r in tau_rows:
            f.write(f"{r[0]}\t{r[1]:.6e}\t{r[2]:.6e}\t{r[3]:.6e}\t{r[4]:+.4f}\t{n}\n")
    with open(RES / "T1_cluster_contrast_pairs.tsv", "w") as f:
        f.write("contrast\tdelta_tau\tSE_delta\tZ\tP_two_sided\n")
        for r in pair_rows:
            f.write(f"{r[0]}\t{r[1]:+.6e}\t{r[2]:.6e}\t{r[3]:+.4f}\t{r[4]:.4g}\n")
    print(f"joint model {M0}; n_blocks={n}")
    for r in tau_rows:
        print(f"{r[0]}\ttau={r[1]:.4e}\tSE={r[2]:.4e}\tSE_jk={r[3]:.4e}\tZ={r[4]:+.2f}")
    for r in pair_rows:
        print(f"{r[0]}\tdelta={r[1]:+.4e}\tSE={r[2]:.4e}\tZ={r[3]:+.2f}\tP={r[4]:.3g}")
    print("\nCONTRAST_DONE")


if __name__ == "__main__":
    main()
