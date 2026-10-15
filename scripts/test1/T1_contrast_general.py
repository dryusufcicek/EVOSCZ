#!/usr/bin/env python3
"""
T1_contrast_general.py — formal between-cluster coefficient contrast (concern 1)
for ANY joint model, generalized over cluster count.

Usage: python3 T1_contrast_general.py <results_prefix> <n_clusters> [cluster_fileidx]
  <results_prefix>   path without extension; reads <prefix>.results + <prefix>.part_delete
  <n_clusters>       2, 3, 4 ...
  cluster_fileidx    L2_ suffix index of the cluster annotation (default 1 = baseline first, cluster second)

Computes per-cluster tau (with jackknife-SE self-check vs .results) and all
pairwise paired block-jackknife contrasts tau_i - tau_j.
"""
import sys
from pathlib import Path
from math import erfc, sqrt
import numpy as np


def main():
    prefix = sys.argv[1]
    ncl = int(sys.argv[2])
    fidx = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    clusters = {f"C{i}": f"C{i}L2_{fidx}" for i in range(ncl)}

    cats, tau, tause = [], {}, {}
    with open(prefix + ".results") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            cats.append(p[0])
            tau[p[0]] = float(p[ci["Coefficient"]])
            tause[p[0]] = float(p[ci["Coefficient_std_error"]])
    arr = np.loadtxt(prefix + ".part_delete")
    n = arr.shape[0]
    idx = {}
    for k, v in clusters.items():
        if v not in cats:
            print(f"WARNING {v} not in .results categories"); continue
        idx[k] = cats.index(v)

    def jk(col):
        return np.sqrt((n - 1) / n * np.sum((col - col.mean()) ** 2))

    tau_rows = []
    for k in clusters:
        if k not in idx:
            continue
        v = clusters[k]; col = arr[:, idx[k]]
        tau_rows.append((k, tau[v], tause[v], jk(col), tau[v] / tause[v]))
    pair_rows = []
    keys = [k for k in clusters if k in idx]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            delta = arr[:, idx[ka]] - arr[:, idx[kb]]
            d = tau[clusters[ka]] - tau[clusters[kb]]
            se = jk(delta)
            z = d / se if se > 0 else float("nan")
            pair_rows.append((f"{ka}-{kb}", d, se, z, erfc(abs(z) / sqrt(2))))
    with open(prefix + "_contrast_tau.tsv", "w") as f:
        f.write("cluster\ttau\ttau_SE_results\ttau_SE_jackknife\tZ\tn_blocks\n")
        for r in tau_rows:
            f.write(f"{r[0]}\t{r[1]:.6e}\t{r[2]:.6e}\t{r[3]:.6e}\t{r[4]:+.4f}\t{n}\n")
    with open(prefix + "_contrast_pairs.tsv", "w") as f:
        f.write("contrast\tdelta_tau\tSE_delta\tZ\tP_two_sided\n")
        for r in pair_rows:
            f.write(f"{r[0]}\t{r[1]:+.6e}\t{r[2]:.6e}\t{r[3]:+.4f}\t{r[4]:.4g}\n")
    print(f"{Path(prefix).name}: n_clusters={ncl}, n_blocks={n}")
    for r in tau_rows:
        print(f"{r[0]}\ttau={r[1]:.4e}\tSE={r[2]:.4e}\tSE_jk={r[3]:.4e}\tZ={r[4]:+.2f}")
    for r in pair_rows:
        print(f"{r[0]}\tdelta={r[1]:+.4e}\tSE={r[2]:.4e}\tZ={r[3]:+.2f}\tP={r[4]:.3g}")
    print("\nCONTRAST_DONE")


if __name__ == "__main__":
    main()
