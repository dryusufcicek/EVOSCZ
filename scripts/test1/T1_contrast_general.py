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

    out = [f"# Between-cluster contrast — {Path(prefix).name} (n_clusters={ncl}, n_blocks={n})\n",
           "| cluster | tau | tau_SE(.results) | tau_SE(jk) | Z |", "|---|---:|---:|---:|---:|"]
    for k in clusters:
        if k not in idx:
            continue
        v = clusters[k]; col = arr[:, idx[k]]
        out.append(f"| {k} | {tau[v]:.4e} | {tause[v]:.4e} | {jk(col):.4e} | {tau[v]/tause[v]:+.2f} |")
    out.append("\n## Pairwise contrasts Δ = tau_i − tau_j (paired block-jackknife)\n")
    out.append("| contrast | Δtau | SE(Δ) | Z | P(2-sided) | interpretation |")
    out.append("|---|---:|---:|---:|---:|---|")
    keys = [k for k in clusters if k in idx]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            delta = arr[:, idx[ka]] - arr[:, idx[kb]]
            d = tau[clusters[ka]] - tau[clusters[kb]]
            se = jk(delta)
            z = d / se if se > 0 else float("nan")
            p = erfc(abs(z) / sqrt(2))
            interp = (f"{ka} NOT distinguishable from {kb} (n.s.)" if abs(z) < 1.96
                      else f"{ka} {'>' if d > 0 else '<'} {kb} (sig)")
            out.append(f"| {ka} − {kb} | {d:+.4e} | {se:.4e} | {z:+.2f} | {p:.3g} | {interp} |")
    text = "\n".join(out)
    print(text)
    Path(prefix + "_contrast.md").write_text(text + "\n")
    print("\nCONTRAST_DONE")


if __name__ == "__main__":
    main()
