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
    out = ["# Robustness check 1 — formal between-cluster coefficient contrast (joint model M0)\n",
           f"n_blocks={n}; tau (conditional coefficient) per cluster:\n",
           "| cluster | tau | tau_SE(.results) | tau_SE(jackknife) | Z |",
           "|---|---:|---:|---:|---:|"]
    for k, v in CLUST.items():
        col = arr[:, idx[k]]
        se_jk = jk_se(col, n)
        out.append(f"| {k} | {tau[v]:.4e} | {tause[v]:.4e} | {se_jk:.4e} | {tau[v]/tause[v]:+.2f} |")
    out.append("\n## Pairwise contrasts  Δ = tau_i − tau_j  (paired block-jackknife SE)\n")
    out.append("| contrast | Δtau | SE(Δ) | Z | P(2-sided) | interpretation |")
    out.append("|---|---:|---:|---:|---:|---|")
    from math import erfc, sqrt
    pairs = [("C0", "C2"), ("C0", "C1"), ("C1", "C2")]
    for a, b in pairs:
        ca, cb = arr[:, idx[a]], arr[:, idx[b]]
        delta = ca - cb
        d = tau[CLUST[a]] - tau[CLUST[b]]
        se = jk_se(delta, n)
        z = d / se if se > 0 else float("nan")
        p = erfc(abs(z) / sqrt(2))
        if abs(z) < 1.96:
            interp = f"{a} NOT distinguishable from {b} (n.s.)"
        else:
            interp = f"{a} {'>' if d > 0 else '<'} {b} (significant)"
        out.append(f"| {a} − {b} | {d:+.4e} | {se:.4e} | {z:+.2f} | {p:.3g} | {interp} |")
    out.append("\n_Concern 1 headline: is C0 (Young) significantly > C2 (Old)? See C0−C2 row. "
               "Note C0 vs C1 (C1 had higher Z) reported honestly._\n")
    text = "\n".join(out)
    print(text)
    (RES / "T1_cluster_contrast.md").write_text(text)
    print("\nCONTRAST_DONE")


if __name__ == "__main__":
    main()
