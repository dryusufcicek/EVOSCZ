#!/usr/bin/env python3
"""
T1_jackknife_diff.py — rigorous paired block-jackknife difference SE for
Delta tau_C0 = tau_C0(model_with_age) - tau_C0(M0)  (methodology Fix-3).

Comparing Z across models is invalid (SE changes). The correct test of "did age
absorb C0" is the change in the COEFFICIENT tau_C0, with a SE that accounts for the
pairing (same sumstats/weights/SNPs/blocks in both models). Using LDSC's per-block
coefficient delete-values (<out>.part_delete), the paired difference SE is:
    delta_b = dC0^{age}[b] - dC0^{M0}[b]
    SE = sqrt((n-1)/n * sum_b (delta_b - mean(delta))^2)
    Z  = (tau_age - tau_M0) / SE

FORMAT IS SELF-IDENTIFIED (not assumed): we confirm a candidate delete-file/column
reproduces the .results Coefficient_std_error for C0 via the standard jackknife SE
sqrt((n-1)/n * sum (d-mean)^2). Only then is it used. If no candidate reproduces it,
we abort and report (fall back to the tau±SE overlap already in T1_SUMMARY).

Usage: python3 T1_jackknife_diff.py M0_baseline_cluster M1d_cluster_age_mut_dec [...]
       (first arg = baseline model M0; the rest = age models to contrast against M0)
"""
import sys
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
import numpy as np

RES = Path((_SCRATCH + "/test1_age_conditioning/results"))
C0 = "C0L2_1"


def results_rows(model):
    """Return (ordered annotation Category list, dict cat->(tau, tau_se))."""
    cats, info = [], {}
    with open(RES / f"{model}.results") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            cats.append(p[0])
            info[p[0]] = (float(p[ci["Coefficient"]]), float(p[ci["Coefficient_std_error"]]))
    return cats, info


def load_delete(model):
    """Load <model>.part_delete (try .delete fallback). Returns ndarray (n_blocks x n_annot) or None."""
    for ext in (".part_delete", ".delete"):
        f = RES / f"{model}{ext}"
        if f.exists():
            try:
                arr = np.loadtxt(f)
                if arr.ndim == 2:
                    return arr, ext
            except Exception as e:
                print(f"  load {f.name} failed: {e}")
    return None, None


def jk_se(delete_col, n):
    m = delete_col.mean()
    return np.sqrt((n - 1) / n * np.sum((delete_col - m) ** 2))


def c0_delete_vector(model):
    """Return (delete_vec_for_C0, n_blocks) after self-verifying it reproduces .results C0 SE."""
    cats, info = results_rows(model)
    if C0 not in info:
        print(f"  {model}: {C0} not in .results"); return None, None
    tau_se_target = info[C0][1]
    arr, ext = load_delete(model)
    if arr is None:
        print(f"  {model}: no delete file"); return None, None
    nb, na = arr.shape
    # column index of C0 = its row index among annotations in .results (same order)
    if na != len(cats):
        print(f"  {model}: {ext} has {na} cols but .results has {len(cats)} annots — mapping by index uncertain")
    idx = cats.index(C0) if C0 in cats else None
    if idx is None or idx >= na:
        print(f"  {model}: cannot locate C0 column"); return None, None
    col = arr[:, idx]
    se_recon = jk_se(col, nb)
    ok = abs(se_recon - tau_se_target) / max(tau_se_target, 1e-300) < 0.05
    print(f"  {model}: {ext} shape={arr.shape}, C0 col idx={idx}, "
          f"jackknife-SE={se_recon:.4e} vs .results tau_SE={tau_se_target:.4e} "
          f"-> {'MATCH (tau delete-vals confirmed)' if ok else 'NO MATCH'}")
    return (col, nb) if ok else (None, None)


def main():
    if len(sys.argv) < 3:
        print("usage: T1_jackknife_diff.py <M0> <age_model> [<age_model> ...]"); sys.exit(1)
    m0 = sys.argv[1]
    _, info0 = results_rows(m0)
    tau0 = info0[C0][0]
    col0, n0 = c0_delete_vector(m0)
    if col0 is None:
        print("ABORT: could not self-verify M0 tau delete-values; "
              "fall back to tau±SE overlap in T1_SUMMARY."); sys.exit(2)

    out = ["\n# Delta tau_C0 paired block-jackknife (Fix-3)\n",
           f"M0={m0}: tau_C0={tau0:.4e}, n_blocks={n0}\n",
           "| age model | tau_C0(age) | Delta tau | SE(Delta) | Z(Delta) | interpretation |",
           "|---|---:|---:|---:|---:|---|"]
    for mage in sys.argv[2:]:
        _, infoA = results_rows(mage)
        if C0 not in infoA:
            out.append(f"| {mage} | MISSING | | | | |"); continue
        tauA = infoA[C0][0]
        colA, nA = c0_delete_vector(mage)
        if colA is None or nA != n0:
            out.append(f"| {mage} | {tauA:.4e} | (no paired delete) | | | tau±SE only |"); continue
        delta = colA - col0
        dtau = tauA - tau0
        se = jk_se(delta, n0)
        z = dtau / se if se > 0 else float("nan")
        # interpret: significant negative Delta => age absorbs C0; n.s. => C0 stable
        if abs(z) < 1.96:
            interp = "Delta n.s. -> tau_C0 STABLE (C0 beyond age)"
        elif dtau < 0:
            interp = "Delta<0 sig -> age SIGNIFICANTLY attenuates C0"
        else:
            interp = "Delta>0 sig -> C0 strengthens with age cond."
        out.append(f"| {mage} | {tauA:.4e} | {dtau:+.4e} | {se:.4e} | {z:+.2f} | {interp} |")

    text = "\n".join(out)
    print(text)
    with open(RES / "T1_jackknife_diff.md", "w") as f:
        f.write(text + "\n")
    print("\nJACKKNIFE_DIFF_DONE")


if __name__ == "__main__":
    main()
