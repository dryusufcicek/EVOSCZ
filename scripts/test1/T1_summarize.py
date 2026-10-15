#!/usr/bin/env python3
"""
T1_summarize.py (v2 — coefficient/tau-based per methodology bug review)

Reads M0/M1/M1d/M2 partitioned-LDSC .results and reports whether the C0 (Young)
cluster's CONDITIONAL COEFFICIENT (tau) survives the addition of continuous /
decile allele age. Verdict is on tau_C0 (NOT the Z-score — Z is not comparable
across models because SEs change with added/collinear annotations; reported
descriptively only). Pre-registered guard in DECISIONS.md.

Category naming (verified): <annot_col>L2_<refld_file_index>. Our --ref-ld-chr
order is baselineLD.(0), cluster.(1), age.(2) -> C0 = "C0L2_1", age cols = "*L2_2".
"""
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

W = Path((_SCRATCH + "/test1_age_conditioning"))
RES = W / "results"
OBS_PRIMARY_Z = 3.05      # published C0 conditional Z under baseline-LD v2.2 (NOT the +3.55 plain-baseline)
BONF = 2.81

MODELS = {
    "M0_baseline_cluster":     "baseline-LD v2.2 + cluster (already conditions on baseline allele-age proxies) — SANITY, expect C0 Z≈+3.05",
    "M1_cluster_age_mut":      "+ GEVA continuous Mut-clock age (linear)",
    "M1d_cluster_age_mut_dec": "+ GEVA Mut-clock age DECILES (flexible age control) — PRIMARY",
    "M2_cluster_age_jnt":      "+ GEVA continuous Jnt-clock age (linear) — robustness",
}
CLUSTER = {"C0": "C0L2_1", "C1": "C1L2_1", "C2": "C2L2_1"}


def parse(path):
    if not path.exists():
        return None
    rows = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            rows[p[0]] = {
                "tau": p[ci["Coefficient"]] if "Coefficient" in ci else "NA",
                "tau_se": p[ci["Coefficient_std_error"]] if "Coefficient_std_error" in ci else "NA",
                "z": p[ci["Coefficient_z-score"]] if "Coefficient_z-score" in ci else "NA",
                "enr": p[ci["Enrichment"]] if "Enrichment" in ci else "NA",
            }
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get(rows, cat):
    if rows and cat in rows:
        return rows[cat]
    # fallback: startswith (in case suffix index differs)
    if rows:
        base = cat.split("L2_")[0]
        for k in rows:
            if k.startswith(base + "L2_"):
                return rows[k]
    return None


def main():
    parsed = {n: parse(RES / f"{n}.results") for n in MODELS}

    with open(RES / "T1_SUMMARY_annotations.tsv", "w") as f:
        f.write("model\tmodel_description\tannotation\tcategory\ttau\ttau_SE\tZ\tenrichment\n")
        for n, desc in MODELS.items():
            rows = parsed[n]
            if rows is None:
                f.write(f"{n}\t{desc}\tMISSING\t\t\t\t\t\n")
                continue
            for label, cat in CLUSTER.items():
                r = get(rows, cat)
                if r:
                    f.write(f"{n}\t{desc}\t{label}\t{cat}\t{r['tau']}\t{r['tau_se']}\t{r['z']}\t{r['enr']}\n")
            for k, v in rows.items():
                if k.endswith("L2_2"):
                    f.write(f"{n}\t{desc}\t{k}\t{k}\t{v['tau']}\t{v['tau_se']}\t{v['z']}\t{v['enr']}\n")

    def c0(n):
        r = get(parsed.get(n), CLUSTER["C0"])
        return (num(r["tau"]), num(r["tau_se"]), num(r["z"])) if r else (None, None, None)

    t0, se0, z0 = c0("M0_baseline_cluster")
    with open(RES / "T1_SUMMARY_C0.tsv", "w") as f:
        f.write("model\ttau_C0\tSE\tZ\tdelta_tau_vs_M0\tpct_change_vs_M0\twithin_2SE_of_M0\tZ_ge_bonferroni\n")
        for n in MODELS:
            t, se, z = c0(n)
            if t is None:
                f.write(f"{n}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n")
                continue
            dt = (t - t0) if t0 is not None else float("nan")
            pct = (1 - t / t0) * 100 if t0 not in (None, 0) else float("nan")
            overlap = int(se is not None and se0 is not None and abs(dt) <= 2 * max(se, se0))
            s_se = "NA" if se is None else f"{se:.6e}"
            s_z = "NA" if z is None else f"{z:+.4f}"
            s_bonf = "NA" if z is None else str(int(z >= BONF))
            f.write(f"{n}\t{t:.6e}\t{s_se}\t{s_z}\t{dt:+.6e}\t{pct:.2f}\t{overlap}\t{s_bonf}\n")
    if z0 is not None:
        print(f"M0 C0 Z = {z0:+.3f}; published v2.2 value {OBS_PRIMARY_Z:+.2f}; difference {z0 - OBS_PRIMARY_Z:+.3f}")
    print((RES / "T1_SUMMARY_C0.tsv").read_text())
    print("\nSUMMARY_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
