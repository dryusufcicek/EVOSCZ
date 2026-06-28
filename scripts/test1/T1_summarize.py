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
    out = ["# Test 1 — Allele-Age Conditioning of C0 — RESULTS (coefficient/tau-based)\n",
           "_Context: baseline-LD v2.2 ALREADY includes MAF_Adj_Predicted_Allele_Age + MAF_Adj_ASMC, "
           "so M0's C0 is already conditional on allele-age proxies. Test adds the cluster-defining GEVA "
           "age (linear + deciles) on top. Verdict on tau_C0; Z descriptive only._\n"]
    parsed = {n: parse(RES / f"{n}.results") for n in MODELS}

    for n, desc in MODELS.items():
        rows = parsed[n]
        out.append(f"\n## {n}\n_{desc}_\n")
        if rows is None:
            out.append("  **MISSING .results**\n"); continue
        out.append("| annotation | tau | tau_SE | Z | Enrichment |")
        out.append("|---|---:|---:|---:|---:|")
        for label, cat in CLUSTER.items():
            r = get(rows, cat)
            if r:
                out.append(f"| {label} ({cat}) | {r['tau']} | {r['tau_se']} | {r['z']} | {r['enr']} |")
        # age annotation rows (file index 2)
        age_rows = {k: v for k, v in rows.items() if k.endswith("L2_2")}
        for k, v in age_rows.items():
            out.append(f"| {k} | {v['tau']} | {v['tau_se']} | {v['z']} | {v['enr']} |")
        out.append("")

    # ---- headline: tau_C0 across models ----
    def c0(n):
        r = get(parsed.get(n), CLUSTER["C0"])
        return (num(r["tau"]), num(r["tau_se"]), num(r["z"])) if r else (None, None, None)

    t0, se0, z0 = c0("M0_baseline_cluster")
    t1, se1, z1 = c0("M1_cluster_age_mut")
    td, sed, zd = c0("M1d_cluster_age_mut_dec")
    t2, se2, z2 = c0("M2_cluster_age_jnt")

    out.append("\n## HEADLINE — C0 conditional COEFFICIENT (tau) across models\n")
    out.append("| model | tau_C0 | SE | Z (descriptive) |")
    out.append("|---|---:|---:|---:|")
    for lbl, (t, se, z) in [("M0 (no GEVA age)", (t0, se0, z0)),
                            ("M1 (+Mut linear)", (t1, se1, z1)),
                            ("M1d (+Mut deciles) PRIMARY", (td, sed, zd)),
                            ("M2 (+Jnt linear)", (t2, se2, z2))]:
        st = f"{t:.3e}" if t is not None else "NA"
        ss = f"{se:.3e}" if se is not None else "NA"
        sz = f"{z:+.3f}" if z is not None else "NA"
        out.append(f"| {lbl} | {st} | {ss} | {sz} |")
    out.append("")

    notes = []
    # sanity
    if z0 is not None:
        notes.append(("SANITY PASS" if abs(z0 - OBS_PRIMARY_Z) <= 0.30 else "⚠️ SANITY FAIL") +
                     f": M0 C0 Z={z0:+.2f} vs published v2.2 +{OBS_PRIMARY_Z:.2f} (Δ={z0-OBS_PRIMARY_Z:+.2f}).")
    # tau attenuation verdict (M0 -> M1d primary)
    def verdict(name, t, se, z):
        if t0 is None or t is None:
            return f"{name}: tau missing — cannot judge."
        dt = t - t0
        pct = (1 - t / t0) * 100 if t0 != 0 else float("nan")
        # 2-SE overlap heuristic (proxy; rigorous Δtau jackknife SE computed separately)
        overlap = (se is not None and se0 is not None and abs(dt) <= 2 * max(se, se0))
        survives = (z is not None and z >= BONF) and (pct < 50)
        tag = ("C0 SURVIVES (tau stable, beyond age)" if survives else
               "C0 ATTENUATED by age (concern not cleared)")
        return (f"{name}: tau_C0 {t0:.2e} -> {t:.2e} (Δ={dt:+.2e}, {pct:.0f}% change; "
                f"{'within' if overlap else 'OUTSIDE'} 2·SE; Z={z:+.2f}). => **{tag}**")
    if td is not None:
        notes.append("PRIMARY " + verdict("M1d deciles", td, sed, zd))
    if t1 is not None:
        notes.append(verdict("M1 linear", t1, se1, z1))
    if t2 is not None:
        notes.append(verdict("M2 jnt", t2, se2, z2))
    notes.append("NOTE: Z not directly comparable across models (SE changes); verdict is on tau. "
                 "Rigorous Δtau block-jackknife difference SE: see T1_jackknife_diff (uses .part_delete).")
    notes.append("NOTE: also inspect the GEVA-age annotation's own coefficient (above) — if ~null, it "
                 "adds little beyond baseline's existing allele-age proxies (Fix-1 caveat).")

    out.append("## VERDICT / NOTES\n")
    out += [f"- {n}" for n in notes]
    out.append("\n_Auto-generated; analyst reviews tau-contrast + jackknife diff before final call._\n")

    text = "\n".join(out)
    (RES / "T1_SUMMARY.md").write_text(text)
    with open(W / "DECISIONS.md", "a") as f:
        f.write("\n\n---\n\n# RESULTS (auto-appended)\n\n" + text + "\n")
    print(text)
    print("\nSUMMARY_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
