#!/usr/bin/env python3
"""
T1_extract_genome_age.py — genome-wide GEVA allele age per rsid, via the
EXACT recipe used by phase11/P11_build_variant_master.py (FIX-A1).

Recipe (verified to reproduce variant_master.age_median_yr for 20,565/20,565
credible-set variants, max abs diff 0.0000):
  - per rsid (VariantID), source priority Combined > TGP > SGDP (first wins)
  - clock = AgeMedian_Mut (mutation-clock median), units = GENERATIONS
We ALSO keep AgeMedian_Jnt (joint clock) for the robustness variant (M2).

Memory-safe: streams each atlas chr CSV line-by-line, keeps one dict entry per
rsid (the highest-priority source seen). Writes age_tables/age_chr{N}.tsv with
columns: rsid, age_mut_gen, age_jnt_gen, source.

Usage:  python3 T1_extract_genome_age.py [chr ...]   (default 1..22)
"""
import gzip
import sys
from pathlib import Path
import os
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")

ATLAS = Path((_SCRATCH + "/test1_age_conditioning/geva_atlas"))
OUT = Path((_SCRATCH + "/test1_age_conditioning/age_tables"))
OUT.mkdir(parents=True, exist_ok=True)

PRIORITY = {"Combined": 0, "TGP": 1, "SGDP": 2}
REQUIRED = ["VariantID", "DataSource", "AgeMedian_Mut", "AgeMedian_Jnt"]


def process_chr(c):
    f = ATLAS / f"atlas.chr{c}.csv.gz"
    if not f.exists():
        print(f"chr{c}: ERROR missing atlas file {f}", flush=True)
        return False
    best = {}            # rsid -> (priority, age_mut, age_jnt, source)
    idx = None
    n_rows = 0
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            fields = [x.strip() for x in line.rstrip("\n").split(",")]
            if idx is None:                       # header line
                idx = {col: i for i, col in enumerate(fields)}
                missing = [c2 for c2 in REQUIRED if c2 not in idx]
                if missing:
                    print(f"chr{c}: ERROR missing columns {missing}; header={fields[:8]}...",
                          flush=True)
                    return False
                continue
            n_rows += 1
            try:
                rsid = fields[idx["VariantID"]]
                src = fields[idx["DataSource"]]
            except IndexError:
                continue                          # malformed row -> skip
            p = PRIORITY.get(src, 99)
            cur = best.get(rsid)
            if cur is None or p < cur[0]:         # strictly higher priority; keep-first on ties
                best[rsid] = (p,
                              fields[idx["AgeMedian_Mut"]] if idx["AgeMedian_Mut"] < len(fields) else "",
                              fields[idx["AgeMedian_Jnt"]] if idx["AgeMedian_Jnt"] < len(fields) else "",
                              src)
    outf = OUT / f"age_chr{c}.tsv"
    with open(outf, "w") as o:
        o.write("rsid\tage_mut_gen\tage_jnt_gen\tsource\n")
        for rsid, (p, am, aj, src) in best.items():
            o.write(f"{rsid}\t{am}\t{aj}\t{src}\n")
    src_dist = {}
    for _, (p, am, aj, src) in best.items():
        src_dist[src] = src_dist.get(src, 0) + 1
    print(f"chr{c}: scanned {n_rows:,} rows -> {len(best):,} unique rsids "
          f"(selected-source {src_dist}) -> {outf}", flush=True)
    return True


if __name__ == "__main__":
    chrs = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 23))
    ok = all(process_chr(c) for c in chrs)
    print("DONE_OK" if ok else "DONE_WITH_ERRORS", flush=True)
    sys.exit(0 if ok else 1)
