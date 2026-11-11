#!/usr/bin/env python3
"""
Phase 14n: repair the A1/A2 labels of the phase14j LDSC-format summary statistics.

Background
----------
P14j/P14k/P14l/P14m sign Z with respect to the HapMap3 A1 allele (harmonise()),
but until 2026-11-11 they wrote the INPUT alleles as A1/A2. Every SNP whose input
A1 differs from the HapMap3 A1 therefore carried a Z of the wrong sign relative to
its labels. Analyses that use only Z^2 (ldsc.py --h2, partitioned heritability)
are unaffected; signed analyses (ldsc.py --rg, Genomic SEM, MR, PRS) are not.

What this script does
---------------------
For every *.sumstats.gz in --dir it rewrites A1/A2 to the HapMap3 alleles of that
SNP and leaves SNP, N and Z byte-for-byte unchanged. The result is identical to
re-running the corrected munge scripts. The script is idempotent: a file whose
labels already equal the HapMap3 alleles is reported as "already consistent".

Usage
-----
  python3 P14n_fix_allele_labels.py --dir results/phase14j            # dry run
  python3 P14n_fix_allele_labels.py --dir results/phase14j --apply    # rewrite

Output (with --apply): files are replaced atomically (temporary file in the same
directory, then rename) and ALLELE_LABEL_FIX.tsv is written to --dir.
"""
import argparse
import glob
import gzip
import hashlib
import os
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
RC = {"A": "T", "T": "A", "C": "G", "G": "C"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_hm3(path):
    hm3 = {}
    with open(path) as fh:
        header = fh.readline().split()
        if header[:3] != ["SNP", "A1", "A2"]:
            sys.exit(f"FATAL: unexpected HapMap3 header {header}")
        for line in fh:
            snp, a1, a2 = line.split()[:3]
            hm3[snp] = (a1.upper(), a2.upper())
    return hm3


def classify(a1, a2, h1, h2):
    if (a1, a2) == (h1, h2):
        return "same"
    if (a1, a2) == (h2, h1):
        return "swap"
    r1, r2 = RC.get(a1), RC.get(a2)
    if (r1, r2) == (h1, h2):
        return "rc_same"
    if (r1, r2) == (h2, h1):
        return "rc_swap"
    return "bad"


def process(path, hm3, apply):
    counts = {"same": 0, "swap": 0, "rc_same": 0, "rc_swap": 0, "bad": 0, "missing": 0}
    tmp = path + ".tmp_relabel"
    out = gzip.open(tmp, "wt", compresslevel=6) if apply else None
    n = 0
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n")
        if header.split("\t") != ["SNP", "A1", "A2", "N", "Z"]:
            if out:
                out.close(); os.remove(tmp)
            return None, f"unexpected header: {header}"
        if out:
            out.write(header + "\n")
        for line in fh:
            snp, a1, a2, n_str, z_str = line.rstrip("\n").split("\t")
            n += 1
            h = hm3.get(snp)
            if h is None:
                counts["missing"] += 1
                new1, new2 = a1, a2
            else:
                kind = classify(a1.upper(), a2.upper(), h[0], h[1])
                counts[kind] += 1
                new1, new2 = (h[0], h[1]) if kind != "bad" else (a1, a2)
            if out:
                out.write(f"{snp}\t{new1}\t{new2}\t{n_str}\t{z_str}\n")
    if out:
        out.close()
    if counts["bad"] or counts["missing"]:
        if apply:
            os.remove(tmp)
        return counts, f"{counts['bad']} unmatched and {counts['missing']} non-HapMap3 SNPs: file left untouched"
    changed = counts["swap"] + counts["rc_same"] + counts["rc_swap"]
    if not apply:
        return counts, f"dry run: {changed:,} of {n:,} labels would change"
    if changed == 0:
        os.remove(tmp)
        return counts, "already consistent"
    # verify the temporary file before replacing: same rows, same SNP, N and Z strings
    with gzip.open(path, "rt") as a, gzip.open(tmp, "rt") as b:
        for la, lb in zip(a, b):
            fa, fb = la.rstrip("\n").split("\t"), lb.rstrip("\n").split("\t")
            if (fa[0], fa[3], fa[4]) != (fb[0], fb[3], fb[4]):
                os.remove(tmp)
                return counts, "verification failed: SNP/N/Z differ"
        if a.readline() or b.readline():
            os.remove(tmp)
            return counts, "verification failed: row count differs"
    old = sha256(path)
    os.replace(tmp, path)
    return counts, f"rewritten: {changed:,} of {n:,} labels changed|{old}|{sha256(path)}"


def main():
    ap = argparse.ArgumentParser(description="Repair A1/A2 labels of phase14j sumstats")
    ap.add_argument("--dir", default=str(BASE / "results/phase14j"))
    ap.add_argument("--hm3", default=str(BASE / "data/ldsc/sldsc_ref/w_hm3.snplist"))
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    args = ap.parse_args()
    hm3 = load_hm3(args.hm3)
    print(f"HapMap3 SNPs: {len(hm3):,} ({args.hm3})")
    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.sumstats.gz"))):
        counts, msg = process(path, hm3, args.apply)
        name = os.path.basename(path)
        print(f"{name}: {msg.split('|')[0]} | {counts}")
        rows.append((name, counts, msg))
    if args.apply:
        manifest = os.path.join(args.dir, "ALLELE_LABEL_FIX.tsv")
        new = not os.path.exists(manifest)
        with open(manifest, "a") as o:
            if new:
                o.write("date\tfile\tsame\tswap\trc_same\trc_swap\tunmatched\tnot_in_hm3\tstatus\tsha256_before\tsha256_after\n")
            for name, counts, msg in rows:
                parts = msg.split("|")
                c = counts or {}
                o.write("\t".join([time.strftime("%Y-%m-%d %H:%M"), name] + [str(c.get(k, "")) for k in
                                  ("same", "swap", "rc_same", "rc_swap", "bad", "missing")] +
                                  [parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""]) + "\n")
        print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
