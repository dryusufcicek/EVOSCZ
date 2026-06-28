#!/usr/bin/env python3
"""
Phase 14p5a — Per-arm Wohns 2022 mutation midpoint age for PGC3 credible-set
=============================================================================
Input (env): ARM_ID (e.g. "1_p", "22_q") — passed by SLURM task

For each PGC3 credible-set variant on the specified chromosome ARM:
  1. Look up its hg38 position in the Wohns dated tree (.tsz)
  2. Find all mutations at that site
  3. Match by derived_state to PGC3 effect_allele/other_allele
  4. For the matched mutation, compute:
     - child_node_time = node.time
     - parent_node_time (via local tree)
     - midpoint = (child + parent) / 2  ← mutation age estimate (generations)
     - age_yr = midpoint × 28.1 (Wohns generation time)
  5. Save TSV: rsid, pos_hg38, mut_child_gen, mut_parent_gen, mut_midpoint_gen, mut_age_yr

Notes:
  - Wohns dated tree mutations have time=NaN; canonical mutation age estimate
    is the midpoint of the branch on which the mutation sits (tsdate convention).
  - For recurrent mutations matching multiple times, take the OLDEST (highest
    midpoint age) — assumption: deepest matching event is the ancestral biological
    mutation; shallower recurrences are likely sequencing/back-mutation events.
"""

import os, sys, time, json
from pathlib import Path
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
from collections import defaultdict
import numpy as np
import pandas as pd
import tszip, tskit

ARM = os.environ.get("ARM_ID", "22_q")
DATA = Path((_SCRATCH + "/v11_data/wohns2022"))
OUT  = Path((_SCRATCH + "/v11_data/phase14p5"))
OUT.mkdir(parents=True, exist_ok=True)

WOHNS_GEN = 28.1  # Wohns 2022 / Speidel convention
tsz_path = DATA / f"hgdp_tgp_sgdp_high_cov_ancients_chr{ARM}.dated.trees.tsz"

print(f"[{time.strftime('%H:%M:%S')}] Loading {tsz_path.name}", flush=True)
t0 = time.time()
ts = tszip.decompress(str(tsz_path))
print(f"  Loaded in {time.time()-t0:.1f}s: {ts.num_samples:,} samples, "
      f"{ts.num_sites:,} sites, {ts.num_mutations:,} mutations", flush=True)

# Parse chromosome number from ARM (e.g. "1_p" → 1)
chrom_num = int(ARM.split("_")[0])

# Load PGC3 credible-set for this chromosome
vm = pd.read_parquet((_ROOT + "/results/phase11/variant_master_v4.parquet"))
vm["chr"] = pd.to_numeric(vm["chr"], errors="coerce").astype("Int64")
vm["pos_hg38"] = pd.to_numeric(vm["pos_hg38"], errors="coerce").astype("Int64")
pgc_chr = vm[vm["chr"] == chrom_num].dropna(subset=["pos_hg38"]).copy()
pgc_chr["pos_hg38_int"] = pgc_chr["pos_hg38"].astype(int)

# Restrict to positions within the arm's sequence range
arm_lo = int(np.asarray(ts.tables.sites.position).min()) if ts.num_sites else 0
arm_hi = int(np.asarray(ts.tables.sites.position).max()) if ts.num_sites else 0
pgc_arm = pgc_chr[
    (pgc_chr["pos_hg38_int"] >= arm_lo) & (pgc_chr["pos_hg38_int"] <= arm_hi)
].copy()
print(f"  chr{chrom_num} arm range hg38: [{arm_lo:,}, {arm_hi:,}]", flush=True)
print(f"  PGC3 chr{chrom_num} credible-set: {len(pgc_chr)} total, "
      f"{len(pgc_arm)} within arm", flush=True)

# Fast PGC3 position → row map
pgc_by_pos = {int(r["pos_hg38_int"]): r for _, r in pgc_arm.iterrows()}
pgc_positions = set(pgc_by_pos.keys())

node_time = np.asarray(ts.tables.nodes.time)

# Iterate trees, match mutations
hits = []
t0 = time.time()
n_trees_visited = 0
n_progress = max(1, ts.num_trees // 10)
for tree in ts.trees():
    n_trees_visited += 1
    if n_trees_visited % n_progress == 0:
        print(f"  {n_trees_visited}/{ts.num_trees} trees ({time.time()-t0:.0f}s, hits so far: {len(hits)})", flush=True)
    for s in tree.sites():
        if int(s.position) not in pgc_positions:
            continue
        pgc_row = pgc_by_pos[int(s.position)]
        pgc_a1 = str(pgc_row["effect_allele"]).upper()
        pgc_a2 = str(pgc_row["other_allele"]).upper()
        for m in s.mutations:
            d = m.derived_state.upper()
            matched = (d == pgc_a1 or d == pgc_a2)
            if not matched:
                continue
            n = m.node
            ct = float(node_time[n])
            parent_n = tree.parent(n)
            pt = float(node_time[parent_n]) if parent_n != tskit.NULL else ct
            mid = (ct + pt) / 2
            hits.append({
                "rsid": pgc_row["rsid"],
                "pos_hg38": int(s.position),
                "pgc_a1": pgc_a1, "pgc_a2": pgc_a2,
                "wohns_derived": d,
                "wohns_child_node_time_gen": ct,
                "wohns_parent_node_time_gen": pt,
                "wohns_midpoint_age_gen": mid,
                "wohns_midpoint_age_yr": mid * WOHNS_GEN,
            })

raw = pd.DataFrame(hits)
print(f"\n[{time.strftime('%H:%M:%S')}] Total allele-matched mutations: {len(raw)} "
      f"({time.time()-t0:.0f}s elapsed)", flush=True)
print(f"  Unique rsids: {raw['rsid'].nunique() if len(raw) else 0}", flush=True)

if len(raw) > 0:
    # Per rsid, take the OLDEST allele-matched mutation
    final = raw.sort_values("wohns_midpoint_age_yr", ascending=False).drop_duplicates(
        subset=["rsid"], keep="first"
    ).reset_index(drop=True)
    print(f"  After dedup (oldest per rsid): {len(final)}", flush=True)

    out_file = OUT / f"P14p5a_chr{ARM}.tsv"
    final.to_csv(out_file, sep="\t", index=False)
    print(f"  Saved: {out_file}", flush=True)

    # Quick stats
    print(f"  Wohns midpoint age (yr): "
          f"median={final['wohns_midpoint_age_yr'].median():,.0f}, "
          f"q99={final['wohns_midpoint_age_yr'].quantile(0.99):,.0f}, "
          f"max={final['wohns_midpoint_age_yr'].max():,.0f}", flush=True)
else:
    out_file = OUT / f"P14p5a_chr{ARM}.tsv"
    pd.DataFrame(columns=["rsid","pos_hg38","wohns_midpoint_age_yr"]).to_csv(out_file, sep="\t", index=False)
    print(f"  No matches; empty file saved: {out_file}", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] chr{ARM} complete", flush=True)
