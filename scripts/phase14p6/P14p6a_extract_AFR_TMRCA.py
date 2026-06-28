#!/usr/bin/env python3
"""
Phase 14p6a — AFR-context TMRCA per PGC3 credible-set variant
==============================================================
For each PGC3 credible-set variant (rsid, chr, pos_hg38) on the chromosome
arm specified by env ARM_ID:

  1. Find the site at the variant's hg38 position in the Wohns dated tree
  2. Identify mutations at this site whose derived_state matches the
     PGC3 effect_allele / other_allele (allele match)
  3. For each matched mutation, traverse its subtree and collect sample
     descendants
  4. Intersect descendants with AFR sample set (HGDP+TGP+SGDP region=Africa
     or super_population=AFR; excluding admixed ACB/ASW from the 1000G TGP set
     for analytical clarity)
  5. Pool AFR carriers across all matched mutations at this site (union):
     `AFR_carriers = ∪ samples(mut.node) ∩ AFR` for each matched mut
  6. Compute AFR_TMRCA = MRCA(AFR_carriers) node time using local tree
     - 0 carriers → AFR_absent = True, AFR_TMRCA = NaN
     - 1 carrier  → singleton, AFR_TMRCA = 0
     - 2+         → tree.mrca(*carriers).time

Output (per arm): rsid, pos_hg38, n_matched_mutations,
  n_carriers_total, n_AFR_carriers, AFR_DAF_in_tree,
  AFR_TMRCA_gen, AFR_TMRCA_yr, AFR_absent_flag

Methodological notes:
  - This is NOT the mutation midpoint (the previous flawed metric).
  - AFR_TMRCA measures: among AFR samples carrying the derived allele,
    when did their MRCA exist?  For an EUR-private variant: AFR_carriers
    is empty → AFR_absent.  For a deeply-shared variant: AFR_carriers is
    large → AFR_TMRCA is deep node time.  For a recent AFR-arising
    variant: AFR_carriers small + shallow MRCA.
  - This is the correct AFR-context age metric.
"""

import os, sys, time, json
from pathlib import Path
_ROOT = os.environ.get("EVOSCZ_ROOT") or str(Path(__file__).resolve().parents[2])
_SCRATCH = os.environ.get("EVOSCZ_SCRATCH") or (_ROOT + "/scratch")
import numpy as np
import pandas as pd
import tszip, tskit

ARM = os.environ.get("ARM_ID", "22_q")
DATA = Path((_SCRATCH + "/v11_data/wohns2022"))
OUT  = Path((_SCRATCH + "/v11_data/phase14p6"))
OUT.mkdir(parents=True, exist_ok=True)

GEN_TO_YR = 28.1  # Wohns 2022 generation time

tsz = DATA / f"hgdp_tgp_sgdp_high_cov_ancients_chr{ARM}.dated.trees.tsz"
print(f"[{time.strftime('%H:%M:%S')}] Loading {tsz.name}", flush=True)
t0 = time.time()
ts = tszip.decompress(str(tsz))
print(f"  loaded in {time.time()-t0:.0f}s: samples={ts.num_samples:,}, "
      f"sites={ts.num_sites:,}, muts={ts.num_mutations:,}", flush=True)

# Build AFR sample set:
#   HGDP region == AFRICA, SGDP region == Africa, TGP super_population == AFR
#   Exclude ACB + ASW (admixed African American/Caribbean) to keep pure AFR.
AFR_NAMES_EXCLUDE = {"ACB", "ASW"}
afr_pops = set()
for p in ts.populations():
    md = json.loads(p.metadata) if isinstance(p.metadata, bytes) else (p.metadata or {})
    region = (md.get("region") or "").upper()
    sp = md.get("super_population", "")
    name = md.get("name", "")
    is_afr = (region in ("AFRICA",) or sp == "AFR")
    if is_afr and name not in AFR_NAMES_EXCLUDE:
        afr_pops.add(p.id)

afr_samples = set()
for s in ts.samples():
    pop = ts.node(s).population
    if pop in afr_pops:
        afr_samples.add(s)
print(f"  AFR populations: {len(afr_pops)}, AFR sample nodes: {len(afr_samples)}", flush=True)

# Load PGC3 credible-set for this chr
chrom_num = int(ARM.split("_")[0])
vm = pd.read_parquet((_ROOT + "/results/phase11/variant_master_v4.parquet"))
vm["chr"] = pd.to_numeric(vm["chr"], errors="coerce").astype("Int64")
vm["pos_hg38"] = pd.to_numeric(vm["pos_hg38"], errors="coerce").astype("Int64")
pgc = vm[vm["chr"] == chrom_num].dropna(subset=["pos_hg38"]).copy()
pgc["pos_hg38_int"] = pgc["pos_hg38"].astype(int)

# Restrict to arm sequence range
if ts.num_sites > 0:
    arm_lo = int(np.asarray(ts.tables.sites.position).min())
    arm_hi = int(np.asarray(ts.tables.sites.position).max())
    pgc = pgc[(pgc["pos_hg38_int"] >= arm_lo) & (pgc["pos_hg38_int"] <= arm_hi)].copy()
print(f"  PGC3 chr{chrom_num} arm-restricted variants: {len(pgc)}", flush=True)

pgc_by_pos = {int(r["pos_hg38_int"]): r for _, r in pgc.iterrows()}
pgc_positions = set(pgc_by_pos.keys())
node_time = np.asarray(ts.tables.nodes.time)

# Iterate trees, find PGC3 sites
results = []
t0 = time.time()
n_trees = 0
n_progress = max(1, ts.num_trees // 10)
for tree in ts.trees():
    n_trees += 1
    if n_trees % n_progress == 0:
        print(f"  trees {n_trees}/{ts.num_trees} ({time.time()-t0:.0f}s, hits {len(results)})", flush=True)
    for site in tree.sites():
        if int(site.position) not in pgc_positions:
            continue
        row = pgc_by_pos[int(site.position)]
        pgc_a1 = str(row["effect_allele"]).upper()
        pgc_a2 = str(row["other_allele"]).upper()
        # Pool AFR carriers across all allele-matched mutations
        afr_carriers_union = set()
        all_carriers_union = set()
        matched_mut_nodes = []
        for m in site.mutations:
            d = m.derived_state.upper()
            if not (d == pgc_a1 or d == pgc_a2):
                continue
            # All samples below mutation node
            subtree_samples = set(tree.samples(m.node))
            all_carriers_union |= subtree_samples
            afr_subset = subtree_samples & afr_samples
            afr_carriers_union |= afr_subset
            matched_mut_nodes.append(m.node)

        n_total = len(all_carriers_union)
        n_afr = len(afr_carriers_union)

        if n_afr == 0:
            afr_tmrca_gen = np.nan
            afr_absent = True
        elif n_afr == 1:
            afr_tmrca_gen = 0.0
            afr_absent = False
        else:
            # MRCA across multiple AFR carriers
            try:
                mrca_node = tree.mrca(*afr_carriers_union)
                afr_tmrca_gen = float(node_time[mrca_node]) if mrca_node != tskit.NULL else np.nan
            except Exception:
                afr_tmrca_gen = np.nan
            afr_absent = False

        afr_daf_tree = n_afr / max(1, len(afr_samples))

        results.append({
            "rsid": row["rsid"],
            "chr": chrom_num, "arm": ARM,
            "pos_hg38": int(site.position),
            "pgc_a1": pgc_a1, "pgc_a2": pgc_a2,
            "n_matched_mutations": len(matched_mut_nodes),
            "n_carriers_total": n_total,
            "n_AFR_carriers": n_afr,
            "AFR_DAF_in_tree": float(afr_daf_tree),
            "AFR_TMRCA_gen": float(afr_tmrca_gen),
            "AFR_TMRCA_yr": float(afr_tmrca_gen * GEN_TO_YR) if not np.isnan(afr_tmrca_gen) else np.nan,
            "AFR_absent": bool(afr_absent),
        })

df = pd.DataFrame(results)
out = OUT / f"P14p6a_chr{ARM}.tsv"
df.to_csv(out, sep="\t", index=False)
print(f"\n[{time.strftime('%H:%M:%S')}] Saved {out}: {len(df)} variants", flush=True)
if len(df) > 0:
    print(f"  AFR_absent: {df['AFR_absent'].sum()} ({df['AFR_absent'].mean()*100:.1f}%)", flush=True)
    valid = df[~df["AFR_absent"]]
    if len(valid) > 0:
        print(f"  AFR_TMRCA (yr) of AFR-present: median={valid['AFR_TMRCA_yr'].median():,.0f}, "
              f"q99={valid['AFR_TMRCA_yr'].quantile(0.99):,.0f}, max={valid['AFR_TMRCA_yr'].max():,.0f}", flush=True)
        print(f"  AFR_DAF (tree-based) of AFR-present: median={valid['AFR_DAF_in_tree'].median():.3f}", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] chr{ARM} complete", flush=True)
