#!/usr/bin/env python3
"""
Phase 11c: Variant Master Safety Check (v5 — post-fix-A1)
=========================================================
After Faz A correction (FIX-A1: GEVA dedup BEFORE merge in P11), the
variant_master.parquet emitted by P11 already has 1 row per unique rsid
(no GEVA-source inflation). This script is now a SAFETY CHECK that:

  1. Verifies the master has 1 row per rsid (no inflation).
  2. Re-emits a `variant_master_clean.parquet` mirror for backwards
     compatibility with downstream scripts that expect this filename.
  3. Reports per-source GEVA distribution and key annotation coverage.

If inflation is detected (>1 row per rsid), it falls back to the old
priority-sort dedup (Combined > TGP > SGDP > other) and prints a warning.

Output:
  - results/phase11/variant_master_clean.parquet (mirror of v5 master,
    20,766 rows × N cols)
"""

import pandas as pd
from pathlib import Path
import os
from datetime import datetime

BASE = Path(os.environ.get("EVOSCZ_ROOT") or Path(__file__).resolve().parents[2])
OUT = BASE / "results/phase11"

print(f"Phase 11c v5: Safety Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

m = pd.read_parquet(OUT / "variant_master.parquet")
print(f"Input: {len(m)} rows × {len(m.columns)} cols")

n_unique_rsid = m["rsid"].nunique()
n_total = len(m)
inflated = n_total > n_unique_rsid

if inflated:
    print(f"  ! WARNING: master has {n_total} rows but only {n_unique_rsid} unique rsids")
    print(f"  ! Falling back to priority-sort dedup (Combined > TGP > SGDP > other)")
    priority_order = {"Combined": 0, "TGP": 1, "SGDP": 2}
    m["_geva_rank"] = m["geva_source"].fillna("").map(
        lambda s: priority_order.get(s, 99)
    )
    m_clean = (m.sort_values(["rsid", "_geva_rank"])
                .drop_duplicates("rsid", keep="first")
                .drop(columns=["_geva_rank"]))
    print(f"  Output after dedup: {len(m_clean)} unique rsids")
else:
    print(f"  ✓ No inflation detected (rows = unique rsids = {n_total})")
    m_clean = m

# Verify per-source GEVA distribution
if "geva_source" in m_clean.columns:
    src_counts = m_clean["geva_source"].value_counts(dropna=False)
    print(f"\nGEVA source distribution:")
    for s, n in src_counts.items():
        print(f"  {s}: {n}")

# Save mirror for backwards compatibility
out_path = OUT / "variant_master_clean.parquet"
m_clean.to_parquet(out_path, index=False, compression="snappy")
print(f"\nSaved: {out_path}")

tsv_path = OUT / "variant_master_clean.tsv.gz"
m_clean.to_csv(tsv_path, sep="\t", index=False, compression="gzip")
print(f"Saved: {tsv_path}")

# Annotation coverage
print("\nKey annotation coverage:")
for col in ["age_median_yr", "sds", "gtex_brain_minp", "gtex_blood_minp",
            "atac_any", "har_overlap", "desert_tier",
            "smr_topsnp_match", "smr_gene_supported"]:
    if col not in m_clean.columns:
        continue
    s = m_clean[col]
    if s.dtype.kind in {"i", "u", "f"}:
        if col in {"atac_any", "har_overlap", "smr_topsnp_match", "smr_gene_supported"}:
            n = int((s == 1).sum())
        elif col == "desert_tier":
            n = int((s > 0).sum())
        else:
            n = int(s.notna().sum())
    else:
        n = int(s.notna().sum())
    print(f"  {col:<28s}: {n:>6d} ({n/len(m_clean)*100:.1f}%)")
