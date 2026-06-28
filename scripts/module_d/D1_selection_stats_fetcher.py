#!/usr/bin/env python3
"""
EVOSCZ Pipeline — Module D: Selection Signatures
==================================================
Step D1: FTP Stream-based Selection Statistic Computation
Instead of downloading 15 GB of VCFs, this script dynamically
streams 200kb windows around each PGC3 locus using tabix/pysam
from the 1000 Genomes EBI FTP server.

Computes:
- Tajima's D (balancing / directional selection)
- Garud's H12, H2H1 (hard vs soft sweeps)
- nSL (number of segregating sites by length, max over window)
"""

import sys
import time
import pysam
import allel
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_TABLE = PROJECT_ROOT / "data/processed/pgc3_master_variants.tsv"
EUR_SAMPLES  = PROJECT_ROOT / "data/raw/1kgp/eur_samples.txt"
OUTPUT_TSV   = PROJECT_ROOT / "results/module_d/D1_empirical_selection_stats.tsv"

FTP_TEMPLATE = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

def compute_statistics_for_locus(chrom, pos, eur_samples):
    window_start = max(1, pos - 100000)
    window_end = pos + 100000
    
    url = FTP_TEMPLATE.format(chrom)
    
    try:
        vf = pysam.VariantFile(url)
    except Exception as e:
        print(f"  Error connecting to {url}: {e}")
        return None
    
    samples = [s for s in vf.header.samples if s in eur_samples]
    vf.subset_samples(samples)
    
    try:
        records = list(vf.fetch(str(chrom), window_start, window_end))
    except Exception as e:
        print(f"  Error fetching chr{chrom}:{window_start}-{window_end}: {e}")
        return None
        
    if not records:
        return None

    # Parse genotypes
    gt_array = []
    pos_array = []
    for r in records:
        # Only use biallelic SNPs
        if len(r.alleles) == 2 and len(r.ref) == 1 and len(r.alts[0]) == 1:
            gts = [s['GT'] for s in r.samples.values()]
            if any(None in gt for gt in gts):
                continue
            gt_array.append(gts)
            pos_array.append(r.pos)
    
    if len(pos_array) < 10:
        return None

    gt = allel.GenotypeArray(gt_array)
    pos_arr = np.array(pos_array)
    h = gt.to_haplotypes()
    
    # 1. Tajima's D
    ac = gt.count_alleles()
    # Ensure segregating sites only
    is_seg = ac.is_segregating()
    if not np.any(is_seg):
        return None
    
    td = allel.tajima_d(ac, pos_arr, window_start, window_end)
    if np.isnan(td): td = 0.0
    
    # 2. Garud's H12, H2H1
    # We restrict to segregating sites for H computation
    h_seg = h.compress(is_seg, axis=0)
    if h_seg.shape[0] > 0:
        h1, h12, h123, h2h1 = allel.garud_h(h_seg)
    else:
        h12, h2h1 = np.nan, np.nan
        
    # 3. nSL (using a proxy: we take the maximum absolute nSL in the window to detect local sweeps)
    try:
        nsl = allel.nsl(h_seg)
        valid_nsl = nsl[~np.isnan(nsl)]
        if len(valid_nsl) > 0:
            max_nsl = np.max(np.abs(valid_nsl))
        else:
            max_nsl = np.nan
    except:
        max_nsl = np.nan

    return {
        'tajimas_d': td,
        'h12': h12,
        'h2h1': h2h1,
        'max_abs_nsl': max_nsl,
        'n_snps': len(pos_array)
    }

def main():
    print("=" * 60)
    print("EVOSCZ Module D: Streaming 1KGP Selection Statistics")
    print("=" * 60)

    # Load samples
    if not EUR_SAMPLES.exists():
        print("Missing EUR samples list!")
        sys.exit(1)
        
    with open(EUR_SAMPLES) as f:
        eur_set = set(f.read().splitlines())
        
    df = pd.read_csv(MASTER_TABLE, sep='\t')
    df_auto = df[df['chr'].apply(lambda x: str(x).isdigit())].copy()
    
    # Find lead SNP for each locus
    lead_snps = df_auto.sort_values('pip', ascending=False).groupby('credible_set_id').first().reset_index()
    total = len(lead_snps)
    print(f"Targeting {total} loci...")
    
    # Checkpoint logic
    if OUTPUT_TSV.exists():
        results = pd.read_csv(OUTPUT_TSV, sep='\t').to_dict('records')
        processed = {r['credible_set_id'] for r in results}
        print(f"Resuming: {len(processed)} already processed.")
    else:
        results = []
        processed = set()
    
    start_time = time.time()
    
    for i, row in lead_snps.iterrows():
        locus = row['credible_set_id']
        chrom = int(row['chr'])
        pos = int(row['pos'])
        
        if locus in processed:
            continue
            
        print(f"[{len(results)+1}/{total}] Fetching {locus} (chr{chrom}:{pos})...", end=" ", flush=True)
        stats = compute_statistics_for_locus(chrom, pos, eur_set)
        
        if stats:
            stats.update({
                'credible_set_id': locus,
                'chr': chrom,
                'lead_pos': pos,
                'lead_rsid': row['rsid']
            })
            results.append(stats)
            print(f"Done (n={stats['n_snps']})")
        else:
            print("Failed.")
            
        # Save incrementally
        if len(results) % 10 == 0 or len(results) == total:
            pd.DataFrame(results).to_csv(OUTPUT_TSV, sep='\t', index=False)
            
    print(f"Completed in {(time.time() - start_time)/60:.1f} minutes.")
    print(f"Saved to {OUTPUT_TSV}")

if __name__ == "__main__":
    main()
