#!/bin/bash
# Phase 17C: Compute LD scores for power-matched cluster annotations (22 chr).
# Parallel via xargs -P 3 (RAM cap ~6 GB, safe alongside P17b).
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline
OUT=$BASE/results/phase17c/cluster_annot_power_matched

mkdir -p /tmp/p17c_logs

# Pre-extract HM3 SNP lists if not done
for chrom in {1..22}; do
  if [ ! -f /tmp/baseline_chr${chrom}_snplist.txt ]; then
    gzip -cd $BASELINE/baseline.${chrom}.l2.ldscore.gz | awk 'NR>1{print $2}' > /tmp/baseline_chr${chrom}_snplist.txt
  fi
done

run_one() {
  local chrom=$1
  if [ -f $OUT/cluster.${chrom}.l2.ldscore.gz ]; then
    echo "  chr${chrom}: SKIP (already done)"
    return 0
  fi
  python3 $LDSC \
    --l2 \
    --bfile $PLINK/1000G.EUR.QC.${chrom} \
    --ld-wind-cm 1 \
    --annot $OUT/cluster.${chrom}.annot.gz \
    --print-snps /tmp/baseline_chr${chrom}_snplist.txt \
    --out $OUT/cluster.${chrom} \
    > /tmp/p17c_logs/chr${chrom}.log 2>&1
  echo "  chr${chrom}: done"
}
export -f run_one
export OUT LDSC PLINK BASELINE BASE

seq 1 22 | xargs -P 3 -I{} bash -c 'run_one "$@"' _ {}

echo "All P17c LD scores computed."
