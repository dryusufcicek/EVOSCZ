#!/bin/bash
# Compute LD scores for C0-only annotation, 4-way parallel
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline
OUT=$BASE/results/phase17e/c0_only_annot

mkdir -p /tmp/p17e_logs
for chrom in {1..22}; do
  if [ ! -f /tmp/baseline_chr${chrom}_snplist.txt ]; then
    gzip -cd $BASELINE/baseline.${chrom}.l2.ldscore.gz | awk 'NR>1{print $2}' > /tmp/baseline_chr${chrom}_snplist.txt
  fi
done

run_one() {
  local c=$1
  if [ -f $OUT/c0_only.${c}.l2.ldscore.gz ]; then echo "  chr${c}: SKIP"; return; fi
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.${c} --ld-wind-cm 1 \
    --annot $OUT/c0_only.${c}.annot.gz --print-snps /tmp/baseline_chr${c}_snplist.txt \
    --out $OUT/c0_only.${c} > /tmp/p17e_logs/chr${c}.log 2>&1
  echo "  chr${c}: done"
}
export -f run_one
export LDSC PLINK OUT

seq 1 22 | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "All P17e LD scores done."
