#!/bin/bash
# Phase 17B: Compute LD scores for brain_spec-only annotations (22 chr × 2 annot).
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline

OUT_TOP=$BASE/results/phase17b/brain_spec_top1742_annot
OUT_CONT=$BASE/results/phase17b/brain_spec_continuous_annot

mkdir -p /tmp/p17b_logs

# Pre-extract HM3 SNP lists per chromosome from baseline LD scores
for chrom in {1..22}; do
  if [ ! -f /tmp/baseline_chr${chrom}_snplist.txt ]; then
    gzip -cd $BASELINE/baseline.${chrom}.l2.ldscore.gz | awk 'NR>1{print $2}' > /tmp/baseline_chr${chrom}_snplist.txt
  fi
done

# Annotation A: top-1742 binary
for chrom in {1..22}; do
  if [ -f $OUT_TOP/brain_spec_high.${chrom}.l2.ldscore.gz ]; then
    continue
  fi
  python3 $LDSC \
    --l2 \
    --bfile $PLINK/1000G.EUR.QC.${chrom} \
    --ld-wind-cm 1 \
    --annot $OUT_TOP/brain_spec_high.${chrom}.annot.gz \
    --print-snps /tmp/baseline_chr${chrom}_snplist.txt \
    --out $OUT_TOP/brain_spec_high.${chrom} \
    > /tmp/p17b_logs/top_chr${chrom}.log 2>&1
  echo "  A chr${chrom} done"
done

# Annotation B: continuous (--thin-annot would clash; build_annot already include CHR/BP/SNP/CM)
for chrom in {1..22}; do
  if [ -f $OUT_CONT/brain_spec.${chrom}.l2.ldscore.gz ]; then
    continue
  fi
  python3 $LDSC \
    --l2 \
    --bfile $PLINK/1000G.EUR.QC.${chrom} \
    --ld-wind-cm 1 \
    --annot $OUT_CONT/brain_spec.${chrom}.annot.gz \
    --print-snps /tmp/baseline_chr${chrom}_snplist.txt \
    --out $OUT_CONT/brain_spec.${chrom} \
    > /tmp/p17b_logs/cont_chr${chrom}.log 2>&1
  echo "  B chr${chrom} done"
done

echo "All LD scores computed."
