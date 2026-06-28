#!/bin/bash
# Parallel LD score computation for P17b annotations
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline
OUT_TOP=$BASE/results/phase17b/brain_spec_top1742_annot
OUT_CONT=$BASE/results/phase17b/brain_spec_continuous_annot

mkdir -p /tmp/p17b_logs
for chrom in {1..22}; do
  if [ ! -f /tmp/baseline_chr${chrom}_snplist.txt ]; then
    gzip -cd $BASELINE/baseline.${chrom}.l2.ldscore.gz | awk 'NR>1{print $2}' > /tmp/baseline_chr${chrom}_snplist.txt
  fi
done

run_top() {
  local c=$1
  if [ -f $OUT_TOP/brain_spec_high.${c}.l2.ldscore.gz ]; then echo "  Atop chr${c}: SKIP"; return; fi
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.${c} --ld-wind-cm 1 \
    --annot $OUT_TOP/brain_spec_high.${c}.annot.gz --print-snps /tmp/baseline_chr${c}_snplist.txt \
    --out $OUT_TOP/brain_spec_high.${c} > /tmp/p17b_logs/top_chr${c}.log 2>&1
  echo "  A chr${c}: done"
}

run_cont() {
  local c=$1
  if [ -f $OUT_CONT/brain_spec.${c}.l2.ldscore.gz ]; then echo "  Bcont chr${c}: SKIP"; return; fi
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.${c} --ld-wind-cm 1 \
    --annot $OUT_CONT/brain_spec.${c}.annot.gz --print-snps /tmp/baseline_chr${c}_snplist.txt \
    --out $OUT_CONT/brain_spec.${c} > /tmp/p17b_logs/cont_chr${c}.log 2>&1
  echo "  B chr${c}: done"
}

export -f run_top run_cont
export LDSC PLINK BASELINE BASE OUT_TOP OUT_CONT

# Annotation A (top1742): parallel 4
seq 1 22 | xargs -P 4 -I{} bash -c 'run_top "$@"' _ {}
echo "Annotation A done."

# Annotation B (continuous): parallel 4
seq 1 22 | xargs -P 4 -I{} bash -c 'run_cont "$@"' _ {}
echo "Annotation B done."

echo "All P17b LD scores computed."
