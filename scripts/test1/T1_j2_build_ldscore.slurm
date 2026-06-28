#!/bin/bash
#SBATCH -J t1_build
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 03:55:00
#SBATCH -c 22
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH -o logs/j2_build_%j.out
#SBATCH -e logs/j2_build_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
AGE=${EVOSCZ_SCRATCH}/test1_age_conditioning/age_annot
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
PLINK=$B/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
SNPLIST=$B/results/phase17d_v22/snplist

# 1) build standardized + decile age annotations aligned to baseline
python3 $S/T1_build_age_annot.py

# 2) LD scores for all 3 annotation sets, 22 chr each (66 tasks). Keep per-task log (no /dev/null).
run_ld(){
  local name=$1 c=$2
  local out=$AGE/${name}.$c
  [ -s ${out}.l2.ldscore.gz ] && [ -s ${out}.l2.M_5_50 ] && [ -s ${out}.l2.M ] && { echo "skip $name $c (exists)"; return 0; }
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.$c --ld-wind-cm 1 \
    --annot $AGE/${name}.$c.annot.gz \
    --print-snps $SNPLIST/baselineLD_v22_chr${c}_snplist.txt \
    --out ${out} > ${out}.ldlog 2>&1 || { echo "LDFAIL $name $c (see ${out}.ldlog)"; return 1; }
}
export -f run_ld; export LDSC PLINK SNPLIST AGE
PAIRS=${EVOSCZ_SCRATCH}/test1_age_conditioning/logs/ldpairs.txt
: > $PAIRS
for name in age_mut age_mut_dec age_jnt; do for c in $(seq 1 22); do echo "$name $c" >> $PAIRS; done; done
xargs -P 22 -L1 -a $PAIRS bash -c 'run_ld "$@"' _
echo "J2_BUILD_LDSCORE_DONE"
