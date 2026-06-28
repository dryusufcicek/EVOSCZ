#!/bin/bash
#SBATCH -J t1_subset
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 02:00:00
#SBATCH -c 22
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH -o logs/subset_%j.out
#SBATCH -e logs/subset_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
W=${EVOSCZ_SCRATCH}/test1_age_conditioning
O=$W/results_robust; mkdir -p $O

python3 $S/T1_subset_build.py

PLINK=$B/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
SNPLIST=$B/results/phase17d_v22/snplist
RA=$W/recluster_annot/subset_c0
run_ld(){
  local c=$1; local out=$RA/cluster.$c
  [ -s ${out}.l2.ldscore.gz ] && [ -s ${out}.l2.M ] && [ -s ${out}.l2.M_5_50 ] && return 0
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.$c --ld-wind-cm 1 \
    --annot $RA/cluster.$c.annot.gz --print-snps $SNPLIST/baselineLD_v22_chr${c}_snplist.txt \
    --out ${out} > ${out}.ldlog 2>&1 || { echo "LDFAIL $c"; return 1; }
}
export -f run_ld; export LDSC PLINK SNPLIST RA
seq 1 22 | xargs -P 22 -I{} bash -c 'run_ld "$@"' _ {}
echo "SUBSET_LDSCORE_DONE"

BASE=$B/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2/baselineLD.
WLD=$B/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$B/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SS=$B/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
python3 $LDSC --h2 $SS --ref-ld-chr "${BASE},$RA/cluster." --w-ld-chr $WLD --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients --print-delete-vals --out $O/subset_c0_baseline > $O/subset_c0.runlog 2>&1 \
  || { echo "MODELFAIL subset_c0"; exit 1; }
echo "T1_SUBSET_DONE"
