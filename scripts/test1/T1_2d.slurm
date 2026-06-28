#!/bin/bash
#SBATCH -J t1_2d
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 03:55:00
#SBATCH -c 22
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH -o logs/t2d_%j.out
#SBATCH -e logs/t2d_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
W=${EVOSCZ_SCRATCH}/test1_age_conditioning
O=$W/results_robust; mkdir -p $O

# 1) re-cluster the 3 two-feature decompositions (concern 4)
for combo in age_brain ihs_brain age_ihs; do
  python3 $S/T1_recluster_2d.py $combo
  python3 $S/T1_recluster_build_annot.py $W/recluster/assign_2d_${combo}_k3.tsv.gz 2d_${combo} 3
done

# 2) LD scores (EUR plink + EUR v22 snplist)
PLINK=$B/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
SNPLIST=$B/results/phase17d_v22/snplist
RA=$W/recluster_annot
run_ld(){
  local sub=$1 c=$2; local out=$RA/$sub/cluster.$c
  [ -s ${out}.l2.ldscore.gz ] && [ -s ${out}.l2.M ] && [ -s ${out}.l2.M_5_50 ] && return 0
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.$c --ld-wind-cm 1 \
    --annot $RA/$sub/cluster.$c.annot.gz --print-snps $SNPLIST/baselineLD_v22_chr${c}_snplist.txt \
    --out ${out} > ${out}.ldlog 2>&1 || { echo "LDFAIL $sub $c"; return 1; }
}
export -f run_ld; export LDSC PLINK SNPLIST RA
P=$W/logs/t2d_pairs.txt; : > $P
for combo in age_brain ihs_brain age_ihs; do for c in $(seq 1 22); do echo "2d_${combo} $c" >> $P; done; done
xargs -P 22 -L1 -a $P bash -c 'run_ld "$@"' _
echo "T2D_LDSCORE_DONE"

# 3) joint models (baseline-LD v2.2 + 2D cluster, EUR sumstats)
BASE=$B/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2/baselineLD.
WLD=$B/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$B/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SS=$B/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
for combo in age_brain ihs_brain age_ihs; do
  python3 $LDSC --h2 $SS --ref-ld-chr "${BASE},$RA/2d_${combo}/cluster." --w-ld-chr $WLD --frqfile-chr $FRQ \
    --overlap-annot --print-coefficients --print-delete-vals --out $O/2d_${combo}_baseline_cluster > $O/2d_${combo}.runlog 2>&1 \
    || { echo "MODELFAIL 2d_${combo}"; exit 1; }
done
echo "T1_2D_DONE"
