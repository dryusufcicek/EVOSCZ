#!/bin/bash
#SBATCH -J t1_jnt
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 03:55:00
#SBATCH -c 22
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH -o logs/jnt_%j.out
#SBATCH -e logs/jnt_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
W=${EVOSCZ_SCRATCH}/test1_age_conditioning
O=$W/results_robust; mkdir -p $O

# 1) re-cluster: mut (sanity vs published) + jnt (preferred clock)
python3 $S/T1_recluster.py mut
python3 $S/T1_recluster.py jnt

# 2) build one-hot annots (jnt k3, jnt k2) aligned to baseline-LD v2.2
python3 $S/T1_recluster_build_annot.py $W/recluster/assign_jnt_k3.tsv.gz jnt_k3 3
python3 $S/T1_recluster_build_annot.py $W/recluster/assign_jnt_k2.tsv.gz jnt_k2 2

# 3) LD scores (EUR plink + EUR v22 snplist)
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
P=$W/logs/jnt_pairs.txt; : > $P
for sub in jnt_k3 jnt_k2; do for c in $(seq 1 22); do echo "$sub $c" >> $P; done; done
xargs -P 22 -L1 -a $P bash -c 'run_ld "$@"' _
echo "JNT_LDSCORE_DONE"

# 4) joint models (baseline-LD v2.2 + jnt cluster, EUR sumstats)
BASE=$B/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2/baselineLD.
WLD=$B/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$B/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SS=$B/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
python3 $LDSC --h2 $SS --ref-ld-chr "${BASE},$RA/jnt_k3/cluster." --w-ld-chr $WLD --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients --print-delete-vals --out $O/jnt_k3_baseline_cluster > $O/jnt_k3.runlog 2>&1 \
  || { echo "MODELFAIL jnt_k3"; exit 1; }
python3 $LDSC --h2 $SS --ref-ld-chr "${BASE},$RA/jnt_k2/cluster." --w-ld-chr $WLD --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients --print-delete-vals --out $O/jnt_k2_baseline_cluster > $O/jnt_k2.runlog 2>&1 \
  || { echo "MODELFAIL jnt_k2"; exit 1; }
echo "T1_JNT_DONE"
