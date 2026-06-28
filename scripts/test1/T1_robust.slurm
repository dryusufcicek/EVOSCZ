#!/bin/bash
#SBATCH -J t1_robust
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 03:55:00
#SBATCH -c 22
#SBATCH --mem=48G
#SBATCH --no-requeue
#SBATCH -o logs/robust_%j.out
#SBATCH -e logs/robust_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
W=${EVOSCZ_SCRATCH}/test1_age_conditioning
O=$W/results_robust; mkdir -p $O

# ===== EAS joint model (EAS reference + existing EAS cluster LD scores) =====
R=$B/data/ldsc/sldsc_ref_eas
python3 $LDSC --h2 $B/results/phase14h/PGC3_SCZ_EAS.sumstats.gz \
  --ref-ld-chr "$R/1000G_EAS_Phase3_baseline/baseline.,$B/results/phase14h/cluster_annot_eas/cluster." \
  --w-ld-chr $R/1000G_Phase3_EAS_weights_hm3_no_MHC/weights.EAS.hm3_noMHC. \
  --frqfile-chr $R/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC. \
  --overlap-annot --print-coefficients --print-delete-vals \
  --out $O/EAS_baseline_cluster > $O/EAS.runlog 2>&1 || { echo "EAS_MODELFAIL (see $O/EAS.runlog)"; exit 1; }
echo "EAS_DONE"

# ===== build k=2 / k=4 multi-cluster annotations (EUR, aligned to baseline-LD v2.2) =====
python3 $S/T1_build_kcluster_annot.py 2 4

# ===== LD scores for k2/k4 (EUR plink + EUR snplist) =====
PLINK=$B/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
SNPLIST=$B/results/phase17d_v22/snplist
KA=$W/kcluster_annot
run_ld(){
  local k=$1 c=$2; local out=$KA/k$k/cluster.$c
  [ -s ${out}.l2.ldscore.gz ] && [ -s ${out}.l2.M ] && [ -s ${out}.l2.M_5_50 ] && return 0
  python3 $LDSC --l2 --bfile $PLINK/1000G.EUR.QC.$c --ld-wind-cm 1 \
    --annot $KA/k$k/cluster.$c.annot.gz --print-snps $SNPLIST/baselineLD_v22_chr${c}_snplist.txt \
    --out ${out} > ${out}.ldlog 2>&1 || { echo "LDFAIL k$k $c"; return 1; }
}
export -f run_ld; export LDSC PLINK SNPLIST KA
P=$W/logs/krobust_pairs.txt; : > $P
for k in 2 4; do for c in $(seq 1 22); do echo "$k $c" >> $P; done; done
xargs -P 22 -L1 -a $P bash -c 'run_ld "$@"' _
echo "KLDSCORE_DONE"

# ===== k2/k4 joint models (EUR baseline-LD v2.2) =====
BASE=$B/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2/baselineLD.
WLD=$B/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$B/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SS=$B/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
for k in 2 4; do
  python3 $LDSC --h2 $SS --ref-ld-chr "${BASE},$KA/k$k/cluster." --w-ld-chr $WLD --frqfile-chr $FRQ \
    --overlap-annot --print-coefficients --print-delete-vals \
    --out $O/k${k}_baseline_cluster > $O/k${k}.runlog 2>&1 || { echo "MODELFAIL k$k"; exit 1; }
done
echo "T1_ROBUST_DONE"
