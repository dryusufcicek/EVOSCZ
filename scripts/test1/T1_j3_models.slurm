#!/bin/bash
#SBATCH -J t1_models
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 02:30:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --no-requeue
#SBATCH -o logs/j3_models_%j.out
#SBATCH -e logs/j3_models_%j.err
set -e
B=${EVOSCZ_ROOT}
S=$B/scripts/test1
W=${EVOSCZ_SCRATCH}/test1_age_conditioning
LDSC=$B/scripts/phase14e/ldsc/ldsc.py
BASE=$B/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2/baselineLD.
CLUST=$B/results/phase14e/cluster_annot/cluster.
AGEM=$W/age_annot/age_mut.
AGEMD=$W/age_annot/age_mut_dec.
AGEJ=$W/age_annot/age_jnt.
WLD=$B/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$B/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SS=$B/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
O=$W/results; mkdir -p $O

run(){
  python3 $LDSC --h2 $SS --ref-ld-chr "$1" --w-ld-chr $WLD --frqfile-chr $FRQ \
    --overlap-annot --print-coefficients --print-delete-vals --out "$2" > "$2.runlog" 2>&1 \
    || { echo "MODELFAIL $2 (see $2.runlog)"; return 1; }
  echo "OK $2"
}
# M0 — baseline + cluster (already conditions on baseline allele-age proxies); reproduce C0 tau/Z (~+3.05)
run "${BASE},${CLUST}"          $O/M0_baseline_cluster
# M1 — + GEVA continuous Mut-clock age (linear)
run "${BASE},${CLUST},${AGEM}"  $O/M1_cluster_age_mut
# M1d — PRIMARY: + GEVA Mut-clock age DECILES (flexible age control)
run "${BASE},${CLUST},${AGEMD}" $O/M1d_cluster_age_mut_dec
# M2 — robustness: + GEVA continuous Jnt-clock age (linear)
run "${BASE},${CLUST},${AGEJ}"  $O/M2_cluster_age_jnt

python3 $S/T1_summarize.py
echo "J3_MODELS_DONE"
