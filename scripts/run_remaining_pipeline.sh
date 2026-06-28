#!/bin/bash
# Sequential remainder of the code-review-fix pipeline (2-way parallel max)
# Runs after current EUR LDSC + 2 regressions finish; user is away ~8h.

set -uo pipefail
LOG=/tmp/run_remaining_pipeline.log
exec > >(tee -a "$LOG") 2>&1

BASE=${EVOSCZ_ROOT}
LDSC_PY=$BASE/scripts/phase14e/ldsc/ldsc.py
EUR_REF=$BASE/data/ldsc/sldsc_ref
EAS_REF=$BASE/data/ldsc/sldsc_ref_eas
EUR_ANNOT=$BASE/results/phase14e/cluster_annot
EAS_ANNOT=$BASE/results/phase14h/cluster_annot_eas

step() {
  echo ""
  echo "=========================================================="
  echo "STEP: $1 — $(date '+%H:%M:%S')"
  echo "=========================================================="
}

wait_for_ldsc_quiet() {
  # wait until no ldsc.py procs are running (regression + lds finish)
  while pgrep -f "ldsc.py" >/dev/null 2>&1; do
    sleep 30
  done
  echo "  no ldsc.py procs running"
  sleep 10
}

# ── Step 1: wait for current SCZ + AD regression + chr 9-12 batch to finish ──
step "WAIT for current EUR LDSC + 2 h2 regressions to finish"
wait_for_ldsc_quiet

# ── Step 2: re-extract SCZ EUR results timestamp check ──
step "Verify EUR SCZ + AD regression results"
ls -la $BASE/results/phase14e/PGC3_SCZ_clusters_baseline.results 2>/dev/null
ls -la $BASE/results/phase14f/Wightman_AD_clusters_baseline.results 2>/dev/null

# ── Step 3: EAS LDSC LD scoring at 2-way parallel ──
step "EAS LDSC LD scoring (2-way parallel)"
mkdir -p $BASE/logs/phase14h_ldsc
for batch_start in 1 3 5 7 9 11 13 15 17 19 21; do
  for offset in 0 1; do
    CHR=$((batch_start + offset))
    if [ $CHR -gt 22 ]; then continue; fi
    if [ -f $EAS_ANNOT/cluster.${CHR}.l2.ldscore.gz ]; then
      # Check if it's NEW (post-cutoff)
      MT=$(stat -f '%m' $EAS_ANNOT/cluster.${CHR}.l2.ldscore.gz)
      CUTOFF=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-05-03 12:30:00' '+%s')
      if [ $MT -ge $CUTOFF ]; then
        echo "  chr${CHR}: SKIP (NEW already done)"
        continue
      fi
    fi
    python3 $LDSC_PY --l2 \
      --bfile $EAS_REF/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC.${CHR} \
      --ld-wind-cm 1 \
      --annot $EAS_ANNOT/cluster.${CHR}.annot.gz \
      --out $EAS_ANNOT/cluster.${CHR} \
      --print-snps $EAS_REF/eas_baseline_snps.txt \
      > $BASE/logs/phase14h_ldsc/chr${CHR}.log 2>&1 &
  done
  wait
  echo "  EAS batch ${batch_start} done at $(date '+%H:%M:%S')"
  sleep 5
done

# ── Step 4: filter EAS cluster ldscores to baseline regression-SNP set ──
step "Filter EAS cluster ldscores to baseline regression set"
python3 $BASE/scripts/phase14h/P14h_filter_cluster_ldscores.py 2>&1 | tail -25

# ── Step 5: EAS h² regression ──
step "EAS h² regression"
cd $BASE/results/phase14h
python3 $LDSC_PY --h2 PGC3_SCZ_EAS.sumstats.gz \
  --ref-ld-chr cluster_annot_eas/cluster.,$EAS_REF/1000G_EAS_Phase3_baseline/baseline. \
  --w-ld-chr $EAS_REF/1000G_Phase3_EAS_weights_hm3_no_MHC/weights.EAS.hm3_noMHC. \
  --overlap-annot \
  --frqfile-chr $EAS_REF/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC. \
  --out PGC3_SCZ_EAS_clusters_baseline \
  --print-coefficients 2>&1 | tail -20

# ── Step 6: P14g LR-LD masking + LDSC + regression ──
step "P14g LR-LD masking + LD scoring (2-way parallel) + regression"
python3 $BASE/scripts/phase14g/P14g_lr_ld_exclusion.py 2>&1 | tail -10
sleep 5
mkdir -p $BASE/logs/phase14g_ldsc
LRLD_ANNOT=$BASE/results/phase14g/cluster_annot_no_lrld
for batch_start in 1 3 5 7 9 11 13 15 17 19 21; do
  for offset in 0 1; do
    CHR=$((batch_start + offset))
    if [ $CHR -gt 22 ]; then continue; fi
    python3 $LDSC_PY --l2 \
      --bfile $EUR_REF/1000G_EUR_Phase3_plink/1000G.EUR.QC.${CHR} \
      --ld-wind-cm 1 \
      --annot $LRLD_ANNOT/cluster.${CHR}.annot.gz \
      --out $LRLD_ANNOT/cluster.${CHR} \
      --print-snps $EUR_REF/w_hm3.snplist \
      > $BASE/logs/phase14g_ldsc/chr${CHR}.log 2>&1 &
  done
  wait
  echo "  P14g LDSC batch ${batch_start} done at $(date '+%H:%M:%S')"
  sleep 5
done
cd $BASE/results/phase14g
python3 $LDSC_PY --h2 $BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz \
  --ref-ld-chr cluster_annot_no_lrld/cluster.,$EUR_REF/1000G_EUR_Phase3_baseline/baseline. \
  --w-ld-chr $EUR_REF/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. \
  --overlap-annot \
  --frqfile-chr $EUR_REF/1000G_Phase3_frq/1000G.EUR.QC. \
  --out PGC3_SCZ_clusters_no_lrld \
  --print-coefficients 2>&1 | tail -20
python3 $LDSC_PY --h2 $BASE/results/phase14f/Wightman_AD.sumstats.gz \
  --ref-ld-chr cluster_annot_no_lrld/cluster.,$EUR_REF/1000G_EUR_Phase3_baseline/baseline. \
  --w-ld-chr $EUR_REF/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. \
  --overlap-annot \
  --frqfile-chr $EUR_REF/1000G_Phase3_frq/1000G.EUR.QC. \
  --out Wightman_AD_clusters_no_lrld \
  --print-coefficients 2>&1 | tail -20

# ── Step 7: P13a paired matched control test ──
step "P13a paired matched control test"
python3 $BASE/scripts/phase13/P13a_matched_control_comparison.py 2>&1 | tail -30

step "ALL PIPELINE STEPS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Manuscript + master doc update will be done by Claude in foreground."
