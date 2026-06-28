#!/bin/bash
# Phase 18b: Comprehensive sensitivity battery with baseline-LD v2.2
# All sensitivity tests re-run with v2.2 for consistency with primary analysis.
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
V22_EUR=$BASE/data/ldsc/sldsc_ref_v2.2/EUR/baselineLD.
V22_EAS=$BASE/data/ldsc/sldsc_ref_v2.2/EAS/baselineLD.
WLD=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
WLD_EAS=$BASE/data/ldsc/sldsc_ref_eas/1000G_Phase3_EAS_weights_hm3_no_MHC/weights.EAS.hm3_noMHC.
FRQ=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
FRQ_EAS=$BASE/data/ldsc/sldsc_ref_eas/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC.
PLINK_EUR=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink/1000G.EUR.QC.

# Sumstats
SCZ_EUR=$BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
SCZ_EAS=$BASE/results/phase14h/PGC3_SCZ_EAS.sumstats.gz
AD=$BASE/results/phase14f/Wightman_AD.sumstats.gz
SCZ_EUR_M=$BASE/results/phase14j/PGC3_SCZ_EUR_male.sumstats.gz
SCZ_EUR_F=$BASE/results/phase14j/PGC3_SCZ_EUR_female.sumstats.gz
SCZ_EAS_M=$BASE/results/phase14j/PGC3_SCZ_EAS_male.sumstats.gz
SCZ_EAS_F=$BASE/results/phase14j/PGC3_SCZ_EAS_female.sumstats.gz

# Cluster annotations
PRIMARY_EUR=$BASE/results/phase14e/cluster_annot/cluster.
PRIMARY_EAS=$BASE/results/phase14h/cluster_annot_eas/cluster.
LRLD=$BASE/results/phase14g/cluster_annot_no_lrld/cluster.
TWO_D=$BASE/results/phase14b/cluster_annot_2d/cluster.
MAPT=$BASE/results/phase17a/cluster_annot_mapt/cluster.
BS_TOP=$BASE/results/phase17b/brain_spec_top1742_annot/brain_spec_high.
BS_CONT=$BASE/results/phase17b/brain_spec_continuous_annot/brain_spec.
POWMATCH=$BASE/results/phase17c/cluster_annot_power_matched/cluster.
C0_ONLY=$BASE/results/phase17e/c0_only_annot/c0_only.

OUT=$BASE/results/phase18_v22
mkdir -p $OUT $OUT/logs

# Helper for HM3 SNP list
HM3_LIST=/tmp/hm3_snplist_full.txt
if [ ! -f $HM3_LIST ]; then
  for chr in {1..22}; do
    if [ ! -f /tmp/baseline_chr${chr}_snplist.txt ]; then
      gzip -cd $BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline/baseline.${chr}.l2.ldscore.gz | awk 'NR>1{print $2}' > /tmp/baseline_chr${chr}_snplist.txt
    fi
  done
fi

# === COMPUTE MISSING LD SCORES (young_annot_k2/k4) ===
echo "============================================================"
echo "[A] Computing missing LD scores for young_annot_k2/k4"
echo "============================================================"

for K in 2 4; do
  YOUNG_ANNOT=$BASE/results/phase14b/young_annot_k${K}
  for chr in {1..22}; do
    if [ ! -f $YOUNG_ANNOT/young.${chr}.l2.ldscore.gz ]; then
      python3 $LDSC --l2 \
        --bfile ${PLINK_EUR}${chr} \
        --ld-wind-cm 1 \
        --annot $YOUNG_ANNOT/young.${chr}.annot.gz \
        --print-snps /tmp/baseline_chr${chr}_snplist.txt \
        --out $YOUNG_ANNOT/young.${chr} \
        > $OUT/logs/k${K}_chr${chr}.log 2>&1
    fi
  done
  echo "  young_annot_k${K}: LD scores ready"
done

# === S-LDSC HELPER ===
run_sldsc_v22() {
  local tag=$1
  local sumstats=$2
  local cluster=$3
  local baseline=$4
  local wld=$5
  local frq=$6
  local extra_flags="${7:-}"
  if [ -f "$OUT/$tag.results" ]; then echo "  $tag: SKIP"; return 0; fi
  echo "  Running $tag..."
  python3 $LDSC \
    --h2 $sumstats \
    --ref-ld-chr $baseline,$cluster \
    --w-ld-chr $wld \
    --frqfile-chr $frq \
    --overlap-annot --print-coefficients $extra_flags \
    --out $OUT/$tag > $OUT/logs/$tag.log 2>&1
  if [ -f "$OUT/$tag.results" ]; then
    # Extract last annotation row
    last=$(grep -E "^L2_1|^C0L2|^BRAIN" $OUT/$tag.results | head -1)
    if [ -n "$last" ]; then
      enr=$(echo "$last" | awk -F'\t' '{print $5}')
      p=$(echo "$last" | awk -F'\t' '{print $7}')
      z=$(echo "$last" | awk -F'\t' '{print $10}')
      echo "    Last annot: enrich=$enr  P=$p  Z=$z"
    fi
  else
    echo "    FAILED"
  fi
}

# === BATTERY ===
echo ""
echo "============================================================"
echo "[B] Sensitivity battery v2.2 (S-LDSC runs)"
echo "============================================================"

# Sex-stratified
run_sldsc_v22 "SCZ_EUR_M_v22"    $SCZ_EUR_M $PRIMARY_EUR $V22_EUR $WLD $FRQ
run_sldsc_v22 "SCZ_EUR_F_v22"    $SCZ_EUR_F $PRIMARY_EUR $V22_EUR $WLD $FRQ
run_sldsc_v22 "SCZ_EAS_M_v22"    $SCZ_EAS_M $PRIMARY_EAS $V22_EAS $WLD_EAS $FRQ_EAS
run_sldsc_v22 "SCZ_EAS_F_v22"    $SCZ_EAS_F $PRIMARY_EAS $V22_EAS $WLD_EAS $FRQ_EAS

# LR-LD masked
run_sldsc_v22 "SCZ_EUR_LRLD_v22" $SCZ_EUR  $LRLD $V22_EUR $WLD $FRQ
run_sldsc_v22 "AD_LRLD_v22"      $AD       $LRLD $V22_EUR $WLD $FRQ

# n_blocks=1000 Tashman
run_sldsc_v22 "SCZ_EUR_n1000_v22" $SCZ_EUR $PRIMARY_EUR $V22_EUR $WLD $FRQ "--n-blocks 1000"

# k=2 / k=4 GMM sensitivity
run_sldsc_v22 "SCZ_EUR_k2_v22"   $SCZ_EUR $BASE/results/phase14b/young_annot_k2/young. $V22_EUR $WLD $FRQ
run_sldsc_v22 "SCZ_EUR_k4_v22"   $SCZ_EUR $BASE/results/phase14b/young_annot_k4/young. $V22_EUR $WLD $FRQ

# 2D feature-space
run_sldsc_v22 "SCZ_EUR_2D_v22"   $SCZ_EUR $TWO_D $V22_EUR $WLD $FRQ

# MAPT-included
run_sldsc_v22 "SCZ_EUR_MAPT_v22" $SCZ_EUR $MAPT $V22_EUR $WLD $FRQ

# Brain_spec-only
run_sldsc_v22 "SCZ_EUR_BSTOP_v22" $SCZ_EUR $BS_TOP $V22_EUR $WLD $FRQ
run_sldsc_v22 "SCZ_EUR_BSCONT_v22" $SCZ_EUR $BS_CONT $V22_EUR $WLD $FRQ

# Power-matched
run_sldsc_v22 "SCZ_EUR_POWMATCH_v22" $SCZ_EUR $POWMATCH $V22_EUR $WLD $FRQ

# C0 single-annotation
run_sldsc_v22 "SCZ_EUR_C0ONLY_v22" $SCZ_EUR $C0_ONLY $V22_EUR $WLD $FRQ

# Joint conditional (brain_spec_top + C0_only)
run_sldsc_v22 "SCZ_EUR_BSC0_JOINT_v22" $SCZ_EUR "${BS_TOP},${C0_ONLY}" $V22_EUR $WLD $FRQ

echo ""
echo "============================================================"
echo "[B] Sensitivity battery v2.2 — done"
echo "============================================================"
echo "Results in: $OUT"
ls $OUT/*.results | wc -l | xargs echo "Total .results files:"
