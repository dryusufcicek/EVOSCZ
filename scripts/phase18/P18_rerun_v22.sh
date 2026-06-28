#!/bin/bash
# Phase 18: Re-run S-LDSC with baseline-LD v2.2 (Gazal 2017, 97 annotations)
# Replaces v1.2 (Finucane 2015, 53 annotations) used in original v10 analyses.
# Produces baseline-LD v2.2 partitioned-heritability .results for all annotations.
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
V22_EUR=$BASE/data/ldsc/sldsc_ref_v2.2/EUR/baselineLD.
V22_EAS=$BASE/data/ldsc/sldsc_ref_v2.2/EAS/baselineLD.
WLD=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
WLD_EAS=$BASE/data/ldsc/sldsc_ref_eas/1000G_Phase3_EAS_weights_hm3_no_MHC/weights.EAS.hm3_noMHC.
FRQ=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
FRQ_EAS=$BASE/data/ldsc/sldsc_ref_eas/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC.
CLUSTER_EUR=$BASE/results/phase14e/cluster_annot/cluster.
CLUSTER_EAS=$BASE/results/phase14h/cluster_annot_eas/cluster.
OUT=$BASE/results/phase18_v22
mkdir -p $OUT

# Sumstats lookup
SCZ_EUR=$BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
SCZ_EAS=$BASE/results/phase14h/PGC3_SCZ_EAS.sumstats.gz
AD=$BASE/results/phase14f/Wightman_AD.sumstats.gz
BD=$BASE/results/phase14j/PGC_BD2024_EUR.sumstats.gz
MDD=$BASE/results/phase14j/PGC_MDD2025_EUR.sumstats.gz
ADHD=$BASE/results/phase14j/PGC_ADHD2022.sumstats.gz
ASD=$BASE/results/phase14j/PGC_ASD_Grove2019.sumstats.gz
F3=$BASE/results/phase14j/CDG3_F3_Neurodev.sumstats.gz
F4=$BASE/results/phase14j/CDG3_F4_Internal.sumstats.gz
HT=$BASE/results/phase14j/Yengo2018_height.sumstats.gz
BMI=$BASE/results/phase14j/Yengo2018_BMI.sumstats.gz

run_sldsc() {
  local tag=$1
  local sumstats=$2
  local cluster=$3
  local baseline=$4
  local wld=$5
  local frq=$6
  if [ -f "$OUT/$tag.results" ]; then echo "  $tag: SKIP"; return 0; fi
  echo "  Running $tag..."
  python3 $LDSC \
    --h2 $sumstats \
    --ref-ld-chr $baseline,$cluster \
    --w-ld-chr $wld \
    --frqfile-chr $frq \
    --overlap-annot --print-coefficients \
    --out $OUT/$tag > $OUT/$tag.run.log 2>&1
  if [ -f "$OUT/$tag.results" ]; then
    enr=$(grep -E "^C0L2" $OUT/$tag.results | awk -F'\t' '{print $5}')
    z=$(grep -E "^C0L2" $OUT/$tag.results | awk -F'\t' '{print $10}')
    p=$(grep -E "^C0L2" $OUT/$tag.results | awk -F'\t' '{print $7}')
    echo "    C0 (Young): enrichment=$enr  Z=$z  P=$p"
  else
    echo "    FAILED"
  fi
}

echo "============================================================"
echo "Phase 18 — S-LDSC re-run with baseline-LD v2.2 (Gazal 2017)"
echo "============================================================"

# Primary EUR
run_sldsc "PGC3_SCZ_EUR_v22"    $SCZ_EUR $CLUSTER_EUR $V22_EUR $WLD $FRQ

# Wightman 2021 AD
run_sldsc "Wightman_AD_v22"      $AD $CLUSTER_EUR $V22_EUR $WLD $FRQ

# Cross-disorder
run_sldsc "BD2024_EUR_v22"       $BD $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "MDD2025_EUR_v22"      $MDD $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "ADHD2022_v22"         $ADHD $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "ASD_Grove2019_v22"    $ASD $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "CDG3_F3_v22"          $F3 $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "CDG3_F4_v22"          $F4 $CLUSTER_EUR $V22_EUR $WLD $FRQ

# Negative-control
run_sldsc "Yengo2018_height_v22" $HT $CLUSTER_EUR $V22_EUR $WLD $FRQ
run_sldsc "Yengo2018_BMI_v22"    $BMI $CLUSTER_EUR $V22_EUR $WLD $FRQ

# EAS replication (cluster annotation EAS-specific)
if [ -f "$CLUSTER_EAS"1.l2.ldscore.gz ]; then
  run_sldsc "PGC3_SCZ_EAS_v22"   $SCZ_EAS $CLUSTER_EAS $V22_EAS $WLD_EAS $FRQ_EAS
else
  echo "  EAS cluster annotation missing — skip"
fi

echo ""
echo "============================================================"
echo "Phase 18 done. Comparison summary:"
echo "============================================================"
python3 << 'PYEOF'
import os
import re

OUT = os.environ.get("EVOSCZ_ROOT", ".") + "/results/phase18_v22"
runs = [f.replace(".results","") for f in os.listdir(OUT) if f.endswith(".results")]
print(f"{'Tag':<32} {'Annotation':<8} {'Enrichment':>10} {'s.e.':>8} {'P':>12} {'Coef Z':>8}")
print("-" * 80)
for tag in sorted(runs):
    with open(f"{OUT}/{tag}.results") as f:
        for line in f:
            if line.startswith("C0L2") or line.startswith("C1L2") or line.startswith("C2L2"):
                parts = line.strip().split('\t')
                ann = parts[0]
                enr = float(parts[4])
                se = float(parts[5])
                p = parts[6]
                z = float(parts[9])
                print(f"{tag:<32} {ann:<8} {enr:>10.2f} {se:>8.2f} {p:>12s} {z:>+8.2f}")
PYEOF
