#!/bin/bash
# Phase 17: S-LDSC runner for P17b (brain_spec-only) + P17c (power-matched)
# Triggered after LD scores complete.
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
SUMSTATS=$BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
REF_BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline/baseline.
WLD=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.

# ── P17b Annotation A (top-1742 brain_spec binary) ───────────────────────
python3 $LDSC \
  --h2 $SUMSTATS \
  --ref-ld-chr ${REF_BASELINE},$BASE/results/phase17b/brain_spec_top1742_annot/brain_spec_high. \
  --w-ld-chr $WLD \
  --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients \
  --out $BASE/results/phase17b/h2_pgc3_eur_brain_spec_top1742 2>&1 | tail -20
echo "P17b TOP1742 done."

# ── P17b Annotation B (continuous brain_spec) ────────────────────────────
python3 $LDSC \
  --h2 $SUMSTATS \
  --ref-ld-chr ${REF_BASELINE},$BASE/results/phase17b/brain_spec_continuous_annot/brain_spec. \
  --w-ld-chr $WLD \
  --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients \
  --out $BASE/results/phase17b/h2_pgc3_eur_brain_spec_continuous 2>&1 | tail -20
echo "P17b CONTINUOUS done."

# ── P17c Power-matched cluster annotation ─────────────────────────────────
python3 $LDSC \
  --h2 $SUMSTATS \
  --ref-ld-chr ${REF_BASELINE},$BASE/results/phase17c/cluster_annot_power_matched/cluster. \
  --w-ld-chr $WLD \
  --frqfile-chr $FRQ \
  --overlap-annot --print-coefficients \
  --out $BASE/results/phase17c/h2_pgc3_eur_power_matched 2>&1 | tail -20
echo "P17c POWER-MATCHED done."

echo "All P17b + P17c S-LDSC complete."
