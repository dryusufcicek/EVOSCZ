#!/bin/bash
#SBATCH --job-name=p14p7_h2only
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --output=logs/p14p7_h2only_%j.out
#SBATCH --error=logs/p14p7_h2only_%j.err

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p7_h2only_${SLURM_JOB_ID}.log") 2>&1

echo "Phase 14p7 — h² only (LD scores already exist) $(date)"

# Copy annot files alongside ldscore files (same prefix required by LDSC)
SCRATCH=${EVOSCZ_SCRATCH}/v11_data/phase14p7
for chrom in $(seq 1 22); do
    cp $SCRATCH/annot/AFR_partition.${chrom}.annot.gz $SCRATCH/ldscores/AFR_partition.${chrom}.annot.gz
done
echo "annot files copied to ldscores dir"

# Activate Python 2.7 env
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ldsc_py2

LDSC=$SCRATCH/software/ldsc/ldsc.py
BASELINE=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline/baseline.
AFR_ANNOT=$SCRATCH/ldscores/AFR_partition.
WEIGHTS=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQFILE=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SUMSTATS=${EVOSCZ_ROOT}/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
RESULTS=${EVOSCZ_ROOT}/results/phase14p7
OUT_PREFIX=$RESULTS/P14p7_AFR_partition_h2_PGC3_EUR

echo "Running partitioned h² (PGC3 EUR)..."
python $LDSC \
    --h2 $SUMSTATS \
    --ref-ld-chr $BASELINE,$AFR_ANNOT \
    --w-ld-chr $WEIGHTS \
    --frqfile-chr $FRQFILE \
    --overlap-annot \
    --print-coefficients \
    --print-delete-vals \
    --out $OUT_PREFIX 2>&1 | tail -30

echo
echo "=== .results FILE ==="
cat ${OUT_PREFIX}.results 2>/dev/null | head -100

echo
echo "Done $(date)"
