#!/bin/bash
#SBATCH --job-name=p14p3_master
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/p14p3_master_%j.out
#SBATCH --error=logs/p14p3_master_%j.err
#
# Phase 14p3 — Master orchestrator
# ==================================================================
# Runs the chained analysis (b → c → d → e → f → g → h) after the
# bcftools array job (a) has completed.
# Submission order:
#   1. sbatch P14p3a_extract_pure_AFR_AF.slurm  (array 1-22, parallel)
#   2. sbatch --dependency=afterok:<a-jobid>  P14p3_pipeline.slurm

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p3_master_${SLURM_JOB_ID}.log") 2>&1

echo "==================================================================="
echo "Phase 14p3 MASTER  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "==================================================================="

CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate v11
which python bcftools

SCRIPTS=${EVOSCZ_ROOT}/scripts/phase14p3
RESULTS=${EVOSCZ_ROOT}/results/phase14p3
DATA=${EVOSCZ_SCRATCH}/v11_data/phase14p3
mkdir -p $RESULTS

# --- Sanity: did all 22 chr extracts complete? ---
echo
echo "[CHECK] Per-chr extract status"
missing=0
for chr in $(seq 1 22); do
    f=$DATA/pure_AFR_chr${chr}.tsv
    if [ ! -s "$f" ]; then
        echo "  MISSING chr${chr}: $f"
        missing=$((missing+1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "FATAL: $missing chromosomes failed extraction; cannot proceed."
    exit 3
fi
echo "  All 22 per-chr extracts present"

# --- Run scripts in order ---
echo
echo "[B] Merge + harmonize"
python3 $SCRIPTS/P14p3b_merge_and_harmonize.py

echo
echo "[C] Primary KS + MWU + permutation"
python3 $SCRIPTS/P14p3c_primary_cluster_DAF_tests.py

echo
echo "[D] EUR-MAF stratified sensitivity"
python3 $SCRIPTS/P14p3d_MAF_stratified_sensitivity.py

echo
echo "[E] HLA + LD sensitivity"
python3 $SCRIPTS/P14p3e_HLA_LD_sensitivity.py

echo
echo "[F] Per-AFR-subpop replication"
python3 $SCRIPTS/P14p3f_subpop_decomposition.py

echo
echo "[G] Block-bootstrap CI"
python3 $SCRIPTS/P14p3g_bootstrap_CI.py

echo
echo "[H] Figure"
python3 $SCRIPTS/P14p3h_figure.py

echo
echo "==================================================================="
echo "Phase 14p3 MASTER COMPLETE $(date)"
echo "==================================================================="
echo "Outputs in: $RESULTS"
ls -lh $RESULTS/
