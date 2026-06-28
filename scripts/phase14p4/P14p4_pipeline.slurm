#!/bin/bash
#SBATCH --job-name=p14p4_master
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/p14p4_master_%j.out
#SBATCH --error=logs/p14p4_master_%j.err
#
# Phase 14p4 — H2 master orchestrator
# ==================================================================
# Runs the H2 analysis chain (a → b → c) sequentially using v11
# conda environment. All data already staged locally — no I/O bottleneck.

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p4_master_${SLURM_JOB_ID}.log") 2>&1

echo "==================================================================="
echo "Phase 14p4 MASTER  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "==================================================================="

CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate v11
which python

SCRIPTS=${EVOSCZ_ROOT}/scripts/phase14p4
RESULTS=${EVOSCZ_ROOT}/results/phase14p4
mkdir -p $RESULTS

# Verify inputs
[ -s "${EVOSCZ_SCRATCH}/v11_data/aux/akbari_347_loci.tsv" ] || { echo "FATAL: akbari_347_loci.tsv missing"; exit 2; }
[ -s "${EVOSCZ_ROOT}/results/phase11/variant_master_v4.parquet" ] || { echo "FATAL: variant_master_v4 missing"; exit 2; }
[ -s "${EVOSCZ_ROOT}/results/phase14b/P14b_v3_cluster_assignments.tsv.gz" ] || { echo "FATAL: cluster assignments missing"; exit 2; }

echo
echo "[A] Akbari overlap main: |S| dist, Fisher overlap, direction-of-effect, HLA sensitivity"
python3 $SCRIPTS/P14p4a_akbari_overlap_main.py

echo
echo "[B] Locus-window overlap (5/50/250 kb)"
python3 $SCRIPTS/P14p4b_locus_window_overlap.py

echo
echo "[C] Multi-panel figure"
python3 $SCRIPTS/P14p4c_figure.py

echo
echo "==================================================================="
echo "Phase 14p4 MASTER COMPLETE $(date)"
echo "==================================================================="
ls -lh $RESULTS/
