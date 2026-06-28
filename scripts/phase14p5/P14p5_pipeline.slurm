#!/bin/bash
#SBATCH --job-name=p14p5_master
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/p14p5_master_%j.out
#SBATCH --error=logs/p14p5_master_%j.err

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p5_master_${SLURM_JOB_ID}.log") 2>&1

echo "==================================================================="
echo "Phase 14p5 MASTER  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "==================================================================="

CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate v11

# Verify all 39 arm extracts present
DATA=${EVOSCZ_SCRATCH}/v11_data/phase14p5
n=$(ls $DATA/P14p5a_chr*.tsv 2>/dev/null | wc -l)
echo "Per-arm extract files: $n/39"
if [ "$n" -lt 39 ]; then
    echo "WARNING: only $n/39 arms found"
fi

python3 ${EVOSCZ_ROOT}/scripts/phase14p5/P14p5b_merge_and_analyze.py

echo
echo "Phase 14p5 MASTER COMPLETE $(date)"
ls -lh ${EVOSCZ_ROOT}/results/phase14p5/
