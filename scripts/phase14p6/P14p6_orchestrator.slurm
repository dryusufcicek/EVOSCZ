#!/bin/bash
#SBATCH --job-name=p14p6_orch
#SBATCH --partition=debug
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/p14p6_orch_%j.out
#SBATCH --error=logs/p14p6_orch_%j.err

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p6_orch_${SLURM_JOB_ID}.log") 2>&1

echo "==================================================================="
echo "Phase 14p6 ORCHESTRATOR  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "==================================================================="

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate v11

SCRIPTS=${EVOSCZ_ROOT}/scripts/phase14p6
RESULTS=${EVOSCZ_ROOT}/results/phase14p6
DATA=${EVOSCZ_SCRATCH}/v11_data/phase14p6
mkdir -p $RESULTS

# Verify all 39 arm extracts present
n=$(ls $DATA/P14p6a_chr*.tsv 2>/dev/null | wc -l)
echo "Per-arm extract files: $n/39"
if [ "$n" -lt 39 ]; then
    echo "ERROR: only $n/39 arms; aborting"
    exit 3
fi

echo "[B] Merge + partition + h² proxy"
python3 $SCRIPTS/P14p6b_merge_partition_heritability.py

echo
echo "[C] Evolutionary dynamics figure"
python3 $SCRIPTS/P14p6c_dynamics_figures.py

echo
echo "==================================================================="
echo "Phase 14p6 ORCHESTRATOR COMPLETE $(date)"
echo "==================================================================="
ls -lh $RESULTS/
