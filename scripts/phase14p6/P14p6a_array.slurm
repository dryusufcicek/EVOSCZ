#!/bin/bash
#SBATCH --job-name=p14p6a_afr_tmrca
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-38
#SBATCH --output=logs/p14p6a_%a_%A.out
#SBATCH --error=logs/p14p6a_%a_%A.err

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p6a_arm${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log") 2>&1

ARMS=(
  1_p 1_q 2_p 2_q 3_p 3_q 4_p 4_q 5_p 5_q
  6_p 6_q 7_p 7_q 8_p 8_q 9_p 9_q 10_p 10_q
  11_p 11_q 12_p 12_q 13_q 14_q 15_q
  16_p 16_q 17_p 17_q 18_p 18_q 19_p 19_q
  20_p 20_q 21_q 22_q
)

ARM_ID=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID → arm $ARM_ID"

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate v11

export ARM_ID="$ARM_ID"
python3 ${EVOSCZ_ROOT}/scripts/phase14p6/P14p6a_extract_AFR_TMRCA.py
