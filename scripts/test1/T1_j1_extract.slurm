#!/bin/bash
#SBATCH -J t1_extract
# (chdir removed for portability — submit this job from your intended working directory)
#SBATCH -p debug
#SBATCH -t 03:30:00
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --no-requeue
#SBATCH -o logs/j1_extract_%j.out
#SBATCH -e logs/j1_extract_%j.err
set -e
mkdir -p ${EVOSCZ_SCRATCH}/test1_age_conditioning/logs
S=${EVOSCZ_ROOT}/scripts/test1
# stream all 22 atlas chr in parallel (one python proc per chr) — memory-safe
seq 1 22 | xargs -P 11 -I{} python3 $S/T1_extract_genome_age.py {}
echo "J1_EXTRACT_DONE"
