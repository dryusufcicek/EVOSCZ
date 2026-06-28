#!/bin/bash
#SBATCH --job-name=p19_polarity_raf
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --output=logs/p19_polarity_%j.out
#SBATCH --error=logs/p19_polarity_%j.err
# Resource budget per the cluster scheduling policy (precedent: Phase 14p7):
#   the cluster may require --cpus-per-task=20 (node-level allocation).
#   mem=12G, time=00:30:00 within QOS limits.
#   Phase 19 is single-threaded pandas — extra cores unused but required by policy.
# Memory peak estimate (post H10 gc.collect cleanup) ~8-10 GB.

# ============================================================================
# Phase 19 — Polarity-aware RAF (SLURM submission)
# ============================================================================
# Replaces the earlier (rejected) Phase 18a approach. Verdict tree:
#   { NULL, DIRECTIONAL_protective, DIRECTIONAL_risk, WINNER_CURSE_CONFOUND }
# Pre-registered falsification gates and methodological requirements
# (analysis protocol available from the authors).
#
# Hard stop conditions (script exits non-zero):
#   - Hand-test fails (P0.9)
#   - P14p3 vs P14p2 Pearson r < 0.90 (P0.10)
#   - Form1/Form2 verdict mismatch on ANY variant (P0.1)
# ============================================================================

# Fix per code review HIGH #2:
#   set -e was killing the trap+RC capture path on python failure (hand-test
#   fail, P0.10 fail, Form1/Form2 assertion). We drop -e specifically around
#   the python call below so RC is captured and the summary block always runs.
set -uo pipefail
trap 'echo "[$(date +%H:%M:%S)] ERROR on line $LINENO (exit $?)"' ERR

mkdir -p ${EVOSCZ_SCRATCH}/v11_logs
# Redirect via append to log; avoid tee subshell + pipefail interaction.
exec >> "${EVOSCZ_SCRATCH}/v11_logs/p19_polarity_${SLURM_JOB_ID}.log" 2>&1

echo "============================================================================"
echo "Phase 19 — Polarity-aware RAF"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Started:   $(date)"
echo "Host:      $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-?}"
echo "Memory:    ${SLURM_MEM_PER_NODE:-?}M"
echo "============================================================================"

# Activate v11 conda env (must contain pandas, scipy, numpy, statsmodels)
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh

# IMPORTANT: confirm env name before submitting. Common candidates:
#   v11_env / v11 / evoscz_v11 / sldsc_py3
# If 'v11' fails, change here.
CONDA_ENV_NAME="${CONDA_ENV:-v11}"
conda activate "${CONDA_ENV_NAME}"
echo "Active conda env: ${CONDA_DEFAULT_ENV}"
python --version
python -c "import pandas, numpy, scipy, statsmodels; \
    print('pandas', pandas.__version__); \
    print('numpy', numpy.__version__); \
    print('scipy', scipy.__version__); \
    print('statsmodels', statsmodels.__version__)"

# Paths (SLURM)
WORKDIR=${EVOSCZ_ROOT}/scripts/phase19
OUTDIR=${EVOSCZ_ROOT}/results/phase19
mkdir -p "${OUTDIR}"

cd "${WORKDIR}"
echo
echo "Working dir: $(pwd)"
echo "Output dir:  ${OUTDIR}"
echo "Script:      P19_polarity_RAF.py"
echo

# Run analysis. Capture exit code without aborting the script — we want the
# summary block (ls outputs, verdict grep, final exit) to run on failure too.
set +e
python P19_polarity_RAF.py
RC=$?
set -e

echo
echo "============================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Exit code: ${RC}"
echo "Finished:  $(date)"
echo "============================================================================"

# List outputs
echo
echo "=== Outputs in ${OUTDIR} ==="
ls -lh "${OUTDIR}" || true

# Echo the verdict line(s) for quick scanning of the SLURM log
echo
echo "=== PRIMARY VERDICT EXCERPT ==="
grep -E "^=== (PRIMARY VERDICT|WINNER-CURSE TEST|SUBPOP AGREEMENT|MERGE RULE|CROSS-CHECK)" \
    "${EVOSCZ_SCRATCH}/v11_logs/p19_polarity_${SLURM_JOB_ID}.log" || \
    echo "(verdict markers not found — check log)"

exit ${RC}
