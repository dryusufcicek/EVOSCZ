#!/bin/bash
#SBATCH --job-name=p14p7_sldsc
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --output=logs/p14p7_%j.out
#SBATCH --error=logs/p14p7_%j.err
#
# Phase 14p7 — s-LDSC partitioned heritability for AFR_TMRCA partitions
# =======================================================================
# Steps:
#   (A) Install LDSC (clone official Bulik repo, create Python 2.7 conda env)
#   (B) Generate AFR-TMRCA partition annotation files
#       - 5 partitions: AFR_absent / <50kyr / 50-200kyr / 200-500kyr / ≥500kyr
#       - One column per partition, 0/1 membership over HM3 SNPs
#   (C) Compute LD scores per annotation × chromosome (ldsc.py --l2)
#   (D) Run partitioned h² regression with baseline-LD v2.2
#       - PGC3 EUR (existing munged sumstats)
#       - Output: h² enrichment + conditional Z per partition
# =======================================================================

set -euo pipefail
exec > >(tee -a "${EVOSCZ_SCRATCH}/v11_logs/p14p7_${SLURM_JOB_ID}.log") 2>&1

echo "==================================================================="
echo "Phase 14p7 — s-LDSC pipeline  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "==================================================================="

SCRATCH=${EVOSCZ_SCRATCH}/v11_data/phase14p7
mkdir -p $SCRATCH/annot $SCRATCH/ldscores
RESULTS=${EVOSCZ_ROOT}/results/phase14p7
mkdir -p $RESULTS

# ── (A) Install LDSC ───────────────────────────────────────────────
echo
echo "=== (A) LDSC install check ==="
LDSC_DIR=$SCRATCH/software/ldsc
if [ ! -f "$LDSC_DIR/ldsc.py" ]; then
    echo "Cloning LDSC..."
    mkdir -p $SCRATCH/software
    cd $SCRATCH/software
    git clone --depth 1 https://github.com/bulik/ldsc.git
    cd $LDSC_DIR
    echo "  Cloned: $(ls ldsc.py)"
fi

# Python 2.7 env for LDSC
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh

if ! conda env list | grep -q '^ldsc_py2'; then
    echo "Creating ldsc_py2 conda env..."
    conda create -n ldsc_py2 -c conda-forge -c bioconda -y \
        python=2.7 numpy=1.16 scipy=1.2 pandas=0.20 bitarray=0.8 2>&1 | tail -8
fi
conda activate ldsc_py2
which python
python --version
python -c "import numpy, scipy, pandas, bitarray; print('Dependencies OK')"

# Test ldsc.py
cd $LDSC_DIR
python ldsc.py --help > /dev/null && echo "ldsc.py runs OK" || { echo "FATAL: ldsc.py broken"; exit 4; }
LDSC=$LDSC_DIR/ldsc.py
MUNGE=$LDSC_DIR/munge_sumstats.py

# ── (B) Build annotation files ─────────────────────────────────────
echo
echo "=== (B) AFR-TMRCA partition annotations ==="

# Switch to v11 (pandas) to build annot files
conda deactivate
conda activate v11

python3 <<'PYEOF'
import pandas as pd, numpy as np, gzip
from pathlib import Path

BASE = Path("${EVOSCZ_ROOT}")
ANNOT_DIR = Path("${EVOSCZ_SCRATCH}/v11_data/phase14p7/annot")
BASELINE_DIR = BASE / "data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline"

# Load AFR partition assignments
part = pd.read_csv(BASE / "results/phase14p6/P14p6b_AFR_partition_assignments.tsv", sep="\t")
print(f"AFR partition assignments: {len(part)}")
print(part["AFR_partition"].value_counts())

# Map rsid → partition
part_map = dict(zip(part["rsid"].astype(str), part["AFR_partition"]))

PARTITIONS = ["AFR_absent", "AFR_TMRCA_lt_50kyr", "AFR_TMRCA_50_200kyr",
              "AFR_TMRCA_200_500kyr", "AFR_TMRCA_ge_500kyr"]

# Build annot per chromosome: read baseline.{N}.annot.gz, extract SNP+CHR+BP+CM
# then add binary partition columns. Force int32 dtype to avoid LDSC parser issues.
for chrom in range(1, 23):
    base_annot = BASELINE_DIR / f"baseline.{chrom}.annot.gz"
    if not base_annot.exists():
        print(f"  WARNING: {base_annot} missing"); continue
    base = pd.read_csv(base_annot, sep="\t", compression="gzip", usecols=["CHR","BP","SNP","CM"])
    # Set partition columns with EXPLICIT int dtype
    for p in PARTITIONS:
        col = base["SNP"].astype(str).map(lambda r: 1 if part_map.get(r) == p else 0)
        base[p] = np.asarray(col.values, dtype=np.int32)
    out_f = ANNOT_DIR / f"AFR_partition.{chrom}.annot.gz"
    base[["CHR","BP","SNP","CM"] + PARTITIONS].to_csv(out_f, sep="\t", index=False, compression="gzip")
    counts = {p: int(base[p].sum()) for p in PARTITIONS}
    print(f"  chr{chrom}: {len(base):,} SNPs, partition hits: {counts}")
print("\nAnnotation files written.")
PYEOF

# ── (C) Compute LD scores per chromosome ───────────────────────────
echo
echo "=== (C) LD scores per chromosome ==="
conda activate ldsc_py2

LDSCORE_DIR=$SCRATCH/ldscores
PLINK=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
HM3=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/hm_snp.txt

mkdir -p $LDSCORE_DIR
for chrom in $(seq 1 22); do
    out_pre=$LDSCORE_DIR/AFR_partition.${chrom}
    if [ -f "${out_pre}.l2.ldscore.gz" ]; then
        echo "  chr${chrom}: LD scores already computed, skipping"
        continue
    fi
    echo "  chr${chrom}: computing LD scores..."
    python $LDSC \
        --l2 \
        --bfile $PLINK/1000G.EUR.QC.${chrom} \
        --ld-wind-cm 1 \
        --annot $SCRATCH/annot/AFR_partition.${chrom}.annot.gz \
        --out $out_pre \
        --print-snps $HM3 2>&1 | tail -3
done

echo
echo "LD scores done. Files:"
ls $LDSCORE_DIR/*.ldscore.gz | head -3

# ── (D) Partitioned h² ─────────────────────────────────────────────
echo
echo "=== (D) Partitioned h² with baseline-LD v2.2 ==="

BASELINE=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline/baseline.
AFR_ANNOT=$LDSCORE_DIR/AFR_partition.
WEIGHTS=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQFILE=${EVOSCZ_ROOT}/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SUMSTATS=${EVOSCZ_ROOT}/results/phase14e/PGC3_SCZ_EUR.sumstats.gz

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
    --out $OUT_PREFIX 2>&1 | tail -20

# Display results
echo
echo "=== RESULTS ==="
cat ${OUT_PREFIX}.results | head -100 || echo "no results file"

echo
echo "Phase 14p7 COMPLETE $(date)"
ls -lh $RESULTS/
