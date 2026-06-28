#!/bin/bash
#SBATCH --job-name=p14p3a_afr_af
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=1-22
#SBATCH --output=logs/p14p3a_chr%a_%A.out
#SBATCH --error=logs/p14p3a_chr%a_%A.err
#
# Phase 14p3a — Extract pure-AFR allele frequencies per chromosome
# ==================================================================
# Per chromosome (SLURM array task = chromosome number):
#   1. Restrict 1000G Phase 3 VCF to (a) pure-AFR 504 samples,
#      (b) PGC3 credible-set positions (from variant_master_v4)
#   2. Compute AF per variant in this pure-AFR subset using
#      `bcftools +fill-tags -- -t AF,AN,AC`
#   3. Output a per-chr TSV with rsid, chr, pos, A1, A2, AF_AFR, AN, AC
#      plus per-subpop AF (ESN/GWD/LWK/MSL/YRI) for downstream replication
#
# Anti-confound design:
#   - SAMPLE-LEVEL FILTERING via bcftools view -S (not population panel
#     trust); guarantees pure-AFR is what's reported.
#   - REGION BED restriction reduces I/O ~1000× per chr.
#   - REF/ALT alignment harmonization handled in next step (P14p3b),
#     not here (avoid premature loss of variants).
#   - "fill-tags -t AF" recomputes AF from genotypes (does not trust
#     header INFO/AF which is the full 2504-sample frequency).

set -euo pipefail

# conda env activation
CONDA_BASE="${CONDA_BASE:?set CONDA_BASE to your conda/miniforge install}"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate v11

CHR=${SLURM_ARRAY_TASK_ID}
DATA=${EVOSCZ_SCRATCH}/v11_data
VCF=$DATA/1000G_phase3/ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
PANEL=$DATA/1000G_phase3/integrated_call_samples_v3.20130502.ALL.panel
OUT=$DATA/phase14p3
mkdir -p $OUT
LOG=${EVOSCZ_SCRATCH}/v11_logs/p14p3a_chr${CHR}_${SLURM_JOB_ID}.log
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo "Phase 14p3a — chr${CHR}  $(date)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "=================================================================="

# ── Step 0: verify inputs ──────────────────────────────────────────
[ -s "$VCF" ]   || { echo "FATAL: VCF missing: $VCF"; exit 2; }
[ -s "${VCF}.tbi" ] || { echo "FATAL: TBI missing"; exit 2; }
[ -s "$PANEL" ] || { echo "FATAL: PANEL missing"; exit 2; }
[ -s "$DATA/aux/pure_AFR_samples.txt" ] || { echo "FATAL: pure_AFR_samples missing"; exit 2; }
bcftools --version | head -1

# ── Step 1: build credible-set position BED for this chr (one shot) ──
# We'll produce a per-chr BED from variant_master_v4. The fast way:
# call Python once at chr1, save BED files for chr1-22; subsequent
# tasks reuse them.
BED=$OUT/credset_chr${CHR}.bed
if [ ! -s "$BED" ]; then
    # Use a per-task lock so only chr1 runs the Python step,
    # and chr2-22 wait briefly then read the file. But simpler:
    # just generate from Python each time — it's ~3 sec.
    python3 - <<PYEOF
import pandas as pd
df = pd.read_parquet("${EVOSCZ_ROOT}/results/phase11/variant_master_v4.parquet")
df = df[["rsid","chr","pos"]].copy()
df["chr"] = pd.to_numeric(df["chr"], errors="coerce").astype("Int64")
df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
df = df.dropna(subset=["chr","pos"])
sub = df[df["chr"] == ${CHR}].copy()
sub["start"] = sub["pos"] - 1
sub["end"]   = sub["pos"]
sub[["chr","start","end","rsid"]].to_csv("${BED}", sep="\t", header=False, index=False)
print(f"chr${CHR}: {len(sub):,} credible-set positions written to ${BED}")
PYEOF
fi
wc -l "$BED"

# ── Step 2: pure-AFR + per-subpop sample lists ──────────────────────
AUX=$DATA/aux
PURE_AFR=$AUX/pure_AFR_samples.txt
# Per-subpop lists (also useful for P14p3f subpop decomposition)
for SUB in ESN GWD LWK MSL YRI; do
    F=$AUX/subpop_${SUB}_samples.txt
    if [ ! -s "$F" ]; then
        awk -F'\t' -v pop="$SUB" '$2==pop {print $1}' "$PANEL" > "$F"
    fi
done
echo "Sample list counts:"
echo "  pure-AFR: $(wc -l < $PURE_AFR)"
for SUB in ESN GWD LWK MSL YRI; do
    echo "  ${SUB}: $(wc -l < $AUX/subpop_${SUB}_samples.txt)"
done

# ── Step 3: bcftools extract — pure-AFR pooled ──────────────────────
OUT_AFR=$OUT/pure_AFR_chr${CHR}.tsv
echo
echo "[3] Extract + fill-tags pure-AFR chr${CHR}..."
echo "    rsid chr pos REF ALT AF_AFR AN_AFR AC_AFR"

bcftools view --threads 2 \
        -S "$PURE_AFR" \
        -R "$BED" \
        --force-samples \
        "$VCF" 2>/dev/null \
  | bcftools +fill-tags - -- -t AF,AN,AC 2>/dev/null \
  | bcftools query -f '%ID\t%CHROM\t%POS\t%REF\t%ALT\t%INFO/AF\t%INFO/AN\t%INFO/AC\n' \
  > $OUT_AFR

N_OUT=$(wc -l < "$OUT_AFR")
echo "  pure-AFR chr${CHR}: ${N_OUT} variants extracted"

# ── Step 4: per-subpop extraction ───────────────────────────────────
echo
echo "[4] Per-subpop extraction..."
for SUB in ESN GWD LWK MSL YRI; do
    SUB_OUT=$OUT/subpop_${SUB}_chr${CHR}.tsv
    bcftools view --threads 2 \
            -S "$AUX/subpop_${SUB}_samples.txt" \
            -R "$BED" \
            --force-samples \
            "$VCF" 2>/dev/null \
      | bcftools +fill-tags - -- -t AF 2>/dev/null \
      | bcftools query -f '%ID\t%CHROM\t%POS\t%REF\t%ALT\t%INFO/AF\n' \
      > $SUB_OUT
    N=$(wc -l < "$SUB_OUT")
    printf "  %s chr${CHR}: %d variants\n" "$SUB" "$N"
done

# ── Step 5: spot-check ──────────────────────────────────────────────
echo
echo "[5] Spot-check (first 3 records pure-AFR chr${CHR}):"
head -3 "$OUT_AFR"
echo
echo "[6] Allele freq distribution (pure-AFR chr${CHR}):"
awk -F'\t' '{
    af=$6
    if (af=="." || af=="") next
    af=af+0
    bin=int(af*10)
    cnt[bin]++
    n++
} END {
    print "  N variants with AF: " n
    for(b=0;b<=10;b++) printf "    AF[%.1f-%.1f]: %d\n", b/10, (b+1)/10, cnt[b]+0
}' "$OUT_AFR"

echo
echo "[$(date)] Phase 14p3a chr${CHR} COMPLETE"
