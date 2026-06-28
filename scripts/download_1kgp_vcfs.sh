#!/bin/bash
# Download all autosomal 1000G Phase 3 VCFs for selscan / Relate / IBDmix.
# Resume-friendly: -C - flag.
# Run with: bash scripts/download_1kgp_vcfs.sh

set -e
DEST="${EVOSCZ_ROOT}/data/raw/1kgp/vcf"
BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
mkdir -p "$DEST"
cd "$DEST"

# Download in order from smallest (chr22) to largest (chr1) for fast feedback
ORDER=(22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1)

for CHR in "${ORDER[@]}"; do
    FILE="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    if [[ -f "$FILE" ]] && gunzip -t "$FILE" 2>/dev/null; then
        echo "[skip] $FILE already complete"
        continue
    fi
    echo "[download] chr${CHR}..."
    curl -L -O -C - "${BASE}/${FILE}"
    echo "[verify] chr${CHR}..."
    gunzip -t "$FILE" && echo "  OK" || echo "  CORRUPT — re-run script"
done

echo ""
echo "All autosomal 1000G VCFs downloaded to: $DEST"
ls -lh "$DEST"/ALL.chr*.vcf.gz | awk '{print $5, $NF}'
