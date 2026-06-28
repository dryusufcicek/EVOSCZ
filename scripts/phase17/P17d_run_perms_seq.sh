#!/bin/bash
# P17d sequential: 15 perms, no parallelism, low RAM footprint.
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_baseline
WLD=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SUMSTATS=$BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
PERM_BASE=$BASE/results/phase17d

mkdir -p $PERM_BASE/logs

N_PERM=${N_PERM:-15}
# Reset summary
echo -e "perm\tProp_SNPs\tProp_h2\tProp_h2_se\tEnrichment\tEnrichment_se\tEnrichment_P\tCoef\tCoef_se\tZ" > $PERM_BASE/perm_summary.tsv

for p in $(seq 0 $((N_PERM-1))); do
  pdir="$PERM_BASE/perm_$(printf '%03d' $p)"
  if [ ! -d "$pdir" ]; then continue; fi

  echo "perm $p starting LD scores at $(date '+%H:%M:%S')"
  # Sequential LD scores per chr
  for chrom in {1..22}; do
    if [ -f $pdir/cluster_perm.${chrom}.l2.ldscore.gz ]; then continue; fi
    python3 $LDSC --l2 \
      --bfile $PLINK/1000G.EUR.QC.${chrom} \
      --ld-wind-cm 1 \
      --annot $pdir/cluster_perm.${chrom}.annot.gz \
      --print-snps /tmp/baseline_chr${chrom}_snplist.txt \
      --out $pdir/cluster_perm.${chrom} \
      > $PERM_BASE/logs/p${p}_chr${chrom}.log 2>&1
  done
  echo "  perm $p LD done at $(date '+%H:%M:%S'), starting S-LDSC"

  # S-LDSC
  python3 $LDSC \
    --h2 $SUMSTATS \
    --ref-ld-chr $BASELINE/baseline.,$pdir/cluster_perm. \
    --w-ld-chr $WLD \
    --frqfile-chr $FRQ \
    --overlap-annot --print-coefficients \
    --out $pdir/h2_perm \
    > $PERM_BASE/logs/p${p}_sldsc.log 2>&1

  if [ -f $pdir/h2_perm.results ]; then
    line=$(grep -E "^L2_1" $pdir/h2_perm.results | head -1)
    echo -e "${p}\t$(echo "$line" | awk -F'\t' '{for(i=2;i<=NF;i++) printf "%s%s", $i, (i==NF?"\n":"\t")}')" >> $PERM_BASE/perm_summary.tsv
    echo "  perm $p: DONE at $(date '+%H:%M:%S')"
  else
    echo "  perm $p: FAILED"
  fi
done

echo "All $N_PERM permutations done."
echo ""
cat $PERM_BASE/perm_summary.tsv
