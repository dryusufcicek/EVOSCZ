#!/bin/bash
# P17d v22 — disk-conscious version of run_perms.sh.
# After S-LDSC completes for each perm, deletes the heavy LD-score files
# (keeps only .results + .log). Peak disk use = 1 perm × ~660 MB instead of 15 × ~660 MB.
set -e
BASE=${EVOSCZ_ROOT}
LDSC=$BASE/scripts/phase14e/ldsc/ldsc.py
PLINK=$BASE/data/ldsc/sldsc_ref/1000G_EUR_Phase3_plink
BASELINE_V22=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_baselineLD_v2.2
WLD=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$BASE/data/ldsc/sldsc_ref/1000G_Phase3_frq/1000G.EUR.QC.
SUMSTATS=$BASE/results/phase14e/PGC3_SCZ_EUR.sumstats.gz
PERM_BASE=$BASE/results/phase17d_v22
N_PERM=${1:-15}

mkdir -p $PERM_BASE/logs

if [ ! -d "$BASELINE_V22" ]; then
  echo "ERROR: baseline-LD v2.2 not found"; exit 1
fi
if [ ! -f "/tmp/baselineLD_v22_chr1_snplist.txt" ]; then
  echo "ERROR: snplist missing — run P17d_v22_build_snplist.py"; exit 1
fi

# (Re)write header
echo -e "perm\tProp_SNPs\tProp_h2\tProp_h2_se\tEnrichment\tEnrichment_se\tEnrichment_P\tCoef\tCoef_se\tZ" > $PERM_BASE/perm_summary_v22.tsv

run_chr() {
  local p=$1
  local chrom=$2
  local pdir="$PERM_BASE/perm_$(printf '%03d' $p)"
  if [ -f $pdir/cluster_perm.${chrom}.l2.ldscore.gz ]; then return 0; fi
  python3 $LDSC --l2 \
    --bfile $PLINK/1000G.EUR.QC.${chrom} \
    --ld-wind-cm 1 \
    --annot $pdir/cluster_perm.${chrom}.annot.gz \
    --print-snps /tmp/baselineLD_v22_chr${chrom}_snplist.txt \
    --out $pdir/cluster_perm.${chrom} \
    > $PERM_BASE/logs/p${p}_chr${chrom}.log 2>&1
}
export -f run_chr
export LDSC PLINK PERM_BASE

echo "[v22 PERM RUN clean] N_PERM=$N_PERM"; echo

for p in $(seq 0 $((N_PERM - 1))); do
  pdir="$PERM_BASE/perm_$(printf '%03d' $p)"
  if [ ! -d "$pdir" ]; then
    echo "  perm $p: missing — skip"; continue
  fi

  # Resume: skip if S-LDSC already done for this perm
  if [ -f $pdir/h2_perm_v22.results ]; then
    line=$(grep -E "^L2_" $pdir/h2_perm_v22.results | tail -1)
    if [ -n "$line" ]; then
      echo -e "${p}\t$(echo "$line" | awk -F'\t' '{for(i=2;i<=NF;i++) printf "%s%s", $i, (i==NF?"\n":"\t")}')" >> $PERM_BASE/perm_summary_v22.tsv
      echo "  perm $p: already done, summary appended"
      # Cleanup leftover LD scores if any
      rm -f $pdir/cluster_perm.*.l2.ldscore.gz $pdir/cluster_perm.*.l2.M $pdir/cluster_perm.*.l2.M_5_50
      continue
    fi
  fi

  echo "  perm $p: LD scores (4-way parallel, RAM-safe)..."
  seq 1 22 | xargs -P 4 -I{} bash -c 'run_chr "$@"' _ $p {}

  echo "  perm $p: S-LDSC..."
  python3 $LDSC \
    --h2 $SUMSTATS \
    --ref-ld-chr $BASELINE_V22/baselineLD.,$pdir/cluster_perm. \
    --w-ld-chr $WLD \
    --frqfile-chr $FRQ \
    --overlap-annot --print-coefficients \
    --out $pdir/h2_perm_v22 \
    > $PERM_BASE/logs/p${p}_sldsc.log 2>&1

  if [ -f $pdir/h2_perm_v22.results ]; then
    line=$(grep -E "^L2_" $pdir/h2_perm_v22.results | tail -1)
    echo -e "${p}\t$(echo "$line" | awk -F'\t' '{for(i=2;i<=NF;i++) printf "%s%s", $i, (i==NF?"\n":"\t")}')" >> $PERM_BASE/perm_summary_v22.tsv

    # CLEANUP: free disk (LD score + M files are the heavy ~660 MB/perm)
    rm -f $pdir/cluster_perm.*.l2.ldscore.gz $pdir/cluster_perm.*.l2.M $pdir/cluster_perm.*.l2.M_5_50
    echo "  perm $p: done + cleaned ($(df -h . | tail -1 | awk '{print $4}') free)"
  else
    echo "  perm $p: FAILED — see $PERM_BASE/logs/p${p}_sldsc.log"
  fi
done

echo
echo "All done. Summary: $PERM_BASE/perm_summary_v22.tsv ($(wc -l < $PERM_BASE/perm_summary_v22.tsv) lines)"
echo "Next: python3 P17d_v22_summarize.py"
