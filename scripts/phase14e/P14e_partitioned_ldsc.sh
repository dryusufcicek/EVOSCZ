#!/bin/bash
# Phase 14e: Partitioned LDSC heritability for 3 evolutionary clusters
# Tests whether SCZ heritability concentrates differently across the 3 GMM clusters
# identified in Phase 14b v3 (C0 young / C1 mid / C2 old).

set -e

BASE=${EVOSCZ_ROOT}
LDSC_DIR=$BASE/scripts/phase14e/ldsc
DATA=$BASE/data/ldsc
WORK=$BASE/results/phase14e
mkdir -p $WORK
cd $WORK

# Step 0: extract reference files
if [ ! -d $DATA/sldsc_ref ]; then
  echo "[0] Extracting reference files..."
  cd $DATA
  tar -xzf sldsc_ref.tar.gz
  ls sldsc_ref/ | head
  cd $WORK
fi

# Step 1: Format PGC3 EUR sumstats for ldsc
SUMSTATS=$BASE/data/raw/gwas/pgc3/PGC3_SCZ_EUR.txt
mkdir -p $BASE/data/raw/gwas/pgc3

if [ ! -f $SUMSTATS ]; then
  echo "[1] Formatting PGC3 EUR sumstats..."
  gunzip -c ${PGC_SUMSTATS_DIR}/PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv.gz | \
    grep -v "^##" > $SUMSTATS
  wc -l $SUMSTATS
fi

# Step 2: HapMap3 SNP list (need this for munge_sumstats merge-alleles)
HM3_LIST=$DATA/w_hm3.snplist
if [ ! -f $HM3_LIST ]; then
  echo "[2] Downloading HapMap3 SNP list..."
  curl -sL "https://zenodo.org/api/records/8367200/files/sldsc_ref.tar.gz/content" -o /dev/null  # ensure ref ext
  # Find w_hm3.snplist in extracted files
  find $DATA/sldsc_ref -name "w_hm3.snplist*" | head -3
fi

# Step 3: Munge sumstats
MUNGED=$WORK/PGC3_SCZ_EUR.sumstats.gz
if [ ! -f $MUNGED ]; then
  echo "[3] Running munge_sumstats..."
  python3 $LDSC_DIR/munge_sumstats.py \
    --sumstats $SUMSTATS \
    --N-cas-col NCAS --N-con-col NCON \
    --snp ID --a1 A1 --a2 A2 --p PVAL --signed-sumstats BETA,0 \
    --merge-alleles $HM3_LIST \
    --out $WORK/PGC3_SCZ_EUR
fi

echo "Phase 14e setup complete. Run actual partitioned heritability with cluster annotations next."
