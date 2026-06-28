#!/bin/bash
# EVOSCZ pipeline monitor — code-review fix overnight run
# Usage:  bash ${EVOSCZ_ROOT}/scripts/monitor.sh
#         watch -n 30 'bash ${EVOSCZ_ROOT}/scripts/monitor.sh'  (refresh every 30s)
#
# Code-review-fix cutoff: any output file modified before 2026-05-02 22:00 is
# the OLD (pre-fix) version; after that timestamp = NEW (post-fix). Files older
# than the cutoff display as ⊝ (stale, pre-fix) so it's obvious which fixes are
# still pending.

PER_CHR=${EVOSCZ_ROOT}/results/phase12/per_chr
LOG_DIR=${EVOSCZ_ROOT}/logs/phase12d

# Code-review run start cutoff (epoch seconds), 2026-05-02 22:00 +03
CUTOFF=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-05-02 22:00:00' '+%s' 2>/dev/null || echo 0)

mark() {
  # Args: <abs_path>  → emits "✓ NEW (HH:MM, SIZE)" or "⊝ STALE (HH:MM, SIZE)"
  local p=$1
  if [ ! -f "$p" ]; then
    echo "⏳ pending"
    return
  fi
  local mt=$(stat -f '%m' "$p")
  local hm=$(stat -f '%Sm' -t '%H:%M' "$p")
  local sz=$(ls -lh "$p" | awk '{print $5}')
  if [ "$mt" -ge "$CUTOFF" ]; then
    echo "✓ NEW   ($hm, $sz)"
  else
    echo "⊝ stale ($hm, $sz) — pre-fix; will be overwritten"
  fi
}

clear 2>/dev/null
echo "=========================================================================="
echo " EVOSCZ Pipeline Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Cutoff: outputs ≥ 2026-05-02 22:00 = NEW (post-code-review-fix);"
echo "         older = stale (pre-fix), will be overwritten by re-run"
echo "=========================================================================="
echo ""

# ── 1) Faz B (iHS) progress ──
N_DONE=$(ls $PER_CHR/P12d_ihs_chr*.tsv.gz 2>/dev/null | wc -l | tr -d ' ')
echo "▸ FAZ B (P12d iHS, polarized + Voight-2006 standardized):"
echo "  $N_DONE / 22 chromosomes done"
DONE_LIST=$(ls $PER_CHR/P12d_ihs_chr*.tsv.gz 2>/dev/null | sed 's/.*chr//; s/.tsv.gz//' | sort -n | tr '\n' ' ')
echo "  Done: $DONE_LIST"

RUNNING=$(ps aux | grep "P12d_ihs_per_chr" | grep -v grep)
if [ -n "$RUNNING" ]; then
  N_RUN=$(echo "$RUNNING" | wc -l | tr -d ' ')
  echo "  Active workers: $N_RUN"
  echo "$RUNNING" | awk '{print "    PID="$2"  CPU="$3"%  MEM="$4"%  CHR="$13}'
  for line in $(echo "$RUNNING" | awk '{print $13}'); do
    CHR=$line
    LOG=$LOG_DIR/chr${CHR}.log
    if [ -f $LOG ]; then
      LAST=$(tail -1 $LOG)
      echo "    chr$CHR latest: $LAST"
    fi
  done
fi

DRIVER=$(ps aux | grep "p12d_2way\|for batch_start" | grep -v grep | wc -l | tr -d ' ')
if [ "$DRIVER" -gt 0 ]; then
  echo "  Batch driver: alive"
else
  if [ "$N_DONE" -lt 22 ]; then
    echo "  ⚠ Batch driver NOT running (N_done=$N_DONE/22) — needs restart"
  fi
fi

# ── 2) Aggregated iHS ──
echo ""
echo "▸ Aggregated iHS (post-iHS, n_bins=50 standardize):"
echo "  P12d_genomewide_ihs.tsv.gz:    $(mark ${EVOSCZ_ROOT}/results/phase12/P12d_genomewide_ihs.tsv.gz)"
echo "  P12d_ihs_per_variant.tsv:      $(mark ${EVOSCZ_ROOT}/results/phase12/P12d_ihs_per_variant.tsv)"

# ── 3) Variant masters chain ──
echo ""
echo "▸ Variant master chain (re-build after iHS done):"
echo "  variant_master.parquet (Faz A v5):  $(mark ${EVOSCZ_ROOT}/results/phase11/variant_master.parquet)"
echo "  variant_master_v2.parquet (P12e): $(mark ${EVOSCZ_ROOT}/results/phase11/variant_master_v2.parquet)"
echo "  variant_master_v3.parquet (P13_5):$(mark ${EVOSCZ_ROOT}/results/phase11/variant_master_v3.parquet)"
echo "  variant_master_v4.parquet (P14d): $(mark ${EVOSCZ_ROOT}/results/phase11/variant_master_v4.parquet)"

# ── 4) Sumstats / matched-controls (iHS-independent re-runs already done) ──
echo ""
echo "▸ Sumstats + matched controls (iHS-independent re-runs, code-review fixes):"
echo "  matched_controls.tsv.gz (U3 reuse cap):              $(mark ${EVOSCZ_ROOT}/data/processed/matched_controls.tsv.gz)"
echo "  PGC3_SCZ_EAS.sumstats.gz (P14h Neff + revcomp):      $(mark ${EVOSCZ_ROOT}/results/phase14h/PGC3_SCZ_EAS.sumstats.gz)"
echo "  Wightman_AD.sumstats.gz (P14f palindromic+revcomp):  $(mark ${EVOSCZ_ROOT}/results/phase14f/Wightman_AD.sumstats.gz)"

# ── 5) Phase 13 robustness re-runs ──
echo ""
echo "▸ Phase 13 robustness (re-run after v2 master rebuilt):"
for f in P13a_matched_control_comparison.tsv \
         P13b_maf_residualized_results.tsv \
         P13c_bootstrap_ci.tsv \
         P13d_robustness_results.tsv; do
  P=${EVOSCZ_ROOT}/results/phase13/$f
  printf "  %-45s %s\n" "$f" "$(mark $P)"
done
echo "  P12g_decomposition_results.tsv (P12g): $(mark ${EVOSCZ_ROOT}/results/phase12/P12g_decomposition_results.tsv)"
echo "  P12h_final_results.tsv (P12h):         $(mark ${EVOSCZ_ROOT}/results/phase12/P12h_final_results.tsv)"

# ── 6) Phase 14 LDSC re-runs ──
echo ""
echo "▸ Phase 14 LDSC h² regressions (re-run after cluster re-build):"
echo "  P14e SCZ EUR clusters:    $(mark ${EVOSCZ_ROOT}/results/phase14e/PGC3_SCZ_clusters_baseline.results)"
echo "  P14f AD EUR clusters:     $(mark ${EVOSCZ_ROOT}/results/phase14f/Wightman_AD_clusters_baseline.results)"
echo "  P14g LR-LD masked SCZ:    $(mark ${EVOSCZ_ROOT}/results/phase14g/PGC3_SCZ_clusters_no_lrld.results)"
echo "  P14h SCZ EAS clusters:    $(mark ${EVOSCZ_ROOT}/results/phase14h/PGC3_SCZ_EAS_clusters_baseline.results)"

# ── 7) Manuscript / master doc ──
echo ""
echo "▸ Final outputs:"
echo "  Manuscript v3: $(mark ${EVOSCZ_ROOT}/manuscript/EVOSCZ_manuscript_v3.md)"
echo "  Master doc:    $(mark ${EVOSCZ_ROOT}/docs/EVOSCZ_FULL_PIPELINE_DOCUMENTATION.md)"

# ── 8) System health ──
echo ""
echo "▸ System health:"
LOAD=$(uptime | sed 's/.*load averages*: *//')
echo "  Load avg: $LOAD"
DISK=$(df -h ${EVOSCZ_ROOT} | tail -1 | awk '{print $4}')
echo "  Disk free: $DISK"
FREE_PAGES=$(vm_stat | awk '/Pages free/ {gsub("\\.",""); print $3}')
FREE_MB=$((FREE_PAGES * 16 / 1024))
COMP_PAGES=$(vm_stat | awk '/compressor/ {gsub("\\.",""); print $5; exit}')
COMP_MB=$((COMP_PAGES * 16 / 1024))
echo "  RAM: ${FREE_MB} MB free, ${COMP_MB} MB compressor (swap pressure if high)"

# ── 9) ETA ──
echo ""
echo "▸ ETA (rough):"
if [ "$N_DONE" -lt 22 ]; then
  REM=$((22 - N_DONE))
  PAIRS=$(( (REM + 1) / 2 ))
  EST_MIN=$((PAIRS * 25))
  EST_END=$(date -j -v+${EST_MIN}M '+%H:%M' 2>/dev/null)
  echo "  iHS: ~${EST_MIN} min ($PAIRS×~25min pairs) → ETA $EST_END for iHS"
  EST_END_FULL=$(date -j -v+$((EST_MIN+150))M '+%H:%M' 2>/dev/null)
  echo "  + downstream ~150 min → final ETA ~$EST_END_FULL"
else
  if [ ! -f ${EVOSCZ_ROOT}/results/phase12/P12d_genomewide_ihs.tsv.gz ] || \
     [ "$(stat -f '%m' ${EVOSCZ_ROOT}/results/phase12/P12d_genomewide_ihs.tsv.gz 2>/dev/null)" -lt "$CUTOFF" ]; then
    echo "  iHS done; aggregate + downstream ~150 min"
  fi
fi

echo ""
echo "=========================================================================="
echo "Logs:    $LOG_DIR/chr*.log"
echo "Driver:  /tmp/P12d_2way.log"
echo "Master:  ${EVOSCZ_ROOT}/docs/EVOSCZ_FULL_PIPELINE_DOCUMENTATION.md"
echo "=========================================================================="
