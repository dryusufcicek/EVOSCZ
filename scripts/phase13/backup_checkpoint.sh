#!/bin/bash
# Backup checkpoint - run periodically
mkdir -p /tmp/EVOSCZ_phase13_backup
mkdir -p ${EVOSCZ_ROOT}_backup_persistent

while true; do
    # /tmp backup (fast, RAM-resident, lost on reboot)
    cp ${EVOSCZ_ROOT}/results/phase13/*.tsv /tmp/EVOSCZ_phase13_backup/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase13/*.tsv.gz /tmp/EVOSCZ_phase13_backup/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase13/*.md /tmp/EVOSCZ_phase13_backup/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase11/variant_master_v2.parquet /tmp/EVOSCZ_phase13_backup/ 2>/dev/null
    
    # Persistent backup (survives reboot)
    cp ${EVOSCZ_ROOT}/results/phase13/*.tsv ${EVOSCZ_ROOT}_backup_persistent/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase13/*.tsv.gz ${EVOSCZ_ROOT}_backup_persistent/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase13/*.md ${EVOSCZ_ROOT}_backup_persistent/ 2>/dev/null
    cp ${EVOSCZ_ROOT}/results/phase11/variant_master_v2.parquet ${EVOSCZ_ROOT}_backup_persistent/ 2>/dev/null
    
    sleep 300  # 5 min
done
