#!/bin/bash
# Phase 17D v22 CORRECT — outer loop, fresh Python process per perm (frees RAM).
# Uses P17d_v22_correct_perm.py (MAF + LD-impute, full 1744 C0 coverage).
# Usage:  bash P17d_v22_correct_loop.sh [N_PERM]
set -e
N=${1:-15}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for p in $(seq 0 $((N - 1))); do
  python3 "$SCRIPT_DIR/P17d_v22_correct_perm.py" "$p"
done
echo "[correct loop] all perms done."
