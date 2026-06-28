#!/bin/bash
# Phase 17D v22 LEAN — outer loop, fresh Python process per perm (frees RAM).
# Usage:  bash P17d_v22_lean_loop.sh [N_PERM]
set -e
N=${1:-15}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for p in $(seq 0 $((N - 1))); do
  python3 "$SCRIPT_DIR/P17d_v22_lean_perm.py" "$p"
done
echo "[lean loop] all perms done."
