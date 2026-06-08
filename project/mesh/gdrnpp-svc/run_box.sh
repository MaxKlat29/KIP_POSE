#!/usr/bin/env bash
# Launch gdrnpp-svc on the GPU box as a SEPARATE native-venv daemon (port 8012).
# Isolated from the live :8078 worker (different process, different VRAM fraction).
# NEVER pkills uvicorn / the live worker.
#
# Usage (on the box, from the rsynced project tree root):
#   bash project/mesh/gdrnpp-svc/run_box.sh            # foreground
#   nohup bash project/mesh/gdrnpp-svc/run_box.sh &    # background
set -euo pipefail

GDRN_ROOT="${GDRN_ROOT:-/mnt/data/bop/repos/gdrnpp}"
GDRNPP_VENV="${GDRNPP_VENV:-/mnt/data/bop/gdrnpp-venv}"
GDRNPP_PORT="${GDRNPP_PORT:-8012}"
GDRNPP_MEMFRAC="${GDRNPP_MEMFRAC:-0.30}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../project/mesh/gdrnpp-svc
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"                   # repo root
BOX_SRC="$PROJECT_ROOT/box_src"                               # kip_infer_worker.py (T-115 det-root)

# kip_infer_worker imports torch at top -> import it under the gdrnpp venv. Put both
# box_src and the gdrnpp repo on PYTHONPATH so `import kip_infer_worker` resolves.
export PYTHONPATH="$BOX_SRC:$GDRN_ROOT:$GDRN_ROOT/core/gdrn_modeling:${PYTHONPATH:-}"
export GDRN_ROOT GDRNPP_PORT GDRNPP_MEMFRAC

echo "[run_box] gdrnpp-svc :$GDRNPP_PORT  venv=$GDRNPP_VENV  memfrac=$GDRNPP_MEMFRAC" >&2
echo "[run_box] PYTHONPATH=$PYTHONPATH" >&2
exec "$GDRNPP_VENV/bin/python" "$HERE/app.py"
