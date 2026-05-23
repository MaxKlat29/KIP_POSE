#!/usr/bin/env bash
# =============================================================================
# eval_bop.sh - thin local wrapper that runs the BOP eval harness ON the GPU box
# -----------------------------------------------------------------------------
# Ships eval_bop.py to the box, runs it inside the bop-venv via gpu_run.sh, and
# pulls the report (report.json + report.txt) back to the worktree.
#
# It does NOT touch training and reads the BOP dataset read-only.
#
# USAGE
#   # Self-validation (no predictions needed):
#   box_src/eval_bop.sh --self-test
#
#   # Score real predictions (a BOP-results CSV that lives ON the box):
#   box_src/eval_bop.sh --preds /mnt/data/bop/results/gdrnpp/preds.csv
#
#   # Add VSD (slower; needs the vispy/EGL renderer + depth maps):
#   box_src/eval_bop.sh --self-test --vsd
#
# Any extra args are forwarded verbatim to eval_bop.py.
#
# ENV OVERRIDES
#   DATASET_DIR  (/mnt/data/kip_pose/project/bop/pose_isaac)
#   SPLIT        (val)
#   REMOTE_PY    (/mnt/data/bop/bop-venv/bin/python)
#   REMOTE_DIR   (/mnt/data/bop)                 # where eval_bop.py is placed
#   OUT_REMOTE   (/mnt/data/bop/results/eval_local)
#   OUT_LOCAL    (./results/eval)                # pulled report destination
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_RUN="${GPU_RUN:-$HERE/../../S-204/box_src/gpu_run.sh}"
[[ -x "$GPU_RUN" ]] || GPU_RUN="$HERE/gpu_run.sh"   # fallback if co-located

DATASET_DIR="${DATASET_DIR:-/mnt/data/kip_pose/project/bop/pose_isaac}"
SPLIT="${SPLIT:-val}"
REMOTE_PY="${REMOTE_PY:-/mnt/data/bop/bop-venv/bin/python}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/data/bop}"
OUT_REMOTE="${OUT_REMOTE:-/mnt/data/bop/results/eval_local}"
OUT_LOCAL="${OUT_LOCAL:-./results/eval}"

BOX_USER="${BOX_USER:-max}"
BOX_HOST="${BOX_HOST:-100.85.216.95}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

echo "[eval_bop.sh] shipping eval_bop.py -> ${BOX_USER}@${BOX_HOST}:${REMOTE_DIR}/" >&2
scp $SSH_OPTS "$HERE/eval_bop.py" "${BOX_USER}@${BOX_HOST}:${REMOTE_DIR}/eval_bop.py"

REMOTE_CMD="${REMOTE_PY} ${REMOTE_DIR}/eval_bop.py \
  --dataset-dir '${DATASET_DIR}' --split '${SPLIT}' \
  --out '${OUT_REMOTE}' $*"

echo "[eval_bop.sh] running on box:" >&2
echo "    ${REMOTE_CMD}" >&2

"$GPU_RUN" -p "${OUT_REMOTE}:${OUT_LOCAL}" -- "$REMOTE_CMD"
echo "[eval_bop.sh] report pulled to ${OUT_LOCAL}/" >&2
