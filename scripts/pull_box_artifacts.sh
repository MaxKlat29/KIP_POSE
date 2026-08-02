#!/usr/bin/env bash
# pull_box_artifacts.sh — fetch all e2e_finish outputs + trained models from
# the GPU box to the local repo so the Three.js viewer + infer.ipynb can run
# without ssh.
#
# What it pulls:
#   project/temp/                — pose_result_*.json (all stages), overlays,
#                                  splitscreen, viewer-snap PNGs
#   project/models/              — trained GDRNPP best.pth per object + detector
#   project/docs/RESULTS_PHASE2.md, AB_LEVERS.md, PROJECT_REPORT.md — final report
#
# Usage:
#   scripts/pull_box_artifacts.sh
#
# Re-runnable: rsync skips unchanged files.
set -euo pipefail

BOX="${BOX:-${GPU_HOST:?set GPU_HOST or BOX (user@host) — see project/.env.example}}"
BOX_REPO="${BOX_REPO:-/mnt/data/kip_pose}"
BOX_GDRN_OUT="${BOX_GDRN_OUT:-/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO}"

# Resolve repo-local POSE/ root (script lives at POSE/scripts/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PROJECT="$ROOT/project"
mkdir -p "$PROJECT/temp" "$PROJECT/models" "$PROJECT/docs"

say() { printf '[pull] %s\n' "$*"; }

say "pulling artifacts from $BOX:$BOX_REPO  ->  $ROOT/"

# --- 1) temp/ (viewer artifacts: pose_result_*.json, overlays, snaps) -------
say "1/3 temp/  (pose_result_*.json, overlays, viewer snaps)"
rsync -avz --info=progress2 --partial \
  --include='pose_result*.json' --include='*.png' --include='*.jpg' \
  --include='*/' --exclude='*' \
  "$BOX:$BOX_REPO/project/temp/" "$PROJECT/temp/" || say "WARN rsync temp/ failed"

# --- 2) models/  (per-object best-by-val + detector) ------------------------
say "2/3 models/ (anker_kurz / anker_lang / zahnrad best.pth + detector.pt)"
for obj in anker_kurz anker_lang zahnrad; do
  remote_pth="$BOX_GDRN_OUT/$obj/model_best.pth"
  # model_best.pth on the box is a symlink to the AR-best checkpoint; -L follows.
  if ssh "$BOX" "test -e $remote_pth"; then
    say "  $obj  <- $remote_pth"
    rsync -avL --info=progress2 --partial \
      "$BOX:$remote_pth" "$PROJECT/models/$obj.pth" || say "WARN rsync $obj failed"
  else
    say "  $obj  -> not yet produced on box (training/best-by-val not done) — skipping"
  fi
done
# Detector
if ssh "$BOX" "test -e $BOX_REPO/data/detector_armvis/detector.pt"; then
  rsync -avz --info=progress2 --partial \
    "$BOX:$BOX_REPO/data/detector_armvis/detector.pt" "$PROJECT/models/detector.pt" \
    || say "WARN detector.pt rsync failed"
fi

# --- 3) docs/  (RESULTS_PHASE2.md, AB_LEVERS.md, PROJECT_REPORT.md) ---------
say "3/3 docs/ (RESULTS_PHASE2.md + AB_LEVERS.md + PROJECT_REPORT.md)"
for doc in RESULTS_PHASE2.md AB_LEVERS.md PROJECT_REPORT.md; do
  if ssh "$BOX" "test -e $BOX_REPO/project/docs/$doc"; then
    rsync -avz --partial "$BOX:$BOX_REPO/project/docs/$doc" "$PROJECT/docs/$doc" \
      || say "WARN $doc rsync failed"
  fi
done

say "DONE."
ls -la "$PROJECT/temp" | grep -E "pose_result|overlay|viewer" | head -10 || true
ls -la "$PROJECT/models" | grep -v "^d" | head -5 || true
