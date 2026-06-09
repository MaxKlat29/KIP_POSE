#!/usr/bin/env bash
# Launch the yolo26m-seg ANKER training DETACHED on the box (Story S-008 / T-134).
#
# Runs in a tmux session so it survives ssh disconnect; logs to a file the
# Queen can poll for `best.pt`. Training is a ~6-12h long-runner on the 3090.
#
# VRAM budget (24GB total): live-worker ~1.7GB + sam3 transient ~3.3GB +
# this training ~10-12GB (imgsz1024 batch12) -> headroom for live inference.
#
# Usage (on box):  bash box_src/launch_seg_train.sh
set -euo pipefail

VENV=/mnt/data/train-venv/bin/python
SRC=/mnt/data/bop/sdg_zivid_10k
OUT=/mnt/data/kip_pose/data/anker_seg
MODEL=/mnt/data/kip_pose/yolo26m-seg.pt
LOG=$OUT/train.log
SESS=yolo_seg_train

EPOCHS=${EPOCHS:-120}
IMGSZ=${IMGSZ:-1024}
BATCH=${BATCH:-12}

cd /mnt/data/kip_pose

if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "tmux session '$SESS' already exists — refusing to double-launch." >&2
  exit 1
fi

# Dataset must already be built (run train_yolo_seg_anker.py --prep-only first,
# or let the trainer rebuild — but we pass an already-built --out so it reuses).
# --skip-prep: reuse the dataset already built by `--prep-only` (no 20-min rescan,
# no rmtree of the built images). data.yaml + dataset_stats.json must exist in $OUT.
tmux new-session -d -s "$SESS" \
  "$VENV box_src/train_yolo_seg_anker.py \
     --src $SRC --out $OUT --model $MODEL --skip-prep \
     --epochs $EPOCHS --imgsz $IMGSZ --batch $BATCH \
     2>&1 | tee $LOG"

echo "LAUNCHED tmux session '$SESS'"
echo "  monitor:  ssh max@100.85.216.95 'tail -f $LOG'"
echo "  best.pt:  $OUT/_runs/seg/weights/best.pt  (canonical copy: $OUT/best.pt)"
echo "  metrics:  $OUT/metrics.json"
