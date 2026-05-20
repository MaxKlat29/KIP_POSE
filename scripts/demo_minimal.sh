#!/usr/bin/env bash
#
# Minimal end-to-end demo for KIP_POSE.
#
# Wakes the GPU workstation (Wake-on-LAN via the ai-desk Raspberry Pi), reaches
# it over Tailscale, runs the minimal Isaac Sim SDG (spawns the real Anker parts
# and renders a top-down image + labels), pulls the result back, and overlays
# the labels into an annotated preview image.
#
# One command: simulation -> top-down image with generated labels.
#
#   bash scripts/demo_minimal.sh
#
set -euo pipefail

PI_HOST="${PI_HOST:-admin@100.117.146.46}"
GPU_MAC="${GPU_MAC:-24:4b:fe:4b:79:e0}"
GPU_HOST="${GPU_HOST:-max@100.85.216.95}"
VENV="${VENV:-/mnt/data/isaacsim-venv}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/kip_pose}"
USD_DIR="${USD_DIR:-$REMOTE_REPO/data/SDG/IsaacSim/USD-Files}"
OUTPUT="${OUTPUT:-$REMOTE_REPO/data/output/minimal}"
NUM_RENDERS="${NUM_RENDERS:-3}"
NUM_OBJECTS="${NUM_OBJECTS:-6}"
LOCAL_OUT="${LOCAL_OUT:-data/output/minimal}"

log() { echo "[$(date '+%T')] $*"; }

# 1. wake
if ssh -o ConnectTimeout=5 "$GPU_HOST" true 2>/dev/null; then
  log "workstation already up"
else
  log "Wake-on-LAN via $PI_HOST ..."
  ssh -o ConnectTimeout=10 "$PI_HOST" "wakeonlan $GPU_MAC"
  log "waiting for SSH (up to 180s) ..."
  for i in $(seq 1 36); do
    sleep 5
    ssh -o ConnectTimeout=5 "$GPU_HOST" true 2>/dev/null && { log "up after $((i*5))s"; break; }
    [ "$i" -eq 36 ] && { log "ERROR: did not come up"; exit 1; }
  done
fi

# 2. sync repo
log "syncing repo ..."
ssh "$GPU_HOST" "[ -d '$REMOTE_REPO/.git' ] && (cd '$REMOTE_REPO' && git pull -q --ff-only) || git clone -q https://github.com/MaxKlat29/KIP_POSE '$REMOTE_REPO'"

# 3. run the minimal SDG in tmux, wait for completion
log "running minimal SDG (renders=$NUM_RENDERS objects=$NUM_OBJECTS) ..."
ssh "$GPU_HOST" "
  rm -f '$REMOTE_REPO/demo.log'; rm -rf '$OUTPUT'
  tmux kill-session -t demo 2>/dev/null || true
  tmux new-session -d -s demo '
    $VENV/bin/python -u $REMOTE_REPO/sim_code/run_minimal.py \
      --usd-dir $USD_DIR --output $OUTPUT \
      --num-renders $NUM_RENDERS --num-objects $NUM_OBJECTS \
      > $REMOTE_REPO/demo.log 2>&1; echo DEMO_DONE_\$? >> $REMOTE_REPO/demo.log'
"
for i in $(seq 1 90); do
  ssh "$GPU_HOST" "grep -q DEMO_DONE '$REMOTE_REPO/demo.log'" 2>/dev/null && break
  sleep 5
done
ssh "$GPU_HOST" "grep -E '\[minimal|DEMO_DONE' '$REMOTE_REPO/demo.log' | tail -8"

# 4. pull results
log "pulling results -> $LOCAL_OUT ..."
mkdir -p "$LOCAL_OUT"
rsync -az "$GPU_HOST:$OUTPUT/" "$LOCAL_OUT/"

# 5. overlay labels
log "rendering annotated previews ..."
python3 sim_code/visualize_labels.py --dir "$LOCAL_OUT"

log "done. Annotated top-down images:"
ls "$LOCAL_OUT"/annotated_*.png
# open the first one on macOS
command -v open >/dev/null && open "$LOCAL_OUT/annotated_0000.png" || true
