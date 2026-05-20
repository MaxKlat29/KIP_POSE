#!/usr/bin/env bash
#
# Remote SDG run for KIP_POSE.
#
# Wakes the GPU workstation via Wake-on-LAN (magic packet sent from the ai-desk
# Raspberry Pi on the same LAN), reaches it over Tailscale, runs the headless
# synthetic-data-generation pipeline in a tmux session, and (optionally) pulls
# the rendered dataset back.
#
# All settings are overridable via environment variables — see the block below.
#
# Usage:
#   bash scripts/wake_and_run.sh
#   NUM_RENDERS=200 MODE=random bash scripts/wake_and_run.sh
#
set -euo pipefail

# ── config ────────────────────────────────────────────────────
PI_HOST="${PI_HOST:-admin@100.117.146.46}"          # ai-desk, sends the magic packet
GPU_MAC="${GPU_MAC:-24:4b:fe:4b:79:e0}"             # workstation NIC MAC
GPU_HOST="${GPU_HOST:-max@100.85.216.95}"           # maxgpuserverobk over Tailscale
VENV="${VENV:-/mnt/data/isaacsim-venv}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/kip_pose}"
REPO_URL="${REPO_URL:-https://github.com/MaxKlat29/KIP_POSE}"
USD_DIR="${USD_DIR:-$REMOTE_REPO/data/usd}"
SCENE="${SCENE:-$USD_DIR/scene.usd}"
OUTPUT="${OUTPUT:-$REMOTE_REPO/data/output}"
NUM_RENDERS="${NUM_RENDERS:-50}"
MODE="${MODE:-tray}"
LOCAL_OUT="${LOCAL_OUT:-data/output}"
PULL_RESULTS="${PULL_RESULTS:-0}"                   # set to 1 to block + rsync back

log() { echo "[$(date '+%T')] $*"; }

# ── 1. wake the workstation ───────────────────────────────────
if ssh -o ConnectTimeout=5 "$GPU_HOST" true 2>/dev/null; then
  log "workstation already up"
else
  log "sending Wake-on-LAN magic packet via $PI_HOST ..."
  ssh -o ConnectTimeout=10 "$PI_HOST" "wakeonlan $GPU_MAC"
  log "waiting for SSH (up to 180s) ..."
  for i in $(seq 1 36); do
    sleep 5
    if ssh -o ConnectTimeout=5 "$GPU_HOST" true 2>/dev/null; then
      log "up after $((i * 5))s"; break
    fi
    [ "$i" -eq 36 ] && { log "ERROR: workstation did not come up"; exit 1; }
  done
fi

# ── 2. ensure the repo is present + current ───────────────────
log "syncing repo at $REMOTE_REPO ..."
ssh "$GPU_HOST" "[ -d '$REMOTE_REPO/.git' ] && (cd '$REMOTE_REPO' && git pull --ff-only) || git clone '$REPO_URL' '$REMOTE_REPO'"

# ── 3. write + launch the run in tmux (survives SSH drops) ─────
log "launching headless SDG run (renders=$NUM_RENDERS mode=$MODE) ..."
ssh "$GPU_HOST" "cat > '$REMOTE_REPO/_sdg_run.sh'" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_REPO"
export SDG_USD_DIR="$USD_DIR"
"$VENV/bin/python" sim_code/run_sdg.py \\
  --scene "$SCENE" --output "$OUTPUT" --usd-dir "$USD_DIR" \\
  --num-renders $NUM_RENDERS --mode $MODE
EOF

ssh "$GPU_HOST" "
  tmux kill-session -t sdg 2>/dev/null || true
  tmux new-session -d -s sdg 'bash \"$REMOTE_REPO/_sdg_run.sh\" > \"$REMOTE_REPO/sdg-run.log\" 2>&1; echo SDG_DONE_\$? >> \"$REMOTE_REPO/sdg-run.log\"'
"
log "run launched in tmux session 'sdg'. Follow with:"
log "  ssh $GPU_HOST 'tail -f $REMOTE_REPO/sdg-run.log'"

# ── 4. (optional) block until done, pull results, sleep host ──
if [ "$PULL_RESULTS" = "1" ]; then
  log "waiting for run to finish ..."
  while ! ssh "$GPU_HOST" "grep -q SDG_DONE '$REMOTE_REPO/sdg-run.log'" 2>/dev/null; do
    sleep 30
  done
  mkdir -p "$LOCAL_OUT"
  log "pulling results -> $LOCAL_OUT ..."
  rsync -avz "$GPU_HOST:$OUTPUT/" "$LOCAL_OUT/"
  log "done."
fi
