#!/usr/bin/env bash
# =============================================================================
# gpu_run.sh — Reusable GPU-box run harness for the POSE project
# -----------------------------------------------------------------------------
# Wakes the RTX-3090 workstation (Wake-on-LAN via the Raspberry-Pi relay if it
# is asleep), waits for SSH to come up, runs an arbitrary command ON the box,
# and optionally rsyncs result artifacts back to the local machine.
#
# It is intentionally dumb and composable: ONE command in, optional ONE pull
# path back. Heavy/long jobs should be wrapped by the *caller* in nohup on the
# box (this script does not daemonize for you) — see examples below.
#
# -----------------------------------------------------------------------------
# USAGE
#   box_src/gpu_run.sh [OPTIONS] -- <remote command ...>
#
# OPTIONS
#   -c, --cmd     STR   Remote command to run (alt. to the `-- <cmd>` form).
#   -p, --pull    SPEC  rsync pull spec "REMOTE_PATH:LOCAL_PATH". Repeatable.
#                       REMOTE_PATH is on the box, LOCAL_PATH on this machine.
#   -d, --workdir DIR   cd into DIR on the box before running the command.
#                       Default: $REMOTE_REPO (/mnt/data/kip_pose).
#   -t, --timeout SEC   SSH command timeout (default 0 = no timeout).
#       --no-wake       Do not attempt Wake-on-LAN; fail fast if box is down.
#       --tty           Allocate a TTY (ssh -t) — for interactive/sudo cmds.
#   -h, --help          Show this help.
#
# ENV OVERRIDES (defaults match Brain environment/gpu-workstation.md)
#   BOX_USER          (max)
#   BOX_HOST          (100.85.216.95)            # Tailscale IP
#   BOX_MAC           (24:4b:fe:4b:79:e0)
#   WOL_RELAY         (admin@100.117.146.46)      # Raspberry-Pi WoL relay
#   REMOTE_REPO       (/mnt/data/kip_pose)
#   SSH_OPTS          (-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
#   WAKE_WAIT_SECS    (90)   total seconds to wait for SSH after a wake
#
# EXIT CODES
#   0  success
#   2  bad arguments
#   3  box unreachable (and --no-wake, or wake timed out)
#   >0 remote command's own exit code
#
# -----------------------------------------------------------------------------
# EXAMPLES
#
#   # 1. Quick one-shot — check GPU on the box:
#   box_src/gpu_run.sh -- 'nvidia-smi'
#
#   # 2. Run a smoke import inside the bop-venv:
#   box_src/gpu_run.sh -- '/mnt/data/bop/bop-venv/bin/python -c "import torch; print(torch.cuda.is_available())"'
#
#   # 3. Kick off a HEAVY download as a detached nohup job (non-blocking),
#   #    then poll its log later with another gpu_run.sh call:
#   box_src/gpu_run.sh -- \
#     'cd /mnt/data/bop && nohup ./fetch_weights.sh > logs/fetch.log 2>&1 & echo "PID=$!"'
#   # ...later...
#   box_src/gpu_run.sh -- 'tail -n 30 /mnt/data/bop/logs/fetch.log; pgrep -f fetch_weights || echo DONE'
#
#   # 4. Run an eval and pull the results dir back to the worktree:
#   box_src/gpu_run.sh \
#     -p '/mnt/data/bop/results/run42:./results/run42' \
#     -- 'cd /mnt/data/bop && bop-venv/bin/python eval.py --out results/run42'
#
#   # 5. Force-fail if box is asleep (CI-style, no WoL):
#   box_src/gpu_run.sh --no-wake -- 'echo alive'
#
# NOTE on nohup: this harness does NOT background remote commands itself; the
# SSH session blocks until the remote command returns. For long jobs, put the
# `nohup ... &` INSIDE the remote command (example 3) so SSH returns immediately
# while the work continues on the box. Poll with a second invocation.
# =============================================================================
set -euo pipefail

# ---- defaults (overridable via env) -----------------------------------------
BOX_USER="${BOX_USER:-max}"
BOX_HOST="${BOX_HOST:-100.85.216.95}"
BOX_MAC="${BOX_MAC:-24:4b:fe:4b:79:e0}"
WOL_RELAY="${WOL_RELAY:-admin@100.117.146.46}"
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/kip_pose}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"
WAKE_WAIT_SECS="${WAKE_WAIT_SECS:-90}"

# ---- arg parsing ------------------------------------------------------------
CMD=""
WORKDIR=""
TIMEOUT="0"
NO_WAKE=0
USE_TTY=0
declare -a PULLS=()

die()  { echo "gpu_run: $*" >&2; exit 2; }
info() { echo "[gpu_run] $*" >&2; }

print_help() { sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cmd)     CMD="${2:-}"; shift 2 ;;
    -p|--pull)    PULLS+=("${2:-}"); shift 2 ;;
    -d|--workdir) WORKDIR="${2:-}"; shift 2 ;;
    -t|--timeout) TIMEOUT="${2:-0}"; shift 2 ;;
    --no-wake)    NO_WAKE=1; shift ;;
    --tty)        USE_TTY=1; shift ;;
    -h|--help)    print_help ;;
    --)           shift; CMD="$*"; break ;;
    *)            die "unknown option: $1 (use -- before the remote command)" ;;
  esac
done

[[ -n "$CMD" ]] || die "no remote command given (use -- <cmd> or -c)"

SSH=(ssh $SSH_OPTS)
[[ "$USE_TTY" -eq 1 ]] && SSH+=(-t)
TARGET="${BOX_USER}@${BOX_HOST}"

# ---- reachability + wake-on-LAN ---------------------------------------------
box_ssh_ok() { ${SSH[@]} -o BatchMode=yes "$TARGET" 'true' >/dev/null 2>&1; }

ensure_box_up() {
  if box_ssh_ok; then info "box reachable: $TARGET"; return 0; fi
  if [[ "$NO_WAKE" -eq 1 ]]; then info "box down and --no-wake set"; exit 3; fi
  info "box not reachable — sending Wake-on-LAN via $WOL_RELAY"
  ssh $SSH_OPTS -o BatchMode=yes "$WOL_RELAY" "wakeonlan $BOX_MAC" \
    || info "WoL relay command returned non-zero (continuing to poll anyway)"
  local waited=0 step=10
  while (( waited < WAKE_WAIT_SECS )); do
    sleep "$step"; waited=$((waited+step))
    if box_ssh_ok; then info "box came up after ${waited}s"; return 0; fi
    info "  ...waiting for SSH (${waited}/${WAKE_WAIT_SECS}s)"
  done
  info "box did not come up within ${WAKE_WAIT_SECS}s"; exit 3
}

ensure_box_up

# ---- build + run remote command ---------------------------------------------
EFFECTIVE_WORKDIR="${WORKDIR:-$REMOTE_REPO}"
REMOTE="cd '${EFFECTIVE_WORKDIR}' 2>/dev/null || true; ${CMD}"
[[ "$TIMEOUT" != "0" ]] && REMOTE="timeout ${TIMEOUT} bash -lc \"\$(cat)\" <<'__GPU_RUN_EOF__'
${CMD}
__GPU_RUN_EOF__"

info "running on box [workdir=${EFFECTIVE_WORKDIR}]:"
info "  ${CMD}"
set +e
${SSH[@]} "$TARGET" "bash -lc $(printf '%q' "$REMOTE")"
RC=$?
set -e
info "remote command exit code: ${RC}"

# ---- optional rsync pulls ---------------------------------------------------
for spec in "${PULLS[@]:-}"; do
  [[ -z "$spec" ]] && continue
  remote_path="${spec%%:*}"
  local_path="${spec#*:}"
  [[ "$remote_path" == "$spec" || -z "$local_path" ]] && die "bad --pull spec '$spec' (need REMOTE:LOCAL)"
  mkdir -p "$(dirname "$local_path")"
  info "rsync pull: ${TARGET}:${remote_path}  ->  ${local_path}"
  rsync -az --info=progress2 -e "ssh $SSH_OPTS" \
    "${TARGET}:${remote_path}" "$local_path"
done

exit "$RC"
