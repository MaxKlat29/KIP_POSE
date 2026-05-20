#!/usr/bin/env bash
#
# Isaac Sim 5.1 headless install for the KIP_POSE GPU workstation.
#
# Target: Ubuntu Server 24.04 (headless), NVIDIA RTX 3090, no GUI.
# Installs Isaac Sim into a Python 3.11 venv on the large data disk via uv.
# Everything heavy (uv cache, python toolchain, venv) lives on /mnt/data so the
# small system SSD does not fill up.
#
# Idempotent: re-running skips uv / python 3.11 if already present.
#
# Usage (on the workstation, or via ssh):
#   bash setup_isaacsim_workstation.sh
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-/mnt/data}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/isaacsim-venv}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$DATA_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$DATA_DIR/uv-python"
export OMNI_KIT_ACCEPT_EULA=YES

log() { echo "[$(date '+%F %T')] $*"; }

log "=== Isaac Sim ${ISAACSIM_VERSION} headless install start ==="
log "DATA_DIR=$DATA_DIR  VENV_DIR=$VENV_DIR"

# 1. uv (fast Python package manager) — installs to ~/.local/bin, no sudo
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
log "uv: $(uv --version)"

# 2. Python 3.11 (Isaac Sim 5.x requires exactly 3.11)
log "ensuring Python 3.11 ..."
uv python install 3.11

# 3. Virtual environment on the data disk
if [ ! -d "$VENV_DIR" ]; then
  log "creating venv at $VENV_DIR ..."
  uv venv --python 3.11 "$VENV_DIR"
fi

# 4. Isaac Sim — the big download (~10-15 GB of wheels + extension caches)
log "installing isaacsim[all,extscache]==${ISAACSIM_VERSION} (this takes a while) ..."
VIRTUAL_ENV="$VENV_DIR" uv pip install \
  "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com

log "=== install done ==="
log "venv: $VENV_DIR"
log "verify headless with: scripts/verify_isaacsim.py"
