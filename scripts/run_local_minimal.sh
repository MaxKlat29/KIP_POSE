#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local_minimal.sh  —  Run the minimal SDG demo on this machine
#
# Builds a self-contained scene (ground + dome-light + top-down camera),
# places real Anker parts at random positions, renders annotated frames
# (RGB + 2D bbox + semantic/instance seg + depth).
#
# No scene file needed — the scene is constructed programmatically.
# No Tailscale / Wake-on-LAN needed — runs on the local GPU.
#
# Usage:
#   bash scripts/run_local_minimal.sh
#
# Overridable env vars:
#   ISAACSIM_DIR   Isaac Sim root dir          (default: /home/age/isaacsim)
#   USD_DIR        Directory with part USDs     (default: ../USD-Files)
#   OUTPUT_DIR     Where to write results       (default: data/output/minimal)
#   NUM_RENDERS    Number of rendered frames    (default: 3)
#   NUM_OBJECTS    Parts per frame              (default: 6)
#   SEED           Random seed                  (default: 42)
#   HEADLESS       Set to 0 to open the GUI     (default: 1)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ISAACSIM_DIR="${ISAACSIM_DIR:-/home/age/isaacsim}"
USD_DIR="${USD_DIR:-/home/age/Downloads/pivarom-SDG-SDG-IsaacSim-USD-Files/SDG/IsaacSim/USD-Files}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/output/minimal}"
NUM_RENDERS="${NUM_RENDERS:-3}"
NUM_OBJECTS="${NUM_OBJECTS:-6}"
SEED="${SEED:-42}"
HEADLESS="${HEADLESS:-1}"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$ISAACSIM_DIR/python.sh" ]; then
  echo "ERROR: Isaac Sim not found at $ISAACSIM_DIR"
  echo "       Set ISAACSIM_DIR to your installation path."
  exit 1
fi

if [ ! -f "$USD_DIR/Anker_Kurz.usd" ] || [ ! -f "$USD_DIR/Anker_Lang.usd" ]; then
  echo "ERROR: Part USD files not found in $USD_DIR"
  echo "       Expected: Anker_Kurz.usd  Anker_Lang.usd"
  exit 1
fi

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  KIP_POSE  —  Minimal SDG (local)"
echo "  Isaac Sim : $ISAACSIM_DIR"
echo "  USD dir   : $USD_DIR"
echo "  Output    : $OUTPUT_DIR"
echo "  Renders   : $NUM_RENDERS   Objects/frame: $NUM_OBJECTS"
echo "═══════════════════════════════════════════════════════"

HEADLESS_FLAG=""
[ "$HEADLESS" = "0" ] && HEADLESS_FLAG="--no-headless"

"$ISAACSIM_DIR/python.sh" -u "$REPO_DIR/sim_code/run_minimal.py" \
  --usd-dir    "$USD_DIR"      \
  --output     "$OUTPUT_DIR"   \
  --num-renders "$NUM_RENDERS" \
  --num-objects "$NUM_OBJECTS" \
  --seed        "$SEED"        \
  $HEADLESS_FLAG

echo ""
echo "✅  Done! Output in: $OUTPUT_DIR"
echo ""
echo "   Overlay labels onto the RGB images:"
echo "   python3 $REPO_DIR/sim_code/visualize_labels.py --dir $OUTPUT_DIR"
