#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local_sdg.sh  —  Full synthetic-data-generation run on this machine
#
# Opens a scene that contains the tray + Zivid camera, spawns Anker parts
# (tray or random mode), lets physics settle, then captures annotated frames.
# Uses run_sdg.py — the headless standalone runner.
#
# Usage:
#   bash scripts/run_local_sdg.sh
#   MODE=random NUM_RENDERS=20 bash scripts/run_local_sdg.sh
#
# Overridable env vars:
#   ISAACSIM_DIR   Isaac Sim root dir             (default: /home/age/isaacsim)
#   USD_DIR        Directory with all USD files    (default: ../USD-Files)
#   SCENE          Scene USD (must have tray+cam)  (default: NEURA_LARA5_Pose_Zivid_Detection.usd)
#   CAMERA         Camera prim path               (default: /World/Zivid)
#   MODE           tray | random                  (default: tray)
#   NUM_RENDERS    Frames to generate             (default: 10)
#   NUM_OBJECTS    Parts per frame                (default: 5)
#   OUTPUT_DIR     Where to write results          (default: data/output/sdg)
#   HEADLESS       Set to 0 to show the GUI        (default: 1)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ISAACSIM_DIR="${ISAACSIM_DIR:-/home/age/isaacsim}"
USD_DIR="${USD_DIR:-/home/age/Downloads/pivarom-SDG-SDG-IsaacSim-USD-Files/SDG/IsaacSim/USD-Files}"
SCENE_FILE="${SCENE:-NEURA_LARA5_Pose_Zivid_Detection.usd}"
SCENE_PATH="$USD_DIR/$SCENE_FILE"
CAMERA="${CAMERA:-/World/Zivid}"
MODE="${MODE:-tray}"
NUM_RENDERS="${NUM_RENDERS:-10}"
NUM_OBJECTS="${NUM_OBJECTS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/output/sdg}"
HEADLESS="${HEADLESS:-1}"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$ISAACSIM_DIR/python.sh" ]; then
  echo "ERROR: Isaac Sim not found at $ISAACSIM_DIR"
  echo "       Set ISAACSIM_DIR to your installation path."
  exit 1
fi

if [ ! -f "$SCENE_PATH" ]; then
  echo "ERROR: Scene file not found: $SCENE_PATH"
  echo ""
  echo "Available USD files in $USD_DIR:"
  ls "$USD_DIR"/*.usd 2>/dev/null | xargs -n1 basename || echo "  (none)"
  echo ""
  echo "NOTE: run_sdg.py requires a scene that already contains the tray and"
  echo "      the Zivid camera prim ($CAMERA). If that scene has a different"
  echo "      camera path, pass CAMERA=/Your/CamPath."
  echo ""
  echo "TIP:  For a camera-free minimal demo, use run_local_minimal.sh instead."
  exit 1
fi

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  KIP_POSE  —  Full SDG (local)"
echo "  Isaac Sim : $ISAACSIM_DIR"
echo "  Scene     : $SCENE_PATH"
echo "  Camera    : $CAMERA"
echo "  USD dir   : $USD_DIR"
echo "  Mode      : $MODE   Objects: $NUM_OBJECTS"
echo "  Renders   : $NUM_RENDERS"
echo "  Output    : $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════"

HEADLESS_FLAG=""
[ "$HEADLESS" = "0" ] && HEADLESS_FLAG="--no-headless"

SDG_USD_DIR="$USD_DIR" \
SDG_OUTPUT_DIR="$OUTPUT_DIR" \
SDG_CAMERA_PATH="$CAMERA" \
"$ISAACSIM_DIR/python.sh" -u "$REPO_DIR/sim_code/run_sdg.py" \
  --scene       "$SCENE_PATH"  \
  --usd-dir     "$USD_DIR"     \
  --output      "$OUTPUT_DIR"  \
  --camera      "$CAMERA"      \
  --mode        "$MODE"        \
  --num-renders "$NUM_RENDERS" \
  --num-objects "$NUM_OBJECTS" \
  $HEADLESS_FLAG

echo ""
echo "✅  Done! $NUM_RENDERS samples written to: $OUTPUT_DIR"
echo ""
echo "   Overlay labels:"
echo "   python3 $REPO_DIR/sim_code/visualize_labels.py --dir $OUTPUT_DIR"
