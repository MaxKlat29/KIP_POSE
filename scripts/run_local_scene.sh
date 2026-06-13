#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local_scene.sh  —  Render a full assembled scene locally
#
# Opens one of the real USD scenes (GST_Scene, NEURA_LARA5_Pose_Zivid_Detection,
# …), optionally spawns Anker parts, and writes RGB + 2D bbox + oriented-bbox +
# semantic/instance segmentation + depth.
#
# Uses run_scene.py which adds oriented 2D bounding boxes (PCA on instance masks).
#
# Usage:
#   bash scripts/run_local_scene.sh
#   SCENE=GST_Scene.usd bash scripts/run_local_scene.sh
#   SPAWN=random NUM_OBJECTS=8 bash scripts/run_local_scene.sh
#
# Overridable env vars:
#   ISAACSIM_DIR   Isaac Sim root dir             (default: /home/age/isaacsim)
#   USD_DIR        Directory with all USD files    (default: ../USD-Files)
#   SCENE          Scene file name inside USD_DIR  (default: NEURA_LARA5_Pose_Zivid_Detection.usd)
#   CAMERA         Camera prim path in the scene   (default: /World/Zivid)
#   SPAWN          none | tray | random            (default: tray)
#   NUM_OBJECTS    Parts to spawn per frame        (default: 6)
#   NUM_RENDERS    Number of rendered frames        (default: 5)
#   OUTPUT_DIR     Where to write results           (default: data/output/scene)
#   ADD_LIGHT      Set to 1 to add a fallback dome  (default: 0)
#   HIDE           Comma-sep prim name substrings to hide (default: "")
#   HEADLESS       Set to 0 to open the GUI         (default: 1)
#   KEEP_OPEN      Set to 1 to keep the UI open after rendering — press Enter to close (default: 0)
#   PHYSICS_Z      Z height (m) for an invisible flat ground collider (fallback; default: unset)
#
# Note: CollisionAPI is applied automatically to /World/Basiswagen/Basiswagen if that prim
#       exists in the scene (GST_Scene.usd). No extra flag needed.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ISAACSIM_DIR="${ISAACSIM_DIR:-/home/age/isaacsim}"
USD_DIR="${USD_DIR:-/home/age/Downloads/pivarom-SDG-SDG-IsaacSim-USD-Files/SDG/IsaacSim/USD-Files}"
SCENE_FILE="${SCENE:-GST_Scene.usd}"
SCENE_PATH="$USD_DIR/$SCENE_FILE"
CAMERA="${CAMERA:-/World/Zivid}"
SPAWN="${SPAWN:-tray}"
NUM_OBJECTS="${NUM_OBJECTS:-6}"
NUM_RENDERS="${NUM_RENDERS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/output/scene}"
ADD_LIGHT="${ADD_LIGHT:-0}"
HIDE="${HIDE:-}"
HEADLESS="${HEADLESS:-1}"
KEEP_OPEN="${KEEP_OPEN:-0}"
PHYSICS_Z="${PHYSICS_Z:-}"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$ISAACSIM_DIR/python.sh" ]; then
  echo "ERROR: Isaac Sim not found at $ISAACSIM_DIR"
  echo "       Set ISAACSIM_DIR to your installation path."
  exit 1
fi

if [ ! -f "$SCENE_PATH" ]; then
  echo "ERROR: Scene file not found: $SCENE_PATH"
  echo ""
  echo "Available scenes in $USD_DIR:"
  ls "$USD_DIR"/*.usd 2>/dev/null | xargs -n1 basename || echo "  (none found)"
  echo ""
  echo "Set SCENE=<filename.usd> to pick a different scene."
  exit 1
fi

if [ ! -f "$USD_DIR/Anker_Kurz.usd" ]; then
  echo "ERROR: Anker_Kurz.usd not found in $USD_DIR"
  exit 1
fi

# ── run ───────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  KIP_POSE  —  Scene SDG (local)"
echo "  Isaac Sim : $ISAACSIM_DIR"
echo "  Scene     : $SCENE_PATH"
echo "  Camera    : $CAMERA"
echo "  USD dir   : $USD_DIR"
echo "  Spawn     : $SPAWN   Objects: $NUM_OBJECTS"
echo "  Output    : $OUTPUT_DIR"
echo "  Renders   : $NUM_RENDERS"
echo "═══════════════════════════════════════════════════════"

EXTRA_FLAGS=""
[ "$ADD_LIGHT"  = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --add-light"
[ -n "$HIDE"          ] && EXTRA_FLAGS="$EXTRA_FLAGS --hide $HIDE"
[ "$HEADLESS"   = "0" ] && EXTRA_FLAGS="$EXTRA_FLAGS --no-headless"
[ "$KEEP_OPEN"  = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --keep-open"
[ -n "$PHYSICS_Z"     ] && EXTRA_FLAGS="$EXTRA_FLAGS --physics-z $PHYSICS_Z"

"$ISAACSIM_DIR/python.sh" -u "$REPO_DIR/sim_code/run_scene.py" \
  --scene       "$SCENE_PATH"  \
  --usd-dir     "$USD_DIR"     \
  --output      "$OUTPUT_DIR"  \
  --camera      "$CAMERA"      \
  --spawn       "$SPAWN"       \
  --num-objects "$NUM_OBJECTS" \
  --num-renders "$NUM_RENDERS" \
  $EXTRA_FLAGS

echo ""
echo "✅  Done! Output in: $OUTPUT_DIR"
echo ""
echo "   Overlay labels:"
echo "   python3 $REPO_DIR/sim_code/visualize_labels.py --dir $OUTPUT_DIR"
echo ""
echo "   Scene variants you can try:"
echo "   SCENE=GST_Foto_Scene.usd bash $0"
echo "   SCENE=NEURA_LARA5_Pose_Zivid_Detection.usd bash $0"
echo "   SPAWN=random bash $0"
echo ""
echo "   Explore with the UI open:"
echo "   HEADLESS=0 KEEP_OPEN=1 SPAWN=none bash $0"
