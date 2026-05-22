#!/usr/bin/env bash
#
# End-to-end POSE smoke run (S-012) — one command, no trained models needed.
#
# Runs the inference + alignment pipeline on the bundled dummy SDG scene,
# validates the produced pose_result.json against the contract schema, and
# prints how to open the 3D viewer on the result.
#
#   bash scripts/run_e2e.sh
#
# Flow:
#   1. pick a python with numpy + PIL (system python3, else .venv-faces)
#   2. pipeline.run_pipeline  data/examples/dummy_scene -> data/output/pose_result.json
#      (the runner gates the result against docs/pose_result.schema.json BEFORE
#       writing — stdlib gate always, jsonschema gate when available)
#   3. independent re-validation of the written file (belt + suspenders)
#   4. print PASS/FAIL + the viewer command
#
# Idempotent: the output file is overwritten each run. Green without any trained
# checkpoint (classifier fallback path -> confidence 0.0, still schema-valid).
#
set -euo pipefail

# --- resolve repo root regardless of CWD --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SCENE_DIR="${SCENE_DIR:-data/examples/dummy_scene}"
OUT="${OUT:-data/output/pose_result.json}"
SCHEMA="${SCHEMA:-docs/pose_result.schema.json}"
VIEWER_PORT="${VIEWER_PORT:-8000}"

log()  { echo "[$(date '+%T')] $*"; }
fail() { echo; echo "  ============================================"; \
         echo "  ❌  E2E FAIL: $*"; \
         echo "  ============================================"; exit 1; }

# --- 1. pick a python with the runtime deps -----------------------------------
pick_python() {
  for cand in "${PYTHON:-}" python3 ".venv-faces/bin/python"; do
    [ -z "$cand" ] && continue
    if "$cand" -c "import numpy, PIL" >/dev/null 2>&1; then
      echo "$cand"; return 0
    fi
  done
  return 1
}
PY="$(pick_python)" || fail "no python with numpy + PIL found (tried \$PYTHON, python3, .venv-faces/bin/python)"
log "python: $PY ($("$PY" --version 2>&1))"
if "$PY" -c "import jsonschema" >/dev/null 2>&1; then
  log "schema gate: stdlib + jsonschema"
else
  log "schema gate: stdlib only (jsonschema not installed — optional)"
fi

# --- 2. inputs present? -------------------------------------------------------
[ -d "$SCENE_DIR" ]                       || fail "scene dir missing: $SCENE_DIR"
[ -f "$SCENE_DIR/rgb_0000.png" ]          || fail "scene rgb missing: $SCENE_DIR/rgb_0000.png"
[ -f "$SCENE_DIR/bbox_2d_0000.json" ]     || fail "scene bboxes missing: $SCENE_DIR/bbox_2d_0000.json"
[ -f "$SCHEMA" ]                          || fail "contract schema missing: $SCHEMA"

# --- 3. run the pipeline (idempotent: fresh output) ---------------------------
rm -f "$OUT"
log "running pipeline on $SCENE_DIR ..."
"$PY" -m pipeline.run_pipeline "$SCENE_DIR" --out "$OUT" || fail "pipeline returned non-zero"
[ -f "$OUT" ]                             || fail "pipeline did not write $OUT (schema gate likely failed before write)"

# --- 4. independent re-validation of the written file -------------------------
log "re-validating $OUT against $SCHEMA ..."
N_PARTS="$("$PY" scripts/validate_pose_result.py "$OUT" "$SCHEMA")" \
  || fail "produced pose_result.json is NOT schema-valid"

# --- 5. PASS banner + viewer instructions -------------------------------------
echo
echo "  ============================================"
echo "  ✅  E2E PASS"
echo "  ============================================"
echo "    scene   : $SCENE_DIR"
echo "    output  : $OUT"
echo "    parts   : $N_PARTS  (schema-valid)"
echo
echo "  Open the 3D viewer on this result:"
echo "    python3 -m http.server $VIEWER_PORT"
echo "    -> http://127.0.0.1:$VIEWER_PORT/viewer/?file=../data/output/pose_result.json"
echo
