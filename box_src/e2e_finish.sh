#!/usr/bin/env bash
# =============================================================================
# e2e_finish.sh — ONE-COMMAND FINISH HARNESS for POSE  (T-048 / S-045)
# -----------------------------------------------------------------------------
# Given trained GDRNPP per-object checkpoints + the arm-visible detector, this
# drives the whole project finish, end to end, in five gated steps:
#
#   1. BOP-EVAL      symmetry-aware AR / trans-median / rot-median on the
#                    >20%-visibility val split, all objects, scored vs the §0
#                    in-house baseline (Anker_Kurz 0.59 · Anker_Lang 0.61 ·
#                    Zahnrad 0.36).                       [box_src/eval_bop.py]
#   2. REAL POSES    GDRNPP inference on a few val scenes -> real predictions ->
#                    the production bop_adapter -> schema-valid pose_result.json
#                    (+ detection overlay). rsync'd into project/temp/.
#                    Optionally planar-refined if the refine strand is wired.
#                                                    [box_src/real_pose_result.py]
#   3. VIEWER-VERIFY headless Playwright: the real pose_result renders the real
#                    CAD meshes in the 3D viewer with ZERO JS errors.
#                                          [project/frontend/test/viewer_*.py]
#   4. SPLIT-SCREEN  project/temp/final_2d_vs_3d.png  (2D top-down+arm+dets |
#                    3D real CAD at predicted poses).  [box_src/make_splitscreen.py]
#   5. REPORT        regenerate project/docs/RESULTS_PHASE2.md + patch the §4.2
#                    table in PROJECT_REPORT.md, with the Baseline->Phase-2
#                    comparison.                            [box_src/e2e_report.py]
#
# -----------------------------------------------------------------------------
# TWO MODES
#
#   --dry-run   Prove the harness end to end against ARTEFACTS THAT ALREADY
#               EXIST. No GPU inference is run: step 1 re-uses an existing
#               eval report.json, step 2 re-uses the existing pose_result /
#               det_overlay (or falls back to a fixture), steps 3-5 run for
#               real on the laptop. This is the gate the dry-run must pass.
#
#   (default)   FULL finish. Step 1 runs eval_bop on the box, step 2 runs real
#               GDRNPP inference on the box and rsyncs the artefacts back, then
#               steps 3-5 run on the laptop. Requires the box to be free (no
#               GPU contention) and the checkpoints to exist.
#
# -----------------------------------------------------------------------------
# USAGE
#   box_src/e2e_finish.sh --dry-run        # offline proof against existing art.
#   box_src/e2e_finish.sh                   # full GPU finish
#   box_src/e2e_finish.sh --scenes "0 1 2"  # full, choose which val scenes
#   box_src/e2e_finish.sh --planar-refine   # opt into the refine strand (full)
#   box_src/e2e_finish.sh --skip-viewer     # full but no Playwright (no browser)
#
# KEY PATHS (laptop)
#   PROJECT_DIR   project/                  (served at :PORT by step 3)
#   TEMP_DIR      project/temp/             (pose_result.json, overlays, split)
#   DOCS_DIR      project/docs/
#   EVAL_LOCAL    results/eval/             (report.json + report.txt pulled)
#
# ENV OVERRIDES
#   PY              python with playwright+PIL (default: python3)
#   PORT            local http port for the viewer (default: 8099)
#   REMOTE_REPO     /mnt/data/kip_pose
#   DATASET_DIR     /mnt/data/kip_pose/project/bop/pose_isaac
#   PREDS_CSV       /mnt/data/bop/results/preds_all.csv   (full-mode eval input)
#   DRY_REPORT      project/docs/eval_gdrnpp_val_filtered_T038.json (dry-run eval)
#   DRY_POSE        an existing pose_result.json to feed the viewer in --dry-run
# =============================================================================
set -uo pipefail

# ── resolve worktree root (this script lives in <root>/box_src) ──────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT" || { echo "FATAL: cannot cd $ROOT"; exit 1; }

# ── config ───────────────────────────────────────────────────────────────────
PY="${PY:-python3}"
PORT="${PORT:-8099}"
PROJECT_DIR="$ROOT/project"
TEMP_DIR="$PROJECT_DIR/temp"
DOCS_DIR="$PROJECT_DIR/docs"
EVAL_LOCAL="$ROOT/results/eval"
FRONTEND_TEST="$PROJECT_DIR/frontend/test"

# remote (box) paths — only used in full mode
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/kip_pose}"
DATASET_DIR="${DATASET_DIR:-/mnt/data/kip_pose/project/bop/pose_isaac}"
PREDS_CSV="${PREDS_CSV:-/mnt/data/bop/results/preds_all.csv}"
GDRN_OUT="${GDRN_OUT:-/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/mnt/data/bop/results/e2e_finish}"
GPU_RUN="$HERE/gpu_run.sh"
EVAL_SH="$HERE/eval_bop.sh"

# dry-run fallbacks (existing artefacts that prove the harness without the GPU)
DRY_REPORT="${DRY_REPORT:-$DOCS_DIR/eval_gdrnpp_val_filtered_T038.json}"
DRY_POSE="${DRY_POSE:-}"   # auto-detected below if empty

# defaults
DRY_RUN=0
PLANAR_REFINE=0
REFINE_RC=0
RC_SCORER="cpu_edge"
SKIP_VIEWER=0
SCENES="0 1"
PRIMARY_SCENE=0
PRIMARY_IM=92

# ── arg parse ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    --planar-refine)  PLANAR_REFINE=1; shift ;;
    --no-planar-refine) PLANAR_REFINE=0; shift ;;
    --refine-rc)      REFINE_RC=1; shift ;;
    --rc-scorer)      RC_SCORER="${2:-cpu_edge}"; shift 2 ;;
    --skip-viewer)    SKIP_VIEWER=1; shift ;;
    --scenes)         SCENES="${2:-}"; shift 2 ;;
    --scene)          PRIMARY_SCENE="${2:-0}"; shift 2 ;;
    --im)             PRIMARY_IM="${2:-92}"; shift 2 ;;
    -h|--help)        sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done

ts()   { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say()  { echo "[$(ts)] [e2e] $*"; }
step() { echo; echo "=============================================================="; \
         echo "[$(ts)] [e2e] STEP $*"; \
         echo "=============================================================="; }
fail() { say "FAILED: $1"; say "E2E_FINISH_FAILED"; exit "${2:-1}"; }

MODE="full"; [[ "$DRY_RUN" -eq 1 ]] && MODE="dry-run"
mkdir -p "$TEMP_DIR" "$DOCS_DIR" "$EVAL_LOCAL"

say "=== E2E FINISH HARNESS START (T-048) — mode=$MODE ==="
say "root=$ROOT  port=$PORT  planar_refine=$PLANAR_REFINE  scenes='$SCENES'"

# Detect the planar-refine helper from the parallel refine strand. It is wired
# but optional: pass the flag through ONLY if a refiner is actually present.
PLANAR_REFINE_AVAILABLE=0
REFINER=""
for cand in "$HERE/planar_refine.py" "$PROJECT_DIR/planar_refine.py"; do
  [[ -f "$cand" ]] && { PLANAR_REFINE_AVAILABLE=1; REFINER="$cand"; break; }
done
if [[ "$PLANAR_REFINE" -eq 1 && "$PLANAR_REFINE_AVAILABLE" -eq 0 ]]; then
  say "WARN --planar-refine requested but no refiner found (refine strand not landed yet) — continuing WITHOUT refine"
  PLANAR_REFINE=0
fi
[[ "$PLANAR_REFINE_AVAILABLE" -eq 1 ]] && say "planar refiner available: $REFINER"

# Detect the M2 render-and-compare refiner (T-058). Wired but optional: pass
# --refine-rc through ONLY if project/refine_rc.py is present.
REFINE_RC_AVAILABLE=0
RC_MODULE="$PROJECT_DIR/refine_rc.py"
[[ -f "$RC_MODULE" ]] && REFINE_RC_AVAILABLE=1
if [[ "$REFINE_RC" -eq 1 && "$REFINE_RC_AVAILABLE" -eq 0 ]]; then
  say "WARN --refine-rc requested but project/refine_rc.py not found — continuing WITHOUT RC"
  REFINE_RC=0
fi
[[ "$REFINE_RC_AVAILABLE" -eq 1 ]] && say "M2 RC refiner available: $RC_MODULE (scorer=$RC_SCORER)"
if [[ "$REFINE_RC" -eq 1 && "$RC_SCORER" == "megapose" ]]; then
  say "NOTE RC scorer=megapose is the GPU path — finish-time validation (GPU must be free)"
fi

# =============================================================================
# STEP 1 — BOP-EVAL  (symmetry-aware, >20% val, all objects, vs §0 baseline)
# =============================================================================
step "1/5: BOP-EVAL"
REPORT_JSON="$EVAL_LOCAL/report.json"
if [[ "$DRY_RUN" -eq 1 ]]; then
  say "dry-run: re-using existing eval report -> $DRY_REPORT"
  [[ -f "$DRY_REPORT" ]] || fail "dry-run eval report not found: $DRY_REPORT" 11
  cp "$DRY_REPORT" "$REPORT_JSON"
  say "eval report staged at $REPORT_JSON"
else
  say "full: running eval_bop on the box (preds=$PREDS_CSV, split=val, >20% dataset)"
  [[ -x "$EVAL_SH" ]] || fail "eval_bop.sh not executable: $EVAL_SH" 12
  OUT_LOCAL="$EVAL_LOCAL" SPLIT=val DATASET_DIR="$DATASET_DIR" \
    "$EVAL_SH" --preds "$PREDS_CSV" \
    || fail "eval_bop.sh failed (box eval)" 12
  [[ -f "$REPORT_JSON" ]] || fail "no report.json pulled to $REPORT_JSON" 12
fi
# surface the headline AR straight away
$PY - "$REPORT_JSON" <<'PYEOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
r = d.get("results", d); ov = r.get("overall", {})
po = r.get("per_object", {})
items = po.items() if isinstance(po, dict) else enumerate(po)
print(f"[e2e]   overall AR = {ov.get('AR')}")
for k, p in items:
    oid = p.get("obj_id", k)
    print(f"[e2e]   obj {oid} {p.get('name')}: AR={p.get('AR')} "
          f"trans_med={p.get('trans_err_median_mm')} rot_med={p.get('rot_err_median_deg')}")
PYEOF
say "STEP1_EVAL_DONE"

# =============================================================================
# STEP 2 — REAL POSES  (GDRNPP inference -> bop_adapter -> pose_result.json)
# =============================================================================
step "2/5: REAL POSES (pose_result.json + det overlay)"
POSE_JSON="$TEMP_DIR/pose_result.json"
DET_OVERLAY="$TEMP_DIR/det_overlay.png"
if [[ "$DRY_RUN" -eq 1 ]]; then
  # Prefer an existing real pose_result; else the populated main-repo temp; else
  # the committed viewer fixture. Whichever exists proves steps 3-5.
  if [[ -z "$DRY_POSE" ]]; then
    for cand in \
        "$TEMP_DIR/pose_result.json" \
        "/Users/Admin/POSE/project/temp/pose_result.json" \
        "$FRONTEND_TEST/fixtures/full.json"; do
      [[ -f "$cand" ]] && { DRY_POSE="$cand"; break; }
    done
  fi
  [[ -n "$DRY_POSE" && -f "$DRY_POSE" ]] || fail "dry-run: no pose_result/fixture to feed the viewer" 21
  say "dry-run: feeding viewer with existing pose -> $DRY_POSE"
  [[ "$DRY_POSE" != "$POSE_JSON" ]] && cp "$DRY_POSE" "$POSE_JSON"
  # det overlay: keep an existing one if present, else fall back to the main temp.
  if [[ ! -f "$DET_OVERLAY" ]]; then
    for cand in "/Users/Admin/POSE/project/temp/det_overlay.png"; do
      [[ -f "$cand" ]] && { cp "$cand" "$DET_OVERLAY"; break; }
    done
  fi
  [[ -f "$DET_OVERLAY" ]] && say "det overlay staged -> $DET_OVERLAY" \
                          || say "WARN no det_overlay.png (split-screen left panel will fall back)"
else
  say "full: GDRNPP inference + bop_adapter on the box for scenes='$SCENES'"
  REMOTE_POSE="$REMOTE_RESULTS/pose_result.json"
  REMOTE_OVERLAY="$REMOTE_RESULTS/det_overlay.png"
  REFINE_ARGS=""
  if [[ "$PLANAR_REFINE" -eq 1 ]]; then
    REFINE_ARGS="--planar-refine"
    say "planar refine ENABLED — passing --planar-refine to real_pose_result.py"
  fi
  if [[ "$REFINE_RC" -eq 1 ]]; then
    REFINE_ARGS="$REFINE_ARGS --refine-rc --rc-scorer $RC_SCORER"
    say "M2 RC refine ENABLED — passing --refine-rc --rc-scorer $RC_SCORER to real_pose_result.py"
  fi
  # ship the real-pose script + adapter + (optional) RC refiner to the box, run it
  # inside the gdrnpp/bop venv, and pull pose_result.json + det_overlay.png back.
  REMOTE_CMD="mkdir -p '$REMOTE_RESULTS'; \
    /mnt/data/bop/bop-venv/bin/python '$REMOTE_REPO/box_src/real_pose_result.py' \
      --dataset-dir '$DATASET_DIR' --split val \
      --scene $PRIMARY_SCENE --im $PRIMARY_IM \
      --preds '$PREDS_CSV' \
      --out-json '$REMOTE_POSE' --out-overlay '$REMOTE_OVERLAY' $REFINE_ARGS"
  say "  scp real_pose_result.py + bop_adapter.py + refine_rc.py -> box"
  scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "$HERE/real_pose_result.py" "max@100.85.216.95:$REMOTE_REPO/box_src/real_pose_result.py" \
      || fail "scp real_pose_result.py to box failed" 22
  scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "$PROJECT_DIR/bop_adapter.py" "max@100.85.216.95:$REMOTE_REPO/project/bop_adapter.py" \
      || fail "scp bop_adapter.py to box failed" 22
  if [[ "$REFINE_RC_AVAILABLE" -eq 1 ]]; then
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$RC_MODULE" "max@100.85.216.95:$REMOTE_REPO/project/refine_rc.py" \
        || fail "scp refine_rc.py to box failed" 22
  fi
  "$GPU_RUN" \
    -p "$REMOTE_POSE:$POSE_JSON" \
    -p "$REMOTE_OVERLAY:$DET_OVERLAY" \
    -- "$REMOTE_CMD" \
    || fail "GDRNPP inference / real_pose_result.py on box failed" 22
  [[ -f "$POSE_JSON" ]] || fail "no pose_result.json pulled to $POSE_JSON" 22
fi

# validate the pose_result against the frozen contract before the viewer sees it
SCHEMA="$PROJECT_DIR/pose_result.schema.json"
if [[ -f "$SCHEMA" ]]; then
  $PY - "$POSE_JSON" "$SCHEMA" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1]))
sch = json.load(open(sys.argv[2]))
# lightweight contract check (no jsonschema dep): required top-level + result keys
assert "meta" in doc and "results" in doc, "pose_result missing meta/results"
req = ["part", "face", "R_world", "t_world", "confidence"]
for i, r in enumerate(doc["results"]):
    miss = [k for k in req if k not in r]
    assert not miss, f"result {i} missing keys {miss}"
print(f"[e2e]   pose_result contract OK: {len(doc['results'])} results, "
      f"schema_version={doc['meta'].get('schema_version')}")
PYEOF
  [[ $? -ne 0 ]] && fail "pose_result.json failed the contract check" 23
fi
say "STEP2_REALPOSE_DONE  ($POSE_JSON)"

# =============================================================================
# STEP 3 — VIEWER-VERIFY  (headless Playwright, real CAD, zero JS errors)
# =============================================================================
VIEWER_SHOT="$TEMP_DIR/viewer_real.png"
if [[ "$SKIP_VIEWER" -eq 1 ]]; then
  step "3/5: VIEWER-VERIFY — SKIPPED (--skip-viewer)"
  [[ -f "$VIEWER_SHOT" ]] || cp "/Users/Admin/POSE/project/temp/viewer_real.png" "$VIEWER_SHOT" 2>/dev/null || true
else
  step "3/5: VIEWER-VERIFY (headless Playwright)"
  # serve project/ so the viewer is /frontend/ and the pose is /temp/ (its default).
  say "starting static server on :$PORT (cwd=$PROJECT_DIR)"
  ( cd "$PROJECT_DIR" && exec "$PY" -m http.server "$PORT" >/dev/null 2>&1 ) &
  SRV_PID=$!
  cleanup_srv() { kill "$SRV_PID" >/dev/null 2>&1 || true; }
  trap cleanup_srv EXIT
  # wait for the server to accept connections
  for i in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$PORT/frontend/index.html" >/dev/null 2>&1; then break; fi
    sleep 0.3
  done
  # default ?file resolves to ../temp/pose_result.json — exactly what we wrote.
  say "running viewer_headless.py (file=../temp/pose_result.json)"
  POSE_PORT="$PORT" POSE_FILE="../temp/pose_result.json" POSE_SHOT="$VIEWER_SHOT" \
    "$PY" "$FRONTEND_TEST/viewer_headless.py" "$PORT"
  VRC=$?
  cleanup_srv; trap - EXIT
  [[ $VRC -eq 0 ]] || fail "viewer headless check FAILED (JS errors / blank canvas / 404s)" 30
  [[ -f "$VIEWER_SHOT" ]] || fail "viewer screenshot not produced: $VIEWER_SHOT" 30
fi
say "STEP3_VIEWER_DONE  ($VIEWER_SHOT)"

# =============================================================================
# STEP 4 — SPLIT-SCREEN  (2D top-down+arm+dets | 3D real CAD at predicted poses)
# =============================================================================
step "4/5: SPLIT-SCREEN"
SPLIT_OUT="$TEMP_DIR/final_2d_vs_3d.png"
LEFT="$DET_OVERLAY"
[[ -f "$LEFT" ]] || LEFT="/Users/Admin/POSE/project/temp/det_overlay.png"
[[ -f "$LEFT" ]] || fail "no left-panel image (det_overlay.png) for the split-screen" 40
[[ -f "$VIEWER_SHOT" ]] || fail "no right-panel image (viewer_real.png) for the split-screen" 40
"$PY" "$HERE/make_splitscreen.py" --left "$LEFT" --right "$VIEWER_SHOT" --out "$SPLIT_OUT" \
  || fail "make_splitscreen.py failed" 41
[[ -f "$SPLIT_OUT" ]] || fail "split-screen not written: $SPLIT_OUT" 41
say "STEP4_SPLITSCREEN_DONE  ($SPLIT_OUT)"

# =============================================================================
# STEP 5 — REPORT  (RESULTS_PHASE2.md + patch PROJECT_REPORT §4.2, vs baseline)
# =============================================================================
step "5/5: REPORT (RESULTS_PHASE2.md + PROJECT_REPORT §4.2)"
PREDS_NOTE="preds_all.csv (val, gt-bbox)"
[[ "$DRY_RUN" -eq 1 ]] && PREDS_NOTE="$(basename "$DRY_REPORT") (existing >20% eval, dry-run)"
REFINE_FLAG=""; [[ "$PLANAR_REFINE" -eq 1 ]] && REFINE_FLAG="--planar-refine"
"$PY" "$HERE/e2e_report.py" \
  --report "$REPORT_JSON" \
  --report-md "$DOCS_DIR/PROJECT_REPORT.md" \
  --out-md "$DOCS_DIR/RESULTS_PHASE2.md" \
  --mode "$MODE" \
  --preds-note "$PREDS_NOTE" \
  $REFINE_FLAG \
  || fail "e2e_report.py failed" 50
say "STEP5_REPORT_DONE"

# =============================================================================
# DONE
# =============================================================================
echo
say "=============================================================="
say "E2E FINISH OK (mode=$MODE)"
say "  eval report     : $REPORT_JSON"
say "  pose_result     : $POSE_JSON"
say "  viewer shot     : $VIEWER_SHOT"
say "  split-screen    : $SPLIT_OUT"
say "  results doc     : $DOCS_DIR/RESULTS_PHASE2.md"
say "  project report  : $DOCS_DIR/PROJECT_REPORT.md (§4.2 patched)"
say "E2E_FINISH_DONE"
