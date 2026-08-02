#!/usr/bin/env bash
# =============================================================================
# e2e_finish.sh — ONE-COMMAND FINISH HARNESS for POSE  (T-048 / S-045)
# -----------------------------------------------------------------------------
# Given trained GDRNPP per-object checkpoints + the arm-visible detector, this
# drives the whole project finish, end to end, in SIX gated steps. The improved
# pipeline's three levers (planar Z-snap, M2 render-and-compare, TTA) are A/B'd
# against each other and the WINNING config is the one materialised + reported:
#
#   1. BOP-EVAL      symmetry-aware AR / trans-median / rot-median on the
#                    >20%-visibility val split, all objects, scored vs the §0
#                    in-house baseline (Anker_Kurz 0.59 · Anker_Lang 0.61 ·
#                    Zahnrad 0.36).                       [box_src/eval_bop.py]
#   2. A/B + AUTO-BEST  four configs — {planar-only} {+M2} {+TTA} {+M2+TTA} —
#                    each producing one refined predictions CSV scored by
#                    eval_bop.py; the config with the highest mean AR over the
#                    trainable objects is auto-selected; decision flags emitted
#                    (Zahnrad<0.70 -> train SO(3) head; Anker<0.80 -> check
#                    MegaPose).                               [box_src/e2e_ab.py
#                    + box_src/refine_eval.py (planar) + box_src/rc_refine_eval.py
#                    (M2) + project/tta_pose.py (TTA)]
#   3. REAL POSES    GDRNPP inference (best config's levers) on a val frame ->
#                    real predictions -> the production bop_adapter -> schema-
#                    valid pose_result.json (+ detection overlay).
#                                                    [box_src/real_pose_result.py]
#   4. VIEWER-VERIFY headless Playwright: the real pose_result renders the real
#                    CAD meshes in the 3D viewer with ZERO JS errors.
#                                          [project/frontend/test/viewer_*.py]
#   5. SPLIT-SCREEN  project/temp/final_2d_vs_3d.png  (2D top-down+arm+dets |
#                    3D real CAD at predicted poses).  [box_src/make_splitscreen.py]
#   6. REPORT        regenerate project/docs/RESULTS_PHASE2.md + AB_LEVERS.md +
#                    patch the §4.2 table in PROJECT_REPORT.md, best-config vs
#                    §0 + §Phase-1 baselines.                [box_src/e2e_report.py]
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
#   box_src/e2e_finish.sh --dry-run         # offline proof against existing art.
#                                           # (A/B configs synthesised, auto-best
#                                           #  + decision flags run for real)
#   box_src/e2e_finish.sh                    # full GPU finish + A/B + auto-best
#   box_src/e2e_finish.sh --scenes "0 1 2"   # full, choose which val scenes
#   box_src/e2e_finish.sh --no-auto-best \
#       --planar-refine --refine-rc --tta    # full, force a specific config
#   box_src/e2e_finish.sh --rc-scorer megapose   # full, M2 with the GPU scorer
#   box_src/e2e_finish.sh --skip-viewer      # full but no Playwright (no browser)
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
#   BOX             ssh target of the GPU box, user@host (full mode; see
#                   project/.env.example — falls back to $GPU_HOST)
#   MAIN_REPO       primary repo checkout used as artifact fallback (default: ROOT)
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

# remote (box) target — only used in full mode. Set BOX (user@host) in your
# environment or project/.env; see project/.env.example.
BOX="${BOX:-${GPU_HOST:-}}"
# gpu_run.sh takes the target split in two; derive it from BOX so callers only
# have to set one variable. Validated in full mode only (dry-run stays offline).
if [ -n "$BOX" ]; then
    export BOX_USER="${BOX_USER:-${BOX%@*}}"
    export BOX_HOST="${BOX_HOST:-${BOX#*@}}"
fi
REMOTE_REPO="${REMOTE_REPO:-/mnt/data/kip_pose}"
DATASET_DIR="${DATASET_DIR:-/mnt/data/kip_pose/project/bop/pose_isaac}"

# Primary repo checkout. When this script runs from a git worktree, artifacts
# from the main checkout are used as a fallback; override with MAIN_REPO.
MAIN_REPO="${MAIN_REPO:-$ROOT}"

# [Box-local autonomy] If ROOT==REMOTE_REPO we are running ON the box itself
# (typically launched by box_src/finish_watcher.sh after PHASE2_TRAIN_DONE, with
# no Mac-side ssh available). All the "scp helper -> box" calls below would then
# be scp-to-self with identical source/destination paths — which would either
# no-op or trip "|| fail". Stub scp() to a no-op so the file-shipping blocks
# below succeed cleanly; the helpers (refine_eval.py, rc_refine_eval.py,
# refine_rc.py, tta_pose.py, bop_adapter.py, real_pose_result.py) are expected
# to already be in place in $REMOTE_REPO. Off-box (Mac) runs are unaffected
# because ROOT (the local checkout) won't match REMOTE_REPO.
if [ "$ROOT" = "$REMOTE_REPO" ]; then
    echo "[e2e] DETECTED on-box mode (ROOT=REMOTE_REPO=$ROOT) — scp stubbed (helpers pre-deployed)"
    scp() { return 0; }
fi
# [T-068] track whether the caller pinned PREDS_CSV (then we don't rebuild it from
# the best/final per-object CSVs in STEP 1).
PREDS_CSV_PINNED="${PREDS_CSV:+1}"
PREDS_CSV="${PREDS_CSV:-/mnt/data/bop/results/preds_all.csv}"
GDRN_OUT="${GDRN_OUT:-/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/mnt/data/bop/results/e2e_finish}"
# the per-object trained-GDRNPP val-prediction CSVs (verified box layout, S-050).
# refine_eval.py takes name=csv; these feed the planar A/B config. Overridable.
# [T-068 — OVERFITTING GUARD] Prefer the BEST-by-val checkpoint's predictions
# ($GDRN_OUT/<obj>/preds_best.csv, staged by select_best_ckpt.py) over the
# model_final ones. The model_final CSV (inference_/.../<obj>-iter0...val.csv) is
# the LAST epoch, a possible overfit; preds_best.csv is the AR-best checkpoint on
# the held-out val. Fall back to the model_final CSV when no best-selection ran.
# NB: GDRN_OUT lives on the BOX; e2e_finish runs on the LAPTOP in full mode, so the
# preds_best-vs-model_final choice is made BOX-SIDE in STEP 1's combine command
# (the `_PRED_*` defaults below are the model_final paths; the box picks best if
# present). The PRED_* values stay model_final paths for the A/B planar config.
PRED_KURZ="${PRED_KURZ:-$GDRN_OUT/anker_kurz/inference_/pose_isaac_anker_kurz_val/anker-kurz-iter0_pose_isaac-val.csv}"
PRED_LANG="${PRED_LANG:-$GDRN_OUT/anker_lang/inference_/pose_isaac_anker_lang_val/anker-lang-iter0_pose_isaac-val.csv}"
PRED_ZAHN="${PRED_ZAHN:-$GDRN_OUT/zahnrad/inference_/pose_isaac_zahnrad_val/zahnrad-iter0_pose_isaac-val.csv}"
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
TTA=0
SKIP_VIEWER=0
AUTO_BEST=1            # let the A/B step pick the winning config (default ON)
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
    --tta)            TTA=1; shift ;;
    --no-auto-best)   AUTO_BEST=0; shift ;;
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

# Full mode ships files to the box and calls gpu_run.sh — abort early with a
# clear message instead of failing halfway through on an empty scp target.
# On-box runs (ROOT==REMOTE_REPO) never leave the machine and need no BOX.
if [[ "$DRY_RUN" -eq 0 && -z "$BOX" && "$ROOT" != "$REMOTE_REPO" ]]; then
    fail "full mode needs the GPU box: set BOX (or GPU_HOST) to user@host — see project/.env.example" 2
fi

mkdir -p "$TEMP_DIR" "$DOCS_DIR" "$EVAL_LOCAL"

say "=== E2E FINISH HARNESS START (T-048) — mode=$MODE ==="
say "root=$ROOT  port=$PORT  planar_refine=$PLANAR_REFINE  scenes='$SCENES'"

# ── LEVER WIRING (the real ones — NOT the never-landed box_src/planar_refine.py) ──
# The improved pipeline's three levers, each a real module already in the tree:
#   • Planar  : project/bop_adapter.planar_refine (M1 stable-pose + Z-snap). The
#               A/B "planar-only" CSV is produced by box_src/refine_eval.py
#               --config m1_zsnap (the §Phase-1 config: Anker 0.645/0.650).
#   • M2       : project/refine_rc.refine_detection (render-and-compare). The A/B
#               "+M2" CSV is produced by box_src/rc_refine_eval.py on top of the
#               planar CSV (scorer cpu_edge laptop / megapose = finish-time GPU).
#   • TTA      : project/tta_pose.tta_call_gdrnpp — a rotation test-time-aug vote;
#               applied at GDRNPP inference (real_pose_result.py --tta), so the
#               "+TTA" CSV is an inference-time variant produced on the box.
# All three are detected here; the A/B step (STEP 2) wires whichever are present.
PLANAR_LEVER="$HERE/refine_eval.py"          # planar (training-free) lever -> CSV
M2_LEVER="$HERE/rc_refine_eval.py"           # M2 render-and-compare lever  -> CSV
RC_MODULE="$PROJECT_DIR/refine_rc.py"        # the M2 engine (imported by both)
TTA_MODULE="$PROJECT_DIR/tta_pose.py"        # TTA engine (inference-time)
ADAPTER="$PROJECT_DIR/bop_adapter.py"        # the planar engine (planar_refine)
AB_PY="$HERE/e2e_ab.py"                       # A/B + auto-best + decision flags

PLANAR_AVAILABLE=0;  [[ -f "$PLANAR_LEVER" && -f "$ADAPTER" ]] && PLANAR_AVAILABLE=1
M2_AVAILABLE=0;      [[ -f "$M2_LEVER" && -f "$RC_MODULE" ]] && M2_AVAILABLE=1
TTA_AVAILABLE=0;     [[ -f "$TTA_MODULE" ]] && TTA_AVAILABLE=1
[[ -f "$AB_PY" ]] || fail "A/B orchestrator missing: $AB_PY" 10

say "levers: planar=$PLANAR_AVAILABLE(refine_eval+bop_adapter)  m2=$M2_AVAILABLE(rc_refine_eval+refine_rc)  tta=$TTA_AVAILABLE(tta_pose)"
[[ "$PLANAR_REFINE" -eq 1 && "$PLANAR_AVAILABLE" -eq 0 ]] && { say "WARN planar lever unavailable"; PLANAR_REFINE=0; }
[[ "$REFINE_RC" -eq 1 && "$M2_AVAILABLE" -eq 0 ]] && { say "WARN M2 lever unavailable"; REFINE_RC=0; }
if [[ "$RC_SCORER" == "megapose" ]]; then
  say "NOTE RC scorer=megapose is the GPU path — measured to ceiling Phase-2 (see RESULTS_PHASE2.md); cpu_edge is the operational default"
fi

# =============================================================================
# STEP 1 — BOP-EVAL  (symmetry-aware, >20% val, all objects, vs §0 baseline)
# =============================================================================
step "1/6: BOP-EVAL (baseline GDRNPP predictions)"
REPORT_JSON="$EVAL_LOCAL/report.json"
if [[ "$DRY_RUN" -eq 1 ]]; then
  say "dry-run: re-using existing eval report -> $DRY_REPORT"
  [[ -f "$DRY_REPORT" ]] || fail "dry-run eval report not found: $DRY_REPORT" 11
  cp "$DRY_REPORT" "$REPORT_JSON"
  say "eval report staged at $REPORT_JSON"
else
  # [T-068] Build the combined preds CSV from the per-object PRED_* paths, which
  # now point at preds_best.csv (best-by-val checkpoint) when a best-selection ran,
  # falling back to the model_final CSV otherwise. Only do this if the caller did
  # NOT pin PREDS_CSV explicitly (env override wins). This is the step that makes
  # the WHOLE finish honour the overfitting-guarded checkpoint, not model_final.
  if [[ -z "${PREDS_CSV_PINNED:-}" ]]; then
    say "full: combining per-object best-by-val/final CSVs -> $PREDS_CSV on the box"
    say "  (box picks \$GDRN_OUT/<obj>/preds_best.csv if present, else the model_final CSV)"
    # The choice happens BOX-SIDE: for each object use preds_best.csv (the best-by-val
    # checkpoint staged by select_best_ckpt.py) when it exists, otherwise the
    # model_final CSV. This keeps e2e_finish correct whether it runs on the laptop
    # (GDRN_OUT is a remote path the laptop can't stat) or on the box.
    COMBINE_CMD="set -e; mkdir -p '$(dirname "$PREDS_CSV")'; \
      pick() { if [ -f \"$GDRN_OUT/\$1/preds_best.csv\" ]; then echo \"$GDRN_OUT/\$1/preds_best.csv\"; else echo \"\$2\"; fi; }; \
      CK=\$(pick anker_kurz '$PRED_KURZ'); \
      CL=\$(pick anker_lang '$PRED_LANG'); \
      CZ=\$(pick zahnrad    '$PRED_ZAHN'); \
      echo \"[combine] kurz=\$CK\"; echo \"[combine] lang=\$CL\"; echo \"[combine] zahn=\$CZ\"; \
      /mnt/data/bop/bop-venv/bin/python '$REMOTE_REPO/box_src/combine_preds.py' \
        '$PREDS_CSV' \"\$CK\" \"\$CL\" \"\$CZ\""
    "$GPU_RUN" -- "$COMBINE_CMD" \
      || say "WARN combine_preds on box failed — falling back to pre-existing $PREDS_CSV"
  else
    say "full: PREDS_CSV pinned by caller -> $PREDS_CSV (skipping best/final combine)"
  fi
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
# STEP 2 — A/B OVER THE LEVERS  (planar | +M2 | +TTA | +M2+TTA -> auto-best)
# -----------------------------------------------------------------------------
# Each config materialises one refined predictions CSV, each CSV is scored by
# eval_bop.py into one per-config report.json, and e2e_ab.py picks the config
# with the highest mean AR over the trainable objects + emits the decision flags.
#
#   DRY-RUN : only the *planar* report is real (re-used from STEP 1). The other
#             three configs are SYNTHESISED by e2e_ab.py (--synth) so the
#             auto-best + decision logic runs offline. Proof of orchestration.
#   FULL    : every config is scored for real on the box (GPU for TTA inference
#             + megapose scorer), reports pulled back, then auto-best here.
# =============================================================================
step "2/6: A/B OVER THE LEVERS (auto-best + decision flags)"
AB_DIR="$EVAL_LOCAL/ab"
AB_RESULT="$AB_DIR/ab_result.json"
AB_MD="$DOCS_DIR/AB_LEVERS.md"
mkdir -p "$AB_DIR"

# the four configs and how each is produced (full mode). All paths are on the
# box; only the planar config can be produced without the GPU/inference variants.
declare -a AB_REPORT_ARGS=()
if [[ "$DRY_RUN" -eq 1 ]]; then
  # planar = the real STEP-1 report; the rest are synthesised by e2e_ab --synth.
  cp "$REPORT_JSON" "$AB_DIR/planar.json"
  AB_REPORT_ARGS+=(--report "planar=$AB_DIR/planar.json" --synth)
  say "dry-run: planar=real STEP-1 report; m2/tta/m2_tta synthesised (e2e_ab --synth)"
else
  # FULL: produce the four refined CSVs on the box, score each, pull reports.
  # 1) planar-only  : refine_eval.py --config m1_zsnap (§Phase-1 planar)
  # 2) +M2          : rc_refine_eval.py on the planar CSV (scorer=$RC_SCORER)
  # 3) +TTA         : GDRNPP inference with --tta -> planar refine of that CSV
  # 4) +M2+TTA      : rc_refine_eval.py on the +TTA CSV
  REMOTE_AB="$REMOTE_RESULTS/ab"
  PRED_PLANAR="$REMOTE_AB/preds_m1_zsnap.csv"
  PRED_M2="$REMOTE_AB/preds_rc_all.csv"
  PRED_TTA="$REMOTE_AB/tta/preds_m1_zsnap.csv"
  PRED_M2TTA="$REMOTE_AB/tta_rc/preds_rc_all.csv"
  RC_FLAG=""; [[ "$M2_AVAILABLE" -eq 1 ]] && RC_FLAG="1"
  TTA_FLAG=""; [[ "$TTA_AVAILABLE" -eq 1 ]] && TTA_FLAG="1"

  say "full: shipping levers (refine_eval.py, rc_refine_eval.py, refine_rc.py, tta_pose.py, bop_adapter.py) -> box"
  for f in "$PLANAR_LEVER" "$M2_LEVER"; do
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$f" "$BOX:$REMOTE_REPO/box_src/$(basename "$f")" \
        || fail "scp $(basename "$f") to box failed" 13
  done
  for f in "$RC_MODULE" "$TTA_MODULE" "$ADAPTER"; do
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$f" "$BOX:$REMOTE_REPO/project/$(basename "$f")" \
        || fail "scp $(basename "$f") to box failed" 13
  done

  # The remote A/B driver: build the 4 CSVs, score each with eval_bop.py.
  # NOTE the GPU touches: TTA needs a fresh GDRNPP inference pass with the
  # rotation augmentation (real_pose_result.py --tta over the val split); the
  # megapose scorer (RC_SCORER=megapose) is the GPU render-and-compare. Both are
  # finish-time — they run here in FULL mode against the trained checkpoints.
  AB_REMOTE_CMD="set -e; mkdir -p '$REMOTE_AB' '$REMOTE_AB/tta' '$REMOTE_AB/tta_rc'; \
    PY=/mnt/data/bop/bop-venv/bin/python; \
    cd '$REMOTE_REPO'; \
    echo '[ab] (1) planar-only via refine_eval.py m1_zsnap'; \
    \$PY box_src/refine_eval.py --bop-root '$DATASET_DIR' \
      --preds 'anker_kurz=$PRED_KURZ' \
              'anker_lang=$PRED_LANG' \
              'zahnrad=$PRED_ZAHN' \
      --out-dir '$REMOTE_AB' --config m1_zsnap; \
    \$PY box_src/eval_bop.py --dataset-dir '$DATASET_DIR' --split val \
      --preds '$PRED_PLANAR' --out '$REMOTE_AB/eval_planar'; \
    if [ -n '$RC_FLAG' ]; then \
      echo '[ab] (2) +M2 via rc_refine_eval.py rc_all (scorer=$RC_SCORER)'; \
      \$PY box_src/rc_refine_eval.py --bop-root '$DATASET_DIR' \
        --preds '$PRED_PLANAR' --out-dir '$REMOTE_AB' --config rc_all; \
      \$PY box_src/eval_bop.py --dataset-dir '$DATASET_DIR' --split val \
        --preds '$PRED_M2' --out '$REMOTE_AB/eval_m2'; \
    fi; \
    if [ -n '$TTA_FLAG' ]; then \
      echo '[ab] (3) +TTA: GDRNPP inference --tta over val (GPU), then planar refine'; \
      \$PY box_src/real_pose_result.py --dataset-dir '$DATASET_DIR' --split val \
        --tta --emit-preds '$REMOTE_AB/tta/preds_raw.csv' --val-all || true; \
      \$PY box_src/refine_eval.py --bop-root '$DATASET_DIR' \
        --preds anker_kurz=$REMOTE_AB/tta/preds_raw.csv \
                anker_lang=$REMOTE_AB/tta/preds_raw.csv \
                zahnrad=$REMOTE_AB/tta/preds_raw.csv \
        --out-dir '$REMOTE_AB/tta' --config m1_zsnap || true; \
      [ -f '$PRED_TTA' ] && \$PY box_src/eval_bop.py --dataset-dir '$DATASET_DIR' \
        --split val --preds '$PRED_TTA' --out '$REMOTE_AB/eval_tta' || true; \
      if [ -n '$RC_FLAG' ] && [ -f '$PRED_TTA' ]; then \
        echo '[ab] (4) +M2+TTA: rc_refine_eval.py on the +TTA CSV'; \
        \$PY box_src/rc_refine_eval.py --bop-root '$DATASET_DIR' \
          --preds '$PRED_TTA' --out-dir '$REMOTE_AB/tta_rc' --config rc_all || true; \
        [ -f '$PRED_M2TTA' ] && \$PY box_src/eval_bop.py --dataset-dir '$DATASET_DIR' \
          --split val --preds '$PRED_M2TTA' --out '$REMOTE_AB/eval_m2_tta' || true; \
      fi; \
    fi"
  "$GPU_RUN" -p "$REMOTE_AB:$AB_DIR/remote" -- "$AB_REMOTE_CMD" \
    || fail "A/B lever scoring on box failed" 13

  # collect whichever per-config reports came back
  for cfg in planar m2 tta m2_tta; do
    rp="$AB_DIR/remote/eval_$cfg/report.json"
    [[ -f "$rp" ]] && AB_REPORT_ARGS+=(--report "$cfg=$rp")
  done
  [[ "${#AB_REPORT_ARGS[@]}" -gt 0 ]] || fail "no per-config reports pulled for A/B" 13
fi

if [[ "$AUTO_BEST" -eq 1 ]]; then
  "$PY" "$AB_PY" "${AB_REPORT_ARGS[@]}" \
    --mode "$MODE" --out-json "$AB_RESULT" --out-md "$AB_MD" \
    || fail "e2e_ab.py (A/B auto-best) failed" 14
  BEST_CONFIG="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_config"])' "$AB_RESULT")"
  say "AUTO-BEST config = $BEST_CONFIG  (see $AB_MD)"
else
  # auto-best off -> honour the explicit flags as the chosen config
  BEST_CONFIG="planar"
  [[ "$REFINE_RC" -eq 1 && "$TTA" -eq 1 ]] && BEST_CONFIG="m2_tta"
  [[ "$REFINE_RC" -eq 1 && "$TTA" -eq 0 ]] && BEST_CONFIG="m2"
  [[ "$REFINE_RC" -eq 0 && "$TTA" -eq 1 ]] && BEST_CONFIG="tta"
  say "auto-best OFF -> chosen config from flags = $BEST_CONFIG"
fi
# translate the winning config into the real-pose lever flags (STEP 3)
case "$BEST_CONFIG" in
  planar) PLANAR_REFINE=1; REFINE_RC=0; TTA=0 ;;
  m2)     PLANAR_REFINE=1; REFINE_RC=1; TTA=0 ;;
  tta)    PLANAR_REFINE=1; REFINE_RC=0; TTA=1 ;;
  m2_tta) PLANAR_REFINE=1; REFINE_RC=1; TTA=1 ;;
esac
[[ "$PLANAR_AVAILABLE" -eq 0 ]] && PLANAR_REFINE=0
[[ "$M2_AVAILABLE" -eq 0 ]] && REFINE_RC=0
[[ "$TTA_AVAILABLE" -eq 0 ]] && TTA=0
say "STEP2_AB_DONE  (best=$BEST_CONFIG -> planar=$PLANAR_REFINE m2=$REFINE_RC tta=$TTA)"

# =============================================================================
# STEP 3 — REAL POSES  (best config: GDRNPP inference -> bop_adapter -> pose_result)
# =============================================================================
step "3/6: REAL POSES from best config '$BEST_CONFIG' (pose_result.json + det overlay)"
POSE_JSON="$TEMP_DIR/pose_result.json"
DET_OVERLAY="$TEMP_DIR/det_overlay.png"
if [[ "$DRY_RUN" -eq 1 ]]; then
  # Prefer an existing real pose_result; else the populated main-repo temp; else
  # the committed viewer fixture. Whichever exists proves steps 3-5.
  if [[ -z "$DRY_POSE" ]]; then
    for cand in \
        "$TEMP_DIR/pose_result.json" \
        "$MAIN_REPO/project/temp/pose_result.json" \
        "$FRONTEND_TEST/fixtures/full.json"; do
      [[ -f "$cand" ]] && { DRY_POSE="$cand"; break; }
    done
  fi
  [[ -n "$DRY_POSE" && -f "$DRY_POSE" ]] || fail "dry-run: no pose_result/fixture to feed the viewer" 21
  say "dry-run: feeding viewer with existing pose -> $DRY_POSE"
  [[ "$DRY_POSE" != "$POSE_JSON" ]] && cp "$DRY_POSE" "$POSE_JSON"
  # det overlay: keep an existing one if present, else fall back to the main temp.
  if [[ ! -f "$DET_OVERLAY" ]]; then
    for cand in "$MAIN_REPO/project/temp/det_overlay.png"; do
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
  if [[ "$TTA" -eq 1 ]]; then
    REFINE_ARGS="$REFINE_ARGS --tta"
    say "TTA ENABLED — passing --tta to real_pose_result.py (rotation test-time-aug at inference, GPU)"
  fi
  # ship the real-pose script + adapter + (optional) RC + TTA modules to the box,
  # run it inside the gdrnpp/bop venv, pull pose_result.json + det_overlay.png back.
  REMOTE_CMD="mkdir -p '$REMOTE_RESULTS'; \
    /mnt/data/bop/bop-venv/bin/python '$REMOTE_REPO/box_src/real_pose_result.py' \
      --dataset-dir '$DATASET_DIR' --split val \
      --scene $PRIMARY_SCENE --im $PRIMARY_IM \
      --preds '$PREDS_CSV' \
      --out-json '$REMOTE_POSE' --out-overlay '$REMOTE_OVERLAY' $REFINE_ARGS"
  say "  scp real_pose_result.py + bop_adapter.py + refine_rc.py + tta_pose.py -> box"
  scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "$HERE/real_pose_result.py" "$BOX:$REMOTE_REPO/box_src/real_pose_result.py" \
      || fail "scp real_pose_result.py to box failed" 22
  scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
      "$PROJECT_DIR/bop_adapter.py" "$BOX:$REMOTE_REPO/project/bop_adapter.py" \
      || fail "scp bop_adapter.py to box failed" 22
  if [[ "$M2_AVAILABLE" -eq 1 ]]; then
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$RC_MODULE" "$BOX:$REMOTE_REPO/project/refine_rc.py" \
        || fail "scp refine_rc.py to box failed" 22
  fi
  if [[ "$TTA_AVAILABLE" -eq 1 ]]; then
    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
        "$TTA_MODULE" "$BOX:$REMOTE_REPO/project/tta_pose.py" \
        || fail "scp tta_pose.py to box failed" 22
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
say "STEP3_REALPOSE_DONE  ($POSE_JSON)"

# =============================================================================
# STEP 4 — VIEWER-VERIFY  (headless Playwright, real CAD, zero JS errors)
# =============================================================================
VIEWER_SHOT="$TEMP_DIR/viewer_real.png"
if [[ "$SKIP_VIEWER" -eq 1 ]]; then
  step "4/6: VIEWER-VERIFY — SKIPPED (--skip-viewer)"
  [[ -f "$VIEWER_SHOT" ]] || cp "$MAIN_REPO/project/temp/viewer_real.png" "$VIEWER_SHOT" 2>/dev/null || true
else
  step "4/6: VIEWER-VERIFY (headless Playwright)"
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
say "STEP4_VIEWER_DONE  ($VIEWER_SHOT)"

# =============================================================================
# STEP 5 — SPLIT-SCREEN  (2D top-down+arm+dets | 3D real CAD at predicted poses)
# =============================================================================
step "5/6: SPLIT-SCREEN"
SPLIT_OUT="$TEMP_DIR/final_2d_vs_3d.png"
LEFT="$DET_OVERLAY"
[[ -f "$LEFT" ]] || LEFT="$MAIN_REPO/project/temp/det_overlay.png"
[[ -f "$LEFT" ]] || fail "no left-panel image (det_overlay.png) for the split-screen" 40
[[ -f "$VIEWER_SHOT" ]] || fail "no right-panel image (viewer_real.png) for the split-screen" 40
"$PY" "$HERE/make_splitscreen.py" --left "$LEFT" --right "$VIEWER_SHOT" --out "$SPLIT_OUT" \
  || fail "make_splitscreen.py failed" 41
[[ -f "$SPLIT_OUT" ]] || fail "split-screen not written: $SPLIT_OUT" 41
say "STEP5_SPLITSCREEN_DONE  ($SPLIT_OUT)"

# =============================================================================
# STEP 6 — REPORT  (RESULTS_PHASE2.md + PROJECT_REPORT §4.2, best config vs base)
# -----------------------------------------------------------------------------
# The report reflects the AUTO-BEST config. In FULL mode that is the best config's
# own eval report (the levers actually applied); in dry-run the planar report
# stands in (the synthesised configs have no scored CSV). The A/B result JSON is
# threaded through so the doc carries the winning config + decision flags.
# =============================================================================
step "6/6: REPORT (RESULTS_PHASE2.md + AB_LEVERS.md + PROJECT_REPORT §4.2)"
# pick the report that matches the winning config (full mode); fall back to the
# baseline STEP-1 report (dry-run, or if the best-config report wasn't pulled).
BEST_REPORT="$REPORT_JSON"
if [[ "$DRY_RUN" -eq 0 ]]; then
  cand="$AB_DIR/remote/eval_$BEST_CONFIG/report.json"
  [[ -f "$cand" ]] && BEST_REPORT="$cand"
fi
PREDS_NOTE="best config '$BEST_CONFIG' (val, gt-bbox, >20%)"
[[ "$DRY_RUN" -eq 1 ]] && PREDS_NOTE="$(basename "$DRY_REPORT") + synth A/B (dry-run, best=$BEST_CONFIG)"
REFINE_FLAG=""; [[ "$PLANAR_REFINE" -eq 1 ]] && REFINE_FLAG="--planar-refine"
AB_FLAG=""; [[ -f "$AB_RESULT" ]] && AB_FLAG="--ab-result $AB_RESULT"
"$PY" "$HERE/e2e_report.py" \
  --report "$BEST_REPORT" \
  --report-md "$DOCS_DIR/PROJECT_REPORT.md" \
  --out-md "$DOCS_DIR/RESULTS_PHASE2.md" \
  --mode "$MODE" \
  --preds-note "$PREDS_NOTE" \
  --best-config "$BEST_CONFIG" \
  $AB_FLAG \
  $REFINE_FLAG \
  || fail "e2e_report.py failed" 50
say "STEP6_REPORT_DONE"

# =============================================================================
# DONE
# =============================================================================
echo
say "=============================================================="
say "E2E FINISH OK (mode=$MODE)"
say "  baseline eval   : $REPORT_JSON"
say "  A/B result      : $AB_RESULT  (best=$BEST_CONFIG)"
say "  A/B doc         : $AB_MD"
say "  pose_result     : $POSE_JSON"
say "  viewer shot     : $VIEWER_SHOT"
say "  split-screen    : $SPLIT_OUT"
say "  results doc     : $DOCS_DIR/RESULTS_PHASE2.md"
say "  project report  : $DOCS_DIR/PROJECT_REPORT.md (§4.2 patched)"
# surface the decision flags one more time at the very end
if [[ -f "$AB_RESULT" ]]; then
  $PY - "$AB_RESULT" <<'PYEOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
dec = d.get("decision", {})
flags = dec.get("flags", [])
if not flags:
    print("[e2e]   DECISION: no flag — pipeline finish-ready")
for f in flags:
    print(f"[e2e]   DECISION FLAG: {f['flag']} -> {f['action']}")
PYEOF
fi
say "E2E_FINISH_DONE"
