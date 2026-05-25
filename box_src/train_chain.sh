#!/usr/bin/env bash
# =============================================================================
# train_chain.sh — sequential single-GPU training chain  (S-401 + S-402)
# -----------------------------------------------------------------------------
# ONE RTX-3090, so detector-retrain and GDRNPP-training must NOT run
# concurrently. This chains them: detector (~1h) THEN GDRNPP per-object
# (multi-day). Launch with nohup and poll the log — do NOT wait inline.
#
#   cd /mnt/data/kip_pose
#   nohup bash box_src/train_chain.sh > /mnt/data/bop/logs/train_chain.log 2>&1 &
#   echo "CHAIN_PID=$!"
#
# Poll:
#   tail -40 /mnt/data/bop/logs/train_chain.log
#   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
#
# Stages (each gated on the previous succeeding):
#   1. Detector retrain (arm-visible)        train-venv      ~1h
#   2. OBB->AABB val detections (bridge)      train-venv      ~min
#   3. GDRNPP deploy (ref+loader+cfg+fps)     gdrnpp-venv     ~min
#   4. GDRNPP SO train: anker_kurz            gdrnpp-venv     ~1-2 days
#   5. GDRNPP SO train: anker_lang            gdrnpp-venv     ~1-2 days
#   6. GDRNPP SO train: zahnrad               gdrnpp-venv     ~1-2 days
# =============================================================================
set -uo pipefail

REPO=/mnt/data/kip_pose
GDRN=/mnt/data/bop/repos/gdrnpp
BOP=$REPO/project/bop/pose_isaac
TRAIN_VENV=/mnt/data/train-venv/bin/python
GDRN_VENV=/mnt/data/bop/gdrnpp-venv/bin/python
LOGDIR=/mnt/data/bop/logs
DETOUT=$REPO/data/detector_armvis
mkdir -p "$LOGDIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[$(ts)] [chain] $*"; }

# --gdrnpp-only : skip stages 1-3 (detector retrain + bridge + deploy) and go
# straight to the GDRNPP per-object training. Used by the T-068 restart: the
# detector (mAP50 0.99) is already trained & must NOT be retrained, and the
# OBB->AABB bridge + GDRNPP deploy artifacts already exist. Verifies the
# prerequisite artifacts exist before skipping.
GDRNPP_ONLY=0
SMOKE=0
# [T-050] --det-src lets the caller point the detector retrain at a different SDG
# bundle than the legacy data/sdg_armvis_full (Phase-2 retrains on the new
# 0-10/full-table data/sdg_armvis_dr5k). Defaults to the legacy path so the
# canonical S-401/402 invocation is unchanged.
DET_SRC="$REPO/data/sdg_armvis_full"
# robust flag parse (order-independent) — old positional `--gdrnpp-only [--smoke]`
# still works because both are matched here.
while [ $# -gt 0 ]; do
    case "$1" in
        --gdrnpp-only) GDRNPP_ONLY=1; shift ;;
        --smoke)       SMOKE=1; shift ;;
        --det-src)     DET_SRC="$2"; shift 2 ;;
        *) echo "[chain] WARN ignoring unknown arg: $1"; shift ;;
    esac
done

if [ "$GDRNPP_ONLY" -eq 1 ]; then
    say "MODE: --gdrnpp-only (skip detector retrain + bridge + deploy)"
    # Verify the upstream artifacts the GDRNPP stages depend on.
    [ -f "$DETOUT/detector.pt" ]                                || { say "ABORT: $DETOUT/detector.pt missing"; exit 11; }
    [ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/fps_points.pkl" ]   || { say "ABORT: fps_points.pkl missing (run deploy first)"; exit 12; }
    [ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/keypoints_3d.pkl" ] || { say "ABORT: keypoints_3d.pkl missing (run deploy first)"; exit 12; }
    # Ensure the EGL/PL/config patches are applied (idempotent, fast).
    say "applying GDRNPP source patches (idempotent)"
    /mnt/data/bop/gdrnpp-venv/bin/python "$REPO/box_src/gdrnpp/patch_gdrnpp_py311.py"
    /mnt/data/bop/gdrnpp-venv/bin/python "$REPO/box_src/gdrnpp/patch_pl19_compat.py"
    /mnt/data/bop/gdrnpp-venv/bin/python "$REPO/box_src/gdrnpp/patch_egl_calc_normals.py"
    # [T-068/PHASE-2] ConvPnPNet final_spatial_size<-OUTPUT_RES (no-op at 64,
    # fixes the 320/80 12800-vs-8192 mat-mul shape mismatch).
    /mnt/data/bop/gdrnpp-venv/bin/python "$REPO/box_src/gdrnpp/patch_pnp_spatial_size.py"
    cp "$REPO/box_src/gdrnpp/config_base_so.py" "$GDRN/configs/gdrn/poseIsaacPbrSO/base_so.py"
    cp "$REPO/box_src/gdrnpp/so_configs/"*.py "$GDRN/configs/gdrn/poseIsaacPbrSO/"   # [T-050] refresh per-obj cfgs too
    rm -rf "$GDRN/.cache"   # drop any stale mesh cache
    say "patches applied, base_so.py + so_configs refreshed, mesh cache cleared"

    # [T-050 / PHASE2] The new base_so.py raises INPUT_RES 256->320 (Zahnrad
    # small-feature fix). This changes the feature-map size into the geo/PnP head
    # and MUST be smoke-tested on the GPU before the multi-day run, or a shape
    # mismatch / OOM only surfaces hours in. Pass --smoke to run a few-iter probe
    # on anker_kurz and ABORT the chain if it fails. Skip with no flag once the
    # res change is validated. Also re-confirm the DR-5k train data is converted
    # into the BOP dataset (PHASE2_PLAN.md §Retrain validation, wiring A or B).
    if [ "$SMOKE" -eq 1 ]; then
        # [T-050] phase2_chain.sh drives the OOM/shape fallback by overriding the
        # smoke batch/resolution via env (SMOKE_BATCH / SMOKE_INPUT_RES /
        # SMOKE_OUTPUT_RES). Defaults match base_so.py (batch 16, 320/80). The
        # config-file values are overridden on the command line so the smoke probe
        # itself is config-pure (no file edits); phase2_chain bakes the WINNING
        # combo into base_so.py before the full run.
        SMK_BATCH="${SMOKE_BATCH:-16}"
        SMK_IN="${SMOKE_INPUT_RES:-320}"
        SMK_OUT="${SMOKE_OUTPUT_RES:-80}"
        say "SMOKE: few-iter GPU probe (anker_kurz) batch=$SMK_BATCH res=$SMK_IN/$SMK_OUT"
        cd "$GDRN"
        # [T-068 FIX] GDRNPP's main_gdrn.py overrides config via the `--opts`
        # flag, but it uses mmcv's DictAction (NOT detectron2's plain opts list):
        #   parser.add_argument("--opts", nargs="+", action=DictAction)
        #   cfg.merge_from_dict(args.opts)
        # So overrides must be `KEY=VALUE` (with '='), space-separated, AFTER
        # `--opts`. The original bug appended them as bare trailing args
        # (`KEY VALUE`) with no `--opts` at all -> argparse "unrecognized
        # arguments" -> rc=2, which was then misread as OOM/shape and sent
        # phase2_chain.sh down its whole fallback ladder for nothing. Two layers:
        #   (1) add --opts                  -> fixes "unrecognized arguments"
        #   (2) use KEY=VALUE not KEY VALUE -> fixes mmcv DictAction split('=')
        # Keys use dot-notation (SOLVER.IMS_PER_BATCH); merge_from_dict resolves
        # them into the nested config dict. We drop the old detectron2-style
        # TRAIN.MAX_ITER (no such key in the mmcv config — TRAIN is a dataset
        # tuple, not a dict); SOLVER.TOTAL_EPOCHS=1 already bounds the probe.
        # The REAL per-object training (run_gdrn below) uses ONLY --config-file
        # with no overrides, so it never hit this — only the smoke probe (which
        # overrides batch/res on the CLI) needs --opts.
        # We capture stderr so the caller can tell a genuine CUDA OOM from an
        # argparse/config error (see phase2_chain.sh smoke ladder).
        # NB: this block runs at script top-level (inside `if`), NOT in a
        # function, so plain vars (no `local` keyword) like the existing
        # local_rc below.
        smoke_err="$LOGDIR/smoke_${SMK_BATCH}_${SMK_IN}_${SMK_OUT}.stderr"
        # [T-068] GDRNPP bounds training by TOTAL_EPOCHS only (no MAX_ITER key in
        # the mmcv config). One epoch on the full train set is too long for a
        # smoke probe, so we cap wall-time with `timeout` and treat "ran many
        # iterations with a finite loss and no OOM/shape error" as the PASS
        # signal — that is exactly what a forward/backward/optimizer-step
        # OOM-and-shape probe needs to prove. SMOKE_SECS=240 -> ~60 iters at
        # ~3.6s/iter (plenty to surface any OOM, which would hit on the FIRST
        # backward, or any shape mismatch, which hits on the FIRST forward).
        SMK_SECS="${SMOKE_SECS:-240}"
        # stderr -> file (synchronous, no process-substitution race so the grep
        # below sees the complete output), then mirror the tail to our own stderr.
        PYTHONPATH="$GDRN" CUDA_VISIBLE_DEVICES=0 timeout "$SMK_SECS" $GDRN_VENV \
            "$GDRN/core/gdrn_modeling/main_gdrn.py" \
            --config-file "$GDRN/configs/gdrn/poseIsaacPbrSO/anker_kurz.py" \
            --num-gpus 1 \
            --opts \
            SOLVER.TOTAL_EPOCHS=1 \
            SOLVER.IMS_PER_BATCH="$SMK_BATCH" \
            MODEL.POSE_NET.INPUT_RES="$SMK_IN" MODEL.POSE_NET.OUTPUT_RES="$SMK_OUT" \
            2>"$smoke_err"
        local_rc=$?
        # surface the smoke stderr into the chain log too (tail is enough)
        tail -25 "$smoke_err" 2>/dev/null | sed 's/^/[smoke-stderr] /' >&2

        # ---- classify the result (in priority order) -------------------------
        # Hard errors are checked FIRST regardless of rc — an OOM or shape error
        # can co-exist with some early iterations, but if it's in the log the
        # combo is NOT viable.
        if grep -qiE "out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|RuntimeError: CUDA out of memory" "$smoke_err" 2>/dev/null; then
            say "SMOKE OOM rc=$local_rc — genuine CUDA out-of-memory at batch=$SMK_BATCH res=$SMK_IN/$SMK_OUT (fallback is valid)."
            exit 41
        fi
        if grep -qiE "mat1 and mat2 shapes cannot be multiplied|size mismatch|Expected .* but got|RuntimeError: shape" "$smoke_err" 2>/dev/null; then
            say "SMOKE SHAPE-ERROR rc=$local_rc — tensor shape mismatch at res=$SMK_IN/$SMK_OUT, NOT OOM. A smaller batch will not help (this is an arch wiring issue, e.g. ConvPnPNet final_spatial_size). HARD STOP."
            exit 42
        fi
        if grep -qiE "unrecognized arguments|error: argument|usage: main_gdrn|invalid choice|the following arguments are required|not enough values to unpack|DictAction|KeyError|merge_from_dict" "$smoke_err" 2>/dev/null; then
            say "SMOKE ARG-ERROR rc=$local_rc — argparse/config error, NOT OOM. A smaller batch/res will not help. HARD STOP."
            say "  (see $smoke_err — likely a --opts / config-key bug, fix the invocation, do NOT keep lowering res)"
            exit 42
        fi

        # PASS path: the probe completed cleanly (rc=0) OR it was cut off by the
        # wall-time cap (rc=124) AFTER proving real training iterations ran with a
        # finite loss. Count the per-iter training log lines (`total_loss:` is
        # emitted once per logged iter). >=5 logged iters with no hard error =
        # forward+backward+optimizer-step all fit and run.
        n_iters=$(grep -cE "iter: [0-9]+/[0-9]+.*total_loss:" "$smoke_err" 2>/dev/null || echo 0)
        if [ "$local_rc" -eq 0 ]; then
            say "SMOKE OK (rc=0, completed) — batch=$SMK_BATCH res=$SMK_IN/$SMK_OUT, $n_iters logged iters, forward/backward fits."
        elif { [ "$local_rc" -eq 124 ] || [ "$local_rc" -eq 137 ]; } && [ "${n_iters:-0}" -ge 5 ]; then
            mm=$(grep -oE "max_mem: [0-9]+M" "$smoke_err" 2>/dev/null | tail -1)
            say "SMOKE OK (rc=$local_rc, wall-time cap after $n_iters iters) — batch=$SMK_BATCH res=$SMK_IN/$SMK_OUT trains cleanly ($mm peak, no OOM/shape error)."
        else
            # Unknown non-zero failure with no iteration progress (import error,
            # segfault, config KeyError, data-load crash...). Not OOM, no proof of
            # a working forward/backward — do not blindly fall back.
            say "SMOKE FAILED rc=$local_rc — non-OOM failure, only $n_iters training iters logged (import/segfault/config/data). HARD STOP, inspect $smoke_err."
            exit 43
        fi
    else
        say "NOTE: skipping GPU smoke-test (no --smoke). Run once after the INPUT_RES bump (PHASE2_PLAN.md)."
    fi
else
    # ---------------------------------------------------------------------------
    say "STAGE 1/6: detector retrain (arm-visible) — train-venv  [src=$DET_SRC]"
    $TRAIN_VENV "$REPO/box_src/train_detector_armvis.py" \
        --src  "$DET_SRC" \
        --out  "$DETOUT" \
        --epochs 100 --imgsz 1280 --batch 8 --max-occ 0.85
    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$DETOUT/detector.pt" ]; then
        say "STAGE 1 FAILED rc=$RC (no detector.pt) — ABORTING CHAIN"; exit 10
    fi
    say "STAGE 1 OK: $DETOUT/detector.pt"

    # ---------------------------------------------------------------------------
    say "STAGE 2/6: OBB->AABB val detections (BOP detection bridge)"
    $TRAIN_VENV "$REPO/box_src/obb_to_aabb_dets.py" \
        --weights "$DETOUT/detector.pt" \
        --bop-root "$BOP" --split val \
        --out "$BOP/val/det_obb2aabb_pose_isaac_val.json" \
        --conf 0.1 --imgsz 1280 || say "STAGE 2 WARN: det bridge failed (non-fatal, GDRNPP uses GT boxes for val)"

    # ---------------------------------------------------------------------------
    say "STAGE 3a/6: GDRNPP env prep (mmcv/PL/numpy pins + py311/PL19/EGL patches + EGL build)"
    bash "$REPO/box_src/gdrnpp/prep_gdrnpp_env.sh" || say "STAGE 3a WARN: prep returned non-zero (may already be set up)"
    say "STAGE 3b/6: deploy pose_isaac into GDRNPP (ref+loader+cfg+symlink+fps)"
    bash "$REPO/box_src/gdrnpp/deploy_gdrnpp_pose_isaac.sh"
    RC=$?
    if [ $RC -ne 0 ]; then say "STAGE 3 FAILED rc=$RC — ABORTING CHAIN"; exit 30; fi
    say "STAGE 3 OK"
fi   # end of full-pipeline (stages 1-3) block

# ---------------------------------------------------------------------------
# GDRNPP single-object training.
#
# HARDENING (T-068): the previous version captured `rc` but NEVER checked it,
# so a segfault (rc=139) was logged as "finished rc=139" and the chain still
# printed TRAIN_CHAIN_DONE — a false success signal that hid 0 checkpoints.
# Now: a non-zero rc is reported as STAGE FAILED and is NOT a success. We also
# verify a real checkpoint .pth actually landed (rc=0 alone is not enough —
# that was the exact failure mode), and we summarize per-object pass/fail at the
# end. Exit code is non-zero unless ALL objects produced a checkpoint.
GDRN_OUT="$GDRN/output/gdrn/poseIsaacPbrSO"
FAILED_OBJS=()
OK_OBJS=()

run_gdrn() {
    local obj="$1" stage="$2"
    say "STAGE $stage: GDRNPP SO train obj=$obj — gdrnpp-venv (multi-day)"
    cd "$GDRN"
    PYTHONPATH="$GDRN" CUDA_VISIBLE_DEVICES=0 $GDRN_VENV \
        "$GDRN/core/gdrn_modeling/main_gdrn.py" \
        --config-file "$GDRN/configs/gdrn/poseIsaacPbrSO/${obj}.py" \
        --num-gpus 1
    local rc=$?

    # rc check — segfault (139) / any non-zero is a hard FAILURE, not "finished".
    if [ $rc -ne 0 ]; then
        say "STAGE $stage obj=$obj FAILED rc=$rc (139=segfault/core-dumped)"
        FAILED_OBJS+=("$obj(rc=$rc)")
        return $rc
    fi

    # rc=0 is necessary but NOT sufficient: confirm a checkpoint .pth exists.
    local ckpt
    ckpt=$(ls -t "$GDRN_OUT/$obj"/*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        say "STAGE $stage obj=$obj FAILED: rc=0 but NO checkpoint .pth in $GDRN_OUT/$obj"
        FAILED_OBJS+=("$obj(no-ckpt)")
        return 1
    fi

    say "STAGE $stage obj=$obj OK rc=0 — checkpoint: $ckpt"
    OK_OBJS+=("$obj")
    return 0
}

# Train all three. Do NOT abort the whole chain on one object's failure —
# a later object may still train — but track failures so the final status
# is honest. (Single GPU: these run strictly sequentially regardless.)
run_gdrn anker_kurz "4/6" || say "continuing chain after anker_kurz failure"
run_gdrn anker_lang "5/6" || say "continuing chain after anker_lang failure"
run_gdrn zahnrad    "6/6" || say "continuing chain after zahnrad failure"

# ---------------------------------------------------------------------------
say "GDRNPP summary: OK=[${OK_OBJS[*]:-none}] FAILED=[${FAILED_OBJS[*]:-none}]"
if [ ${#FAILED_OBJS[@]} -ne 0 ]; then
    say "TRAIN_CHAIN_FAILED — ${#FAILED_OBJS[@]}/3 GDRNPP objects did not produce a checkpoint"
    exit 60
fi
say "TRAIN_CHAIN_DONE (detector + 3 GDRNPP SO models, all checkpointed)"
