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
[ "${1:-}" = "--gdrnpp-only" ] && GDRNPP_ONLY=1

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
    if [ "${2:-}" = "--smoke" ]; then
        say "SMOKE: 1-config few-iter GPU probe of the INPUT_RES=320 change (anker_kurz)"
        cd "$GDRN"
        PYTHONPATH="$GDRN" CUDA_VISIBLE_DEVICES=0 timeout 900 $GDRN_VENV \
            "$GDRN/core/gdrn_modeling/main_gdrn.py" \
            --config-file "$GDRN/configs/gdrn/poseIsaacPbrSO/anker_kurz.py" \
            --num-gpus 1 \
            SOLVER.TOTAL_EPOCHS 1 TRAIN.MAX_ITER 20 SOLVER.IMS_PER_BATCH 16
        local_rc=$?
        if [ $local_rc -ne 0 ]; then
            say "SMOKE FAILED rc=$local_rc — INPUT_RES=320 shape/OOM issue. ABORT before the multi-day run."
            say "Fix: drop INPUT_RES to 256/64 in base_so.py OR lower IMS_PER_BATCH to 12, then re-smoke."
            exit 40
        fi
        say "SMOKE OK — INPUT_RES=320 forward/backward fits. Proceeding to full training."
    else
        say "NOTE: skipping GPU smoke-test (no --smoke). Run once after the INPUT_RES bump (PHASE2_PLAN.md)."
    fi
else
    # ---------------------------------------------------------------------------
    say "STAGE 1/6: detector retrain (arm-visible) — train-venv"
    $TRAIN_VENV "$REPO/box_src/train_detector_armvis.py" \
        --src  "$REPO/data/sdg_armvis_full" \
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
