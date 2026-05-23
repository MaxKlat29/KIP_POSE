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
say "STAGE 3/6: deploy pose_isaac into GDRNPP (ref+loader+cfg+symlink+fps)"
bash "$REPO/box_src/gdrnpp/deploy_gdrnpp_pose_isaac.sh"
RC=$?
if [ $RC -ne 0 ]; then say "STAGE 3 FAILED rc=$RC — ABORTING CHAIN"; exit 30; fi
say "STAGE 3 OK"

# ---------------------------------------------------------------------------
# GDRNPP single-object training. train_gdrn.sh CFG GPU_IDS [extra opts]
run_gdrn() {
    local obj="$1" stage="$2"
    say "STAGE $stage: GDRNPP SO train obj=$obj — gdrnpp-venv (multi-day)"
    cd "$GDRN"
    PYTHONPATH="$GDRN" CUDA_VISIBLE_DEVICES=0 $GDRN_VENV \
        "$GDRN/core/gdrn_modeling/main_gdrn.py" \
        --config-file "$GDRN/configs/gdrn/poseIsaacPbrSO/${obj}.py" \
        --num-gpus 1
    local rc=$?
    say "STAGE $stage obj=$obj finished rc=$rc"
    return $rc
}

run_gdrn anker_kurz "4/6"
run_gdrn anker_lang "5/6"
run_gdrn zahnrad    "6/6"

say "TRAIN_CHAIN_DONE (detector + 3 GDRNPP SO models)"
