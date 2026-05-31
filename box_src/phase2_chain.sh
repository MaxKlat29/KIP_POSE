#!/usr/bin/env bash
# =============================================================================
# phase2_chain.sh — AUTONOMOUS Phase-2 box pipeline  (T-050)
# -----------------------------------------------------------------------------
# Runs UNATTENDED from "render still going" all the way to "trained models",
# on the one RTX-3090. The GPU is currently busy rendering sdg_armvis_dr5k, so
# stage 1 WAITS for the render to finish before touching the GPU. Launch with
# setsid+nohup and poll the log — do NOT wait inline.
#
#   cd /mnt/data/kip_pose
#   setsid nohup bash box_src/phase2_chain.sh \
#       > /mnt/data/bop/logs/phase2_chain.log 2>&1 </dev/null &
#   echo "PHASE2_PID=$!"
#
# Poll:
#   tail -50 /mnt/data/bop/logs/phase2_chain.log
#   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
#
# MILESTONE MARKERS (grep these in the log):
#   WAIT_RENDER_DONE       render finished, GPU free
#   CONVERT_DONE           BOP dataset built + bop_toolkit-validated
#   SYM_CHECK_DONE         Anker C_2 decision made (+ keypoints regenerated)
#   SMOKE_DONE             INPUT_RES smoke passed (winning batch/res baked in)
#   DETECTOR_TRAIN_DONE    detector retrained on the new full-table data
#   PHASE2_TRAIN_DONE      all GDRNPP objects checkpointed — SUCCESS
#   PHASE2_FAILED          a hard stage failed — STOP, inspect (NEVER a false DONE)
#
# STAGES (sequential, each gated on the previous):
#   1. WAIT-RENDER   poll until gen_sdg_arm_visible gone OR ~8000 scenes
#   2. CONVERT       convert_full_to_bop.py --min-visib 0.20 + bop_toolkit validate
#   3. SYM-CHECK     anker_c2_check.py (geometric C_2 decision) + deploy(fps/kpts)
#   4. SMOKE         320/80 GPU probe with auto OOM/shape fallback
#   5. RETRAIN       detector (NEW data) + GDRNPP all 3 objects (research fixes)
# =============================================================================
set -uo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
REPO=/mnt/data/kip_pose
GDRN=/mnt/data/bop/repos/gdrnpp
BOP=$REPO/project/bop/pose_isaac
RAW="${RAW:-/mnt/data/bop/sdg_zivid_10k}"              # the render output (raw bundle)
                                                       # default: 2026-05-26 10k Marc-Zivid run
BOP_VENV=/mnt/data/bop/bop-venv/bin/python
LOGDIR=/mnt/data/bop/logs
DETOUT=$REPO/data/detector_armvis
EXPECT_SCENES="${EXPECT_SCENES:-10000}"
mkdir -p "$LOGDIR"
cd "$REPO" || { echo "FATAL: cannot cd $REPO"; echo "PHASE2_FAILED"; exit 1; }

ts()   { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say()  { echo "[$(ts)] [phase2] $*"; }
fail() { say "STAGE FAILED: $*"; say "PHASE2_FAILED"; exit "${2:-1}"; }

# [T-068] --from-train : skip stages 1-3 (WAIT-RENDER, CONVERT, SYM-CHECK+deploy)
# and resume at stage 4 (SMOKE) + stage 5 (RETRAIN). Used by the T-068 relaunch:
# the render finished, the BOP convert (8000->train/val, ~3.4h) is DONE, and the
# sym-check + deploy (fps/keypoints) are DONE. Re-running them with --fresh would
# throw away 3.4h of conversion for nothing. This flag verifies the convert+deploy
# artifacts exist, then jumps straight to the smoke + training stages.
#
# [T-068 / S-053] --from-gdrnpp : skip stages 1-4 (WAIT-RENDER, CONVERT, SYM-CHECK,
# SMOKE) AND stage 5a (detector retrain) + 5b (OBB->AABB bridge), and go straight
# to GDRNPP per-object training (stage 5c). Used by the "abfahrt" relaunch: the
# detector converged early (best.pt = ep55, mAP50 0.992, val flat) and was
# published as detector.pt by hand, so retraining it 100 epochs again would waste
# ~7h on the one 3090. The smoke already baked the winning 320/80@16 combo into
# base_so.py. This flag verifies the EXISTING detector.pt + fps/keypoints, logs a
# DETECTOR_TRAIN_DONE (skipped/published) marker so the phase-bar monitor still
# sees the milestone, then jumps to GDRNPP. PHASE2_TRAIN_DONE / PHASE2_FAILED are
# written at the end exactly as in the full run.
FROM_TRAIN=0
FROM_GDRNPP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-train)  FROM_TRAIN=1; shift ;;
        --from-gdrnpp) FROM_GDRNPP=1; shift ;;
        *) say "WARN ignoring unknown arg: $1"; shift ;;
    esac
done
# --from-gdrnpp implies everything --from-train skips, plus the detector; treat it
# as a superset so the stage-1-3 guard below also short-circuits.
[ "$FROM_GDRNPP" -eq 1 ] && FROM_TRAIN=1

say "=== PHASE-2 AUTONOMOUS CHAIN START (T-050) ==="
say "repo=$REPO  raw=$RAW  bop=$BOP  expect_scenes=$EXPECT_SCENES  from_train=$FROM_TRAIN  from_gdrnpp=$FROM_GDRNPP"

if [ "$FROM_TRAIN" -eq 1 ]; then
    if [ "$FROM_GDRNPP" -eq 1 ]; then
        say "MODE: --from-gdrnpp — skipping WAIT-RENDER + CONVERT + SYM-CHECK + SMOKE + DETECTOR (detector.pt already published, smoke combo already baked)."
    else
        say "MODE: --from-train — skipping WAIT-RENDER + CONVERT + SYM-CHECK (already done)."
    fi
    # Verify the upstream artifacts the smoke + training stages depend on, so we
    # fail loudly here instead of hours into a run if convert/deploy didn't land.
    [ -d "$BOP/train_pbr" ] && [ -n "$(ls -A "$BOP/train_pbr" 2>/dev/null)" ] || fail "--from-train: $BOP/train_pbr missing/empty (convert not done?)" 2
    [ -d "$BOP/val" ]       && [ -n "$(ls -A "$BOP/val" 2>/dev/null)" ]       || fail "--from-train: $BOP/val missing/empty (convert not done?)" 2
    [ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/fps_points.pkl" ]   || fail "--from-train: fps_points.pkl missing (deploy not done?)" 3
    [ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/keypoints_3d.pkl" ] || fail "--from-train: keypoints_3d.pkl missing (deploy not done?)" 3
    say "--from-train artifacts verified: train_pbr=$(ls "$BOP/train_pbr" | wc -l) scenes, val=$(ls "$BOP/val" | wc -l) scenes, fps+keypoints present."
    if [ "$FROM_GDRNPP" -eq 1 ]; then
        # --from-gdrnpp skips the detector stage, so the published detector.pt is a
        # hard prerequisite — fail loudly NOW if it's not there (don't discover it
        # only when inference/the bridge needs it). Also confirm it loads.
        [ -f "$DETOUT/detector.pt" ] || fail "--from-gdrnpp: $DETOUT/detector.pt missing (publish the converged detector best.pt first)" 4
        say "--from-gdrnpp: published detector.pt present ($(ls -la "$DETOUT/detector.pt" | awk '{print $5" bytes, mtime "$6" "$7" "$8}'))."
    fi
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/[phase2]   GPU: /'
fi

if [ "$FROM_TRAIN" -ne 1 ]; then
# =============================================================================
# STAGE 1 — WAIT-RENDER  (GPU is busy; do NOT touch it until the render frees it)
# =============================================================================
say "STAGE 1/5: WAIT-RENDER — polling until gen_sdg_arm_visible exits OR ~$EXPECT_SCENES scenes"
POLL=300            # 5 min between polls
WAITED=0
while true; do
    # render process alive?  (match the script name, not a substring of our poll cmd)
    if pgrep -f "gen_sdg_arm_visible.py" >/dev/null 2>&1; then
        ALIVE=1
    else
        ALIVE=0
    fi
    NSC=$(ls "$RAW"/gt_raw_*.json 2>/dev/null | wc -l | tr -d ' ')
    say "  poll: render_alive=$ALIVE scenes=$NSC/$EXPECT_SCENES waited=${WAITED}s"
    # done when the process is gone OR we have enough scenes
    if [ "$ALIVE" -eq 0 ]; then
        say "  render process gone."
        break
    fi
    if [ "${NSC:-0}" -ge "$EXPECT_SCENES" ]; then
        say "  scene target reached while process still alive — wait 60s for it to flush+exit."
        sleep 60
        break
    fi
    sleep "$POLL"
    WAITED=$((WAITED + POLL))
done
# Settle: ensure no render python is still holding the GPU before we proceed.
for i in $(seq 1 10); do
    pgrep -f "gen_sdg_arm_visible.py" >/dev/null 2>&1 || break
    say "  render still finishing... ($i/10)"; sleep 30
done
FINAL_NSC=$(ls "$RAW"/gt_raw_*.json 2>/dev/null | wc -l | tr -d ' ')
[ "${FINAL_NSC:-0}" -lt 100 ] && fail "only $FINAL_NSC raw scenes in $RAW — render produced too little" 20
say "render done: $FINAL_NSC raw scenes available."
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/[phase2]   GPU: /'
say "WAIT_RENDER_DONE"

# =============================================================================
# STAGE 2 — CONVERT  (Isaac raw bundle -> BOP train/val, >20% visibility filter)
# =============================================================================
say "STAGE 2/5: CONVERT — convert_full_to_bop.py --min-visib 0.20 --fresh"
$BOP_VENV "$REPO/box_src/convert_full_to_bop.py" \
    --raw-dir   "$RAW" \
    --bop-root  "$BOP" \
    --python    "$BOP_VENV" \
    --val-frac 0.1 --scene-size 100 --seed 7 --min-visib 0.20 --fresh
RC=$?
[ $RC -ne 0 ] && fail "convert_full_to_bop.py rc=$RC" 21
# sanity: train_pbr + val dirs exist and are non-empty
[ -d "$BOP/train_pbr" ] && [ -n "$(ls -A "$BOP/train_pbr" 2>/dev/null)" ] || fail "no $BOP/train_pbr scenes after convert" 21
[ -d "$BOP/val" ]       && [ -n "$(ls -A "$BOP/val" 2>/dev/null)" ]       || fail "no $BOP/val scenes after convert" 21

# --- bop_toolkit native-load validation + visib stats + per-part GT counts ---
say "STAGE 2: bop_toolkit validation + visib_fract stats + per-part >20% GT counts"
$BOP_VENV - "$BOP" <<'PYEOF'
import json, os, sys
import numpy as np
bop = sys.argv[1]
# (a) bop_toolkit native load over all scenes (key alignment = the hard gate)
try:
    from bop_toolkit_lib import inout
except Exception as e:
    print(f"[validate] FATAL bop_toolkit import: {e}"); sys.exit(2)

OBJ = {1:"Anker_Kurz",2:"Anker_Lang",3:"Buerstenhalter_2polig",
       4:"Getriebegehaeuse_typ4",5:"Ringmagnet",6:"Zahnrad"}
per_obj = {o:0 for o in OBJ}          # instances surviving the >20% filter
visibs = []
n_frames = 0; n_inst = 0
bad = 0
for split in ("train_pbr","val"):
    sdir = os.path.join(bop, split)
    if not os.path.isdir(sdir): continue
    for scene in sorted(os.listdir(sdir)):
        sp = os.path.join(sdir, scene)
        if not os.path.isdir(sp): continue
        try:
            cam = inout.load_scene_camera(os.path.join(sp,"scene_camera.json"))
            gt  = inout.load_scene_gt(os.path.join(sp,"scene_gt.json"))
            gti = json.load(open(os.path.join(sp,"scene_gt_info.json")))
        except Exception as e:
            print(f"[validate] LOAD FAIL {split}/{scene}: {e}"); bad += 1; continue
        # key alignment across the three files
        if set(map(str,gt.keys())) != set(map(str,cam.keys())):
            print(f"[validate] KEY MISMATCH gt vs cam in {split}/{scene}"); bad += 1
        for im, insts in gt.items():
            n_frames += 1
            infos = gti.get(str(im), [])
            for j, a in enumerate(insts):
                n_inst += 1
                oid = a["obj_id"]
                if oid in per_obj: per_obj[oid] += 1
                if j < len(infos) and "visib_fract" in infos[j]:
                    visibs.append(float(infos[j]["visib_fract"]))
visibs = np.array(visibs) if visibs else np.array([0.0])
print(f"[validate] frames={n_frames} instances={n_inst} bad_scenes={bad}")
print(f"[validate] visib_fract: min={visibs.min():.3f} mean={visibs.mean():.3f} "
      f"max={visibs.max():.3f}  (all should be >0.20 after --min-visib 0.20)")
under = int((visibs <= 0.20).sum())
if under: print(f"[validate] WARN {under} instances <=0.20 slipped the filter (expected ~0)")
print("[validate] per-part >20% GT instances:")
THIN = 200    # warn if a trainable part has too few >20% GT instances
thin_parts = []
for o,name in OBJ.items():
    c = per_obj[o]
    flag = "  <-- THIN" if (o in (1,2,6) and c < THIN) else ""
    if flag: thin_parts.append(name)
    print(f"   obj {o} {name:24s}: {c}{flag}")
if thin_parts:
    print(f"[validate] WARN trainable parts with <{THIN} >20%-GT instances: {thin_parts} "
          f"(zahnrad/anker need enough — flagging, NOT aborting; 8k scenes should cover)")
if bad:
    print(f"[validate] FATAL {bad} scene(s) failed native load"); sys.exit(3)
print("VALIDATE_BOP_OK")
PYEOF
RC=$?
[ $RC -ne 0 ] && fail "BOP validation rc=$RC (bop_toolkit native load / key alignment)" 22
say "CONVERT_DONE"

# =============================================================================
# STAGE 3 — SYM-CHECK  (Anker C_2 geometric decision, then deploy regen fps/kpts)
# =============================================================================
say "STAGE 3/5: SYM-CHECK — anker_c2_check.py (geometric C_2 decision, --apply)"
$BOP_VENV "$REPO/box_src/gdrnpp/anker_c2_check.py" \
    --bop-root "$BOP" --apply \
    --json-out "$LOGDIR/anker_c2_verdict.json"
RC=$?
[ $RC -ne 0 ] && fail "anker_c2_check.py rc=$RC" 30
# echo the machine-parseable verdict line for the log
grep -h "ANKER_C2_VERDICT" "$LOGDIR/phase2_chain.log" 2>/dev/null | tail -1 || true
say "SYM-CHECK: models_info.json reflects the C_2 decision (added iff geometrically true)."

# GDRNPP deploy: copies ref/loader/config + symlinks BOP + regenerates
# fps_points.pkl & keypoints_3d.pkl FROM the (possibly C_2-patched) models_info.
# This is the step that propagates a symmetry change into the training keypoints.
say "STAGE 3: deploy GDRNPP (regenerate fps_points + keypoints_3d from current models_info)"
bash "$REPO/box_src/gdrnpp/deploy_gdrnpp_pose_isaac.sh"
RC=$?
[ $RC -ne 0 ] && fail "deploy_gdrnpp_pose_isaac.sh rc=$RC (fps/keypoints regen)" 31
[ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/fps_points.pkl" ]   || fail "fps_points.pkl missing after deploy" 31
[ -f "$GDRN/datasets/BOP_DATASETS/pose_isaac/models/keypoints_3d.pkl" ] || fail "keypoints_3d.pkl missing after deploy" 31
say "SYM_CHECK_DONE"
fi   # end of "if FROM_TRAIN != 1" (stages 1-3: WAIT-RENDER + CONVERT + SYM-CHECK)

# =============================================================================
# STAGE 4 — SMOKE  (320/80 GPU probe; auto OOM/shape fallback before the long run)
# =============================================================================
if [ "$FROM_GDRNPP" -eq 1 ]; then
    # The smoke already ran in the previous launch and baked the winning combo
    # (320/80@16) into base_so.py + config_base_so.py. Re-running it would waste a
    # few GPU-minutes for no new info. Skip and emit the marker for the monitor.
    say "STAGE 4/5: SMOKE — SKIPPED (--from-gdrnpp; winning 320/80@16 already baked into base_so.py)"
    say "SMOKE_DONE (skipped — combo already baked)"
else
say "STAGE 4/5: SMOKE — INPUT_RES=320 probe with auto fallback (batch 16 -> 12 -> crop 288/72 -> 256/64)"
# Fallback ladder: (batch, in_res, out_res). First that passes wins and is baked
# into base_so.py before the multi-day run. 320/80 @16 is the target.
SMOKE_LADDER=(
    "16 320 80"
    "12 320 80"
    "12 288 72"
    "16 256 64"
)
SMOKE_OK=0
WIN_BATCH=""; WIN_IN=""; WIN_OUT=""
# [T-068] The fallback ladder is ONLY valid for genuine CUDA OOM. train_chain.sh
# now returns a discriminating exit code from the smoke probe (and treats a
# wall-time cap with >=5 clean training iters as PASS, since GDRNPP has no
# MAX_ITER key — see train_chain.sh):
#   0  = PASS (completed or wall-time cap after clean iters) -> bake, stop ladder
#   41 = genuine CUDA OOM           -> try the next (smaller) rung
#   42 = argparse/config OR tensor-shape error -> HARD STOP (smaller res/batch
#        won't fix an arch wiring or arg bug)
#   43 = other non-OOM crash with no iter progress -> HARD STOP (import/segfault/
#        config/data)
# The original version treated EVERY non-zero as "OOM" and walked the whole
# ladder, which is exactly how a trivial --opts argparse bug (rc=2) masqueraded
# as "INPUT_RES change unworkable / OOM" and crashed the chain.
for combo in "${SMOKE_LADDER[@]}"; do
    set -- $combo
    b=$1; ir=$2; orr=$3
    say "  smoke try: batch=$b res=$ir/$orr"
    SMOKE_BATCH=$b SMOKE_INPUT_RES=$ir SMOKE_OUTPUT_RES=$orr \
        bash "$REPO/box_src/train_chain.sh" --gdrnpp-only --smoke
    smk_rc=$?
    case $smk_rc in
        0)
            SMOKE_OK=1; WIN_BATCH=$b; WIN_IN=$ir; WIN_OUT=$orr
            say "  smoke PASS at batch=$b res=$ir/$orr"
            break ;;
        41)
            say "  smoke OOM at batch=$b res=$ir/$orr — genuine CUDA OOM, trying next (smaller) fallback" ;;
        42)
            fail "smoke ARG/CONFIG/SHAPE error at batch=$b res=$ir/$orr (rc=42) — argparse/config bug OR tensor-shape mismatch, NOT OOM. A smaller batch/res would not fix it. See $LOGDIR/smoke_${b}_${ir}_${orr}.stderr" 42 ;;
        43)
            fail "smoke NON-OOM crash at batch=$b res=$ir/$orr (rc=43) — import/segfault/config/data error, NOT OOM. See $LOGDIR/smoke_${b}_${ir}_${orr}.stderr" 43 ;;
        *)
            # train_chain.sh's own pre-flight aborts (missing detector/fps/etc.)
            # return non-OOM codes (10/11/12/30). Those are not OOM either.
            fail "smoke probe returned unexpected rc=$smk_rc at batch=$b res=$ir/$orr — NOT a known OOM path. HARD STOP." "$smk_rc" ;;
    esac
done
[ "$SMOKE_OK" -eq 1 ] || fail "all smoke fallbacks OOMed (320/80@16 .. 256/64@16) — INPUT_RES change genuinely does not fit on the 3090 even at 256/64@16" 40

# Bake the winning combo into base_so.py on the box AND back into the source copy
# so the full per-object run uses exactly the validated combo. Edit both the
# deployed config and the box source-of-truth box_src copy.
say "  baking winning combo into base_so.py: batch=$WIN_BATCH res=$WIN_IN/$WIN_OUT"
for cfg in "$GDRN/configs/gdrn/poseIsaacPbrSO/base_so.py" "$REPO/box_src/gdrnpp/config_base_so.py"; do
    [ -f "$cfg" ] || continue
    $BOP_VENV - "$cfg" "$WIN_BATCH" "$WIN_IN" "$WIN_OUT" <<'PYEOF'
import re, sys
path, b, ir, orr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(path).read()
s = re.sub(r"IMS_PER_BATCH=\d+", f"IMS_PER_BATCH={b}", s, count=1)
s = re.sub(r"INPUT_RES=\d+", f"INPUT_RES={ir}", s, count=1)
s = re.sub(r"OUTPUT_RES=\d+", f"OUTPUT_RES={orr}", s, count=1)
open(path, "w").write(s)
print(f"[bake] {path}: IMS_PER_BATCH={b} INPUT_RES={ir} OUTPUT_RES={orr}")
PYEOF
done
say "SMOKE_DONE (winning combo batch=$WIN_BATCH res=$WIN_IN/$WIN_OUT)"
fi   # end of "if FROM_GDRNPP != 1" (stage 4: SMOKE)

# =============================================================================
# STAGE 5 — RETRAIN  (detector on NEW full-table data, then GDRNPP all 3 objects)
# =============================================================================
# STAGE 5a — detector retrain on the NEW 0-10/full-table data (sdg_armvis_dr5k).
# The full-table render has obb_2d_*.json per scene; the detector trains on those
# directly (NOT BOP). We run it ourselves (rather than via train_chain stage 1) so
# the data source + the sym/deploy ordering above are correct.
if [ "$FROM_GDRNPP" -eq 1 ]; then
    # Detector converged early and was published by hand (best.pt -> detector.pt).
    # Don't retrain it for another ~7h on the one 3090. Verify it's there (already
    # checked in the --from-gdrnpp artifact block above) and emit the milestone
    # marker as "published/skipped" so the phase-bar monitor still advances.
    say "STAGE 5a/5: detector retrain — SKIPPED (--from-gdrnpp; using PUBLISHED detector.pt at $DETOUT/detector.pt)"
    [ -f "$DETOUT/detector.pt" ] || fail "--from-gdrnpp: detector.pt vanished before GDRNPP (was present at start)" 50
    say "DETECTOR_TRAIN_DONE (skipped/published — converged detector best.pt published as detector.pt)"
else
say "STAGE 5a/5: detector retrain (arm-visible) on NEW data — $RAW"
/mnt/data/train-venv/bin/python "$REPO/box_src/train_detector_armvis.py" \
    --src "$RAW" --out "$DETOUT" \
    --epochs 100 --imgsz 1280 --batch 8 --max-occ 0.80
RC=$?
if [ $RC -ne 0 ] || [ ! -f "$DETOUT/detector.pt" ]; then
    fail "detector retrain rc=$RC (no detector.pt at $DETOUT)" 50
fi
say "DETECTOR_TRAIN_DONE"
fi   # end of "if FROM_GDRNPP != 1" (stage 5a: detector retrain)

# STAGE 5b — OBB->AABB val detection bridge (non-fatal: GDRNPP val uses GT boxes).
say "STAGE 5b/5: OBB->AABB val detections (BOP detection bridge, non-fatal)"
/mnt/data/train-venv/bin/python "$REPO/box_src/obb_to_aabb_dets.py" \
    --weights "$DETOUT/detector.pt" \
    --bop-root "$BOP" --split val \
    --out "$BOP/val/det_obb2aabb_pose_isaac_val.json" \
    --conf 0.1 --imgsz 1280 || say "  det bridge failed (non-fatal)"

# STAGE 5c — GDRNPP all 3 objects with the research fixes (--gdrnpp-only skips the
# detector + bridge + deploy we already did; re-applies patches + refreshes configs;
# we DON'T pass --smoke here, the smoke already validated the res). train_chain.sh
# is rc-hardened (T-068): per-object rc + checkpoint-existence check; exits 60 if
# any object fails to produce a .pth.
say "STAGE 5c/5: GDRNPP per-object training (anker_kurz -> anker_lang -> zahnrad), research fixes baked in"
bash "$REPO/box_src/train_chain.sh" --gdrnpp-only
RC=$?
if [ $RC -ne 0 ]; then
    fail "train_chain.sh --gdrnpp-only rc=$RC (>=1 GDRNPP object did not checkpoint)" 60
fi
# double-check all three checkpoints exist (belt-and-suspenders over train_chain's own check)
GDRN_OUT="$GDRN/output/gdrn/poseIsaacPbrSO"
MISSING=()
for obj in anker_kurz anker_lang zahnrad; do
    ls -t "$GDRN_OUT/$obj"/*.pth >/dev/null 2>&1 || MISSING+=("$obj")
done
[ ${#MISSING[@]} -ne 0 ] && fail "GDRNPP checkpoints missing for: ${MISSING[*]}" 61

say "all 3 GDRNPP checkpoints present: $(for o in anker_kurz anker_lang zahnrad; do echo -n "$o=$(ls -t "$GDRN_OUT/$o"/*.pth 2>/dev/null | head -1) "; done)"
say "PHASE2_TRAIN_DONE (detector retrained on full-table data + 3 GDRNPP SO models, all checkpointed, research fixes applied)"
