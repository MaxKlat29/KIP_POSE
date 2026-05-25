#!/usr/bin/env bash
# =============================================================================
# deploy_gdrnpp_pose_isaac.sh — wire the pose_isaac dataset into the GDRNPP repo
# -----------------------------------------------------------------------------
# Idempotent. Run ON the box. Copies ref + loader + configs into the GDRNPP repo,
# symlinks the BOP dataset into datasets/BOP_DATASETS, patches the two
# registration points (ref/__init__.py + dataset_factory.py), and runs the fast
# CPU preprocessing (fps_points + keypoints_3d). XYZ crops are rendered ONLINE
# during training (XYZ_ONLINE=True). [T-068] We measured offline XYZ precompute
# as a speed fix and it was speed-NEUTRAL (3.0 s/iter both ways) — the run is
# compute-bound on convnext_base@320, not render-bound. The offline tool
# (pose_isaac_pbr_gen_xyz.py) is kept for the record but NOT run here.
#
#   bash deploy_gdrnpp_pose_isaac.sh
# =============================================================================
set -euo pipefail

GDRN=/mnt/data/bop/repos/gdrnpp
BOP=/mnt/data/kip_pose/project/bop/pose_isaac
SRC=/mnt/data/kip_pose/box_src/gdrnpp        # where the staged files live on the box
PY=/mnt/data/bop/gdrnpp-venv/bin/python

echo "[deploy] 1) ref/pose_isaac.py"
cp "$SRC/ref_pose_isaac.py" "$GDRN/ref/pose_isaac.py"

echo "[deploy] 2) dataset loader"
cp "$SRC/pose_isaac_pbr.py" "$GDRN/core/gdrn_modeling/datasets/pose_isaac_pbr.py"

echo "[deploy] 3) configs"
mkdir -p "$GDRN/configs/gdrn/poseIsaacPbrSO"
cp "$SRC/config_base_so.py" "$GDRN/configs/gdrn/poseIsaacPbrSO/base_so.py"
cp "$SRC"/so_configs/*.py    "$GDRN/configs/gdrn/poseIsaacPbrSO/"

echo "[deploy] 4) symlink BOP dataset into datasets/BOP_DATASETS"
mkdir -p "$GDRN/datasets/BOP_DATASETS"
ln -sfn "$BOP" "$GDRN/datasets/BOP_DATASETS/pose_isaac"

echo "[deploy] 5) patch ref/__init__.py"
$PY - <<PY
p = "$GDRN/ref/__init__.py"
s = open(p).read()
if "pose_isaac" not in s:
    # append to the existing 'from . import ...' line
    import re
    s2 = re.sub(r"(from \. import [^\n]+)", r"\1, pose_isaac", s, count=1)
    if s2 == s:  # no match -> add a fresh import line
        s2 = s.rstrip() + "\nfrom . import pose_isaac\n"
    open(p, "w").write(s2)
    print("  patched ref/__init__.py")
else:
    print("  ref/__init__.py already has pose_isaac")
PY

echo "[deploy] 5b) GDRNPP py311 + PL1.9 source patches (idempotent)"
$PY "$SRC/patch_gdrnpp_py311.py"
$PY "$SRC/patch_pl19_compat.py"

echo "[deploy] 6) patch dataset_factory.py"
$PY - <<PY
p = "$GDRN/core/gdrn_modeling/datasets/dataset_factory.py"
s = open(p).read()
changed = False
if "pose_isaac_pbr," not in s and "pose_isaac_pbr\n" not in s:
    # add to the 'from core.gdrn_modeling.datasets import (' block (after itodd_d2,)
    s = s.replace("    itodd_d2,\n)", "    itodd_d2,\n    pose_isaac_pbr,\n)", 1)
    changed = True
if '"pose_isaac_pbr"' not in s:
    # add to the _DSET_MOD_NAMES list (after "itodd_d2",)
    s = s.replace('    "itodd_d2",\n]', '    "itodd_d2",\n    "pose_isaac_pbr",\n]', 1)
    changed = True
open(p, "w").write(s)
print("  patched dataset_factory.py" if changed else "  dataset_factory.py already patched")
PY

echo "[deploy] 7) preprocessing: fps_points + keypoints_3d (CPU, fast)"
cp "$SRC/pose_isaac_compute_fps.py"          "$GDRN/core/gdrn_modeling/tools/pose_isaac_compute_fps.py" 2>/dev/null || true
cp "$SRC/pose_isaac_compute_keypoints_3d.py" "$GDRN/core/gdrn_modeling/tools/pose_isaac_compute_keypoints_3d.py" 2>/dev/null || true
cd "$GDRN"
PYTHONPATH="$GDRN" $PY core/gdrn_modeling/tools/pose_isaac_compute_fps.py
PYTHONPATH="$GDRN" $PY core/gdrn_modeling/tools/pose_isaac_compute_keypoints_3d.py

echo "[deploy] 7b) stage offline XYZ tool (NOT run — XYZ_ONLINE=True; see T-068 measurement)"
# pose_isaac_pbr_gen_xyz.py renders the dense XYZ target once per GT instance into
# <split>/xyz_crop/*.pkl (the path pose_isaac_pbr.py populates as xyz_path +
# GDRN_DatasetFromList loads). It works and is fast (~72 s for 17041+1964 inst),
# but offline XYZ was speed-neutral vs online (compute-bound on convnext_base@320),
# so we keep online rendering. To activate offline later: set XYZ_ONLINE=False in
# config_base_so.py and run:
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$GDRN $PY $SRC/pose_isaac_pbr_gen_xyz.py --repo $GDRN --split train_pbr --gpu 0
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$GDRN $PY $SRC/pose_isaac_pbr_gen_xyz.py --repo $GDRN --split val --gpu 0
cp "$SRC/pose_isaac_pbr_gen_xyz.py" "$GDRN/core/gdrn_modeling/tools/pose_isaac_pbr_gen_xyz.py" 2>/dev/null || true

echo "[deploy] 8) sanity: register + count one split"
PYTHONPATH="$GDRN" $PY - <<PY
import sys; sys.path.insert(0, "$GDRN")
import detectron2.data.datasets  # noqa
from core.gdrn_modeling.datasets import pose_isaac_pbr
print("available:", pose_isaac_pbr.get_available_datasets())
pose_isaac_pbr.register_with_name_cfg("pose_isaac_zahnrad_train_pbr")
from detectron2.data import DatasetCatalog
print("DEPLOY_SANITY_OK: registered pose_isaac_zahnrad_train_pbr")
PY

echo "DEPLOY_GDRNPP_POSE_ISAAC_DONE"
