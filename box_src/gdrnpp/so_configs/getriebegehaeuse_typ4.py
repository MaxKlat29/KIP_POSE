"""GDRNPP per-object SO config for pose_isaac obj=getriebegehaeuse_typ4 — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/getriebegehaeuse_typ4.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/getriebegehaeuse_typ4"
DATASETS = dict(
    TRAIN=("pose_isaac_getriebegehaeuse_typ4_train_pbr",),
    TEST=("pose_isaac_getriebegehaeuse_typ4_val",),
)
