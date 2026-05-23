"""GDRNPP per-object SO config for pose_isaac obj=anker_kurz — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/anker_kurz.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/anker_kurz"
DATASETS = dict(
    TRAIN=("pose_isaac_anker_kurz_train_pbr",),
    TEST=("pose_isaac_anker_kurz_val",),
)
