"""GDRNPP per-object SO config for pose_isaac obj=zahnrad — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/zahnrad.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/zahnrad"
DATASETS = dict(
    TRAIN=("pose_isaac_zahnrad_train_pbr",),
    TEST=("pose_isaac_zahnrad_val",),
)
