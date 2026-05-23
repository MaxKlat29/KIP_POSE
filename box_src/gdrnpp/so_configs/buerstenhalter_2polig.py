"""GDRNPP per-object SO config for pose_isaac obj=buerstenhalter_2polig — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/buerstenhalter_2polig.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/buerstenhalter_2polig"
DATASETS = dict(
    TRAIN=("pose_isaac_buerstenhalter_2polig_train_pbr",),
    TEST=("pose_isaac_buerstenhalter_2polig_val",),
)
