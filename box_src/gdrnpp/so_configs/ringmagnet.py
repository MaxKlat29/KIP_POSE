"""GDRNPP per-object SO config for pose_isaac obj=ringmagnet — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/ringmagnet.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/ringmagnet"
DATASETS = dict(
    TRAIN=("pose_isaac_ringmagnet_train_pbr",),
    TEST=("pose_isaac_ringmagnet_val",),
)
