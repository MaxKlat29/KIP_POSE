"""GDRNPP per-object SO config for pose_isaac obj=anker_lang — RGB-only.
Deployed to configs/gdrn/poseIsaacPbrSO/anker_lang.py."""
_base_ = ["./base_so.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/anker_lang"
DATASETS = dict(
    TRAIN=("pose_isaac_anker_lang_train_pbr",),
    TEST=("pose_isaac_anker_lang_val",),
)
