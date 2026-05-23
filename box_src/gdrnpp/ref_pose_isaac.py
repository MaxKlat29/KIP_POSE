# encoding: utf-8
"""GDRNPP dataset reference for the Isaac arm-visible BOP dataset `pose_isaac`.

Modeled on ref/itodd.py. Deployed to gdrnpp/ref/pose_isaac.py and registered in
ref/__init__.py. obj_id mapping is FROZEN (adr.md §1.2):
  1 Anker_Kurz · 2 Anker_Lang · 3 Buerstenhalter_2polig · 4 Getriebegehaeuse_typ4
  · 5 Ringmagnet · 6 Zahnrad

RGB-only (Max HARD): depth is present in the BOP dataset but the GDRN net runs
in_chans=3 and DEPTH_BACKBONE is disabled — depth is only used to backproject
XYZ crops during preprocessing.
"""
import os.path as osp
import mmcv
import numpy as np

cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# pose_isaac DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "pose_isaac")
train_dir = osp.join(dataset_root, "train_pbr")
val_dir = osp.join(dataset_root, "val")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001

# object info (FROZEN, adr.md §1.2)
id2obj = {
    1: "anker_kurz",
    2: "anker_lang",
    3: "buerstenhalter_2polig",
    4: "getriebegehaeuse_typ4",
    5: "ringmagnet",
    6: "zahnrad",
}
objects = list(id2obj.values())
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]

# diameters (mm -> m), order = id2obj order. From models_info.json.
diameters = (
    np.array([112.1202, 115.1634, 63.7169, 117.7517, 10.8809, 49.9346]) / 1000.0
)

# Camera info — top-down render, 1280x720. Default K from the render pinhole
# (per-image scene_camera always wins at load time; this is a fallback only).
width = 1280
height = 720
zNear = 0.01
zFar = 6.5
# fx = focal_mm / aperture_mm * W = 16.6703 / 20.955 * 1280 ≈ 1018.5; cx=640, cy=360
camera_matrix = np.array([[1018.5, 0.0, 640.0], [0.0, 1018.5, 360.0], [0.0, 0.0, 1.0]])


def get_models_info():
    """key is str(obj_id). Carries the symmetries (continuous/discrete) so
    GDRNPP's get_symmetry_transformations resolves the rotation ambiguity."""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    return mmcv.load(models_info_path)


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    return mmcv.load(fps_points_path)


def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    return mmcv.load(keypoints_3d_path)
