# compute bbox3d + fps keypoints for pose_isaac models
# adapted from tools/itodd/itodd_1_compute_keypoints_3d.py — only ref_key swapped.
import os.path as osp
import sys
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../.."))
import mmcv
from lib.pysixd import inout, misc
import ref
from core.utils.data_utils import get_fps_and_center

ref_key = "pose_isaac"
data_ref = ref.__dict__[ref_key]
model_dir = data_ref.model_dir
id2obj = data_ref.id2obj


def main():
    vertex_scale = 0.001
    kpts3d_dict = {}
    for obj_id in tqdm(id2obj):
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        kpts3d_dict[str(obj_id)] = {"bbox3d_and_center": misc.get_bbox3d_and_center(model["pts"])}
        for n in (4, 8, 12, 16, 20, 32, 64, 128, 256):
            kpts3d_dict[str(obj_id)][f"fps{n}_and_center"] = get_fps_and_center(
                model["pts"], num_fps=n, init_center=True
            )
    save_path = osp.join(model_dir, "keypoints_3d.pkl")
    mmcv.dump(kpts3d_dict, save_path)
    print(f"saved keypoints_3d -> {save_path}")


if __name__ == "__main__":
    main()
