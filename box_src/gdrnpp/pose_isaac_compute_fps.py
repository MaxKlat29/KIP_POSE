# compute fps (farthest point sampling) for pose_isaac models
# adapted from tools/itodd/itodd_1_compute_fps.py — only ref_key swapped.
import os.path as osp
import sys
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../.."))
import mmcv
from lib.pysixd import inout, misc  # noqa
import ref
from core.utils.data_utils import get_fps_and_center

ref_key = "pose_isaac"
data_ref = ref.__dict__[ref_key]
model_dir = data_ref.model_dir
id2obj = data_ref.id2obj


def main():
    vertex_scale = 0.001
    fps_dict = {}
    for obj_id in tqdm(id2obj):
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        fps_dict[str(obj_id)] = {}
        for n in (4, 8, 12, 16, 20, 32, 64, 128, 256):
            fps_dict[str(obj_id)][f"fps{n}_and_center"] = get_fps_and_center(
                model["pts"], num_fps=n, init_center=True
            )
    save_path = osp.join(model_dir, "fps_points.pkl")
    mmcv.dump(fps_dict, save_path)
    print(f"saved fps -> {save_path}")


if __name__ == "__main__":
    main()
