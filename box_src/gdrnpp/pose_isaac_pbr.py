"""GDRNPP dataset loader for the Isaac arm-visible BOP dataset `pose_isaac`.

Adapted from core/gdrn_modeling/datasets/itodd_pbr.py. Deployed to
core/gdrn_modeling/datasets/pose_isaac_pbr.py and registered in dataset_factory.py.

Differences from itodd_pbr:
  * rgb is PNG (our converter writes .png; itodd uses .jpg)
  * scenes are DISCOVERED from the split dir (not hardcoded range(50))
  * ref_key = "pose_isaac"; both train_pbr and val splits registered
  * height/width = 720/1280 (our top-down render)
  * single-obj splits auto-generated for per-object SO training (adr.md §5-R1)

RGB-only: with_depth loads the depth PATH (used by xyz preprocessing) but the
GDRN net runs in_chans=3 with DEPTH_BACKBONE disabled.
"""
import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint, lazy_property

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


def _discover_scenes(split_root):
    if not osp.isdir(split_root):
        return []
    return sorted(
        d for d in os.listdir(split_root)
        if osp.isdir(osp.join(split_root, d)) and d.isdigit()
    )


class POSE_ISAAC_PBR_Dataset:
    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg
        self.objs = data_cfg["objs"]
        self.dataset_root = data_cfg["dataset_root"]
        self.xyz_root = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop"))
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]
        self.scale_to_meter = data_cfg["scale_to_meter"]
        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]
        self.height = data_cfg["height"]
        self.width = data_cfg["width"]
        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        self.ref_key = data_cfg["ref_key"]
        self.data_ref = ref.__dict__[self.ref_key]

        self.cat_ids = [cid for cid, oname in self.data_ref.id2obj.items() if oname in self.objs]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}
        self.label2cat = {l: c for c, l in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, i) for i, obj in enumerate(self.objs))

        self.scenes = _discover_scenes(self.dataset_root)

    def __call__(self):
        hashed = hashlib.md5(
            ("".join(str(fn) for fn in self.objs)
             + "dd_{}_{}_{}_{}_{}".format(self.name, self.dataset_root,
                                          self.with_masks, self.with_depth, __name__)
             ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed))
        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []
        for scene in tqdm(self.scenes):
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in gt_dict:
                int_im_id = int(str_im_id)
                rgb_path = osp.join(scene_root, "rgb/{:06d}.png".format(int_im_id))
                assert osp.exists(rgb_path), rgb_path
                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))
                scene_im_id = f"{scene_id}/{int_im_id}"
                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr",
                }
                insts = []
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    cur_label = self.cat2label[obj_id]
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(R).astype("float32")
                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid and (h <= 1 or w <= 1):
                        self.num_instances_without_valid_box += 1
                        continue

                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    mask_single = mmcv.imread(mask_visib_file, "unchanged").astype("bool")
                    if mask_single.sum() < 30:
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                    mask_full = mmcv.imread(mask_file, "unchanged").astype("bool")
                    mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)
                    visib_fract = gt_info_dict[str_im_id][anno_i].get("visib_fract", 1.0)
                    xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")

                    inst = {
                        "category_id": cur_label,
                        "bbox": bbox_visib,
                        "bbox_obj": bbox_obj,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": t,
                        "centroid_2d": proj,
                        "segmentation": mask_rle,
                        "mask_full": mask_full_rle,
                        "visib_fract": visib_fract,
                        "xyz_path": xyz_path,
                    }
                    inst["model_info"] = self.models_info[str(obj_id)]
                    inst["bbox3d_and_center"] = self.models[cur_label]["bbox3d_and_center"]
                    insts.append(inst)
                if len(insts) == 0:
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning("Filtered {} insts w/o valid segm.".format(self.num_instances_without_valid_segmentation))
        if self.num_instances_without_valid_box > 0:
            logger.warning("Filtered {} insts w/o valid box.".format(self.num_instances_without_valid_box))
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dicts in {:.2f}s".format(len(dataset_dicts), time.perf_counter() - t_start))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        return dataset_dicts

    @lazy_property
    def models_info(self):
        p = osp.join(self.models_root, "models_info.json")
        assert osp.exists(p), p
        return mmcv.load(p)

    @lazy_property
    def models(self):
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            return mmcv.load(cache_path)
        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{self.data_ref.obj2id[obj_name]:06d}.ply"),
                vertex_scale=self.scale_to_meter,
            )
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])
            models.append(model)
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height


def get_pose_isaac_metadata(obj_names, ref_key):
    data_ref = ref.__dict__[ref_key]
    cur_sym_infos = {}
    loaded = data_ref.get_models_info()
    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        mi = loaded[str(obj_id)]
        if "symmetries_discrete" in mi or "symmetries_continuous" in mi:
            syms = misc.get_symmetry_transformations(mi, max_sym_disc_step=0.01)
            cur_sym_infos[i] = np.array([s["R"] for s in syms], dtype=np.float32)
        else:
            cur_sym_infos[i] = None
    return {"thing_classes": obj_names, "sym_infos": cur_sym_infos}


def _cfg_for(name, objs, split):
    split_dir = "train_pbr" if split == "train_pbr" else "val"
    droot = osp.join(DATASETS_ROOT, "BOP_DATASETS/pose_isaac", split_dir)
    return dict(
        name=name, objs=objs,
        dataset_root=droot,
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/pose_isaac/models"),
        xyz_root=osp.join(droot, "xyz_crop"),
        scale_to_meter=0.001,
        with_masks=True, with_depth=True,
        height=720, width=1280,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True, num_to_load=-1, filter_invalid=True,
        ref_key="pose_isaac",
    )


SPLITS_POSE_ISAAC = {}
# all-objects splits
SPLITS_POSE_ISAAC["pose_isaac_train_pbr"] = _cfg_for("pose_isaac_train_pbr", ref.pose_isaac.objects, "train_pbr")
SPLITS_POSE_ISAAC["pose_isaac_val"] = _cfg_for("pose_isaac_val", ref.pose_isaac.objects, "val")
# single-obj splits (per-object SO training + its val)
for _obj in ref.pose_isaac.objects:
    SPLITS_POSE_ISAAC[f"pose_isaac_{_obj}_train_pbr"] = _cfg_for(f"pose_isaac_{_obj}_train_pbr", [_obj], "train_pbr")
    SPLITS_POSE_ISAAC[f"pose_isaac_{_obj}_val"] = _cfg_for(f"pose_isaac_{_obj}_val", [_obj], "val")


def register_with_name_cfg(name, data_cfg=None):
    dprint("register dataset: {}".format(name))
    used_cfg = SPLITS_POSE_ISAAC[name] if name in SPLITS_POSE_ISAAC else data_cfg
    assert used_cfg is not None, f"dataset name {name} is not registered"
    DatasetCatalog.register(name, POSE_ISAAC_PBR_Dataset(used_cfg))
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"], objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"], evaluator_type="bop",
        **get_pose_isaac_metadata(used_cfg["objs"], used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_POSE_ISAAC.keys())
