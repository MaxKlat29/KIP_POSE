#!/usr/bin/env python3
"""OBB -> AABB detection bridge for GDRNPP/BOP  (Story S-401 / T-067, adr.md §4).

Our detector is yolov8s-obb (Oriented Bounding Boxes). GDRNPP (and every BOP
pose stage) consumes the BOP DEFAULT DETECTION FORMAT: a flat JSON list, one
entry per detection, with an AXIS-ALIGNED bbox [x, y, w, h] and category_id =
obj_id (1-based). This module is the pure adapter (no retrain needed for the
format) that:

  1. runs the (arm-visible-retrained) OBB detector on a set of RGB images,
  2. converts each rotated box's 4 corners to the enclosing AABB,
  3. maps detector class (0-based) -> obj_id (1-based) via +1  (adr.md §1.2),
  4. writes the BOP detection json keyed for GDRNPP's DET_FILES_TEST.

It also exposes obb_corners_to_aabb() as a library function so the inference
glue (S-302) can reuse the exact same conversion, and an ADR-017 variant
that emits [x0,y0,x1,y1] for pose_result.bbox_2d.

BOP detection entry (adr.md §4.1):
    { "scene_id": int, "image_id": int, "category_id": obj_id(1-based),
      "bbox": [x,y,w,h], "score": float, "time": float }

USAGE  (train-venv, on the box — ultralytics lives there)
  /mnt/data/train-venv/bin/python box_src/obb_to_aabb_dets.py \
      --weights /mnt/data/kip_pose/data/detector_armvis/detector.pt \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
      --split val \
      --out /mnt/data/kip_pose/project/bop/pose_isaac/val/det_obb2aabb_pose_isaac_val.json \
      --conf 0.1 --imgsz 1280
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


# detector class order (0-based) is FROZEN to match obj_id-1 (adr.md §1.2).
# anker_kurz=0 -> obj_id 1, ... zahnrad=5 -> obj_id 6.
CLASS_TO_OBJ_ID = {i: i + 1 for i in range(6)}


def obb_corners_to_aabb(corners):
    """corners: (4,2) array of (x,y). Returns BOP AABB [x, y, w, h] (top-left+wh)."""
    c = np.asarray(corners, float).reshape(-1, 2)
    x0, y0 = c[:, 0].min(), c[:, 1].min()
    x1, y1 = c[:, 0].max(), c[:, 1].max()
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def aabb_xywh_to_xyxy(bbox):
    """BOP [x,y,w,h] -> ADR-017 pose_result.bbox_2d [x0,y0,x1,y1] (adr.md §3.4)."""
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _xyxyxyxy_from_result(obb):
    """Pull the 4-corner representation out of an ultralytics OBB result, robust
    to version differences (xyxyxyxy may be (N,4,2) or (N,8))."""
    pts = obb.xyxyxyxy
    arr = pts.cpu().numpy() if hasattr(pts, "cpu") else np.asarray(pts)
    if arr.ndim == 3:               # (N,4,2)
        return arr
    return arr.reshape(arr.shape[0], 4, 2)  # (N,8) -> (N,4,2)


def run_detector_to_bop(weights, image_paths, scene_im_ids, conf, imgsz, device=0):
    """image_paths parallel to scene_im_ids=[(scene_id, im_id), ...].
    Returns the BOP detection list."""
    from ultralytics import YOLO
    model = YOLO(weights)
    dets = []
    # ultralytics batches internally; iterate per image to keep scene/im pairing exact
    for (scene_id, im_id), img in zip(scene_im_ids, image_paths):
        res = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            continue
        corners = _xyxyxyxy_from_result(res.obb)            # (N,4,2)
        cls = res.obb.cls.cpu().numpy().astype(int)         # (N,)
        scores = res.obb.conf.cpu().numpy().astype(float)   # (N,)
        spd = float(sum(res.speed.values())) / 1000.0 if hasattr(res, "speed") else 0.0
        for k in range(len(cls)):
            obj_id = CLASS_TO_OBJ_ID.get(int(cls[k]))
            if obj_id is None:
                continue
            dets.append({
                "scene_id": int(scene_id),
                "image_id": int(im_id),
                "category_id": int(obj_id),
                "bbox": [round(v, 2) for v in obb_corners_to_aabb(corners[k])],
                "score": round(float(scores[k]), 4),
                "time": round(spd, 4),
            })
    return dets


def collect_split_images(bop_root, split):
    """Return (image_paths, scene_im_ids) over every rgb image in a BOP split."""
    paths, ids = [], []
    sdir = os.path.join(bop_root, split)
    for scene in sorted(os.listdir(sdir)):
        rgb_dir = os.path.join(sdir, scene, "rgb")
        if not os.path.isdir(rgb_dir):
            continue
        sid = int(scene)
        for p in sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                        glob.glob(os.path.join(rgb_dir, "*.jpg"))):
            paths.append(p)
            ids.append((sid, int(os.path.splitext(os.path.basename(p))[0])))
    return paths, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="retrained yolov8-obb detector.pt")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    paths, ids = collect_split_images(args.bop_root, args.split)
    if not paths:
        print(f"[obb2aabb] no images in {args.bop_root}/{args.split}", file=sys.stderr)
        sys.exit(1)
    print(f"[obb2aabb] {len(paths)} images in {args.split}; running OBB detector...", flush=True)
    dev = int(args.device) if args.device.isdigit() else args.device
    dets = run_detector_to_bop(args.weights, paths, ids, args.conf, args.imgsz, dev)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dets, open(args.out, "w"))
    by_obj = {}
    for d in dets:
        by_obj[d["category_id"]] = by_obj.get(d["category_id"], 0) + 1
    print(f"[obb2aabb] wrote {len(dets)} BOP detections -> {args.out}", flush=True)
    print(f"[obb2aabb] per obj_id: {dict(sorted(by_obj.items()))}", flush=True)
    print("OBB2AABB_DONE", flush=True)


if __name__ == "__main__":
    main()
