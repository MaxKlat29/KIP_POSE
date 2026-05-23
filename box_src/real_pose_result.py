#!/usr/bin/env python3
"""real_pose_result.py - REAL end-to-end pose_result.json from trained GDRNPP.

NO MOCK. This drives the *real* chain output through the *real* bop_adapter:

  trained GDRNPP (model_final.pth) inference  ->  BOP predictions CSV (R_m2c,t_m2c)
       +  real scene_camera.json (cam_R_w2c, cam_t_w2c, cam_K)
       ->  project/bop_adapter.detection_to_result (the production adapter)
       ->  schema-valid pose_result.json

GDRNPP runs per-object SO models and already wrote predictions for the val split
(output/.../<obj>/inference_/.../*.csv, combined to preds_all.csv). Those poses
ARE the trained-network output (TEST_BBOX_TYPE=gt inference, real weights). We
take one val frame, pull its real predictions, and run them through the same
adapter the laptop e2e_infer.py uses -> a real pose_result the viewer renders.

Also renders the detection overlay (GT bbox of each trained object on the RGB)
so the 2D side of the split-screen is a real image with real boxes + the arm.

Usage (on the box, bop-venv):
  python real_pose_result.py --scene 0 --im 92 \
     --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
     --preds /mnt/data/bop/results/preds_all.csv \
     --out-json /mnt/data/bop/results/pose_result.json \
     --out-overlay /mnt/data/bop/results/det_overlay.png
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# the PRODUCTION adapter (same module the laptop e2e_infer.py imports)
sys.path.insert(0, "/mnt/data/kip_pose/project")
import bop_adapter as BOP  # noqa: E402

SCHEMA_VERSION = "1.0.0"
COORD_CONVENTION = ("Z-up world; column rotation world = R @ body; "
                    "origin = table-plane null-point")
# same table origin the laptop pipeline uses (e2e_infer.TABLE_ORIGIN_SCENE)
TABLE_ORIGIN_SCENE = (0.0, 0.0, 0.08)

# distinct overlay colors per obj_id
OBJ_COLORS = {
    1: (66, 135, 245), 2: (52, 199, 89), 3: (255, 159, 10),
    4: (191, 90, 242), 5: (255, 214, 10), 6: (255, 69, 58),
}


def load_preds_for_frame(preds_csv, scene_id, im_id):
    """Return list of {obj_id, score, R(3x3), t(3,) mm} for this exact frame."""
    from bop_toolkit_lib import inout
    raw = inout.load_bop_results(preds_csv, version="bop19")
    out = []
    for r in raw:
        if r["scene_id"] == scene_id and r["im_id"] == im_id:
            out.append({
                "obj_id": int(r["obj_id"]),
                "score": float(r["score"]),
                "R": np.array(r["R"], dtype=np.float64).reshape(3, 3),
                "t": np.array(r["t"], dtype=np.float64).reshape(3, 1),
            })
    return out


def load_gt_bboxes(dataset_dir, split, scene_id, im_id):
    """GT bbox per instance from scene_gt_info.json (for the det overlay)."""
    scene_dir = os.path.join(dataset_dir, split, f"{scene_id:06d}")
    info = json.load(open(os.path.join(scene_dir, "scene_gt_info.json")))
    gt = json.load(open(os.path.join(scene_dir, "scene_gt.json")))
    insts = gt[str(im_id)]
    binfo = info[str(im_id)]
    boxes = []
    for inst, bi in zip(insts, binfo):
        x, y, w, h = bi["bbox_visib"]
        boxes.append({"obj_id": int(inst["obj_id"]),
                      "bbox": [int(x), int(y), int(x + w), int(y + h)],
                      "visib": float(bi.get("visib_fract", 1.0))})
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--im", type=int, required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-overlay", required=True)
    a = ap.parse_args()

    scene_dir = os.path.join(a.dataset_dir, a.split, f"{a.scene:06d}")
    cam_all = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    cam = cam_all[str(a.im)]
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    R_w2c = np.array(cam["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.array(cam["cam_t_w2c"], dtype=np.float64).reshape(3)  # mm

    rgb_path = os.path.join(scene_dir, "rgb", f"{a.im:06d}.png")
    img = Image.open(rgb_path).convert("RGB")

    preds = load_preds_for_frame(a.preds, a.scene, a.im)
    # only trained objects have a model -> predictions exist only for 1,2,6
    trained = {p["obj_id"] for p in preds}
    sys.stderr.write(
        f"[real_pose] frame scene={a.scene} im={a.im}: {len(preds)} real GDRNPP "
        f"predictions for obj_ids {sorted(trained)}\n")

    # --- REAL chain: GDRNPP pose -> production bop_adapter -> pose_result ---
    results = []
    for i, p in enumerate(preds):
        r = BOP.detection_to_result(
            instance_id=i,
            obj_id=p["obj_id"],
            R_m2c=p["R"],
            t_m2c_mm=p["t"],
            R_w2c=R_w2c,
            t_w2c_mm=t_w2c,
            table_origin_m=TABLE_ORIGIN_SCENE,
            bbox_2d=[0, 0, 1, 1],  # placeholder; replaced from GT-info below
            confidence=p["score"],
        )
        results.append((p["obj_id"], r))

    # attach the real detection bbox (GT-info visib bbox per matched obj instance)
    gt_boxes = load_gt_bboxes(a.dataset_dir, a.split, a.scene, a.im)
    # greedy: give each result the next unused GT bbox of its obj_id
    used = set()
    for obj_id, r in results:
        for bi, gb in enumerate(gt_boxes):
            if bi in used or gb["obj_id"] != obj_id:
                continue
            r["bbox_2d"] = gb["bbox"]
            used.add(bi)
            break

    doc = {
        "meta": {
            "source_image": rgb_path,
            "table_origin": [float(v) for v in TABLE_ORIGIN_SCENE],
            "units": "m",
            "coordinate_convention": COORD_CONVENTION,
            "schema_version": SCHEMA_VERSION,
            "pose_backend": "GDRNPP",
            "checkpoint_note": "model_final.pth per-object SO (anker_kurz/anker_lang/zahnrad)",
            "frame": {"scene_id": a.scene, "im_id": a.im, "split": a.split},
        },
        "results": [r for _, r in results],
    }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(doc, open(a.out_json, "w"), indent=2)
    sys.stderr.write(f"[real_pose] wrote {len(doc['results'])} REAL poses -> {a.out_json}\n")

    # --- detection overlay: GT-visib boxes of trained objects + arm scene ---
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for gb in gt_boxes:
        if gb["obj_id"] not in trained:
            continue
        x0, y0, x1, y1 = gb["bbox"]
        col = OBJ_COLORS.get(gb["obj_id"], (255, 255, 255))
        draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f"{BOP.part_for_obj_id(gb['obj_id'])} v{gb['visib']:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x0, y0 - 22, x0 + tw + 6, y0], fill=col)
        draw.text((x0 + 3, y0 - 21), label, fill=(0, 0, 0), font=font)
    img.save(a.out_overlay)
    sys.stderr.write(f"[real_pose] wrote detection overlay -> {a.out_overlay}\n")

    # echo to stdout for the laptop
    print(json.dumps({
        "n_results": len(doc["results"]),
        "obj_ids": sorted(trained),
        "results": [{"part": r["part"], "face": r["face"],
                     "t_world": [round(v, 3) for v in r["t_world"]],
                     "upright": r["upright"], "conf": round(r["confidence"], 2)}
                    for _, r in results],
    }, indent=2))


if __name__ == "__main__":
    main()
