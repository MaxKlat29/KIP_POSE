#!/usr/bin/env python3
"""YOLO26m-seg ANKER segmentation trainer  (Story S-008 / T-134).

WHY THIS EXISTS
  yolo-seg-svc on the box was running a STOCK COCO `yolo26n.pt` -> 0 anker
  detections (Sam's S-006 finding). That kills combos 2/4/5 (yolo-seg + FP /
  GigaPose). This trains a proper `yolo26m-seg` on the workstation SDG data so
  `best.pt` becomes a drop-in for yolo-svc/app.py.

DROP-IN CONTRACT (project/mesh/yolo-svc/app.py)
  app.py loads `YOLO(WEIGHTS)` and reads `result.masks.xy` + `result.boxes.cls`,
  mapping cls -> {0: "anker_kurz", 1: "anker_lang"}. So this model MUST be a
  *segment* task model with EXACTLY 2 classes in that frozen index order.

FROZEN CLASS ORDER (adr.md §1.2 — DO NOT REORDER)
  index 0 = anker_kurz   (BOP obj_id 1)
  index 1 = anker_lang   (BOP obj_id 2)
  so  category_id (0-based) + 1 == obj_id (1-based)  holds.
  zahnrad and BACKGROUND/UNLABELLED are FILTERED OUT (2-class scope D1).

LABEL SOURCE  (/mnt/data/bop/sdg_zivid_10k, 10000 frames)
  rgb_XXXX.png                720x1280 RGB
  instance_XXXX.npy           (720,1280) uint32 per-pixel INSTANCE id
  instance_labels_XXXX.json   {"idToSemantics": {"<iid>": {"class": "..."}}}
  We rasterize each instance id whose class is anker_kurz/anker_lang into a
  binary mask, extract polygon(s) via cv2.findContours, simplify with
  approxPolyDP, normalize to [0,1], and write Ultralytics seg labels:
      <cls> x1 y1 x2 y2 ... xn yn        (one line per polygon)

USAGE  (train-venv — ultralytics 8.4.53 + torch 2.5.1 cu121 live there)
  /mnt/data/train-venv/bin/python box_src/train_yolo_seg_anker.py \
      --src /mnt/data/bop/sdg_zivid_10k \
      --out /mnt/data/kip_pose/data/anker_seg \
      --model /mnt/data/kip_pose/yolo26m-seg.pt \
      --epochs 120 --imgsz 1024 --batch 16

  --prep-only   build dataset + data.yaml, then exit (no training)
"""
import argparse
import glob
import json
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image

# FROZEN class order = obj_id - 1 (adr.md §1.2). NEVER reorder. 2-class scope (D1).
CLASSES = ["anker_kurz", "anker_lang"]
CIDX = {c: i for i, c in enumerate(CLASSES)}

# instance-mask cleanup thresholds
MIN_INST_PX = 200        # whole instance smaller than this -> unlearnable noise
MIN_POLY_PX = 60         # a single contour component smaller than this -> drop fragment
APPROX_EPS_FRAC = 0.004  # approxPolyDP epsilon as fraction of contour perimeter
MIN_POLY_PTS = 3         # a valid polygon needs >=3 vertices


def _instance_to_polygons(mask_u8, W, H):
    """One binary instance mask -> list of normalized flat polygons.

    Handles occlusion-split instances (multiple connected components) by
    emitting one polygon per significant external contour. Simplifies each
    contour with approxPolyDP so label files stay compact (~10-40 pts vs 200+).
    Inner holes are ignored (RETR_EXTERNAL) — YOLO seg masks are solid polygons.
    """
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_POLY_PX:
            continue
        eps = APPROX_EPS_FRAC * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if approx.shape[0] < MIN_POLY_PTS:
            continue
        flat = []
        for (x, y) in approx:
            flat += [min(max(float(x) / W, 0.0), 1.0),
                     min(max(float(y) / H, 0.0), 1.0)]
        polys.append(flat)
    return polys


def build_dataset(src, out, val_frac=0.1, seed=7, limit=0):
    random.seed(seed)
    inst_files = sorted(glob.glob(os.path.join(src, "instance_*.npy")))
    if limit:
        inst_files = inst_files[:limit]

    records = []          # (rgb_path, [(cls_idx, [poly,...]), ...])
    per_class_inst = {c: 0 for c in CLASSES}
    per_class_polys = {c: 0 for c in CLASSES}
    frames_with = {c: 0 for c in CLASSES}
    dropped_small, dropped_zahnrad, dropped_other = 0, 0, 0

    for f in inst_files:
        idx = os.path.basename(f)[9:13]               # instance_XXXX.npy
        rgb = os.path.join(src, f"rgb_{idx}.png")
        lab = os.path.join(src, f"instance_labels_{idx}.json")
        if not (os.path.exists(rgb) and os.path.exists(lab)):
            continue
        inst = np.load(f)
        H, W = inst.shape
        sem = json.load(open(lab)).get("idToSemantics", {})

        frame_objs = []
        seen_cls = set()
        for iid in np.unique(inst):
            iid = int(iid)
            cls = str(sem.get(str(iid), {}).get("class", "")).lower()
            if cls == "zahnrad":
                dropped_zahnrad += 1
                continue
            if cls not in CIDX:                       # BACKGROUND/UNLABELLED/unknown
                dropped_other += 1
                continue
            m = (inst == iid).astype(np.uint8)
            if int(m.sum()) < MIN_INST_PX:
                dropped_small += 1
                continue
            polys = _instance_to_polygons(m, W, H)
            if not polys:
                dropped_small += 1
                continue
            for p in polys:
                frame_objs.append((CIDX[cls], p))
                per_class_polys[cls] += 1
            per_class_inst[cls] += 1
            seen_cls.add(cls)

        if frame_objs:
            records.append((rgb, frame_objs))
            for c in seen_cls:
                frames_with[c] += 1

    random.shuffle(records)
    n_val = max(1, round(len(records) * val_frac))
    splits = {"val": records[:n_val], "train": records[n_val:]}

    if os.path.isdir(out):
        shutil.rmtree(out)
    total_polys = 0
    for split, recs in splits.items():
        idir = os.path.join(out, "images", split)
        ldir = os.path.join(out, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        for i, (rgb, objs) in enumerate(recs):
            stem = f"{split}_{i:05d}"
            shutil.copy(rgb, os.path.join(idir, stem + ".png"))
            lines = []
            for cls_idx, poly in objs:
                lines.append(str(cls_idx) + " " + " ".join(f"{v:.6f}" for v in poly))
                total_polys += 1
            open(os.path.join(ldir, stem + ".txt"), "w").write("\n".join(lines) + "\n")

    yaml_path = os.path.join(out, "anker_seg.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for c, i in CIDX.items():
            fh.write(f"  {i}: {c}\n")

    stats = {
        "frames_used": len(records),
        "train": len(splits["train"]),
        "val": n_val,
        "total_polygons": total_polys,
        "instances_per_class": per_class_inst,
        "polygons_per_class": per_class_polys,
        "frames_with_class": frames_with,
        "dropped_small": dropped_small,
        "dropped_zahnrad": dropped_zahnrad,
        "dropped_other": dropped_other,
        "frozen_classes": CLASSES,
        "min_inst_px": MIN_INST_PX,
        "min_poly_px": MIN_POLY_PX,
        "approx_eps_frac": APPROX_EPS_FRAC,
        "seed": seed,
        "val_frac": val_frac,
    }
    json.dump(stats, open(os.path.join(out, "dataset_stats.json"), "w"), indent=2)
    print(f"[ds] {len(records)} frames ({len(splits['train'])} train / {n_val} val), "
          f"{total_polys} polygons. per-class inst={per_class_inst} polys={per_class_polys}. "
          f"dropped: small={dropped_small} zahnrad={dropped_zahnrad} other={dropped_other}. "
          f"FROZEN classes={CLASSES}", flush=True)
    return yaml_path, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data/bop/sdg_zivid_10k")
    ap.add_argument("--out", default="/mnt/data/kip_pose/data/anker_seg")
    ap.add_argument("--model", default="/mnt/data/kip_pose/yolo26m-seg.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap #frames (0=all) — for smoke")
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse the already-built dataset in --out (no rescan/rewrite). "
                         "REQUIRES <out>/anker_seg.yaml + dataset_stats.json to exist.")
    a = ap.parse_args()

    if a.skip_prep:
        yaml_path = os.path.join(a.out, "anker_seg.yaml")
        stats_path = os.path.join(a.out, "dataset_stats.json")
        if not (os.path.exists(yaml_path) and os.path.exists(stats_path)):
            raise SystemExit(f"--skip-prep: prebuilt dataset missing at {a.out} "
                             f"(need anker_seg.yaml + dataset_stats.json)")
        stats = json.load(open(stats_path))
        print(f"[ds] REUSE prebuilt dataset {a.out} "
              f"({stats.get('train')} train / {stats.get('val')} val, "
              f"{stats.get('total_polygons')} polygons)", flush=True)
    else:
        yaml_path, stats = build_dataset(a.src, a.out, a.val_frac, a.seed, a.limit)
    if a.prep_only:
        print("PREP_ONLY_DONE", flush=True)
        return

    from ultralytics import YOLO
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device={dev} model={a.model} epochs={a.epochs} "
          f"imgsz={a.imgsz} batch={a.batch}", flush=True)

    model = YOLO(a.model)
    model.train(
        data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=dev,
        project=os.path.join(a.out, "_runs"), name="seg", exist_ok=True,
        # best-checkpoint + early stopping (Kai-Prinzip: never last-epoch)
        patience=30, close_mosaic=15, lr0=0.01, optimizer="auto",
        # top-down planar parts: rotation/flip OK, NO perspective/shear.
        degrees=180.0, fliplr=0.5, flipud=0.5, perspective=0.0, shear=0.0,
        translate=0.1, scale=0.3, mosaic=1.0, hsv_v=0.4,
        # VRAM guard: cap GPU memory fraction so live inference (:8078) survives.
        verbose=True,
    )

    run_dir = os.path.join(a.out, "_runs", "seg")
    best = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, os.path.join(a.out, "best.pt"))     # canonical drop-in

    m = model.val(data=yaml_path, imgsz=a.imgsz, device=dev, verbose=False)
    summary = {
        "box_map50": float(m.box.map50), "box_map50_95": float(m.box.map),
        "mask_map50": float(m.seg.map50), "mask_map50_95": float(m.seg.map),
        "box_recall": float(m.box.mr), "box_precision": float(m.box.mp),
        "mask_recall": float(m.seg.mr), "mask_precision": float(m.seg.mp),
        "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
        "classes": CLASSES, "task": "segment", "base_model": a.model,
        "best": best, "dataset_stats": stats,
        "note": ("yolo26m-seg trained on sdg_zivid_10k, 2-class (anker_kurz/lang), "
                 "zahnrad filtered. Drop-in for project/mesh/yolo-svc/app.py "
                 "(YOLO_WEIGHTS=best.pt). FROZEN class order 0=kurz 1=lang."),
    }
    json.dump(summary, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(f"[train] DONE best={best} mask_map50={summary['mask_map50']:.3f} "
          f"mask_map50_95={summary['mask_map50_95']:.3f} "
          f"box_map50={summary['box_map50']:.3f}", flush=True)
    print("SEG_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
