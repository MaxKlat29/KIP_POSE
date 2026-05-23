#!/usr/bin/env python3
"""Arm-VISIBLE YOLOv8-OBB detector retrain  (Story S-401 / T-067, Viktor adr.md §4).

The old detector.pt was trained arm-HIDDEN (mAP50 0.987 on the easier arm-free
distribution). With the LARA5 arm visible (the actual task), the old detector
breaks under occlusion. This retrains on the NEW arm-visible SDG bundle
(sdg_armvis_full, 2000 frames with obb_2d_*.json).

KEY DIFFERENCE from the old build_train_detector.py: the class order is FROZEN to
the global obj_id mapping (anker_kurz=0 ... zahnrad=5) so that
    category_id (detector, 0-based) + 1 == obj_id (BOP, 1-based)   [adr.md §1.2]
holds even if a class is rare/absent in a given epoch's val. The old script
derived classes from the data (sorted(set)) which would silently renumber if a
class were missing -> breaks the BOP detection bridge. We pin it.

Occlusion handling: obb_2d carries per-box `occlusion`. We KEEP arm-occluded
boxes up to --max-occ (default 0.85) so the detector LEARNS to find partly
occluded parts (that is the whole point of the arm-visible retrain). Boxes more
occluded than that are dropped as unlearnable noise. occlusion==-1 (no value)
is treated as 0 (fully visible).

USAGE  (train-venv — ultralytics 8.4.x + torch cu121 live there)
  /mnt/data/train-venv/bin/python box_src/train_detector_armvis.py \
      --src /mnt/data/kip_pose/data/sdg_armvis_full \
      --out /mnt/data/kip_pose/data/detector_armvis \
      --epochs 100 --imgsz 1280 --batch 8 --max-occ 0.85
"""
import argparse
import glob
import json
import os
import random
import shutil

from PIL import Image

# FROZEN class order = obj_id - 1 (adr.md §1.2). NEVER reorder.
CLASSES = ["anker_kurz", "anker_lang", "buerstenhalter_2polig",
           "getriebegehaeuse_typ4", "ringmagnet", "zahnrad"]
CIDX = {c: i for i, c in enumerate(CLASSES)}


def build_dataset(src, out, max_occ, val_frac=0.1, seed=7):
    random.seed(seed)
    records = []
    kept, dropped_occ, dropped_cls = 0, 0, 0
    for rgb in sorted(glob.glob(os.path.join(src, "rgb_*.png"))):
        idx = os.path.basename(rgb)[4:8]
        oj = os.path.join(src, f"obb_2d_{idx}.json")
        if not os.path.exists(oj):
            continue
        boxes = []
        for b in json.load(open(oj)).get("boxes", []):
            cls = str(b.get("class", "")).lower()
            corners = b.get("corners", [])
            occ = float(b.get("occlusion", 0.0))
            if occ < 0:                      # -1 = no value -> fully visible
                occ = 0.0
            if cls not in CIDX:
                dropped_cls += 1
                continue
            if len(corners) != 4:
                continue
            if occ > max_occ:                # too occluded to learn
                dropped_occ += 1
                continue
            boxes.append((cls, corners))
            kept += 1
        if boxes:
            records.append((rgb, boxes))

    random.shuffle(records)
    n_val = max(1, round(len(records) * val_frac))
    splits = {"val": records[:n_val], "train": records[n_val:]}
    if os.path.isdir(out):
        shutil.rmtree(out)
    tot = 0
    for split, recs in splits.items():
        idir = os.path.join(out, "images", split)
        ldir = os.path.join(out, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        for i, (rgb, boxes) in enumerate(recs):
            W, H = Image.open(rgb).size
            stem = f"{split}_{i:04d}"
            shutil.copy(rgb, os.path.join(idir, stem + ".png"))
            lines = []
            for cls, corners in boxes:
                xy = []
                for (x, y) in corners:
                    xy += [min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0)]
                lines.append(str(CIDX[cls]) + " " + " ".join(f"{v:.6f}" for v in xy))
                tot += 1
            open(os.path.join(ldir, stem + ".txt"), "w").write("\n".join(lines) + "\n")

    yaml_path = os.path.join(out, "detect.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for c, i in CIDX.items():
            f.write(f"  {i}: {c}\n")
    print(f"[ds] {len(records)} frames ({len(splits['train'])} train / {n_val} val), "
          f"{tot} OBBs kept; dropped {dropped_occ} (occ>{max_occ}), {dropped_cls} (bad class). "
          f"FROZEN classes={CLASSES}", flush=True)
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default="yolov8s-obb.pt")
    ap.add_argument("--max-occ", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    yaml_path = build_dataset(a.src, a.out, a.max_occ)

    from ultralytics import YOLO
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device={dev} model={a.model} epochs={a.epochs} imgsz={a.imgsz} batch={a.batch}",
          flush=True)
    model = YOLO(a.model)
    model.train(
        data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=dev,
        project=os.path.join(a.out, "_runs"), name="obb", exist_ok=True,
        patience=40, close_mosaic=15, lr0=0.01, optimizer="auto",
        # top-down planar parts: rotation/flip OK, NO perspective/shear.
        degrees=180.0, fliplr=0.5, flipud=0.5, perspective=0.0, shear=0.0,
        translate=0.1, scale=0.3, mosaic=1.0, hsv_v=0.4, verbose=True,
    )
    run_dir = os.path.join(a.out, "_runs", "obb")
    best = os.path.join(run_dir, "weights", "best.pt")
    # publish best.pt as the canonical detector.pt
    if os.path.exists(best):
        shutil.copy(best, os.path.join(a.out, "detector.pt"))
    m = model.val(data=yaml_path, imgsz=a.imgsz, device=dev, verbose=False)
    summary = {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "recall": float(m.box.mr), "precision": float(m.box.mp),
        "epochs": a.epochs, "imgsz": a.imgsz, "max_occ": a.max_occ,
        "classes": CLASSES, "arm_visible": True,
        "best": best, "model": a.model,
        "note": ("arm-VISIBLE retrain; NOT comparable to old arm-free map50=0.987 "
                 "(detect_topdown) — different, harder distribution with arm occlusion."),
    }
    json.dump(summary, open(os.path.join(a.out, "detector.metrics.json"), "w"), indent=2)
    print(f"[train] DONE best={best} map50={summary['map50']:.3f} "
          f"map50_95={summary['map50_95']:.3f} recall={summary['recall']:.3f}", flush=True)
    print("DETECTOR_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
