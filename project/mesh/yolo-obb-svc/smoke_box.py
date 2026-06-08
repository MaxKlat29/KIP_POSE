#!/usr/bin/env python3
"""Box smoke for yolo-obb-svc — runs the real /segment code path against a real
image with the real detector.pt, then validates the output against CONTRACT.md §2.

Run on the box (train-venv has ultralytics 8.4.x + torch cu121):
  YOLO_OBB_WEIGHTS=/mnt/data/kip_pose/data/detector_armvis/detector.pt \
  /mnt/data/train-venv/bin/python smoke_box.py --image <rgb.png> [--device cpu]

Prints a JSON sample of the response + a PASS/FAIL contract verdict. Does NOT
start a server (avoids port juggling on the shared box); it exercises the exact
segment() function so the smoke == the service behaviour.
"""
import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None, help="cpu to force CPU (keep VRAM 0)")
    a = ap.parse_args()

    if a.device:
        # ultralytics honours predict(device=...); we set it via env-driven app
        # by monkey-patching after import. Simpler: force CUDA hidden for CPU.
        if a.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Import the actual service module so we test the real code path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import app as svc

    t0 = time.time()
    svc.load_model()
    t_load = time.time() - t0

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"FAIL: could not read image {a.image}")
        return 2
    ok, buf = cv2.imencode(".png", img)
    rgb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    t1 = time.time()
    resp = svc.segment(svc.SegmentReq(rgb_b64=rgb_b64))
    t_infer = time.time() - t1

    dets = resp["detections"]
    H, W = img.shape[:2]

    # ---- Contract §2 validation ----
    errors = []
    valid_classes = {"anker_kurz", "anker_lang"}
    for i, d in enumerate(dets):
        p = f"detections[{i}]"
        if d.get("id") != i:
            errors.append(f"{p}.id should be {i}, got {d.get('id')}")
        if d.get("class") not in valid_classes:
            errors.append(f"{p}.class '{d.get('class')}' not in {valid_classes}")
        c = d.get("conf")
        if not (isinstance(c, float) and 0.0 <= c <= 1.0):
            errors.append(f"{p}.conf invalid: {c}")
        if not isinstance(d.get("mask_b64"), str) or not d["mask_b64"]:
            errors.append(f"{p}.mask_b64 missing")
        else:
            m = cv2.imdecode(np.frombuffer(base64.b64decode(d["mask_b64"]), np.uint8),
                             cv2.IMREAD_UNCHANGED)
            if m is None:
                errors.append(f"{p}.mask_b64 not decodable")
            elif m.shape[:2] != (H, W):
                errors.append(f"{p}.mask_b64 shape {m.shape[:2]} != full image {(H, W)}")
            elif not set(np.unique(m)).issubset({0, 255}):
                errors.append(f"{p}.mask_b64 not 0/255 binary")
        obb = d.get("obb")
        if not (isinstance(obb, list) and len(obb) == 5 and all(isinstance(v, float) for v in obb)):
            errors.append(f"{p}.obb must be [cx,cy,w,h,theta] floats, got {obb}")

    # health
    health_ok = svc.health() == {"ok": True}
    if not health_ok:
        errors.append("/health did not return {'ok': True}")

    sample = {
        "n_detections": len(dets),
        "classes": [d["class"] for d in dets],
        "detections_preview": [
            {k: (v if k != "mask_b64" else f"<{len(v)} b64 chars>")
             for k, v in d.items()} for d in dets[:4]
        ],
        "timings_s": {"model_load": round(t_load, 2), "segment": round(t_infer, 3)},
        "image": {"path": a.image, "wh": [W, H]},
        "device": "cpu" if a.device == "cpu" else "cuda-if-available",
    }
    print(json.dumps(sample, indent=2))
    if errors:
        print("\nCONTRACT VERDICT: FAIL")
        for e in errors:
            print("  - " + e)
        return 1
    print("\nCONTRACT VERDICT: PASS — §2 /segment schema satisfied (obb + mask + 2-Anker classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
