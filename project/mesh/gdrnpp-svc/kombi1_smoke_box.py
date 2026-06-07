#!/usr/bin/env python3
"""Kombi-1 wire smoke (yolo-obb → gdrnpp), real models, on the box.

Reproduces the gateway's Pipeline-A composition end to end against the REAL
services, without needing the dockerised gateway up:
  1. run the REAL yolo-obb detector (train-venv ultralytics) on a real image ->
     contract obb detections [{id,class,conf,obb:[cx,cy,w,h,theta]}],
  2. apply the gateway gating (gdrnpp ⇔ yolo-obb — satisfied here),
  3. build /pose instances WITH obb, POST to the REAL gdrnpp-svc:8012,
  4. assert a valid T_cam_obj comes back (OpenCV cam, metres, mesh→cam).

This is the exact obb→GDRNPP wire path the gateway drives for combo 1. It does
NOT touch the live :8078 worker.

Run on the box:
  /mnt/data/bop/gdrnpp-venv/bin/python kombi1_smoke_box.py \
     --rgb /mnt/data/kip_pose/project/bop/pose_isaac/val/000002/rgb/000000.png
(the detector subprocess uses train-venv; gdrnpp-venv runs this driver + requests)
"""
import argparse
import base64
import json
import os
import subprocess
import sys

import numpy as np
import requests

BOX_SRC = "/mnt/data/kip_pose/box_src"
TRAIN_VENV = "/mnt/data/train-venv/bin/python"
DETECTOR = "/mnt/data/kip_pose/data/detector_armvis/detector.pt"
VAL = "/mnt/data/kip_pose/project/bop/pose_isaac/val"

# detector class id (0-based) -> mesh class (lowercase). D1: 2 Anker only.
CLS_TO_NAME = {0: "anker_kurz", 1: "anker_lang"}


def run_yolo_obb(rgb_path):
    """Run the real OBB detector -> contract detections with the `obb` field.
    Mirrors yolo-obb-svc (xywhr -> [cx,cy,w,h,theta]; class id>=2 filtered)."""
    code = (
        "import sys,json;sys.path.insert(0,'%s')\n"
        "from ultralytics import YOLO\n"
        "m=YOLO('%s')\n"
        "r=m.predict('%s',conf=0.40,imgsz=1280,device=0,verbose=False)[0]\n"
        "out=[]\n"
        "if r.obb is not None and len(r.obb):\n"
        "  xywhr=r.obb.xywhr.cpu().numpy();cls=r.obb.cls.cpu().numpy().astype(int);conf=r.obb.conf.cpu().numpy()\n"
        "  names={0:'anker_kurz',1:'anker_lang'}\n"
        "  for i in range(len(cls)):\n"
        "    c=int(cls[i])\n"
        "    if c not in names: continue\n"
        "    cx,cy,w,h,th=[float(v) for v in xywhr[i]]\n"
        "    out.append({'id':i,'class':names[c],'conf':round(float(conf[i]),4),'obb':[cx,cy,w,h,th]})\n"
        "print('JSON'+json.dumps(out))"
    ) % (BOX_SRC, DETECTOR, rgb_path)
    r = subprocess.run([TRAIN_VENV, "-c", code], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"detector failed: {r.stderr[-400:]}")
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")][-1]
    return json.loads(line[4:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default=f"{VAL}/000002/rgb/000000.png")
    ap.add_argument("--svc", default="http://localhost:8012")
    args = ap.parse_args()

    # 1. real OBB detector -> detections with obb (the yolo-obb seg stage)
    dets = run_yolo_obb(args.rgb)
    print(f"[kombi1] yolo-obb: {len(dets)} detection(s) "
          f"{[(d['class'], round(d['conf'],2)) for d in dets]}")
    if not dets:
        print("[kombi1] FAIL: detector produced no anker detections")
        sys.exit(1)

    # camera K (use the scene's; for an arbitrary upload the gateway takes fx/fy/cx/cy)
    scene = os.path.normpath(args.rgb).split(os.sep)[-3]
    cam = json.load(open(f"{VAL}/{scene}/scene_camera.json"))["0"]
    K = [float(x) for x in cam["cam_K"]]
    rgb_b64 = base64.b64encode(open(args.rgb, "rb").read()).decode()

    # 2/3. gateway gating (gdrnpp ⇔ yolo-obb satisfied) + build /pose WITH obb
    instances = [{"id": d["id"], "class": d["class"], "obb": d["obb"]} for d in dets]
    req = {"rgb_b64": rgb_b64, "K": K, "iterations": 5, "instances": instances}
    r = requests.post(f"{args.svc}/pose", json=req, timeout=180).json()
    poses = r.get("poses") or []
    print(f"[kombi1] gdrnpp-svc: {len(poses)} pose(s) for {len(instances)} instance(s)")

    # 4. validate each T_cam_obj is a proper SE(3) in the cam frame (z forward, metres)
    ok = bool(poses)
    for p in poses:
        T = np.asarray(p["T_cam_obj"], float)
        R = T[:3, :3]
        t = T[:3, 3]
        det = float(np.linalg.det(R))
        orth = float(np.linalg.norm(R @ R.T - np.eye(3)))
        z_fwd = t[2] > 0
        depth_m = float(t[2])
        sane = abs(det - 1.0) < 1e-3 and orth < 1e-3 and z_fwd and 0.1 < depth_m < 3.0
        ok &= sane
        print(f"  id={p['id']} {p['class']}: t_m={[round(v,3) for v in t]} "
              f"det(R)={det:.4f} orth_err={orth:.1e} depth={depth_m:.3f}m  "
              f"{'OK' if sane else 'BAD'}")

    print(f"[kombi1] VERDICT: {'PASS' if ok else 'FAIL'} — "
          f"real yolo-obb → gdrnpp produced valid T_cam_obj")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
