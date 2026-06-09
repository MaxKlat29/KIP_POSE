#!/usr/bin/env python3
"""Round-trip verification: gdrnpp-svc:8012 (contract /pose) vs the live :8078 worker.

Both wrap the SAME GDRNPP checkpoints. Feeding the same RGB + K + detection box to
both must yield the same T_cam_obj (within tolerance) — proving the new service does
not regress the live pose. Runs ON THE BOX (reads BOP val data + base64-encodes RGB).

Usage (on the box, any python3 with numpy/requests; the gdrnpp-venv has both):
  /mnt/data/bop/gdrnpp-venv/bin/python roundtrip_box.py [--scene 0] [--im 0]

It:
  1. asks the live :8078 worker to pose <scene>/<im> (its full det-driven path),
  2. for each posed obj_id, builds a contract /pose request to :8012 using the SAME
     RGB + scene K + a det box (the GT bbox of that instance — same box both sides
     effectively see via the live worker's own GT-bbox det), and compares T_cam_obj.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import requests

VAL = "/mnt/data/kip_pose/project/bop/pose_isaac/val"
OBJ_TO_CLASS = {1: "anker_kurz", 2: "anker_lang"}  # D1; obj 6 (zahnrad) out-of-scope


def _bbox_from_gt_info(scene, im, instance_idx):
    gi = json.load(open(f"{VAL}/{scene:06d}/scene_gt_info.json"))[str(im)]
    return gi[instance_idx]["bbox_visib"]  # [x,y,w,h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--im", type=int, default=0)
    ap.add_argument("--svc", default="http://localhost:8012")
    ap.add_argument("--live", default="http://localhost:8078")
    args = ap.parse_args()

    sdir = f"{VAL}/{args.scene:06d}"
    rgb_path = f"{sdir}/rgb/{args.im:06d}.png"
    if not os.path.exists(rgb_path):
        rgb_path = f"{sdir}/rgb/{args.im:06d}.jpg"
    rgb_b64 = base64.b64encode(open(rgb_path, "rb").read()).decode()
    cam = json.load(open(f"{sdir}/scene_camera.json"))[str(args.im)]
    K = [float(x) for x in cam["cam_K"]]
    gt = json.load(open(f"{sdir}/scene_gt.json"))[str(args.im)]

    # 1. live worker poses the whole scene/im (its own det-driven path).
    live = requests.get(f"{args.live}/infer",
                        params={"scene": args.scene, "im": args.im}, timeout=120).json()
    live_preds = live.get("preds") or []
    # group live preds by obj_id, in order
    live_by_obj = {}
    for p in live_preds:
        live_by_obj.setdefault(p["obj_id"], []).append(p)

    print(f"[roundtrip] scene {args.scene}/{args.im}: live posed obj_ids "
          f"{sorted({p['obj_id'] for p in live_preds})}, {len(live_preds)} preds")

    # 2. for each in-scope obj instance in the GT, build a /pose request to :8012 with
    #    that instance's GT bbox as the det box, compare against the live pred.
    results = []
    for obj_id in sorted(OBJ_TO_CLASS):
        if obj_id not in live_by_obj:
            continue
        # find the first GT instance of this obj_id -> its bbox
        idxs = [i for i, o in enumerate(gt) if o["obj_id"] == obj_id]
        if not idxs:
            continue
        bbox = _bbox_from_gt_info(args.scene, args.im, idxs[0])
        cls = OBJ_TO_CLASS[obj_id]
        # contract /pose with a mask-less obb-equivalent: gdrnpp reads obb; here we hand
        # it the axis-aligned GT bbox AS an obb with theta=0 (cx,cy,w,h,0) so the
        # service's obb->AABB returns exactly this box, matching what the live worker's
        # det box is for this instance.
        x, y, w, h = bbox
        obb = [x + w / 2.0, y + h / 2.0, float(w), float(h), 0.0]
        req = {"rgb_b64": rgb_b64, "K": K, "iterations": 5,
               "instances": [{"id": 0, "class": cls, "obb": obb}]}
        r = requests.post(f"{args.svc}/pose", json=req, timeout=120).json()
        svc_poses = r.get("poses") or []
        if not svc_poses:
            print(f"  obj {obj_id} ({cls}): svc returned NO pose -> FAIL")
            results.append((obj_id, None))
            continue
        T = np.asarray(svc_poses[0]["T_cam_obj"], float)
        svc_R = T[:3, :3]
        svc_t_mm = T[:3, 3] * 1000.0  # back to mm for comparison
        live_p = live_by_obj[obj_id][0]
        live_R = np.asarray(live_p["R"], float).reshape(3, 3)
        live_t_mm = np.asarray(live_p["t"], float)

        d_t = float(np.linalg.norm(svc_t_mm - live_t_mm))
        # geodesic rotation distance (deg)
        Rrel = svc_R @ live_R.T
        cos = (np.trace(Rrel) - 1.0) / 2.0
        d_R = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        results.append((obj_id, (d_t, d_R, svc_t_mm, live_t_mm)))
        print(f"  obj {obj_id} ({cls}): "
              f"svc t_mm={[round(v,1) for v in svc_t_mm]}  "
              f"live t_mm={[round(v,1) for v in live_t_mm]}  "
              f"Δt={d_t:.2f}mm  ΔR={d_R:.3f}°")

    # verdict
    ok = all(r[1] is not None and r[1][0] < 5.0 and r[1][1] < 1.0 for r in results) and results
    print(f"[roundtrip] VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(tol Δt<5mm ΔR<1°; same model+box must reproduce)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
