#!/usr/bin/env python3
"""Repair existing BOP depth PNGs: stored euclid ray length -> planar Z (T-177).

The Isaac->BOP converter wrote Isaac's distance_to_camera annotator (EUCLIDEAN
ray length) 1:1 into the BOP depth PNGs, but BOP depth is by convention PLANAR
Z. Read-as-planar, the euclid values are +3.4% off-axis (= +40 mm @ 1.1 m),
which radially biased every RGB-D combo in the batch eval (FoundationPose
+41 mm, GigaPose-3D +35 mm radial — matching the 1/cos prediction, corr 0.91).
The converter is fixed (isaac_to_bop.planar_cos_map); this script repairs
datasets that were already converted.

Per scene dir:
  1. SKIP if depth/.planar_repaired marker exists (idempotency guard — running
     twice would double-apply the cos factor).
  2. Backup depth/ -> depth_euclid_orig/ (only created once, never overwritten).
  3. depth/<im>.png *= cos(theta)(u,v) from that frame's cam_K, uint16 round.
  4. Write marker with stats.

Usage:
  python repair_depth_planar.py --dataset /mnt/data/kip_pose/project/bop/pose_isaac \
      [--split val] [--dry-run]
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaac_to_bop import planar_cos_map  # noqa: E402

MARKER = ".planar_repaired"


def repair_scene(scene_dir, dry_run=False):
    depth_dir = os.path.join(scene_dir, "depth")
    marker = os.path.join(depth_dir, MARKER)
    if not os.path.isdir(depth_dir):
        return "no-depth", None
    if os.path.exists(marker):
        return "already-repaired", None

    cam = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    pngs = sorted(f for f in os.listdir(depth_dir) if f.endswith(".png"))
    if not pngs:
        return "empty", None

    backup_dir = os.path.join(scene_dir, "depth_euclid_orig")
    if dry_run:
        return "would-repair", {"frames": len(pngs)}
    if not os.path.isdir(backup_dir):
        shutil.copytree(depth_dir, backup_dir)

    factors = []
    cos_cache = {}
    for f in pngs:
        im_id = str(int(os.path.splitext(f)[0]))
        K = np.asarray(cam[im_id]["cam_K"], np.float64).reshape(3, 3)
        path = os.path.join(depth_dir, f)
        d = np.asarray(Image.open(path), dtype=np.float64)
        key = (tuple(K.reshape(-1)), d.shape)
        if key not in cos_cache:
            cos_cache[key] = planar_cos_map(K, d.shape[1], d.shape[0])
        repaired = np.clip(np.rint(d * cos_cache[key]), 0, 65535).astype(np.uint16)
        Image.fromarray(repaired).save(path)
        nz = d > 0
        if nz.any():
            factors.append(float(cos_cache[key][nz].mean()))

    stats = {
        "applied": True,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frames": len(pngs),
        "mean_cos_factor": float(np.mean(factors)) if factors else None,
        "backup": "depth_euclid_orig/",
        "reason": "T-177 euclid->planar depth convention repair",
    }
    with open(marker, "w") as fh:
        json.dump(stats, fh, indent=1)
    return "repaired", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.join(args.dataset, args.split)
    scenes = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)))
    n_done = 0
    for sd in scenes:
        status, stats = repair_scene(os.path.join(root, sd), dry_run=args.dry_run)
        extra = f" ({stats['frames']} frames)" if stats else ""
        print(f"[{sd}] {status}{extra}")
        n_done += status in ("repaired", "would-repair")
    print(f"\n{n_done}/{len(scenes)} scenes {'would be ' if args.dry_run else ''}repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
