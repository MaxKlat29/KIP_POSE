#!/usr/bin/env python3
"""Full Isaac SDG bundle (2000 arm-visible frames) -> BOP dataset with a clean
train/val split  (Story S-202 / T-061, follow-up to the S-201 small-batch).

WHAT IT DOES
  * Deterministically shuffles the 2000 raw frames (seed) into ~90/10 train/val.
  * Groups TRAIN frames into multiple BOP scenes of ~SCENE_SIZE frames each
    (BOP train_pbr convention: many scene dirs 000000, 000001, ...). VAL frames
    likewise go into scenes under the `val` split.
  * Calls isaac_to_bop.py once per scene (via --frame-list) so each scene gets
    its own contiguous 6-digit im-ids (000000.png ...). Scene/im ids are clean
    and non-overlapping within a split.
  * Emits camera.json (default intrinsics fallback) + dataset_info.json at the
    BOP root, and a GDRNPP-style test_targets json for the val split (so the
    pose eval has explicit targets).

WHY MULTI-SCENE
  Detectors don't care, but GDRNPP's pbr loader iterates scene dirs, and the BOP
  toolkit / eval addresses GT by (scene_id, im_id). A single 2000-frame scene
  works too, but multi-scene keeps per-scene json files small and matches the
  layout every BOP repo expects. Splitting at the SCENE level (not per-frame
  inside one scene) also gives a clean train/val boundary the loader honours.

USAGE  (bop-venv, on the box)
  /mnt/data/bop/bop-venv/bin/python box_src/convert_full_to_bop.py \
      --raw-dir   /mnt/data/kip_pose/data/sdg_armvis_full \
      --bop-root  /mnt/data/kip_pose/project/bop/pose_isaac \
      --val-frac 0.1 --scene-size 100 --seed 7 --fresh

  --fresh wipes existing train_pbr/ and val/ first (the S-201 small batch).
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "isaac_to_bop.py")


def discover_frames(raw_dir):
    fs = sorted(glob.glob(os.path.join(raw_dir, "gt_raw_*.json")))
    return [int(os.path.basename(f)[7:11]) for f in fs]


def chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def run_scene(py, raw_dir, bop_root, split, scene_id, frames, depth_scale, min_visib=0.0):
    cmd = [
        py, CONVERTER,
        "--raw-dir", raw_dir,
        "--bop-root", bop_root,
        "--split", split,
        "--scene-id", str(scene_id),
        "--start-im", "0",
        "--depth-scale", str(depth_scale),
        "--frame-list", ",".join(str(i) for i in frames),
        "--min-visib", str(min_visib),
    ]
    print(f"[convert_full] -> {split}/{scene_id:06d}  ({len(frames)} frames)", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[convert_full] scene {split}/{scene_id:06d} FAILED rc={r.returncode}",
              file=sys.stderr)
        sys.exit(r.returncode)


def write_camera_json(raw_dir, bop_root):
    """Default intrinsics fallback (per-image scene_camera always wins). Derived
    from the first frame's pinhole params via the same formula isaac_to_bop uses."""
    f0 = sorted(glob.glob(os.path.join(raw_dir, "gt_raw_*.json")))[0]
    g = json.load(open(f0))
    W, H = g["width"], g["height"]
    fx = g["focal_length_mm"] / g["horizontal_aperture_mm"] * W
    cam = {"cx": W / 2.0, "cy": H / 2.0, "depth_scale": 0.1,
           "fx": fx, "fy": fx, "height": H, "width": W}
    json.dump(cam, open(os.path.join(bop_root, "camera.json"), "w"), indent=2)


def write_dataset_info(bop_root, n_train_scenes, n_val_scenes, n_train, n_val):
    info = {
        "name": "pose_isaac",
        "description": "Isaac SDG top-down, LARA5 arm visible as occluder. RGB-only.",
        "obj_ids": [1, 2, 3, 4, 5, 6],
        "obj_names": {"1": "Anker_Kurz", "2": "Anker_Lang", "3": "Buerstenhalter_2polig",
                      "4": "Getriebegehaeuse_typ4", "5": "Ringmagnet", "6": "Zahnrad"},
        "splits": {"train_pbr": {"scenes": n_train_scenes, "frames": n_train},
                   "val": {"scenes": n_val_scenes, "frames": n_val}},
        "depth_scale_note": "depth_png(uint16) * 0.1 = mm",
        "modality": "rgb",  # depth present but NOT fed to the net (RGB-only, Max HARD)
    }
    json.dump(info, open(os.path.join(bop_root, "dataset_info.json"), "w"), indent=2)


def write_val_targets(bop_root, val_split="val"):
    """GDRNPP/bop test_targets: one row per (scene, im, obj_id) instance in val."""
    targets = []
    sdir = os.path.join(bop_root, val_split)
    for scene in sorted(os.listdir(sdir)):
        sp = os.path.join(sdir, scene)
        gt = json.load(open(os.path.join(sp, "scene_gt.json")))
        sid = int(scene)
        for im, insts in gt.items():
            counts = {}
            for a in insts:
                counts[a["obj_id"]] = counts.get(a["obj_id"], 0) + 1
            for oid, c in counts.items():
                targets.append({"im_id": int(im), "inst_count": c,
                                "obj_id": int(oid), "scene_id": sid})
    out = os.path.join(bop_root, "test_targets_pose_isaac_val.json")
    json.dump(targets, open(out, "w"))
    print(f"[convert_full] wrote {len(targets)} val targets -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--scene-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--depth-scale", type=float, default=0.1)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe existing train_pbr/ and val/ first")
    ap.add_argument("--min-visib", type=float, default=0.0,
                    help="T-038 >20% scoping passthrough: drop instances with "
                         "visib_fract <= this at convert-time (use 0.20 for the filter)")
    args = ap.parse_args()

    frames = discover_frames(args.raw_dir)
    if not frames:
        print(f"[convert_full] no gt_raw in {args.raw_dir}", file=sys.stderr); sys.exit(1)
    print(f"[convert_full] {len(frames)} raw frames discovered", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(frames)
    n_val = max(1, round(len(frames) * args.val_frac))
    val_frames = sorted(frames[:n_val])
    train_frames = sorted(frames[n_val:])
    print(f"[convert_full] split: {len(train_frames)} train / {len(val_frames)} val "
          f"(seed={args.seed}, val_frac={args.val_frac})", flush=True)

    if args.fresh:
        for sp in ("train_pbr", "val"):
            d = os.path.join(args.bop_root, sp)
            if os.path.isdir(d):
                print(f"[convert_full] --fresh: removing {d}", flush=True)
                shutil.rmtree(d)

    train_scenes = chunk(train_frames, args.scene_size)
    val_scenes = chunk(val_frames, args.scene_size)

    for sid, fr in enumerate(train_scenes):
        run_scene(args.python, args.raw_dir, args.bop_root, "train_pbr", sid, fr,
                  args.depth_scale, args.min_visib)
    for sid, fr in enumerate(val_scenes):
        run_scene(args.python, args.raw_dir, args.bop_root, "val", sid, fr,
                  args.depth_scale, args.min_visib)

    write_camera_json(args.raw_dir, args.bop_root)
    write_dataset_info(args.bop_root, len(train_scenes), len(val_scenes),
                       len(train_frames), len(val_frames))
    write_val_targets(args.bop_root, "val")

    print(f"[convert_full] DONE: train_pbr {len(train_scenes)} scenes / {len(train_frames)} frames, "
          f"val {len(val_scenes)} scenes / {len(val_frames)} frames", flush=True)
    print("CONVERT_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
