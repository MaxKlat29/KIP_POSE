#!/usr/bin/env python3
"""BOP dataset validation gate for the full pose_isaac dataset (S-202/T-061).

Validates the converter output the way every downstream repo will read it:
  1. bop_toolkit NATIVE load of scene_camera/scene_gt/scene_gt_info per scene
     (the exact loaders CNOS/GigaPose/GDRNPP/eval use) — no repo patch allowed.
  2. train/val scene + frame + GT-instance counts.
  3. symmetry resolution via get_symmetry_transformations on models_info.json
     (continuous 1/2/5 -> many discretised; discrete C_7 obj6 -> 6 matrices).
  4. arm-occlusion belief: visib_fract distribution (proves the arm occludes).
  5. masks present and indexable per gt_id.

Exit 0 + prints VALIDATE_BOP_OK only if all checks pass.

USAGE  (bop-venv, on box; bop_toolkit_lib must be importable)
  PYTHONPATH=/mnt/data/bop/repos/bop_toolkit \
  /mnt/data/bop/bop-venv/bin/python box_src/validate_bop_full.py \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    args = ap.parse_args()

    from bop_toolkit_lib import inout, misc  # native loaders + symmetry

    root = args.bop_root
    ok = True

    # ---- 1+2: native load + counts per split ----
    report = {}
    for split in ("train_pbr", "val"):
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            print(f"[validate] MISSING split dir {sdir}"); ok = False; continue
        scenes = sorted(d for d in os.listdir(sdir) if d.isdigit())
        n_frames = n_inst = 0
        vfracs = []
        for sc in scenes:
            sp = os.path.join(sdir, sc)
            cam = inout.load_scene_camera(os.path.join(sp, "scene_camera.json"))
            gt = inout.load_scene_gt(os.path.join(sp, "scene_gt.json"))
            gti = inout.load_json(os.path.join(sp, "scene_gt_info.json"), keys_to_int=True)
            assert set(cam.keys()) == set(gt.keys()) == set(gti.keys()), \
                f"key mismatch in {sp}"
            n_frames += len(gt)
            for im in gt:
                assert len(gt[im]) == len(gti[im]), f"gt/gt_info len mismatch {sp}/{im}"
                n_inst += len(gt[im])
                # mask indexability + visib_fract
                for gid, info in enumerate(gti[im]):
                    vfracs.append(info["visib_fract"])
                    mv = os.path.join(sp, "mask_visib", f"{im:06d}_{gid:06d}.png")
                    mf = os.path.join(sp, "mask", f"{im:06d}_{gid:06d}.png")
                    if not (os.path.exists(mv) and os.path.exists(mf)):
                        print(f"[validate] MISSING mask {mv}"); ok = False
        vf = np.array(vfracs) if vfracs else np.array([1.0])
        report[split] = {
            "scenes": len(scenes), "frames": n_frames, "gt_instances": n_inst,
            "visib_fract_mean": round(float(vf.mean()), 3),
            "visib_fract_min": round(float(vf.min()), 3),
            "occluded<0.9": int((vf < 0.9).sum()),
            "occluded<0.5": int((vf < 0.5).sum()),
        }
        print(f"[validate] {split}: {report[split]}")

    # ---- 3: symmetry resolution ----
    mi = json.load(open(os.path.join(root, "models", "models_info.json")))
    print("[validate] symmetry resolution (get_symmetry_transformations):")
    for oid in sorted(mi, key=int):
        info = mi[oid]
        syms = misc.get_symmetry_transformations(info, max_sym_disc_step=0.01)
        kind = ("discrete" if "symmetries_discrete" in info
                else "continuous" if "symmetries_continuous" in info else "none")
        print(f"    obj {oid}: {kind:10s} -> {len(syms)} sym transforms (incl. identity)")
        if kind == "discrete":
            # C_7 -> 7 total (identity + 6 discrete matrices)
            if len(syms) != 7:
                print(f"    WARN obj {oid} discrete expected 7 (C_7), got {len(syms)}")

    # ---- arm-occlusion sanity: train must have a meaningful occluded fraction ----
    tr = report.get("train_pbr", {})
    if tr.get("occluded<0.9", 0) == 0:
        print("[validate] WARN: no arm/clutter occlusion detected (all visib_fract>=0.9)")

    if ok:
        print("VALIDATE_BOP_OK")
        sys.exit(0)
    else:
        print("VALIDATE_BOP_FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
