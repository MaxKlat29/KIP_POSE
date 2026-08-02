#!/usr/bin/env python3
"""GT-pose overlay sanity check for the BOP dataset  (Story S-201/S-202 AK).

Projects each part's PLY mesh at its scene_gt pose onto the RGB image using the
scene_camera intrinsics. If the converter math (mm, row-major, w2c inversion,
OpenGL->OpenCV flip, scale-strip) is correct, the projected mesh wireframe/points
land exactly on the part in the image — WITH the LARA5 arm visible as an
occluder (parts behind the arm partly hidden).

This is the visual acceptance gate (adr.md §1.6 vis_gt_poses equivalent),
written self-contained so it does not need the bop_toolkit hydra/EGL renderer.

Output: <out>/overlay_{im:06d}.png  — RGB with per-instance coloured mesh-point
projection + obj_id label + projected origin axis.
"""
import argparse, json, os
import numpy as np
import trimesh
from PIL import Image, ImageDraw

COLORS = {1: (255, 60, 60), 2: (60, 160, 255), 3: (60, 255, 120),
          4: (255, 200, 40), 5: (220, 60, 255), 6: (60, 255, 240)}
NAMES = {1: "Anker_Kurz", 2: "Anker_Lang", 3: "Buerstenhalter",
         4: "Getriebe", 5: "Ringmagnet", 6: "Zahnrad"}


def load_models(models_dir):
    out = {}
    for oid in range(1, 7):
        p = os.path.join(models_dir, f"obj_{oid:06d}.ply")
        if os.path.exists(p):
            m = trimesh.load(p, force="mesh")
            V = np.asarray(m.vertices, float)
            if len(V) > 1500:
                V = V[np.random.default_rng(0).choice(len(V), 1500, replace=False)]
            out[oid] = V
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--split", default="train_pbr")
    ap.add_argument("--scene-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    sd = os.path.join(args.bop_root, args.split, f"{args.scene_id:06d}")
    models = load_models(os.path.join(args.bop_root, "models"))
    cam = json.load(open(os.path.join(sd, "scene_camera.json")))
    gt = json.load(open(os.path.join(sd, "scene_gt.json")))
    info = json.load(open(os.path.join(sd, "scene_gt_info.json")))
    os.makedirs(args.out, exist_ok=True)

    done = 0
    for key in sorted(gt.keys(), key=int):
        if done >= args.limit:
            break
        im_id = int(key)
        rgb = Image.open(os.path.join(sd, "rgb", f"{im_id:06d}.png")).convert("RGB")
        dr = ImageDraw.Draw(rgb)
        W, H = rgb.size
        K = np.asarray(cam[key]["cam_K"], float).reshape(3, 3)
        n_on = 0
        for gi, (g, ginf) in enumerate(zip(gt[key], info[key])):
            oid = g["obj_id"]
            if oid not in models:
                continue
            R = np.asarray(g["cam_R_m2c"], float).reshape(3, 3)
            t = np.asarray(g["cam_t_m2c"], float)
            Pc = (R @ models[oid].T).T + t          # model-mm -> cam-mm
            z = Pc[:, 2]
            valid = z > 1e-6
            uv = (K @ Pc[valid].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            col = COLORS.get(oid, (255, 255, 255))
            inb = 0
            for (u, v) in uv:
                if 0 <= u < W and 0 <= v < H:
                    dr.point((u, v), fill=col)
                    inb += 1
            # projected origin + label
            o = K @ t
            if o[2] > 1e-6:
                ou, ov = o[0] / o[2], o[1] / o[2]
                if 0 <= ou < W and 0 <= ov < H:
                    dr.ellipse([ou - 4, ov - 4, ou + 4, ov + 4], outline=col, width=2)
                    dr.text((ou + 6, ov - 6), f"{oid}:{NAMES.get(oid,'')[:6]} vf={ginf['visib_fract']:.2f}", fill=col)
                    n_on += 1
            if inb > 50:
                pass
        op = os.path.join(args.out, f"overlay_{im_id:06d}.png")
        rgb.save(op)
        print(f"frame {im_id:06d}: {len(gt[key])} GT, {n_on} origins in-frame -> {op}")
        done += 1
    print(f"[overlay] wrote {done} overlays -> {args.out}")


if __name__ == "__main__":
    main()
