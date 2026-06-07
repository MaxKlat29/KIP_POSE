#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
Convert a raw Isaac-Sim SDG frame into the KIP Pose Viewer's ground-truth
*overlay* bundle (uploadable via the viewer's "Ground-truth overlay (JSON)"
slot). Standalone / offline equivalent of the gateway's POST /gt_overlay — no
services required.

The viewer is convention-agnostic: it only ever consumes ``T_cam_obj`` (OpenCV
camera frame, metres) + an optional instance mask. Raw SDG output is NOT that —
``bbox_3d_*.json`` stores a row-major native->world transform with a 0.001 asset
scale baked in. This script performs the same conversion as
``kip-pose-detection/pose_eval/fp_build_scene.py`` (D_FLIP USD->OpenCV, 0.001 unscale,
world->cam extrinsic), verified bit-identical to it.

Inputs per frame (default names, --frame N):
  bbox_3d_<N>.json          6-DoF GT (native->world transform + class + occlusion)
  instance_<N>.png          instance segmentation (optional: masks + visibility)
  instance_labels_<N>.json  pixel-id -> prim path (optional: robust linkage)
Camera (shared across a render run):
  --world2cam world2cam_view.txt   4x4 row-major USD world->camera "view"
                                   (optional; defaults to the GST_Scene Zivid
                                   extrinsic, matching poc500 / most GST_Scene runs)
  --cam-k     cam_K.txt            3x3 pinhole intrinsics (reprojection self-check)

Output: {"frame":N,"source":...,"instances":[{id,class,occlusion,T_cam_obj[,mask_b64]}]}

Example:
  uv run tools/sdg_to_gt_overlay.py \
      --sdg-dir ../kip-pose-detection/data/output/poc500 --frame 0 \
      --cam-k   ../kip-pose-detection/foundationpose/cam_K.txt \
      --out gt_overlay_0000.json
  # add --world2cam <file> only to override the default GST_Scene extrinsic.
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

D_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera -> OpenCV camera

# Default camera extrinsic: the GST_Scene /World/Zivid world->camera "view"
# matrix (row-major, USD). Matches the bundled poc500 frames and most SDG runs
# against that scene. Identical to the gateway's DEFAULT_WORLD2CAM.
DEFAULT_WORLD2CAM = np.array([
    [-0.99939083, 0.00000000, 0.03489950, 0.00000000],
    [-0.00665914, -0.98162718, -0.19069276, 0.00000000],
    [0.03425829, -0.19080900, 0.98102920, 0.00000000],
    [0.41262822, 0.29634180, -1.07804259, 1.00000000],
], dtype=np.float64)


def load_K(path):
    return np.loadtxt(path).reshape(3, 3)


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def gt_pose_cam_obj(bbox_transform, T_world2cvcam):
    """bbox_3d transform (row-major native->world, 0.001 scale) -> T_cam_obj."""
    M = np.asarray(bbox_transform, dtype=np.float64)
    T_n2w = M.T
    R = orthonormalize(T_n2w[:3, :3] * 1000.0)  # strip the 0.001 asset scale
    t = T_n2w[:3, 3]
    T_mesh2world = np.eye(4)
    T_mesh2world[:3, :3] = R
    T_mesh2world[:3, 3] = t
    return T_world2cvcam @ T_mesh2world


def instance_id_to_prim(sdg_dir, idx):
    p = os.path.join(sdg_dir, f"instance_labels_{idx:04d}.json")
    if not os.path.exists(p):
        return None
    info = json.load(open(p))
    m = info.get("idToLabels", info)
    out = {}
    for k, v in m.items():
        try:
            iid = int(k)
        except (ValueError, TypeError):
            continue
        prim = v if isinstance(v, str) else (
            v.get("instanceId") or v.get("prim") or v.get("class") or json.dumps(v)
        )
        out[iid] = prim
    return out


def mask_b64(inst_img, iid):
    arr = np.where(inst_img == iid, 255, 0).astype(np.uint8)
    out = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), int((arr > 0).sum())


def convert(sdg_dir, idx, inst_img, id2prim, T_world2cvcam, K):
    """[{id,class,occlusion,T_cam_obj[,mask_b64]}] from raw SDG data.

    With an instance image: visible ids only, linked to bbox_3d rows (via
    instance_labels, else 3D-centroid projection), with masks. Without one:
    every bbox_3d row as a pose (no masks, no visibility filter)."""
    b = json.load(open(os.path.join(sdg_dir, f"bbox_3d_{idx:04d}.json")))
    data = b["data"]
    id2label = b["info"]["idToLabels"]
    prim_paths = b["info"].get("primPaths", [None] * len(data))
    by_prim = {}
    for i, row in enumerate(data):
        by_prim[prim_paths[i]] = (row, id2label[str(int(row[0]))]["class"].lower())

    if inst_img is None:
        return [
            {"id": i, "class": id2label[str(int(row[0]))]["class"].lower(),
             "occlusion": float(row[8]) if len(row) > 8 else None,
             "T_cam_obj": gt_pose_cam_obj(row[7], T_world2cvcam).tolist()}
            for i, row in enumerate(data)
        ]

    vis_ids = [int(i) for i in np.unique(inst_img) if int(i) != 0]
    out = []
    for iid in vis_ids:
        entry = None
        if id2prim is not None:
            prim = id2prim.get(iid)
            if prim in by_prim:
                entry = by_prim[prim]
        else:
            mask = inst_img == iid
            for row, cls in by_prim.values():
                M = np.asarray(row[7]).T
                c = np.array([(row[1] + row[4]) / 2, (row[2] + row[5]) / 2,
                              (row[3] + row[6]) / 2, 1.0])
                cc = T_world2cvcam @ (M @ c)
                if cc[2] <= 0:
                    continue
                u = int(round(K[0, 0] * cc[0] / cc[2] + K[0, 2]))
                v = int(round(K[1, 1] * cc[1] / cc[2] + K[1, 2]))
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                    entry = (row, cls)
                    break
        if entry is None:
            continue
        row, cls = entry
        b64, npx = mask_b64(inst_img, iid)
        if npx == 0:
            continue
        out.append({
            "id": int(iid),
            "class": cls,
            "occlusion": float(row[8]) if len(row) > 8 else None,
            "T_cam_obj": gt_pose_cam_obj(row[7], T_world2cvcam).tolist(),
            "mask_b64": b64,
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdg-dir", required=True,
                    help="dir with bbox_3d_/instance_/instance_labels_*.json|png")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cam-k", required=True, help="3x3 intrinsics txt")
    ap.add_argument("--world2cam", default=None,
                    help="4x4 USD world->camera view txt (default: GST_Scene Zivid)")
    ap.add_argument("--out", required=True, help="output overlay JSON path")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the origin-on-mask reprojection self-check")
    args = ap.parse_args()

    idx = args.frame
    inst_path = os.path.join(args.sdg_dir, f"instance_{idx:04d}.png")
    inst_img = np.array(Image.open(inst_path)) if os.path.exists(inst_path) else None
    id2prim = instance_id_to_prim(args.sdg_dir, idx) if inst_img is not None else None
    K = load_K(args.cam_k)
    view = np.loadtxt(args.world2cam).reshape(4, 4) if args.world2cam else DEFAULT_WORLD2CAM
    T_world2cvcam = D_FLIP @ view.T

    instances = convert(args.sdg_dir, idx, inst_img, id2prim, T_world2cvcam, K)
    if not instances:
        sys.exit("ERROR: no instances converted — check bbox_3d / instance / labels names")

    bundle = {
        "frame": idx,
        "source": f"converted from SDG frame {idx} ({os.path.basename(args.sdg_dir.rstrip('/'))})",
        "instances": instances,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(bundle, open(args.out, "w"))

    msg = f"wrote {len(instances)} instances -> {args.out}"
    if not args.no_check and inst_img is not None:
        hits = 0
        for inst in instances:
            T = np.asarray(inst["T_cam_obj"])
            z = T[2, 3]
            if z <= 1e-6:
                continue
            u = int(round(K[0, 0] * T[0, 3] / z + K[0, 2]))
            v = int(round(K[1, 1] * T[1, 3] / z + K[1, 2]))
            m = inst_img == inst["id"]
            H, W = m.shape
            if 0 <= u < W and 0 <= v < H:
                y0, y1 = max(0, v - 12), min(H, v + 13)
                x0, x1 = max(0, u - 12), min(W, u + 13)
                hits += bool(m[y0:y1, x0:x1].any())
        msg += f"  (self-check: {hits}/{len(instances)} pose origins near their mask)"
    print(msg)


if __name__ == "__main__":
    main()
