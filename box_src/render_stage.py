#!/usr/bin/env python3
"""render_stage.py — Server-side photorealistic render einer Stage auf der Box.

Pro Stage (raw / zsnap / m2 / final) rendert dieses Skript ein Bild:
  • Source-RGB als Hintergrund (das echte Isaac-Top-Down)
  • CAD-Mesh jeder vorhergesagten Part-Pose, ge-rendert AUS DER ECHTEN KAMERA
    (cam_K + cam_R_w2c + cam_t_w2c aus dem BOP-Datensatz)
  • Composite: CAD-overlay über das echte Foto

Damit sieht Max die predicted Posen GENAU dort wo das Modell sie hingepackt hat,
in echter Auflösung, mit echten CAD-Meshes (BOP-PLY, 1:1 Isaac).

Gebraucht: pyrender + trimesh + PIL (alle auf bop-venv vorhanden).
GPU optional (EGL); CPU-Pfad funktioniert auch (OSMESA fällt zurück).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Headless GL backend — egl wenn verfügbar, sonst osmesa.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image


# ── Mapping obj_id -> BOP-PLY path ───────────────────────────────────────────
def ply_path(dataset_dir: str, obj_id: int) -> str:
    return os.path.join(dataset_dir, "models", f"obj_{obj_id:06d}.ply")


# ── pose_result schema convention ─────────────────────────────────────────────
# t_world is in pose-frame METER (relative to table_origin). To get back into the
# DATASET-world (which matches the camera extrinsics), add meta.table_origin.
def pose_to_dataset_world(t_world_m, table_origin_m):
    return (np.array(t_world_m) + np.array(table_origin_m)) * 1000.0  # → mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--im", type=int, required=True)
    ap.add_argument("--pose-result", required=True,
                    help="path to one pose_result_<stage>.json")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--opacity", type=float, default=0.85,
                    help="overlay alpha (0..1) — how strongly CAD wins over the photo")
    a = ap.parse_args()

    scene_dir = os.path.join(a.dataset_dir, a.split, f"{a.scene:06d}")
    cam_all = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    cam = cam_all[str(a.im)]
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    R_w2c = np.array(cam["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.array(cam["cam_t_w2c"], dtype=np.float64)  # mm

    rgb_path = os.path.join(scene_dir, "rgb", f"{a.im:06d}.png")
    bg = Image.open(rgb_path).convert("RGB")
    W, H = bg.size

    pose_doc = json.load(open(a.pose_result))
    table_origin = pose_doc["meta"]["table_origin"]

    # Build pyrender scene. We render in CAMERA frame: camera at origin, looking
    # down -Z (pyrender convention is +Y up, -Z forward like OpenGL). BUT BOP
    # cam_R_w2c uses CV convention (+Y down, +Z forward). Convert.
    cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])  # flip Y and Z for the camera

    # World->camera (BOP cv) -> world->camera (GL)
    Twc_cv = np.eye(4)
    Twc_cv[:3, :3] = R_w2c
    Twc_cv[:3, 3] = t_w2c

    # In pyrender, the camera POSE is camera->world (inverse).
    Tcw_cv = np.linalg.inv(Twc_cv)
    Tcw_gl = Tcw_cv @ cv2gl
    cam_pose = Tcw_gl  # camera-to-world in GL conv (Y-up)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=10.0, zfar=5000.0)

    scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.96, 1.0],
                           ambient_light=[0.55, 0.55, 0.58])
    scene.add(camera, pose=cam_pose)

    # Soft directional light from above-front of the camera (like top-down Isaac lighting)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_pose = cam_pose.copy()
    light_pose[:3, 3] += cam_pose[:3, :3] @ np.array([0, 0.3, -0.2])  # offset
    scene.add(light, pose=light_pose)

    # Add each part at its predicted pose. Pose is in pose-frame (m); we convert to
    # dataset-world (mm) for compatibility with the camera-mm extrinsics.
    n_parts = 0
    for r in pose_doc.get("results", []):
        oid_name = r["part"]
        # part -> obj_id
        OID = {"Anker_Kurz": 1, "Anker_Lang": 2, "Buerstenhalter_2polig": 3,
               "Getriebegehaeuse_typ4": 4, "Ringmagnet": 5, "Zahnrad": 6}
        if oid_name not in OID:
            continue
        plypath = ply_path(a.dataset_dir, OID[oid_name])
        if not os.path.isfile(plypath):
            print(f"[render] PLY missing for {oid_name}: {plypath}", file=sys.stderr)
            continue
        mesh_tm = trimesh.load(plypath, force="mesh")
        # PLYs are in mm, world is mm too → no rescale needed.
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)

        R_world = np.array(r["R_world"]).reshape(3, 3)
        t_world_dataset_mm = pose_to_dataset_world(r["t_world"], table_origin)
        T = np.eye(4)
        T[:3, :3] = R_world
        T[:3, 3] = t_world_dataset_mm
        scene.add(mesh_pr, pose=T)
        n_parts += 1

    print(f"[render] scene with {n_parts} parts; image {W}x{H}", file=sys.stderr)

    # Offscreen renderer
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, depth = r.render(scene)
    r.delete()

    # Composite: CAD wherever depth>0, source-RGB elsewhere
    mask = depth > 0
    out = np.array(bg).copy()
    if mask.any():
        cad_rgba = color.astype(np.float32)
        alpha = float(a.opacity)
        out[mask] = (alpha * cad_rgba[mask] + (1 - alpha) * out[mask]).astype(np.uint8)

    Image.fromarray(out).save(a.out_png)
    print(f"[render] wrote {a.out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
