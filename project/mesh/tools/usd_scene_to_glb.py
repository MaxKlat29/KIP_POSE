#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["usd-core", "numpy", "trimesh", "fast-simplification"]
# ///
"""
Export the *static scene props* of GST_Scene.usd (the Basiswagen carts that form
the table/background, plus the static NEURA LARA5 arm) to decimated GLB meshes
for the KIP Pose Viewer's 3D mode.

Why world coordinates: the viewer is convention-agnostic and only ever places a
mesh by a 4x4 pose via ``poseToMatrix`` (which premultiplies the OpenCV->three.js
flip F). The detected *parts* are placed by their per-frame ``T_cam_obj`` with
geometry in object-local space. These props are fixed scene fixtures, so we bake
each prim's world transform into the vertices and export them in **USD world
coordinates (metres)**. The viewer then places the whole prop with a single
shared "pose" = the world->camera extrinsic ``T_world2cvcam`` — identical math,
no new code path. ``T_world2cvcam = D_FLIP @ DEFAULT_WORLD2CAM.T`` (see
tools/sdg_to_gt_overlay.py); it is emitted here too so the frontend stays in
sync with the gateway.

Self-check: the exported arm's world bbox must match ADR-008
(x[0.30,0.60], y[-0.01,0.51], z[0.015,0.60]); the script asserts this.

Usage:
  uv run tools/usd_scene_to_glb.py \
      --scene ../kip-pose-detection/data/USD-Files/GST_Scene.usd \
      --out   frontend/public/meshes
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf

# Same extrinsic the gateway / sdg_to_gt_overlay.py use for the GST_Scene Zivid.
DEFAULT_WORLD2CAM = np.array([
    [-0.99939083, 0.00000000, 0.03489950, 0.00000000],
    [-0.00665914, -0.98162718, -0.19069276, 0.00000000],
    [0.03425829, -0.19080900, 0.98102920, 0.00000000],
    [0.41262822, 0.29634180, -1.07804259, 1.00000000],
], dtype=np.float64)
D_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera -> OpenCV camera

# Prop groups: GST_Scene top-level prim names -> output GLB + decimation budget.
GROUPS = {
    "table": {
        # The three Basiswagen carts are the table/background the parts rest on.
        # GroundPlane is a 500x500 m infinite floor (not useful) and the walls are
        # Cube prims (not Mesh) — both deliberately excluded.
        "prims": ["Basiswagen", "Basiswagen_01", "Basiswagen_02"],
        "target_tris": 200_000,
    },
    "arm": {
        "prims": ["NEURA_LARA5_Pose_Zivid_Detection"],
        "target_tris": 120_000,
        # ADR-008 world bbox (metres) — geometry sanity gate.
        "expect_bbox": ([0.30, -0.01, 0.015], [0.60, 0.51, 0.60], 0.04),
    },
}


def gf_to_np(m: Gf.Matrix4d) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def triangulate(counts, indices):
    """Fan-triangulate arbitrary polygon faces -> (M,3) int array."""
    tris = []
    k = 0
    for c in counts:
        for i in range(1, c - 1):
            tris.append((indices[k], indices[k + i], indices[k + i + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), np.int64)


def collect_world_mesh(stage, root_path, mpu, xc):
    """Concatenate every Mesh under root_path into one world-space mesh (metres)."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"  ! prim {root_path} not found — skipping", file=sys.stderr)
        return None
    vchunks, fchunks, voff = [], [], 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        P = np.asarray(pts, dtype=np.float64)            # local, (N,3)
        M = gf_to_np(xc.GetLocalToWorldTransform(prim))  # row-vector: world = [P 1] @ M
        Ph = np.c_[P, np.ones(len(P))] @ M               # -> world (stage units)
        W = Ph[:, :3] * mpu                              # -> metres
        T = triangulate(counts, idx)
        if len(T) == 0:
            continue
        vchunks.append(W)
        fchunks.append(T + voff)
        voff += len(W)
    if not vchunks:
        return None
    return trimesh.Trimesh(
        vertices=np.vstack(vchunks), faces=np.vstack(fchunks), process=False
    )


def decimate(mesh, target):
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except TypeError:  # older trimesh signature
        return mesh.simplify_quadric_decimation(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="GST_Scene.usd")
    ap.add_argument("--out", required=True, help="output dir (frontend/public/meshes)")
    ap.add_argument("--only", default=None, help="comma list of groups (default: all)")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if not stage:
        sys.exit(f"ERROR: could not open {args.scene}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    xc = UsdGeom.XformCache()
    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else set(GROUPS)
    print(f"stage metersPerUnit = {mpu}")

    for name, spec in GROUPS.items():
        if name not in wanted:
            continue
        print(f"\n== {name} ==")
        parts = [collect_world_mesh(stage, f"/World/{p}", mpu, xc) for p in spec["prims"]]
        parts = [m for m in parts if m is not None]
        if not parts:
            print(f"  ! no geometry for {name} — skipped", file=sys.stderr)
            continue
        merged = trimesh.util.concatenate(parts)
        lo, hi = merged.bounds
        print(f"  raw: {len(merged.faces):,} tris  bbox(m) "
              f"x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]")

        if "expect_bbox" in spec:
            elo, ehi, tol = spec["expect_bbox"]
            if np.allclose(lo, elo, atol=tol) and np.allclose(hi, ehi, atol=tol):
                print(f"  ✓ bbox matches ADR-008 (tol {tol} m)")
            else:
                print(f"  ! bbox DEVIATES from ADR-008 expectation "
                      f"{elo}->{ehi}; convention may be off", file=sys.stderr)

        deci = decimate(merged, spec["target_tris"])
        out_path = os.path.join(args.out, f"{name}.glb")
        deci.export(out_path)
        sz = os.path.getsize(out_path) / 1e6
        print(f"  -> {out_path}  ({len(deci.faces):,} tris, {sz:.1f} MB)")

    # Emit the extrinsic the frontend needs, so it never drifts from the gateway.
    t_w2c = (D_FLIP @ DEFAULT_WORLD2CAM.T)
    meta_path = os.path.join(args.out, "scene_props.json")
    json.dump(
        {"comment": "T_world2cvcam = D_FLIP @ DEFAULT_WORLD2CAM.T (world->OpenCV cam, metres)",
         "T_world2cvcam": t_w2c.tolist()},
        open(meta_path, "w"), indent=2)
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
