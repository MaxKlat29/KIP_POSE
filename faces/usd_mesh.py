#!/usr/bin/env python3
"""Extract a single welded triangle mesh from a USD file.

Walks the stage, bakes every ``UsdGeom.Mesh``'s local-to-world transform into
its points, triangulates polygons (fan) and concatenates everything into one
``(V, 3)`` / ``(F, 3)`` mesh. Scale is irrelevant for stable-pose analysis, but
we keep the USD's native units so thumbnails look right.
"""
from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom, Gf


def _triangulate(counts, indices):
    """Fan-triangulate face-vertex streams into (T, 3) index triples."""
    tris = []
    i = 0
    for c in counts:
        if c < 3:
            i += c
            continue
        v0 = indices[i]
        for k in range(1, c - 1):
            tris.append((v0, indices[i + k], indices[i + k + 1]))
        i += c
    return np.asarray(tris, dtype=np.int64)


def load_usd_mesh(path: str):
    """Return (vertices (V,3) float64, faces (F,3) int64) baked to world space."""
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")

    all_v = []
    all_f = []
    voff = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not counts or not idx:
            continue

        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts_np = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
        # apply 4x4 row-vector transform: p' = p * M
        homog = np.c_[pts_np, np.ones(len(pts_np))]
        M = np.array(m, dtype=np.float64).reshape(4, 4)
        world = homog @ M
        world = world[:, :3]

        tris = _triangulate(list(counts), list(idx))
        if len(tris) == 0:
            continue
        all_v.append(world)
        all_f.append(tris + voff)
        voff += len(world)

    if not all_v:
        raise RuntimeError(f"no mesh prims with geometry found in {path}")

    V = np.concatenate(all_v, axis=0)
    F = np.concatenate(all_f, axis=0)
    return V, F


if __name__ == "__main__":
    import sys
    V, F = load_usd_mesh(sys.argv[1])
    print(f"vertices={len(V)} faces={len(F)} "
          f"bbox_min={V.min(0)} bbox_max={V.max(0)} "
          f"extent_mm={(V.max(0) - V.min(0))}")
