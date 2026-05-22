#!/usr/bin/env python3
"""Stable-Face-Atlas — analytic path.

Computes a part's stable resting poses analytically (``trimesh.poses.
compute_stable_poses`` — the analytic form of "drop it N times and see which
face lands"), de-bloats them into physical faces and renders the atlas collage.
No GPU. The empirical Isaac-Sim path lives in ``atlas_from_drops.py``.

    python faces/face_atlas.py <part.usd> [--out o.png]
        [--g-merge-deg 35] [--sig-tol 0.10] [--min-prob 0.02] [--max-faces 8]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

from usd_mesh import load_usd_mesh
import atlas_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--out", default=None)
    ap.add_argument("--link-tol", type=float, default=0.22,
                    help="contact-descriptor distance (normal + body centroid) to merge settles into one face")
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--max-faces", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=12)
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.part))[0]
    out = args.out or f"faces_out/{name}_atlas.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"[atlas] loading {args.part}")
    V, F = load_usd_mesh(args.part)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
    print(f"[atlas] mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}, extent={np.round(mesh.extents, 1)}")

    com = mesh.center_mass if mesh.is_watertight else mesh.convex_hull.center_mass
    print("[atlas] computing stable poses ...")
    transforms, probs = trimesh.poses.compute_stable_poses(
        mesh, center_mass=com, sigma=0.0, n_samples=args.n_samples, threshold=0.0)
    Rs = [np.asarray(T)[:3, :3] for T in transforms]
    print(f"[atlas] {len(probs)} raw stable poses")

    faces, rare, n_raw, _ = core.cluster_and_merge(
        Rs, np.asarray(probs), mesh, link_tol=args.link_tol, min_prob=args.min_prob)
    print(f"[atlas] {n_raw} contact-clusters | {len(faces)} faces >= {args.min_prob:.0%}" +
          (f" | {rare['n_faces']} seltene gefaltet ({rare['prob']*100:.1f}%)"
           if rare["n_faces"] else ""))

    core.render_atlas(
        mesh, faces, rare,
        f"Stable-Face-Atlas — {name}",
        f"{len(probs)} rohe Ruhelagen  →  {len(faces)} Faces "
        f"(analytisch, trimesh; Kontaktpunkt-Cluster · floor {args.min_prob:.0%})",
        out, max_faces=args.max_faces, min_prob=args.min_prob)
    print(f"[atlas] wrote {out}")


if __name__ == "__main__":
    main()
