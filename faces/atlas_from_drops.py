#!/usr/bin/env python3
"""Stable-Face-Atlas — empirical path.

Reads the settled poses produced by ``sim_code/discover_faces.py`` (N real
Isaac-Sim physics drops of one part) and de-bloats them into physical faces with
the same shared core as the analytic path. The face probabilities here are the
*measured* drop frequencies — the real resting-pose prior the classifier sees.

    python faces/atlas_from_drops.py <part.usd> <drops.jsonl> [--out o.png]
        [--g-merge-deg 35] [--sig-tol 0.10] [--min-prob 0.02] [--max-faces 8]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh

from usd_mesh import load_usd_mesh
import atlas_core as core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("drops")
    ap.add_argument("--out", default=None)
    ap.add_argument("--link-tol", type=float, default=0.22,
                    help="contact-descriptor distance (normal + body centroid) to merge settles into one face")
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--max-faces", type=int, default=8)
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.part))[0]
    out = args.out or f"faces_out/{name}_empirical_atlas.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"[atlas] loading mesh {args.part}")
    V, F = load_usd_mesh(args.part)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)

    Rs = []
    with open(args.drops) as f:
        for line in f:
            line = line.strip()
            if line:
                Rs.append(np.asarray(json.loads(line)["R"], float).reshape(3, 3))
    n = len(Rs)
    print(f"[atlas] {n} settled drops")
    if n == 0:
        raise SystemExit("no drops in file")

    faces, rare, n_raw, _ = core.cluster_and_merge(
        Rs, np.ones(n), mesh, link_tol=args.link_tol, min_prob=args.min_prob)
    print(f"[atlas] {n_raw} clusters | {len(faces)} faces >= {args.min_prob:.0%}: " +
          ", ".join(f'{f["name"]}/{f["tag"]}/diag{f["diag_deg"]:.0f} {f["prob"]*100:.1f}% (n={f["count"]})' for f in faces) +
          (f" | {rare['n_faces']} seltene gefaltet ({rare['prob']*100:.1f}%, n={rare['count']})"
           if rare["n_faces"] else ""))

    core.render_atlas(
        mesh, faces, rare,
        f"Stable-Face-Atlas — {name}  (empirisch)",
        f"{n} Isaac-Sim Physik-Drops  →  {len(faces)} Faces "
        f"(orientation-up-to-yaw · floor {args.min_prob:.0%})",
        out, max_faces=args.max_faces, min_prob=args.min_prob)
    print(f"[atlas] wrote {out}")

    # registration table: Face N <-> canonical body->world rotation R_face.
    # At inference: detected Face N gives R_face (out-of-plane orientation); the
    # OBB yaw + (x,y) back-projection complete the 6D pose -> CAD placement.
    reg = {"part": name, "n_drops": n, "convention": "world = R @ body (column)",
           "faces": [{"name": f["name"], "prob": round(f["prob"], 4),
                      "count": f["count"], "tag": f["tag"],
                      "diag_deg": round(f["diag_deg"], 1),
                      "R": [round(v, 6) for v in np.asarray(f["R"]).flatten().tolist()]}
                     for f in faces]}
    reg_path = os.path.join(os.path.dirname(out), f"faces_{name}.json")
    with open(reg_path, "w") as fh:
        json.dump(reg, fh, indent=2)
    print(f"[atlas] registration -> {reg_path}")


if __name__ == "__main__":
    main()
