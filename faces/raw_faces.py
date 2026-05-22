#!/usr/bin/env python3
"""Diagnostic: render the RAW g-merge clusters of a drop run — before the
symmetry-merge and probability-floor. Shows the true empirical multiplicity of
rest orientations, so over-/under-merging of the atlas can be judged by eye.

    python faces/raw_faces.py <part.usd> <drops.jsonl> [--g-merge-deg 35]
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
    ap.add_argument("--link-frac", type=float, default=0.12)
    ap.add_argument("--max-faces", type=int, default=12)
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.part))[0]
    out = args.out or f"faces_out/{name}_raw_faces.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    V, F = load_usd_mesh(args.part)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)

    Rs = []
    with open(args.drops) as f:
        for line in f:
            line = line.strip()
            if line:
                Rs.append(np.asarray(json.loads(line)["R"], float).reshape(3, 3))
    n = len(Rs)

    clusters = core._contact_merge(Rs, np.ones(n) / n, mesh, args.link_frac)
    clusters.sort(key=lambda c: c["prob"], reverse=True)
    for i, c in enumerate(clusters):
        c["name"] = f"Roh {i + 1}"
        c["tag"] = core.tilt_tag(mesh, c["R"])
    print(f"[raw] {n} drops -> {len(clusters)} contact-clusters (link={args.link_frac}): " +
          ", ".join(f'{c["name"]}/{c["tag"]} {c["prob"]*100:.1f}%(n={c["count"]})' for c in clusters))

    core.render_atlas(
        mesh, clusters, {"n_faces": 0, "prob": 0.0, "count": 0},
        f"ROH Kontaktpunkt-Cluster — {name}",
        f"{n} Drops  →  {len(clusters)} Cluster  (link-frac {args.link_frac}, KEIN floor)",
        out, max_faces=args.max_faces, min_prob=0.0)
    print(f"[raw] wrote {out}")


if __name__ == "__main__":
    main()
