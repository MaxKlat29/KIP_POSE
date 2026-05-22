#!/usr/bin/env python3
"""Richer matplotlib evaluation of an empirical drop run.

Visualises the raw N settles behind the atlas — so a "1 face" result for a
rotationally-symmetric part is *shown*, not just asserted:

  * left   — the settle gravity-directions g_body on the unit sphere, coloured by
             g-merge cluster (a cylinder's side-rests form one dense ring → one
             colour; standing outliers pop out),
  * middle — histogram of the tilt angle θ between g_body and the part's long
             axis (90° = lying, 0/180° = standing on an end),
  * right  — the de-bloated face-probability distribution.

    python faces/drop_analysis.py <part.usd> <drops.jsonl> [--out o.png]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse.csgraph import connected_components
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
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.part))[0]
    out = args.out or f"faces_out/{name}_drop_analysis.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    V, F = load_usd_mesh(args.part)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
    long_axis = np.eye(3)[int(np.argmax(mesh.extents))]

    Rs, G = [], []
    with open(args.drops) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            Rs.append(np.asarray(d["R"], float).reshape(3, 3))
            g = np.asarray(d["g_body"], float)
            G.append(g / (np.linalg.norm(g) + 1e-9))
    G = np.array(G)
    n = len(G)
    print(f"[analysis] {n} drops")

    # diagonal-to-floor angle per drop (the new discriminator)
    d_body = core.main_diagonal(mesh)
    diag = np.array([core.diagonal_floor_angle(mesh, R, d_body) for R in Rs])

    faces, rare, n_raw, _ = core.cluster_and_merge(
        Rs, np.ones(n), mesh, link_tol=args.link_tol, min_prob=args.min_prob)

    fig = plt.figure(figsize=(16, 5.2))

    # ── left: g_body on the unit sphere, coloured by diagonal-to-floor angle ──
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="0.85", linewidth=0.4)
    sc = ax.scatter(G[:, 0], G[:, 1], G[:, 2], c=diag, cmap="turbo",
                    vmin=0, vmax=90, s=8, depthshade=True)
    ax.set_title("Settle-Richtung g_body auf S²\n(Farbe = Diagonalwinkel)", fontsize=11)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label="Diagonale → Boden [°]")

    # ── middle: diagonal-to-floor angle histogram (multimodal = distinct rests) ──
    axh = fig.add_subplot(1, 3, 2)
    axh.hist(diag, bins=np.arange(0, 91, 3), color="#4C72B0", edgecolor="white")
    axh.set_xlabel("Winkel Hauptdiagonale → Boden  [°]   (0 = flach, 90 = aufrecht)",
                   fontsize=10)
    axh.set_ylabel("Anzahl Drops", fontsize=10)
    axh.set_title("Verteilung Diagonalwinkel", fontsize=12, fontweight="bold")
    axh.spines[["top", "right"]].set_visible(False)

    # ── right: face distribution ──
    axb = fig.add_subplot(1, 3, 3)
    ypos = np.arange(len(faces))[::-1]
    axb.barh(ypos, [f["prob"] * 100 for f in faces], color="#55a868")
    axb.set_yticks(ypos)
    axb.set_yticklabels([f'{f["name"]}\n{f["tag"]} {f["diag_deg"]:.0f}°' for f in faces],
                        fontsize=8)
    for y, f in zip(ypos, faces):
        axb.text(f["prob"] * 100 + 1, y, f'{f["prob"]*100:.1f}%  ({f["count"]})',
                 va="center", fontsize=9)
    axb.set_xlim(0, 119)
    axb.set_xlabel("Wahrscheinlichkeit  [%]", fontsize=10)
    axb.set_title(f"Faces (Normale + Diagonale)\n{n} Drops → {len(faces)} Faces",
                  fontsize=12, fontweight="bold")
    axb.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Drop-Auswertung — {name} (empirisch, {n} Isaac-Drops)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"[analysis] wrote {out}")


if __name__ == "__main__":
    main()
