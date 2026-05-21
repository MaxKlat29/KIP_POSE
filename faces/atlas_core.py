#!/usr/bin/env python3
"""Shared stable-face clustering + collage for the Face-Atlas module.

Both discovery paths feed this:
  * analytic  (``face_atlas.py``)      — trimesh.compute_stable_poses, weighted.
  * empirical (``atlas_from_drops.py``)— N Isaac physics settles, weight 1 each.

A "pose record" is just a rotation matrix R (column convention: ``world = R @
body``) plus a weight. The pipeline de-bloats the raw poses into the few
*physical* faces in three stages so the same face is never counted twice:

1. **g-merge** — single-linkage cluster by the gravity direction in the body
   frame (yaw-invariant); chains a near-cylindrical part's continuous ring of
   side-rests, and tessellation duplicates of one flat face, into one cluster.
2. **symmetry-merge** — clusters with the same yaw-invariant contact signature
   (footprint area + rest height), OR equiprobable at the same height, merge:
   the N equal sides of a prism / arcs of a ring torn apart by the angular
   threshold collapse into one face.
3. **probability-floor** — clusters below ``min_prob`` are unstable / numeric
   noise and are reported as a single "selten" note, never as their own face.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── geometry ────────────────────────────────────────────────────────────────

def gravity_in_body(R):
    """World down (0,0,-1) expressed in the body frame for rotation R (on S²)."""
    return np.asarray(R).reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def _ground_transform(R):
    """4x4 that rotates the mesh by R and drops it so it rests on z=0."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    return T


def contact_signature(mesh, R):
    """Yaw-invariant resting signature: (footprint_area, rest_height)."""
    m = mesh.copy(); m.apply_transform(_ground_transform(R))
    V = m.vertices
    height = float(V[:, 2].max() - V[:, 2].min())
    xy = V[:, :2]
    try:
        area = float(ConvexHull(xy).volume)      # 2-D hull "volume" == area
    except Exception:
        area = float(xy[:, 0].ptp() * xy[:, 1].ptp())
    return np.array([area, height])


def _close(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def _g_merge(Rs, w, deg):
    g = np.array([gravity_in_body(R) for R in Rs])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    A = (g @ g.T) >= np.cos(np.radians(deg))          # single-linkage adjacency
    n, labels = connected_components(A, directed=False)
    clusters = []
    for c in range(n):
        idx = np.where(labels == c)[0]
        mean = (g[idx] * w[idx, None]).sum(0)
        mean /= np.linalg.norm(mean) + 1e-9
        rep = idx[int(np.argmax(g[idx] @ mean))]      # member nearest cluster mean
        clusters.append({"R": np.asarray(Rs[rep]).reshape(3, 3),
                         "prob": float(w[idx].sum()), "count": int(len(idx))})
    return clusters


def _symmetry_merge(mesh, clusters, tol, prob_tol):
    sigs = [contact_signature(mesh, c["R"]) for c in clusters]
    G = nx.Graph(); G.add_nodes_from(range(len(clusters)))
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            ai, hi = sigs[i]; aj, hj = sigs[j]
            same_sig = _close(ai, aj, tol) and _close(hi, hj, tol)
            equiprob = (_close(hi, hj, tol) and
                        _close(clusters[i]["prob"], clusters[j]["prob"], prob_tol))
            if same_sig or equiprob:
                G.add_edge(i, j)
    out = []
    for comp in nx.connected_components(G):
        ms = sorted(comp)
        best = ms[int(np.argmax([clusters[m]["prob"] for m in ms]))]
        out.append({"R": clusters[best]["R"],
                    "prob": float(sum(clusters[m]["prob"] for m in ms)),
                    "count": int(sum(clusters[m]["count"] for m in ms))})
    out.sort(key=lambda c: c["prob"], reverse=True)
    return out


def name_face(mesh, R):
    long_axis = np.eye(3)[int(np.argmax(mesh.extents))]
    tilt = abs((np.asarray(R).reshape(3, 3) @ long_axis)[2])
    return "Liegend" if tilt < 0.40 else "Stehend" if tilt > 0.80 else "Schräg"


def cluster_and_merge(Rs, w, mesh, g_merge_deg=35.0, sig_tol=0.10,
                      min_prob=0.02, prob_tol=0.25):
    """Returns (faces, rare, n_after_g, n_after_sym).

    faces: list of {R, prob, count, name} with prob>=min_prob, sorted desc.
    rare:  {n_faces, prob, count} aggregate of the sub-floor clusters.
    """
    Rs = [np.asarray(R).reshape(3, 3) for R in Rs]
    w = np.asarray(w, float)
    w = w / w.sum()
    c1 = _g_merge(Rs, w, g_merge_deg)
    c2 = _symmetry_merge(mesh, c1, sig_tol, prob_tol)
    faces = [c for c in c2 if c["prob"] >= min_prob]
    rare = [c for c in c2 if c["prob"] < min_prob]
    for f in faces:
        f["name"] = name_face(mesh, f["R"])
    rare_info = {"n_faces": len(rare),
                 "prob": float(sum(c["prob"] for c in rare)),
                 "count": int(sum(c["count"] for c in rare))}
    return faces, rare_info, len(c1), len(c2)


# ── rendering ───────────────────────────────────────────────────────────────

def _shaded(verts, faces, base=(0.36, 0.52, 0.86)):
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    light = np.array([0.4, 0.2, 0.9]); light /= np.linalg.norm(light)
    return np.array(base)[None, :] * np.clip(np.abs(n @ light), 0.30, 1.0)[:, None]


def _render_pose(ax, mesh, R):
    m = mesh.copy(); m.apply_transform(_ground_transform(R))
    V = m.vertices.copy(); V[:, 2] -= V[:, 2].min()       # rest on z=0
    F = m.faces
    if len(F) > 9000:
        F = F[np.random.default_rng(0).choice(len(F), 9000, replace=False)]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=_shaded(V, F), edgecolors="none"))
    mn, mx = V.min(0), V.max(0); ctr = (mn + mx) / 2.0
    span = (mx - mn).max() * 0.55 + 1e-6
    gx = [ctr[0] - span, ctr[0] + span]; gy = [ctr[1] - span, ctr[1] + span]
    ax.plot_surface(np.array([[gx[0], gx[1]], [gx[0], gx[1]]]),
                    np.array([[gy[0], gy[0]], [gy[1], gy[1]]]),
                    np.zeros((2, 2)), color=(0.82, 0.82, 0.82), alpha=0.4, zorder=0)
    ax.set_xlim(*gx); ax.set_ylim(*gy); ax.set_zlim(0, 2 * span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=26, azim=-60); ax.set_axis_off()


def render_atlas(mesh, faces, rare, title_main, title_sub, out,
                 max_faces=8, min_prob=0.02):
    k = min(max_faces, len(faces))
    show = faces[:k]
    seen = {}
    for f in show:
        seen[f["name"]] = seen.get(f["name"], 0) + 1
        f["disp"] = f["name"] if seen[f["name"]] == 1 else f'{f["name"]} {seen[f["name"]]}'

    ncol = min(4, max(1, k)); nrow = int(np.ceil(k / ncol))
    fig = plt.figure(figsize=(max(11.0, 3.0 + 3.2 * ncol), max(3.9, 3.3 * nrow)))
    gs = GridSpec(nrow, ncol + 2, figure=fig, wspace=0.05, hspace=0.20,
                  top=0.80, bottom=0.12, left=0.055, right=0.98)

    axb = fig.add_subplot(gs[:, 0:2])
    ypos = np.arange(len(show))[::-1]
    axb.barh(ypos, [f["prob"] * 100 for f in show], color="#4C72B0")
    axb.set_yticks(ypos); axb.set_yticklabels([f["disp"] for f in show], fontsize=10)
    axb.set_xlabel("Wahrscheinlichkeit der Ruhelage  [%]", fontsize=10)
    axb.set_title("Face-Verteilung  (de-bloated)", fontsize=12, fontweight="bold")
    for y, f in zip(ypos, show):
        axb.text(f["prob"] * 100 + 1.0, y, f'{f["prob"]*100:.1f}%  ·  {f["count"]} Lagen',
                 va="center", fontsize=9)
    axb.set_xlim(0, max(f["prob"] for f in show) * 100 * 1.35)
    axb.spines[["top", "right"]].set_visible(False)
    if rare["n_faces"]:
        axb.text(0, -0.6, f'+ {rare["n_faces"]} seltene Lagen < {min_prob:.0%} '
                          f'(instabil, {rare["prob"]*100:.1f}%) — nicht als Face geführt',
                 transform=axb.get_yaxis_transform(), fontsize=8.5, color="#888", va="top")

    for i, f in enumerate(show):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[r, c + 2], projection="3d")
        _render_pose(ax, mesh, f["R"])
        ax.set_title(f'{f["disp"]}  ·  p={f["prob"]*100:.1f}%', fontsize=11, pad=-2)

    fig.suptitle(f"{title_main}\n{title_sub}", fontsize=14, fontweight="bold", y=0.975)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
