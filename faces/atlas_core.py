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
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
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


def top_down_descriptor(mesh, R, nbins=12):
    """Yaw-invariant top-down appearance signature of a resting pose.

    Radial profile (about the footprint centroid) of the top-surface height —
    i.e. *what the overhead camera sees*. Averaged over angle, so it is
    invariant to the part's yaw. Two rests are the *same face* iff they look the
    same from above: this merges true symmetry duplicates (a cylinder's side
    rolls, a prism's equal sides) yet keeps front/back-different rests apart
    (a gear hub-up vs hub-down has a central bump vs a flat back -> different).

    Returns [max-z profile (nbins), mean-z profile (nbins), aspect, footprint].
    """
    m = mesh.copy(); m.apply_transform(_ground_transform(R))
    V = m.vertices.copy()
    V[:, 2] -= V[:, 2].min()
    H = float(V[:, 2].max()) + 1e-9
    xy = V[:, :2]
    c = xy.mean(0)
    r = np.linalg.norm(xy - c, axis=1)
    Rmax = float(r.max()) + 1e-9
    zn = V[:, 2] / H
    b = np.clip((r / Rmax * nbins).astype(int), 0, nbins - 1)
    maxp = np.zeros(nbins); meanp = np.zeros(nbins)
    for k in range(nbins):
        sel = b == k
        if sel.any():
            maxp[k] = zn[sel].max(); meanp[k] = zn[sel].mean()
    aspect = H / (2 * Rmax)                       # tall (on edge) vs flat
    try:
        foot = float(ConvexHull(xy).volume) / (np.pi * Rmax ** 2)  # fill ratio
    except Exception:
        foot = 1.0
    return np.concatenate([maxp, meanp, [aspect, foot]])


def contact_feature(mesh, R, tol_frac=0.05):
    """Identify the face by *which mesh points touch the table*.

    Rotate the part into its rest pose, take the vertices at the lowest z (the
    contact patch), and describe the face by WHERE that patch sits on the body
    (centroid in body frame, part-centred + scale-normalised) plus the contact
    normal (= down-direction in body frame). Two settles are the same face iff
    they touch the same region of the part — the physically correct definition.

    Returns a 6-vector [centroid_x, centroid_y, centroid_z, n_x, n_y, n_z].
    """
    Rm = np.asarray(R).reshape(3, 3)
    Vb = mesh.vertices                                  # body-frame vertices
    z = Vb @ Rm[2, :]                                   # world-z of each vertex (= R·v)[2]
    zmin = z.min()
    h = z.max() - zmin + 1e-9
    contact = Vb[z <= zmin + tol_frac * h]
    scale = float(mesh.extents.max()) + 1e-9
    centroid = (contact.mean(0) - mesh.centroid) / scale
    normal = gravity_in_body(Rm) * 0.5                  # contact normal, weighted
    return np.concatenate([centroid, normal])


def _contact_merge(Rs, w, mesh, link_frac):
    """Single-linkage cluster settles by their contact patch on the table."""
    feats = np.array([contact_feature(mesh, R) for R in Rs])
    D = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
    A = D < link_frac
    n, labels = connected_components(A, directed=False)
    clusters = []
    for c in range(n):
        idx = np.where(labels == c)[0]
        mean = (feats[idx] * w[idx, None]).sum(0) / max(w[idx].sum(), 1e-9)
        rep = idx[int(np.argmin(np.linalg.norm(feats[idx] - mean, axis=1)))]
        clusters.append({"R": np.asarray(Rs[rep]).reshape(3, 3),
                         "prob": float(w[idx].sum()), "count": int(len(idx))})
    return clusters


def _symmetry_merge(mesh, clusters, tol):
    """Merge clusters that look identical from above (top-down descriptor)."""
    desc = [top_down_descriptor(mesh, c["R"]) for c in clusters]
    G = nx.Graph(); G.add_nodes_from(range(len(clusters)))
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            d = np.linalg.norm(desc[i] - desc[j]) / np.sqrt(len(desc[i]))
            if d < tol:
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


def tilt_tag(mesh, R):
    """Coarse descriptive tag for a face (NOT the identity — that's the cluster)."""
    long_axis = np.eye(3)[int(np.argmax(mesh.extents))]
    tilt = abs((np.asarray(R).reshape(3, 3) @ long_axis)[2])
    return "flach" if tilt < 0.35 else "hochkant" if tilt > 0.75 else "schräg"


# back-compat alias (older scripts import name_face)
def name_face(mesh, R):
    return tilt_tag(mesh, R)


def main_diagonal(mesh):
    """Unit vector of the part's main bounding-box diagonal (corner-to-corner)."""
    lo, hi = mesh.bounds
    d = np.asarray(hi) - np.asarray(lo)
    return d / (np.linalg.norm(d) + 1e-9)


def diagonal_floor_angle(mesh, R, d_body=None):
    """Angle (deg, 0..90) of the part's main diagonal to the floor in pose R."""
    if d_body is None:
        d_body = main_diagonal(mesh)
    dz = float((np.asarray(R).reshape(3, 3) @ d_body)[2])
    return float(np.degrees(np.arcsin(min(1.0, abs(dz)))))


def cluster_and_merge(Rs, w, mesh, link_tol=0.22, min_prob=0.02):
    """Group settles into physical faces by *which body region touches the table*.

    Face identity = the contact patch expressed in the body frame: contact
    normal (which way is down) + contact centroid (where on the part it touches).
    Both are yaw-invariant, so rotating the part about the vertical keeps the
    same contact region -> same face (no split for rotation). But genuinely
    different contacts stay apart: a gear resting flat, on its hub only, and
    leaning on hub+teeth are three distinct faces. Single-linkage on that
    descriptor; a cylinder's side-roll ring still chains into one.

    Returns (faces, rare, n_clusters, n_clusters).
    """
    Rs = [np.asarray(R).reshape(3, 3) for R in Rs]
    w = np.asarray(w, float); w = w / w.sum()
    d_body = main_diagonal(mesh)
    clusters = _contact_merge(Rs, w, mesh, link_tol)
    clusters.sort(key=lambda x: x["prob"], reverse=True)
    faces = [x for x in clusters if x["prob"] >= min_prob]
    rare = [x for x in clusters if x["prob"] < min_prob]
    for i, f in enumerate(faces):
        f["name"] = f"Face {i + 1}"
        f["diag_deg"] = diagonal_floor_angle(mesh, f["R"], d_body)
        f["tag"] = tilt_tag(mesh, f["R"])
    rare_info = {"n_faces": len(rare),
                 "prob": float(sum(x["prob"] for x in rare)),
                 "count": int(sum(x["count"] for x in rare))}
    return faces, rare_info, len(clusters), len(clusters)


# ── rendering ───────────────────────────────────────────────────────────────

def _shaded(verts, faces, base=(0.36, 0.52, 0.86)):
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    light = np.array([0.4, 0.2, 0.9]); light /= np.linalg.norm(light)
    return np.array(base)[None, :] * np.clip(np.abs(n @ light), 0.30, 1.0)[:, None]


def _render_pose(ax, mesh, R, mark_contact=True):
    m = mesh.copy(); m.apply_transform(_ground_transform(R))
    V = m.vertices.copy(); V[:, 2] -= V[:, 2].min()       # rest on z=0
    F = m.faces
    if len(F) > 9000:
        F = F[np.random.default_rng(0).choice(len(F), 9000, replace=False)]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=_shaded(V, F), edgecolors="none"))
    if mark_contact:                                      # the points touching the table
        h = V[:, 2].max() + 1e-9
        cp = V[V[:, 2] <= 0.02 * h]
        if len(cp):
            ax.scatter(cp[:, 0], cp[:, 1], np.zeros(len(cp)), c="#e23", s=4,
                       depthshade=False, zorder=5)
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


def _render_topdown(ax, mesh, R, max_tris=40000):
    """Orthographic overhead view — what the camera sees in the plane.

    Painter's algorithm (low z first) so the top surface shows; coloured by
    height like a depth image. Contact points (touching the table) outlined red.
    """
    m = mesh.copy(); m.apply_transform(_ground_transform(R))
    V = np.asarray(m.vertices, float); V[:, 2] -= V[:, 2].min()
    F = np.asarray(m.faces)
    if len(F) > max_tris:
        F = F[np.random.default_rng(0).choice(len(F), max_tris, replace=False)]
    tri = V[F]
    zc = tri[:, :, 2].mean(1)
    order = np.argsort(zc)
    znorm = (zc[order] - zc.min()) / (np.ptp(zc) + 1e-9)
    cols = cm.viridis(0.15 + 0.85 * znorm)
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=cols,
                                     edgecolors="none"))
    h = V[:, 2].max() + 1e-9
    cp = V[V[:, 2] <= 0.02 * h]
    if len(cp):
        ax.scatter(cp[:, 0], cp[:, 1], c="none", edgecolors="#e23", s=10,
                   linewidths=0.6, zorder=5)
    pad = 0.05 * (V[:, :2].max() - V[:, :2].min())
    ax.set_xlim(V[:, 0].min() - pad, V[:, 0].max() + pad)
    ax.set_ylim(V[:, 1].min() - pad, V[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.set_axis_off()


def render_atlas(mesh, faces, rare, title_main, title_sub, out,
                 max_faces=6, min_prob=0.02):
    """Per face: 3D rest view (top row) + overhead camera view (bottom row)."""
    k = min(max_faces, len(faces))
    show = faces[:k]
    for f in show:
        tag = f.get("tag"); dd = f.get("diag_deg")
        d = f["name"]
        if tag:
            d += f" · {tag}"
        if dd is not None:
            d += f" · diag {dd:.0f}°"
        f["disp"] = d

    fig = plt.figure(figsize=(max(11.0, 3.2 + 2.9 * k), 7.2))
    gs = GridSpec(2, k + 2, figure=fig, wspace=0.06, hspace=0.12,
                  top=0.80, bottom=0.06, left=0.05, right=0.98,
                  height_ratios=[1.15, 1.0])

    axb = fig.add_subplot(gs[:, 0:2])
    ypos = np.arange(len(show))[::-1]
    axb.barh(ypos, [f["prob"] * 100 for f in show], color="#4C72B0")
    axb.set_yticks(ypos); axb.set_yticklabels([f["disp"] for f in show], fontsize=10)
    axb.set_xlabel("Wahrscheinlichkeit der Ruhelage  [%]", fontsize=10)
    axb.set_title("Face-Verteilung", fontsize=12, fontweight="bold")
    for y, f in zip(ypos, show):
        axb.text(f["prob"] * 100 + 1.0, y, f'{f["prob"]*100:.1f}%  ·  {f["count"]}',
                 va="center", fontsize=9)
    axb.set_xlim(0, max(f["prob"] for f in show) * 100 * 1.35)
    axb.spines[["top", "right"]].set_visible(False)
    if rare["n_faces"]:
        axb.text(0, -0.6, f'+ {rare["n_faces"]} seltene Lagen < {min_prob:.0%} '
                          f'({rare["prob"]*100:.1f}%) — nicht als Face geführt',
                 transform=axb.get_yaxis_transform(), fontsize=8.5, color="#888", va="top")

    for i, f in enumerate(show):
        ax3d = fig.add_subplot(gs[0, i + 2], projection="3d")
        _render_pose(ax3d, mesh, f["R"])
        ax3d.set_title(f'{f["disp"]}\np={f["prob"]*100:.1f}%', fontsize=10, pad=-2)
        axtd = fig.add_subplot(gs[1, i + 2])
        _render_topdown(axtd, mesh, f["R"])
        if i == 0:
            axtd.set_title("↓ Kamera von oben (planar)", fontsize=9, color="#555")

    fig.suptitle(f"{title_main}\n{title_sub}", fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.055, 0.83, "oben: 3D-Ruhelage (rote Punkte = Tischkontakt)   ·   "
             "unten: Kamera-Draufsicht (Höhe = Farbe)", fontsize=9, color="#555")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
