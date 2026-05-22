#!/usr/bin/env python3
"""Cluster rendered top-down views into distinct face-views (CNN classes).

Two robust, generalised stages (no per-part geometry tuning, no unstable
PCA-deskew in the clustering):

  1. g_body — group by orientation up to yaw (gravity in the body frame). Stable
     (no PCA). Separates which-side-up/down (a part's two flat sides stay apart).
  2. appearance-merge — merge g_body clusters whose top-down *looks the same* via
     a rotation-invariant radial height profile (so e.g. an edge-stand that g_body
     split by axis-direction collapses into one face).

Each resulting cluster is a face-view = a CNN class. Outputs the registration
table (face -> R), a template montage, and the labelled crops per face.

    python faces/cluster_views.py <faceset_dir> [--deg 28] [--merge 0.16]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

NB = 12   # radial bins


def height_map(dep, msk):
    dv = dep[msk]
    h = np.zeros_like(dep, float)
    if dv.size and dv.max() > dv.min():
        h[msk] = (dv.max() - dep[msk]) / (dv.max() - dv.min())
    return h


def radial_desc(dep, msk):
    """Rotation-invariant top-view descriptor: radial height profile + aspect."""
    h = height_map(dep, msk)
    ys, xs = np.nonzero(msk)
    c = np.array([xs.mean(), ys.mean()])
    r = np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2)
    rmax = r.max() + 1e-9
    prof = np.zeros(NB)
    b = np.clip((r / rmax * NB).astype(int), 0, NB - 1)
    hv = h[ys, xs]
    for k in range(NB):
        m = b == k
        prof[k] = hv[m].mean() if m.any() else 0.0
    pts = np.stack([xs, ys], 1).astype(float)
    ev = np.linalg.eigvalsh(np.cov((pts - c).T))
    aspect = float(np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))   # 1=round, <1 elongated
    return np.concatenate([prof, [aspect]])


def deskew(rgb, dep, msk):
    ys, xs = np.nonzero(msk)
    pts = np.stack([xs, ys], 1).astype(float); c = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - c).T))
    major = evec[:, int(np.argmax(ev))]
    ang = np.degrees(np.arctan2(major[1], major[0]))
    rr = ndimage.rotate(rgb, ang, reshape=True, order=1, mode="constant", cval=128)
    mm = ndimage.rotate(msk.astype(np.uint8), ang, reshape=True, order=0) > 0
    if mm.sum() > 20:
        ys2, xs2 = np.nonzero(mm)
        rr = rr[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--part-name", default=None)
    ap.add_argument("--deg", type=float, default=28.0, help="g_body grouping angle")
    ap.add_argument("--match", type=float, default=0.90,
                    help="rotation-search image NCC above which two views are the same shape")
    ap.add_argument("--flip-guard", type=float, default=-0.5,
                    help="merge only if mean g_body dot > this (blocks top/bottom flips)")
    ap.add_argument("--min-prob", type=float, default=0.02)
    args = ap.parse_args()

    ds = args.dataset
    name = args.part_name or os.path.basename(ds.rstrip("/")).replace("faceset_", "")
    out = os.path.join(ds, "faces"); os.makedirs(out, exist_ok=True)

    rows = [json.loads(l) for l in open(os.path.join(ds, "manifest.jsonl")) if l.strip()]
    n = len(rows)
    g = np.array([r["g_body"] for r in rows], float)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    deps = [np.load(os.path.join(ds, r["dep"])) for r in rows]
    msks = [np.load(os.path.join(ds, r["msk"])) for r in rows]
    descs = np.array([radial_desc(deps[i], msks[i]) for i in range(n)])

    # stage 1: g_body single-linkage
    _, lab1 = connected_components(g @ g.T >= np.cos(np.radians(args.deg)), directed=False)
    g_clusters = [np.where(lab1 == c)[0] for c in range(lab1.max() + 1)]

    # stage 2: deterministic appearance merge. Rotate one cluster's planar
    # top-view through 360° and check if it matches another (>match), BUT only
    # merge if they are NOT a top/bottom flip — i.e. the raw 3D orientation
    # (g_body) is not opposed. So pure in-plane re-orientations of the same side
    # collapse, while genuinely different sides (look alike, flipped) stay apart.
    def rep_img(c):                                          # height-map of the medoid, 96x96 square
        med = c[int(np.argmin(np.linalg.norm(descs[c] - descs[c].mean(0), axis=1)))]
        h = height_map(deps[med], msks[med])
        ys, xs = np.nonzero(msks[med])
        h = h[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sd = max(h.shape); sq = np.zeros((sd, sd))
        sq[(sd - h.shape[0]) // 2:(sd - h.shape[0]) // 2 + h.shape[0],
           (sd - h.shape[1]) // 2:(sd - h.shape[1]) // 2 + h.shape[1]] = h
        return np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((96, 96))) / 255.0

    def rot_match(a, b, step=6):                             # best NCC over 360° in-plane rotation
        best = -1.0
        for ang in range(0, 360, step):
            ar = ndimage.rotate(a, ang, reshape=False, order=1)
            m = (ar > 0.02) | (b > 0.02)
            if m.sum() < 20:
                continue
            x = ar[m] - ar[m].mean(); y = b[m] - b[m].mean()
            d = np.linalg.norm(x) * np.linalg.norm(y) + 1e-9
            best = max(best, float((x @ y) / d))
        return best

    K = len(g_clusters)
    reps = [rep_img(c) for c in g_clusters]
    gmean = [g[c].mean(0) / (np.linalg.norm(g[c].mean(0)) + 1e-9) for c in g_clusters]
    Adj = np.eye(K, dtype=bool)
    for i in range(K):
        for j in range(i + 1, K):
            if gmean[i] @ gmean[j] > args.flip_guard:        # not a top/bottom flip
                if rot_match(reps[i], reps[j]) >= args.match:
                    Adj[i, j] = Adj[j, i] = True
    _, lab2 = connected_components(Adj, directed=False)

    merged = {}
    for gi, m in enumerate(lab2):
        merged.setdefault(m, []).extend(g_clusters[gi].tolist())
    faces = [np.array(sorted(v)) for v in merged.values()]
    faces.sort(key=len, reverse=True)
    keep = [f for f in faces if len(f) / n >= args.min_prob]
    rare = sum(len(f) for f in faces if len(f) / n < args.min_prob)
    print(f"[cluster] {n} views: g_body->{len(g_clusters)} clusters, "
          f"appearance-merge->{len(faces)} faces, {len(keep)} >= {args.min_prob:.0%}")

    reg = {"part": name, "n_views": n, "convention": "world = R @ body (column)", "faces": []}
    fig, axs = plt.subplots(1, max(2, len(keep)), figsize=(2.8 * max(2, len(keep)), 3.2))
    for k, f in enumerate(keep):
        med = f[int(np.argmin(np.linalg.norm(descs[f] - descs[f].mean(0), axis=1)))]
        fname = f"Face {k + 1}"
        gm = g[f].mean(0); gm /= np.linalg.norm(gm) + 1e-9
        tilt = float(np.degrees(np.arccos(np.clip(abs(gm[2]), 0, 1))))
        fdir = os.path.join(out, fname.replace(" ", "_")); os.makedirs(fdir, exist_ok=True)
        for j in f:
            rgb = np.asarray(Image.open(os.path.join(ds, rows[j]["rgb"])).convert("RGB"))
            Image.fromarray(deskew(rgb, deps[j], msks[j]).astype(np.uint8)).save(
                os.path.join(fdir, f"{rows[j]['i']:04d}.png"))
        reg["faces"].append({"name": fname, "prob": round(len(f) / n, 4), "count": int(len(f)),
                             "tilt_deg": round(tilt, 1),
                             "R": [round(v, 6) for v in rows[med]["R"]]})
        ax = np.atleast_1d(axs)[k]
        rgb = np.asarray(Image.open(os.path.join(ds, rows[med]["rgb"])).convert("RGB"))
        tmpl = deskew(rgb, deps[med], msks[med]).astype(np.uint8)
        Image.fromarray(tmpl).save(os.path.join(out, f"tmpl_{fname.replace(' ', '')}.png"))
        ax.imshow(tmpl)
        ax.set_title(f"{fname}\n{len(f)/n*100:.0f}% (n={len(f)}) tilt {tilt:.0f}°", fontsize=10)
        ax.set_axis_off()
    for ax in np.atleast_1d(axs)[len(keep):]:
        ax.set_axis_off()
    json.dump(reg, open(os.path.join(out, f"faces_{name}.json"), "w"), indent=2)
    fig.suptitle(f"Face-Views — {name}  ({n} Drops -> {len(keep)} Klassen · "
                 f"g_body {args.deg:.0f}° + appearance-merge)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    mp = os.path.join(out, f"facemontage_{name}.png")
    fig.savefig(mp, dpi=130, facecolor="white"); plt.close(fig)
    print("[cluster] faces: " +
          ", ".join(f'{x["name"]} {x["prob"]*100:.0f}%(tilt{x["tilt_deg"]:.0f})' for x in reg["faces"]))
    print(f"[cluster] -> {out}  | montage {mp}" + (f"  | {rare} rare folded" if rare else ""))


if __name__ == "__main__":
    main()
