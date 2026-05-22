#!/usr/bin/env python3
"""Diagnostic: cluster a rendered face-view dataset by g_body (orientation up to
yaw) and montage the ACTUAL rendered crops per cluster — to judge by eye whether
g_body gives visually clean, distinct top-views.

    python faces/gbody_montage.py <faceset_dir> [--deg 30]
"""
import argparse
import json
import os

import numpy as np
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--deg", type=float, default=30.0)
    ap.add_argument("--per-row", type=int, default=7)
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = args.dataset
    name = os.path.basename(ds.rstrip("/")).replace("faceset_", "")
    out = args.out or os.path.join(ds, f"gbody_montage_{name}.png")

    rows = [json.loads(l) for l in open(os.path.join(ds, "manifest.jsonl")) if l.strip()]
    g = np.array([r["g_body"] for r in rows], float)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    n = len(rows)

    A = (g @ g.T) >= np.cos(np.radians(args.deg))
    nc, labels = connected_components(A, directed=False)
    clusters = sorted([np.where(labels == c)[0] for c in range(nc)],
                      key=len, reverse=True)
    keep = [c for c in clusters if len(c) / n >= args.min_prob]
    print(f"[gbody] {n} views, deg={args.deg} -> {nc} clusters, "
          f"{len(keep)} >= {args.min_prob:.0%}: " +
          ", ".join(f"{len(c)} ({len(c)/n*100:.0f}%)" for c in keep))

    pr = args.per_row
    fig, axs = plt.subplots(len(keep), pr, figsize=(1.6 * pr, 1.7 * len(keep)),
                            squeeze=False)
    for ri, c in enumerate(keep):
        gm = g[c].mean(0); gm /= np.linalg.norm(gm) + 1e-9
        tilt = np.degrees(np.arccos(np.clip(abs(gm[2]), 0, 1)))  # 0=normal up,90=side
        axs[ri][0].set_ylabel(f"F{ri+1}\n{len(c)/n*100:.0f}%\ntilt {tilt:.0f}°",
                              fontsize=9, rotation=0, labelpad=26, va="center")
        sel = c[np.linspace(0, len(c) - 1, min(pr, len(c))).astype(int)]
        for k in range(pr):
            ax = axs[ri][k]; ax.set_xticks([]); ax.set_yticks([])
            if k < len(sel):
                im = Image.open(os.path.join(ds, rows[sel[k]]["rgb"])).convert("RGB")
                ax.imshow(im)
            else:
                ax.set_axis_off()
    fig.suptitle(f"g_body-Cluster (Orientierung bis auf Yaw) — {name}  "
                 f"({n} Views -> {len(keep)} Faces, deg={args.deg})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor="white"); plt.close(fig)
    print(f"[gbody] -> {out}")


if __name__ == "__main__":
    main()
