#!/usr/bin/env python3
"""Validation grid: parts (Y) x their discovered face-views (X).

Reads several clustered facesets (each <faceset>/faces/{faces_<name>.json,
tmpl_FaceK.png}) and lays them out as one matplotlib frame — one row per part,
one column per face-view — to eyeball whether the face discovery makes sense
across different parts.

    python faces/face_grid.py <faceset_dir> [<faceset_dir> ...] --out grid.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("facesets", nargs="+")
    ap.add_argument("--out", default="faces_out/face_grid.png")
    args = ap.parse_args()

    parts = []
    for ds in args.facesets:
        fdir = os.path.join(ds, "faces")
        regs = [f for f in os.listdir(fdir) if f.startswith("faces_") and f.endswith(".json")]
        if not regs:
            continue
        reg = json.load(open(os.path.join(fdir, regs[0])))
        faces = []
        for fc in reg["faces"]:
            tp = os.path.join(fdir, f"tmpl_{fc['name'].replace(' ', '')}.png")
            if os.path.exists(tp):
                faces.append((fc, np.asarray(Image.open(tp).convert("RGB"))))
        if faces:
            parts.append((reg["part"], faces))

    nrow = len(parts)
    ncol = max(len(f) for _, f in parts)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.5 * ncol + 1.2, 2.6 * nrow),
                            squeeze=False)
    for r, (pname, faces) in enumerate(parts):
        for c in range(ncol):
            ax = axs[r][c]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c < len(faces):
                fc, img = faces[c]
                ax.imshow(img)
                ax.set_title(f"{fc['name']} · {fc['prob']*100:.0f}%", fontsize=9)
            else:
                ax.set_axis_off()
        axs[r][0].set_ylabel(pname, fontsize=11, fontweight="bold",
                             rotation=0, ha="right", va="center", labelpad=8)

    fig.suptitle("Face-View Validierung — Teile (Y) × gefundene Varianten (X)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor="white"); plt.close(fig)
    print(f"[grid] {nrow} parts x up to {ncol} faces -> {args.out}")


if __name__ == "__main__":
    main()
