#!/usr/bin/env python3
"""
Build a grid montage of rendered frames with matplotlib — e.g. to eyeball the
physics variation across a batch of SDG renders.

    python sim_code/montage.py --dir data/output/physvar --out montage.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")           # render to file, no interactive backend needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="rgb_*.png")
    ap.add_argument("--out", default=None, help="output PNG (default: <dir>/montage.png)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--title", default="Physik-Variation (Marc tray-mode, Zivid)")
    args = ap.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.dir, args.pattern)),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )[: args.rows * args.cols]
    if not paths:
        print(f"no images matching {args.pattern} in {args.dir}")
        return

    out = args.out or os.path.join(args.dir, "montage.png")
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 4, args.rows * 2.6))
    fig.suptitle(args.title, fontsize=14)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(paths):
            ax.imshow(mpimg.imread(paths[i]))
            ax.set_title(f"Frame {i}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out} ({len(paths)} frames)")


if __name__ == "__main__":
    main()
