#!/usr/bin/env python3
"""
Overlay the generated 2D-bbox labels onto the RGB render → one annotated
top-down preview image per sample. Pure PIL, runs anywhere (no Isaac Sim).

    python sim_code/visualize_labels.py --dir data/output/minimal
"""
import argparse
import glob
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

_COLORS = [
    (255, 80, 80), (80, 180, 255), (90, 230, 120), (255, 200, 40),
    (220, 110, 255), (255, 140, 50), (60, 220, 220), (255, 90, 170),
]


def _font(size):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def annotate(out_dir, idx, font):
    rgb_path = os.path.join(out_dir, f"rgb_{idx:04d}.png")
    bb_path = os.path.join(out_dir, f"bbox_2d_{idx:04d}.json")
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    bb = json.load(open(bb_path))
    id2lab = bb["info"]["idToLabels"]

    n = 0
    for row in bb["data"]:
        sid, x0, y0, x1, y1 = row[0], row[1], row[2], row[3], row[4]
        label = id2lab.get(str(sid), {}).get("class", str(sid))
        if label.lower() == "ground":            # skip the full-frame ground box
            continue
        col = _COLORS[sid % len(_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
        tb = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle([tb[0], tb[1] - 3, tb[2] + 5, tb[3] + 1], fill=col)
        draw.text((x0 + 3, y0 - 2), label, fill=(0, 0, 0), font=font)
        n += 1

    out = os.path.join(out_dir, f"annotated_{idx:04d}.png")
    img.save(out)
    print(f"  {os.path.basename(out)}  ({n} labels)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="output dir with rgb_*.png + bbox_2d_*.json")
    args = ap.parse_args()

    font = _font(20)
    idxs = sorted(
        int(re.search(r"rgb_(\d+)\.png", p).group(1))
        for p in glob.glob(os.path.join(args.dir, "rgb_*.png"))
    )
    if not idxs:
        print(f"no rgb_*.png in {args.dir}")
        return
    print(f"annotating {len(idxs)} sample(s) in {args.dir}:")
    for i in idxs:
        annotate(args.dir, i, font)


if __name__ == "__main__":
    main()
