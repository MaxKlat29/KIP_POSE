#!/usr/bin/env python3
"""make_splitscreen.py - compose the final 2D-vs-3D split-screen.

LEFT : the real top-down Zivid input WITH the LARA5 arm + the detector boxes
       (det_overlay.png from the real chain).
RIGHT: the real 3D viewer render (viewer_render.png) - real CAD meshes placed at
       the GDRNPP-predicted poses on the real cell.glb.

Pure PIL, runs on the laptop. Writes project/temp/final_2d_vs_3d.png.
"""
import argparse

from PIL import Image, ImageDraw, ImageFont


def fit_height(img, h):
    w = int(round(img.width * h / img.height))
    return img.resize((w, h), Image.LANCZOS)


def load_font(size):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    H = 720
    GAP = 14
    BAR = 56
    PAD = 16
    BG = (16, 18, 22)

    left = fit_height(Image.open(a.left).convert("RGB"), H)
    right = fit_height(Image.open(a.right).convert("RGB"), H)

    W = PAD + left.width + GAP + right.width + PAD
    canvas = Image.new("RGB", (W, BAR + H + PAD), BG)
    d = ImageDraw.Draw(canvas)

    title_font = load_font(26)
    sub_font = load_font(15)
    cap_font = load_font(17)

    d.text((PAD, 10), "POSE — 2D Detektion  ·  3D Pose (echte GDRNPP-Gewichte)",
           fill=(235, 238, 245), font=title_font)
    d.text((PAD, 38),
           "val scene1/im92 · Detektor (Arm sichtbar) -> GDRNPP model_final.pth "
           "-> bop_adapter -> echte CAD an predicted Posen",
           fill=(150, 158, 170), font=sub_font)

    lx = PAD
    rx = PAD + left.width + GAP
    canvas.paste(left, (lx, BAR))
    canvas.paste(right, (rx, BAR))

    # captions on the images
    def caption(x, w, text):
        tw = d.textlength(text, font=cap_font)
        bx = x + (w - tw) / 2
        d.rectangle([bx - 8, BAR + 8, bx + tw + 8, BAR + 34], fill=(0, 0, 0))
        d.text((bx, BAR + 10), text, fill=(255, 255, 255), font=cap_font)

    caption(lx, left.width, "2D: Top-Down + Arm + Detektionen")
    caption(rx, right.width, "3D: CAD-Meshes an GDRNPP-Posen")

    canvas.save(a.out)
    print(f"wrote {a.out}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
