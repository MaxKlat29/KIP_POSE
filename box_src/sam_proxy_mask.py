#!/usr/bin/env python3
"""
sam_proxy_mask.py — POSE-INDEPENDENT visible-mask proxy via SAM (T-089)

WHY
---
The visibility-aware refine_rc fix (T-085) needs a per-instance VISIBLE-region
mask (`visib_mask`) to restrict the render-and-compare score to the
unoccluded pixels — that is what defeats the Anker 180deg-transverse flip when
only the shaft (or only the head) is visible. In the EVAL we had the BOP GT
`mask_visib`. **In the live pipeline there is no GT mask.** This module supplies
a POSE-INDEPENDENT proxy: prompt SAM (Segment-Anything, ViT-H) with the DETECTOR
bounding box and take the predicted segment as the visible silhouette of the
part inside that box.

POSE-INDEPENDENCE (load-bearing for the "does the fix still help?" test)
------------------------------------------------------------------------
The ONLY inputs to the proxy are:
  - the RGB image (what the camera sees)
  - the detector AABB (where the part is — comes from the 2D detector, NOT pose)
No rendered silhouette, no GDRNPP pose, no GT visibility ever enters the prompt.
So the proxy CANNOT leak the answer to the flip-disambiguation it is used for.
That is exactly why measuring the fix with THIS mask is the honest "does it
still bring something with a live proxy?" test, not the GT-mask_visib upper bound.

USAGE (library)
---------------
    from sam_proxy_mask import sam_visible_mask
    m = sam_visible_mask(rgb_hwc_uint8, bbox_xyxy)   # -> (H,W) bool

The first call lazily loads the ViT-H checkpoint onto CUDA (cached singleton).

USAGE (CLI smoke)
-----------------
    gdrnpp-venv/bin/python box_src/sam_proxy_mask.py \
        --rgb path/to/rgb.png --bbox x0,y0,x1,y1 --out /tmp/proxy.png
"""
import os
import sys
import numpy as np

# segment_anything lives in the cnos repo on the box
_CNOS = "/mnt/data/bop/repos/cnos"
if _CNOS not in sys.path:
    sys.path.insert(0, _CNOS)

_SAM_CKPT = os.environ.get(
    "SAM_CKPT", "/mnt/data/bop/weights/segment-anything/sam_vit_h_4b8939.pth")
_SAM_TYPE = os.environ.get("SAM_TYPE", "vit_h")

_PREDICTOR = None          # cached SamPredictor (model weights are heavy)
_LAST_IMG_ID = None        # id() of last RGB so we only re-embed on image change


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[_SAM_TYPE](checkpoint=_SAM_CKPT)
        sam.to(dev)
        sam.eval()
        _PREDICTOR = SamPredictor(sam)
    return _PREDICTOR


def sam_visible_mask(rgb, bbox_xyxy, *, multimask=True, set_image=True):
    """Pose-independent visible-mask proxy for ONE detection box.

    rgb        : (H,W,3) uint8 RGB image (full frame).
    bbox_xyxy  : [x0,y0,x1,y1] detector box (pixels, full-frame coords).
    multimask  : SAM returns 3 masks; we pick the highest-IoU-pred one (best
                 single-object segment), which for a box prompt is the part.
    set_image  : re-embed the frame (set False when batching same frame).

    Returns (H,W) bool. On any failure returns None (caller falls back to no
    visib_mask = today's behaviour) — never raises into the live path.
    """
    global _LAST_IMG_ID
    try:
        pred = _get_predictor()
        rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        if set_image or id(rgb) != _LAST_IMG_ID:
            pred.set_image(rgb)
            _LAST_IMG_ID = id(rgb)
        box = np.array(bbox_xyxy, dtype=float).reshape(1, 4)
        masks, scores, _ = pred.predict(box=box, multimask_output=multimask)
        if masks is None or len(masks) == 0:
            return None
        best = int(np.argmax(scores))
        return np.asarray(masks[best], dtype=bool)
    except Exception as exc:  # never break the caller
        sys.stderr.write(f"[sam_proxy] failed: {exc!r}\n")
        return None


def proxy_visib_fract(proxy_mask, bbox_xyxy):
    """Pose-independent proxy for `visib_fract` (the gate input) from the proxy
    mask alone: fraction of the (padded) detector box covered by the SAM segment.
    This is the live-available analogue of BOP visib_fract — it is NOT the GT
    occlusion fraction, just 'how much of the detected box is solid object'.
    Lower => more of the box is background/occluder => 'less visible'.
    """
    if proxy_mask is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    box_area = max(1, (x1 - x0) * (y1 - y0))
    seg_in_box = int(proxy_mask[y0:y1, x0:x1].sum())
    return float(seg_in_box) / float(box_area)


def free_image(predictor=None):
    """Drop the cached image embedding (not the weights) to free VRAM between
    frames if needed. Weights stay resident for warm reuse."""
    global _LAST_IMG_ID
    _LAST_IMG_ID = None


def _cli():
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="SAM proxy visible-mask smoke")
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--bbox", required=True, help="x0,y0,x1,y1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rgb = np.asarray(Image.open(a.rgb).convert("RGB"))
    bbox = [int(v) for v in a.bbox.split(",")]
    m = sam_visible_mask(rgb, bbox)
    if m is None:
        print("FAIL: no mask")
        sys.exit(1)
    vf = proxy_visib_fract(m, bbox)
    print(f"OK mask shape={m.shape} px={int(m.sum())} proxy_visib_fract={vf:.3f}")
    if a.out:
        Image.fromarray((m * 255).astype(np.uint8)).save(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    _cli()
