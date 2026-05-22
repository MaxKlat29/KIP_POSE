#!/usr/bin/env python3
"""Face-classifier inference — the stable contract for the serving layer.

    infer(crop, part) -> {"face": str, "confidence": float}

``crop``  : an HxWx3 (or HxW) uint8/array snippet of one segmented part, ideally
            the deskewed depth-height-map snippet (same domain as training); a
            raw RGB crop also works for the fallback.
``part``  : part name, e.g. "Anker_Lang" — selects the model + face class set.
returns   : {"face": "<face_id>", "confidence": <0..1>}.

Two backends, chosen automatically, NEVER crashes:
  1. checkpoint   — if faces/classifier/checkpoints/<model>.pt exists AND torch
                    is importable, run the trained CNN (softmax confidence).
  2. fallback     — otherwise, nearest-template matching against the per-face
                    reference templates (cluster_views tmpl_Face<k>.png), scored
                    by the same rotation+mirror NCC the clustering uses. Pure
                    numpy/PIL, no torch, no checkpoint, no training required.

This means the serving endpoint Jonas builds is functional from day one (fallback)
and silently upgrades to the trained model the moment a checkpoint is dropped in
— same call signature, same return shape.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

# Make both faces/ and faces/classifier/ importable, then import siblings flatly
# regardless of how this module is loaded (script, package, or sys.path tweak).
_HERE = os.path.dirname(os.path.abspath(__file__))          # faces/classifier
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                  # faces/
import registry          # type: ignore  # noqa: E402
from model import IMG_SIZE  # type: ignore  # noqa: E402

# cache loaded models / templates per part so repeated calls are cheap
_MODEL_CACHE: dict = {}
_TMPL_CACHE: dict = {}


def _prep(crop, size=IMG_SIZE):
    """Square-pad on neutral grey and resize to size×size, RGB uint8."""
    a = np.asarray(crop)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    sd = max(h, w)
    sq = np.full((sd, sd, 3), 128, np.uint8)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = a[:, :, :3]
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.BILINEAR))


# ── backend 1: trained checkpoint (lazy torch) ────────────────────────────────

def _load_model(part):
    if part in _MODEL_CACHE:
        return _MODEL_CACHE[part]
    ckpt = registry.checkpoint_path(part)
    if not os.path.exists(ckpt):
        _MODEL_CACHE[part] = None
        return None
    try:
        import torch
        from model import build_model
        state = torch.load(ckpt, map_location="cpu")
        classes = state.get("classes") or registry.classes_for(part)
        net = build_model(len(classes))
        net.load_state_dict(state["model"])
        net.eval()
        _MODEL_CACHE[part] = (net, classes)
    except Exception:                              # torch missing or bad ckpt
        _MODEL_CACHE[part] = None
    return _MODEL_CACHE[part]


def _infer_model(crop, part):
    loaded = _load_model(part)
    if loaded is None:
        return None
    net, classes = loaded
    import torch
    x = _prep(crop).astype(np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0].numpy()
    k = int(np.argmax(probs))
    return {"face": classes[k], "confidence": float(probs[k])}


# ── backend 2: nearest-template fallback (numpy only) ─────────────────────────

def _ncc_rot_mirror(a, b, step=10):
    """Best NCC of a against b over 360° rotation + mirror (grey, normalised)."""
    from scipy import ndimage
    best = -1.0
    for src in (a, a[:, ::-1]):
        for ang in range(0, 360, step):
            ar = ndimage.rotate(src, ang, reshape=False, order=1)
            m = (ar > 0.02) | (b > 0.02)
            if m.sum() < 20:
                continue
            x = ar[m] - ar[m].mean(); y = b[m] - b[m].mean()
            d = np.linalg.norm(x) * np.linalg.norm(y) + 1e-9
            best = max(best, float((x @ y) / d))
    return best


def _grey(img):
    return np.asarray(img).astype(np.float32).mean(2) / 255.0 if np.asarray(img).ndim == 3 \
        else np.asarray(img).astype(np.float32) / 255.0


def _load_refs(part, per_class=5):
    """Reference grey images per class for the fallback.

    Prefer a handful of training snippets per class (more representative than a
    single medoid template); fall back to the cluster_views tmpl_Face<k>.png if
    no snippets are present. Cached per part.
    """
    import glob
    refs = {}
    pdir = registry.part_dir(part)
    for cls in registry.classes_for(part):
        files = sorted(glob.glob(os.path.join(pdir, cls, "*.png")))
        if files:
            idx = np.linspace(0, len(files) - 1, min(per_class, len(files))).astype(int)
            refs[cls] = [_grey(_prep(np.asarray(Image.open(files[i]).convert("RGB"))))
                         for i in idx]
    if not refs:                                   # no snippets -> use templates
        for cls, t in registry.load_templates(part, size=IMG_SIZE).items():
            refs[cls] = [_grey(t)]
    return refs


def _infer_fallback(crop, part):
    if part not in _TMPL_CACHE:
        _TMPL_CACHE[part] = _load_refs(part)
    refs = _TMPL_CACHE[part]
    classes = registry.classes_for(part)
    if not refs:                                   # nothing to match against
        return {"face": classes[0] if classes else "face_1", "confidence": 0.0}
    q = _grey(_prep(crop))
    scores = {cls: max(_ncc_rot_mirror(q, r) for r in imgs) for cls, imgs in refs.items()}
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_cls, best = items[0]
    # confidence: a single class is trivially confident (its NCC); with rivals,
    # how far the winner separates from the runner-up, relative to the winner.
    if len(items) > 1:
        runner = items[1][1]
        conf = (best - runner) / (abs(best) + 1e-6)
    else:
        conf = best
    return {"face": best_cls, "confidence": float(np.clip(conf, 0.0, 1.0))}


# ── public API ────────────────────────────────────────────────────────────────

def infer(crop, part):
    """Classify ``crop`` for ``part``. STABLE SIGNATURE — do not change.

    Returns {"face": str, "confidence": float in [0,1]}. Uses the trained model
    if a checkpoint exists and torch is available, else the nearest-template
    fallback. Never raises for a missing model/checkpoint.
    """
    out = _infer_model(crop, part)
    if out is not None:
        return out
    return _infer_fallback(crop, part)


def backend(part):
    """Which backend infer() will use for ``part``: 'checkpoint' or 'fallback'."""
    return "checkpoint" if _load_model(part) is not None else "fallback"


if __name__ == "__main__":                         # tiny smoke CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("crop"); ap.add_argument("part")
    a = ap.parse_args()
    img = np.asarray(Image.open(a.crop).convert("RGB"))
    print(f"backend={backend(a.part)}  ->  {infer(img, a.part)}")
