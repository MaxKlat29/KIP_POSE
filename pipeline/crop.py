#!/usr/bin/env python3
"""Cut per-detection image crops out of an SDG scene image (S-008).

Reads the SDG annotator output produced by ``sim_code/run_scene.py`` /
``datagenerationscript.py``:

  * ``rgb_<idx>.png``               — the scene image
  * ``bbox_2d_<idx>.json``          — Isaac ``bounding_box_2d_tight`` annotator,
                                      ``{"data": [[semId, x0, y0, x1, y1, occ], ...],
                                        "info": {"idToLabels": {"<semId>": {"class": "<label>"}}}}``
  * ``semantic_<idx>.png`` + ``semantic_labels_<idx>.json`` (optional) — used to
    build an instance mask for the crop, so the classifier sees the part on a
    neutral background instead of neighbouring clutter.

A crop is the axis-aligned ``bbox_2d`` rectangle clamped to the image. When a
segmentation map is available the crop is additionally background-masked to
neutral grey using the dominant part-class label inside the box. The classifier
square-pads on neutral grey itself, so an un-masked crop also works.

This module is pure ``numpy + PIL`` — no torch, no Isaac, runs anywhere.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

NEUTRAL = 128  # grey used for padding / background, matches faces.classifier._prep


@dataclass
class Detection:
    """One 2D detection cut from the scene image.

    Attributes
    ----------
    instance_id : stable per-scene id (row order in the bbox file).
    part        : canonical part name (e.g. ``Anker_Lang``), normalised from the
                  raw annotator label (which is lower-case, e.g. ``anker_lang``).
    bbox_2d     : ``[x0, y0, x1, y1]`` ints in source-image pixels, clamped, ordered.
    crop        : ``HxWx3`` uint8 RGB crop (optionally background-masked).
    occlusion   : annotator occlusionRatio 0..1 (0 = fully visible).
    raw_label   : the original annotator class string (for diagnostics).
    """

    instance_id: int
    part: str
    bbox_2d: list[int]
    crop: np.ndarray = field(repr=False)
    occlusion: float = 0.0
    raw_label: str = ""


# canonical part-name lookup: the SDG annotator emits lower-case class names
# ("anker_lang"), the registry / classifier routing keys are TitleCase
# ("Anker_Lang"). Map raw -> canonical without guessing for unknown labels.
def _canonical_parts() -> dict[str, str]:
    try:
        from faces.classifier import registry  # type: ignore
        names = list(registry.PART_TO_MODEL.keys())
    except Exception:
        names = [
            "Anker_Lang", "Anker_Kurz", "Zahnrad", "Poltopf_kurz_centered",
            "Getriebegehaeuse_typ4", "Buerstenhalter_2polig",
        ]
    return {n.lower(): n for n in names}


_CANON = _canonical_parts()


def canonical_part(raw_label: str) -> str:
    """Map a raw annotator label to the canonical part name.

    ``anker_lang`` -> ``Anker_Lang``. Unknown labels are returned unchanged so the
    caller can warn + skip rather than silently mis-route.
    """
    return _CANON.get(raw_label.strip().lower(), raw_label)


def _clamp_box(x0, y0, x1, y1, w, h) -> tuple[int, int, int, int]:
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h))
    return x0, y0, x1, y1


def load_scene(scene_dir: str, idx: int = 0) -> tuple[np.ndarray, dict, dict | None]:
    """Load ``(rgb, bbox_data, semantic_labels)`` for render ``idx`` in ``scene_dir``.

    ``semantic_labels`` is ``None`` when no semantic_labels file is present.
    Raises ``FileNotFoundError`` if the rgb or bbox file is missing.
    """
    rgb_path = os.path.join(scene_dir, f"rgb_{idx:04d}.png")
    bbox_path = os.path.join(scene_dir, f"bbox_2d_{idx:04d}.json")
    sem_path = os.path.join(scene_dir, f"semantic_labels_{idx:04d}.json")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"scene rgb not found: {rgb_path}")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"bbox_2d not found: {bbox_path}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    with open(bbox_path) as f:
        bbox_data = json.load(f)
    sem_labels = None
    if os.path.exists(sem_path):
        with open(sem_path) as f:
            sem_labels = json.load(f)
    return rgb, bbox_data, sem_labels


def _load_seg(scene_dir: str, idx: int) -> np.ndarray | None:
    """Load the semantic segmentation map (id image) if present, else None."""
    sem_png = os.path.join(scene_dir, f"semantic_{idx:04d}.png")
    if not os.path.exists(sem_png):
        return None
    seg = np.asarray(Image.open(sem_png))
    return seg[..., 0] if seg.ndim > 2 else seg


def _mask_crop(crop_rgb: np.ndarray, seg_crop: np.ndarray | None,
               keep_ids: set[int]) -> np.ndarray:
    """Neutral-grey out everything in ``crop_rgb`` not in ``keep_ids``.

    No-op (returns the crop unchanged) when there is no usable seg, so a raw
    bbox crop still flows through.
    """
    if seg_crop is None or not keep_ids:
        return crop_rgb
    keep = np.isin(seg_crop, list(keep_ids))
    if keep.sum() < 10:                              # seg didn't line up — keep raw
        return crop_rgb
    out = crop_rgb.copy()
    out[~keep] = NEUTRAL
    return out


def crops_from_bbox(rgb: np.ndarray, bbox_data: dict,
                    seg: np.ndarray | None = None,
                    sem_labels: dict | None = None,
                    min_side: int = 4) -> list[Detection]:
    """Cut one :class:`Detection` per row of the bbox annotator output.

    Parameters
    ----------
    rgb         : ``HxWx3`` scene image.
    bbox_data   : the ``bbox_2d_<idx>.json`` dict (``data`` + ``info.idToLabels``).
    seg         : optional semantic-id image (same HxW) for background masking.
    sem_labels  : optional ``semantic_labels_<idx>.json`` (``info`` with idToLabels)
                  mapping seg ids -> class; used to pick which seg ids belong to
                  the box's part for masking.
    min_side    : boxes smaller than this in either dimension are dropped (noise).

    An empty / missing ``data`` yields an empty list — never raises.
    """
    h, w = rgb.shape[:2]
    rows = bbox_data.get("data", []) if isinstance(bbox_data, dict) else []
    id2label = (bbox_data.get("info", {}) or {}).get("idToLabels", {}) if isinstance(bbox_data, dict) else {}

    # class -> set of seg ids, for optional masking
    seg_ids_for_class: dict[str, set[int]] = {}
    if sem_labels is not None:
        sem_map = (sem_labels.get("info", sem_labels) or {}).get("idToLabels", {})
        for sid, meta in sem_map.items():
            cls = (meta or {}).get("class", "").lower()
            seg_ids_for_class.setdefault(cls, set()).add(int(sid))

    out: list[Detection] = []
    for inst_id, row in enumerate(rows):
        if len(row) < 5:
            continue
        sem_id = int(row[0])
        raw_label = (id2label.get(str(sem_id), {}) or {}).get("class", "")
        x0, y0, x1, y1 = _clamp_box(row[1], row[2], row[3], row[4], w, h)
        if (x1 - x0) < min_side or (y1 - y0) < min_side:
            continue
        occ = float(row[5]) if len(row) > 5 else 0.0
        crop = rgb[y0:y1, x0:x1].copy()
        seg_crop = seg[y0:y1, x0:x1] if seg is not None else None
        keep_ids = seg_ids_for_class.get(raw_label.lower(), set())
        crop = _mask_crop(crop, seg_crop, keep_ids)
        out.append(Detection(
            instance_id=inst_id,
            part=canonical_part(raw_label),
            bbox_2d=[x0, y0, x1, y1],
            crop=crop,
            occlusion=occ,
            raw_label=raw_label,
        ))
    return out
