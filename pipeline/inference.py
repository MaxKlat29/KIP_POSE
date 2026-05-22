#!/usr/bin/env python3
"""S-008 — inference stage: SDG scene -> face-id per detection.

Loads an SDG scene render (rgb + bbox_2d + optional semantic seg), cuts a crop
per detection, routes each crop to the right face-classifier by its part label
(Kai's ``faces.classifier.infer.infer(crop, part)``), and returns the
intermediate per-detection list that the alignment stage (S-009) consumes:

    [{instance_id, part, face, confidence, bbox_2d, crop_ref, occlusion, ...}]

Routing is purely by part name — ``infer`` itself picks the model (checkpoint or
nearest-template fallback) via ``faces/classifier/registry.py``. So this module
never imports torch and runs end-to-end with no trained checkpoint.

Robustness contract (S-008 AKs):
  * unknown part label  -> a warning is recorded, the detection is SKIPPED
    (kept out of the result, no crash);
  * empty / no-bbox scene -> an empty list (the producer then writes an empty,
    schema-valid pose_result).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Make the repo root importable so ``faces.classifier`` resolves regardless of
# how the pipeline is launched (module, script, or cwd elsewhere).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.crop import Detection, crops_from_bbox, load_scene, _load_seg  # noqa: E402


@dataclass
class InferredDetection:
    """A detection enriched with the classifier's face + confidence (S-008 output)."""

    instance_id: int
    part: str
    face: str
    confidence: float
    bbox_2d: list[int]
    crop_ref: np.ndarray = field(repr=False)          # the crop the face came from
    occlusion: float = 0.0
    backend: str = "fallback"                          # 'checkpoint' | 'fallback'
    raw_label: str = ""


def _known_parts() -> set[str]:
    try:
        from faces.classifier import registry  # type: ignore
        return set(registry.PART_TO_MODEL.keys())
    except Exception:
        return set()


def infer_detections(detections: list[Detection],
                     warn=print) -> list[InferredDetection]:
    """Run the face classifier over a list of crops, routing by part.

    Unknown part labels are warned about and skipped (no crash). Returns the
    enriched list; an empty input yields an empty list.
    """
    # lazy import: keeps numpy-only callers (crop tests) torch-free, and lets the
    # fallback path work without any ML deps installed.
    from faces.classifier.infer import backend as infer_backend
    from faces.classifier.infer import infer as infer_face

    known = _known_parts()
    out: list[InferredDetection] = []
    for det in detections:
        if known and det.part not in known:
            warn(f"[inference] unknown part label '{det.raw_label}' "
                 f"(canonical '{det.part}') — no model/registry, skipping "
                 f"instance {det.instance_id}")
            continue
        try:
            res = infer_face(det.crop, det.part)
            be = infer_backend(det.part)
        except Exception as exc:                       # infer() is contracted not to
            warn(f"[inference] infer() failed for part '{det.part}' "       # raise,
                 f"instance {det.instance_id}: {exc!r} — skipping")          # belt+braces
            continue
        out.append(InferredDetection(
            instance_id=det.instance_id,
            part=det.part,
            face=str(res.get("face", "")),
            confidence=float(res.get("confidence", 0.0)),
            bbox_2d=list(det.bbox_2d),
            crop_ref=det.crop,
            occlusion=det.occlusion,
            backend=be,
            raw_label=det.raw_label,
        ))
    return out


def run_inference(scene_dir: str, idx: int = 0,
                  use_seg: bool = True,
                  warn=print) -> list[InferredDetection]:
    """End-to-end S-008: load scene ``idx`` from ``scene_dir`` -> inferred list.

    ``use_seg`` background-masks crops with the semantic seg map when present.
    A scene with no detections returns an empty list (schema-valid downstream).
    """
    rgb, bbox_data, sem_labels = load_scene(scene_dir, idx)
    seg = _load_seg(scene_dir, idx) if use_seg else None
    dets = crops_from_bbox(rgb, bbox_data, seg=seg, sem_labels=sem_labels)
    if not dets:
        warn(f"[inference] scene {scene_dir} render {idx}: no usable detections")
        return []
    return infer_detections(dets, warn=warn)


if __name__ == "__main__":                             # smoke CLI
    import argparse
    ap = argparse.ArgumentParser(description="S-008 inference smoke")
    ap.add_argument("scene_dir")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    res = run_inference(a.scene_dir, a.idx)
    print(f"{len(res)} detections inferred:")
    for d in res:
        print(f"  #{d.instance_id:>2} {d.part:<22} {d.face:<10} "
              f"conf={d.confidence:.3f} backend={d.backend} bbox={d.bbox_2d}")
