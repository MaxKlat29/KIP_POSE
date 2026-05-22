#!/usr/bin/env python3
"""End-to-end POSE pipeline runner: SDG scene -> pose_result.json (S-008 + S-009).

    python -m pipeline.run_pipeline data/examples/dummy_scene \
        --out data/output/pose_result.json

Stages:
  1. S-008 inference  (pipeline.inference.run_inference): scene -> per-detection
     face + confidence + bbox + crop.
  2. S-009 alignment  (pipeline.alignment.align_detection): per detection -> yaw
     (template MSE-min) -> R_world = Rz(yaw)·R_face (world = R @ body); upright
     from the registry face tilt.
  3. back-projection  (pipeline.backproject): bbox centre -> t_world on the table
     plane with the documented top-down default intrinsics.
  4. assemble + **validate** the pose_result against docs/pose_result.schema.json
     (pure-stdlib gate always; jsonschema gate when available) BEFORE writing.

Green with no trained checkpoint (classifier fallback) and on a bare python3
(stdlib schema gate). A scene with no detections yields an empty, schema-valid
results list — still a valid pose_result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make the repo root importable regardless of launch style.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.alignment import YAW_STEP_DEG, align_detection                # noqa: E402
from pipeline.backproject import Intrinsics, bbox_center_to_world           # noqa: E402
from pipeline.crop import load_scene                                        # noqa: E402
from pipeline.inference import run_inference                                # noqa: E402
from pipeline.registry_paths import load_part_registry                     # noqa: E402
from pipeline.schema import (SCHEMA_VERSION, jsonschema_available,         # noqa: E402
                             validate_or_raise)

COORD_CONVENTION = (
    "Z-up world; column rotation world = R @ body (same as faces_<part>.json "
    "registry); origin = table-plane null-point"
)


def build_pose_result(scene_dir: str, idx: int = 0,
                      intr: Intrinsics | None = None,
                      yaw_step_deg: float = YAW_STEP_DEG,
                      use_seg: bool = True,
                      warn=print) -> dict:
    """Run S-008 + S-009 over scene ``idx`` and return the pose_result dict.

    Intrinsics default to the documented top-down ``render_dataset`` model sized
    to the actual image. The result is NOT yet validated — :func:`run` validates
    before writing.
    """
    rgb, _bbox, _sem = load_scene(scene_dir, idx)
    h, w = rgb.shape[:2]
    if intr is None:
        intr = Intrinsics.for_image(width=w, height=h)

    inferred = run_inference(scene_dir, idx, use_seg=use_seg, warn=warn)

    # cache part registries so we read each faces_<part>.json once
    reg_cache: dict[str, object] = {}

    results = []
    for det in inferred:
        if det.part not in reg_cache:
            reg_cache[det.part] = load_part_registry(det.part)
        registry = reg_cache[det.part]
        aligned = align_detection(det.part, det.face, det.crop_ref,
                                  registry, step_deg=yaw_step_deg)
        t_world = bbox_center_to_world(det.bbox_2d, intr,
                                       rest_height=aligned.rest_height)
        results.append({
            "instance_id": int(det.instance_id),
            "part": det.part,
            "face": aligned.face_name,
            "R_world": [float(v) for v in aligned.R_world],
            "t_world": [float(v) for v in t_world],
            "confidence": float(max(0.0, min(1.0, det.confidence))),
            "bbox_2d": [int(v) for v in det.bbox_2d],
            "upright": bool(aligned.upright),
        })

    source_image = os.path.join(scene_dir, f"rgb_{idx:04d}.png")
    return {
        "meta": {
            "source_image": source_image,
            "table_origin": [float(v) for v in intr.table_origin],
            "units": "m",
            "coordinate_convention": COORD_CONVENTION,
            "schema_version": SCHEMA_VERSION,
        },
        "results": results,
    }


def run(scene_dir: str, out_path: str, idx: int = 0,
        intr: Intrinsics | None = None,
        yaw_step_deg: float = YAW_STEP_DEG,
        use_seg: bool = True,
        warn=print) -> dict:
    """Build, validate, and write a pose_result. Returns the validated dict.

    Raises :class:`pipeline.schema.PoseResultInvalid` if the assembled result is
    not contract-valid — the file is written only on a passing gate.
    """
    doc = build_pose_result(scene_dir, idx, intr=intr,
                            yaw_step_deg=yaw_step_deg, use_seg=use_seg, warn=warn)
    validate_or_raise(doc)                              # contract gate before write
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    gate = "stdlib + jsonschema" if jsonschema_available() else "stdlib"
    warn(f"[run_pipeline] wrote {out_path} — {len(doc['results'])} parts, "
         f"schema gate PASS ({gate})")
    return doc


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="POSE E2E pipeline: SDG scene -> pose_result.json")
    ap.add_argument("scene_dir", help="dir with rgb_<idx>.png + bbox_2d_<idx>.json + (optional) semantic_labels")
    ap.add_argument("--idx", type=int, default=0, help="render index in the scene dir")
    ap.add_argument("--out", default="data/output/pose_result.json", help="output pose_result.json path")
    ap.add_argument("--yaw-step", type=float, default=YAW_STEP_DEG, help="yaw search step in degrees")
    ap.add_argument("--cam-h", type=float, default=None, help="camera height above table (m); default 0.16")
    ap.add_argument("--focal", type=float, default=None, help="focal length (mm); default 24.0")
    ap.add_argument("--no-seg", action="store_true", help="skip semantic-seg background masking of crops")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    intr = None
    if a.cam_h is not None or a.focal is not None:
        rgb, _b, _s = load_scene(a.scene_dir, a.idx)
        h, w = rgb.shape[:2]
        kw = {}
        if a.cam_h is not None:
            kw["cam_h"] = a.cam_h
        if a.focal is not None:
            kw["focal_mm"] = a.focal
        intr = Intrinsics.for_image(width=w, height=h, **kw)
    doc = run(a.scene_dir, a.out, idx=a.idx, intr=intr,
              yaw_step_deg=a.yaw_step, use_seg=not a.no_seg)
    print(f"[run_pipeline] {len(doc['results'])} parts -> {a.out}")
    for r in doc["results"]:
        print(f"  #{r['instance_id']:>2} {r['part']:<22} {r['face']:<10} "
              f"conf={r['confidence']:.2f} t={[round(v, 3) for v in r['t_world']]} "
              f"upright={r['upright']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
