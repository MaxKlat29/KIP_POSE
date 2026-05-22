"""POSE inference + alignment pipeline.

Turns an SDG scene image plus 2D detections (bbox_2d + labels) into a
schema-valid ``pose_result.json`` (see ``docs/pose_result.schema.json``).

Two stages, wired by ``run_pipeline.py``:

  S-008  inference  : crop per bbox -> faces.classifier.infer(crop, part)
                      -> [{instance_id, part, face, confidence, bbox_2d, crop_ref}]
  S-009  alignment  : per detection -> in-plane yaw (template MSE-min) composed
                      onto the registry R_face -> R_world (world = R @ body);
                      bbox-centre back-projected onto the table plane -> t_world.

No trained checkpoint is required: the classifier falls back to nearest-template
matching, so the whole chain is end-to-end green out of the box.
"""
from __future__ import annotations

__all__ = [
    "crop",
    "inference",
    "alignment",
    "backproject",
    "run_pipeline",
    "schema",
]
