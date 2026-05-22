"""Per-part face-view classifier (crop -> face_id).

Public surface:
    from faces.classifier.infer import infer        # infer(crop, part) -> {"face","confidence"}
    from faces.classifier.registry import (model_name, checkpoint_path,
                                           PART_TO_MODEL, classes_for)

Inference works with no checkpoint and no torch (nearest-template fallback);
drop a trained checkpoint at registry.checkpoint_path(part) to upgrade silently.
"""
