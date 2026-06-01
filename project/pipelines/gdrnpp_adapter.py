"""gdrnpp_adapter.py — REFERENZ-Adapter, der die gelieferte GDRNPP-Pipeline wrappt.

Das ist "Pipeline A": die bestehende Hauptlinie (YOLOv8-OBB -> GDRNPP -> bop_adapter),
durch den PipelineAdapter-Seam exponiert, damit der Vergleichs-Harness eine BASELINE
hat, gegen die fremde Pipelines gemessen werden.

WICHTIG: das ist eine DUENNE Huelle um e2e_infer.run — sie aendert den Live-Web-
Viewer-Pfad (kip_server) in KEINER Weise. Sie wird NUR vom Vergleich genutzt.

Lokal lauffaehig: e2e_infer faellt ohne GDRNPP-Checkpoint auf MOCK zurueck, der
Detektor auf SDG-/Dummy-Boxen — der Seam ist also auch ohne GPU-Box ausfuehrbar.
Der echte Detektor/GDRNPP greift nur auf der Box (mit Checkpoints).
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

from .base import PipelineAdapter
from . import contract

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


class GdrnppReferenceAdapter(PipelineAdapter):
    id = "gdrnpp"
    name = "GDRNPP (RGB, Referenz)"
    description = (
        "Gelieferte Hauptlinie: YOLOv8-OBB Detektor -> GDRNPP per-Objekt -> "
        "bop_adapter (World-Transform + Boden-Snap). Baseline fuer den Vergleich."
    )

    @property
    def available(self) -> bool:
        # e2e_infer ist lokal importierbar (Mock-Fallback) -> der Seam ist immer
        # ausfuehrbar. Der ECHTE Detektor/GDRNPP greift nur auf der Box.
        try:
            import e2e_infer  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def infer(self, image_path: str, camera: dict, table_origin: list) -> dict:
        """Referenz-Pipeline auf ein Bild. e2e_infer.run loest Kamera + table_origin
        selbst auf (load_scene_camera / TABLE_ORIGIN_SCENE) und schreibt bereits ein
        frozen-schema-valides Doc; die camera/table_origin-Args des Contracts werden
        hier daher ignoriert (fuer fremde Pipelines sind sie verpflichtend)."""
        import e2e_infer
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "pose_result.json"
            doc = e2e_infer.run(str(image_path), out, warn=lambda *a, **k: None)
        return contract.assert_valid(doc)
