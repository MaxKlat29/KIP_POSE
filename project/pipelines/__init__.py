"""pipelines — pluggable Multi-Pipeline-Vergleichs-Seam fuer KIP_POSE.

Erlaubt es, KOMPLETT ANDERE (eigenstaendige) 6D-Pose-Pipelines hinter einer
gemeinsamen Schnittstelle gegeneinander zu vergleichen — Genauigkeit (vs GT),
visueller Side-by-Side, Latenz, Robustheit/Coverage.

Eine "Pipeline" ist ein vollstaendiges System Bild -> 6D-Posen. Die gelieferte
Hauptlinie (gdrnpp: YOLOv8-OBB -> GDRNPP -> bop_adapter) ist nur EINE Auspraegung
davon und dient als Referenz/Baseline. Fremde Pipelines docken als rohe Python-
Projekte unter ``pipelines/<id>/vendor/`` an, gekapselt durch einen Adapter, der
ihren Output auf den eingefrorenen ``pose_result.schema.json``-Contract bringt.

Siehe README.md (Integrations-Playbook) + ../docs/PIPELINE_INTEGRATION.md.

Status 2026-06-01: SCAFFOLD — nur die Huelle. Fremde Pipelines sind leer
(NotImplementedError), kommen spaeter von Max.
"""
from __future__ import annotations

from .base import PipelineAdapter, PipelineResult
from . import contract, registry


def _autoload_combos() -> None:
    """Registriert die 6 NICHT-A-ComposedPipelines (Kombi 2..7, CONTRACT.md §5).
    Defensiv gekapselt — genau wie registry._autoload_reference: ein Fehler beim
    Bauen einer Kombi (z.B. fehlende optionale Dependency) darf `import pipelines`
    NIE brechen. Der Live-Monolith (gdrnpp = Pipeline A) bleibt unberuehrt."""
    try:
        from . import combos
        combos.register_combos()
    except Exception:  # noqa: BLE001 — Kombis sind best-effort beim Import
        pass


_autoload_combos()

__all__ = ["PipelineAdapter", "PipelineResult", "contract", "registry"]
