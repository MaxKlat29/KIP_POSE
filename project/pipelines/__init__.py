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

__all__ = ["PipelineAdapter", "PipelineResult", "contract", "registry"]
