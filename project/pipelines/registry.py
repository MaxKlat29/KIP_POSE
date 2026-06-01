"""registry.py — id -> PipelineAdapter-Registry fuer Harness + API + Dropdown.

Die Referenz-Pipeline (gdrnpp) registriert sich automatisch beim Import. Fremde
Pipelines registrieren sich selbst beim Import ihres adapter.py (siehe
_template/adapter.py, untere Zeilen) — ABER erst wenn sie wirklich angebunden sind.
"""
from __future__ import annotations

from .base import PipelineAdapter

_REGISTRY: "dict[str, PipelineAdapter]" = {}


def register(adapter: PipelineAdapter) -> PipelineAdapter:
    """Adapter-Instanz registrieren. Doppelte id -> Fehler (kein stilles Ueberschreiben)."""
    if not isinstance(adapter, PipelineAdapter):
        raise TypeError(f"register() erwartet eine PipelineAdapter-Instanz, nicht {type(adapter)!r}")
    if adapter.id in _REGISTRY:
        raise ValueError(f"Pipeline-id {adapter.id!r} ist bereits registriert")
    _REGISTRY[adapter.id] = adapter
    return adapter


def get(pipeline_id: str) -> PipelineAdapter:
    if pipeline_id not in _REGISTRY:
        raise KeyError(f"Unbekannte Pipeline {pipeline_id!r}. Registriert: {sorted(_REGISTRY)}")
    return _REGISTRY[pipeline_id]


def has(pipeline_id: str) -> bool:
    return pipeline_id in _REGISTRY


def all_pipelines() -> list:
    """Metadaten aller registrierten Pipelines (fuer /api/pipelines + Dropdown)."""
    return [
        {"id": a.id, "name": a.name, "description": a.description, "available": bool(a.available)}
        for a in _REGISTRY.values()
    ]


def available_adapters() -> list:
    """Nur die tatsaechlich lauffaehigen Pipelines (available=True)."""
    return [a for a in _REGISTRY.values() if a.available]


def available_ids() -> list:
    return [a.id for a in _REGISTRY.values() if a.available]


def _autoload_reference() -> None:
    """Registriert die eingebaute Referenz (gdrnpp). Defensiv gekapselt: eine
    fehlende optionale Dependency darf `import pipelines` NIE brechen."""
    try:
        from .gdrnpp_adapter import GdrnppReferenceAdapter
        if "gdrnpp" not in _REGISTRY:
            register(GdrnppReferenceAdapter())
    except Exception:  # noqa: BLE001 — Referenz ist best-effort beim Import
        pass


_autoload_reference()
