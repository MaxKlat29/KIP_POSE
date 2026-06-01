#!/usr/bin/env python3
"""Tests fuer das Multi-Pipeline-Vergleichs-Scaffold (T-076).

Prueft die HUELLE (nicht die fremden Pipelines, die sind leer):
  * registry kennt die gdrnpp-Referenz,
  * der eingefrorene Contract akzeptiert ein sauberes Doc + lehnt fehlende Felder ab,
  * assemble_doc leitet face/upright ab und produziert ein schema-valides Doc,
  * das Template wirft NotImplementedError + ist nicht verfuegbar/nicht registriert,
  * world_to_bop_cam ist die exakte Inverse von bop_adapter.bop_pose_to_world,
  * compare_pipelines --dry-run laeuft offline durch.

Lauf:  python3 -m pytest project/tests/test_pipelines_scaffold.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# project/ importierbar machen (gleiche Konvention wie die anderen Tests).
_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

import bop_adapter as A  # noqa: E402
from pipelines import contract, registry  # noqa: E402
import compare_pipelines as C  # noqa: E402


# ── Registry / Referenz ───────────────────────────────────────────────────────
def test_gdrnpp_reference_registered():
    ids = [p["id"] for p in registry.all_pipelines()]
    assert "gdrnpp" in ids, f"gdrnpp-Referenz fehlt in der Registry: {ids}"
    gd = registry.get("gdrnpp")
    assert gd.name and gd.description
    assert gd.available is True  # e2e_infer ist lokal importierbar (Mock-Fallback)


def test_no_empty_stub_floods_registry():
    # Das Template darf NICHT registriert sein (sonst flutet ein leerer Stub das Dropdown).
    assert not registry.has("template")


# ── Contract-Gate (exaktes frozen Schema) ─────────────────────────────────────
def _valid_entry():
    return {
        "instance_id": 0,
        "part": "Anker_Lang",
        "R_world": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "t_world": [0.1, -0.2, 0.0],
        "confidence": 0.9,
        "bbox_2d": [10, 20, 100, 200],
    }


def test_assemble_doc_is_schema_valid():
    doc = contract.assemble_doc("scene_x.png", [_valid_entry()])
    assert contract.validate(doc) == [], contract.validate(doc)
    r = doc["results"][0]
    assert isinstance(r["face"], str) and r["face"]      # analytisch abgeleitet
    assert isinstance(r["upright"], bool)
    assert r["bbox_2d"] == [10, 20, 100, 200]


def test_contract_rejects_missing_face():
    doc = contract.assemble_doc("scene_x.png", [_valid_entry()])
    del doc["results"][0]["face"]
    assert contract.validate(doc), "fehlendes 'face' muss ein Verstoss sein"


def test_contract_rejects_extra_field_when_jsonschema():
    if not contract.jsonschema_available():
        return  # additionalProperties wird nur vom jsonschema-Gate erzwungen
    doc = contract.assemble_doc("scene_x.png", [_valid_entry()])
    doc["results"][0]["color"] = "pred"   # Viewer-Runtime-Feld, NICHT im frozen Schema
    assert contract.validate(doc), "fremdes Feld muss vom jsonschema-Gate abgelehnt werden"


# ── Template-Stub ─────────────────────────────────────────────────────────────
def test_template_stub_raises_not_implemented():
    import importlib.util
    tmpl = _PROJECT / "pipelines" / "_template" / "adapter.py"
    spec = importlib.util.spec_from_file_location("kip_template_adapter", tmpl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.TemplatePipeline()
    assert p.available is False
    try:
        p.infer("x.png", {}, [0, 0, 0.08])
        assert False, "Template-infer haette NotImplementedError werfen muessen"
    except NotImplementedError:
        pass


# ── run_timed: Crash- + Contract-Robustheit ──────────────────────────────────
from pipelines.base import PipelineAdapter  # noqa: E402


class _BadAdapter(PipelineAdapter):
    id, name = "bad", "Bad"
    def infer(self, image_path, camera, table_origin):
        return None                       # vertraglich invalid (kein dict)


class _CrashAdapter(PipelineAdapter):
    id, name = "crash", "Crash"
    def infer(self, image_path, camera, table_origin):
        raise RuntimeError("boom")


class _OkAdapter(PipelineAdapter):
    id, name = "ok", "Ok"
    def infer(self, image_path, camera, table_origin):
        return contract.assemble_doc("scene.png", [_valid_entry()])


def test_run_timed_invalid_output_is_failure():
    pr = _BadAdapter().run_timed("x.png", {}, [0, 0, 0.08])
    assert pr.ok is False and pr.doc is None
    assert "contract" in pr.error or "kein dict" in pr.error


def test_run_timed_crash_is_failure_not_raise():
    pr = _CrashAdapter().run_timed("x.png", {}, [0, 0, 0.08])
    assert pr.ok is False and "boom" in pr.error


def test_run_timed_valid_output_ok():
    pr = _OkAdapter().run_timed("x.png", {}, [0, 0, 0.08])
    assert pr.ok is True and pr.n_results == 1 and pr.doc is not None


# ── Achse 1: World -> BOP-cam Rueckrechnung = Inverse von bop_pose_to_world ────
def _rand_SO3(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_world_to_bop_cam_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(50):
        R_world = _rand_SO3(rng)
        t_world = rng.standard_normal(3) * 0.3          # m
        R_w2c = _rand_SO3(rng)
        t_w2c = rng.standard_normal(3) * 500.0          # mm
        table_origin = np.array([0.0, 0.0, 0.08])
        # world -> cam
        R_m2c, t_m2c = C.world_to_bop_cam(R_world, t_world, R_w2c, t_w2c, table_origin)
        # cam -> world (autoritativer Vorwaerts-Transform)
        R_back, t_back = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
        assert np.allclose(R_back, R_world, atol=1e-9)
        assert np.allclose(t_back, t_world, atol=1e-9)


# ── Harness Dry-Run ───────────────────────────────────────────────────────────
def test_compare_dry_run(tmp_path):
    out = tmp_path / "cmp"
    rc = C.main(["--dry-run", "--out", str(out)])
    assert rc == 0
    assert (out / "compare_result.json").exists()
    assert (out / "COMPARE.md").exists()
    md = (out / "COMPARE.md").read_text()
    assert "Pipeline-Vergleich" in md and "gdrnpp" in md


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
