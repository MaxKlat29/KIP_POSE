#!/usr/bin/env python3
"""Tests fuer die pure (fastapi-/httpx-freie) Proxy-Logik (S-013 / T-139).

Prueft die Naht, die kip_server:8077 als Proxy fuer die 6 NICHT-A-Kombis nutzt:

  * Kombi-id ↔ Gateway-source-Map (yolo-seg→'yolo', combo-id-Suffix→pose-source).
  * Pipeline-A-Guard: pipeline=gdrnpp ODER seg=yolo-obb&pose=gdrnpp ODER leer.
  * resolve_combo_id: 2-Achsen-Form (seg,pose) → Kombi-id; ungueltig → InvalidCombo.
  * gateway_predict_to_pose_result: Gateway-Antwort (T_cam_obj) → frozen pose_result
    (gleiche Mathe wie composed.py — §6-CamelCase-Mapping, §1-mm-Einheit, Klassenfilter).
  * pipelines_status + unavailable_reason: service_down vs training (Mia S-010 §12 Pflicht).

Lauf:  .venv/bin/python -m pytest project/tests/test_gateway_proxy.py -q
"""
from __future__ import annotations

import base64
import io
import pathlib
import sys

import numpy as np
import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from pipelines import contract  # noqa: E402
from pipelines import gateway_proxy as g  # noqa: E402
from pipelines.combos import COMBO_WHITELIST  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _mask_b64(x0, y0, x1, y1, hw=(64, 64)) -> str:
    from PIL import Image
    m = np.zeros(hw, dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture()
def camera():
    return {"cam_K": [600.0, 0, 32.0, 0, 600.0, 32.0, 0, 0, 1],
            "cam_R_w2c": list(np.eye(3).reshape(9)),
            "cam_t_w2c": [0.0, 0.0, 500.0]}


def _gateway_resp(*classes):
    T = np.eye(4); T[2, 3] = 0.30
    insts = []
    for i, cls in enumerate(classes):
        insts.append({"id": i, "class": cls, "conf": 0.9 - 0.1 * i,
                      "T_cam_obj": T.tolist(), "mask_b64": _mask_b64(5 + i, 5 + i, 25 + i, 25 + i)})
    return {"instances": insts, "num_detections": len(insts),
            "timings": {"seg_ms": 10.0, "pose_ms": 50.0, "num_posed": len(insts)}}


# ── 1) Kombi-id ↔ Gateway-source-Map ──────────────────────────────────────────
def test_combo_to_gateway_covers_exactly_6_non_a():
    assert len(g.COMBO_TO_GATEWAY) == 6, "genau die 6 NICHT-A-Kombis proxyen (Pipeline A nicht)"
    assert "gdrnpp" not in g.COMBO_TO_GATEWAY, "Kombi 1 (Pipeline A) laeuft NICHT ueber das Gateway"


def test_seg_source_maps_yolo_seg_to_gateway_yolo():
    # combos-seg 'yolo-seg' == Gateway INFER_SOURCES 'yolo' (gateway/app.py:51-54).
    assert g.combo_to_gateway("yolo_seg__foundationpose")["seg_source"] == "yolo"
    assert g.combo_to_gateway("sam3__foundationpose")["seg_source"] == "sam3"


def test_pose_source_is_combo_id_suffix():
    # Gateway POSE_SOURCES ids == combo-id-Suffix (foundationpose, gigapose_rgbd, gigapose_rgb).
    assert g.combo_to_gateway("yolo_seg__gigapose_rgbd")["pose_source"] == "gigapose_rgbd"
    assert g.combo_to_gateway("sam3__gigapose_rgb")["pose_source"] == "gigapose_rgb"


def test_needs_depth_matches_whitelist():
    by_id = {c["id"]: c for c in COMBO_WHITELIST}
    for cid, gw in g.COMBO_TO_GATEWAY.items():
        assert gw["needs_depth"] == bool(by_id[cid]["needs_depth"]), cid


def test_combo_to_gateway_rejects_pipeline_a_and_unknown():
    with pytest.raises(g.InvalidCombo):
        g.combo_to_gateway("gdrnpp")                  # Pipeline A laeuft nicht ueber Gateway
    with pytest.raises(g.InvalidCombo):
        g.combo_to_gateway("does_not_exist")


# ── 2) Pipeline-A-Guard ───────────────────────────────────────────────────────
def test_is_pipeline_a_all_forms():
    assert g.is_pipeline_a(pipeline="gdrnpp")
    assert g.is_pipeline_a(pipeline="")                # leer = Default = Pipeline A
    assert g.is_pipeline_a(pipeline=None)
    assert g.is_pipeline_a(seg="yolo-obb", pose="gdrnpp")
    # NICHT Pipeline A:
    assert not g.is_pipeline_a(pipeline="yolo_seg__foundationpose")
    assert not g.is_pipeline_a(seg="yolo-seg", pose="FoundationPose")
    assert not g.is_pipeline_a(seg="yolo-obb", pose="FoundationPose")  # falsche Kopplung


# ── 3) resolve_combo_id (2-Achsen + Fehler) ───────────────────────────────────
def test_resolve_combo_id_pipeline_a():
    assert g.resolve_combo_id(pipeline="gdrnpp") == "gdrnpp"
    assert g.resolve_combo_id(seg="yolo-obb", pose="gdrnpp") == "gdrnpp"
    assert g.resolve_combo_id(pipeline="") == "gdrnpp"


def test_resolve_combo_id_by_combo_id():
    for cid in g.COMBO_TO_GATEWAY:
        assert g.resolve_combo_id(pipeline=cid) == cid


def test_resolve_combo_id_by_seg_pose_axes():
    # FE-Achsen (Labels aus Mia S-010 §2) → Kombi-id.
    assert g.resolve_combo_id(seg="yolo-seg", pose="FoundationPose") == "yolo_seg__foundationpose"
    assert g.resolve_combo_id(seg="sam3", pose="FoundationPose") == "sam3__foundationpose"
    assert g.resolve_combo_id(seg="yolo-seg", pose="GigaPose-3D") == "yolo_seg__gigapose_rgbd"
    assert g.resolve_combo_id(seg="yolo-seg", pose="GigaPose-2D") == "yolo_seg__gigapose_rgb"
    assert g.resolve_combo_id(seg="sam3", pose="GigaPose-3D") == "sam3__gigapose_rgbd"
    assert g.resolve_combo_id(seg="sam3", pose="GigaPose-2D") == "sam3__gigapose_rgb"
    # Auch Gateway-pose-ids direkt:
    assert g.resolve_combo_id(seg="yolo-seg", pose="gigapose_rgbd") == "yolo_seg__gigapose_rgbd"


def test_resolve_combo_id_invalid_raises():
    # yolo-obb nur mit gdrnpp (§4) — yolo-obb + FoundationPose ist KEINE Whitelist-Kombi.
    with pytest.raises(g.InvalidCombo):
        g.resolve_combo_id(seg="yolo-obb", pose="FoundationPose")
    with pytest.raises(g.InvalidCombo):
        g.resolve_combo_id(seg="bogus", pose="FoundationPose")
    with pytest.raises(g.InvalidCombo):
        g.resolve_combo_id(pipeline="not_a_combo")


# ── 4) gateway_predict_to_pose_result (Mapping) ───────────────────────────────
def test_mapping_produces_valid_pose_result(camera):
    resp = _gateway_resp("anker_kurz", "anker_lang")
    doc = g.gateway_predict_to_pose_result(resp, camera=camera,
                                           table_origin=[0, 0, 0.08],
                                           source_image="t", snap=False)
    assert contract.validate(doc) == [], contract.validate(doc)
    assert len(doc["results"]) == 2
    assert {r["part"] for r in doc["results"]} == {"Anker_Kurz", "Anker_Lang"}  # §6 CamelCase


def test_mapping_filters_unknown_class(camera):
    resp = _gateway_resp("anker_kurz", "schraube", "anker_lang")
    doc = g.gateway_predict_to_pose_result(resp, camera=camera,
                                           table_origin=[0, 0, 0.08],
                                           source_image="t", snap=False)
    # 'schraube' ist nicht im 2-Klassen-Scope (§6) → gefiltert.
    assert len(doc["results"]) == 2
    assert all(r["part"] in ("Anker_Kurz", "Anker_Lang") for r in doc["results"])


def test_mapping_skips_instances_without_pose(camera):
    # §3-Skip: eine Instanz ohne T_cam_obj (Gateway hat sie skippt) → fehlt im Doc.
    resp = _gateway_resp("anker_kurz", "anker_lang")
    resp["instances"][1]["T_cam_obj"] = None
    doc = g.gateway_predict_to_pose_result(resp, camera=camera,
                                           table_origin=[0, 0, 0.08],
                                           source_image="t", snap=False)
    assert len(doc["results"]) == 1


def test_mapping_uses_mm_unit_correctly(camera):
    # §1: T_cam_obj ist Meter; bop_pose_to_world braucht mm. Identitaet 0.30 m vor Cam,
    # R_w2c=I, t_w2c=500 mm → t_world.z = (300 - 500) mm = -0.2 m; minus table z 0.08.
    resp = _gateway_resp("anker_kurz")
    doc = g.gateway_predict_to_pose_result(resp, camera=camera,
                                           table_origin=[0, 0, 0.08],
                                           source_image="t", snap=False)
    t = doc["results"][0]["t_world"]
    assert np.allclose(t, [0.0, 0.0, -0.2 - 0.08], atol=1e-6), t


def test_mapping_requires_extrinsics():
    with pytest.raises(ValueError):
        g.gateway_predict_to_pose_result(_gateway_resp("anker_kurz"),
                                         camera={"cam_K": [1] * 9},
                                         table_origin=[0, 0, 0.08], source_image="t")


# ── 5) pipelines_status + unavailable_reason (Mia S-010 §12) ──────────────────
def test_pipelines_status_has_all_mia_fields():
    ps = g.pipelines_status(gateway_health=None, live_pipeline_a_available=True)
    assert len(ps) == 12  # T-147: Gating-Relax 7→12 (volle FEASIBLE_COMBOS-Matrix)
    for p in ps:
        for f in ("id", "name", "seg", "pose", "needs_depth", "is_pipeline_a",
                  "available", "unavailable_reason"):
            assert f in p, f"{p['id']} fehlt {f}"


def test_pipeline_a_available_when_live_path_alive():
    ps = g.pipelines_status(gateway_health=None, live_pipeline_a_available=True)
    a = next(p for p in ps if p["is_pipeline_a"])
    assert a["available"] is True and a["unavailable_reason"] is None
    # Live-Pfad tot → Kombi 1 nicht available (kein Reason ist service_down hier).
    ps2 = g.pipelines_status(gateway_health=None, live_pipeline_a_available=False)
    a2 = next(p for p in ps2 if p["is_pipeline_a"])
    assert a2["available"] is False


def test_gateway_down_makes_non_a_service_down():
    ps = g.pipelines_status(gateway_health=None, live_pipeline_a_available=True)
    for p in ps:
        if p["is_pipeline_a"]:
            continue
        assert p["available"] is False
        assert p["unavailable_reason"] == g.REASON_SERVICE_DOWN


def test_available_when_both_stages_up():
    gh = {"ok": True, "yolo": {"ok": True}, "fp": {"ok": True},
          "gigapose": {"status": "ok"}, "sam3": {"ok": True}}
    ps = g.pipelines_status(gateway_health=gh, live_pipeline_a_available=True)
    by = {p["id"]: p for p in ps}
    # alle 6 NICHT-A available (alle Services up, kein Training).
    for cid in g.COMBO_TO_GATEWAY:
        assert by[cid]["available"] is True, cid
        assert by[cid]["unavailable_reason"] is None


def test_training_reason_takes_precedence_over_service_down():
    # yolo up, fp up, gigapose loading, sam3 down; yolo-seg trainiert (S-008).
    gh = {"ok": True, "yolo": {"ok": True}, "fp": {"ok": True},
          "gigapose": {"status": "loading"}, "sam3": {"ok": False}}
    ps = g.pipelines_status(gateway_health=gh, live_pipeline_a_available=True,
                            training_segs={"yolo-seg"})
    by = {p["id"]: p for p in ps}
    # yolo-seg-Kombis: training (auch wenn pose-Service up waere).
    assert by["yolo_seg__foundationpose"]["unavailable_reason"] == g.REASON_TRAINING
    assert by["yolo_seg__gigapose_rgbd"]["unavailable_reason"] == g.REASON_TRAINING
    # sam3-Kombis: service_down (sam3 down), NICHT training.
    assert by["sam3__foundationpose"]["unavailable_reason"] == g.REASON_SERVICE_DOWN
    assert by["sam3__gigapose_rgbd"]["unavailable_reason"] == g.REASON_SERVICE_DOWN


def test_partial_health_gigapose_only():
    # nur gigapose up (yolo+fp+sam3 down): nur die gigapose-Kombis mit up-seg available.
    gh = {"ok": False, "yolo": {"ok": True}, "fp": {"ok": False},
          "gigapose": {"status": "ok"}, "sam3": {"ok": False}}
    ps = g.pipelines_status(gateway_health=gh, live_pipeline_a_available=True)
    by = {p["id"]: p for p in ps}
    # yolo up + gigapose up → yolo_seg__gigapose_* available; foundationpose-Kombis down (fp).
    assert by["yolo_seg__gigapose_rgbd"]["available"] is True
    assert by["yolo_seg__gigapose_rgb"]["available"] is True
    assert by["yolo_seg__foundationpose"]["available"] is False        # fp down
    assert by["sam3__gigapose_rgbd"]["available"] is False             # sam3 down


def test_available_combo_ids_helper():
    gh = {"ok": True, "yolo": {"ok": True}, "fp": {"ok": True},
          "gigapose": {"status": "error"}, "sam3": {"ok": False}}
    ps = g.pipelines_status(gateway_health=gh, live_pipeline_a_available=True)
    avail = g.available_combo_ids(ps)
    assert "gdrnpp" in avail                          # Live-Pfad
    assert "yolo_seg__foundationpose" in avail        # yolo+fp up
    assert "sam3__foundationpose" not in avail        # sam3 down
    assert "yolo_seg__gigapose_rgbd" not in avail     # gigapose error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
