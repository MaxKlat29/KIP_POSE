#!/usr/bin/env python3
"""Integrationstests fuer den kip_server-Proxy (S-013 / T-139) gegen ein GEMOCKTES
Gateway (kein Box/GPU noetig — Gateway+Services laufen nicht permanent).

Prueft die HTTP-Naht (FastAPI TestClient):

  * /api/pipelines → 7 Kombis mit Mias Pflicht-Feldern (unavailable_reason etc.).
  * /api/predict Routing:
      - Pipeline A (pipeline=gdrnpp ODER seg=yolo-obb&pose=gdrnpp ODER leer)
        → UNVERAENDERTER Live-Pfad: delegiert byte-identisch an _real_infer_job,
          ruft das Gateway NIE auf (Guard returnt frueh). → {job, mode:'live'}.
      - 6 NICHT-A-Kombis → Gateway-Proxy (gemockt) → pose_result-Doc zurueck.
      - ungueltige Kombi → 400 (niemals 12 Kombis).
      - needs_depth-Kombi ohne Tiefenbild → 400.
  * /api/compare-Gating: <2 available → 501 „braucht >=2"; >=2 → 501 „folgt".

Lauf:  .venv/bin/python -m pytest project/tests/test_kip_server_predict.py -q
"""
from __future__ import annotations

import base64
import importlib
import io
import os
import pathlib
import sys
import tempfile

import numpy as np
import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

# kip_server liest /mnt-Pfade beim Import; in Tests auf ein tmp + leeres root_path
# umlenken, damit der Modul-Import ueberall laeuft (Box hat /mnt, CI nicht).
os.environ.setdefault("KIP_LIVE_ROOT", tempfile.mkdtemp(prefix="kip_live_test_"))
os.environ.setdefault("KIP_BASE_PATH", "")
os.environ.setdefault("KIP_GATEWAY_URL", "http://gateway-mock:8000")

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

import kip_server as K  # noqa: E402
from pipelines import contract  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(K.app)


def _png_bytes(hw=(64, 64), val=0, mode="RGB"):
    from PIL import Image
    if mode == "RGB":
        arr = np.full((*hw, 3), val, dtype=np.uint8)
    else:
        arr = np.full(hw, val, dtype=np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mask_b64(x0, y0, x1, y1, hw=(64, 64)) -> str:
    from PIL import Image
    m = np.zeros(hw, dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Explizite Welt-Extrinsics (Upload-Tab-Form): I + 0.5 m vor der Kamera. Damit braucht
# der Proxy-Mapping-Schritt KEIN Zivid-val-Template (das gibt es nur auf der Box).
_EXTRINSICS = {
    "cam_R_w2c": "1 0 0 0 1 0 0 0 1",
    "cam_t_w2c": "0 0 500",
}


def _gateway_resp():
    T = np.eye(4); T[2, 3] = 0.30
    return {
        "width": 64, "height": 64, "K": {"fx": 600, "fy": 600, "cx": 32, "cy": 32},
        "seg_source": "yolo", "pose_source": "foundationpose", "num_detections": 2,
        "instances": [
            {"id": 0, "class": "anker_kurz", "conf": 0.9, "T_cam_obj": T.tolist(),
             "mask_b64": _mask_b64(5, 5, 25, 25)},
            {"id": 1, "class": "anker_lang", "conf": 0.8, "T_cam_obj": T.tolist(),
             "mask_b64": _mask_b64(30, 30, 60, 60)},
        ],
        "pointcloud": None,
        "timings": {"seg_ms": 11.0, "pose_ms": 53.0, "num_posed": 2},
    }


# ── /api/pipelines ────────────────────────────────────────────────────────────
def test_pipelines_returns_7_with_mia_fields(client, monkeypatch):
    monkeypatch.setattr(K, "_gateway_health", lambda: None)
    monkeypatch.setattr(K, "_live_pipeline_a_available", lambda: True)
    r = client.get("/api/pipelines")
    assert r.status_code == 200
    body = r.json()
    assert body["seam"] == "ok"
    pls = body["pipelines"]
    assert len(pls) == 7
    a = [p for p in pls if p["is_pipeline_a"]]
    assert len(a) == 1 and a[0]["id"] == "gdrnpp" and a[0]["available"] is True
    for p in pls:
        # Mia S-010 §12 Pflicht-Felder:
        assert {"id", "seg", "pose", "needs_depth", "is_pipeline_a",
                "available", "unavailable_reason"} <= set(p)
        # unavailable_reason ist enum oder None:
        assert p["unavailable_reason"] in (None, "service_down", "training")
        # back-compat: alte Felder (name/description/available) noch da.
        assert "name" in p and "description" in p


def test_pipelines_reports_training_reason(client, monkeypatch):
    gh = {"ok": True, "yolo": {"ok": True}, "fp": {"ok": True},
          "gigapose": {"status": "ok"}, "sam3": {"ok": True}}
    monkeypatch.setattr(K, "_gateway_health", lambda: gh)
    monkeypatch.setattr(K, "_live_pipeline_a_available", lambda: True)
    monkeypatch.setattr(K, "KIP_TRAINING_SEGS", {"yolo-seg"})
    pls = {p["id"]: p for p in client.get("/api/pipelines").json()["pipelines"]}
    assert pls["yolo_seg__foundationpose"]["unavailable_reason"] == "training"
    assert pls["sam3__foundationpose"]["available"] is True       # sam3 up, kein Training


# ── /api/predict — Pipeline A (Live-Pfad-Guard, byte-identisch) ───────────────
def test_predict_pipeline_a_delegates_to_live_job_not_gateway(client, monkeypatch):
    """pipeline=gdrnpp → der Guard returnt frueh und startet _real_infer_job
    (UNVERAENDERTER Live-Pfad). Das Gateway wird NIE aufgerufen."""
    calls = {"real_infer_job": 0, "gateway": 0}

    def fake_real_infer_job(job, img_bytes, fname):
        calls["real_infer_job"] += 1
        K._job_set(job, phase="Fertig", pct=100)

    def fake_gateway(*a, **k):
        calls["gateway"] += 1
        return _gateway_resp()

    monkeypatch.setattr(K, "_real_infer_job", fake_real_infer_job)
    monkeypatch.setattr(K, "_gateway_predict_multipart", fake_gateway)

    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files, data={"pipeline": "gdrnpp"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_pipeline_a"] is True and body["mode"] == "live"
    assert "job" in body and body["poll_url"].startswith("api/real/job/")
    assert calls["real_infer_job"] == 1, "Live-Pfad MUSS _real_infer_job aufrufen"
    assert calls["gateway"] == 0, "Pipeline A darf das Gateway NIE aufrufen"


def test_predict_pipeline_a_via_seg_pose_axes(client, monkeypatch):
    """seg=yolo-obb & pose=gdrnpp ist ebenfalls Pipeline A (2-Achsen-Form, S-010)."""
    calls = {"n": 0, "gw": 0}
    monkeypatch.setattr(K, "_real_infer_job",
                        lambda j, b, f: (calls.__setitem__("n", calls["n"] + 1),
                                         K._job_set(j, phase="Fertig", pct=100)))
    monkeypatch.setattr(K, "_gateway_predict_multipart",
                        lambda *a, **k: calls.__setitem__("gw", calls["gw"] + 1))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files,
                    data={"seg": "yolo-obb", "pose": "gdrnpp"})
    assert r.status_code == 200, r.text
    assert r.json()["is_pipeline_a"] is True
    assert calls["n"] == 1 and calls["gw"] == 0


def test_predict_empty_pipeline_defaults_to_pipeline_a(client, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(K, "_real_infer_job",
                        lambda j, b, f: (calls.__setitem__("n", calls["n"] + 1),
                                         K._job_set(j, phase="Fertig", pct=100)))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files, data={"pipeline": ""})
    assert r.status_code == 200 and r.json()["is_pipeline_a"] is True
    assert calls["n"] == 1


# ── /api/predict — 6 NICHT-A-Kombis (Gateway-Proxy, gemockt) ──────────────────
def test_predict_mesh_combo_proxies_to_gateway(client, monkeypatch):
    captured = {}

    def fake_gateway(combo_id, **kw):
        captured["combo_id"] = combo_id
        captured["kw"] = kw
        return _gateway_resp()

    def fake_real_infer_job(*a, **k):
        raise AssertionError("Mesh-Kombi darf den Live-Pfad NIE anfassen")

    monkeypatch.setattr(K, "_gateway_predict_multipart", fake_gateway)
    monkeypatch.setattr(K, "_real_infer_job", fake_real_infer_job)

    files = {"image": ("rgb.png", _png_bytes(), "image/png"),
             "depth": ("depth.png", _png_bytes(mode="L16", val=300), "image/png")}
    r = client.post("/api/predict", files=files, data={
        "pipeline": "yolo_seg__foundationpose",
        "fx": "600", "fy": "600", "cx": "32", "cy": "32", **_EXTRINSICS,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_pipeline_a"] is False and body["mode"] == "mesh"
    assert body["pipeline"] == "yolo_seg__foundationpose"
    assert body["seg_source"] == "yolo" and body["pose_source"] == "foundationpose"
    # Antwort traegt ein VALIDES pose_result (T_cam_obj → Welt gemappt).
    assert contract.validate(body["pose_result"]) == []
    assert len(body["pose_result"]["results"]) == 2
    assert body["timings"]["pose_ms"] == 53.0
    assert captured["combo_id"] == "yolo_seg__foundationpose"


def test_predict_mesh_combo_by_seg_pose_axes(client, monkeypatch):
    monkeypatch.setattr(K, "_gateway_predict_multipart", lambda cid, **k: _gateway_resp())
    files = {"image": ("rgb.png", _png_bytes(), "image/png"),
             "depth": ("depth.png", _png_bytes(mode="L16", val=300), "image/png")}
    r = client.post("/api/predict", files=files, data={
        "seg": "sam3", "pose": "FoundationPose",
        "fx": "600", "fy": "600", "cx": "32", "cy": "32", **_EXTRINSICS,
    })
    assert r.status_code == 200, r.text
    assert r.json()["pipeline"] == "sam3__foundationpose"


def test_predict_invalid_combo_is_400(client, monkeypatch):
    monkeypatch.setattr(K, "_gateway_predict_multipart",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nie aufrufen")))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    # yolo-obb nur mit gdrnpp (§4) → yolo-obb + FoundationPose ist ungueltig.
    r = client.post("/api/predict", files=files, data={
        "seg": "yolo-obb", "pose": "FoundationPose",
        "fx": "600", "fy": "600", "cx": "32", "cy": "32"})
    assert r.status_code == 400


def test_predict_needs_depth_without_depth_is_400(client, monkeypatch):
    monkeypatch.setattr(K, "_gateway_predict_multipart",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nie aufrufen")))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}     # KEIN depth
    r = client.post("/api/predict", files=files, data={
        "pipeline": "yolo_seg__foundationpose",                   # needs_depth=True
        "fx": "600", "fy": "600", "cx": "32", "cy": "32"})
    assert r.status_code == 400
    assert "tiefenbild" in r.json()["detail"].lower()


def test_predict_rgb_only_combo_without_depth_ok(client, monkeypatch):
    # GigaPose-2D (rgb) braucht KEIN Depth → ohne Tiefenbild gueltig.
    monkeypatch.setattr(K, "_gateway_predict_multipart", lambda cid, **k: _gateway_resp())
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files, data={
        "pipeline": "yolo_seg__gigapose_rgb",
        "fx": "600", "fy": "600", "cx": "32", "cy": "32", **_EXTRINSICS})
    assert r.status_code == 200, r.text


def test_predict_mesh_combo_no_extrinsics_no_zivid_is_400(client, monkeypatch):
    # Kein Zivid-Template (kein val auf der Test-Maschine) + keine cam_R/t_w2c-Felder
    # → klare 400 statt 500-Traceback (defensiver Guard).
    monkeypatch.setattr(K, "_gateway_predict_multipart", lambda cid, **k: _gateway_resp())
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files, data={
        "pipeline": "yolo_seg__gigapose_rgb",                 # rgb-only, kein depth noetig
        "fx": "600", "fy": "600", "cx": "32", "cy": "32"})    # KEINE Extrinsics
    assert r.status_code == 400
    assert "extrinsics" in r.json()["detail"].lower()


def test_predict_mesh_combo_missing_K_is_400(client, monkeypatch):
    monkeypatch.setattr(K, "_gateway_predict_multipart", lambda cid, **k: _gateway_resp())
    files = {"image": ("rgb.png", _png_bytes(), "image/png"),
             "depth": ("depth.png", _png_bytes(mode="L16", val=300), "image/png")}
    r = client.post("/api/predict", files=files, data={"pipeline": "yolo_seg__foundationpose"})
    assert r.status_code == 400      # fx/fy/cx/cy fehlen


# ── /api/compare-Gating ───────────────────────────────────────────────────────
def test_compare_needs_two_available(client, monkeypatch):
    # nur Pipeline A available → <2 → 501 „braucht >=2".
    monkeypatch.setattr(K, "_gateway_health", lambda: None)
    monkeypatch.setattr(K, "_live_pipeline_a_available", lambda: True)
    r = client.get("/api/compare")
    assert r.status_code == 501
    assert "braucht >=2" in r.json()["detail"]


def test_compare_active_with_two_available(client, monkeypatch):
    gh = {"ok": True, "yolo": {"ok": True}, "fp": {"ok": True},
          "gigapose": {"status": "ok"}, "sam3": {"ok": True}}
    monkeypatch.setattr(K, "_gateway_health", lambda: gh)
    monkeypatch.setattr(K, "_live_pipeline_a_available", lambda: True)
    r = client.get("/api/compare")
    assert r.status_code == 501
    assert "folgt" in r.json()["detail"].lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
