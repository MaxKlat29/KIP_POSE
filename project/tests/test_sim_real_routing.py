#!/usr/bin/env python3
"""Integrationstests fuer das Sim/Upload-Inferenz-Routing (S-014 / T-140) gegen ein
GEMOCKTES Gateway (kein Box/GPU noetig — Isaac/Detektor/Gateway laufen nicht permanent).

Vor T-140 war der interaktive Sim-/Upload-Pfad auf gdrnpp (Pipeline A) hardcodiert:
`_resolve_pipeline(pipeline)` warf bei jeder Nicht-gdrnpp-Pipeline 501 „Routing fuer
Pipeline '…' folgt — bisher nur gdrnpp live." (= Max' Befund im Sim-Tab). Jetzt routen
`/api/sim/generate_async` + `/api/real/infer_async` die GEWAEHLTE Kombi: Pipeline A
weiter ueber den :8078-Worker, die 6 NICHT-A-Kombis ueber das Gateway.

Geprueft (FastAPI TestClient + pure Naht):
  * sim/generate_async mit seg=sam3&pose=foundationpose → KEIN Stub-Fehler mehr, die
    combo_id geht an _sim_generate_job; Sofort-Antwort traegt used_combo/used_seg/
    used_pose/modality (T-164).
  * pipeline=gdrnpp → Pipeline-A-Kombi (Live-Pfad byte-identisch).
  * ungueltige Kombi → 400 (niemals 12 Kombis).
  * image-only-Guard: seg=gt → 400 (GT-Masken nur Batch-Eval), Sim/Upload.
  * _sim_generate_job/_real_infer_job: NICHT-A faehrt das Gateway (gemockt), Pipeline A
    den Worker; GT-Overlay (blau) NUR im Sim (Upload hat keins, image-only).
  * pure: combo_result_meta-Felder + gateway_instances_to_viewer_preds (color='pred').

Lauf:  .venv/bin/python -m pytest project/tests/test_sim_real_routing.py -q
"""
from __future__ import annotations

import io
import os
import pathlib
import sys
import tempfile

import numpy as np
import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

os.environ.setdefault("KIP_LIVE_ROOT", tempfile.mkdtemp(prefix="kip_live_test_"))
os.environ.setdefault("KIP_BASE_PATH", "")
os.environ.setdefault("KIP_GATEWAY_URL", "http://gateway-mock:8000")

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

import kip_server as K  # noqa: E402
from pipelines import gateway_proxy as gwp  # noqa: E402


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


def _gateway_resp():
    """Gemockte Gateway-/predict-Antwort: 2 Instanzen mit T_cam_obj (0.30 m vor Cam)."""
    T = np.eye(4); T[2, 3] = 0.30
    return {
        "width": 64, "height": 64, "K": {"fx": 600, "fy": 600, "cx": 32, "cy": 32},
        "seg_source": "sam3", "pose_source": "foundationpose", "num_detections": 2,
        "instances": [
            {"id": 0, "class": "anker_kurz", "conf": 0.9, "T_cam_obj": T.tolist()},
            {"id": 1, "class": "anker_lang", "conf": 0.8, "T_cam_obj": T.tolist()},
        ],
        "timings": {"seg_ms": 12.0, "pose_ms": 60.0, "num_posed": 2},
    }


# ── pure: combo_result_meta (T-164 Result-Felder) ────────────────────────
def test_combo_result_meta_for_all_modes():
    """used_* + modality fuer Pipeline A, eine RGB-D- und eine RGB-Kombi."""
    a = gwp.combo_result_meta("gdrnpp")
    assert a["used_combo"] == "gdrnpp" and a["used_seg"] == "yolo-obb"
    assert a["used_pose"] == "GDRNPP" and a["modality"] == "RGB"

    fp = gwp.combo_result_meta("sam3__foundationpose")
    assert fp["used_seg"] == "sam3" and fp["used_pose"] == "FoundationPose"
    assert fp["modality"] == "RGBD" and fp["needs_depth"] is True

    g2 = gwp.combo_result_meta("yolo_seg__gigapose_rgb")
    assert g2["used_pose"] == "GigaPose-2D" and g2["modality"] == "RGB"


def test_combo_result_meta_unknown_raises():
    with pytest.raises(gwp.InvalidCombo):
        gwp.combo_result_meta("bogus__combo")


# ── pure: gateway_instances_to_viewer_preds (color='pred', Viewer-Overlay-Form) ─
def test_gateway_instances_to_viewer_preds_shape():
    """Gateway-Instanzen → Viewer-Pred-Eintraege: alle color='pred', Viewer-Felder da,
    keine GT-Faerbung (image-only — das Gateway liefert NIE GT)."""
    R_w2c = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    t_w2c = [0.0, 0.0, 500.0]                     # 0.5 m vor der Kamera (mm)
    preds = gwp.gateway_instances_to_viewer_preds(
        _gateway_resp(), R_w2c=R_w2c, t_w2c=t_w2c, table_origin=[0.0, 0.0, 0.08],
        start_inst_id=5, snap=False)
    assert len(preds) == 2
    for p in preds:
        assert p["color"] == "pred"               # NIE 'gt' aus dem Gateway
        assert {"instance_id", "part", "face", "confidence",
                "t_world", "R_world", "upright", "color"} <= set(p)
        assert len(p["t_world"]) == 3 and len(p["R_world"]) == 9
    # start_inst_id respektiert (Pred-iids hinter den GT-iids):
    assert preds[0]["instance_id"] == 5


def test_gateway_instances_skip_out_of_scope_class():
    """Klassen ausserhalb des 2-Klassen-Scope werden still gefiltert (§6)."""
    resp = _gateway_resp()
    resp["instances"].append(
        {"id": 2, "class": "voellig_unbekannt", "conf": 0.5,
         "T_cam_obj": np.eye(4).tolist()})
    preds = gwp.gateway_instances_to_viewer_preds(
        resp, R_w2c=[1, 0, 0, 0, 1, 0, 0, 0, 1], t_w2c=[0, 0, 500],
        table_origin=[0, 0, 0.08], snap=False)
    assert len(preds) == 2                          # die 3. (out-of-scope) ist raus


# ── /api/sim/generate_async — Routing statt Stub ──────────────────────────────
def test_sim_generate_no_more_stub_for_non_a_combo(client, monkeypatch):
    """seg=sam3&pose=foundationpose → KEIN 501 „Routing folgt" mehr; die combo_id geht
    an _sim_generate_job, die Sofort-Antwort traegt used_combo/used_seg/used_pose."""
    captured = {}

    def fake_job(job, combo_id="gdrnpp", seg_prompts=None):
        captured["combo_id"] = combo_id

    monkeypatch.setattr(K, "_sim_generate_job", fake_job)
    r = client.get("/api/sim/generate_async",
                   params={"seg": "sam3", "pose": "foundationpose"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job" in body
    assert body["used_combo"] == "sam3__foundationpose"
    assert body["used_seg"] == "sam3" and body["used_pose"] == "FoundationPose"
    assert body["modality"] == "RGBD"
    assert captured["combo_id"] == "sam3__foundationpose"


def test_sim_generate_pipeline_a_default(client, monkeypatch):
    """pipeline=gdrnpp (Default) → Pipeline-A-Kombi an den Job."""
    captured = {}
    monkeypatch.setattr(K, "_sim_generate_job",
                        lambda job, combo_id="gdrnpp", seg_prompts=None:
                        captured.__setitem__("combo_id", combo_id))
    r = client.get("/api/sim/generate_async", params={"pipeline": "gdrnpp"})
    assert r.status_code == 200
    assert r.json()["used_combo"] == "gdrnpp"
    assert captured["combo_id"] == "gdrnpp"


def test_sim_generate_invalid_combo_is_400(client, monkeypatch):
    monkeypatch.setattr(K, "_sim_generate_job",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nie starten")))
    r = client.get("/api/sim/generate_async",
                   params={"seg": "bogus-seg", "pose": "foundationpose"})
    assert r.status_code == 400


def test_sim_generate_gt_seg_source_blocked(client, monkeypatch):
    """image-only (ADR-020): die `gt`-Seg-Quelle (supplied GT masks) ist im interaktiven
    Sim-Pfad gesperrt — GT-Masken sind nur fuers Batch-Eval."""
    monkeypatch.setattr(K, "_sim_generate_job",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nie starten")))
    r = client.get("/api/sim/generate_async",
                   params={"seg": "gt", "pose": "foundationpose"})
    assert r.status_code == 400
    assert "gt" in r.json()["detail"].lower()


# ── /api/real/infer_async — Routing statt Stub (image-only, kein GT) ──────────
def test_real_infer_async_routes_non_a(client, monkeypatch):
    captured = {}

    def fake_job(job, img_bytes, fname, combo_id="gdrnpp", depth_bytes=None,
                 seg_prompts=None):
        captured["combo_id"] = combo_id
        captured["has_depth"] = depth_bytes is not None

    monkeypatch.setattr(K, "_real_infer_job", fake_job)
    files = {"image": ("rgb.png", _png_bytes(), "image/png"),
             "depth": ("depth.png", _png_bytes(mode="L16", val=300), "image/png")}
    r = client.post("/api/real/infer_async", files=files,
                    data={"seg": "yolo-seg", "pose": "GigaPose-3D"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["used_combo"] == "yolo_seg__gigapose_rgbd"
    assert body["modality"] == "RGBD"
    assert captured["combo_id"] == "yolo_seg__gigapose_rgbd"
    assert captured["has_depth"] is True


def test_real_infer_async_pipeline_a_default_unchanged(client, monkeypatch):
    """Default-Aufruf (kein pipeline/seg/pose) bleibt Pipeline A (Live-Pfad)."""
    captured = {}
    monkeypatch.setattr(K, "_real_infer_job",
                        lambda job, img_bytes, fname, combo_id="gdrnpp", *a, **k:
                        captured.__setitem__("combo_id", combo_id))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/real/infer_async", files=files)
    assert r.status_code == 200
    assert r.json()["used_combo"] == "gdrnpp"
    assert captured["combo_id"] == "gdrnpp"


def test_real_infer_async_gt_seg_blocked(client, monkeypatch):
    monkeypatch.setattr(K, "_real_infer_job",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nie starten")))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/real/infer_async", files=files,
                    data={"seg": "gt", "pose": "foundationpose"})
    assert r.status_code == 400


# ── _real_infer_job: NICHT-A faehrt das Gateway, Pipeline A den Worker ────────
def test_real_infer_job_non_a_uses_gateway_not_worker(monkeypatch, tmp_path):
    """_real_infer_job mit einer NICHT-A-Kombi → Pose ueber das Gateway (gemockt), NIE
    ueber den :8078-Worker. Result-Doc traegt used_* + nur color='pred' (kein GT, Upload).
    Detektor wird gemockt (kein train-venv/GPU)."""
    calls = {"worker": 0, "gateway": 0}

    monkeypatch.setattr(K, "_run_detector", lambda src, out: 2)
    monkeypatch.setattr(K, "_worker_infer_upload",
                        lambda *a, **k: calls.__setitem__("worker", calls["worker"] + 1) or [])
    monkeypatch.setattr(K, "_zivid_cam", lambda: {
        "cam_K": [600, 0, 32, 0, 600, 32, 0, 0, 1],
        "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_w2c": [0, 0, 500]})
    monkeypatch.setattr(K, "_table_origin", lambda: [0.0, 0.0, 0.08])

    def fake_gateway(combo_id, **kw):
        calls["gateway"] += 1
        return _gateway_resp()

    monkeypatch.setattr(K, "_gateway_predict_multipart", fake_gateway)
    monkeypatch.setattr(K, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(K, "RENDERS", tmp_path / "renders")
    (tmp_path / "uploads").mkdir(); (tmp_path / "renders").mkdir()

    job = "deadbeef"
    K._real_infer_job(job, _png_bytes(), "rgb.png",
                      combo_id="sam3__foundationpose",
                      depth_bytes=_png_bytes(mode="L16", val=300))

    st = K._job_get(job)
    assert st.get("pct") == 100, st
    assert calls["gateway"] == 1, "NICHT-A muss das Gateway fahren"
    assert calls["worker"] == 0, "NICHT-A darf den Live-Worker NIE anfassen"
    assert st["used_combo"] == "sam3__foundationpose"
    # image-only: Upload-Doc hat KEIN GT — alle results color='pred'.
    doc = K._SIM_DOCS.get(job) or {}
    docp = K.RENDERS / f"real_{job}.json"
    import json as _json
    saved = _json.load(open(docp))
    assert saved["meta"]["used_combo"] == "sam3__foundationpose"
    assert all(r["color"] == "pred" for r in saved["results"])
    assert len(saved["results"]) == 2


def test_real_infer_job_pipeline_a_uses_worker_not_gateway(monkeypatch, tmp_path):
    """Pipeline A (default combo_id='gdrnpp') → :8078-Worker, NIE das Gateway."""
    calls = {"worker": 0, "gateway": 0}
    monkeypatch.setattr(K, "_run_detector", lambda src, out: 1)
    monkeypatch.setattr(K, "_worker_infer_upload",
                        lambda *a, **k: (calls.__setitem__("worker", calls["worker"] + 1) or []))
    monkeypatch.setattr(K, "_gateway_predict_multipart",
                        lambda *a, **k: calls.__setitem__("gateway", calls["gateway"] + 1))
    monkeypatch.setattr(K, "_zivid_cam", lambda: {
        "cam_K": [600, 0, 32, 0, 600, 32, 0, 0, 1],
        "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_w2c": [0, 0, 500]})
    monkeypatch.setattr(K, "_table_origin", lambda: [0.0, 0.0, 0.08])
    monkeypatch.setattr(K, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(K, "RENDERS", tmp_path / "renders")
    (tmp_path / "uploads").mkdir(); (tmp_path / "renders").mkdir()

    K._real_infer_job("cafe", _png_bytes(), "rgb.png")        # default = Pipeline A
    st = K._job_get("cafe")
    assert st.get("pct") == 100, st
    assert calls["worker"] == 1, "Pipeline A muss den Live-Worker fahren"
    assert calls["gateway"] == 0, "Pipeline A darf das Gateway NIE anfassen"
    assert st["used_combo"] == "gdrnpp"


def test_real_infer_job_needs_depth_without_depth_errors(monkeypatch, tmp_path):
    """needs_depth-Kombi im Upload OHNE Tiefenbild → klare Fehlermeldung im Job-State
    (keine stille Falschpose, UX-Spec §5). Das Gateway wird NIE aufgerufen."""
    calls = {"gateway": 0}
    monkeypatch.setattr(K, "_run_detector", lambda src, out: 2)
    monkeypatch.setattr(K, "_gateway_predict_multipart",
                        lambda *a, **k: calls.__setitem__("gateway", calls["gateway"] + 1))
    monkeypatch.setattr(K, "_zivid_cam", lambda: {
        "cam_K": [600, 0, 32, 0, 600, 32, 0, 0, 1],
        "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_w2c": [0, 0, 500]})
    monkeypatch.setattr(K, "_table_origin", lambda: [0.0, 0.0, 0.08])
    monkeypatch.setattr(K, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(K, "RENDERS", tmp_path / "renders")
    (tmp_path / "uploads").mkdir(); (tmp_path / "renders").mkdir()

    K._real_infer_job("nodep", _png_bytes(), "rgb.png",
                      combo_id="sam3__foundationpose", depth_bytes=None)  # needs_depth, kein depth
    st = K._job_get("nodep")
    assert st.get("pct") == -1, st                 # Fehlerpfad
    assert "tiefenbild" in st.get("error", "").lower()
    assert calls["gateway"] == 0


# ── /api/predict — used_*-Konsistenz (gleiches Result-Schema wie Sim/Real) ─────
def test_predict_result_carries_used_fields(client, monkeypatch):
    """Auch /api/predict (S-013) traegt jetzt used_combo/used_seg/used_pose/modality —
    EIN Result-Schema fuer alle Pfade (T-164)."""
    monkeypatch.setattr(K, "_gateway_predict_multipart", lambda cid, **k: _gateway_resp())
    files = {"image": ("rgb.png", _png_bytes(), "image/png"),
             "depth": ("depth.png", _png_bytes(mode="L16", val=300), "image/png")}
    r = client.post("/api/predict", files=files, data={
        "pipeline": "sam3__foundationpose",
        "fx": "600", "fy": "600", "cx": "32", "cy": "32",
        "cam_R_w2c": "1 0 0 0 1 0 0 0 1", "cam_t_w2c": "0 0 500"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["used_combo"] == "sam3__foundationpose"
    assert body["used_seg"] == "sam3" and body["used_pose"] == "FoundationPose"
    assert body["modality"] == "RGBD"


def test_predict_pipeline_a_result_carries_used_fields(client, monkeypatch):
    monkeypatch.setattr(K, "_real_infer_job",
                        lambda j, b, f, *a, **k: K._job_set(j, phase="Fertig", pct=100))
    files = {"image": ("rgb.png", _png_bytes(), "image/png")}
    r = client.post("/api/predict", files=files, data={"pipeline": "gdrnpp"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["used_combo"] == "gdrnpp" and body["modality"] == "RGB"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
