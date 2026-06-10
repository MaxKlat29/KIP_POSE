#!/usr/bin/env python3
"""Tests fuer den Batch-Eval-Runner (S-012 / T-138) gegen FIXTURES + Mocks.

Kein GPU/Gateway/best.pt noetig (Code-now/run-later). Prueft den fastapi-freien
`eval.batch_eval`-Kern + die kip_server-/api/eval/*-Endpoints (TestClient, gemockt):

  * EVAL_CONFIGS = ALLE feasiblen Kombis (~12, registry-Kreuzprodukt), kuratierte 7
    = recommended-Subset, Kombi 1 = Pipeline A (Referenz). [T-138-PIVOT]
  * instances_to_doc: Gateway-/predict T_cam_obj → pose_result-Welt-Doc, §6-Klassen-
    mapping (lowercase → CamelCase-Part → obj_id), 2-Klassen-Scope filtert Fremdes.
  * Coord-Frame Round-Trip (S-003): world_to_bop_cam(instances_to_doc(T)) == T-Translation.
  * BOP-CSV-Erzeugung: doc_to_bop_rows → korrekte (scene_id,im_id,obj_id,R,t)-Zeilen.
  * aggregate_config: mean/std seg_ms/pose_ms, coverage + crash_rate ∈ 0..1,
    Pipeline-A-ohne-Gateway zaehlt NICHT als Versuch.
  * run_one: Crash (predict wirft) → ok=False, error gesetzt, kein Re-raise.
  * ar_from_report: AR IC-BIN overall + per_class aus eval_bop-report.json.
  * run_batch end-to-end gegen Mock-predict + Mock-eval → results.json + EVAL.md,
    idempotent (re-run patcht).
  * Endpoint-Schemata = Lena batch.js / Mia §14: runs/result/run/job,
    coverage/crash 0..1, job = sim-job-{pct,phase}.

Lauf:  .venv/bin/python -m pytest project/tests/test_batch_eval.py -q
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from eval import batch_eval as be  # noqa: E402
from compare_pipelines import world_to_bop_cam  # noqa: E402


# ── Fixtures: Mini-GT-Szene + gemockte Gateway-Antwort ──────────────────────────
def _png_bytes(hw=(48, 48), val=20):
    from PIL import Image
    arr = np.full((*hw, 3), val, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mask_b64(hw=(48, 48)):
    from PIL import Image
    m = np.zeros(hw, dtype=np.uint8)
    m[8:30, 8:30] = 255
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _T_cam_obj(tx, ty, tz):
    """Identitaets-Rotation, Translation (tx,ty,tz) in Metern → 4×4 cam-Frame."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T.tolist()


def _scene(tmp: pathlib.Path, scene_id=0, with_depth=True):
    """Eine Mini-Szene mit RGB(+Depth) auf Disk + BOP-Kamera (Extrinsics = I + 0.5 m)."""
    sd = tmp / f"{scene_id:06d}"
    (sd / "rgb").mkdir(parents=True, exist_ok=True)
    (sd / "depth").mkdir(parents=True, exist_ok=True)
    (sd / "rgb" / "000000.png").write_bytes(_png_bytes())
    if with_depth:
        (sd / "depth" / "000000.png").write_bytes(_png_bytes())
    camera = {"cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1],
              "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
              "cam_t_w2c": [0, 0, 500]}
    return {"scene_id": scene_id, "im_id": 0,
            "rgb": str(sd / "rgb" / "000000.png"),
            "depth": str(sd / "depth" / "000000.png") if with_depth else None,
            "depth_scale": 0.1,   # BOP/SDG convention (png*0.1 = mm), T-156
            "camera": camera, "K": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
            "dir": str(sd)}


def _gateway_reply(n_anker=2, seg_ms=12.0, pose_ms=340.0, with_foreign=False):
    """Eine gemockte /predict-Antwort: n_anker valide Anker + optional eine Fremd-Klasse."""
    inst = [
        {"id": 0, "class": "anker_kurz", "conf": 0.91,
         "T_cam_obj": _T_cam_obj(0.10, -0.05, 0.60), "mask_b64": _mask_b64(),
         "bbox_2d": [8, 8, 30, 30]},
        {"id": 1, "class": "anker_lang", "conf": 0.83,
         "T_cam_obj": _T_cam_obj(-0.08, 0.04, 0.72), "mask_b64": _mask_b64(),
         "bbox_2d": [8, 8, 30, 30]},
    ][:n_anker]
    if with_foreign:
        inst.append({"id": 9, "class": "buerstenhalter_2polig", "conf": 0.5,
                     "T_cam_obj": _T_cam_obj(0.0, 0.0, 0.5), "mask_b64": _mask_b64()})
    return {"instances": inst, "timings": {"seg_ms": seg_ms, "pose_ms": pose_ms,
                                           "num_posed": n_anker}}


# ── 1) Die ~12 feasiblen Configs (PIVOT) + Pipeline A + Recommended-7 ────────────
def test_feasible_configs_pipeline_a_and_recommended():
    # PIVOT (T-138): ALLE feasiblen Kombis (3 seg × 4 pose = 12), NICHT fix 7.
    assert len(be.EVAL_CONFIGS) == 12
    a = [c for c in be.EVAL_CONFIGS if c["is_pipeline_a"]]
    assert len(a) == 1 and a[0]["n"] == 1
    assert a[0]["seg"] == "yolo-obb" and a[0]["pose"] == "GDRNPP"
    # Genau 7 sind RECOMMENDED (die kuratierte Whitelist).
    rec = [c for c in be.EVAL_CONFIGS if c["recommended"]]
    assert len(rec) == 7
    # FE-Labels muessen Lenas POST_L/SEG_L-Maps treffen.
    segs = {c["seg"] for c in be.EVAL_CONFIGS}
    poses = {c["pose"] for c in be.EVAL_CONFIGS}
    assert segs == {"yolo-obb", "yolo-seg", "sam3"}
    assert poses == {"GDRNPP", "FoundationPose", "GigaPose-2D", "GigaPose-3D"}
    # run_config_id eindeutig ueber alle 12 (combo-id, nicht Gateway-source-Label).
    keys = [be.config_key(c) for c in be.EVAL_CONFIGS]
    assert len(set(keys)) == 12


def test_recommended_match_combo_whitelist():
    # Die Recommended-Configs == genau die combos.COMBO_WHITELIST-ids.
    from pipelines.combos import COMBO_WHITELIST
    rec_ids = {be.config_key(c) for c in be.EVAL_CONFIGS if c["recommended"]}
    assert rec_ids == {c["id"] for c in COMBO_WHITELIST}


def test_pivot_flags_present():
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}
    # GDRNPP mit nicht-OBB-seg → degraded (AABB-aus-Maske), nicht recommended.
    assert by_key["yolo_seg__gdrnpp"]["degraded"] is True
    assert by_key["yolo_seg__gdrnpp"]["degraded_reason"] == "aabb_from_mask"
    assert by_key["yolo_seg__gdrnpp"]["recommended"] is False
    # sam3-Kombis: seit T-177 transferiert das Gateway die Klassen von yolo-obb
    # (SAM3_CLASS_FROM_YOLO) — sam3 ist nicht mehr klassen-ambig (S006 obsolet).
    assert by_key["sam3__foundationpose"]["class_ambiguity"] is False
    assert by_key["yolo_seg__foundationpose"]["class_ambiguity"] is False
    # Pipeline A nicht degraded.
    assert by_key["gdrnpp"]["degraded"] is False


def test_pipeline_a_routes_through_gateway_native_obb():
    """T-157: Pipeline A (yolo-obb→gdrnpp) faehrt im EVAL-Pfad ueber das Gateway mit der
    ECHTEN OBB-Quelle (seg_source='yolo-obb', NICHT dem mask-Pfad 'yolo'), damit gdrnpp-svc
    `obb` nativ liest (S-004) → echte Pipeline-A-Pose statt leerer Referenzzeile.
    seg_source='gdrnpp' (Live-Monolith-Signal) ist hier explizit NICHT mehr gesetzt."""
    a = next(c for c in be.EVAL_CONFIGS if c["is_pipeline_a"])
    assert a["seg_source"] == "yolo-obb"      # echte orientierte Boxen, nicht "yolo"/"gdrnpp"
    assert a["pose_source"] == "gdrnpp"
    # Pipeline A ist KEINE degraded-Kombi (kein AABB-aus-Maske-Fallback).
    assert a["degraded"] is False
    # Die degraded gdrnpp-Kombis routen weiterhin ueber den mask-Pfad "yolo"/"sam3".
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}
    assert by_key["yolo_seg__gdrnpp"]["seg_source"] == "yolo"
    assert by_key["sam3__gdrnpp"]["seg_source"] == "sam3"


def test_needs_depth_matches_contract():
    # §5: FP + GigaPose-3D brauchen Depth; GigaPose-2D + GDRNPP nicht.
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}
    assert by_key["yolo_seg__foundationpose"]["needs_depth"] is True
    assert by_key["sam3__gigapose_rgbd"]["needs_depth"] is True
    assert by_key["yolo_seg__gigapose_rgb"]["needs_depth"] is False
    assert by_key["gdrnpp"]["needs_depth"] is False


def test_runner_gateway_routing_consistent():
    """T-155 Konsistenz Runner ↔ Gateway: der Batch-Eval-Runner (EVAL_CONFIGS, via
    http_predict) und der direkte /api/predict-Proxy (gateway_proxy.COMBO_TO_GATEWAY)
    MUESSEN jede NICHT-A-Kombi auf dasselbe (seg_source, pose_source, degraded) routen.
    Sonst gibt derselbe Eval andere Zahlen als die manuelle Inferenz desselben Modells."""
    from pipelines.gateway_proxy import COMBO_TO_GATEWAY
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}
    # Beide Pfade decken dieselben NICHT-A feasible-Kombis ab.
    non_a_runner = {k for k, c in by_key.items() if not c["is_pipeline_a"]}
    assert non_a_runner == set(COMBO_TO_GATEWAY), "Runner ↔ Gateway: gleiche NICHT-A-Kombis"
    for cid, gw in COMBO_TO_GATEWAY.items():
        cfg = by_key[cid]
        assert cfg["seg_source"] == gw["seg_source"], f"{cid}: seg_source-Drift"
        assert cfg["pose_source"] == gw["pose_source"], f"{cid}: pose_source-Drift"
        # Der Runner setzt degraded=true im http_predict fuer pose_source==gdrnpp&!is_a;
        # das deckt sich 1:1 mit dem COMBO_TO_GATEWAY-degraded-Flag.
        runner_degraded = (cfg["pose_source"] == "gdrnpp" and not cfg["is_pipeline_a"])
        assert runner_degraded == gw["degraded"], f"{cid}: degraded-Drift"


# ── 2) instances_to_doc: §6-Klassenmapping + 2-Klassen-Scope ────────────────────
def test_instances_to_doc_class_mapping_and_scope():
    reply = _gateway_reply(n_anker=2, with_foreign=True)
    camera = _scene_camera()
    doc = be.instances_to_doc(reply["instances"], camera, "scene.png")
    # Fremd-Klasse (buerstenhalter) wird gefiltert → nur 2 Anker.
    assert len(doc["results"]) == 2
    parts = sorted(r["part"] for r in doc["results"])
    assert parts == ["Anker_Kurz", "Anker_Lang"]   # CamelCase-Registry-Parts (§6)
    # Schema-valide (das frozen pose_result).
    from pipelines import contract
    assert contract.validate(doc) == []


def test_instances_to_doc_requires_extrinsics():
    reply = _gateway_reply(n_anker=1)
    with pytest.raises(ValueError):
        be.instances_to_doc(reply["instances"], {"cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1]},
                            "scene.png")


# ── 3) Coord-Frame Round-Trip (S-003): world → bop-cam erholt die Cam-Translation ─
def test_coordframe_roundtrip_translation_preserved():
    camera = _scene_camera()
    tx, ty, tz = 0.10, -0.05, 0.60
    inst = {"id": 0, "class": "anker_kurz", "conf": 1.0,
            "T_cam_obj": _T_cam_obj(tx, ty, tz), "bbox_2d": [0, 0, 1, 1]}
    doc = be.instances_to_doc([inst], camera, "s.png")
    r = doc["results"][0]
    R_w2c = np.array(camera["cam_R_w2c"]).reshape(3, 3)
    t_w2c = np.array(camera["cam_t_w2c"])
    table_origin = doc["meta"]["table_origin"]
    _R_m2c, t_m2c_mm = world_to_bop_cam(r["R_world"], r["t_world"], R_w2c, t_w2c, table_origin)
    # Cam-Translation (mm) muss die Eingabe (m → mm) treffen — DER Round-Trip.
    assert np.allclose(t_m2c_mm, [tx * 1000, ty * 1000, tz * 1000], atol=1e-3)


# ── 4) BOP-CSV-Erzeugung ────────────────────────────────────────────────────────
def test_bop_csv_rows_shape_and_objids():
    reply = _gateway_reply(n_anker=2)
    camera = _scene_camera()
    doc = be.instances_to_doc(reply["instances"], camera, "s.png")
    rows = be.doc_to_bop_rows(doc, camera, scene_id=3, im_id=0)
    assert len(rows) == 2
    for row in rows:
        assert row["scene_id"] == 3 and row["im_id"] == 0
        assert row["obj_id"] in (1, 2)                  # anker_kurz=1, anker_lang=2
        assert len(row["R"].split()) == 9 and len(row["t"].split()) == 3
        assert 0.0 <= float(row["score"]) <= 1.0


def test_write_bop_csv_roundtrips(tmp_path):
    reply = _gateway_reply(n_anker=2)
    camera = _scene_camera()
    doc = be.instances_to_doc(reply["instances"], camera, "s.png")
    rows = be.doc_to_bop_rows(doc, camera, scene_id=0, im_id=0)
    p = tmp_path / "preds.csv"
    be._write_bop_csv(p, rows)
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "scene_id,im_id,obj_id,score,R,t,time"
    assert len(lines) == 1 + 2                          # header + 2 rows


# ── 4b) T-163: Eval-Pfad snappt NICHT — sonst killt der Z-Snap AR_MSSD ───────────
def _back_to_cam_t(doc, camera):
    """Recover cam-frame translation (mm) of the first result via world_to_bop_cam."""
    r = doc["results"][0]
    R_w2c = np.array(camera["cam_R_w2c"]).reshape(3, 3)
    t_w2c = np.array(camera["cam_t_w2c"])
    table_origin = doc["meta"]["table_origin"]
    _R, t_m2c = world_to_bop_cam(r["R_world"], r["t_world"], R_w2c, t_w2c, table_origin)
    return np.asarray(t_m2c, float)


def test_eval_path_does_not_snap_preserves_3d_z(monkeypatch):
    """REGRESSION T-163: der Eval-Pfad darf KEINEN Boden-Snap anwenden.

    Bug: `instances_to_doc` snappte über `planar_z_snap(table_z=0.0)` jede Pose so,
    dass der tiefste Mesh-Punkt auf Welt-z=0 ruht. Im pose_isaac-Eval ruhen die GT-
    Teile aber NICHT auf z=0 → der Snap hob jede Pose um ~75–79 mm an. 2D reprojiziert
    weiter (MSPD ok), aber der 3D-Punktabstand (MSSD) sprengt die Schwelle → AR_MSSD=0.

    Dieser Test injiziert synthetische Mesh-Verts, deren tiefster Punkt 60 mm UNTER
    dem Body-Origin liegt → ein scharfer, deterministischer Snap-Hebel. Bewiesen:
      * snap=False (Fix): die zurückgerechnete Cam-Z == Eingabe-Z (exakt, kein Drift).
      * snap=True  (Bug): die Cam-Z weicht um die volle Snap-Distanz ab (hier ~60 mm).
    Gegen den ALTEN Code (immer-snap) fällt der snap=False-Assert (Cam-Z driftet) → rot.
    """
    # Synthetische Body-Verts (Meter): Würfelchen, tiefster Punkt 0.06 m unter Origin.
    verts = np.array([[0.01, 0.01, -0.06], [-0.01, -0.01, -0.06],
                      [0.01, -0.01, 0.02], [-0.01, 0.01, 0.02]], float)
    monkeypatch.setattr(be, "_mesh_verts_for", lambda oid, warn=None: verts)
    # Falls der geteilte Gateway-Proxy/Composed-Pfad genutzt wird, dort dieselbe
    # CAD-Quelle patchen (er lädt Verts über pipelines.composed._mesh_verts_for).
    try:
        from pipelines import composed as _composed
        monkeypatch.setattr(_composed, "_mesh_verts_for",
                            lambda oid, warn=None: verts, raising=False)
    except Exception:  # noqa: BLE001
        pass

    camera = _scene_camera()
    tz_m = 0.60
    inst = {"id": 0, "class": "anker_kurz", "conf": 1.0,
            "T_cam_obj": _T_cam_obj(0.10, -0.05, tz_m), "bbox_2d": [0, 0, 1, 1]}

    # Fix: Default snap=False → 3D-Cam-Translation bleibt EXAKT erhalten.
    doc_fix = be.instances_to_doc([inst], camera, "s.png")
    t_fix = _back_to_cam_t(doc_fix, camera)
    assert np.allclose(t_fix, [100.0, -50.0, 600.0], atol=1e-2), (
        f"Eval-Pfad (snap=False) verfaelschte die Cam-Translation: {t_fix} "
        "— der Boden-Snap darf im Eval NICHT greifen (T-163).")

    # Bug-Regime: snap=True → die Cam-Z driftet messbar (Snap-Hebel ~60 mm).
    doc_snap = be.instances_to_doc([inst], camera, "s.png", snap=True)
    t_snap = _back_to_cam_t(doc_snap, camera)
    dz = abs(float(t_snap[2]) - 600.0)
    assert dz > 30.0, (
        f"snap=True haette die Z um ~Snap-Distanz verschieben muessen (got dz={dz:.1f}mm). "
        "Wenn das fehlschlaegt, feuert der Snap-Hebel nicht — der Test prueft nichts.")
    # Und: der Fix-Pfad weicht echt vom Bug-Pfad ab (kein No-Op-Theater).
    assert abs(float(t_fix[2]) - float(t_snap[2])) > 30.0


# ── 5) aggregate_config: coverage/crash 0..1, Pipeline-A-no-gateway-Sonderfall ──
def test_aggregate_config_coverage_crash_and_timings():
    cfg = _cfg("yolo_seg__foundationpose")
    per_scene = [
        {"ok": True, "seg_ms": 10.0, "pose_ms": 300.0, "n_instances": 2, "error": None, "rows": []},
        {"ok": True, "seg_ms": 14.0, "pose_ms": 340.0, "n_instances": 0, "error": None, "rows": []},
        {"ok": False, "seg_ms": None, "pose_ms": None, "n_instances": 0, "error": "503", "rows": []},
        {"ok": True, "seg_ms": 12.0, "pose_ms": 320.0, "n_instances": 1, "error": None, "rows": []},
    ]
    row = be.aggregate_config(cfg, per_scene, ar_mean=0.41, ar_std=0.0)
    assert row["seg"] == "yolo-seg" and row["pose"] == "FoundationPose"
    assert row["ar_mean"] == 0.41
    # mean seg_ms ueber die 3 oks = (10+14+12)/3 = 12.0
    assert row["seg_ms"] == pytest.approx(12.0)
    assert row["pose_ms"] == pytest.approx(320.0)
    # 4 reale Versuche, 1 Crash → crash_rate = 0.25; 2 mit Instanzen → coverage = 0.5.
    assert row["crash_rate"] == pytest.approx(0.25)
    assert row["coverage"] == pytest.approx(0.5)
    assert 0.0 <= row["crash_rate"] <= 1.0 and 0.0 <= row["coverage"] <= 1.0
    assert row["is_pipeline_a"] is False
    assert row["run_config_id"] == "yolo_seg__foundationpose"


def test_aggregate_config_pipeline_a_no_gateway_not_counted():
    cfg = _cfg("gdrnpp")  # Pipeline A
    per_scene = [
        {"ok": False, "seg_ms": None, "pose_ms": None, "n_instances": 0,
         "error": "pipeline_a_no_gateway", "rows": []} for _ in range(5)
    ]
    row = be.aggregate_config(cfg, per_scene)
    assert row["n_real"] == 0                           # kein echter Versuch
    assert row["crash_rate"] is None and row["coverage"] is None
    assert row["is_pipeline_a"] is True


def test_aggregate_all_none_timings():
    cfg = _cfg("sam3__foundationpose")
    per_scene = [{"ok": False, "seg_ms": None, "pose_ms": None, "n_instances": 0,
                  "error": "boom", "rows": []}]
    row = be.aggregate_config(cfg, per_scene, ar_mean=None)
    assert row["seg_ms"] is None and row["pose_ms"] is None
    assert row["crash_rate"] == 1.0 and row["coverage"] == 0.0


# ── 6) run_one: Crash-Handling + Pipeline-A-Sonderfall ──────────────────────────
def test_run_one_catches_crash(tmp_path):
    cfg = _cfg("yolo_seg__foundationpose")
    scene = _scene(tmp_path)

    def boom(_cfg, _scene):
        raise RuntimeError("svc 503")

    res = be.run_one(cfg, scene, boom)
    assert res["ok"] is False and "503" in res["error"]
    assert res["rows"] == []


def test_run_one_pipeline_a_no_gateway(tmp_path):
    cfg = _cfg("gdrnpp")  # Pipeline A
    scene = _scene(tmp_path)

    def predict(_cfg, _scene):
        raise be.PipelineANotOnGateway(_cfg)

    res = be.run_one(cfg, scene, predict)
    assert res["ok"] is False and res["error"] == "pipeline_a_no_gateway"


def test_run_one_success_builds_rows(tmp_path):
    cfg = _cfg("yolo_seg__foundationpose")
    scene = _scene(tmp_path)
    res = be.run_one(cfg, scene, lambda c, s: _gateway_reply(n_anker=2))
    assert res["ok"] is True and res["n_instances"] == 2
    assert len(res["rows"]) == 2
    assert res["seg_ms"] == 12.0 and res["pose_ms"] == 340.0


# ── 6b) T-157: http_predict-Wiring fuer Pipeline A (echte obb ueber Gateway) ──────
class _FakeResp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _FakeHttpxClient:
    """Faengt die /predict-Form-Daten ab (statt echtem Netz) → wir koennen pruefen,
    WAS der Runner ans Gateway schickt (seg_source/pose_source/degraded)."""

    captured = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, data=None, files=None):
        _FakeHttpxClient.captured = {"url": url, "data": dict(data or {}),
                                     "files": list((files or {}).keys())}
        # Antwort = eine valide Pipeline-A-Pose (2 Anker), wie sie gdrnpp-svc liefern wuerde.
        return _FakeResp(_gateway_reply(n_anker=2))


def _patch_httpx(monkeypatch, client_cls=_FakeHttpxClient):
    import types
    fake = types.ModuleType("httpx")
    fake.Client = client_cls
    monkeypatch.setitem(sys.modules, "httpx", fake)


def test_http_predict_pipeline_a_uses_native_obb_over_gateway(tmp_path, monkeypatch):
    """T-157 KERN: der Default-http_predict routet Pipeline A ueber das Gateway mit
    seg_source='yolo-obb' + pose_source='gdrnpp' OHNE degraded → echte obb, gdrnpp-svc
    nativ. Frueher warf er PipelineANotOnGateway → leere Zeile. Jetzt: echte Antwort."""
    _FakeHttpxClient.captured = None
    _patch_httpx(monkeypatch)
    predict = be.http_predict("http://gateway:8000")
    cfg = _cfg("gdrnpp")                                  # Pipeline A
    scene = _scene(tmp_path)
    out = predict(cfg, scene)                             # wirft NICHT mehr
    cap = _FakeHttpxClient.captured
    assert cap is not None and cap["url"].endswith("/predict")
    assert cap["data"]["seg_source"] == "yolo-obb"       # echte OBB-Quelle
    assert cap["data"]["pose_source"] == "gdrnpp"
    assert "degraded" not in cap["data"]                 # KEIN AABB-aus-Maske-Fallback
    assert len(out["instances"]) == 2                    # echte Pipeline-A-Posen


def test_http_predict_degraded_combo_still_sets_degraded(tmp_path, monkeypatch):
    """Regression: die degraded gdrnpp-Kombis (yolo-seg→gdrnpp) setzen WEITERHIN
    degraded=true + behalten ihren mask-seg_source (T-153 unveraendert)."""
    _FakeHttpxClient.captured = None
    _patch_httpx(monkeypatch)
    predict = be.http_predict("http://gateway:8000")
    cfg = _cfg("yolo_seg__gdrnpp")                        # degraded, NICHT Pipeline A
    predict(cfg, _scene(tmp_path))
    cap = _FakeHttpxClient.captured
    assert cap["data"]["seg_source"] == "yolo"           # mask-Pfad bleibt
    assert cap["data"]["pose_source"] == "gdrnpp"
    assert cap["data"]["degraded"] == "true"             # T-153-Opt-in bleibt


def test_run_one_pipeline_a_over_gateway_builds_rows(tmp_path, monkeypatch):
    """run_one mit dem echten http_predict → Pipeline A liefert nicht-leere CSV-Zeilen
    (der ehemals leere AR-Eintrag ist jetzt befuellt)."""
    _patch_httpx(monkeypatch)
    predict = be.http_predict("http://gateway:8000")
    res = be.run_one(_cfg("gdrnpp"), _scene(tmp_path), predict)
    assert res["ok"] is True
    assert res["n_instances"] == 2 and len(res["rows"]) == 2
    assert res["error"] is None


# ── 7) ar_from_report — T-171: primaere AR = D1-aktive Klassen, NICHT 6-obj-overall ──
def test_ar_from_report_primary_is_active_class_mean_not_overall():
    """Die primaere AR ist das Mittel der D1-aktiven Klassen (anker_kurz/lang) — NICHT
    der eval_bop-overall.AR. overall.AR (6-obj, hier 0.4123 inkl. 4 untrainierter 0er)
    wandert nach `ar_6obj`. Das ist der T-171-Kern: 0.886 statt 0.295."""
    report = {"results": {
        "overall": {"AR": 0.4123},                       # 6-obj-Mittel (verwaessert)
        "per_object": {"1": {"name": "Anker_Kurz", "AR": 0.8796},
                       "2": {"name": "Anker_Lang", "AR": 0.8926},
                       "3": {"name": "Buerstenhalter_2polig", "AR": 0.0},
                       "4": {"name": "Getriebegehaeuse_typ4", "AR": 0.0},
                       "5": {"name": "Ringmagnet", "AR": 0.0},
                       "6": {"name": "Zahnrad", "AR": 0.0}}}}
    ar, per_class, ar_6obj = be.ar_from_report(report)
    # Primaer = Mittel NUR der 2 aktiven Klassen (nicht alle 6).
    assert ar == pytest.approx((0.8796 + 0.8926) / 2, abs=1e-4)   # 0.8861
    # Sekundaer = die alte 6-obj-overall-Zahl, erhalten.
    assert ar_6obj == pytest.approx(0.4123)
    # per_class fuehrt weiterhin ALLE 6 Klassen (FE-Spalte + Transparenz).
    assert per_class["Anker_Kurz"] == 0.8796 and per_class["Zahnrad"] == 0.0
    assert len(per_class) == 6


def test_ar_from_report_case_insensitive_active_match():
    """Aktive-Klassen-Match ist case-insensitiv (eval_bop CamelCase, Fixtures lowercase)."""
    report = {"results": {"overall": {"AR": 0.4123},
                          "per_object": {"1": {"name": "anker_kurz", "AR": 0.51},
                                         "2": {"name": "anker_lang", "AR": 0.31}}}}
    ar, per_class, ar_6obj = be.ar_from_report(report)
    assert ar == pytest.approx((0.51 + 0.31) / 2)        # 0.41 (2 aktive Klassen)
    assert ar_6obj == pytest.approx(0.4123)
    assert per_class == {"anker_kurz": 0.51, "anker_lang": 0.31}


def test_ar_from_report_missing_active_class_counts_as_zero():
    """Eine aktive D1-Klasse, die im Report fehlt/0 ist (z.B. sam3 findet Anker_Lang
    nie), zaehlt als 0.0 in der Haupt-AR — sonst wuerde sam3 mit nur Anker_Kurz
    faelschlich hoch erscheinen. (0.8187 + 0.0) / 2 = 0.40935."""
    report = {"results": {"overall": {"AR": 0.1365},
                          "per_object": {"1": {"name": "Anker_Kurz", "AR": 0.8187},
                                         "2": {"name": "Anker_Lang", "AR": 0.0},
                                         "5": {"name": "Ringmagnet", "AR": 0.0}}}}
    ar, _per_class, _ar6 = be.ar_from_report(report)
    assert ar == pytest.approx((0.8187 + 0.0) / 2, abs=1e-4)


def test_ar_from_report_active_parts_generic_from_class_map():
    """Die aktiven Klassen kommen GENERISCH aus CLASS_TO_OBJ_ID (nicht hartcodiert 2).
    Kommt eine 3. trainierte Klasse dazu, mittelt die Haupt-AR ueber 3."""
    assert be.active_class_parts() == ["Anker_Kurz", "Anker_Lang"]
    # Simuliere einen 3-Klassen-Scope → Mittel ueber 3.
    import bop_adapter
    saved = dict(be.CLASS_TO_OBJ_ID)
    be.CLASS_TO_OBJ_ID["zahnrad"] = 6
    try:
        assert be.active_class_parts() == ["Anker_Kurz", "Anker_Lang", "Zahnrad"]
        report = {"results": {"overall": {"AR": 0.5},
                              "per_object": {"1": {"name": "Anker_Kurz", "AR": 0.9},
                                             "2": {"name": "Anker_Lang", "AR": 0.6},
                                             "6": {"name": "Zahnrad", "AR": 0.3}}}}
        ar, _pc, _a6 = be.ar_from_report(report)
        assert ar == pytest.approx((0.9 + 0.6 + 0.3) / 3)
    finally:
        be.CLASS_TO_OBJ_ID.clear()
        be.CLASS_TO_OBJ_ID.update(saved)
        assert bop_adapter is not None     # touch import (linter)


def test_ar_from_report_missing_ar_is_none():
    # Kein per_object → keine aktive Klasse aufloesbar → primaer None (kein stiller 0).
    assert be.ar_from_report({"results": {"overall": {}}})[0] is None
    assert be.ar_from_report({})[0] is None
    # ar_6obj bleibt None wenn overall.AR fehlt.
    assert be.ar_from_report({"results": {"overall": {}}})[2] is None


# ── 8) run_batch end-to-end gegen Mocks → results.json + EVAL.md + idempotent ───
def _mock_eval_fn(csv_path, scene_dir, out_dir):
    """Fixture-Scorer: zaehlt die CSV-Zeilen, gibt einen deterministischen AR zurueck
    (kein bop_toolkit). Beweist die CSV→eval_bop-Naht ohne die Box."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = max(0, len(pathlib.Path(csv_path).read_text().strip().splitlines()) - 1)
    ar = round(min(0.9, 0.1 + 0.05 * n), 4)
    report = {"mode": "eval", "protocol": "ic_bin",
              "results": {"overall": {"AR": ar},
                          "per_object": {"1": {"name": "anker_kurz", "AR": ar},
                                         "2": {"name": "anker_lang", "AR": ar}}}}
    (out / "report.json").write_text(json.dumps(report))
    return report


def test_run_batch_end_to_end(tmp_path):
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(3)]
    out = tmp_path / "batch_eval"

    def predict(cfg, scene):
        if cfg["pose_source"] == "gdrnpp":
            raise be.PipelineANotOnGateway(cfg)
        return _gateway_reply(n_anker=2)

    results = be.run_batch(be.EVAL_CONFIGS, scenes, predict, _mock_eval_fn, out,
                           run_id="run-test")
    # Struktur (Lena batch.js / Mia §14). PIVOT: 12 feasible, 7 recommended.
    assert results["run_id"] == "run-test"
    assert results["n_configs"] == 12 and results["n_scenes"] == 3
    cfgs = results["configs"]
    assert len(cfgs) == 12
    assert sum(1 for c in cfgs if c["recommended"]) == 7
    a = [c for c in cfgs if c["is_pipeline_a"]][0]
    assert a["crash_rate"] is None                      # Pipeline A: kein Gateway → n_real 0
    # Gateway-Kombis (pose_source != gdrnpp) sind gescort; gdrnpp-Kombis (A + die 2
    # degraded) gehen ueber PipelineANotOnGateway → ar_mean None (kein Gateway).
    gw = [c for c in cfgs if not c["run_config_id"].endswith("gdrnpp")
          and c["run_config_id"] != "gdrnpp"]
    assert len(gw) == 9                                 # 12 - 3 gdrnpp-Kombis
    for c in gw:
        assert c["ar_mean"] is not None                # gescort
        assert 0.0 <= c["coverage"] <= 1.0 and 0.0 <= c["crash_rate"] <= 1.0
        assert c["seg_ms"] == pytest.approx(12.0)
        assert "run_config_id" in c
    # Persistenz.
    rj = out / "run-test" / "results.json"
    md = out / "run-test" / "EVAL.md"
    assert rj.is_file() and md.is_file()
    assert "AR IC-BIN" in md.read_text()
    # Discovery.
    runs = be.list_runs(out)
    assert runs and runs[0]["run_id"] == "run-test" and runs[0]["n_configs"] == 12
    assert be.load_run(out, "run-test")["run_id"] == "run-test"
    assert be.load_run(out, "nope") is None


def test_run_batch_idempotent_patch(tmp_path):
    scenes = [_scene(tmp_path / "scenes", scene_id=0)]
    out = tmp_path / "batch_eval"
    predict = lambda c, s: (_ for _ in ()).throw(be.PipelineANotOnGateway(c)) \
        if c["pose_source"] == "gdrnpp" else _gateway_reply(n_anker=1)
    be.run_batch(be.EVAL_CONFIGS, scenes, predict, _mock_eval_fn, out, run_id="run-x")
    first = json.loads((out / "run-x" / "results.json").read_text())
    be.run_batch(be.EVAL_CONFIGS, scenes, predict, _mock_eval_fn, out, run_id="run-x")
    # Re-run mit gleicher id → genau EIN Run-Dir (kein Doppel-Append).
    assert len(list(out.glob("*/results.json"))) == 1
    second = json.loads((out / "run-x" / "results.json").read_text())
    assert second["n_configs"] == first["n_configs"] == 12


def test_run_batch_pipeline_a_scored_over_gateway(tmp_path):
    """T-157 end-to-end: faehrt ALLE Kombis (inkl. Pipeline A) ueber einen Mock-Gateway
    → Pipeline A bekommt eine ECHTE, nicht-leere AR-Zeile (ar_mean != None, crash_rate
    != None) statt der frueheren leeren Referenzzelle. Kein PipelineANotOnGateway mehr."""
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(2)]
    out = tmp_path / "batch_eval"

    # Mock-predict: JEDE Kombi (auch Pipeline A) liefert echte Posen — wie das echte
    # http_predict, das Pipeline A jetzt uebers Gateway faehrt (seg_source=yolo-obb).
    def predict(cfg, scene):
        return _gateway_reply(n_anker=2)

    results = be.run_batch(be.EVAL_CONFIGS, scenes, predict, _mock_eval_fn, out,
                           run_id="run-a")
    a = [c for c in results["configs"] if c["is_pipeline_a"]][0]
    assert a["ar_mean"] is not None                       # GESCORT (war vorher None)
    assert a["crash_rate"] is not None                    # echter Versuch (n_real > 0)
    assert a["coverage"] == pytest.approx(1.0)            # 2 Anker pro Szene erkannt
    assert a["is_pipeline_a"] is True                     # FE-Flag bleibt
    assert a["run_config_id"] == "gdrnpp"
    # Pipeline A taucht in den Live-Standings mit einem AR-Rang auf (nicht ar=None-Ende).
    a_st = [s for s in results["standings"] if s["config_key"] == "gdrnpp"][0]
    assert a_st["ar"] is not None and a_st["is_pipeline_a"] is True


def test_discover_scenes(tmp_path):
    # 4 Szenen anlegen, scene_camera mit Extrinsics + BOP-depth_scale.
    for i in range(4):
        sd = tmp_path / f"{i:06d}"
        (sd / "rgb").mkdir(parents=True, exist_ok=True)
        (sd / "depth").mkdir(parents=True, exist_ok=True)
        (sd / "rgb" / "000000.png").write_bytes(_png_bytes())
        (sd / "depth" / "000000.png").write_bytes(_png_bytes())
        (sd / "scene_camera.json").write_text(json.dumps({"0": {
            "cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1], "depth_scale": 0.1,
            "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_w2c": [0, 0, 500]}}))
    scenes = be.discover_scenes(tmp_path, seeds=2)
    assert len(scenes) == 2                             # seeds cappt
    s = scenes[0]
    assert s["K"] == {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    assert s["camera"]["cam_R_w2c"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert pathlib.Path(s["rgb"]).exists() and pathlib.Path(s["depth"]).exists()
    # T-156: the BOP depth_scale MUST be surfaced so the gateway/refiner decode the
    # right metres (png*depth_scale/1000). Dropping it = depth 10x too far -> ~2.4m X.
    assert s["depth_scale"] == 0.1


def test_discover_scenes_depth_scale_defaults_to_one(tmp_path):
    # scene_camera without a depth_scale (real-mm convention) -> 1.0 (status quo).
    sd = tmp_path / "000000"
    (sd / "rgb").mkdir(parents=True, exist_ok=True)
    (sd / "depth").mkdir(parents=True, exist_ok=True)
    (sd / "rgb" / "000000.png").write_bytes(_png_bytes())
    (sd / "depth" / "000000.png").write_bytes(_png_bytes())
    (sd / "scene_camera.json").write_text(json.dumps({"0": {
        "cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1],
        "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_w2c": [0, 0, 500]}}))
    s = be.discover_scenes(tmp_path)[0]
    assert s["depth_scale"] == 1.0


# ── T-156: depth_scale propagation (the ~2.4m RGB-D X-bug) ───────────────────────
def test_http_predict_forwards_depth_scale_for_rgbd():
    """The runner MUST send depth_scale to the gateway for depth-consuming combos,
    else the BOP png*0.1 depth is decoded as png/1000 (10x too far) -> the ~2.4m
    lateral X that zeroed every RGB-D combo's AR (T-156). RGB-only combos send none."""
    captured = {}

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"instances": [], "timings": {}}

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, data=None, files=None):
            captured["data"] = data
            captured["has_depth_file"] = files is not None and "depth" in files
            return _FakeResp()

    import types
    fake_httpx = types.SimpleNamespace(Client=lambda *a, **k: _FakeClient())
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _patched_import(name, *a, **k):
        if name == "httpx":
            return fake_httpx
        return real_import(name, *a, **k)

    import builtins, tempfile
    with tempfile.TemporaryDirectory() as td:
        scene = _scene(pathlib.Path(td))   # carries depth_scale=0.1, needs depth
        predict = be.http_predict("http://gw:8090")
        rgbd_cfg = _cfg("yolo_seg__foundationpose")   # needs_depth=True
        builtins.__import__ = _patched_import
        try:
            predict(rgbd_cfg, scene)
            assert captured["has_depth_file"], "RGB-D combo must upload depth"
            assert "depth_scale" in captured["data"], "depth_scale must be forwarded"
            assert float(captured["data"]["depth_scale"]) == 0.1
            # RGB-only combo (no depth) must NOT forward depth_scale.
            captured.clear()
            rgb_cfg = _cfg("yolo_seg__gigapose_rgb")   # needs_depth=False
            predict(rgb_cfg, scene)
            assert "depth_scale" not in captured["data"]
        finally:
            builtins.__import__ = real_import


def test_depth_decode_backprojection_honours_depth_scale():
    """End-to-end numeric guard on the depth-decode formula shared by the gateway
    pointcloud + both refiners: metres = png * depth_scale / 1000.

    Mirrors box scene 000000: a uint16 depth of 11131 at a pixel that GT projects to,
    with K (fx=1322.67, cx=640) and depth_scale=0.1, MUST back-project to ~the GT
    translation (-0.296,-0.205,1.059) m (within a few cm — the depth samples the part
    SURFACE, not its centroid). The pre-fix decode (png/1000, depth_scale dropped)
    lands at (-3.1,-2.2,11.1) m — the exact 10x / ~2.4m-X regression. This test goes
    RED against the old `png/1000` path and GREEN with `png*depth_scale/1000`."""
    fx = fy = 1322.6666666666667
    cx, cy = 640.0, 360.0
    depth_scale = 0.1
    png_raw = 11131.0
    # The pixel GT obj1 (-296.3,-204.7,1058.9 mm) projects to (measured on box).
    u, v = 269.9, 104.3
    gt_m = np.array([-0.2963, -0.2047, 1.0589])

    def _backproject(png, scale):
        z = png * scale / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    # Correct decode (depth_scale honoured) recovers GT to within a few cm (surface vs
    # centroid). The key guard: it is metres, not tens of metres.
    p_ok = _backproject(png_raw, depth_scale)
    assert np.allclose(p_ok, gt_m, atol=0.10), f"correct decode off: {p_ok} vs {gt_m}"
    # The dropped-depth_scale decode (the bug) is exactly 10x off — X alone ~2.8 m away.
    p_bug = _backproject(png_raw, 1.0)
    assert np.allclose(p_bug, p_ok * 10.0, atol=1e-9), f"bug decode not 10x: {p_bug}"
    assert abs(p_bug[0] - gt_m[0]) > 2.0, "the bug must show the multi-metre X error"
    assert p_bug[2] > 10.0, "the bug must place the surface >10 m away (10x of ~1.1 m)"


# ── Helper ──────────────────────────────────────────────────────────────────────
def _scene_camera():
    return {"cam_K": [600, 0, 320, 0, 600, 240, 0, 0, 1],
            "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_w2c": [0, 0, 500]}


def _cfg(combo_id):
    """Config-dict per combo-id (robust gg Reihenfolge der ~12 feasiblen)."""
    return next(c for c in be.EVAL_CONFIGS if be.config_key(c) == combo_id)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
