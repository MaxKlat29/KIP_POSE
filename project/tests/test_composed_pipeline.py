#!/usr/bin/env python3
"""Tests fuer den 2-stufigen (Seg→Pose) Multi-Pipeline-Seam (S-003 / T-129).

Prueft den NEUEN Seam additiv (der Scaffold-Test test_pipelines_scaffold.py bleibt
gruen — Live-Pfad gdrnpp ist unberuehrt):

  * Jede der 6 NICHT-A-ComposedPipelines ist gegen einen MOCK-Service instanziierbar
    + lauffaehig (KEIN echter HTTP-Call, KEINE GPU) → liefert ein schema-valides
    pose_result-Doc (contract.validate == []).
  * Coord-Frame Round-Trip: world_to_bop_cam(bop_pose_to_world(T)) == T (±eps) —
    DIE Vorzeichen-/Transpose-Falle (CONTRACT.md §1).
  * Klassen-Mapping (§6): obj_id → CamelCase-Part korrekt, Symmetrie greift
    (PART_SYMMETRY ueber den CamelCase-Part, NICHT die lowercase Mesh-Klasse).
  * needs_depth-Gating: nur Estimatoren mit needs_depth=True bekommen depth_b64.
  * Skip-Verhalten (§3): die Pose-Liste darf kuerzer sein als die Detections.
  * Whitelist = GENAU 7 (nicht 12); GDRNPP ist Kombi 1 = der Monolith, KEINE Composed.

Lauf:  python3 -m pytest project/tests/test_composed_pipeline.py -q
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

import bop_adapter as A  # noqa: E402
import compare_pipelines as C  # noqa: E402
from pipelines import contract, registry, combos  # noqa: E402
from pipelines.seg_base import SegAdapter, Detection  # noqa: E402
from pipelines.pose_base import PoseAdapter, Pose, _as_4x4  # noqa: E402
from pipelines.composed import ComposedPipeline, CLASS_TO_OBJ_ID, _mask_b64_to_bbox  # noqa: E402


# ── Test-Fixtures: ein echtes RGB + eine echte Voll-Bild-Maske als base64-PNG ──
def _png_b64(arr) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_b64(x0, y0, x1, y1, hw=(64, 64)) -> str:
    """Voll-Bild-Maske (0/255) mit einem gefuellten Rechteck — wie der Seg-Contract."""
    m = np.zeros(hw, dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return _png_b64(m)


@pytest.fixture()
def rgb_file(tmp_path):
    from PIL import Image
    p = tmp_path / "rgb_000001.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(p)
    return str(p)


@pytest.fixture()
def camera():
    """Ein BOP scene_camera-Eintrag: cam_K flat 9, cam_R_w2c flat 9, cam_t_w2c mm."""
    K = [600.0, 0, 32.0, 0, 600.0, 32.0, 0, 0, 1]
    R_w2c = list(np.eye(3).reshape(9))                 # Welt == Cam-Orientierung
    t_w2c = [0.0, 0.0, 500.0]                          # 0.5 m vor der Kamera (mm)
    return {"cam_K": K, "cam_R_w2c": R_w2c, "cam_t_w2c": t_w2c}


# ── Mock-Service: ersetzt post_json (KEIN Socket, KEINE GPU) ──────────────────
class _MockSeg(SegAdapter):
    """Liefert deterministisch 2 Detections (anker_kurz + anker_lang) + eine
    unbekannte Klasse (muss gefiltert werden, §6)."""
    def post_json(self, path, payload):
        assert path == "/segment"
        assert "rgb_b64" in payload
        self.last_payload = payload
        return {"detections": [
            {"id": 0, "class": "anker_kurz", "conf": 0.91, "mask_b64": _mask_b64(5, 5, 25, 25)},
            {"id": 1, "class": "anker_lang", "conf": 0.80, "mask_b64": _mask_b64(30, 30, 60, 60)},
            {"id": 2, "class": "schraube", "conf": 0.40, "mask_b64": _mask_b64(0, 0, 4, 4)},
        ]}


class _MockPose(PoseAdapter):
    """Echoot fuer jede Instanz eine Identitaets-T_cam_obj 0.3 m vor der Kamera.
    Skippt absichtlich id=2 (existiert ohnehin nicht mehr — Klassenfilter) und
    demonstriert das §3-Skip: liefert NUR Posen fuer die Instanzen die ankommen."""
    def post_json(self, path, payload):
        assert path == "/pose"
        self.last_payload = payload
        poses = []
        for inst in payload["instances"]:
            T = np.eye(4)
            T[2, 3] = 0.30                              # 30 cm vor der Kamera (Meter, §1)
            poses.append({"id": inst["id"], "class": inst["class"],
                          "T_cam_obj": T.tolist(), "score": 0.77})
        return {"poses": poses}


def _composed_with_mocks(pose_needs_depth=True, pipeline=None):
    seg = _MockSeg("http://mock/seg"); seg.id = "mock-seg"
    pose = _MockPose("http://mock/pose"); pose.id = "mock-pose"
    pose.needs_depth, pose.pipeline = pose_needs_depth, pipeline
    return ComposedPipeline(combo_id="mock_combo", name="Mock", seg=seg, pose=pose)


# ── 1) Coord-Frame Round-Trip (DIE Falle, §1) ─────────────────────────────────
def _rand_SO3(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_coord_frame_roundtrip_world_to_bop_cam():
    """world_to_bop_cam ist die exakte Inverse von bop_pose_to_world — fuer ≥1 Kombi
    (hier ueber Zufalls-SO3, deckt Vorzeichen/Transpose ab). Das ist der Frame, den
    die ComposedPipeline durchschiebt (T_cam_obj → Welt → ... → eval BOP-cam)."""
    rng = np.random.default_rng(129)
    for _ in range(50):
        R_world = _rand_SO3(rng)
        t_world = rng.standard_normal(3) * 0.3
        R_w2c = _rand_SO3(rng)
        t_w2c = rng.standard_normal(3) * 500.0
        table_origin = np.array([0.0, 0.0, 0.08])
        R_m2c, t_m2c = C.world_to_bop_cam(R_world, t_world, R_w2c, t_w2c, table_origin)
        R_back, t_back = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
        assert np.allclose(R_back, R_world, atol=1e-9)
        assert np.allclose(t_back, t_world, atol=1e-9)


def test_composed_tcamobj_to_world_uses_mm_correctly(camera):
    """Die ComposedPipeline zerlegt T_cam_obj (Meter, §1) und ruft bop_pose_to_world
    mit t in **mm**. Verifiziert die Einheit: Identitaet 0.30 m vor Cam, R_w2c=I,
    t_w2c=500 mm → t_world.z ≈ (300 - 0) mm ... in Welt minus table_origin."""
    R_w2c = np.eye(3)
    t_w2c = np.array(camera["cam_t_w2c"], float)        # [0,0,500] mm
    to = [0.0, 0.0, 0.08]
    T = np.eye(4); T[2, 3] = 0.30                        # 0.30 m (§1)
    R_m2c, t_m2c_mm = T[:3, :3], T[:3, 3] * 1000.0       # → mm (wie composed)
    R_world, t_world = A.bop_pose_to_world(R_m2c, t_m2c_mm, R_w2c, t_w2c, to)
    # t_m2w = R_w2c^T @ (t_m2c - t_w2c) = (300 - 500) = -200 mm → -0.2 m; minus table z 0.08
    assert np.allclose(t_world, [0.0, 0.0, -0.2 - 0.08], atol=1e-9)
    assert np.allclose(R_world, np.eye(3), atol=1e-9)


# ── 2) Klassen-Mapping (§6): obj_id → CamelCase-Part, Symmetrie greift ─────────
def test_class_mapping_via_obj_id_not_lowercase():
    """Die §6-Falle: die lowercase Mesh-Klasse darf NICHT direkt in PART_SYMMETRY.
    Ueber obj_id muss der CamelCase-Part kommen, und DER muss Symmetrie haben."""
    assert CLASS_TO_OBJ_ID == {"anker_kurz": 1, "anker_lang": 2}
    for cls, obj_id in CLASS_TO_OBJ_ID.items():
        part = A.part_for_obj_id(obj_id)
        assert part in ("Anker_Kurz", "Anker_Lang")     # CamelCase!
        # Die direkte (falsche) Variante liefert None — DAS ist die Falle:
        assert A.PART_SYMMETRY.get(cls) is None
        # Der korrekte (CamelCase) Part hat die continuous-Y-Symmetrie:
        sym = A.PART_SYMMETRY.get(part)
        assert sym is not None and sym["type"] == "continuous"
        assert sym["axis"] == [0.0, 1.0, 0.0]


def test_canonicalize_is_applied_for_symmetric_anker():
    """canonicalize_rotation greift ueber den CamelCase-Part (sonst No-Op). Ein um die
    Symmetrieachse gedrehtes R muss auf den kanonischen Repraesentanten projiziert
    werden (deterministisch, gegen Yaw-Flackern). continuous wird ueber n_steps=64
    diskretisiert → Repraesentanten sind bis auf die Schrittweite (2π/64 ≈ 5.6°)
    gleich, nicht exakt; ein um 70.7° gedrehtes R muss klar Richtung Identitaet
    projiziert werden (Yaw kollabiert)."""
    part = A.part_for_obj_id(1)                          # Anker_Kurz (continuous-Y)
    R0 = np.eye(3)
    R_yawed = R0 @ A.axis_angle_matrix([0, 1, 0], 1.234) # 70.7° um Body-Y
    R_canon_a = A.canonicalize_rotation(R0, part)
    R_canon_b = A.canonicalize_rotation(R_yawed, part)
    step = 2.0 * np.pi / 64.0                            # Diskretisierungs-Schrittweite
    # Zwei symmetrie-aequivalente Posen → derselbe kanonische Repraesentant
    # (bis auf eine Schrittweite). Geo-Distanz ueber den Rotationswinkel.
    R_rel = R_canon_a.T @ R_canon_b
    ang = np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    assert ang <= step + 1e-6, f"kanonische Repraesentanten zu weit auseinander: {np.degrees(ang):.1f}°"
    # Die Kanonisierung muss den Yaw kollabieren: canon(yawed) ist viel naeher an
    # Identitaet als das rohe yawed-R (sonst greift die Symmetrie nicht).
    def _ang_to_I(R):
        return np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    assert _ang_to_I(R_canon_b) < _ang_to_I(R_yawed) - 1e-3
    assert _ang_to_I(R_canon_b) <= step + 1e-6          # auf <= eine Schrittweite kollabiert
    # Gegenprobe: mit der lowercase-Klasse (= None-Symmetrie) waere es ein No-Op:
    assert np.allclose(A.canonicalize_rotation(R_yawed, "anker_kurz"), R_yawed)


# ── 3) Jede ComposedPipeline gegen Mock-Service: schema-valides Doc ───────────
def test_composed_infer_produces_valid_pose_result(rgb_file, camera):
    cp = _composed_with_mocks(pose_needs_depth=False)   # ohne Depth → kein File noetig
    doc = cp.infer(rgb_file, camera, [0.0, 0.0, 0.08])
    assert contract.validate(doc) == [], contract.validate(doc)
    # Unbekannte Klasse (schraube) gefiltert → genau 2 Ergebnisse (anker_kurz/lang).
    assert len(doc["results"]) == 2
    parts = {r["part"] for r in doc["results"]}
    assert parts == {"Anker_Kurz", "Anker_Lang"}        # CamelCase im Welt-Doc (§6)
    for r in doc["results"]:
        assert isinstance(r["face"], str) and r["face"]
        assert isinstance(r["upright"], bool)
        assert len(r["bbox_2d"]) == 4 and r["bbox_2d"][2] > r["bbox_2d"][0]


def test_each_whitelist_combo_runs_against_mock(rgb_file, camera, monkeypatch, tmp_path):
    """ALLE 6 NICHT-A-ComposedPipelines: gegen Mock instanziierbar + lauffaehig.
    Wir patchen post_json der gebauten Adapter auf die Mocks (kein Netz, keine GPU).
    needs_depth-Kombis bekommen eine echte Depth-PNG daneben."""
    # Depth-Datei neben dem RGB (uint16 mm) fuer die needs_depth-Kombis.
    from PIL import Image
    depth = (np.ones((64, 64), dtype=np.uint16) * 300)
    Image.fromarray(depth).save(pathlib.Path(rgb_file).with_name("depth_000001.png"))

    seg_resp = _MockSeg("x").post_json("/segment", {"rgb_b64": "x"})
    pose_mock = _MockPose("x")

    def fake_seg_post(self, path, payload):
        return seg_resp

    def fake_pose_post(self, path, payload):
        return pose_mock.post_json(path, payload)

    monkeypatch.setattr(SegAdapter, "post_json", fake_seg_post, raising=True)
    monkeypatch.setattr(PoseAdapter, "post_json", fake_pose_post, raising=True)

    built = combos.build_combos()
    assert len(built) == 6, f"erwarte 6 NICHT-A-Kombis, got {len(built)}"
    for cp in built:
        doc = cp.infer(rgb_file, camera, [0.0, 0.0, 0.08])
        assert contract.validate(doc) == [], f"{cp.id}: {contract.validate(doc)}"
        assert len(doc["results"]) == 2, cp.id


# ── 4) needs_depth-Gating (§5) ────────────────────────────────────────────────
def test_needs_depth_gating_forwards_depth_only_when_required(rgb_file, camera, tmp_path):
    from PIL import Image
    Image.fromarray((np.ones((64, 64), dtype=np.uint16) * 250)).save(
        pathlib.Path(rgb_file).with_name("depth_000001.png"))

    # needs_depth=True → depth_b64 muss im Pose-Payload landen.
    cp_d = _composed_with_mocks(pose_needs_depth=True)
    cp_d.infer(rgb_file, camera, [0.0, 0.0, 0.08])
    assert "depth_b64" in cp_d.pose.last_payload, "needs_depth=True muss depth forwarden"

    # needs_depth=False → KEIN depth_b64 (RGB-only-Pfad, §5).
    cp_rgb = _composed_with_mocks(pose_needs_depth=False)
    cp_rgb.infer(rgb_file, camera, [0.0, 0.0, 0.08])
    assert "depth_b64" not in cp_rgb.pose.last_payload, "needs_depth=False darf KEIN depth senden"


def test_gigapose_pipeline_flag_in_payload(rgb_file, camera, tmp_path):
    from PIL import Image
    Image.fromarray((np.ones((64, 64), dtype=np.uint16) * 250)).save(
        pathlib.Path(rgb_file).with_name("depth_000001.png"))
    cp = _composed_with_mocks(pose_needs_depth=True, pipeline="rgbd")
    cp.infer(rgb_file, camera, [0.0, 0.0, 0.08])
    assert cp.pose.last_payload.get("pipeline") == "rgbd"  # §5: Kabsch-Gate-Flag


# ── 5) Whitelist-Invarianten (§5) ─────────────────────────────────────────────
def test_whitelist_is_exactly_7_not_12():
    wl = combos.COMBO_WHITELIST
    assert len(wl) == 7, "GENAU 7 Kombis (GDRNPP-Kopplung verbietet die uebrigen, §4)"
    a = [c for c in wl if c["is_pipeline_a"]]
    assert len(a) == 1 and a[0]["id"] == "gdrnpp" and a[0]["seg"] == "yolo-obb"
    # needs_depth-Regel (§5): FP + GigaPose-3D brauchen Depth; GDRNPP + GigaPose-2D nicht.
    by_id = {c["id"]: c for c in wl}
    assert by_id["gdrnpp"]["needs_depth"] is False
    assert by_id["yolo_seg__foundationpose"]["needs_depth"] is True
    assert by_id["sam3__foundationpose"]["needs_depth"] is True
    assert by_id["yolo_seg__gigapose_rgbd"]["needs_depth"] is True
    assert by_id["yolo_seg__gigapose_rgb"]["needs_depth"] is False
    assert by_id["sam3__gigapose_rgbd"]["needs_depth"] is True
    assert by_id["sam3__gigapose_rgb"]["needs_depth"] is False


def test_registry_has_all_7_and_gdrnpp_is_monolith():
    ids = [p["id"] for p in registry.all_pipelines()]
    for c in combos.COMBO_WHITELIST:
        assert c["id"] in ids, f"{c['id']} fehlt in der Registry"
    # gdrnpp ist NICHT eine ComposedPipeline (= der bestehende Monolith-Adapter).
    assert not isinstance(registry.get("gdrnpp"), ComposedPipeline)
    # Die 6 NICHT-A sind ComposedPipelines.
    for c in combos.COMBO_WHITELIST:
        if not c["is_pipeline_a"]:
            assert isinstance(registry.get(c["id"]), ComposedPipeline)


# ── 6) Skip-Verhalten + Detection-Helfer ──────────────────────────────────────
def test_pose_list_shorter_than_detections_is_ok(rgb_file, camera):
    """§3: ein Estimator darf Instanzen still skippen → die Pose-Liste ist kuerzer.
    Die ComposedPipeline darf NICHT crashen, nur weniger Ergebnisse liefern."""
    class _SkipPose(_MockPose):
        def post_json(self, path, payload):
            # Nur die ERSTE Instanz posieren (Rest skippen).
            full = super().post_json(path, payload)
            full["poses"] = full["poses"][:1]
            return full
    seg = _MockSeg("x"); seg.id = "mock-seg"
    pose = _SkipPose("x"); pose.id = "skip-pose"; pose.needs_depth = False
    cp = ComposedPipeline(combo_id="skip", name="Skip", seg=seg, pose=pose)
    doc = cp.infer(rgb_file, camera, [0.0, 0.0, 0.08])
    assert contract.validate(doc) == []
    assert len(doc["results"]) == 1                     # 2 detections, 1 pose → 1 result


def test_detection_from_wire_requires_mask():
    with pytest.raises(ValueError):
        Detection.from_wire({"id": 0, "class": "anker_kurz", "conf": 0.5})  # kein mask_b64


def test_pose_as_4x4_accepts_flat16_and_nested():
    flat = list(range(16))
    nested = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert _as_4x4(flat) == [[float(x) for x in r] for r in nested]
    assert _as_4x4(nested) == [[float(x) for x in r] for r in nested]
    with pytest.raises(ValueError):
        _as_4x4([1, 2, 3])


def test_mask_b64_to_bbox():
    b = _mask_b64(10, 12, 30, 40, hw=(64, 64))
    assert _mask_b64_to_bbox(b) == [10, 12, 30, 40]
    empty = _png_b64(np.zeros((64, 64), dtype=np.uint8))
    assert _mask_b64_to_bbox(empty) == [0, 0, 0, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
