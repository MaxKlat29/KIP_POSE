#!/usr/bin/env python3
"""Unit-Tests für den BOP -> pose_result Adapter (Viktor adr.md §3).

Kritischer Test: **Round-Trip**. Eine bekannte Welt-Pose wird in den Kamera-Frame
projiziert (R_m2c, t_m2c erzeugt, BOP-Konvention), dann durch den Adapter geschickt
— er MUSS die Original-Welt-Pose (±1e-6) zurückgeben. Geprüft mit mm/m-Skalierung,
table_origin, zufälligen SO(3)-Rotationen und nicht-trivialen Extrinsics.

Lauf:  python3 -m pytest project/tests/test_bop_adapter.py -q
   oder python3 project/tests/test_bop_adapter.py   (Self-Runner ohne pytest)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Adapter aus dem Geschwister-Verzeichnis importierbar machen (kein Package).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bop_adapter as A  # noqa: E402


# ── Hilfen: SO(3) erzeugen + BOP-Forward-Projektion (Inverse des Adapters) ────
def rand_SO3(rng) -> np.ndarray:
    """Gleichverteilte zufällige Rotationsmatrix via QR mit Determinanten-Fix."""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))          # eindeutige QR
    if np.linalg.det(Q) < 0:                       # in SO(3) zwingen
        Q[:, 0] = -Q[:, 0]
    return Q


def world_to_bop(R_world, t_world_m, R_w2c, t_w2c_mm, table_origin_m):
    """Forward = exakte Inverse der Adapter-Kette (R_model_to_body = I).

    Adapter: R_m2w = R_w2c.T @ R_m2c ; t_m2w = R_w2c.T @ (t_m2c - t_w2c)
             t_world = t_m2w/1000 - table_origin
    Invertiert:
             t_m2w_mm = (t_world + table_origin) * 1000
             R_m2c = R_w2c @ R_world            (R_world == R_m2w bei R_mb=I)
             t_m2c = R_w2c @ t_m2w_mm + t_w2c
    """
    t_m2w_mm = (np.asarray(t_world_m) + np.asarray(table_origin_m)) * 1000.0
    R_m2c = R_w2c @ R_world
    t_m2c = R_w2c @ t_m2w_mm + np.asarray(t_w2c_mm)
    return R_m2c, t_m2c


# ── Round-Trip-Tests ──────────────────────────────────────────────────────────
def test_roundtrip_identity_extrinsics():
    """Triviale Extrinsics (Kamera = Welt): Adapter muss exakt zurückgeben."""
    R_world = np.eye(3)
    t_world = np.array([0.05, -0.02, 0.10])
    R_w2c = np.eye(3)
    t_w2c = np.zeros(3)
    table_origin = np.array([0.0, 0.0, 0.0])

    R_m2c, t_m2c = world_to_bop(R_world, t_world, R_w2c, t_w2c, table_origin)
    R_out, t_out = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
    assert np.allclose(R_out, R_world, atol=1e-6)
    assert np.allclose(t_out, t_world, atol=1e-6)


def test_roundtrip_with_table_origin_and_mm():
    """table_origin != 0 + mm->m: Skalierung + Verschiebung müssen exakt zurück."""
    R_world = A.axis_angle_matrix([0, 0, 1], np.radians(37.0))
    t_world = np.array([0.123, 0.456, 0.0789])
    R_w2c = np.eye(3)
    t_w2c = np.zeros(3)
    table_origin = np.array([0.10, -0.05, 0.08])   # nicht-trivialer Tisch-Nullpunkt

    R_m2c, t_m2c = world_to_bop(R_world, t_world, R_w2c, t_w2c, table_origin)
    R_out, t_out = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
    assert np.allclose(R_out, R_world, atol=1e-6)
    assert np.allclose(t_out, t_world, atol=1e-6)


def test_roundtrip_nontrivial_extrinsics():
    """Nicht-triviale Kamera-Extrinsics (gedreht + verschoben in mm)."""
    rng = np.random.default_rng(0)
    R_world = rand_SO3(rng)
    t_world = np.array([0.2, -0.15, 0.05])
    R_w2c = rand_SO3(rng)
    t_w2c = np.array([120.0, -340.0, 560.0])       # mm
    table_origin = np.array([0.03, 0.07, 0.08])

    R_m2c, t_m2c = world_to_bop(R_world, t_world, R_w2c, t_w2c, table_origin)
    R_out, t_out = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
    assert np.allclose(R_out, R_world, atol=1e-6)
    assert np.allclose(t_out, t_world, atol=1e-6)


def test_roundtrip_many_random():
    """50 zufällige SO(3)-Posen + zufällige Extrinsics: alle ±1e-6 zurück."""
    rng = np.random.default_rng(42)
    max_R_err, max_t_err = 0.0, 0.0
    for _ in range(50):
        R_world = rand_SO3(rng)
        t_world = rng.uniform(-0.3, 0.3, size=3)
        R_w2c = rand_SO3(rng)
        t_w2c = rng.uniform(-800.0, 800.0, size=3)   # mm
        table_origin = rng.uniform(-0.1, 0.1, size=3)

        R_m2c, t_m2c = world_to_bop(R_world, t_world, R_w2c, t_w2c, table_origin)
        R_out, t_out = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
        max_R_err = max(max_R_err, float(np.max(np.abs(R_out - R_world))))
        max_t_err = max(max_t_err, float(np.max(np.abs(t_out - t_world))))
    assert max_R_err < 1e-6, f"max R err {max_R_err}"
    assert max_t_err < 1e-6, f"max t err {max_t_err}"


def test_roundtrip_with_R_model_to_body():
    """R_model_to_body (90°-Achsenpermutation) muss korrekt mitkomponiert werden.

    Forward mit body-Frame: R_m2c = R_w2c @ R_m2w, R_m2w = R_world @ R_mb^T,
    weil Adapter R_world = R_m2w @ R_mb rechnet.
    """
    rng = np.random.default_rng(7)
    R_world = rand_SO3(rng)                          # gewünschtes Contract-R
    t_world = np.array([0.1, 0.2, 0.05])
    R_w2c = rand_SO3(rng)
    t_w2c = np.array([10.0, 20.0, 700.0])
    table_origin = np.array([0.0, 0.0, 0.08])
    # 90° um X als feste Modell->Body-Brücke.
    R_mb = A.axis_angle_matrix([1, 0, 0], np.radians(90.0))

    R_m2w = R_world @ R_mb.T                          # so dass R_m2w @ R_mb = R_world
    R_m2c = R_w2c @ R_m2w
    t_m2w_mm = (t_world + table_origin) * 1000.0
    t_m2c = R_w2c @ t_m2w_mm + t_w2c

    R_out, t_out = A.bop_pose_to_world(
        R_m2c, t_m2c, R_w2c, t_w2c, table_origin, R_model_to_body=R_mb)
    assert np.allclose(R_out, R_world, atol=1e-6)
    assert np.allclose(t_out, t_world, atol=1e-6)


def test_output_is_valid_SO3():
    """Adapter-Ausgabe bleibt eine gültige Rotationsmatrix (orthonormal, det=+1)."""
    rng = np.random.default_rng(99)
    R_world = rand_SO3(rng)
    R_w2c = rand_SO3(rng)
    R_m2c, t_m2c = world_to_bop(R_world, [0.1, 0, 0.05], R_w2c, [0, 0, 500],
                                [0, 0, 0.08])
    R_out, _ = A.bop_pose_to_world(R_m2c, t_m2c, R_w2c, [0, 0, 500], [0, 0, 0.08])
    assert np.allclose(R_out @ R_out.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R_out) - 1.0) < 1e-6


# ── obj_id-Mapping (§1.2) ─────────────────────────────────────────────────────
def test_obj_id_mapping():
    assert A.part_for_obj_id(1) == "Anker_Kurz"
    assert A.part_for_obj_id(2) == "Anker_Lang"
    assert A.part_for_obj_id(6) == "Zahnrad"
    # Detektor-Klasse (0-basiert) +1 = obj_id.
    assert A.category_id_to_obj_id(0) == 1            # anker_kurz
    assert A.category_id_to_obj_id(5) == 6            # zahnrad
    # Konsistenz mit der Detektor-Klassen-Reihenfolge.
    for cls0, name in enumerate(A.DETECTOR_CLASS_ORDER):
        obj_id = A.category_id_to_obj_id(cls0)
        assert A.part_for_obj_id(obj_id).lower() == name


# ── face/upright-Ableitung (§3.4 / R8) ────────────────────────────────────────
def test_upright_true_when_long_axis_vertical():
    """Anker (Body-Längsachse Y) steht aufrecht: body-Y -> Welt-Z => upright."""
    # world = R @ body. body-Y soll auf world-Z zeigen: R[:,1] = [0,0,1].
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])                  # body-Y -> world+Z
    face, upright = A.face_and_upright_from_R(R, part="Anker_Lang")
    assert upright is True


def test_upright_false_when_long_axis_horizontal():
    """Anker liegt flach: body-Y in der Tischebene => nicht upright."""
    R = np.eye(3)                                     # body-Y -> world+Y (flach)
    face, upright = A.face_and_upright_from_R(R, part="Anker_Lang")
    assert upright is False


def test_face_is_axis_pointing_down():
    """Die Body-Achse Richtung Welt-DOWN (-Z) bestimmt den face-Namen."""
    # Identität: body-Z -> world+Z, also body-(-Z) zeigt nach unten => "face_z-".
    face, _ = A.face_and_upright_from_R(np.eye(3), part="Zahnrad")
    assert face == "face_z-"
    # 180° um X: body-Z -> world-Z, body+Z zeigt nach unten => "face_z+".
    R = A.axis_angle_matrix([1, 0, 0], np.pi)
    face, _ = A.face_and_upright_from_R(R, part="Zahnrad")
    assert face == "face_z+"


def test_face_upright_deterministic():
    """Gleiche Rotation -> gleicher face/upright (Determinismus für Viewer)."""
    rng = np.random.default_rng(3)
    R = rand_SO3(rng)
    a = A.face_and_upright_from_R(R, part="Anker_Lang")
    b = A.face_and_upright_from_R(R, part="Anker_Lang")
    assert a == b


# ── Symmetrie-Kanonisierung (§3.3) ────────────────────────────────────────────
def test_continuous_canonicalization_is_idempotent():
    """Zweimal kanonisieren == einmal (stabiler Fixpunkt)."""
    rng = np.random.default_rng(11)
    R = rand_SO3(rng)
    c1 = A.canonicalize_rotation(R, "Anker_Lang")
    c2 = A.canonicalize_rotation(c1, "Anker_Lang")
    assert np.allclose(c1, c2, atol=1e-2)            # diskretisiert -> grobe Toleranz


def test_continuous_canon_collapses_yaw_about_axis():
    """Zwei Posen, die sich NUR um die Symmetrieachse unterscheiden, kanonisieren
    auf (nahezu) dieselbe Rotation — das ist die analytische 91°-Auflösung."""
    rng = np.random.default_rng(13)
    base = rand_SO3(rng)
    axis = [0, 1, 0]
    Ra = base @ A.axis_angle_matrix(axis, np.radians(20.0))
    Rb = base @ A.axis_angle_matrix(axis, np.radians(200.0))
    ca = A.canonicalize_rotation(Ra, "Anker_Lang", n_steps_continuous=720)
    cb = A.canonicalize_rotation(Rb, "Anker_Lang", n_steps_continuous=720)
    # Restdifferenz ist eine reine Drehung um die Achse mit kleinem Winkel.
    rel = ca.T @ cb
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert angle < 1.0, f"canon residual {angle} deg"


def test_no_symmetry_part_unchanged():
    """Teil ohne Symmetrie: Kanonisierung ist Identität."""
    rng = np.random.default_rng(5)
    R = rand_SO3(rng)
    out = A.canonicalize_rotation(R, "Getriebegehaeuse_typ4")
    assert np.allclose(out, R, atol=1e-12)


def test_discrete_unknown_N_unchanged():
    """Zahnrad mit n_fold=None (N noch nicht gezählt): unverändert."""
    rng = np.random.default_rng(6)
    R = rand_SO3(rng)
    out = A.canonicalize_rotation(R, "Zahnrad")        # n_fold default None
    assert np.allclose(out, R, atol=1e-12)


def test_discrete_canon_with_N():
    """Mit N gesetzt: Zahnrad-Posen, die sich um 2pi/N unterscheiden, kollabieren."""
    sym = {"type": "discrete", "axis": [0, 1, 0], "n_fold": 12}
    rng = np.random.default_rng(8)
    base = rand_SO3(rng)
    Ra = base @ A.axis_angle_matrix([0, 1, 0], 2 * np.pi / 12 * 3)   # 3 Schritte
    Rb = base @ A.axis_angle_matrix([0, 1, 0], 2 * np.pi / 12 * 7)   # 7 Schritte
    ca = A.canonicalize_rotation(Ra, "Zahnrad", symmetry=sym)
    cb = A.canonicalize_rotation(Rb, "Zahnrad", symmetry=sym)
    assert np.allclose(ca, cb, atol=1e-6)


# ── detection_to_result: voller Eintrag, schema-Form ──────────────────────────
def test_detection_to_result_shape():
    rng = np.random.default_rng(21)
    R_world = rand_SO3(rng)
    R_w2c = np.eye(3)
    t_w2c = np.zeros(3)
    table_origin = [0.0, 0.0, 0.08]
    R_m2c, t_m2c = world_to_bop(R_world, [0.1, 0.0, 0.05], R_w2c, t_w2c,
                                table_origin)
    r = A.detection_to_result(
        instance_id=0, obj_id=4,                       # Getriebe = keine Symmetrie
        R_m2c=R_m2c, t_m2c_mm=t_m2c, R_w2c=R_w2c, t_w2c_mm=t_w2c,
        table_origin_m=table_origin, bbox_2d=[10, 20, 100, 200], confidence=0.9)
    assert r["part"] == "Getriebegehaeuse_typ4"
    assert len(r["R_world"]) == 9 and all(isinstance(v, float) for v in r["R_world"])
    assert len(r["t_world"]) == 3
    assert r["bbox_2d"] == [10, 20, 100, 200]
    assert 0.0 <= r["confidence"] <= 1.0
    assert isinstance(r["upright"], bool)
    assert r["face"].startswith("face_")
    # Da Getriebe keine Symmetrie hat, ist der Round-Trip exakt:
    assert np.allclose(np.array(r["R_world"]).reshape(3, 3), R_world, atol=1e-6)


# ── Self-Runner (ohne pytest) ─────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} adapter tests green")
