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


# ── Planares Tisch-Ebenen-Refinement (§3.5 / T-055) ───────────────────────────
def _box_verts(hx, hy, hz):
    """8 Eckpunkte einer achsenparallelen Box mit Halb-Ausdehnungen (Body-Frame)."""
    import itertools
    return np.array([[sx * hx, sy * hy, sz * hz]
                     for sx, sy, sz in itertools.product((-1, 1), (-1, 1), (-1, 1))],
                    dtype=np.float64)


def test_lowest_contact_z_identity():
    """Bei R=I ist der tiefste Punkt -hz (Box-Unterkante im Body-Frame)."""
    V = _box_verts(0.01, 0.05, 0.01)                 # 2cm x 10cm x 2cm Box
    assert abs(A.lowest_contact_z(np.eye(3), V) - (-0.01)) < 1e-12


def test_lowest_contact_z_rotated():
    """Body 90° um X gedreht: die lange Y-Achse zeigt in Welt-Z -> tiefster Punkt -hy."""
    V = _box_verts(0.01, 0.05, 0.01)
    R = A.axis_angle_matrix([1, 0, 0], np.radians(90.0))  # body-Y -> world-±Z
    assert abs(A.lowest_contact_z(R, V) - (-0.05)) < 1e-9


def test_z_snap_restores_resting_height_after_offset():
    """Liegendes Teil mit künstlichem Z-Offset: Z-Snap stellt die Auflage (z=0) her."""
    V = _box_verts(0.02, 0.05, 0.015)                # flach liegende Quader-Box
    R = np.eye(3)                                     # liegt flach
    # korrekte ruhende Höhe: tiefster Punkt auf z=0 -> origin.z = +hz = 0.015
    t_true = np.array([0.1, -0.2, 0.015])
    # künstlicher Z-Fehler (RGB-Tiefenfehler), x/y unverändert:
    t_noisy = t_true + np.array([0.0, 0.0, 0.042])
    t_snapped, dz = A.planar_z_snap(R, t_noisy, V, table_z=0.0)
    assert np.allclose(t_snapped[:2], t_noisy[:2], atol=1e-12)   # x/y unangetastet
    assert abs(t_snapped[2] - t_true[2]) < 1e-9                  # Auflage wiederhergestellt
    # der tiefste Welt-Punkt sitzt jetzt exakt auf der Tischebene:
    contact = t_snapped[2] + A.lowest_contact_z(R, V)
    assert abs(contact) < 1e-9
    assert dz < 0                                                # nach unten korrigiert


def test_z_snap_idempotent():
    """Zweimal snappen == einmal (Fixpunkt; das Teil ruht schon)."""
    V = _box_verts(0.02, 0.05, 0.015)
    R = A.axis_angle_matrix([0, 0, 1], np.radians(33.0))   # nur Yaw, liegt flach
    t = np.array([0.05, 0.05, 0.5])
    t1, _ = A.planar_z_snap(R, t, V, table_z=0.0)
    t2, dz2 = A.planar_z_snap(R, t1, V, table_z=0.0)
    assert np.allclose(t1, t2, atol=1e-12)
    assert abs(dz2) < 1e-12


def test_z_snap_works_for_standing_part_too():
    """Stehendes/hochkant Teil: Z-Snap setzt korrekt den tiefsten Punkt auf z=0
    (verschlimmert NICHTS, x/y/Rotation unverändert)."""
    V = _box_verts(0.01, 0.05, 0.01)
    R = A.axis_angle_matrix([1, 0, 0], np.radians(90.0))   # steht hochkant (Y->Z)
    t = np.array([0.0, 0.0, 0.2])
    t_snapped, dz = A.planar_z_snap(R, t, V, table_z=0.0)
    contact = t_snapped[2] + A.lowest_contact_z(R, V)
    assert abs(contact) < 1e-9                              # tiefster Punkt auf Tisch
    assert np.allclose(t_snapped[:2], t[:2], atol=1e-12)   # x/y unangetastet


def test_z_snap_guard_skips_far_off_part():
    """Guard: ein weit über dem Tisch platziertes Teil (Greifer/in der Luft) wird
    NICHT gesnappt — der Planar-Prior gilt dort nicht (T-055-Held-Teil-Schutz)."""
    V = _box_verts(0.02, 0.05, 0.015)
    R = np.eye(3)
    # Teil 0.5m über dem Tisch -> dz ~ -0.515m, weit über dem Guard (0.05m).
    t_far = np.array([0.0, 0.0, 0.5])
    _, t_out, info = A.planar_refine(R, t_far, V, table_z=0.0, max_snap_m=0.05)
    assert info["snap_skipped"] is True and info["z_snap"] is False
    assert np.allclose(t_out, t_far, atol=1e-12)         # Pose unverändert
    # mit max_snap_m=None (kein Guard) WIRD gesnappt:
    _, t_snap, info2 = A.planar_refine(R, t_far, V, table_z=0.0, max_snap_m=None)
    assert info2["z_snap"] is True
    assert abs(t_snap[2] + A.lowest_contact_z(R, V)) < 1e-9


def test_z_snap_guard_allows_resting_part():
    """Guard lässt resting-Teile (kleiner Tiefenfehler) durch."""
    V = _box_verts(0.02, 0.05, 0.015)
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.015 + 0.03])               # 30mm Tiefenfehler < 50mm Guard
    _, t_out, info = A.planar_refine(R, t, V, table_z=0.0, max_snap_m=0.05)
    assert info["z_snap"] is True and info["snap_skipped"] is False
    assert abs(t_out[2] - 0.015) < 1e-9                  # auf die Auflage gesnappt


def test_planar_refine_default_only_z_snap():
    """planar_refine() default: nur Z-Snap, Rotation BLEIBT (tilt_correct aus).
    Höhe innerhalb des Guards (resting), daher wird gesnappt."""
    V = _box_verts(0.02, 0.05, 0.015)
    R = A.axis_angle_matrix([0, 0, 1], np.radians(20.0))    # flach, nur Yaw
    t = np.array([0.1, 0.1, 0.015 + 0.025])                 # 25mm Tiefenfehler < 50mm Guard
    R_out, t_out, info = A.planar_refine(R, t, V, table_z=0.0)
    assert np.allclose(R_out, R, atol=1e-12)               # Rotation unverändert
    assert info["z_snap"] is True and info["tilt"] is False
    assert abs(t_out[2] + A.lowest_contact_z(R, V)) < 1e-9


def _flat_plate_verts(hx, hy, n=12):
    """Dichtes, planares Bodengitter (z=-eps) + dünner Korpus darüber.

    Modelliert ein Teil mit einer echten planaren Auflagefläche: ein nxn-Gitter
    bei z=-0.003 (die Auflage) plus ein paar Punkte darüber. So fällt eine ganze
    Vertex-Fläche ins Kontakt-Band -> die Tilt-Heuristik greift (wie bei einem
    realen CAD-Mesh mit flacher Unterseite)."""
    gx = np.linspace(-hx, hx, n)
    gy = np.linspace(-hy, hy, n)
    floor = np.array([[x, y, -0.003] for x in gx for y in gy], dtype=np.float64)
    top = np.array([[x, y, 0.003] for x in (-hx, hx) for y in (-hy, hy)], dtype=np.float64)
    return np.vstack([floor, top])


def test_planar_tilt_correct_fixes_small_tilt_on_flat_part():
    """Konservative Tilt-Korrektur: ein flach liegendes Teil mit echter planarer
    Auflagefläche + KLEINER Kippung wird zur Tischnormalen geradegerückt."""
    V = _flat_plate_verts(0.03, 0.04, n=12)          # planare Bodenfläche
    tilt = np.radians(6.0)                            # kleine Kippung um X
    R = A.axis_angle_matrix([1, 0, 0], tilt)
    t = np.array([0.0, 0.0, 0.1])
    R_out, t_out, info = A.planar_refine(R, t, V, table_z=0.0, tilt_correct=True)
    assert info["tilt"] is True
    # nach der Korrektur liegt die Plattennormale (~Body-Z) ~ auf Welt-Z:
    plate_normal_world = R_out[:, 2] / np.linalg.norm(R_out[:, 2])
    cos = abs(float(plate_normal_world @ np.array([0.0, 0.0, 1.0])))
    assert cos > 0.999                               # praktisch flach


def test_planar_tilt_correct_skips_standing_part():
    """Stehendes/kantiges Teil: Tilt-Korrektur fasst die Rotation NICHT an
    (Auflage ist kein planares Band -> keine Verschlimmerung)."""
    V = _box_verts(0.01, 0.06, 0.01)                 # langer dünner Stab
    R = A.axis_angle_matrix([1, 0, 0], np.radians(90.0))   # steht hochkant
    t = np.array([0.0, 0.0, 0.1])
    R_out, t_out, info = A.planar_refine(R, t, V, table_z=0.0, tilt_correct=True)
    # Rotation darf NICHT verändert worden sein (nur Z-Snap):
    assert np.allclose(R_out, R, atol=1e-12)
    assert info["tilt"] is False


def test_detection_to_result_with_planar_snap():
    """detection_to_result(apply_planar=True): t_world.z wird auf die Tischebene
    gesnappt, Schema-Form bleibt; ohne Mesh kein Snap (kein Crash)."""
    rng = np.random.default_rng(31)
    V = _box_verts(0.02, 0.05, 0.015)
    R_world = A.axis_angle_matrix([0, 0, 1], np.radians(45.0))  # flach, Yaw
    R_w2c = np.eye(3); t_w2c = np.zeros(3); table_origin = [0.0, 0.0, 0.0]
    # absichtlich falsche Höhe (z weit über dem Tisch):
    R_m2c, t_m2c = world_to_bop(R_world, [0.1, 0.0, 0.25], R_w2c, t_w2c, table_origin)
    r = A.detection_to_result(
        instance_id=0, obj_id=4, R_m2c=R_m2c, t_m2c_mm=t_m2c,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, table_origin_m=table_origin,
        bbox_2d=[1, 2, 3, 4], confidence=0.8,
        apply_planar=True, mesh_verts_m=V, table_z=0.0, max_snap_m=None)
    contact = r["t_world"][2] + A.lowest_contact_z(np.array(r["R_world"]).reshape(3, 3), V)
    assert abs(contact) < 1e-9                        # auf der Tischebene
    # ohne Mesh: kein Snap, Pose unverändert (Höhe bleibt 0.25):
    r2 = A.detection_to_result(
        instance_id=0, obj_id=4, R_m2c=R_m2c, t_m2c_mm=t_m2c,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, table_origin_m=table_origin,
        bbox_2d=[1, 2, 3, 4], confidence=0.8, apply_planar=True, mesh_verts_m=None)
    assert abs(r2["t_world"][2] - 0.25) < 1e-9


# ── Self-Runner (ohne pytest) ─────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} adapter tests green")
