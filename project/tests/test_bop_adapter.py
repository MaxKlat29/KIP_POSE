#!/usr/bin/env python3
"""Unit-Tests für den BOP -> pose_result Adapter (adr.md §3).

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


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — M1 Stable-Pose-Snap / M3 Top-Down-C_2 / M4 Contour-Yaw-Lock (T-041)
# ══════════════════════════════════════════════════════════════════════════════
import pytest  # noqa: E402

try:
    import trimesh as _tm
except Exception:  # pragma: no cover
    _tm = None
requires_trimesh = pytest.mark.skipif(_tm is None, reason="trimesh nicht installiert")


def _disk_mesh(radius=0.02, height=0.004, sections=48):
    """Symmetrischer Zylinder (trimesh) — liegt flach, Top-Down-flip-identisch."""
    return _tm.creation.cylinder(radius=radius, height=height, sections=sections)


# ── M1: _min_rot_align ─────────────────────────────────────────────────────────
def test_min_rot_align_basic():
    """Align(a->b) @ a == b, gültige SO(3)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = rng.normal(size=3); b /= np.linalg.norm(b)
        R = A._min_rot_align(a, b)
        assert np.allclose(R @ a, b, atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_min_rot_align_antiparallel():
    """a == -b: 180°-Drehung um eine senkrechte Achse, immer noch a->b."""
    a = np.array([0.0, 0.0, 1.0])
    R = A._min_rot_align(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


# ── M1: stable_pose_snap (synthetische stable-downs, kein trimesh) ─────────────
def test_stable_pose_snap_snaps_tilt_preserves_yaw():
    """Eine flach-liegende Rotation + kleine Kippung snappt auf die Ruhelage zurück;
    der In-Plane-Yaw um Welt-Z bleibt erhalten."""
    down = np.array([0.0, 0.0, -1.0])                 # body-Z zeigt nach Welt-DOWN
    downs = np.array([down])                           # eine Ruhelage
    # exakte flache Pose: body-down == down. Yaw 30° um Welt-Z dazu.
    R_flat = A._min_rot_align(down, np.array([0.0, 0.0, -1.0]))
    R_yaw = A.axis_angle_matrix([0, 0, 1], np.radians(30.0))
    R_flat = R_yaw @ R_flat                            # In-Plane-Yaw (um Welt-Z)
    # 12° Kippung (verlässt die Ruhelage):
    R_tilt = A.axis_angle_matrix([0, 1, 0], np.radians(12.0)) @ R_flat
    R_new, info = A.stable_pose_snap(R_tilt, downs, max_tilt_snap_deg=55.0)
    assert info["snapped"] is True
    assert abs(info["tilt_deg"] - 12.0) < 0.5
    # body-down danach exakt auf die Ruhelage:
    bd = R_new.T @ np.array([0.0, 0.0, -1.0]); bd /= np.linalg.norm(bd)
    assert float((downs @ bd).max()) > 0.99999
    # In-Plane-Yaw erhalten: R_new ~ R_flat (die snap entfernt nur die 12°-Kippung).
    rel = R_flat.T @ R_new
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert ang < 1.0, f"yaw nicht erhalten, residual {ang:.2f}deg"
    # gültige SO(3):
    assert np.allclose(R_new @ R_new.T, np.eye(3), atol=1e-9)


def test_stable_pose_snap_idempotent():
    """Zweimal snappen == einmal (das Teil ruht schon)."""
    downs = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    R = A.axis_angle_matrix([1, 0, 0], np.radians(20.0)) @ \
        A.axis_angle_matrix([1, 0, 0], np.pi)
    R1, i1 = A.stable_pose_snap(R, downs, max_tilt_snap_deg=55.0)
    R2, i2 = A.stable_pose_snap(R1, downs, max_tilt_snap_deg=55.0)
    assert i1["snapped"] is True
    assert abs(i2["tilt_deg"]) < 1e-3                  # zweiter Snap ist ein No-Op
    assert np.allclose(R1, R2, atol=1e-9)


def test_stable_pose_snap_guard_skips_standing():
    """Guard: ein weit gekipptes/stehendes Teil (Tilt > guard) wird NICHT gesnappt."""
    downs = np.array([[0.0, 0.0, -1.0]])              # flach-Ruhelage
    # 80° gekippt — über dem 55°-Guard:
    R = A.axis_angle_matrix([0, 1, 0], np.radians(80.0)) @ \
        A.axis_angle_matrix([1, 0, 0], np.pi)
    R_new, info = A.stable_pose_snap(R, downs, max_tilt_snap_deg=55.0)
    assert info["snap_skipped"] is True and info["snapped"] is False
    assert np.allclose(R_new, R, atol=1e-12)          # Pose unverändert


def test_stable_pose_snap_picks_nearest_of_many():
    """Bei mehreren Ruhelagen wird die zur Vorhersage NÄCHSTE gewählt."""
    downs = np.array([[0.0, 0.0, -1.0],               # 0: flach
                      [1.0, 0.0, 0.0],                # 1: auf der Seite (body-X unten)
                      [0.0, 1.0, 0.0]])               # 2: auf der Seite (body-Y unten)
    # Vorhersage nah an Ruhelage 1 (body-X leicht verkippt nach unten):
    base = A._min_rot_align(downs[1], np.array([0.0, 0.0, -1.0]))
    R = A.axis_angle_matrix([0, 1, 0], np.radians(8.0)) @ base
    R_new, info = A.stable_pose_snap(R, downs, max_tilt_snap_deg=55.0)
    assert info["pose_idx"] == 1


def test_stable_pose_snap_validates_shape():
    with pytest.raises(ValueError):
        A.stable_pose_snap(np.eye(3), np.zeros((0, 3)))


# ── M1 in planar_refine (Flag + Kombination mit Z-Snap) ───────────────────────
def test_planar_refine_default_no_m1():
    """Default planar_refine() rührt die Rotation NICHT an (M1 aus)."""
    V = _box_verts(0.02, 0.05, 0.015)
    R = A.axis_angle_matrix([0, 0, 1], np.radians(20.0))
    t = np.array([0.1, 0.1, 0.015 + 0.02])
    R_out, _, info = A.planar_refine(R, t, V, table_z=0.0)
    assert np.allclose(R_out, R, atol=1e-12)
    assert info["m1_snapped"] is False


def test_planar_refine_m1_requires_downs():
    V = _box_verts(0.02, 0.05, 0.015)
    with pytest.raises(ValueError):
        A.planar_refine(np.eye(3), [0, 0, 0.5], V, stable_pose_snap=True)


def test_planar_refine_m1_plus_zsnap_combine():
    """M1 + Z-Snap zusammen: Rotation auf Ruhelage, Höhe auf Tischebene."""
    V = _box_verts(0.02, 0.05, 0.015)                 # flache Box
    downs = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    # gekippte + zu hohe Pose:
    R = A.axis_angle_matrix([0, 1, 0], np.radians(10.0)) @ \
        A.axis_angle_matrix([1, 0, 0], np.pi)         # ~flach + 10° Kippung
    t = np.array([0.1, 0.1, 0.015 + 0.03])            # 30mm zu hoch (< Guard)
    R_out, t_out, info = A.planar_refine(
        R, t, V, table_z=0.0, z_snap=True, stable_pose_snap=True,
        stable_downs=downs, max_tilt_snap_deg=55.0)
    assert info["m1_snapped"] is True
    assert info["z_snap"] is True
    # nach M1 ist die body-down auf einer Ruhelage:
    bd = R_out.T @ np.array([0.0, 0.0, -1.0]); bd /= np.linalg.norm(bd)
    assert float((downs @ bd).max()) > 0.999
    # tiefster Punkt auf der Tischebene:
    assert abs(t_out[2] + A.lowest_contact_z(R_out, V)) < 1e-9


# ── M1: echte trimesh-Ruhelagen ───────────────────────────────────────────────
@requires_trimesh
def test_stable_pose_body_downs_gear_is_flat():
    """Ein flacher Zylinder hat eine dominante Ruhelage 'flach auf der Fläche'
    (body-Z down/up), nach prob_min-Filter."""
    disk = _disk_mesh()
    downs, probs = A.stable_pose_body_downs(mesh=disk, prob_min=0.05)
    # die dominante Ruhelage liegt flach: body-Z (Zylinderachse) zeigt nach
    # world-down ODER world-up (|Z-Komponente| ~ 1).
    assert abs(abs(downs[0][2]) - 1.0) < 1e-3
    assert probs[0] > 0.3


@requires_trimesh
def test_stable_pose_body_downs_cache():
    disk = _disk_mesh()
    d1 = A.stable_pose_body_downs(mesh=disk, cache_key="diskA")
    d2 = A.stable_pose_body_downs(mesh=disk, cache_key="diskA")
    assert d1[0] is d2[0]                              # gecacht (identisches Objekt)


# ── M3: Top-Down-C_2-Check ────────────────────────────────────────────────────
@requires_trimesh
def test_topdown_c2_symmetric_disk_is_flip_identical():
    """Ein symmetrischer Zylinder ist top-down unter 180°-Flip deckungsgleich."""
    disk = _disk_mesh(radius=0.02, height=0.004)
    ident, iou = A.topdown_c2_flip_identical(mesh=disk, n_px=200, iou_thresh=0.95)
    assert ident is True and iou > 0.95


@requires_trimesh
def test_topdown_c2_asymmetric_L_not_flip_identical():
    """Ein klar asymmetrisches Teil (L-förmig) ist NICHT flip-identisch top-down."""
    # L-Form: zwei Boxen, eine lang, eine kurz quer -> 180° überlappt schlecht.
    a = _tm.creation.box(extents=[0.08, 0.01, 0.004])
    b = _tm.creation.box(extents=[0.01, 0.04, 0.004])
    b.apply_translation([0.035, 0.02, 0.0])
    L = _tm.util.concatenate([a, b])
    # flache Top-Down-Rotation = Identität (Teil liegt in xy):
    ident, iou = A.topdown_c2_flip_identical(mesh=L, R_world=np.eye(3),
                                             n_px=200, iou_thresh=0.90)
    assert ident is False and iou < 0.90


# ── M4: Contour-Yaw-Lock ──────────────────────────────────────────────────────
def _gear_mesh(n_teeth=7, r_in=18.0, r_tooth=24.0, h=10.0, marked=False):
    """Synthetisches C_N-Zahnrad in MM (wie das echte CAD): Scheibe + n_teeth
    radiale Zähne. marked=True: EIN Zahn ist deutlich länger → die Silhouette bricht
    die C_N-Symmetrie (so KANN ein Silhouetten-Yaw-Lock überhaupt diskriminieren —
    ein perfekt C_N-symmetrisches Zahnrad hat eine flip-/rotations-invariante
    Top-Down-Silhouette, s. M4-Doku in bop_adapter)."""
    parts = [_tm.creation.cylinder(radius=r_in, height=h, sections=64)]
    for k in range(n_teeth):
        th = 2 * np.pi * k / n_teeth
        rt = r_tooth * (1.6 if (marked and k == 0) else 1.0)   # Zahn 0 markiert
        tooth = _tm.creation.box(extents=[6.0, (rt - r_in) * 1.1, h])
        tooth.apply_translation([0.0, (r_in + rt) / 2.0, 0.0])
        tooth.apply_transform(_tm.transformations.rotation_matrix(th, [0, 0, 1]))
        parts.append(tooth)
    g = _tm.util.concatenate(parts)
    # Symmetrieachse soll body-Y sein (wie im Projekt): drehe Z->Y.
    g.apply_transform(_tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return g


@requires_trimesh
def test_contour_yaw_lock_picks_correct_of_N():
    """M4 wählt unter den 7 Zahn-Yaws den, der zur WAHREN Maske passt — getestet an
    einem Zahnrad mit EINEM markierten (längeren) Zahn, das die C_7-Silhouette
    bricht (nur dann ist der Silhouetten-Lock überhaupt diskriminativ — beim echten,
    perfekt C_7-symmetrischen Zahnrad ist die Top-Down-Silhouette invariant, s.
    REFINE_T041-Befund: M4 hat dort 0 AR-Wirkung)."""
    n = 7
    gear = _gear_mesh(n_teeth=n, marked=True)
    K = np.array([[1006.0, 0, 640.0], [0, 1006.0, 360.0], [0, 0, 1.0]])
    R_w2c = np.diag([1.0, -1.0, -1.0])                # ~top-down
    t_w2c = np.array([0.0, 0.0, 300.0])              # mm
    table_origin = np.zeros(3)
    # WAHRE flache Welt-Rotation: gear flach (body-Y -> world-Z) + 0° Yaw.
    R_true = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # body-Y -> world+Z
    t_world = np.array([0.0, 0.0, 0.10])             # 10cm vor der Kamera (mm-Mesh)
    # Zielmaske = (gefüllte) Silhouette der WAHREN Pose im Bild.
    R_m2c_true = R_w2c @ R_true
    t_m2c = R_w2c @ ((t_world + table_origin) * 1000.0) + t_w2c
    pts, _ = _tm.sample.sample_surface(gear, 8000)
    mask = A._camera_silhouette(np.asarray(pts, float), R_m2c_true, t_m2c, K,
                                (720, 1280))
    assert mask.sum() > 2000                          # Maske sinnvoll gefüllt
    # Eingabe-Pose: um 3 Zähne verdreht (falscher Yaw, den GDRNPP liefern könnte).
    R_off = R_true @ A.axis_angle_matrix([0, 1, 0], 3 * 2 * np.pi / n)
    R_best, info = A.contour_yaw_lock(
        R_off, mask, K, t_world, table_origin, R_w2c, t_w2c,
        mesh=gear, n_fold=n, sym_axis=(0, 1, 0))
    assert info["applied"] is True
    # der gewählte Yaw bringt R_off zurück auf ~R_true (Restwinkel um Y klein):
    rel = R_true.T @ R_best
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert ang < (360.0 / n) / 2.0, f"yaw-lock residual {ang:.1f}deg"
    # die gewählte Lage muss die BESTE der N sein (Diskriminierung), IoU>0 reicht
    # (Punkt-Scatter-Silhouette → absolute IoU moderat, aber klar maximal bei k*).
    assert info["iou"] == max(info["ious"]) and info["iou"] > 0.2


@requires_trimesh
def test_contour_yaw_lock_idempotent():
    """Bereits korrekt ausgerichtet -> M4 wählt k=0 (markiertes Zahnrad)."""
    n = 7
    gear = _gear_mesh(n_teeth=n, marked=True)
    K = np.array([[1006.0, 0, 640.0], [0, 1006.0, 360.0], [0, 0, 1.0]])
    R_w2c = np.diag([1.0, -1.0, -1.0]); t_w2c = np.array([0.0, 0.0, 300.0])
    R_true = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    t_world = np.array([0.0, 0.0, 0.10])
    R_m2c_true = R_w2c @ R_true
    t_m2c = R_w2c @ (t_world * 1000.0) + t_w2c
    pts, _ = _tm.sample.sample_surface(gear, 8000)
    mask = A._camera_silhouette(np.asarray(pts, float), R_m2c_true, t_m2c, K,
                                (720, 1280))
    R_best, info = A.contour_yaw_lock(R_true, mask, K, t_world, np.zeros(3),
                                      R_w2c, t_w2c, mesh=gear, n_fold=n)
    rel = R_true.T @ R_best
    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert ang < (360.0 / n) / 2.0


def test_contour_yaw_lock_nfold_none_noop():
    """n_fold None/<2 -> No-Op (kontinuierliche/asymmetrische Teile unangetastet)."""
    R = A.axis_angle_matrix([0, 0, 1], np.radians(33.0))
    R_out, info = A.contour_yaw_lock(
        R, np.ones((10, 10), bool), np.eye(3), [0, 0, 0.3], [0, 0, 0],
        np.eye(3), [0, 0, 0], mesh_verts_mm=np.zeros((4, 3)), n_fold=None)
    assert info["applied"] is False
    assert np.allclose(R_out, R, atol=1e-12)


def test_contour_yaw_lock_no_mask_noop():
    """Ohne Maske kein Yaw-Lock (kein Crash)."""
    R = np.eye(3)
    R_out, info = A.contour_yaw_lock(
        R, None, np.eye(3), [0, 0, 0.3], [0, 0, 0], np.eye(3), [0, 0, 0],
        mesh_verts_mm=np.zeros((4, 3)), n_fold=7)
    assert info["applied"] is False
    assert np.allclose(R_out, R, atol=1e-12)


# ── Masken-/IoU-Hilfen ────────────────────────────────────────────────────────
def test_mask_iou_basic():
    a = np.zeros((10, 10), bool); a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool); b[4:8, 4:8] = True
    assert abs(A._mask_iou(a, a) - 1.0) < 1e-12
    assert A._mask_iou(a, np.zeros((10, 10), bool)) == 0.0
    assert 0.0 < A._mask_iou(a, b) < 1.0
    assert A._mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0


# ── Self-Runner (ohne pytest) ─────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for fn in fns:
        # Tests, die trimesh brauchen, überspringen wenn es fehlt (statt zu crashen).
        if _tm is None and getattr(fn, "__wrapped__", None) is not None:
            skipped += 1
            print(f"  SKIP {fn.__name__} (trimesh fehlt)")
            continue
        try:
            fn()
        except pytest.skip.Exception:                 # @requires_trimesh
            skipped += 1
            print(f"  SKIP {fn.__name__}")
            continue
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed} passed, {skipped} skipped / {len(fns)} adapter tests")
