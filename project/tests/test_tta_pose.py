#!/usr/bin/env python3
"""Unit-Tests für den TTA-Wrapper (T-058 / S-049) — kein GPU.

Die load-bearing Teile ohne Checkpoint:
  1. INVERSE-TRANSFORM-ALGEBRA — ein synthetischer, EXAKT in-plane-äquivarianter
     "Modell"-Mock liefert nach Rück-Transform für JEDE View dieselbe wahre
     Rotation (Round-Trip). Das ist die zentrale Korrektheit: dreht man den Crop
     um die optische Achse, dreht das Modell seine Ausgabe mit, und das Undo macht
     es exakt rückgängig.
  2. AGGREGATION auf SO(3) — chordaler Mittelwert, geodätischer Medoid, score-pick;
     bimodal-Robustheit (Medoid kollabiert zwei Becken NICHT zu ihrer Mitte).
  3. INTEGRATION — estimate_poses mit tta=True am MOCK-Backend bleibt schema-/
     contract-konform und identisch in Form zur Nicht-TTA-Kette.

Lauf:  python3 -m pytest project/tests/test_tta_pose.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tta_pose as T  # noqa: E402


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def random_rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def is_SO3(R, tol=1e-9):
    R = np.asarray(R, float)
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < 1e-6)


# ── 1. Inverse-Transform-Algebra: Round-Trip eines äquivarianten Mock-Modells ──
def make_equivariant_model(R_true):
    """Ein synthetisches Modell, das EXAKT in-plane-äquivariant ist: es schaut auf
    den Crop, misst die angewandte 90°-Rotation und gibt R_true MIT der Crop-
    Rotation links-multipliziert zurück — genau das Verhalten, das der TTA-Undo
    rückgängig machen muss.

    Wir kodieren die angewandte View in den Crop selbst: der Crop trägt ein
    'Label' (Pixel-Marke), aus dem das Modell die View-Rotation rekonstruiert.
    np.rot90(k) auf das Label -> das Modell liest k -> liefert Rz(k·90°)·R_true.
    """
    base = np.zeros((8, 8, 3), np.uint8)
    base[0, 0] = 255          # asymmetrische Ecken-Marke -> Drehung detektierbar

    def model(crop, K, bbox, obj_id, cfg=None):
        # rekonstruiere k aus der Position der 255-Marke (rotierte Ecke)
        ys, xs = np.where(crop[..., 0] == 255)
        y, x = int(ys[0]), int(xs[0])
        h, w = crop.shape[:2]
        corner = (y, x)
        # Ecken-Mapping für np.rot90 (CCW): (0,0)->(h-1,0)->(h-1,w-1)->(0,w-1)
        order = [(0, 0), (h - 1, 0), (h - 1, w - 1), (0, w - 1)]
        k = order.index(corner)
        # die Bild-Drehung um die opt. Achse ist +k·90° -> Modell-Ausgabe mitgedreht
        R_aug = Rz(k * np.pi / 2.0) @ R_true
        t = np.array([0.0, 0.0, 500.0])
        return R_aug, t

    return model, base


@pytest.mark.parametrize("seed", [0, 1, 7, 13])
def test_inverse_transform_roundtrip(seed):
    """Jede View, rück-transformiert, ergibt EXAKT R_true (äquivariantes Modell)."""
    R_true = random_rot(seed)
    model, crop = make_equivariant_model(R_true)
    for k, (name, aug_fn, undo) in enumerate(T.rot90_views(4)):
        aug_crop = aug_fn(crop)
        R_aug, _ = model(aug_crop, None, None, 6)
        R_corr = undo @ R_aug
        assert is_SO3(R_corr)
        assert np.allclose(R_corr, R_true, atol=1e-9), (
            f"view {name}: undo failed, ang="
            f"{np.degrees(T.geodesic_angle(R_corr, R_true)):.3f}°")


def test_tta_wrapper_recovers_true_rotation():
    """Der volle Wrapper liefert für ein äquivariantes Modell R_true (alle Views
    rück-transformiert deckungsgleich -> jede Aggregation trifft R_true)."""
    R_true = random_rot(42)
    model, crop = make_equivariant_model(R_true)
    for agg in ("medoid", "chordal"):
        R_agg, t, info = T.tta_call_gdrnpp(
            lambda c, K, b, o, cfg=None: model(c, K, b, o),
            crop, None, None, 6, n_rot=4, agg=agg)
        assert is_SO3(R_agg)
        ang = np.degrees(T.geodesic_angle(R_agg, R_true))
        assert ang < 1e-6, f"agg={agg}: {ang:.4f}° off"
        assert np.allclose(t, [0, 0, 500.0])
        assert info["n"] == 4


# ── 2. H-Flip: Konjugations-Undo ergibt eine gültige SO(3)-Matrix ──────────────
def test_hflip_undo_is_proper_rotation():
    name, aug, undo = T.hflip_view()
    crop = np.zeros((6, 6, 3), np.uint8)
    crop[0, 0] = 255
    flipped = aug(crop)
    assert flipped[0, -1, 0] == 255          # Marke nach rechts gespiegelt
    R = random_rot(3)
    R_corr = undo(R)
    assert is_SO3(R_corr), "H-Flip-Undo muss eine gültige Rotation liefern"
    # zweimaliges Anwenden ist Identität (Spiegelung ist selbst-invers)
    assert np.allclose(undo(undo(R)), R, atol=1e-12)


# ── 3. Aggregation ────────────────────────────────────────────────────────────
def test_chordal_mean_is_so3_and_unbiased_unimodal():
    base = random_rot(5)
    # kleine symmetrische Störung um base -> Mittel ~ base
    Rs = [Rx(d) @ base for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    M = T.chordal_mean(Rs)
    assert is_SO3(M)
    assert np.degrees(T.geodesic_angle(M, base)) < 0.5


def test_medoid_does_not_collapse_two_basins():
    """Bimodal: 3 Posen im Becken A, 2 im (um 120°-geflippten) Becken B. Der Medoid
    MUSS eine echte Pose aus dem dichteren Becken A wählen — NICHT die (physikalisch
    falsche) Mitte zwischen A und B, die ein Mittelwert liefern würde.

    (120° statt 180°: zwei EXAKT antipodale Becken mitteln sich im chordalen
    SVD-Mittel über die det-Korrektur teilweise wieder heraus — ein degeneriertes
    Sonderbeispiel. Ein generisches Becken-B zeigt den Verschmier-Effekt sauber.)"""
    base = random_rot(9)
    flip = Rz(2 * np.pi / 3.0) @ base           # 120°-Becken B (nicht antipodal)
    Rs = [base, Rx(0.01) @ base, Rx(-0.01) @ base, flip, Rx(0.01) @ flip]
    i = T.geodesic_medoid(Rs)
    assert i < 3, "Medoid muss aus dem dichteren Becken A (Index 0..2) kommen"
    # Gegenprobe: der chordale Mittelwert liegt messbar zwischen den Becken
    M = T.chordal_mean(Rs)
    ang_to_base = np.degrees(T.geodesic_angle(M, base))
    assert ang_to_base > 5.0, "Mittelwert sollte (anders als Medoid) verschmieren"


def test_aggregate_score_mode_picks_argmax():
    Rs = [random_rot(s) for s in (1, 2, 3)]
    R, info = T.aggregate(Rs, mode="score", scores=[0.1, 0.9, 0.2])
    assert np.allclose(R, Rs[1])
    assert info["picked"] == 1


def test_aggregate_score_without_scores_falls_back():
    Rs = [random_rot(s) for s in (1, 2, 3)]
    R, info = T.aggregate(Rs, mode="score", scores=None)
    assert "fallback" in info and is_SO3(R)


def test_single_view_is_passthrough():
    R = random_rot(11)
    out, info = T.aggregate([R], mode="medoid")
    assert np.allclose(out, R) and info["mode"] == "single"


def test_n_rot_validation():
    with pytest.raises(ValueError):
        T.rot90_views(3)
    assert len(list(T.rot90_views(1))) == 1
    assert len(list(T.rot90_views(2))) == 2
    assert len(list(T.rot90_views(4))) == 4


# ── 4. Integration mit estimate_poses am MOCK-Backend ─────────────────────────
def test_estimate_poses_with_tta_mock_is_contract_shaped():
    import e2e_infer as E  # noqa: E402
    rgb = np.zeros((480, 640, 3), np.uint8)
    dets = [{"part": "Zahnrad", "bbox_2d": [100, 100, 200, 200],
             "instance_id": 0, "det_conf": 0.9},
            {"part": "Anker_Kurz", "bbox_2d": [300, 200, 360, 280],
             "instance_id": 1, "det_conf": 0.8}]
    cfg = E.GdrnppConfig(mock=True)
    warns = []
    aligned = E.estimate_poses(rgb, dets, cfg=cfg, warn=warns.append,
                               tta=True, tta_n_rot=4, tta_agg="medoid")
    assert len(aligned) == 2
    for a in aligned:
        R = np.asarray(a["R_world"], float).reshape(3, 3)
        assert is_SO3(R, tol=1e-6)
        assert len(a["t_world"]) == 3
        assert "face_name" in a and "upright" in a
    assert any("TTA" in w for w in warns), "TTA-Log-Zeile erwartet"


def test_tta_disabled_matches_plain_mock():
    """tta=False reproduziert die nicht-augmentierte MOCK-Kette bit-nah."""
    import e2e_infer as E  # noqa: E402
    rgb = np.zeros((480, 640, 3), np.uint8)
    dets = [{"part": "Zahnrad", "bbox_2d": [100, 100, 200, 200],
             "instance_id": 0, "det_conf": 0.9}]
    cfg = E.GdrnppConfig(mock=True)
    a_plain = E.estimate_poses(rgb, dets, cfg=cfg, warn=lambda *_: None, tta=False)
    a_tta_off = E.estimate_poses(rgb, dets, cfg=cfg, warn=lambda *_: None,
                                 tta=True, tta_n_rot=1)  # n_rot=1 -> effektiv aus
    assert np.allclose(a_plain[0]["R_world"], a_tta_off[0]["R_world"])
