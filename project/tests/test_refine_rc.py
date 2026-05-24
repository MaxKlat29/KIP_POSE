#!/usr/bin/env python3
"""Unit-Tests für den M2 Render-and-Compare-Refiner (T-058 / S-048).

Zwei testbare NON-GPU-Teile (der MegaPose-GPU-Pfad ist finish-time):
  1. HYPOTHESEN-GENERATOR — erzeugt korrekte Flip/Yaw/Tilt/Ruhelagen-Kandidaten,
     dedupliziert, R0 bleibt index 0.
  2. CPU-KANTEN/SILHOUETTEN-SCORER — wählt bei einem KÜNSTLICH geflippten Anker
     wieder die richtige (ungeflippte) Hypothese (dry-run, synthetische Maske).

Lauf:  python3 -m pytest project/tests/test_refine_rc.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import bop_adapter as A  # noqa: E402
import refine_rc as RC  # noqa: E402


# ── Hilfen: ein einfaches asymmetrisches „Anker"-Mesh (Kopf + dünner Schaft) ──
def anker_like_verts_mm(n=400, seed=0):
    """Synthetischer Anker: fetter Kopf (r=12mm) an einem Y-Ende, dünner Schaft
    (r=4mm) am anderen. Top-down klar asymmetrisch — genau die Eigenschaft, die
    der Flip-Diskriminierung zugrunde liegt (M3-Befund: Self-IoU 0.58/0.64)."""
    rng = np.random.default_rng(seed)
    # Kopf-Kugel bei y=+20mm
    head = rng.standard_normal((n // 2, 3))
    head = head / np.linalg.norm(head, axis=1, keepdims=True) * 12.0
    head[:, 1] += 20.0
    # Schaft-Zylinder von y=-30 bis y=+10, r=4mm
    m = n - n // 2
    ys = rng.uniform(-30.0, 10.0, m)
    ang = rng.uniform(0, 2 * np.pi, m)
    shaft = np.stack([4.0 * np.cos(ang), ys, 4.0 * np.sin(ang)], axis=1)
    return np.vstack([head, shaft])


def gear_like_verts_mm(n=600, n_teeth=7, seed=1):
    """Synthetisches C_7-Zahnrad: flache Scheibe r=25mm, 7 Zähne, Dicke 8mm."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    # Zahn-moduliertes Radius
    r = 25.0 + 4.0 * (np.cos(n_teeth * ang) > 0.5)
    z = rng.uniform(-4.0, 4.0, n)
    return np.stack([r * np.cos(ang), z, r * np.sin(ang)], axis=1)


def topdown_cam():
    """Top-Down-Kamera (Welt Z-up, Blick nach -Z), K + Extrinsics."""
    R_w2c = A.axis_angle_matrix([1.0, 0.0, 0.0], np.pi)      # 180° um X
    cam_origin = np.array([0.0, 0.0, 0.5])
    t_w2c = -(R_w2c @ cam_origin) * 1000.0                   # mm
    W = H = 256
    f = 24.0 / 20.955 * max(W, H)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R_w2c, t_w2c, (H, W)


# ══════════════════════════════════════════════════════════════════════════════
# 1) HYPOTHESEN-GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def test_hyps_coarse_is_index0():
    R0 = A.axis_angle_matrix([0, 0, 1], np.radians(33.0))
    hyps, tags = RC.generate_hypotheses(R0)
    assert tags[0] == "coarse"
    assert np.allclose(hyps[0], R0, atol=1e-9)


def test_hyps_contains_180_flips():
    R0 = np.eye(3)
    hyps, tags = RC.generate_hypotheses(R0, flip_axes=("x", "y", "z"),
                                        tilt_degs=(), n_fold=None)
    # Ein 180°-Flip um X muss als Kandidat existieren (geodätisch ~180° von R0).
    angs = [RC._rot_geodesic_deg(R0, h) for h in hyps]
    assert any(abs(a - 180.0) < 5.0 for a in angs), f"kein 180°-Flip: {angs}"
    assert any(t.startswith("flip180_") for t in tags)


def test_hyps_cn_yaw_variants():
    """C_7: erzeugt (bis zu) 6 Yaw-Varianten um die Sym-Achse (Y)."""
    R0 = np.eye(3)
    hyps, tags = RC.generate_hypotheses(R0, n_fold=7, sym_axis=(0, 1, 0),
                                        flip_axes=(), tilt_degs=())
    yaw_tags = [t for t in tags if t.startswith("yaw_")]
    assert len(yaw_tags) == 6, f"erwartet 6 Yaw-Varianten, got {yaw_tags}"
    # jede Yaw-Variante ist eine reine Rotation um Y -> Body-Y-Achse unverändert.
    y_axis = np.array([0, 1.0, 0])
    for h in hyps:
        assert np.allclose(h @ y_axis, R0 @ y_axis, atol=1e-9)


def test_hyps_dedup():
    """Doppelte/zu-nahe Kandidaten werden zusammengelegt."""
    R0 = np.eye(3)
    # n_fold=1 -> keine Yaws; nur Flips. Dann eng dedup -> weniger als naiv.
    hyps, tags = RC.generate_hypotheses(R0, flip_axes=("y",), n_fold=None,
                                        tilt_degs=(), dedup_tol_deg=3.0)
    # flip180_y um Y bei R0=I ist 180° -> bleibt. coarse + 1 flip = 2.
    assert len(hyps) == 2
    # geodätische Paar-Abstände >= tol
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            assert RC._rot_geodesic_deg(hyps[i], hyps[j]) >= 3.0 - 1e-6


def test_hyps_rest_poses_from_stable_downs():
    """Ruhelagen-Hypothesen aus stable_downs erzeugen rest_<j>-Tags."""
    R0 = np.eye(3)
    downs = np.array([[0, 0, -1.0], [0, -1.0, 0]])         # zwei Auflage-Achsen
    hyps, tags = RC.generate_hypotheses(R0, stable_downs=downs, flip_axes=(),
                                        tilt_degs=(), n_fold=None,
                                        dedup_tol_deg=1.0)
    assert any(t.startswith("rest_") for t in tags)


def test_hyps_respects_max():
    R0 = np.eye(3)
    hyps, _ = RC.generate_hypotheses(R0, n_fold=7, max_hypotheses=5)
    assert len(hyps) <= 5
    assert np.allclose(hyps[0], R0)                         # coarse trotzdem dabei


# ══════════════════════════════════════════════════════════════════════════════
# 2) CPU-KANTEN/SILHOUETTEN-SCORER — Anker-Flip-Korrektur (dry-run)
# ══════════════════════════════════════════════════════════════════════════════
def _world_lay_flat(yaw_deg=0.0):
    """Eine flache Welt-Rotation des Ankers (Längsachse Y in der Tischebene),
    plus In-Plane-Yaw um Welt-Z."""
    # Body-Y (Längsachse) auf Welt-X legen (liegt flach), dann Yaw um Welt-Z.
    R_lay = A._min_rot_align(np.array([0, 1.0, 0]), np.array([1.0, 0, 0]))
    R_yaw = A.axis_angle_matrix([0, 0, 1.0], np.radians(yaw_deg))
    return R_yaw @ R_lay


def test_cpu_scorer_corrects_anker_flip():
    """Der Kern-Test: gegebene WAHRE (ungeflippte) Anker-Pose -> erzeuge die
    Detektor-Maske + Bildkanten DARAUS (Render der wahren Pose). Starte den
    Refiner von der GEFLIPPTEN Pose. Der CPU-Scorer MUSS die ungeflippte
    Hypothese (180°-Flip-Kandidat) wieder als beste wählen.
    """
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3)
    t_world = np.array([0.0, 0.0, 0.0])                     # zentriert über Kamera-Achse

    R_true = _world_lay_flat(yaw_deg=20.0)                  # WAHRE Pose
    # 180°-Flip um Body-X (= End-über-End-Flip der Längsachse) als "falsche" Coarse.
    R_flip = R_true @ A.axis_angle_matrix([1.0, 0, 0], np.pi)

    # Ground-Truth-Maske + Bildkanten aus der WAHREN Pose rendern (synthetisch).
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    # "Bild" = gefüllte wahre Silhouette als Graustufe -> Sobel-Kanten daraus.
    img = np.zeros((hw[0], hw[1], 3), np.uint8)
    img[sil_true] = 200
    image_edges = RC.image_edges(img, thresh_pct=80.0)

    # min_margin=0.0: testet die SCORER-MECHANIK (wählt die richtige Hypothese,
    # wenn die Silhouette sie trennt). Das synthetische Anker-Mesh ist top-down
    # STARK asymmetrisch (fetter Kopf vs. dünner Schaft) → die Silhouette TRENNT
    # den Flip hier. Auf dem ECHTEN Rod ist sie es NICHT (M3: Self-IoU 0.58/0.64,
    # T-058-Messung: kein AR-Gewinn) — daher der konservative Default-Gate 0.15.
    R_ref, info = RC.refine_detection(
        R_flip, verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw,
        target_mask=sil_true, image_edge_mask=image_edges,
        n_fold=None, tilt_degs=(), scorer="cpu_edge", min_margin=0.0)

    # Die verfeinerte Pose muss DEUTLICH näher an der wahren als an der geflippten
    # Coarse sein (Flip korrigiert).
    err_ref = RC._rot_geodesic_deg(R_ref, R_true)
    err_flip = RC._rot_geodesic_deg(R_flip, R_true)
    assert err_flip > 150.0, f"Setup: Flip sollte ~180° daneben sein, ist {err_flip}"
    assert err_ref < 30.0, (
        f"CPU-Scorer hat den Anker-Flip NICHT korrigiert: ref-Fehler {err_ref:.1f}°, "
        f"flip-Fehler {err_flip:.1f}° (best_tag={info.get('best_tag')})")


def test_cpu_scorer_keeps_good_coarse():
    """Ist die Coarse bereits korrekt, darf der Refiner sie NICHT verschlechtern
    (Gate gegen die Coarse, MegaPose-Design)."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3)
    t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=10.0)
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil_true] = 200
    image_edges = RC.image_edges(img)
    R_ref, info = RC.refine_detection(
        R_true, verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw,
        target_mask=sil_true, image_edge_mask=image_edges,
        n_fold=None, tilt_degs=(), scorer="cpu_edge")
    assert RC._rot_geodesic_deg(R_ref, R_true) < 15.0


def test_cpu_score_iou_perfect_for_matching_pose():
    """IoU-Term: dieselbe Pose -> IoU ~1.0."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R = _world_lay_flat(0.0)
    sil = RC.render_silhouette(verts, R, t_world, table_origin, R_w2c, t_w2c, K, hw)
    scores, detail = RC.cpu_edge_score(
        R[None], verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil,
        w_iou=1.0, w_chamfer=0.0)
    assert detail[0]["iou"] > 0.95


def test_select_best_gate_margin():
    """select_best_hypothesis: ohne Margin-Überschuss bleibt Coarse."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.1)])
    # gleiche Scores -> Coarse behalten
    idx, info = RC.select_best_hypothesis(hyps, np.array([0.5, 0.5]), min_margin=0.0)
    assert idx == 0 and info["switched"] is False
    # klar bessere Hyp -> wechseln
    idx2, info2 = RC.select_best_hypothesis(hyps, np.array([0.5, 0.9]), min_margin=0.0)
    assert idx2 == 1 and info2["switched"] is True


def test_megapose_score_unavailable_falls_back():
    """megapose-Scorer ohne Torch/MegaPose -> refine_detection fällt auf cpu_edge."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R = _world_lay_flat(0.0)
    sil = RC.render_silhouette(verts, R, t_world, table_origin, R_w2c, t_w2c, K, hw)
    R_ref, info = RC.refine_detection(
        R, verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil,
        scorer="megapose", n_fold=None, tilt_degs=())
    # Fallback dokumentiert + Ergebnis vom CPU-Scorer.
    assert info["scorer"] == "cpu_edge"
    assert "megapose_fallback" in info


def test_gear_silhouette_cn_invariant_honest():
    """EHRLICH (T-041-Lehre): die Top-Down-Silhouette eines C_7-Zahnrads ist
    rotations-invariant -> die 7 Yaw-Hypothesen haben quasi-identische IoU. Der
    CPU-Scorer kann (und muss) den C_7-Yaw NICHT auflösen. Dieser Test
    DOKUMENTIERT das als bewusste Einschränkung, kein Bug."""
    verts = gear_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_flat = A._min_rot_align(np.array([0, 1.0, 0]), np.array([0, 0, -1.0]))  # flach
    hyps, tags = RC.generate_hypotheses(R_flat, n_fold=7, sym_axis=(0, 1, 0),
                                        flip_axes=(), tilt_degs=())
    sil0 = RC.render_silhouette(verts, hyps[0], t_world, table_origin,
                                R_w2c, t_w2c, K, hw)
    scores, detail = RC.cpu_edge_score(
        hyps, verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil0,
        w_iou=1.0, w_chamfer=0.0)
    ious = np.array([d["iou"] for d in detail])
    # alle 7 Yaw-IoUs liegen eng beieinander (C_7-invariant) -> nicht trennbar.
    assert ious.max() - ious.min() < 0.12, (
        f"C_7-Yaws sollten silhouetten-invariant sein, Spannweite {ious.ptp():.3f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
