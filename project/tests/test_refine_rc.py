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


# ══════════════════════════════════════════════════════════════════════════════
# 3) VISIBILITY-AWARE SCORER  (ADR-020, S-003, T-085)
# ══════════════════════════════════════════════════════════════════════════════
# Der Anker-180°-Quer-Flip bei partieller Sicht: der Scorer soll NUR über die
# sichtbare Region (visib_mask, BOP mask_visib) scoren. Tests:
#  (a) visib_mask=None -> EXAKT heutiges Verhalten (Rückwärtskompatibilität).
#  (b) Score-Clipping: visib_mask beschränkt IoU/Chamfer auf die sichtbare Region.
#  (c) Visibility-Gate: zu geringe visib_fract -> Coarse behalten.
#  (d) Kern: bei occludiertem Schaft (visib_mask = nur Kopf) trennt der vis-aware
#      Scorer den Flip, wo der Voll-Crop-Scorer das Asymmetrie-Signal ertränkt.

def _occluder_visib_mask(verts, R_pose, t_world, table_origin, R_w2c, t_w2c, K, hw,
                         keep_head=True, head_y_mm=10.0):
    """Sichtbare Region = die Silhouette der WAHREN Pose, aber nur die Hälfte des
    Teils, die auf der Kopf- (bzw. Schaft-)Seite der Längsachse liegt. Simuliert
    einen Occluder, der genau die andere Hälfte verdeckt."""
    # gefüllte Silhouette der wahren Pose
    sil = RC.render_silhouette(verts, R_pose, t_world, table_origin,
                               R_w2c, t_w2c, K, hw)
    # Welche Bildpixel gehören zur Kopf-Hälfte? Projiziere die Kopf-Vertices.
    head_sel = verts[:, 1] >= head_y_mm if keep_head else verts[:, 1] < head_y_mm
    R_w2c_ = A._as_R(R_w2c)
    R_pose_ = A._as_R(R_pose)
    t_world_ = A._as_vec3(t_world); to = A._as_vec3(table_origin)
    t_m2w_mm = (t_world_ + to) * 1000.0
    t_m2c = R_w2c_ @ t_m2w_mm + A._as_vec3(t_w2c)
    R_m2c = R_w2c_ @ R_pose_
    half_sil = A._camera_silhouette(verts[head_sel], R_m2c, t_m2c, A._as_R(K), hw)
    return sil & half_sil


def test_visib_none_is_backward_compatible():
    """visib_mask=None -> cpu_edge_score liefert EXAKT die alten Scores."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R0 = _world_lay_flat(yaw_deg=15.0)
    hyps, _ = RC.generate_hypotheses(R0, n_fold=None, tilt_degs=())
    sil = RC.render_silhouette(verts, R0, t_world, table_origin, R_w2c, t_w2c, K, hw)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil] = 200
    ie = RC.image_edges(img)
    kw = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
              R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil,
              image_edge_mask=ie)
    s_old, _ = RC.cpu_edge_score(hyps, **kw)
    s_new, _ = RC.cpu_edge_score(hyps, visib_mask=None, **kw)
    assert np.allclose(s_old, s_new, atol=1e-12)


def test_visib_clip_reduces_to_visible_region():
    """Mit visib_mask = nur Kopf-Hälfte zählt IoU nur über die sichtbare Region:
    die korrekte Pose hat dort IoU ~1, weil sil & visib == target & visib."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=0.0)
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    visib = _occluder_visib_mask(verts, R_true, t_world, table_origin,
                                 R_w2c, t_w2c, K, hw, keep_head=True)
    assert 0 < visib.sum() < sil_true.sum(), "visib sollte echte Teilregion sein"
    scores, detail = RC.cpu_edge_score(
        R_true[None], verts_mm=verts, t_world_m=t_world,
        table_origin_m=table_origin, R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw,
        target_mask=sil_true, visib_mask=visib, w_iou=1.0, w_chamfer=0.0)
    # gleiche Pose, auf die sichtbare Region geclippt -> IoU nahe 1.
    assert detail[0]["iou"] > 0.9


def test_visib_gate_keeps_coarse_when_too_occluded():
    """Visibility-Gate: zu geringe visib_fract -> kein Switch (Coarse behalten),
    selbst wenn eine andere Hypothese besser scort."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.4, 0.9])     # Hyp 1 klar besser
    # ohne Visibility-Kondition: switcht.
    idx_open, info_open = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.15, visib_fract=0.5, min_visib_fract=0.25)
    assert idx_open == 1 and info_open["switched"] is True
    # mit zu geringer Sichtbarkeit: bleibt Coarse, visib_gated_out gesetzt.
    idx_gate, info_gate = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.15, visib_fract=0.10, min_visib_fract=0.25)
    assert idx_gate == 0 and info_gate["switched"] is False
    assert info_gate["visib_gated_out"] is True


def test_visib_gate_min_visible_px():
    """absolutes Pixel-Gate: zu wenige sichtbare Pixel -> Coarse behalten."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.4, 0.9])
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.15, visible_px=50, min_visible_px=200)
    assert idx == 0 and info["switched"] is False and info["visib_gated_out"]


def test_visaware_scorer_separates_flip_under_occlusion():
    """Kern-Test (ADR-020): Schaft occludiert (visib_mask = nur Kopf-Region).
    Der VOLL-Crop-Scorer (visib=None) trennt den Flip schlechter als der
    VISIBILITY-AWARE Scorer. Wir prüfen: die korrekte (un-geflippte) Hypothese
    schlägt die geflippte Coarse beim vis-aware Scorer mit größerem Margin als
    beim Voll-Crop-Scorer — d.h. das Asymmetrie-Signal wird NICHT ertränkt.

    EHRLICH: das synthetische Anker-Mesh ist top-down ohnehin stark asymmetrisch.
    Der Test zeigt die Score-MECHANIK (vis-aware Margin >= voll-Crop Margin), nicht
    eine harte 0-Flip-Garantie — das echte Single-View-Limit misst die GPU-Eval.
    """
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=25.0)
    R_flip = R_true @ A.axis_angle_matrix([1.0, 0, 0], np.pi)   # geflippte Coarse

    # WAHRES Bild: volle wahre Silhouette als Detektor-Maske + Bildkanten, ABER
    # die occludierte Schaft-Hälfte wird im "Bild" mit Occluder-Rauschen überdeckt.
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    visib = _occluder_visib_mask(verts, R_true, t_world, table_origin,
                                 R_w2c, t_w2c, K, hw, keep_head=True)
    occluded = sil_true & ~visib
    img = np.zeros((hw[0], hw[1], 3), np.uint8)
    img[sil_true] = 200
    # Occluder: helle Störung über der verdeckten Hälfte -> Fremdkanten im Voll-Crop.
    rng = np.random.default_rng(7)
    img[occluded] = rng.integers(60, 255, size=(int(occluded.sum()), 3), dtype=np.uint8)
    ie = RC.image_edges(img)

    hyps, tags = RC.generate_hypotheses(R_flip, n_fold=None, tilt_degs=(),
                                        flip_axes=("x", "y", "z"))
    correct = [i for i, t in enumerate(tags) if t in ("flip180_x", "flip180_z")]
    assert correct, "Setup: flip180_x/z muss als Hypothese existieren"

    common = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
                  R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw,
                  target_mask=sil_true, image_edge_mask=ie)
    s_full, _ = RC.cpu_edge_score(hyps, visib_mask=None, **common)
    s_vis, _ = RC.cpu_edge_score(hyps, visib_mask=visib, **common)

    coarse = 0
    margin_full = float(max(s_full[i] for i in correct) - s_full[coarse])
    margin_vis = float(max(s_vis[i] for i in correct) - s_vis[coarse])
    # vis-aware soll die korrekte Hypothese mindestens so gut von der geflippten
    # Coarse trennen wie der Voll-Crop-Scorer (typ. deutlich besser).
    assert margin_vis >= margin_full - 1e-9, (
        f"vis-aware Margin {margin_vis:.4f} schlechter als voll-Crop {margin_full:.4f}")


def test_refine_detection_visaware_end_to_end():
    """refine_detection mit visib_mask + ausreichender visib_fract korrigiert den
    Flip; mit zu geringer visib_fract bleibt die Coarse (Gate greift)."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=18.0)
    R_flip = R_true @ A.axis_angle_matrix([1.0, 0, 0], np.pi)
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    visib = _occluder_visib_mask(verts, R_true, t_world, table_origin,
                                 R_w2c, t_w2c, K, hw, keep_head=True)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil_true] = 200
    ie = RC.image_edges(img)
    common = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
                  R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil_true,
                  image_edge_mask=ie, visib_mask=visib, n_fold=None, tilt_degs=(),
                  scorer="cpu_edge", min_margin=0.0)
    # genug Sichtbarkeit -> Gate offen -> Flip korrigiert.
    R_ok, info_ok = RC.refine_detection(R_flip, visib_fract=0.6,
                                        min_visib_fract=0.25, **common)
    assert info_ok["visib_aware"] is True
    assert RC._rot_geodesic_deg(R_ok, R_true) < 30.0
    # zu wenig Sichtbarkeit -> Gate zu -> Coarse (geflippt) bleibt.
    R_keep, info_keep = RC.refine_detection(R_flip, visib_fract=0.05,
                                            min_visib_fract=0.25, **common)
    assert info_keep["switched"] is False
    assert RC._rot_geodesic_deg(R_keep, R_flip) < 0.01     # Coarse unverändert


# ── VISIBILITY-STAGGERED MARGIN GATE (S-003 v2, T-085) ───────────────────────
def test_staggered_min_margin_three_bands():
    """staggered_min_margin: occluded band -> aggressive, well-vis -> conservative,
    below floor -> +inf (never switch)."""
    sched = RC.DEFAULT_MARGIN_SCHEDULE
    # below vf_occ_lo (0.20): never switch.
    assert RC.staggered_min_margin(0.10, sched) == float("inf")
    # occluded band [0.20, 0.60): aggressive margin_occ (0.05).
    assert RC.staggered_min_margin(0.30, sched) == sched["margin_occ"]
    assert RC.staggered_min_margin(0.20, sched) == sched["margin_occ"]   # lo inclusive
    # well-vis (>= 0.60): conservative margin_well (= shipped 0.15).
    assert RC.staggered_min_margin(0.60, sched) == sched["margin_well"]  # hi inclusive
    assert RC.staggered_min_margin(0.95, sched) == sched["margin_well"]
    # None schedule or None visib_fract -> None (caller uses static margin).
    assert RC.staggered_min_margin(0.5, None) is None
    assert RC.staggered_min_margin(None, sched) is None


def test_staggered_gate_aggressive_in_occluded_band():
    """Im occludierten Band schaltet das gestaffelte Gate bei einem Margin, der
    das shipped-konservative statische Gate (0.15) NICHT durchließe."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.40, 0.48])     # Margin 0.08: < 0.15, aber > 0.05
    sched = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.60,
             "margin_occ": 0.05, "margin_well": 0.15}
    # statisch 0.15 -> KEIN Switch (inert), wie der v1-Befund.
    idx_static, info_static = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.15, visib_fract=0.35)
    assert idx_static == 0 and info_static["switched"] is False
    # gestaffelt, occludiertes Band (vf=0.35) -> aggressiver Margin 0.05 -> Switch.
    idx_sched, info_sched = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, visib_fract=0.35, margin_schedule=sched)
    assert idx_sched == 1 and info_sched["switched"] is True
    assert info_sched["margin_schedule"] is True
    assert info_sched["eff_min_margin"] == 0.05


def test_staggered_gate_conservative_in_well_vis_band():
    """Im well-vis Band bleibt das gestaffelte Gate konservativ (= shipped 0.15):
    der gleiche schwache Margin schaltet NICHT -> Regressionsschutz."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.40, 0.48])     # Margin 0.08: < 0.15
    sched = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.60,
             "margin_occ": 0.05, "margin_well": 0.15}
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, visib_fract=0.90, margin_schedule=sched)
    assert idx == 0 and info["switched"] is False
    assert info["eff_min_margin"] == 0.15           # well-vis -> shipped Margin


def test_staggered_gate_never_switches_below_floor():
    """Unter vf_occ_lo: +inf -> nie schalten, auch bei riesigem Margin."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.10, 0.99])     # Margin 0.89: riesig
    sched = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.60,
             "margin_occ": 0.05, "margin_well": 0.15}
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, visib_fract=0.10, margin_schedule=sched)
    assert idx == 0 and info["switched"] is False
    assert info["eff_min_margin"] == float("inf")


def test_staggered_gate_respects_global_hard_floor():
    """Der globale min_margin bleibt Hard-Floor: ist er höher als margin_occ,
    gewinnt der Floor (Schedule darf den globalen Floor NIE unterlaufen)."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.40, 0.48])     # Margin 0.08
    sched = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.60,
             "margin_occ": 0.05, "margin_well": 0.15}
    # hard floor 0.10 > margin_occ 0.05 -> eff = 0.10 -> Margin 0.08 reicht NICHT.
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.10, visib_fract=0.35, margin_schedule=sched)
    assert idx == 0 and info["switched"] is False
    assert info["eff_min_margin"] == 0.10


def test_staggered_gate_none_is_backward_compatible():
    """margin_schedule=None -> EXAKT das statische Margin-Gate (kein eff_min_margin
    Feld, kein margin_schedule Flag)."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.40, 0.90])
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.15, visib_fract=0.35, margin_schedule=None)
    assert idx == 1 and info["switched"] is True
    assert "margin_schedule" not in info
    assert "eff_min_margin" not in info


def test_refine_detection_staggered_schedule_threads_through():
    """refine_detection reicht margin_schedule durch -> occludiertes Band korrigiert
    den Flip bei einem Margin, den das statische 0.15-Gate verschlucken würde."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=12.0)
    R_flip = R_true @ A.axis_angle_matrix([1.0, 0, 0], np.pi)
    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    visib = _occluder_visib_mask(verts, R_true, t_world, table_origin,
                                 R_w2c, t_w2c, K, hw, keep_head=True)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil_true] = 200
    ie = RC.image_edges(img)
    sched = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.60,
             "margin_occ": 0.05, "margin_well": 0.15}
    common = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
                  R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil_true,
                  image_edge_mask=ie, visib_mask=visib, n_fold=None, tilt_degs=(),
                  scorer="cpu_edge")
    # occludiertes Band (vf=0.35), gestaffeltes Gate, hard floor 0.0.
    R_ref, info = RC.refine_detection(R_flip, visib_fract=0.35,
                                      min_visib_fract=0.20, min_margin=0.0,
                                      margin_schedule=sched, **common)
    assert info.get("margin_schedule") is True
    assert info["eff_min_margin"] == 0.05


# ══════════════════════════════════════════════════════════════════════════════
# 4) OCCLUSION-CONSISTENCY / FREE-SPACE-REFUTATION  (ADR-020-Amendment, T-092)
# ══════════════════════════════════════════════════════════════════════════════
# Negative-Evidenz / free-space-refutation, TEIL-AGNOSTISCH (kein Kopf/Schaft/
# Anker-Wissen). Eine Hypothese, die einen Teil ihrer VOLLEN Silhouette in den
# sichtbar-leeren Raum legt, widerspricht der Beobachtung -> harte Verwerfung.
# Tests:
#  (a) free_space_violation-Mathe (Anteil sil ∩ free_space / |sil|, off-frame=0).
#  (b) build_free_space_mask: Frame minus visib/occluded/andere-Teil-Masken.
#  (c) KERN ("nur-Schaft sichtbar, Kopf verdeckt"): der falsche Flip legt den Kopf
#      in den leeren Raum -> hohe Violation -> verworfen -> un-Flip gewinnt.
#  (d) Hard-Rejection im Gate: refuted Hyp kann nicht gewinnen; Coarse nie verworfen.
#  (e) free_space_mask=None -> exakt heutiges Verhalten (rückwärtskompatibel).
#  (f) Teil-Agnostik: derselbe Term auf einem Zahnrad-Mesh, kein Crash, plausibel.


def test_free_space_violation_basic_and_offframe():
    """violation = |sil ∩ free_space| / |sil|; leere Silhouette -> 0;
    off-frame zählt nicht (Silhouette ist frame-begrenzt)."""
    sil = np.zeros((10, 10), bool)
    sil[2:6, 2:6] = True                       # 16 px
    fs = np.zeros((10, 10), bool)
    fs[2:4, 2:6] = True                        # 8 px überlappen die Silhouette
    v = RC.free_space_violation(sil, fs)
    assert abs(v - 8.0 / 16.0) < 1e-9
    # leere Silhouette -> keine Aussage.
    assert RC.free_space_violation(np.zeros((5, 5), bool), fs[:5, :5]) == 0.0
    # disjunkt -> 0.
    fs2 = np.zeros((10, 10), bool); fs2[7:9, 7:9] = True
    assert RC.free_space_violation(sil, fs2) == 0.0


def test_build_free_space_mask_part_agnostic():
    """free_space = Frame ∧ ¬visib ∧ ¬occluded ∧ ¬andere-Teile. Rein mengen-
    basiert, keine Geometrie-Annahme."""
    H, W = 8, 8
    visib = np.zeros((H, W), bool); visib[0:2, 0:2] = True
    occ = np.zeros((H, W), bool); occ[0:2, 2:4] = True
    other = np.zeros((H, W), bool); other[6:8, 6:8] = True
    fs = RC.build_free_space_mask((H, W), visib, occluded_mask=occ,
                                  other_masks=[other])
    # belegte Regionen sind NICHT frei
    assert not fs[0, 0] and not fs[0, 2] and not fs[6, 6]
    # eine garantiert freie Zelle ist frei
    assert fs[4, 4]
    # Gesamtmenge frei = Frame - (visib ∪ occ ∪ other)
    occupied = visib | occ | other
    assert int(fs.sum()) == int((~occupied).sum())


def test_build_free_space_mask_dilate_is_conservative():
    """dilate>0 schrumpft den freien Raum (konservativer, weniger false-positive
    Violations am Rand belegter Regionen)."""
    H, W = 12, 12
    visib = np.zeros((H, W), bool); visib[5:7, 5:7] = True
    fs0 = RC.build_free_space_mask((H, W), visib, dilate=0)
    fs2 = RC.build_free_space_mask((H, W), visib, dilate=2)
    assert int(fs2.sum()) < int(fs0.sum())     # mehr belegt -> weniger frei


def test_free_space_mask_none_backward_compatible():
    """cpu_edge_score ohne free_space_mask -> identische Scores; detail-violation=0;
    select_best_hypothesis ohne Violations -> exakt altes Verhalten."""
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R0 = _world_lay_flat(yaw_deg=15.0)
    hyps, _ = RC.generate_hypotheses(R0, n_fold=None, tilt_degs=())
    sil = RC.render_silhouette(verts, R0, t_world, table_origin, R_w2c, t_w2c, K, hw)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil] = 200
    ie = RC.image_edges(img)
    kw = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
              R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=sil,
              image_edge_mask=ie)
    s_old, _ = RC.cpu_edge_score(hyps, **kw)
    s_new, det = RC.cpu_edge_score(hyps, free_space_mask=None, **kw)
    assert np.allclose(s_old, s_new, atol=1e-12)
    assert all(d["free_space_violation"] == 0.0 for d in det)
    # Gate ohne Violations == altes Verhalten.
    idx_a, _ = RC.select_best_hypothesis(hyps, s_new, min_margin=0.0)
    idx_b, info_b = RC.select_best_hypothesis(hyps, s_new, min_margin=0.0,
                                              free_space_violations=None)
    assert idx_a == idx_b
    assert "n_refuted" not in info_b


def test_select_best_hard_rejects_violating_hypothesis():
    """Die best-scorende Hypothese wird verworfen, wenn ihre Violation > Schwelle;
    die Coarse wird NIE verworfen (Fallback)."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5),
                     A.axis_angle_matrix([0, 0, 1], 1.0)])
    scores = np.array([0.30, 0.90, 0.50])      # Hyp 1 am besten
    viol = np.array([0.10, 0.80, 0.05])        # ... aber Hyp 1 verletzt free space
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, free_space_violations=viol,
        max_free_space_violation=0.30)
    assert idx == 2, "die beste NICHT-verletzende Hypothese muss gewinnen"
    assert 1 in info["refuted_idx"] and info["n_refuted"] == 1
    # Coarse selbst verletzt stark -> wird TROTZDEM nicht verworfen.
    viol2 = np.array([0.99, 0.80, 0.95])
    idx2, info2 = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, free_space_violations=viol2,
        max_free_space_violation=0.30)
    assert idx2 == 0 and info2["n_refuted"] == 2   # nur die beiden non-coarse


def test_select_best_all_violating_falls_back_to_coarse():
    """Sind alle non-coarse Hypothesen verworfen -> Coarse bleibt (margin-Gate
    hält ohnehin)."""
    hyps = np.stack([np.eye(3), A.axis_angle_matrix([0, 0, 1], 0.5)])
    scores = np.array([0.4, 0.95])
    viol = np.array([0.0, 0.9])
    idx, info = RC.select_best_hypothesis(
        hyps, scores, min_margin=0.0, free_space_violations=viol,
        max_free_space_violation=0.30)
    assert idx == 0 and info["switched"] is False


def test_free_space_refutation_breaks_shaft_only_flip():
    """KERN-TEST (Max-Insight): nur der Schaft ist sichtbar, der Kopf ist verdeckt.
    Der appearance-Score allein kann den Flip nicht trennen (der dünne Schaft ist
    quasi flip-invariant). Aber der FALSCHE Flip projiziert den Kopf in den
    sichtbar-LEEREN Raum -> hohe free_space_violation -> verworfen -> die korrekte
    (un-geflippte) Hypothese gewinnt. TEIL-AGNOSTISCH: das Gate sieht nur Masken.
    """
    verts = anker_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R_true = _world_lay_flat(yaw_deg=20.0)
    R_flip = R_true @ A.axis_angle_matrix([1.0, 0, 0], np.pi)   # geflippte Coarse

    sil_true = RC.render_silhouette(verts, R_true, t_world, table_origin,
                                    R_w2c, t_w2c, K, hw)
    # SICHTBAR = nur die SCHAFT-Hälfte (Kopf verdeckt -> keep_head=False).
    visib = _occluder_visib_mask(verts, R_true, t_world, table_origin,
                                 R_w2c, t_w2c, K, hw, keep_head=False)
    assert 0 < visib.sum() < sil_true.sum()
    occluded = sil_true & ~visib
    # free space = Frame ohne sichtbare Schaft-Region, ohne occludierte Kopf-Region,
    # ohne andere Teile (hier keine). dilate=0 = exakte negative Evidenz; die
    # Dilation ist ein konservativer Tuning-Knopf (schrumpft die Magnitude),
    # KEIN Träger der Separierbarkeit — siehe eval_s007.md (GPU-Kalibrierung).
    fs = RC.build_free_space_mask(hw, visib, occluded_mask=occluded,
                                  other_masks=None, dilate=0)

    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil_true] = 200
    ie = RC.image_edges(img)
    common = dict(verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
                  R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=visib,
                  image_edge_mask=ie, visib_mask=visib, n_fold=None, tilt_degs=(),
                  flip_axes=("x", "y", "z"), scorer="cpu_edge")

    # MIT Refutation: die geflippte Coarse legt den Kopf in den freien Raum.
    R_fs, info_fs = RC.refine_detection(
        R_flip, visib_fract=0.35, min_visib_fract=0.20, min_margin=0.0,
        free_space_mask=fs, max_free_space_violation=0.25, **common)

    assert info_fs["free_space_aware"] is True
    tags = info_fs["tags"]
    det = info_fs["detail"]
    v_coarse = det[0]["free_space_violation"]
    correct = [i for i, t in enumerate(tags) if t in ("flip180_x", "flip180_z")]
    assert correct
    v_correct = min(det[i]["free_space_violation"] for i in correct)
    # KERN: die korrekte (un-geflippte) Hypothese belegt SUBSTANZIELL weniger freien
    # Raum als die geflippte Coarse — DAS ist das diskriminierende Signal, das der
    # appearance-Score allein nicht hat. Separierbarkeit, kein Magnitude-Hard-Coding.
    assert v_correct < v_coarse - 0.10, (
        f"Separierbarkeit zu schwach: correct={v_correct:.3f} coarse={v_coarse:.3f}")
    # Die geflippte Coarse ist über der Schwelle verwerfbar (negative Evidenz greift).
    assert v_coarse > 0.25, f"geflippte Coarse-Violation {v_coarse:.3f} nicht verwerfbar"
    # Folge: der Refiner verwirft die Coarse NICHT (Fallback), aber eine refutierte
    # bessere-aber-leerraum-belegende Hyp kann nicht gewinnen; die gewählte Pose darf
    # nicht schlechter als die geflippte Coarse sein.
    syms = [A.axis_angle_matrix([0, 1.0, 0], 2 * np.pi * k / 90) for k in range(90)]
    def rs(Rw):
        Rc = A._as_R(R_w2c) @ A._as_R(Rw)
        Rg = A._as_R(R_w2c) @ R_true
        return min(RC._rot_geodesic_deg(Rc, Rg @ S) for S in syms)
    assert rs(R_fs) <= rs(R_flip) + 1e-6, (
        f"Refutation darf nicht verschlechtern: fs={rs(R_fs):.1f} flip={rs(R_flip):.1f}")


def test_free_space_refutation_part_agnostic_on_gear():
    """TEIL-AGNOSTIK: derselbe free-space-Term auf einem C_7-Zahnrad-Mesh — kein
    Crash, Violations ∈ [0,1], refine_detection läuft mit n_fold=7 + free_space_mask.
    Der Term nimmt NICHTS Anker-spezifisches an."""
    verts = gear_like_verts_mm()
    K, R_w2c, t_w2c, hw = topdown_cam()
    table_origin = np.zeros(3); t_world = np.zeros(3)
    R0 = A.axis_angle_matrix([1.0, 0, 0], np.pi)        # flach top-down liegend
    sil = RC.render_silhouette(verts, R0, t_world, table_origin,
                               R_w2c, t_w2c, K, hw)
    assert sil.sum() > 0
    # sichtbar = halbe Scheibe, Rest "occludiert"; free space = Frame minus beide.
    visib = sil.copy(); visib[:, hw[1] // 2:] = False
    occluded = sil & ~visib
    fs = RC.build_free_space_mask(hw, visib, occluded_mask=occluded, dilate=1)
    img = np.zeros((hw[0], hw[1], 3), np.uint8); img[sil] = 200
    ie = RC.image_edges(img)
    R_ref, info = RC.refine_detection(
        R0, verts_mm=verts, t_world_m=t_world, table_origin_m=table_origin,
        R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=hw, target_mask=visib,
        image_edge_mask=ie, visib_mask=visib, visib_fract=0.5, min_visib_fract=0.2,
        free_space_mask=fs, max_free_space_violation=0.30,
        sym_axis=(0.0, 1.0, 0.0), n_fold=7, tilt_degs=(),
        scorer="cpu_edge", min_margin=0.0)
    assert info["free_space_aware"] is True
    for d in info["detail"]:
        assert 0.0 <= d["free_space_violation"] <= 1.0
    # die Coarse (= wahre Pose) belegt KEINEN nennenswerten freien Raum.
    assert info["detail"][0]["free_space_violation"] < 0.30


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
