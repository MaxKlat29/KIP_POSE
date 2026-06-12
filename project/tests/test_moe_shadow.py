"""Tests fuer pipelines.moe_shadow (T-178b — IR-Schatten-Depth-Simulation).

Geometrie-Setup: Kamera exakt ueber dem Welt-Ursprung in 1000mm Hoehe, Blick
senkrecht nach unten, Identitaets-aehnliche Extrinsics (R = diag(1,-1,-1)
dreht Cam-z auf Welt--z), K mit fx=fy=1000, cx=cy=50 -> Pixel (50,50) trifft
Welt-(0,0,0). So sind Backprojektion und Polygon-Test von Hand nachrechenbar.
"""
import numpy as np
import pytest

from pipelines.moe_shadow import (
    count_gt_hits, mask_depth_in_ir_shadow, points_in_any_poly, points_in_poly,
)

# Welt->Cam: X_c = R (X_w) + t. Kamera bei Welt (0,0,1000), Blick -z (Welt):
# R = diag(1,-1,-1)  (Welt-x bleibt, Welt-y/-z gespiegelt -> rechtshaendig,
# Cam-+z zeigt auf den Tisch), t = -R @ C = (0, 0, 1000).
R_W2C = [1.0, 0, 0, 0, -1.0, 0, 0, 0, -1.0]
T_W2C = [0.0, 0.0, 1000.0]
K9 = [1000.0, 0, 50.0, 0, 1000.0, 50.0, 0, 0, 1.0]
SQUARE = [[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]]


def test_points_in_poly_basic():
    inside = points_in_poly([0.0, 100.0], [0.0, 0.0], SQUARE)
    assert inside.tolist() == [True, False]


def test_points_in_any_poly_union():
    polys = [SQUARE, [[90, -10], [110, -10], [110, 10], [90, 10]]]
    inside = points_in_any_poly([0, 100, 200], [0, 0, 0], polys)
    assert inside.tolist() == [True, True, False]


def _depth_image():
    # 101x101, uint16; depth_scale 0.1 -> png 10000 = 1000mm (Tischebene z_w=0).
    return np.full((101, 101), 10000, dtype=np.uint16)


def test_mask_depth_zeroes_only_shadow_pixels():
    dep = _depth_image()
    masked, n = mask_depth_in_ir_shadow(dep, K9, R_W2C, T_W2C, [SQUARE],
                                        depth_scale=0.1)
    # Pixel (50,50) -> Welt (0,0,0) liegt im 60x60mm-Quadrat -> maskiert.
    assert masked[50, 50] == 0
    # Pixel (50,90): u-Offset 40px * z/fx = 40mm -> Welt-x=40 > 30 -> bleibt.
    assert masked[50, 90] == 10000
    assert 0 < n < dep.size
    # Original unangetastet (Kopie-Vertrag).
    assert dep[50, 50] == 10000


def test_mask_depth_respects_max_world_z():
    dep = _depth_image()
    # Punkt 500mm ueber dem Tisch (png 5000 -> 500mm Cam-Distanz -> z_w=500):
    dep[50, 50] = 5000
    masked, _ = mask_depth_in_ir_shadow(dep, K9, R_W2C, T_W2C, [SQUARE],
                                        depth_scale=0.1, max_world_z_mm=250.0)
    # z_w = 1000-500 = 500 > 250 -> NICHT maskiert (Arm-Hoehe bleibt).
    assert masked[50, 50] == 5000


def test_mask_depth_no_polys_is_noop():
    dep = _depth_image()
    masked, n = mask_depth_in_ir_shadow(dep, K9, R_W2C, T_W2C, [],
                                        depth_scale=0.1)
    assert n == 0 and (masked == dep).all()


def test_count_gt_hits():
    gts = [(0.0, 0.0), (100.0, 0.0)]
    preds = [(10.0, 10.0), (300.0, 300.0)]
    # GT1: Pred1 in 14.1mm -> Treffer; GT2: naechster Pred 90mm/283mm -> kein Treffer.
    assert count_gt_hits(preds, gts, thresh_mm=50.0) == 1
    assert count_gt_hits([], gts) == 0
    assert count_gt_hits(preds, []) == 0


def test_count_gt_hits_3d_catches_depth_failure():
    # T-178b-Befund: ohne Tiefe landet RGB-D-coarse lateral fast richtig,
    # aber radial (Welt-Z bei Top-Down) weit daneben. Die 3D-Metrik MUSS das
    # als Miss werten — eine XY-only-Metrik waere blind dafuer.
    gts = [(0.0, 0.0, 14.0)]
    pred_depth_broken = [(5.0, 5.0, 120.0)]     # XY 7mm, Z +106mm
    pred_good = [(5.0, 5.0, 20.0)]
    assert count_gt_hits(pred_depth_broken, gts, thresh_mm=50.0) == 0
    assert count_gt_hits(pred_good, gts, thresh_mm=50.0) == 1
