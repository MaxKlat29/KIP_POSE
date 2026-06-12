"""moe_shadow.py — fastapi-freie Helfer fuer die IR-Schatten-SIMULATION (T-178b).

Max will den MoE-Gewinn IN DER SIMULATION sehen (nicht erst mit echten
Aufnahmen): die Isaac-Sim rendert perfekte Tiefe ueberall — die echte Anlage
hat im IR-Schatten des Arms aber KEINE Tiefe. Dieses Modul entfernt die
gerenderte Tiefe genau dort (Backprojektion -> Welt -> Punkt-im-IR-Schatten-
Polygon), bevor die Pose-Stage laeuft. Ergebnis: RGB-D-Schaetzer erleben in
der Sim dieselbe Blindheit wie real, und das MoE-Routing (RGB-Zweig in der
Zone) zeigt seinen Mehrwert messbar (Zonen-GT-Treffer MoE vs RGB-D-only).

Bewusst stdlib+numpy (testbar ohne Box-venv, Muster gateway_proxy).
"""
from __future__ import annotations

import numpy as np


def points_in_poly(px, py, poly) -> np.ndarray:
    """Vektorisiertes Ray-Casting: welche Punkte (px[i],py[i]) liegen im Polygon
    (Liste [[x,y],...], gleiche Einheit wie px/py)."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        cond = (yi > py) != (yj > py)
        if cond.any():
            t = (py - yi) / ((yj - yi) if yj != yi else 1e-12)
            inside ^= cond & (px < xi + t * (xj - xi))
        j = i
    return inside


def points_in_any_poly(px, py, polys) -> np.ndarray:
    out = np.zeros(np.asarray(px).shape, dtype=bool)
    for poly in polys or []:
        out |= points_in_poly(px, py, poly)
    return out


def mask_depth_in_ir_shadow(depth_u16, K9, R_w2c9, t_w2c_mm, polys_mm,
                            depth_scale: float = 0.1,
                            max_world_z_mm: float = 250.0):
    """Setzt Depth-Pixel auf 0 (= ungueltig), deren 3D-Punkt im IR-Schatten
    liegt — genau das Loch, das die echte Anlage im Arm-Schatten hat.

    depth_u16    : uint16-Depth-PNG-Array (png * depth_scale = mm, BOP/T-156)
    K9/R_w2c9    : Kamera-Intrinsics (9) / Welt->Cam-Rotation (9)
    t_w2c_mm     : Welt->Cam-Translation (3, mm)
    polys_mm     : IR-Schatten-Polygone in Welt-mm (moe_shadow.json
                   `ir_shadow_polys_mm` — der VOLLE Projektor-Schatten)
    max_world_z_mm: nur Punkte unterhalb (Tisch + Teile) maskieren — der Arm
                   selbst bleibt; er traegt eh keine greifbaren Teile.
    -> (maskiertes uint16-Array (Kopie), n_maskierte_Pixel)
    """
    depth_u16 = np.asarray(depth_u16)
    d_mm = depth_u16.astype(np.float64) * float(depth_scale)
    ys, xs = np.nonzero(d_mm > 0)
    if len(xs) == 0:
        return depth_u16.copy(), 0
    z = d_mm[ys, xs]
    K = np.asarray(K9, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_w2c9, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_w2c_mm, dtype=np.float64).reshape(3)
    Xc = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                   (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
    Xw = (Xc - t) @ R                      # zeilenweise R^T @ (Xc - t)
    cand = Xw[:, 2] < float(max_world_z_mm)
    in_shadow = np.zeros(len(xs), dtype=bool)
    if cand.any():
        in_shadow[cand] = points_in_any_poly(Xw[cand, 0], Xw[cand, 1], polys_mm)
    out = depth_u16.copy()
    out[ys[in_shadow], xs[in_shadow]] = 0
    return out, int(in_shadow.sum())


def count_gt_hits(pred_pos_mm, gt_pos_mm, thresh_mm: float = 50.0) -> int:
    """Wie viele GT-Positionen haben (mind.) einen Pred innerhalb thresh_mm?
    Dimension folgt den Eingaben (2D oder 3D-Punkte). WICHTIG 3D nehmen, wenn
    der Depth-Ausfall bewertet wird: bei Top-Down-Sicht bricht fehlende Tiefe
    primaer die Z-Komponente — eine XY-only-Metrik ist dafuer blind (T-178b-
    Befund: RGB-D-coarse ohne Tiefe landet lateral fast richtig, nur radial
    daneben). Grobe Demo-Metrik, bewusst kein BOP-Protokoll."""
    if not gt_pos_mm or not pred_pos_mm:
        return 0
    P = np.asarray(pred_pos_mm, dtype=np.float64)
    G = np.asarray(gt_pos_mm, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    dim = min(P.shape[1], G.shape[1])
    hits = 0
    for g in G:
        d = np.linalg.norm(P[:, :dim] - g[None, :dim], axis=1)
        if float(d.min()) <= thresh_mm:
            hits += 1
    return hits
