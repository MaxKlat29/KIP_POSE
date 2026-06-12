#!/usr/bin/env python3
"""moe_shadow_sim.py — Raytracing-Sim des IR-Schattens auf dem Tisch (T-178).

Real-Problem (Max): die Depth-Kamera (Structured-Light-Projektor) sitzt auf
RGB-Cam-Hoehe, aber lateral versetzt (weg vom Nullpunkt; von vorne gesehen
rechts) und leicht dorthin geneigt. Der LARA5-Arm blockt einen Teil der
IR-Strahlen -> auf dem Tisch entsteht ein Depth-Schatten, in dem RGBD-Pose-
Schaetzer keine Tiefe haben. Plan: Schattenbereich per Raytracing formulieren,
Max zeichnet ein, dann MoE-Routing (RGB im Schatten / RGBD sonst).

Modell:
  * Occluder = cell.glb (Arm in Zivid-Detection-Pose + Basiswagen + Trays),
    alles oberhalb der Tischplatte. Punktquellen-Schatten (Projektor-Apertur
    klein gegen Arm-Distanz -> Halbschatten vernachlaessigt).
  * IR-Quelle = RGB-Cam-Weltposition + --offset (Default +100mm x, 0, 0).
    Die NEIGUNG der Depth-Cam aendert nur deren FOV, nicht die Schatten-
    geometrie einer Punktquelle -> Position ist der relevante Parameter.
  * Structured Light braucht Projektor UND Kamera: Depth fehlt wo der
    PROJEKTOR blockiert ist ODER die Kamera den Punkt nicht sieht. Der
    MoE-RGB-Bereich = (IR-Schatten) AND NOT (RGB-View-Schatten) — wo die
    RGB-Kamera den Tisch noch sieht, aber keine Tiefe existiert. Im View-
    Schatten selbst sieht NIEMAND etwas (dort hilft auch RGB nicht).

Output (--out-dir):
  t178_shadow_topdown.png   Top-Down-Schema (Tisch, Arm-Footprint, Konturen)
  t178_shadow_overlay.png   Konturen auf echtem val-RGB-Frame (zum Einzeichnen)
  moe_shadow.json           Polygone (Welt-mm) + Parameter (fuer FE/Routing)

Aufruf (Box):
  /mnt/data/isaacsim-venv/bin/python box_src/moe_shadow_sim.py \
      --offset 0.10,0,0 [--grid-res 0.005] [--out-dir /mnt/data/kip_pose/recon_t178]
"""
import argparse
import json
import os

import numpy as np

CELL_GLB = "/mnt/data/kip_pose/project/frontend/assets/cell.glb"
VAL_SCENE = "/mnt/data/kip_pose/project/bop/pose_isaac/val/000000"
# Tisch-/Spawn-Bereich (gen_sdg_arm_visible Defaults, Meter, Welt-Frame)
TABLE_X = (0.05732, 0.78332)
TABLE_Y = (0.04157, 0.53698)
TABLE_Z = 0.0
RAY_EPS = 0.008          # m: Treffer naeher als (dist - eps) = blockiert
SRC_MIN_Z = 0.02         # Occluder-Filter: nur Geometrie oberhalb der Platte


def load_occluder():
    import trimesh
    sc = trimesh.load(CELL_GLB, process=False)
    mesh = sc.dump(concatenate=True) if hasattr(sc, "dump") else sc
    # nur Dreiecke oberhalb der Platte behalten (Beine/Unterbau blocken nie
    # einen Strahl von oben auf die Platte, sparen aber massiv Tris)
    tz = mesh.triangles.reshape(-1, 9)[:, [2, 5, 8]]
    keep = (tz.max(axis=1) > SRC_MIN_Z)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def rgb_cam_world():
    cam = json.load(open(os.path.join(VAL_SCENE, "scene_camera.json")))["0"]
    R = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
    t = np.array(cam["cam_t_w2c"], float)
    C = -R.T @ t / 1000.0          # Welt, Meter
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    return C, K, R, t


def shadow_mask(mesh, source, grid_xy, eps=RAY_EPS,
                ray_chunk=192, tri_chunk=8192):
    """Bool-Maske: True = Tischpunkt liegt im Schatten der Quelle.

    Eigener vektorisierter Moeller-Trumbore (kein embree auf der Box, und die
    isaacsim-venv bleibt unangetastet). Quelle ist FIX pro Aufruf -> tvec,
    qvec und der t-Zaehler sind reine Dreieck-Konstanten; pro Strahl-Block
    bleibt nur pvec = D x E2 (Kreuzprodukt) + zwei Skalarprodukte. Bereits
    blockierte Strahlen werden in spaeteren Tri-Bloecken uebersprungen."""
    targets = np.column_stack([grid_xy, np.full(len(grid_xy), TABLE_Z)])
    vecs = targets - source[None, :]
    dists = np.linalg.norm(vecs, axis=1)
    dirs = (vecs / dists[:, None]).astype(np.float64)
    tmax = dists - eps

    tri = mesh.triangles.astype(np.float64)        # (T,3,3)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    tvec = source[None, :] - v0                    # (T,3) konstant
    qvec = np.cross(tvec, e1)                      # (T,3) konstant
    t_num = np.einsum("tj,tj->t", e2, qvec)        # (T,)  konstant

    blocked = np.zeros(len(dirs), bool)
    for r0 in range(0, len(dirs), ray_chunk):
        r1 = min(r0 + ray_chunk, len(dirs))
        D = dirs[r0:r1]
        Tm = tmax[r0:r1]
        alive = ~blocked[r0:r1]
        for t0 in range(0, len(v0), tri_chunk):
            if not alive.any():
                break
            t1 = min(t0 + tri_chunk, len(v0))
            Da = D[alive]                                       # (R,3)
            pvec = np.cross(Da[:, None, :], e2[None, t0:t1, :])  # (R,T,3)
            det = np.einsum("tj,rtj->rt", e1[t0:t1], pvec)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / det
                u = np.einsum("tj,rtj->rt", tvec[t0:t1], pvec) * inv
                v = (Da @ qvec[t0:t1].T) * inv
                t = t_num[None, t0:t1] * inv
            hit = ((np.abs(det) > 1e-12) & (u >= 0.0) & (v >= 0.0)
                   & (u + v <= 1.0) & (t > 1e-6) & (t < Tm[alive, None]))
            newly = hit.any(axis=1)
            if newly.any():
                idx_alive = np.flatnonzero(alive)
                blocked[r0 + idx_alive[newly]] = True
                alive[idx_alive[newly]] = False
    return blocked


def mask_to_polygons(mask2d, xs, ys, min_area_px=25):
    """Konturen der Maske -> Liste von Polygonen in Welt-mm."""
    import cv2
    m = (mask2d.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        # Pixel -> Welt (Spalte=x-Index, Zeile=y-Index)
        wx = np.interp(c[:, 0], np.arange(len(xs)), xs) * 1000.0
        wy = np.interp(c[:, 1], np.arange(len(ys)), ys) * 1000.0
        polys.append(np.column_stack([wx, wy]).round(1).tolist())
    return polys


def render_topdown(path, xs, ys, ir_mask, view_mask, foot_mask, cam_c, ir_c):
    """Top-Down-Schema in Operator-Orientierung: man steht VOR dem Tisch,
    vorne (kleines y) = UNTEN im Bild, x nach rechts. Banner unten, damit
    nichts den Tisch ueberlappt."""
    import cv2
    H, W = len(ys), len(xs)
    S = 8
    moe = ir_mask & ~view_mask
    img = np.full((H, W, 3), 248, np.uint8)
    img[view_mask] = (175, 175, 175)             # View-Schatten: grau gefuellt
    img[moe] = (140, 140, 255)                   # MoE-RGB-Zone: rot gefuellt
    img = img[::-1]                              # Operator-Sicht: vorne unten
    img = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_NEAREST)

    def draw_cnt(mask, color, thick):
        m = (mask[::-1].astype(np.uint8)) * 255
        m = cv2.resize(m, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, color, thick)

    draw_cnt(foot_mask, (120, 120, 120), 2)      # Arm/Aufbau-Footprint: nur Linie
    draw_cnt(view_mask, (90, 90, 90), 2)
    draw_cnt(moe, (0, 0, 220), 4)                # MoE-Zone: dicke rote Linie

    def mark(pt_world, color, label, dy):
        px = int(np.interp(pt_world[0], xs, np.arange(W)) * S)
        py = int((H - 1 - np.interp(pt_world[1], ys, np.arange(H))) * S)
        cv2.drawMarker(img, (px, py), color, cv2.MARKER_CROSS, 26, 3)
        cv2.putText(img, label, (px + 14, py + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    mark(cam_c, (200, 120, 0), "RGB-Cam", -12)
    mark(ir_c, (0, 0, 220), "IR-Quelle (Depth)", 24)

    # Banner unterhalb des Tisches
    bh = 96
    banner = np.full((bh, W * S, 3), 255, np.uint8)
    cv2.putText(banner, "T-178 IR-Schatten-Sim  (Blick von oben, VORNE = unten)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(banner, "ROT = MoE-RGB-Zone (Depth tot, RGB ok)   GRAU = View-Schatten "
                        "(keiner sieht)   Linie = Arm/Aufbau-Footprint",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    cv2.putText(banner, f"x {xs[0]*1000:.0f}..{xs[-1]*1000:.0f}mm | "
                        f"y {ys[0]*1000:.0f}..{ys[-1]*1000:.0f}mm (Welt, Platte z=0) | "
                        f"IR-Quelle = RGB-Cam + Offset",
                (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    img = np.vstack([img, banner])
    cv2.imwrite(path, img)


def render_overlay(path, ir_polys, view_polys, K, R, t):
    """Konturen ins echte RGB-Bild projizieren (val 000000 frame 0)."""
    import cv2
    img = cv2.imread(os.path.join(VAL_SCENE, "rgb", "000000.png"))

    def project(poly_mm):
        P = np.array(poly_mm, float)
        Pw = np.column_stack([P, np.full(len(P), TABLE_Z * 1000.0)])  # mm
        Pc = (R @ Pw.T).T + t
        uv = (K @ Pc.T).T
        return (uv[:, :2] / uv[:, 2:3]).astype(np.int32)

    for poly in view_polys:
        cv2.polylines(img, [project(poly)], True, (90, 90, 90), 2)
    for poly in ir_polys:
        cv2.polylines(img, [project(poly)], True, (0, 0, 230), 3)
    cv2.putText(img, "rot = MoE-RGB-Zone (kein Depth, RGB ok) | grau = View-Schatten",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", default=None,
                    help="IR-Quelle relativ zur RGB-Cam, Meter 'dx,dy,dz'")
    ap.add_argument("--ir-pos", default="0.20,0.088,1.10",
                    help="IR-Quelle ABSOLUT in Welt-Metern 'x,y,z' (Max-Rig: "
                         "~20cm rechts vom Nullpunkt an der Decke, RGB-Cam-"
                         "Hoehe 1.10m, Montagelinie y=88mm). --offset gewinnt.")
    ap.add_argument("--grid-res", type=float, default=0.005)
    ap.add_argument("--out-dir", default="/mnt/data/kip_pose/recon_t178")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cam_c, K, R, t = rgb_cam_world()
    if args.offset:
        ir_c = cam_c + np.array([float(v) for v in args.offset.split(",")])
    else:
        ir_c = np.array([float(v) for v in args.ir_pos.split(",")])
    print(f"[sim] RGB-Cam Welt: {np.round(cam_c*1000,1)} mm")
    print(f"[sim] IR-Quelle:    {np.round(ir_c*1000,1)} mm")

    mesh = load_occluder()
    print(f"[sim] Occluder: {len(mesh.faces)} Tris (>z={SRC_MIN_Z}m)")

    xs = np.arange(TABLE_X[0], TABLE_X[1] + 1e-9, args.grid_res)
    ys = np.arange(TABLE_Y[0], TABLE_Y[1] + 1e-9, args.grid_res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    print(f"[sim] Grid {len(xs)}x{len(ys)} = {len(grid)} Strahlen/Quelle")

    ir_mask = shadow_mask(mesh, ir_c, grid).reshape(len(ys), len(xs))
    view_mask = shadow_mask(mesh, cam_c, grid).reshape(len(ys), len(xs))
    # Arm-Footprint: Quelle senkrecht weit oben ueber Tischmitte
    top_src = np.array([(TABLE_X[0]+TABLE_X[1])/2, (TABLE_Y[0]+TABLE_Y[1])/2, 50.0])
    foot_mask = shadow_mask(mesh, top_src, grid).reshape(len(ys), len(xs))

    moe_mask = ir_mask & ~view_mask
    a_cell = (args.grid_res * 1000.0) ** 2 / 100.0  # cm^2 pro Zelle
    print(f"[sim] IR-Schatten:   {ir_mask.sum()*a_cell:8.0f} cm^2 ({ir_mask.mean()*100:.1f}% des Tischs)")
    print(f"[sim] View-Schatten: {view_mask.sum()*a_cell:8.0f} cm^2 ({view_mask.mean()*100:.1f}%)")
    print(f"[sim] MoE-RGB-Zone:  {moe_mask.sum()*a_cell:8.0f} cm^2 ({moe_mask.mean()*100:.1f}%)")

    ir_polys = mask_to_polygons(moe_mask, xs, ys)
    view_polys = mask_to_polygons(view_mask, xs, ys)

    render_topdown(os.path.join(args.out_dir, "t178_shadow_topdown.png"),
                   xs, ys, ir_mask, view_mask, foot_mask, cam_c, ir_c)
    render_overlay(os.path.join(args.out_dir, "t178_shadow_overlay.png"),
                   ir_polys, view_polys, K, R, t)
    json.dump({
        "params": {"offset_m": off.tolist(), "grid_res_m": args.grid_res,
                   "rgb_cam_world_mm": (cam_c*1000).round(1).tolist(),
                   "ir_source_world_mm": (ir_c*1000).round(1).tolist(),
                   "table_x_mm": [TABLE_X[0]*1000, TABLE_X[1]*1000],
                   "table_y_mm": [TABLE_Y[0]*1000, TABLE_Y[1]*1000]},
        "moe_rgb_zone_polys_mm": ir_polys,
        "view_shadow_polys_mm": view_polys,
    }, open(os.path.join(args.out_dir, "moe_shadow.json"), "w"), indent=1)
    print(f"[sim] -> {args.out_dir}/t178_shadow_topdown.png + _overlay.png + moe_shadow.json")


if __name__ == "__main__":
    main()
