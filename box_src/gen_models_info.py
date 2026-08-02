#!/usr/bin/env python3
"""GLB -> BOP models/  +  models_info.json with symmetries  (Story S-203 / T-062).

Implements adr.md §2 EXACTLY — the analytic fix for the 120deg/91deg
rotation problem. Per part:
  * export GLB mesh -> obj_{obj_id:06d}.ply in MILLIMETRES (GLB is metres -> x1000)
  * compute diameter, min_{x,y,z}, size_{x,y,z}  (mm, BOP models_info convention)
  * write symmetries per §2:
      Anker_Kurz / Anker_Lang / Ringmagnet -> symmetries_continuous, axis Y [0,1,0]
      Zahnrad                               -> symmetries_discrete, C_N (N = tooth count)
      Buerstenhalter / Getriebegehaeuse     -> none

Tooth count for the Zahnrad is COUNTED from the mesh (adr.md §2.2): take a thin
slice around the gear mid-plane (perpendicular to the symmetry axis Y), build the
radius-vs-angle profile of the rim, and count the periodic peaks (= teeth).

Runs in bop-venv (trimesh + numpy).
"""
import argparse, json, os
import numpy as np
import trimesh

# label -> obj_id (frozen, adr.md §1.2)
PARTS = [
    ("Anker_Kurz",            1, "continuous"),
    ("Anker_Lang",            2, "continuous"),
    ("Buerstenhalter_2polig", 3, "none"),
    ("Getriebegehaeuse_typ4", 4, "none"),
    ("Ringmagnet",            5, "continuous"),
    ("Zahnrad",               6, "discrete"),
]
SYM_AXIS = np.array([0.0, 1.0, 0.0])  # Y in the model frame (verified from extents)


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def discrete_matrix_flat(theta, center_mm):
    """4x4 row-major flat for a rotation by theta about the Y axis through
    `center_mm`. t = (I - R) @ c so the part rotates about its real centre."""
    R = Ry(theta)
    t = (np.eye(3) - R) @ np.asarray(center_mm, float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return [float(x) for x in M.reshape(-1)]


def count_gear_teeth(mesh, axis=SYM_AXIS):
    """Count the rotational symmetry order N of the gear about `axis` (=Y).

    METHOD (verified 2026-05-22): rasterize the top-down silhouette (projection
    along the symmetry axis) into a filled binary image, then measure the
    periodicity of BOTH the outer rim and the inner bore. The Zahnrad's outer
    rim is a smooth disc (no outer teeth — ptp ~0.25mm = raster noise); the
    rotational feature is the INNER 7-spline hub. We take the dominant FFT
    frequency of whichever boundary has real periodicity (ptp > 0.5mm) and the
    lower stable order. Visually verified: N=7 (7-spline bore).

    Vertex-scatter profiling (the obvious approach) FAILS here: the 2.7k-vertex
    cloud is too sparse, producing spurious single-bin spikes that fool both
    peak-count and FFT (gave 66/67 — pure noise). Rasterizing the filled
    polygon silhouette is the robust fix.
    """
    from PIL import Image, ImageDraw
    V = mesh.vertices[:, [0, 2]]  # axis is Y -> project to (X,Z)
    F = mesh.faces
    mn, mx = V.min(0), V.max(0)
    S = 900
    scale = (S - 20) / (mx - mn).max()
    to_px = lambda p: ((p - mn) * scale + 10).astype(int)
    im = Image.new("L", (S, S), 0); dr = ImageDraw.Draw(im)
    for f in F:
        dr.polygon([tuple(to_px(V[i])) for i in f], fill=255)
    img = np.asarray(im)
    ys, xs = np.nonzero(img); cx, cy = xs.mean(), ys.mean()
    rmax = int(scale * (mx - mn).max())
    nb = 1080
    angs = np.linspace(-np.pi, np.pi, nb, endpoint=False)

    def boundary(which):
        outer = np.zeros(nb); inner = np.zeros(nb)
        for i, a in enumerate(angs):
            rr = np.arange(0, rmax, 0.3)
            px = (cx + rr * np.cos(a)).astype(int); py = (cy + rr * np.sin(a)).astype(int)
            ok = (px >= 0) & (px < S) & (py >= 0) & (py < S)
            filled = np.zeros(len(rr), bool); filled[ok] = img[py[ok], px[ok]] > 0
            fn = np.nonzero(filled)[0]
            if len(fn):
                inner[i] = rr[fn[0]] / scale
                outer[i] = rr[fn[-1]] / scale
        return inner if which == "inner" else outer

    def dom_freq(prof):
        sp = np.abs(np.fft.rfft(prof - prof.mean())); fr = np.arange(len(sp))
        bm = (fr >= 3) & (fr <= 60)
        return int(fr[bm][np.argmax(sp[bm])]), float(prof.ptp())

    inner = boundary("inner"); outer = boundary("outer")
    n_in, ptp_in = dom_freq(inner)
    n_out, ptp_out = dom_freq(outer)
    # pick the boundary with a REAL periodic feature (ptp threshold 0.5mm)
    if ptp_in >= 0.5 and ptp_in >= ptp_out:
        N, src = n_in, f"inner-bore (ptp={ptp_in:.2f}mm)"
    elif ptp_out >= 0.5:
        N, src = n_out, f"outer-rim (ptp={ptp_out:.2f}mm)"
    else:
        N, src = n_in, f"weak (inner ptp={ptp_in:.2f}, outer ptp={ptp_out:.2f})"
    return N, {"n_inner": n_in, "ptp_inner_mm": round(ptp_in, 3),
               "n_outer": n_out, "ptp_outer_mm": round(ptp_out, 3), "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb-dir", default="/mnt/data/kip_pose/tmpl_build/part_glbs")
    ap.add_argument("--bop-root", required=True, help="BOP dataset root (writes models/)")
    ap.add_argument("--also-eval", action="store_true", help="duplicate to models_eval/")
    args = ap.parse_args()

    models_dir = os.path.join(args.bop_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    models_info = {}
    teeth_report = {}

    for name, oid, symkind in PARTS:
        glb = os.path.join(args.glb_dir, f"{name}.glb")
        mesh = trimesh.load(glb, force="mesh")
        # GLB is metres -> millimetres
        mesh.apply_scale(1000.0)
        ply_path = os.path.join(models_dir, f"obj_{oid:06d}.ply")
        mesh.export(ply_path)

        mn = mesh.vertices.min(0)
        mx = mesh.vertices.max(0)
        size = mx - mn
        # BOP diameter = max pairwise vertex distance (approx via convex hull)
        try:
            hull = mesh.convex_hull.vertices
        except Exception:
            hull = mesh.vertices
        # subsample hull for speed if huge
        if len(hull) > 2000:
            idx = np.random.default_rng(0).choice(len(hull), 2000, replace=False)
            hull = hull[idx]
        d = 0.0
        for i in range(len(hull)):
            dd = np.linalg.norm(hull[i + 1:] - hull[i], axis=1)
            if dd.size:
                d = max(d, float(dd.max()))
        center_mm = 0.5 * (mn + mx)

        info = {
            "diameter": round(d, 4),
            "min_x": round(float(mn[0]), 4), "min_y": round(float(mn[1]), 4),
            "min_z": round(float(mn[2]), 4),
            "size_x": round(float(size[0]), 4), "size_y": round(float(size[1]), 4),
            "size_z": round(float(size[2]), 4),
        }

        if symkind == "continuous":
            # axis through the part centre (offset = centre x,z; y arbitrary)
            info["symmetries_continuous"] = [{
                "axis": SYM_AXIS.tolist(),
                "offset": [round(float(center_mm[0]), 4), 0.0, round(float(center_mm[2]), 4)],
            }]
        elif symkind == "discrete":
            N, diag = count_gear_teeth(mesh)
            teeth_report[name] = {**diag, "chosen_N": N}
            mats = [discrete_matrix_flat(k * 2 * np.pi / N, center_mm) for k in range(1, N)]
            info["symmetries_discrete"] = mats
            info["_sym_note"] = f"C_{N} about Y ({diag['source']}; inner={diag['n_inner']} outer={diag['n_outer']})"
        # none -> no symmetry entry

        models_info[str(oid)] = info
        print(f"obj_{oid:06d} {name}: diam={info['diameter']:.1f}mm size="
              f"({info['size_x']:.1f},{info['size_y']:.1f},{info['size_z']:.1f}) "
              f"sym={symkind}" + (f" N={teeth_report.get(name,{}).get('chosen_N')}" if symkind == 'discrete' else ""))

    json.dump(models_info, open(os.path.join(models_dir, "models_info.json"), "w"), indent=2)
    print(f"[models_info] wrote {models_dir}/models_info.json with {len(models_info)} objects")
    if teeth_report:
        print("[models_info] tooth report:", json.dumps(teeth_report))

    if args.also_eval:
        import shutil
        eval_dir = os.path.join(args.bop_root, "models_eval")
        os.makedirs(eval_dir, exist_ok=True)
        for f in os.listdir(models_dir):
            shutil.copy(os.path.join(models_dir, f), os.path.join(eval_dir, f))
        print(f"[models_info] duplicated -> {eval_dir}")


if __name__ == "__main__":
    main()
