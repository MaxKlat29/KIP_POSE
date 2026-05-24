#!/usr/bin/env python3
"""Isaac-SDG  ->  BOP-format converter  (Story S-201 / T-060).

THE central data contract. Turns the raw bundle written by gen_sdg_arm_visible.py
into a BOP dataset that CNOS / GigaPose / MegaPose / GDRNPP / bop_toolkit read
without a single line of repo patching. Implements Viktor adr.md §1 EXACTLY.

Runs in the bop-venv (numpy + PIL only — NO Isaac dependency).

INPUT  (per frame idx, from gen_sdg_arm_visible.py):
    rgb_{idx:04d}.png
    depth_{idx:04d}.npy            metric distance-to-camera, float32, METRES
    instance_{idx:04d}.npy         per-pixel instance ids (uint32)
    instance_labels_{idx:04d}.json idToLabels: id -> {class, instanceId / "..."}
    semantic_{idx:04d}.npy / .json (fallback class lookup)
    gt_raw_{idx:04d}.json          intrinsics + cam_c2w + per-instance T_obj2world

OUTPUT  (BOP, Viktor adr.md §1):
    <bop_root>/<split>/<scene_id:06d>/
        scene_camera.json     cam_K (row-major), depth_scale, cam_R_w2c, cam_t_w2c
        scene_gt.json         per-instance cam_R_m2c, cam_t_m2c (mm), obj_id
        scene_gt_info.json    bbox_obj, bbox_visib [x,y,w,h], px_count_*, visib_fract
        rgb/{idx:06d}.png
        depth/{idx:06d}.png   16-bit, depth_scale aware
        mask_visib/{idx:06d}_{gt:06d}.png   visible silhouette (8-bit 0/255)
        mask/{idx:06d}_{gt:06d}.png         FULL silhouette (provisional = visible;
                                            replace with bop_toolkit calc_gt_masks)

CORRECTNESS (Viktor adr.md §4-R4):
  * w2c: gt_raw stores cam_c2w (Isaac/USD). We INVERT to w2c. [R(c2w) = R(w2c).T;
    t(w2c) = -R(w2c) @ t(c2w)].
  * mm: T_obj2world translation is scene-metres; *1000 -> mm.  cam_t_w2c also *1000.
  * row-major: all 3x3 / 4x4 stored row-major flattened (BOP/numpy native).
  * obj_id: label -> 1-based obj_id from the frozen mapping (= detector class +1).

The arm is NEVER an instance here: it has no semantic label in the scene, so it
carries no class in instance_labels and is skipped. It still occludes -> the
parts behind it simply have fewer visible pixels -> lower visib_fract.
"""
import argparse, glob, json, os, re, sys
import numpy as np
from PIL import Image

# ── FROZEN obj_id mapping (Viktor adr.md §1.2) ──────────────────────────────
# label (CAD/Registry name, case-insensitive) -> obj_id (1-based)
OBJ_ID = {
    "anker_kurz":            1,
    "anker_lang":            2,
    "buerstenhalter_2polig": 3,
    "getriebegehaeuse_typ4": 4,
    "ringmagnet":            5,
    "zahnrad":               6,
}
# Poltopf_kurz_centered intentionally absent (degenerate mesh, adr.md R6).

DEPTH_SCALE = 0.1  # depth_png(uint16) * 0.1 = mm  (0.1mm resolution, range ~6.5m)


def obj_id_for(label: str):
    return OBJ_ID.get((label or "").strip().lower())


def cam_K_from(focal_mm, aperture_mm, W, H):
    """Pinhole intrinsics, row-major 9-list. Isaac renders centred (cx=W/2)."""
    fx = focal_mm / aperture_mm * W
    # square pixels: vertical aperture = aperture * H/W -> fy == fx (px)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def invert_rt(R, t):
    """(R,t) for x' = R x + t  ->  inverse (Rinv, tinv) for x = Rinv x' + tinv."""
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv


def usd_view_to_bop_w2c(c2w_4x4):
    """gt_raw cam_c2w is column-vector row-major (p_world = M_c2w @ p_cam_usd).

    USD/Isaac camera looks down -Z with +Y up (OpenGL-style). BOP cameras look
    down +Z with +Y down (OpenCV-style). We must (a) invert c2w->w2c and
    (b) flip the Y and Z camera axes to go OpenGL->OpenCV.
    """
    M = np.asarray(c2w_4x4, float).reshape(4, 4)
    R_c2w = M[:3, :3]
    t_c2w = M[:3, 3]
    R_w2c, t_w2c = invert_rt(R_c2w, t_c2w)
    # OpenGL(cam) -> OpenCV(cam): flip Y,Z rows of the cam-frame
    flip = np.diag([1.0, -1.0, -1.0])
    R_w2c = flip @ R_w2c
    t_w2c = flip @ t_w2c
    return R_w2c, t_w2c


def m2c_from(R_w2c, t_w2c_m, T_obj2world_4x4):
    """Compose model->cam from world->cam and obj->world.

    UNIT CHAIN (critical, verified against the smoke gt_raw):
      * The PLY model is in MM (= the raw, unscaled USD geometry).
      * Isaac spawns parts at OBJECT_SCALE = 1e-3, so the obj->world transform's
        3x3 block is R_pure * 1e-3 (rotation WITH the metre-scale baked in) and
        the translation is in METRES.
      * A model-mm point p maps to world-metres as:
            p_world_m = t_o2w_m + 1e-3 * R_pure @ p_mm
      * world->cam then *1000 for mm gives:
            p_cam_mm = (R_w2c @ R_pure) @ p_mm + 1000 * (R_w2c @ t_o2w_m + t_w2c_m)
        so R_m2c = R_w2c @ R_pure  and the 1e-3/1000 cancel on the rotation term.

    We recover R_pure by normalising the column scales out of the 3x3 block
    (robust to any uniform/near-uniform scale), then re-orthonormalise via SVD.
    """
    To = np.asarray(T_obj2world_4x4, float).reshape(4, 4)
    A = To[:3, :3]
    t_o2w_m = To[:3, 3]
    # remove per-column scale (= the 1e-3 object scale) to get a pure rotation
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    R_pure = A / col_norms
    # re-orthonormalise (SVD) to kill numeric drift
    U, _, Vt = np.linalg.svd(R_pure)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:        # guard against reflection
        U[:, -1] *= -1
        R_pure = U @ Vt
    R_m2c = R_w2c @ R_pure
    t_m2c_m = R_w2c @ t_o2w_m + t_w2c_m
    return R_m2c, t_m2c_m * 1000.0  # mm


def instance_id_to_path(info: dict):
    """instance-seg idToLabels maps id -> the PRIM PATH string, e.g.
       { "5": "/World/SpawnedObject_001", "2": "/World/Anker_Tray/Anker_Lang" }.
    Return id(int) -> path(str)."""
    out = {}
    for k, v in (info.get("idToLabels", info) or {}).items():
        try:
            iid = int(k)
        except (ValueError, TypeError):
            continue
        out[iid] = v if isinstance(v, str) else json.dumps(v)
    return out


def match_instance_ids(inst_id2path, prim_path):
    """Find the instance-seg mask id(s) for a spawned GT prim by matching the
    SpawnedObject_NNN token in the prim-path string. EXACT (no class ambiguity):
    the instance-seg idToLabels carries the full prim path, so distinct same-class
    instances map to distinct mask ids unambiguously.
    """
    token = prim_path.rsplit("/World/", 1)[-1].split("/")[0]  # SpawnedObject_003
    return [iid for iid, p in inst_id2path.items() if token in p]


def aabb_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def load_bop_models(models_dir):
    """Load each obj PLY (mm). Returns {obj_id: trimesh.Trimesh}."""
    import trimesh
    out = {}
    for oid in OBJ_ID.values():
        p = os.path.join(models_dir, f"obj_{oid:06d}.ply")
        if os.path.exists(p):
            out[oid] = trimesh.load(p, force="mesh")
    return out


def render_full_mask(mesh, K, R_m2c, t_m2c_mm, W, H):
    """Rasterize the GT-posed mesh silhouette (occlusion-FREE full mask) by
    filling the projected triangles. This is the BOP `mask` (the full silhouette
    ignoring scene occlusion) used for px_count_all + visib_fract. Pure
    PIL/numpy — no EGL/vispy renderer needed.
    """
    from PIL import Image as _I, ImageDraw as _D
    Km = np.asarray(K, float).reshape(3, 3)
    V = np.asarray(mesh.vertices, float)
    Pc = (R_m2c @ V.T).T + t_m2c_mm
    z = Pc[:, 2]
    uv = (Km @ Pc.T).T
    front = z > 1e-6
    uv2 = np.full((len(V), 2), -1e6)
    uv2[front] = uv[front, :2] / uv[front, 2:3]
    im = _I.new("L", (W, H), 0); dr = _D.Draw(im)
    F = mesh.faces
    for f in F:
        if not (front[f[0]] and front[f[1]] and front[f[2]]):
            continue
        tri = [tuple(uv2[i]) for i in f]
        # skip degenerate / wildly out-of-frame
        xs = [p[0] for p in tri]; ys = [p[1] for p in tri]
        if max(xs) < 0 or min(xs) >= W or max(ys) < 0 or min(ys) >= H:
            continue
        dr.polygon(tri, fill=255)
    return np.asarray(im) > 0


def convert_frame(raw_dir, gt_raw, bop_scene_dir, im_id, depth_scale, models=None,
                  min_visib=0.0):
    """min_visib: drop any instance whose visib_fract <= min_visib (under-arm scoping,
    Max rule: a part that is <=20% visible does not exist — not in GT, not in masks).
    Returns (key, cam_entry, gt_list, info_list, n_dropped)."""
    idx = gt_raw["image_id"]
    W, H = gt_raw["width"], gt_raw["height"]

    # camera
    K = cam_K_from(gt_raw["focal_length_mm"], gt_raw["horizontal_aperture_mm"], W, H)
    R_w2c, t_w2c_m = usd_view_to_bop_w2c(gt_raw["cam_c2w"])
    cam_entry = {
        "cam_K": [float(x) for x in K],
        "depth_scale": depth_scale,
        "cam_R_w2c": [float(x) for x in R_w2c.reshape(-1)],
        "cam_t_w2c": [float(x) for x in (t_w2c_m * 1000.0)],  # mm
    }

    # instance-seg map (id -> PRIM PATH). Exact prim-path matching to GT.
    inst = np.load(os.path.join(raw_dir, f"instance_{idx:04d}.npy"))
    if inst.ndim > 2:
        inst = inst[..., 0]
    inst_info = json.load(open(os.path.join(raw_dir, f"instance_labels_{idx:04d}.json")))
    inst_id2path = instance_id_to_path(inst_info)

    # rgb + depth copy/convert
    rgb_src = os.path.join(raw_dir, f"rgb_{idx:04d}.png")
    Image.open(rgb_src).convert("RGB").save(
        os.path.join(bop_scene_dir, "rgb", f"{im_id:06d}.png"))
    depth_npy = os.path.join(raw_dir, f"depth_{idx:04d}.npy")
    if os.path.exists(depth_npy):
        dep_m = np.load(depth_npy).astype(np.float32)
        dep_m = np.nan_to_num(dep_m, nan=0.0, posinf=0.0, neginf=0.0)
        if dep_m.ndim > 2:
            dep_m = dep_m[..., 0]
        dep_mm = dep_m * 1000.0
        dep_png = np.clip(dep_mm / depth_scale, 0, 65535).astype(np.uint16)
        Image.fromarray(dep_png).save(os.path.join(bop_scene_dir, "depth", f"{im_id:06d}.png"))

    gt_list = []
    info_list = []
    gt_id = 0
    n_dropped = 0
    # buffer (instance -> computed fields) so we can apply the >min_visib filter
    # BEFORE assigning gt_ids/mask names (keeps gt_ids contiguous, T-038 rule).
    staged = []
    for ins in gt_raw["instances"]:
        oid = obj_id_for(ins["label"])
        if oid is None:
            continue  # not a BOP object (e.g. Poltopf) -> skip
        R_m2c, t_m2c_mm = m2c_from(R_w2c, t_w2c_m, ins["T_obj2world"])
        # visible mask from instance-seg, matched EXACTLY by prim-path token
        cand_ids = match_instance_ids(inst_id2path, ins["prim_path"])
        vis = np.zeros((H, W), bool)
        for cid in cand_ids:
            vis |= (inst == cid)
        px_visib = int(vis.sum())
        mask_visib = (vis * 255).astype(np.uint8) if px_visib else np.zeros((H, W), np.uint8)
        bbox_visib = aabb_from_mask(vis) if px_visib else [0, 0, 0, 0]

        # FULL silhouette (occlusion-free) by rasterising the GT-posed mesh.
        if models is not None and oid in models:
            full = render_full_mask(models[oid], K, R_m2c, t_m2c_mm, W, H)
            # the visible mask must be a SUBSET of the full silhouette; clip to be safe
            vis_in_full = vis & full
            px_all = int(full.sum())
            px_vis_clip = int(vis_in_full.sum())
            mask_full = (full * 255).astype(np.uint8)
            bbox_obj = aabb_from_mask(full) if px_all else bbox_visib
            visib_fract = (px_vis_clip / px_all) if px_all else 0.0
            px_visib_eff = px_vis_clip
        else:
            mask_full = mask_visib
            bbox_obj = bbox_visib
            px_all = px_visib
            visib_fract = 1.0 if px_visib else 0.0
            px_visib_eff = px_visib

        # T-038 >20% scoping: a part <=min_visib visible is fairly not pose-able.
        # Cut it from GT + masks entirely (the occlusion stays in the RGB; only the
        # label/target/eval ignores it). min_visib=0.0 keeps everything (back-compat).
        if float(visib_fract) <= min_visib:
            n_dropped += 1
            continue

        staged.append({
            "oid": oid, "R_m2c": R_m2c, "t_m2c_mm": t_m2c_mm,
            "mask_visib": mask_visib, "mask_full": mask_full,
            "bbox_obj": bbox_obj, "bbox_visib": bbox_visib,
            "px_all": px_all, "px_visib_eff": px_visib_eff,
            "visib_fract": visib_fract,
        })

    for st in staged:
        Image.fromarray(st["mask_visib"]).save(
            os.path.join(bop_scene_dir, "mask_visib", f"{im_id:06d}_{gt_id:06d}.png"))
        Image.fromarray(st["mask_full"]).save(
            os.path.join(bop_scene_dir, "mask", f"{im_id:06d}_{gt_id:06d}.png"))

        gt_list.append({
            "obj_id": st["oid"],
            "cam_R_m2c": [float(x) for x in st["R_m2c"].reshape(-1)],
            "cam_t_m2c": [float(x) for x in st["t_m2c_mm"]],
        })
        info_list.append({
            "bbox_obj": st["bbox_obj"],
            "bbox_visib": st["bbox_visib"],
            "px_count_all": st["px_all"],
            "px_count_valid": st["px_all"],
            "px_count_visib": st["px_visib_eff"],
            "visib_fract": round(float(st["visib_fract"]), 4),
        })
        gt_id += 1

    return str(im_id), cam_entry, gt_list, info_list, n_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with the Isaac raw bundle")
    ap.add_argument("--bop-root", required=True, help="BOP dataset root (project/bop/pose_isaac)")
    ap.add_argument("--split", default="train_pbr")
    ap.add_argument("--scene-id", type=int, default=0, help="BOP scene_id (6-digit dir)")
    ap.add_argument("--start-im", type=int, default=0, help="first BOP image id in this scene")
    ap.add_argument("--depth-scale", type=float, default=DEPTH_SCALE)
    ap.add_argument("--limit", type=int, default=0, help="cap #frames (0=all)")
    ap.add_argument("--frame-list", default="",
                    help="comma-separated raw frame indices to put in THIS scene "
                         "(e.g. 0,1,2,...); overrides the glob. Used by "
                         "convert_full_to_bop.py to drive train/val multi-scene splits.")
    ap.add_argument("--no-full-mask", action="store_true",
                    help="skip GT-posed mesh rasterisation (mask=mask_visib, visib_fract provisional)")
    ap.add_argument("--min-visib", type=float, default=0.0,
                    help="T-038 >20% scoping: drop instances with visib_fract <= this "
                         "(default 0.0 = keep all; use 0.20 for the pipeline filter)")
    args = ap.parse_args()

    raw = args.raw_dir
    scene_dir = os.path.join(args.bop_root, args.split, f"{args.scene_id:06d}")
    for sub in ("rgb", "depth", "mask", "mask_visib"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    models = None
    if not args.no_full_mask:
        mdir = os.path.join(args.bop_root, "models")
        if os.path.isdir(mdir):
            models = load_bop_models(mdir)
            print(f"[isaac_to_bop] loaded {len(models)} models for full-silhouette + visib_fract")
        else:
            print(f"[isaac_to_bop] WARN no models/ at {mdir} -> visib_fract provisional", file=sys.stderr)

    if args.frame_list.strip():
        idxs = [int(x) for x in args.frame_list.split(",") if x.strip() != ""]
        gt_files = [os.path.join(raw, f"gt_raw_{i:04d}.json") for i in idxs]
        missing = [f for f in gt_files if not os.path.exists(f)]
        if missing:
            print(f"[isaac_to_bop] {len(missing)} frame(s) in --frame-list have no gt_raw "
                  f"(first: {missing[0]})", file=sys.stderr); sys.exit(1)
    else:
        gt_files = sorted(glob.glob(os.path.join(raw, "gt_raw_*.json")))
    if args.limit:
        gt_files = gt_files[:args.limit]
    if not gt_files:
        print(f"[isaac_to_bop] no gt_raw_*.json in {raw}", file=sys.stderr); sys.exit(1)

    scene_camera, scene_gt, scene_gt_info = {}, {}, {}
    im_id = args.start_im
    n_inst = 0
    n_dropped_total = 0
    for gf in gt_files:
        gt_raw = json.load(open(gf))
        key, cam, gtl, info, n_drop = convert_frame(
            raw, gt_raw, scene_dir, im_id, args.depth_scale, models,
            min_visib=args.min_visib)
        scene_camera[key] = cam
        scene_gt[key] = gtl
        scene_gt_info[key] = info
        n_inst += len(gtl)
        n_dropped_total += n_drop
        im_id += 1

    json.dump(scene_camera, open(os.path.join(scene_dir, "scene_camera.json"), "w"), indent=2)
    json.dump(scene_gt, open(os.path.join(scene_dir, "scene_gt.json"), "w"), indent=2)
    json.dump(scene_gt_info, open(os.path.join(scene_dir, "scene_gt_info.json"), "w"), indent=2)
    drop_note = (f", dropped {n_dropped_total} (visib_fract<={args.min_visib})"
                 if args.min_visib > 0 else "")
    print(f"[isaac_to_bop] scene {args.scene_id:06d}: {im_id - args.start_im} frames, "
          f"{n_inst} GT instances{drop_note} -> {scene_dir}")
    if models is not None:
        print("[isaac_to_bop] mask/ = true full silhouette (GT-posed mesh raster), "
              "mask_visib/ = arm-occluded visible, scene_gt_info.visib_fract computed.")
    else:
        print("[isaac_to_bop] NOTE: mask/ provisional (=mask_visib); no models loaded.")


if __name__ == "__main__":
    main()
