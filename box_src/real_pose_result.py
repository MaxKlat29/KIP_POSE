#!/usr/bin/env python3
"""real_pose_result.py - REAL end-to-end pose_result.json from trained GDRNPP.

NO MOCK. This drives the *real* chain output through the *real* bop_adapter:

  trained GDRNPP (model_final.pth) inference  ->  BOP predictions CSV (R_m2c,t_m2c)
       +  real scene_camera.json (cam_R_w2c, cam_t_w2c, cam_K)
       ->  project/bop_adapter.detection_to_result (the production adapter)
       ->  schema-valid pose_result.json

GDRNPP runs per-object SO models and already wrote predictions for the val split
(output/.../<obj>/inference_/.../*.csv, combined to preds_all.csv). Those poses
ARE the trained-network output (TEST_BBOX_TYPE=gt inference, real weights). We
take one val frame, pull its real predictions, and run them through the same
adapter the laptop e2e_infer.py uses -> a real pose_result the viewer renders.

Also renders the detection overlay (GT bbox of each trained object on the RGB)
so the 2D side of the split-screen is a real image with real boxes + the arm.

Usage (on the box, bop-venv):
  python real_pose_result.py --scene 0 --im 92 \
     --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
     --preds /mnt/data/bop/results/preds_all.csv \
     --out-json /mnt/data/bop/results/pose_result.json \
     --out-overlay /mnt/data/bop/results/det_overlay.png
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# the PRODUCTION adapter (same module the laptop e2e_infer.py imports). Prefer a
# sibling ../project (so a scp'd scratch copy works), then the box repo location.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "project"), "/mnt/data/kip_pose/project"):
    if os.path.isfile(os.path.join(_p, "bop_adapter.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import bop_adapter as BOP  # noqa: E402
try:                                   # M2 RC refiner (T-058), optional
    import refine_rc as RC  # noqa: E402
except Exception:
    RC = None

OBJ_NFOLD = {1: None, 2: None, 6: 7}   # C_N per obj_id (models_info)

SCHEMA_VERSION = "1.0.0"
COORD_CONVENTION = ("Z-up world; column rotation world = R @ body; "
                    "origin = table-plane null-point")
# same table origin the laptop pipeline uses (e2e_infer.TABLE_ORIGIN_SCENE)
TABLE_ORIGIN_SCENE = (0.0, 0.0, 0.08)

# distinct overlay colors per obj_id
OBJ_COLORS = {
    1: (66, 135, 245), 2: (52, 199, 89), 3: (255, 159, 10),
    4: (191, 90, 242), 5: (255, 214, 10), 6: (255, 69, 58),
}


def load_preds_for_frame(preds_csv, scene_id, im_id):
    """Return list of {obj_id, score, R(3x3), t(3,) mm} for this exact frame."""
    from bop_toolkit_lib import inout
    raw = inout.load_bop_results(preds_csv, version="bop19")
    out = []
    for r in raw:
        if r["scene_id"] == scene_id and r["im_id"] == im_id:
            out.append({
                "obj_id": int(r["obj_id"]),
                "score": float(r["score"]),
                "R": np.array(r["R"], dtype=np.float64).reshape(3, 3),
                "t": np.array(r["t"], dtype=np.float64).reshape(3, 1),
            })
    return out


def load_gt_bboxes(dataset_dir, split, scene_id, im_id):
    """GT bbox per instance from scene_gt_info.json (for the det overlay)."""
    scene_dir = os.path.join(dataset_dir, split, f"{scene_id:06d}")
    info = json.load(open(os.path.join(scene_dir, "scene_gt_info.json")))
    gt = json.load(open(os.path.join(scene_dir, "scene_gt.json")))
    insts = gt[str(im_id)]
    binfo = info[str(im_id)]
    boxes = []
    for inst, bi in zip(insts, binfo):
        x, y, w, h = bi["bbox_visib"]
        boxes.append({"obj_id": int(inst["obj_id"]),
                      "bbox": [int(x), int(y), int(x + w), int(y + h)],
                      "visib": float(bi.get("visib_fract", 1.0))})
    return boxes


_MESH_CACHE = {}       # obj_id -> trimesh.Trimesh (WITH faces — needed for stable poses)


def _mesh(dataset_dir, obj_id):
    """Load the BOP CAD trimesh (mm, WITH faces) for obj_id. None if missing.

    Faces matter: compute_stable_poses on a faceless vertex cloud degenerates
    (hangs / no stable poses). Always load the full mesh."""
    if obj_id in _MESH_CACHE:
        return _MESH_CACHE[obj_id]
    m = None
    for sub in ("models_eval", "models"):
        p = os.path.join(dataset_dir, sub, f"obj_{int(obj_id):06d}.ply")
        if os.path.isfile(p):
            try:
                import trimesh
                m = trimesh.load(p, process=False)
                break
            except Exception:
                pass
    _MESH_CACHE[obj_id] = m
    return m


def _mesh_verts_mm(dataset_dir, obj_id):
    """CAD vertices (mm) for obj_id. None if missing."""
    m = _mesh(dataset_dir, obj_id)
    return None if m is None else np.asarray(m.vertices, dtype=np.float64)


def _matched_masks(dataset_dir, split, scene_id, im_id, preds):
    """pred-index -> matched GT-visib mask (greedy by obj_id + cam-t proximity).

    Mirrors the bbox greedy match; used to feed the RC refiner a real mask."""
    scene_dir = os.path.join(dataset_dir, split, f"{scene_id:06d}")
    gt = json.load(open(os.path.join(scene_dir, "scene_gt.json")))[str(im_id)]
    from PIL import Image as _I
    used, out = set(), {}
    for pi, p in enumerate(preds):
        tp = np.asarray(p["t"], float).reshape(-1)
        best, bd = None, None
        for gi, inst in enumerate(gt):
            if gi in used or int(inst["obj_id"]) != p["obj_id"]:
                continue
            tg = np.array(inst["cam_t_m2c"], float)
            d = float(np.linalg.norm(tp - tg))
            if bd is None or d < bd:
                bd, best = d, gi
        if best is None:
            continue
        used.add(best)
        mp = os.path.join(scene_dir, "mask_visib", f"{im_id:06d}_{best:06d}.png")
        if os.path.isfile(mp):
            out[pi] = np.array(_I.open(mp)) > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="val")
    # scene/im/out-json/out-overlay are required for the single-frame pose_result
    # path, but NOT for the A/B +TTA val-wide CSV path (--emit-preds/--val-all).
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--im", type=int, default=None)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-overlay", default=None)
    ap.add_argument("--planar-refine", action="store_true",
                    help="planar Z-snap on the world pose (T-055)")
    ap.add_argument("--refine-rc", action="store_true",
                    help="M2 multi-hypothesis render-and-compare refiner (T-058)")
    ap.add_argument("--rc-scorer", default="cpu_edge",
                    choices=["cpu_edge", "megapose"],
                    help="RC scorer: cpu_edge (no GPU) | megapose (GPU, finish-time)")
    ap.add_argument("--tta", action="store_true",
                    help="rotation test-time-augmentation at GDRNPP inference "
                         "(project/tta_pose.tta_call_gdrnpp). FINISH-TIME / GPU.")
    ap.add_argument("--emit-preds", default=None,
                    help="(A/B +TTA path) write the TTA-augmented predictions to "
                         "this BOP-results CSV instead of a single pose_result")
    ap.add_argument("--val-all", action="store_true",
                    help="(A/B +TTA path) run over the whole val split, not one frame")
    a = ap.parse_args()

    # --- A/B +TTA inference path (FINISH-TIME / GPU) --------------------------
    # The A/B step asks for TTA predictions over the whole val split as a BOP-
    # results CSV that refine_eval.py + eval_bop.py then score. This needs a LIVE
    # GDRNPP inference pass with the rotation augmentation (tta_pose wraps the
    # network call_fn over rot90 views) — it CANNOT be reconstructed from the
    # precomputed preds CSV. Wired + guarded; runs on the box vs the trained
    # checkpoints. Off-box (no GPU call_fn) this exits 3 = "finish-time".
    if a.emit_preds or a.val_all:
        try:
            sys.path.insert(0, os.path.join(_HERE, "..", "project"))
            import tta_pose as _TTA  # noqa: F401  (the rotation-vote engine)
        except Exception as exc:
            sys.stderr.write(f"[real_pose] FATAL --emit-preds/--val-all needs "
                             f"project/tta_pose.py: {exc!r}\n")
            sys.exit(2)
        sys.stderr.write(
            "[real_pose] +TTA val-wide inference is FINISH-TIME (GPU): it wraps the "
            "GDRNPP network call in tta_pose.tta_call_gdrnpp over rot90 views and "
            "writes the aggregated predictions to --emit-preds. The live GDRNPP "
            "call_fn binding lives in the gdrnpp inference entrypoint on the box; "
            "this guard makes the harness wiring explicit. Run on the box with the "
            "trained weights to materialise the +TTA CSV.\n")
        sys.exit(3)   # finish-time: not runnable off-box without the GPU call_fn

    # single-frame pose_result path requires scene/im/out-json/out-overlay
    for need, val in (("--scene", a.scene), ("--im", a.im),
                      ("--out-json", a.out_json), ("--out-overlay", a.out_overlay)):
        if val is None:
            ap.error(f"{need} is required for the single-frame pose_result path")

    scene_dir = os.path.join(a.dataset_dir, a.split, f"{a.scene:06d}")
    cam_all = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    cam = cam_all[str(a.im)]
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    R_w2c = np.array(cam["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.array(cam["cam_t_w2c"], dtype=np.float64).reshape(3)  # mm

    rgb_path = os.path.join(scene_dir, "rgb", f"{a.im:06d}.png")
    img = Image.open(rgb_path).convert("RGB")

    preds = load_preds_for_frame(a.preds, a.scene, a.im)
    # only trained objects have a model -> predictions exist only for 1,2,6
    trained = {p["obj_id"] for p in preds}
    sys.stderr.write(
        f"[real_pose] frame scene={a.scene} im={a.im}: {len(preds)} real GDRNPP "
        f"predictions for obj_ids {sorted(trained)}\n")

    rgb_np = np.asarray(img)
    H, W = rgb_np.shape[:2]
    do_rc = a.refine_rc and RC is not None
    if a.refine_rc and RC is None:
        sys.stderr.write("[real_pose] WARN --refine-rc but refine_rc.py not importable — skipping RC\n")

    def _build_rc_refiner(obj_id, mask):
        """Per-detection RC refiner callback (refine_rc.refine_detection). The
        mask is the matched GT-visib mask; the RGB crop drives the edge score.
        scorer=megapose is finish-time (GPU); cpu_edge runs here (no GPU)."""
        if not do_rc or mask is None:
            return None
        verts_mm = _mesh_verts_mm(a.dataset_dir, obj_id)
        if verts_mm is None:
            return None
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        bbox = [max(0, int(xs.min()) - 6), max(0, int(ys.min()) - 6),
                min(W, int(xs.max()) + 6), min(H, int(ys.max()) + 6)]
        crop = rgb_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ie = RC.image_edges(crop, bbox=bbox, full_hw=(H, W))
        try:                                   # pass the MESH (with faces) — not verts
            downs, _ = BOP.stable_pose_body_downs(mesh=_mesh(a.dataset_dir, obj_id),
                                                  prob_min=0.02, max_k=6,
                                                  cache_key=obj_id)
        except Exception:
            downs = None
        mk = None
        if a.rc_scorer == "megapose":
            mk = {"crop_rgb": crop, "K_full": K, "bbox": bbox,
                  "obj_label": f"obj_{int(obj_id):06d}",
                  "mesh_dir": os.path.join(a.dataset_dir, "models_eval")}

        def _refine(R_world, t_world):
            R_ref, info = RC.refine_detection(
                R_world, verts_mm=verts_mm, t_world_m=t_world,
                table_origin_m=TABLE_ORIGIN_SCENE, R_w2c=R_w2c, t_w2c_mm=t_w2c,
                K=K, hw=(H, W), target_mask=mask, image_edge_mask=ie,
                sym_axis=(0.0, 1.0, 0.0), n_fold=OBJ_NFOLD.get(obj_id),
                stable_downs=downs, scorer=a.rc_scorer, megapose_kwargs=mk)
            sys.stderr.write(
                f"[rc] obj{obj_id}: {info['n_hyps']} hyps scorer={info['scorer']} "
                f"best={info.get('best_tag')} switched={info.get('switched')}\n")
            return R_ref
        return _refine

    # masks per matched GT instance (for RC) — same greedy match as the bbox loop.
    masks_for = _matched_masks(a.dataset_dir, a.split, a.scene, a.im, preds)

    # --- REAL chain: GDRNPP pose -> production bop_adapter -> pose_result ---
    results = []
    for i, p in enumerate(preds):
        rc_refiner = _build_rc_refiner(p["obj_id"], masks_for.get(i))
        r = BOP.detection_to_result(
            instance_id=i,
            obj_id=p["obj_id"],
            R_m2c=p["R"],
            t_m2c_mm=p["t"],
            R_w2c=R_w2c,
            t_w2c_mm=t_w2c,
            table_origin_m=TABLE_ORIGIN_SCENE,
            bbox_2d=[0, 0, 1, 1],  # placeholder; replaced from GT-info below
            confidence=p["score"],
            apply_planar=a.planar_refine,
            mesh_verts_m=(_mesh_verts_mm(a.dataset_dir, p["obj_id"]) / 1000.0
                          if a.planar_refine else None),
            rc_refiner=rc_refiner,
        )
        results.append((p["obj_id"], r))

    # attach the real detection bbox (GT-info visib bbox per matched obj instance)
    gt_boxes = load_gt_bboxes(a.dataset_dir, a.split, a.scene, a.im)
    # greedy: give each result the next unused GT bbox of its obj_id
    used = set()
    for obj_id, r in results:
        for bi, gb in enumerate(gt_boxes):
            if bi in used or gb["obj_id"] != obj_id:
                continue
            r["bbox_2d"] = gb["bbox"]
            used.add(bi)
            break

    doc = {
        "meta": {
            "source_image": rgb_path,
            "table_origin": [float(v) for v in TABLE_ORIGIN_SCENE],
            "units": "m",
            "coordinate_convention": COORD_CONVENTION,
            "schema_version": SCHEMA_VERSION,
            "pose_backend": "GDRNPP",
            "checkpoint_note": "model_final.pth per-object SO (anker_kurz/anker_lang/zahnrad)",
            "levers": {"planar_refine": bool(a.planar_refine),
                       "refine_rc": bool(a.refine_rc and RC is not None),
                       "rc_scorer": a.rc_scorer if a.refine_rc else None,
                       "tta": bool(a.tta)},
            "frame": {"scene_id": a.scene, "im_id": a.im, "split": a.split},
        },
        "results": [r for _, r in results],
    }

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(doc, open(a.out_json, "w"), indent=2)
    sys.stderr.write(f"[real_pose] wrote {len(doc['results'])} REAL poses -> {a.out_json}\n")

    # --- detection overlay: GT-visib boxes of trained objects + arm scene ---
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for gb in gt_boxes:
        if gb["obj_id"] not in trained:
            continue
        x0, y0, x1, y1 = gb["bbox"]
        col = OBJ_COLORS.get(gb["obj_id"], (255, 255, 255))
        draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f"{BOP.part_for_obj_id(gb['obj_id'])} v{gb['visib']:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x0, y0 - 22, x0 + tw + 6, y0], fill=col)
        draw.text((x0 + 3, y0 - 21), label, fill=(0, 0, 0), font=font)
    img.save(a.out_overlay)
    sys.stderr.write(f"[real_pose] wrote detection overlay -> {a.out_overlay}\n")

    # echo to stdout for the laptop
    print(json.dumps({
        "n_results": len(doc["results"]),
        "obj_ids": sorted(trained),
        "results": [{"part": r["part"], "face": r["face"],
                     "t_world": [round(v, 3) for v in r["t_world"]],
                     "upright": r["upright"], "conf": round(r["confidence"], 2)}
                    for _, r in results],
    }, indent=2))


if __name__ == "__main__":
    main()
