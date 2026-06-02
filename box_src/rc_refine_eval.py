#!/usr/bin/env python3
"""rc_refine_eval.py — M2 Render-and-Compare-Refiner (CPU-Kanten-Scorer) auf
BESTEHENDE GDRNPP-val-Predictions anwenden + neue BOP-results-CSVs schreiben.

Schwester von refine_eval.py (T-041), aber für M2 (T-058): statt der training-
freien Tilt/Yaw-Snaps wendet es den MULTI-HYPOTHESEN-Render-and-Compare-Refiner
(refine_rc.refine_detection, scorer=cpu_edge) an. Pro Prediction:

  cam-Pose -> Welt (bop_adapter) -> generate_hypotheses (Coarse + 180°-Flips +
  C_N-Yaws + Ruhelagen + Tilts) -> CPU-Kanten/Silhouetten-Score gegen die REALE
  Detektor-Maske (mask_visib) UND die REALEN Bildkanten des RGB-Crops -> beste
  Hypothese (gegen Coarse gegated) -> Welt -> cam -> CSV.

Diese CSV wird mit eval_bop.py (symmetrie-bewusst, >20%-gefiltert) gescort — so
misst man EHRLICH, ob der CPU-Kanten-Scorer den Anker-Flip korrigiert (AR rauf?).

KEIN Training, KEINE GPU. numpy + trimesh (CPU) + PIL. Liest die val-Daten read-
only (rgb/, mask_visib/, scene_camera.json, scene_gt.json), schreibt nur --out-dir.

USAGE (Box, bop-venv):
  /mnt/data/bop/bop-venv/bin/python box_src/rc_refine_eval.py \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds /mnt/data/bop/results/val_preds_combined.csv \
    --out-dir /tmp/rc_out --config raw rc_anker rc_all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))
sys.path.insert(0, os.path.dirname(__file__))   # sam_proxy_mask lives here (T-089)
import bop_adapter as A  # noqa: E402
import refine_rc as RC  # noqa: E402

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"FATAL: trimesh nötig: {exc!r}\n")
    raise

# C_N pro obj_id (models_info): Anker continuous-Y -> kein discrete N; Zahnrad C_7.
OBJ_NFOLD = {1: None, 2: None, 6: 7}
SYM_AXIS = (0.0, 1.0, 0.0)
TABLE_ORIGIN = np.zeros(3)
DEFAULT_TABLE_Z = 0.018          # wie refine_eval (val-Welt-Frame Auflage-Median)

# Welche Konfigurationen: rc_anker = nur Anker (1,2) refinen; rc_all = alle.
# T-092: rc_anker_vis = vis-aware (shipped T-085), rc_anker_occ = vis-aware +
# FREE-SPACE-REFUTATION (ADR-020-Amendment). Der AR-before/after-Vergleich nutzt
# rc_anker_vis (before) vs rc_anker_occ (after) — selber vis-aware Score, das occ-
# Gate ist die einzige Differenz.
CONFIGS = {
    "raw": {"objs": set()},
    "rc_anker": {"objs": {1, 2}},
    "rc_all": {"objs": {1, 2, 6}},
    "rc_anker_vis": {"objs": {1, 2}, "visaware": True},
    "rc_anker_occ": {"objs": {1, 2}, "visaware": True, "freespace": True},
    "rc_all_occ": {"objs": {1, 2, 6}, "visaware": True, "freespace": True},
}


def load_cams(bop_root):
    cams = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        cams[sid] = json.load(open(os.path.join(sc, "scene_camera.json")))
    return cams


def load_meshes(bop_root):
    md = os.path.join(bop_root, "models_eval")
    if not os.path.isdir(md):
        md = os.path.join(bop_root, "models")
    meshes, downs = {}, {}
    for oid in (1, 2, 6):
        m = trimesh.load(os.path.join(md, f"obj_{oid:06d}.ply"), process=False)
        meshes[oid] = m
        d, _ = A.stable_pose_body_downs(mesh=m, prob_min=0.02, max_k=6, cache_key=oid)
        downs[oid] = d
    return meshes, downs


def load_visib_index(bop_root):
    """(scene, im, inst_idx) -> visib_fract aus scene_gt_info.json (für das
    visibility-konditionierte Gate + das free-space-Band-Gate, T-092)."""
    idx = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        ip = os.path.join(sc, "scene_gt_info.json")
        if not os.path.isfile(ip):
            continue
        info = json.load(open(ip))
        for im, insts in info.items():
            for i, fi in enumerate(insts):
                idx[(sid, int(im), i)] = float(fi.get("visib_fract", -1.0))
    return idx


def n_insts_in_frame(bop_root, sid, im):
    """Anzahl GT-Instanzen in (sid, im) — aus scene_gt.json."""
    gp = os.path.join(bop_root, "val", f"{sid:06d}", "scene_gt.json")
    if not os.path.isfile(gp):
        return 0
    gt = json.load(open(gp))
    return len(gt.get(str(im), []))


def other_full_masks(bop_root, sid, im, self_inst, n_insts):
    """Volle Masken ALLER ANDEREN Instanzen im Frame (für die other-parts-
    Exklusion in der free-space-Maske, T-092)."""
    out = []
    for j in range(n_insts):
        if j == self_inst:
            continue
        m = load_full_mask(bop_root, sid, im, j)
        if m is None:
            m = load_mask(bop_root, sid, im, j)
        if m is not None:
            out.append(m)
    return out


def load_gt_index(bop_root):
    """(scene,im,obj_id) -> [(inst_idx, cam_t_m2c)] zum Matchen pred<->GT-Maske."""
    idx = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        for im, insts in gt.items():
            for i, inst in enumerate(insts):
                key = (sid, int(im), int(inst["obj_id"]))
                idx.setdefault(key, []).append((i, np.array(inst["cam_t_m2c"], float)))
    return idx


def match_gt_inst(gt_index, sid, im, obj_id, t_m2c, used):
    cands = gt_index.get((sid, im, obj_id), [])
    best, bd = None, None
    for i, tg in cands:
        if (sid, im, obj_id, i) in used:
            continue
        d = float(np.linalg.norm(t_m2c - tg))
        if bd is None or d < bd:
            bd, best = d, i
    if best is not None:
        used.add((sid, im, obj_id, best))
    return best


def load_mask(bop_root, sid, im, inst_idx):
    p = os.path.join(bop_root, "val", f"{sid:06d}", "mask_visib",
                     f"{im:06d}_{inst_idx:06d}.png")
    if not os.path.isfile(p):
        return None
    return np.array(Image.open(p)) > 0


def load_full_mask(bop_root, sid, im, inst_idx):
    """Volle Silhouetten-Maske (mask/, inkl. occludierter Teile) — der Detektor-
    Maske-Proxy für den 'before'-Fall (Scorer über den ganzen Crop)."""
    p = os.path.join(bop_root, "val", f"{sid:06d}", "mask",
                     f"{im:06d}_{inst_idx:06d}.png")
    if not os.path.isfile(p):
        return None
    return np.array(Image.open(p)) > 0


def load_rgb(bop_root, sid, im):
    p = os.path.join(bop_root, "val", f"{sid:06d}", "rgb", f"{im:06d}.png")
    if not os.path.isfile(p):
        return None
    return np.asarray(Image.open(p).convert("RGB"))


def mask_bbox(mask, pad=6):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W, int(xs.max()) + pad), min(H, int(ys.max()) + pad)]


def _sam_proxy_mask(rgb, bbox, hw):
    """POSE-INDEPENDENT visible-mask proxy: SAM segment of the detector bbox
    (= AABB of the full silhouette; carries no pose/visibility). Returns (H,W)
    bool or None on failure (caller falls back to passthrough = today's live)."""
    if rgb is None or bbox is None:
        return None
    import sam_proxy_mask as SP
    x0, y0, x1, y1 = bbox
    m = SP.sam_visible_mask(rgb, [x0, y0, x1, y1])
    if m is None or m.shape != tuple(hw):
        return None
    return m


def read_preds(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "scene_id": int(r["scene_id"]), "im_id": int(r["im_id"]),
                "obj_id": int(r["obj_id"]), "score": float(r["score"]),
                "R": np.array([float(x) for x in r["R"].split()]).reshape(3, 3),
                "t": np.array([float(x) for x in r["t"].split()]),
                "time": float(r.get("time", -1)),
            })
    return rows


def world_to_cam(R_world, t_world, R_w2c, t_w2c):
    t_m2w_mm = (t_world + TABLE_ORIGIN) * 1000.0
    return R_w2c @ R_world, R_w2c @ t_m2w_mm + t_w2c


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for r in rows:
            R = " ".join(f"{v:.9g}" for v in r["R"].reshape(-1))
            t = " ".join(f"{v:.9g}" for v in r["t"].reshape(-1))
            f.write(f"{r['scene_id']},{r['im_id']},{r['obj_id']},{r['score']:.9g},"
                    f"{R},{t},{r['time']:.9g}\n")


def refine_rows(rows, cfg, cams, meshes, downs, gt_index, bop_root,
                table_z=DEFAULT_TABLE_Z, scorer="cpu_edge",
                visib_index=None, min_visib_fract=0.25,
                occ_max_violation=0.30, occ_dilate=2, occ_vf_hi=0.0,
                proxy_mask="gt", use_schedule=False, proxy_vf_scale=1.0,
                proxy_vf_clip=False):
    """proxy_mask: 'gt' = BOP mask_visib (T-085 upper bound), 'sam' = POSE-
    INDEPENDENT SAM segment of the detector box (T-089 live proxy). use_schedule:
    apply the SHIPPED visibility-staggered margin schedule (refine_rc.
    DEFAULT_MARGIN_SCHEDULE = T-085 v2 = what actually ships); with proxy_mask=sam
    the gate reads the GT-scale proxy visib_fract (SAM-px / coarse-silhouette-px)
    = the exact live behaviour.

    proxy_vf_scale / proxy_vf_clip (T-089 gate CALIBRATION, sam-path only): the raw
    proxy gate_vf = SAM-px / coarse-silhouette-px OVERSHOOTS (~median 1.29) and lands
    in the well-vis band so the staggered occluded band [0.20,0.50) never fires.
    proxy_vf_scale linearly maps the proxy onto the GT visib_fract scale (calibrated
    factor ~0.384 = GT-median/proxy-median) and proxy_vf_clip clamps to [0,1] so the
    band semantics line up. gt-path is untouched (=upper bound)."""
    objs = cfg["objs"]
    visaware = bool(cfg.get("visaware"))
    freespace = bool(cfg.get("freespace"))
    sched = RC.DEFAULT_MARGIN_SCHEDULE if use_schedule else None
    used = set()
    out = []
    stats = {"n": 0, "rc": 0, "switched": 0, "no_mask": 0, "refuted": 0,
             "proxy_fail": 0}
    for r in rows:
        sid, im, oid = r["scene_id"], r["im_id"], r["obj_id"]
        cam = cams[sid][str(im)]
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        R_world, t_world = A.bop_pose_to_world(r["R"], r["t"], R_w2c, t_w2c,
                                               TABLE_ORIGIN)
        if oid in objs:
            inst_idx = match_gt_inst(gt_index, sid, im, oid, r["t"], used)
            mask = load_mask(bop_root, sid, im, inst_idx) if inst_idx is not None else None
            if mask is None:
                stats["no_mask"] += 1
            else:
                # target: vis-aware nutzt mask_visib; sonst die volle Maske als
                # Detektor-Proxy (= das alte Verhalten der rc-Configs).
                full = (load_full_mask(bop_root, sid, im, inst_idx)
                        if inst_idx is not None else None)
                if full is None:
                    full = mask
                bbox_src = full
                rgb = load_rgb(bop_root, sid, im)
                bbox = mask_bbox(bbox_src)
                vf = (None if visib_index is None
                      else visib_index.get((sid, im, inst_idx)))
                verts_mm = np.asarray(meshes[oid].vertices, float)
                # ── visible-mask source + gate visib_fract (T-089) ──
                # 'gt'  : BOP mask_visib + GT visib_fract  (T-085 upper bound)
                # 'sam' : POSE-INDEPENDENT SAM segment of the detector box (full-
                #         mask AABB carries no pose) + GT-scale proxy visib_fract
                #         (SAM-px / coarse-silhouette-px) = exact live behaviour.
                vis_mask = mask
                gate_vf = vf
                if proxy_mask == "sam" and visaware:
                    pm = _sam_proxy_mask(rgb, bbox, full.shape)
                    if pm is None:
                        stats["proxy_fail"] += 1
                        # no live proxy -> live path would NOT refine -> passthrough
                        R_world, t_world, _ = A.planar_refine(
                            R_world, t_world,
                            np.asarray(meshes[oid].vertices, float) / 1000.0,
                            table_z=table_z, z_snap=True,
                            max_snap_m=A.DEFAULT_MAX_SNAP_M)
                        R_m2c, t_m2c = world_to_cam(R_world, t_world, R_w2c, t_w2c)
                        out.append({**r, "R": R_m2c, "t": t_m2c})
                        stats["n"] += 1
                        continue
                    vis_mask = pm
                    # GT-scale proxy_vf = SAM-px / rendered-coarse-silhouette-px
                    csil = RC.render_silhouette(
                        verts_mm, R_world, t_world, TABLE_ORIGIN,
                        R_w2c, t_w2c, K, full.shape)
                    csum = int(np.asarray(csil, bool).sum())
                    gate_vf = (float(int(pm.sum()) / csum) if csum > 0 else vf)
                    # T-089 calibration: remap proxy gate_vf onto GT visib_fract
                    # scale so the staggered occluded band [0.20,0.50) can fire.
                    if gate_vf is not None:
                        gate_vf = gate_vf * float(proxy_vf_scale)
                        if proxy_vf_clip:
                            gate_vf = float(min(1.0, max(0.0, gate_vf)))
                target = vis_mask if visaware else full
                ie = None
                if rgb is not None and bbox is not None:
                    crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    ie = RC.image_edges(crop, bbox=bbox, full_hw=bbox_src.shape)
                # free-space-Maske (TEIL-AGNOSTISCH) für die Refutation.
                fs = None
                if freespace and inst_idx is not None:
                    occluded = full & ~vis_mask
                    n_insts = n_insts_in_frame(bop_root, sid, im)
                    others = other_full_masks(bop_root, sid, im, inst_idx, n_insts)
                    fs = RC.build_free_space_mask(
                        full.shape, vis_mask, occluded_mask=occluded,
                        other_masks=others, dilate=occ_dilate)
                kw = dict(
                    verts_mm=verts_mm, t_world_m=t_world,
                    table_origin_m=TABLE_ORIGIN, R_w2c=R_w2c, t_w2c_mm=t_w2c,
                    K=K, hw=full.shape, target_mask=target, image_edge_mask=ie,
                    sym_axis=SYM_AXIS, n_fold=OBJ_NFOLD.get(oid),
                    stable_downs=downs[oid], scorer=scorer)
                if visaware:
                    kw.update(visib_mask=vis_mask, visib_fract=gate_vf,
                              min_visib_fract=min_visib_fract,
                              margin_schedule=sched)
                if fs is not None:
                    kw.update(free_space_mask=fs,
                              max_free_space_violation=occ_max_violation,
                              free_space_vf_hi=(None if occ_vf_hi <= 0 else occ_vf_hi))
                R_new, info = RC.refine_detection(R_world, **kw)
                stats["rc"] += 1
                if info.get("switched"):
                    stats["switched"] += 1
                stats["refuted"] += int(info.get("n_refuted", 0))
                R_world = R_new
        # Z-Snap (Baseline-Default) immer mit anwenden (wie die geshippte Pipeline).
        R_world, t_world, _ = A.planar_refine(
            R_world, t_world, np.asarray(meshes[oid].vertices, float) / 1000.0,
            table_z=table_z, z_snap=True, max_snap_m=A.DEFAULT_MAX_SNAP_M)
        R_m2c, t_m2c = world_to_cam(R_world, t_world, R_w2c, t_w2c)
        out.append({**r, "R": R_m2c, "t": t_m2c})
        stats["n"] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser(description="M2 RC-refiner (cpu_edge) -> eval CSVs")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", required=True, help="combined BOP-results CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z)
    ap.add_argument("--scorer", default="cpu_edge",
                    choices=["cpu_edge", "megapose"],
                    help="hypothesis scorer for M2 (cpu_edge=fast/silhouette, megapose=GPU/RGB)")
    ap.add_argument("--min-visib-fract", type=float, default=0.25)
    ap.add_argument("--occ-max-violation", type=float, default=0.30,
                    help="free-space-refutation threshold (T-092)")
    ap.add_argument("--occ-dilate", type=int, default=2)
    ap.add_argument("--occ-vf-hi", type=float, default=0.0,
                    help="band gate: refute only below this visib_fract (0=off)")
    ap.add_argument("--proxy-mask", choices=["gt", "sam"], default="gt",
                    help="visible-mask source for vis-aware configs (T-089). "
                         "'gt'=BOP mask_visib (upper bound), 'sam'=POSE-INDEPENDENT "
                         "SAM segment of the detector box (live proxy).")
    ap.add_argument("--use-schedule", action="store_true",
                    help="apply the SHIPPED visibility-staggered margin schedule "
                         "(refine_rc.DEFAULT_MARGIN_SCHEDULE = T-085 v2 = what "
                         "actually ships) instead of a flat min_margin.")
    ap.add_argument("--proxy-vf-scale", type=float, default=1.0,
                    help="T-089: linear scale applied to the SAM-proxy gate_vf so "
                         "it lands on the GT visib_fract scale (calibrated ~0.384). "
                         "1.0 = uncalibrated (today, inert). sam-path only.")
    ap.add_argument("--proxy-vf-clip", action="store_true",
                    help="T-089: clamp the (scaled) SAM-proxy gate_vf to [0,1].")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cams = load_cams(args.bop_root)
    meshes, downs = load_meshes(args.bop_root)
    gt_index = load_gt_index(args.bop_root)
    visib_index = load_visib_index(args.bop_root)
    rows = read_preds(args.preds)

    for cname in args.config:
        cfg = CONFIGS[cname]
        ref, st = refine_rows(rows, cfg, cams, meshes, downs, gt_index,
                              args.bop_root, table_z=args.table_z,
                              scorer=args.scorer, visib_index=visib_index,
                              min_visib_fract=args.min_visib_fract,
                              occ_max_violation=args.occ_max_violation,
                              occ_dilate=args.occ_dilate, occ_vf_hi=args.occ_vf_hi,
                              proxy_mask=args.proxy_mask,
                              use_schedule=args.use_schedule,
                              proxy_vf_scale=args.proxy_vf_scale,
                              proxy_vf_clip=args.proxy_vf_clip)
        tag = f"{cname}_{args.proxy_mask}" if cfg.get("visaware") else cname
        if args.use_schedule and cfg.get("visaware"):
            tag += "_sched"
        out = os.path.join(args.out_dir, f"preds_{tag}.csv")
        write_csv(ref, out)
        sys.stderr.write(
            f"[rc] {tag:20s} -> {out}  (n={st['n']} rc={st['rc']} "
            f"switched={st['switched']} refuted={st['refuted']} "
            f"no_mask={st['no_mask']} proxy_fail={st['proxy_fail']})\n")


if __name__ == "__main__":
    main()
