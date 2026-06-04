#!/usr/bin/env python3
"""t122_segmask_visaware.py — KERN-EXPERIMENT T-122 (board T-116).

FRAGE (image-only, kein GT in der Maske):
  Der vis-aware refine_rc (T-085) holt offline mit GT `mask_visib` ~40% der
  partiellen Anker-Flips zurück (report-split partial flip 7.52%->2.26%, fix7/broke0,
  well-vis 6.29% unverändert). LIVE gibt es kein GT `mask_visib` — der OBB-Detektor
  liefert nur eine Box, kip_server schreibt dummy 40x40-Masken. Der EINZIGE non-
  retrain-Hebel ist: eine echte visible-Maske per Segmentierung (FastSAM) auf dem
  Crop beschaffen und damit den vis-aware Score füttern.

  TRÄGT das? D.h.: ersetzt man im T-085-Pfad GT `mask_visib` durch eine FastSAM-
  Maske (box-prompted auf dem RGB-Crop, NUR Bild, kein GT), überlebt dann der Flip-
  Recovery-Gain — OHNE well-vis/AR zu regredieren?

WAS ES MISST (read-only val, kein Retrain):
  1. MASKEN-QUALITÄT: FastSAM-Maske vs GT mask_visib (IoU, recall, precision),
     stratifiziert nach visib_fract-Band. Schlechte Maske im partial-Band = der
     Hebel kann gar nicht greifen.
  2. FLIP-RECOVERY: die exakte T-085-Sweep-Logik (visibility-gestaffeltes Margin-
     Gate, scorer=cpu_edge), aber visib_mask = SEG-Maske statt GT. visib_fract aus
     der Seg-Maske abgeleitet (seg-Pixel / Hypothesen-coarse-Silhouette), NICHT GT
     (= live verfügbar). Vergleich: before (raw) vs after (seg-vis-aware) für
     partial[0.2,0.6) und well[0.6,0.8). Plus die GT-Maske als oberer Referenzwert.

  Wir scoren das EHRLICHE Verdikt: greift es mit Seg-Maske so wie mit GT, oder
  kollabiert der Gain? -> deploy ODER STOP.

USAGE (Box):
  # FastSAM braucht ultralytics -> train-venv. refine_rc braucht trimesh -> bop-venv.
  # Lösung: train-venv segmentiert -> cached Masken als .npz; bop-venv refined.
  # Dieses Script kann BEIDE Phasen, je nach --phase.

  # Phase 1 (train-venv): FastSAM-Masken für alle Anker-Preds cachen.
  /mnt/data/train-venv/bin/python box_src/t122_segmask_visaware.py --phase seg \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds-dir /mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO \
    --seg-cache /tmp/t122_seg.npz --fastsam-weights /tmp/FastSAM-s.pt

  # Phase 2 (bop-venv): mask-quality + vis-aware sweep mit Seg-Maske.
  /mnt/data/bop/bop-venv/bin/python box_src/t122_segmask_visaware.py --phase refine \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds-dir /mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO \
    --seg-cache /tmp/t122_seg.npz --out /tmp/t122_result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "project"))
sys.path.insert(0, HERE)

TABLE_ORIGIN = np.zeros(3)
SYM_AXIS = (0.0, 1.0, 0.0)
OBJ_NFOLD = {1: None, 2: None, 6: 7}
DEFAULT_TABLE_Z = 0.018
ANKER_OBJS = (1, 2)
FLIP_THR_DEG = 90.0
# T-085-Kalibrier-Optimum (DEFAULT_MARGIN_SCHEDULE in refine_rc):
SCHEDULE = {"vf_occ_lo": 0.20, "vf_occ_hi": 0.50, "margin_occ": 0.02, "margin_well": 0.15}
PARTIAL_BAND = (0.20, 0.60)
WELL_BAND = (0.60, 0.80)


# ──────────────────────────────────────────────────────────────────────────────
# gemeinsame Helfer (read-only val)
# ──────────────────────────────────────────────────────────────────────────────
def load_cams(bop_root):
    cams = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        cams[sid] = json.load(open(os.path.join(sc, "scene_camera.json")))
    return cams


def load_gt_index(bop_root):
    """(scene,im,obj_id) -> [(inst_idx, cam_t_m2c, cam_R_m2c)]."""
    idx = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        for im, insts in gt.items():
            for i, inst in enumerate(insts):
                key = (sid, int(im), int(inst["obj_id"]))
                idx.setdefault(key, []).append(
                    (i, np.array(inst["cam_t_m2c"], float),
                     np.array(inst["cam_R_m2c"], float).reshape(3, 3)))
    return idx


def load_visib_index(bop_root):
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


def match_gt_inst(gt_index, sid, im, obj_id, t_m2c, used):
    cands = gt_index.get((sid, im, obj_id), [])
    best, bd = None, None
    for i, tg, _Rg in cands:
        if (sid, im, obj_id, i) in used:
            continue
        d = float(np.linalg.norm(t_m2c - tg))
        if bd is None or d < bd:
            bd, best = d, i
    if best is not None:
        used.add((sid, im, obj_id, best))
    return best


def load_mask_visib(bop_root, sid, im, inst):
    p = os.path.join(bop_root, "val", f"{sid:06d}", "mask_visib", f"{im:06d}_{inst:06d}.png")
    return (np.array(Image.open(p)) > 0) if os.path.isfile(p) else None


def load_full_mask(bop_root, sid, im, inst):
    p = os.path.join(bop_root, "val", f"{sid:06d}", "mask", f"{im:06d}_{inst:06d}.png")
    return (np.array(Image.open(p)) > 0) if os.path.isfile(p) else None


def load_rgb(bop_root, sid, im):
    p = os.path.join(bop_root, "val", f"{sid:06d}", "rgb", f"{im:06d}.png")
    return np.asarray(Image.open(p).convert("RGB")) if os.path.isfile(p) else None


def mask_bbox(mask, pad=6):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W, int(xs.max()) + pad), min(H, int(ys.max()) + pad)]


def read_preds(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "scene_id": int(r["scene_id"]), "im_id": int(r["im_id"]),
                "obj_id": int(r["obj_id"]), "score": float(r["score"]),
                "R": np.array([float(x) for x in r["R"].split()]).reshape(3, 3),
                "t": np.array([float(x) for x in r["t"].split()]),
            })
    return rows


def collect_anker_rows(preds_dir):
    rows = []
    for slug in ("anker_kurz", "anker_lang"):
        p = os.path.join(preds_dir, slug, "preds_best.csv")
        if os.path.isfile(p):
            rows.extend(read_preds(p))
    return [r for r in rows if r["obj_id"] in ANKER_OBJS]


def key_of(r, inst):
    return f"{r['scene_id']}/{r['im_id']}/{r['obj_id']}/{inst}"


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: FastSAM-Segmentierung (train-venv)
# ──────────────────────────────────────────────────────────────────────────────
def phase_seg(args):
    """FastSAM box-prompted auf jedem Anker-Crop -> Seg-Maske (Vollbild-bool, RLE
    in .npz). Box = AABB der GT-vollen-Maske (Proxy für die Detektor-Box; live
    käme sie aus dem OBB-Detektor). NUR das RGB-Bild geht in FastSAM (image-only).
    """
    from ultralytics import FastSAM
    bop = args.bop_root
    gt_index = load_gt_index(bop)
    rows = collect_anker_rows(args.preds_dir)
    model = FastSAM(args.fastsam_weights)
    used = {}
    cache = {}
    rgb_cache = {}
    n = 0
    for r in rows:
        sid, im, oid = r["scene_id"], r["im_id"], r["obj_id"]
        u = used.setdefault((sid, im), set())
        inst = match_gt_inst(gt_index, sid, im, oid, r["t"], u)
        if inst is None:
            continue
        # Crop-Box = AABB der echten vollen Maske (Proxy für Detektor-Box).
        full = load_full_mask(bop, sid, im, inst)
        if full is None:
            continue
        bbox = mask_bbox(full, pad=6)   # [x0,y0,x1,y1]
        if bbox is None:
            continue
        rk = (sid, im)
        if rk not in rgb_cache:
            rgb_cache[rk] = load_rgb(bop, sid, im)
        rgb = rgb_cache[rk]
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        # FastSAM box-prompt: gib ihm die ganze RGB + bbox als Prompt.
        try:
            res = model(rgb, bboxes=[bbox], imgsz=max(H, W), conf=0.25, iou=0.7,
                        device=0, verbose=False, retina_masks=True)
        except TypeError:
            # ältere/neuere API: predict + prompt-process
            res = model.predict(rgb, imgsz=max(H, W), conf=0.25, iou=0.7,
                                device=0, verbose=False, retina_masks=True)
        seg = _extract_box_mask(res, bbox, (H, W))
        if seg is None:
            seg = np.zeros((H, W), bool)
        # RLE-kompakt (nur die bbox-Region speichern + offset)
        x0, y0, x1, y1 = bbox
        sub = seg[y0:y1, x0:x1].astype(np.uint8)
        cache[key_of(r, inst)] = (np.packbits(sub.ravel()), sub.shape, (x0, y0, H, W))
        n += 1
        if n % 200 == 0:
            print(f"[seg] {n} masks ...", flush=True)
    # speichern
    keys = list(cache.keys())
    np.savez_compressed(
        args.seg_cache,
        keys=np.array(keys),
        **{f"p_{i}": cache[k][0] for i, k in enumerate(keys)},
        **{f"s_{i}": np.array(cache[k][1]) for i, k in enumerate(keys)},
        **{f"o_{i}": np.array(cache[k][2]) for i, k in enumerate(keys)},
    )
    print(f"[seg] DONE n={n} -> {args.seg_cache}", flush=True)


def _extract_box_mask(res, bbox, hw):
    """Wähle aus den FastSAM-Masken die mit der höchsten IoU zur Prompt-Box."""
    if res is None:
        return None
    r0 = res[0] if isinstance(res, (list, tuple)) else res
    masks = getattr(r0, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return None
    data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else np.asarray(masks.data)
    H, W = hw
    x0, y0, x1, y1 = bbox
    boxarea = max(1, (x1 - x0) * (y1 - y0))
    best, best_score = None, -1.0
    for m in data:
        mm = m > 0.5
        if mm.shape != (H, W):
            mm = np.array(Image.fromarray(mm.astype(np.uint8) * 255).resize(
                (W, H), Image.NEAREST)) > 127
        # IoU mit der Prompt-Box (box als Maske)
        inter = int(mm[y0:y1, x0:x1].sum())
        union = int(mm.sum()) + boxarea - inter
        score = inter / max(1, union)
        if score > best_score:
            best_score, best = score, mm
    return best


def load_seg_cache(path):
    z = np.load(path, allow_pickle=True)
    keys = [k if isinstance(k, str) else k.decode() for k in z["keys"]]
    out = {}
    for i, k in enumerate(keys):
        packed = z[f"p_{i}"]; shape = tuple(int(v) for v in z[f"s_{i}"])
        x0, y0, H, W = (int(v) for v in z[f"o_{i}"])
        sub = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        full = np.zeros((H, W), bool)
        full[y0:y0 + shape[0], x0:x0 + shape[1]] = sub
        out[k] = full
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: mask-quality + vis-aware sweep mit Seg-Maske (bop-venv)
# ──────────────────────────────────────────────────────────────────────────────
def _mask_iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else 1.0


def _cont_y_syms(n=180):
    """Stack der kontinuierlichen Y-Sym-Rotationen (n,3,3) — vektorisiert."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    S = np.zeros((n, 3, 3))
    S[:, 0, 0] = c;  S[:, 0, 2] = s
    S[:, 1, 1] = 1.0
    S[:, 2, 0] = -s; S[:, 2, 2] = c
    return S


def _rot_err_sym_contY(R_est_cam, R_gt_cam, syms):
    """sym-resolved cont-Y rot err = min über re(R_est, R_gt@S). VEKTORISIERT:
    trace(R_est @ (R_gt@S).T) = trace(R_est @ S.T @ R_gt.T). Identisch zu
    anker_flip_repro.rot_err_sym_contY, nur ohne Python-Schleife."""
    M = R_est_cam @ np.transpose(R_gt_cam @ syms, (0, 2, 1))   # (n,3,3)
    tr = np.einsum("nii->n", M)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).min())


def phase_refine(args):
    """FAITHFUL zum validierten T-085-Pfad (anker_flip_repro.visaware_sweep):
    cache pro Instanz die per-Hyp vis-aware-Scores + per-Hyp rot_sym EINMAL (ein
    Render-Pass), dann das gestaffelte Margin-Gate ANALYTISCH anwenden.

    DREI Score-Varianten gecacht je Instanz:
      • gt  : visib_mask=GT mask_visib, gate-vf = GT visib_fract       (obere Referenz)
      • seg : visib_mask=FastSAM-Maske, gate-vf = seg-abgeleitet       (LIVE-realistisch)
    'raw' = Coarse (Hyp-Index 0, kein Switch).

    KORREKTUR ggü. v1 (false-negative): min_margin=0.0 (globaler Hard-Floor wie im
    validierten Sweep — sonst floored DEFAULT 0.15 das margin_occ=0.02 weg → 0 fix
    sogar mit GT-Maske). gate-vf für GT = echtes scene_gt_info visib_fract.
    """
    import bop_adapter as A
    import refine_rc as RC
    import trimesh

    bop = args.bop_root
    cams = load_cams(bop)
    gt_index = load_gt_index(bop)
    visib_index = load_visib_index(bop)
    seg = load_seg_cache(args.seg_cache)
    syms = _cont_y_syms(180)
    print(f"[refine] seg-cache: {len(seg)} masks", flush=True)

    md = os.path.join(bop, "models_eval")
    if not os.path.isdir(md):
        md = os.path.join(bop, "models")
    meshes, downs = {}, {}
    for oid in ANKER_OBJS:
        m = trimesh.load(os.path.join(md, f"obj_{oid:06d}.ply"), process=False)
        meshes[oid] = m
        d, _ = A.stable_pose_body_downs(mesh=m, prob_min=0.02, max_k=6, cache_key=oid)
        downs[oid] = d

    rows = collect_anker_rows(args.preds_dir)

    # ── PASS 1 (BILLIG, kein Render): matche alle Preds zu GT, bestimme den Flip-
    # Status der COARSE (hyp_rs[0] braucht nur die Coarse-Rotation, kein Rendering),
    # sammle die Inputs. Flip-before-Instanzen sind selten + entscheidungskritisch
    # (fix/broke) → ALLE behalten; non-flip nur fürs Banding-Denominator → cappbar.
    used = {}
    candidates = []   # dict je matched Instanz inkl. flip_before + Maske-Refs
    for r in rows:
        sid, im, oid = r["scene_id"], r["im_id"], r["obj_id"]
        u = used.setdefault((sid, im), set())
        inst = match_gt_inst(gt_index, sid, im, oid, r["t"], u)
        if inst is None:
            continue
        gt_vf = visib_index.get((sid, im, inst), -1.0)
        if gt_vf < 0:
            continue
        band = ("partial" if PARTIAL_BAND[0] <= gt_vf < PARTIAL_BAND[1]
                else "well" if WELL_BAND[0] <= gt_vf < WELL_BAND[1] else "other")
        gtR = t_gt = None
        for i2, t2, R2 in gt_index.get((sid, im, oid), []):
            if i2 == inst:
                gtR = R2; t_gt = t2; break
        if gtR is None:
            continue
        cam = cams[sid][str(im)]
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        # COARSE-Flip-Status (billig): pred-cam vs GT-cam, sym-resolved.
        flip_before = _rot_err_sym_contY(R_w2c @ r["R"], R_w2c @ gtR, syms) > FLIP_THR_DEG
        candidates.append(dict(r=r, sid=sid, im=im, oid=oid, inst=inst, gt_vf=gt_vf,
                               band=band, gtR=gtR, t_gt=t_gt, flip_before=flip_before))

    # Cap pro Band (NUR non-flip subsamplen; ALLE flips behalten). cap<=0 = alles.
    cap = int(getattr(args, "cap", 0) or 0)
    chosen = []
    for b in ("partial", "well"):
        bandc = [c for c in candidates if c["band"] == b]
        flips = [c for c in bandc if c["flip_before"]]
        nonflips = [c for c in bandc if not c["flip_before"]]
        if cap > 0 and len(nonflips) > cap:
            nonflips = nonflips[:cap]
        chosen.extend(flips + nonflips)
    print(f"[refine] candidates={len(candidates)} chosen={len(chosen)} "
          f"(cap_per_band_nonflip={cap}); "
          f"flips partial={sum(c['flip_before'] and c['band']=='partial' for c in candidates)} "
          f"well={sum(c['flip_before'] and c['band']=='well' for c in candidates)}", flush=True)

    cached = []
    mask_q = []   # (band, iou, recall, prec, gt_vf, seg_npx)
    rgb_cache = {}
    n_seen = 0
    for c in chosen:
        sid, im, oid, inst = c["sid"], c["im"], c["oid"], c["inst"]
        r = c["r"]; gt_vf = c["gt_vf"]; gtR = c["gtR"]; t_gt = c["t_gt"]
        cam = cams[sid][str(im)]
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        gtmask = load_mask_visib(bop, sid, im, inst)
        full = load_full_mask(bop, sid, im, inst)
        segmask = seg.get(key_of(r, inst))
        if gtmask is None or full is None or segmask is None:
            continue

        # Masken-Qualität (seg vs GT)
        inter = int((segmask & gtmask).sum())
        rec = inter / max(1, int(gtmask.sum()))
        prec = inter / max(1, int(segmask.sum()))
        band = ("partial" if PARTIAL_BAND[0] <= gt_vf < PARTIAL_BAND[1]
                else "well" if WELL_BAND[0] <= gt_vf < WELL_BAND[1] else "other")
        mask_q.append((band, _mask_iou(segmask, gtmask), rec, prec, gt_vf,
                       int(segmask.sum())))

        # ── Render-Pass EINMAL: Coarse-Welt-Pose (mit GT-t wie validierter Pfad) ──
        rk = (sid, im)
        if rk not in rgb_cache:
            rgb_cache[rk] = load_rgb(bop, sid, im)
        rgb = rgb_cache[rk]
        bbox = mask_bbox(full)
        ie = None
        if rgb is not None and bbox is not None:
            crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            ie = RC.image_edges(crop, bbox=bbox, full_hw=full.shape)
        R_world, t_world = A.bop_pose_to_world(r["R"], t_gt, R_w2c, t_w2c, TABLE_ORIGIN)
        verts_mm = np.asarray(meshes[oid].vertices, float)
        n_fold = OBJ_NFOLD.get(oid)
        hyps, tags = RC.generate_hypotheses(
            R_world, sym_axis=SYM_AXIS, n_fold=n_fold, stable_downs=downs[oid])
        hyps = np.asarray(hyps, float)

        # per-Hyp Scores: GT-Maske-Variante und Seg-Maske-Variante.
        scores_gt, _ = RC.cpu_edge_score(
            hyps, verts_mm=verts_mm, t_world_m=t_world, table_origin_m=TABLE_ORIGIN,
            R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=full.shape,
            target_mask=gtmask, image_edge_mask=ie, visib_mask=gtmask, visib_dilate=2)
        scores_seg, _ = RC.cpu_edge_score(
            hyps, verts_mm=verts_mm, t_world_m=t_world, table_origin_m=TABLE_ORIGIN,
            R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=full.shape,
            target_mask=segmask, image_edge_mask=ie, visib_mask=segmask, visib_dilate=2)
        # per-Hyp rot_sym (Coarse = Index 0).
        hyp_rs = [_rot_err_sym_contY(R_w2c @ h, gtR, syms) for h in hyps]

        # seg-abgeleitetes visib_fract fürs Gate (LIVE-verfügbar): seg-Pixel über die
        # Coarse-Silhouette / Coarse-Silhouette (Index 0 = Coarse-Hyp).
        sil0 = RC.render_silhouette(verts_mm, hyps[0], t_world, TABLE_ORIGIN,
                                    R_w2c, t_w2c, K, full.shape)
        sil_a = int(sil0.sum())
        seg_vf = (int((segmask & sil0).sum()) / sil_a) if sil_a else 0.0

        cached.append({
            "scene_id": sid, "im_id": im, "inst": inst, "obj_id": oid,
            "gt_vf": float(gt_vf), "seg_vf": float(seg_vf),
            "scores_gt": [float(s) for s in scores_gt],
            "scores_seg": [float(s) for s in scores_seg],
            "hyp_rs": [float(x) for x in hyp_rs],
        })
        n_seen += 1
        if n_seen % 300 == 0:
            print(f"[refine] cached {n_seen} ...", flush=True)

    print(f"[refine] matched+cached anker instances: {len(cached)}", flush=True)

    # Mask-quality-Summary
    def _q(band):
        sel = [m for m in mask_q if m[0] == band]
        if not sel:
            return {"n": 0}
        iou = np.array([m[1] for m in sel]); rec = np.array([m[2] for m in sel])
        prec = np.array([m[3] for m in sel])
        return {"n": len(sel), "iou_median": float(np.median(iou)),
                "iou_mean": float(iou.mean()), "recall_median": float(np.median(rec)),
                "prec_median": float(np.median(prec)),
                "frac_iou_lt_0.3": float((iou < 0.3).mean()),
                "frac_empty": float((np.array([m[5] for m in sel]) == 0).mean())}
    mask_quality = {b: _q(b) for b in ("partial", "well", "other")}

    # ── analytischer Sweep: gestaffeltes Gate auf gecachte Scores anwenden ──
    MIN_MARGIN = 0.0          # globaler Hard-Floor (= validierter Sweep, NICHT 0.15!)
    MIN_VF = 0.25
    thr = FLIP_THR_DEG

    def eval_variant(score_key, vf_key, use_schedule):
        """score_key in {scores_gt,scores_seg}; vf_key in {gt_vf,seg_vf}.
        use_schedule=False -> raw (Coarse, kein Gate). Liefert pro Band Listen
        flip-before/flip-after (bool) für fix/broke + flip-rate."""
        per_band = {"partial": {"bef": [], "aft": []}, "well": {"bef": [], "aft": []}}
        for c in cached:
            gv = c["gt_vf"]
            band = ("partial" if PARTIAL_BAND[0] <= gv < PARTIAL_BAND[1]
                    else "well" if WELL_BAND[0] <= gv < WELL_BAND[1] else None)
            if band is None:
                continue
            rs_coarse = c["hyp_rs"][0]
            flip_before = rs_coarse > thr
            if not use_schedule:
                flip_after = flip_before
            else:
                scores = np.asarray(c[score_key], float)
                best_idx, _sel = RC.select_best_hypothesis(
                    scores, scores, coarse_idx=0, min_margin=MIN_MARGIN,
                    visib_fract=c[vf_key], min_visib_fract=MIN_VF,
                    margin_schedule=SCHEDULE)
                flip_after = c["hyp_rs"][best_idx] > thr
            per_band[band]["bef"].append(flip_before)
            per_band[band]["aft"].append(flip_after)
        out = {}
        for b in ("partial", "well"):
            bef = np.array(per_band[b]["bef"], bool); aft = np.array(per_band[b]["aft"], bool)
            out[b] = {
                "n": len(bef),
                "flip_rate_before": float(bef.mean()) if len(bef) else 0.0,
                "flip_rate_after": float(aft.mean()) if len(aft) else 0.0,
                "n_flip_before": int(bef.sum()), "n_flip_after": int(aft.sum()),
                "n_fixed": int((bef & ~aft).sum()), "n_broke": int((~bef & aft).sum()),
            }
        return out

    raw = eval_variant("scores_gt", "gt_vf", use_schedule=False)
    gt = eval_variant("scores_gt", "gt_vf", use_schedule=True)         # obere Referenz
    seg = eval_variant("scores_seg", "seg_vf", use_schedule=True)      # LIVE-realistisch
    # Kontroll-Variante: seg-Score aber GT-vf fürs Gate (isoliert Score-Qualität von
    # vf-Qualität).
    seg_gtvf = eval_variant("scores_seg", "gt_vf", use_schedule=True)

    # WAHRE Band-Populationen (alle matched candidates, kein Cap) + wahre flip-before.
    true_band = {}
    for b in ("partial", "well"):
        bc = [c for c in candidates if c["band"] == b]
        nf = sum(c["flip_before"] for c in bc)
        true_band[b] = {"n_total": len(bc), "n_flip_before": nf,
                        "flip_rate_before": (nf / len(bc)) if bc else 0.0}

    result = {
        "n_matched": len(cached),
        "n_candidates_total": len(candidates),
        "cap_per_band_nonflip": cap,
        "true_band_populations": true_band,
        "schedule": SCHEDULE,
        "min_margin_hard_floor": MIN_MARGIN,
        "min_visib_fract": MIN_VF,
        "bands": {"partial": list(PARTIAL_BAND), "well": list(WELL_BAND)},
        "mask_quality_segVSgt": mask_quality,
        "flip": {
            "raw_coarse": raw,
            "after_gt_reference": gt,
            "after_seg_live": seg,
            "after_seg_score_gt_vf_gate": seg_gtvf,
        },
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[refine] DONE -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["seg", "refine"])
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--seg-cache", required=True)
    ap.add_argument("--fastsam-weights", default="/tmp/FastSAM-s.pt")
    ap.add_argument("--out", default="/tmp/t122_result.json")
    ap.add_argument("--cap", type=int, default=0,
                    help="refine: cap NON-FLIP instances per band (0=all); ALL "
                         "flip-before instances always kept (decision-critical).")
    args = ap.parse_args()
    if args.phase == "seg":
        phase_seg(args)
    else:
        phase_refine(args)


if __name__ == "__main__":
    main()
