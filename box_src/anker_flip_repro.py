#!/usr/bin/env python3
"""anker_flip_repro.py — DETERMINISTIC repro + measurement harness for the
Anker 180-degree end-to-end flip at PARTIAL visibility (Session 2026-06-01,
S-001 / T-083, Theo).

THE QUESTION (read-only, no training, no GPU):
  GDRNPP predicts the Anker (obj 1 Anker_Kurz, obj 2 Anker_Lang) rotated ~180deg
  about a TRANSVERSE axis (head<->tail swapped) much more often when the part is
  only PARTIALLY visible (shaft/tip occluded or cropped) than when fully visible.
  We prove that on the REAL GDRNPP val predictions (preds_best.csv), stratified by
  the GT instance visib_fract from scene_gt_info.json.

WHAT IT MEASURES (per Anker instance, matched pred<->GT by translation):
  1. rot_err NAIVE          : raw geodesic angle pred-vs-GT (deg).
  2. rot_err SYM-RESOLVED   : continuous-Y symmetry resolved (the eval_bop metric).
                              -> does the declared cont-Y symmetry FORGIVE the
                              cross-flip? (load-bearing for S-004 metric audit).
  3. flip residual          : after removing the best cont-Y twist, how much of the
                              residual is a ~180deg rotation about a TRANSVERSE
                              axis? -> isolates the genuine flip from a Y-twist.
  4. FLIP-RATE              : fraction of instances with sym-resolved rot_err > 90deg
                              (a cross-flip survives the symmetry), full-vis vs
                              partial-vis. THE number S-004 needs as before-baseline.

It ALSO proves the metric behaviour analytically (no GDRNPP needed):
  --metric-check : take each Anker GT pose, apply an EXACT 180deg transverse flip,
                   and measure naive vs sym-resolved rot_err. If sym-resolved stays
                   ~180deg the metric CATCHES the flip; if it collapses to ~0 the
                   metric FORGIVES it (the S-004 trap).

And it can probe the refiner (does cpu_edge break the flip?):
  --rc-probe N : for up to N full-vis + N partial-vis flipped instances, run
                 refine_rc.refine_detection(cpu_edge) and log the per-hypothesis
                 scores + whether the correct (non-flipped) hypothesis wins.

USAGE (on the box, bop-venv):
  /mnt/data/bop/bop-venv/bin/python box_src/anker_flip_repro.py \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds-dir /mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO \
    --full-thr 0.95 --partial-lo 0.20 --partial-hi 0.60 \
    --metric-check --rc-probe 6 \
    --out /mnt/data/bop/logs/anker_flip_repro.json

Self-contained: reuses bop_adapter + refine_rc + the rc_refine_eval helpers. The
match is identical to eval_bop (greedy nearest-translation, same obj_id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))
sys.path.insert(0, os.path.dirname(__file__))

import bop_adapter as A          # noqa: E402
import refine_rc as RC           # noqa: E402
import rc_refine_eval as RR      # noqa: E402 (reuse read_preds/load_cams/match/...)

try:
    from bop_toolkit_lib import misc, pose_error
    HAVE_BOP = True
except Exception:                # pragma: no cover - fall back to local geodesic
    HAVE_BOP = False

ANKER_OBJS = (1, 2)
ANKER_NAME = {1: "Anker_Kurz", 2: "Anker_Lang"}
SYM_AXIS = np.array([0.0, 1.0, 0.0])      # cont-Y symmetry axis (models_info)
TRANSVERSE = {"X": np.array([1.0, 0.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
FLIP_THR_DEG = 90.0                       # sym-resolved rot > this = a surviving flip


# ── rotation helpers (cam-frame; pred/GT are model->cam) ─────────────────────
def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_err_naive(R_est, R_gt):
    if HAVE_BOP:
        return float(pose_error.re(R_est, R_gt))
    return geodesic_deg(R_est, R_gt)


def cont_y_syms(n_steps=180):
    """Continuous-Y symmetry transforms (model frame), discretised like
    bop_toolkit's get_symmetry_transformations does for continuous parts."""
    syms = []
    for k in range(n_steps):
        th = 2.0 * np.pi * k / n_steps
        syms.append(A.axis_angle_matrix(SYM_AXIS, th))
    return syms


def rot_err_sym_contY(R_est, R_gt, syms):
    """Sym-resolved rot err under continuous-Y (pick closest GT@R_y(th))."""
    best = None
    for S in syms:
        e = rot_err_naive(R_est, R_gt @ S)
        if best is None or e < best:
            best = e
    return float(best)


def flip_residual_deg(R_est, R_gt, syms):
    """After the best cont-Y twist, how 'flipped about a transverse axis' is the
    residual? Returns (residual_geodesic_deg, best_transverse_axis, twist_deg).

    residual = R_gt_best^T @ R_est, where R_gt_best is the cont-Y representative
    closest to R_est. A pure cross-flip leaves residual ~= 180deg about X or Z.
    """
    best_e, best_S = None, np.eye(3)
    for S in syms:
        e = rot_err_naive(R_est, R_gt @ S)
        if best_e is None or e < best_e:
            best_e, best_S = e, S
    R_gt_best = R_gt @ best_S
    R_res = R_gt_best.T @ R_est                 # residual rotation (model frame)
    res_deg = geodesic_deg(R_res, np.eye(3))
    # which transverse axis best explains the residual? project residual axis.
    # residual axis-angle:
    ang = np.arccos(np.clip((np.trace(R_res) - 1) / 2, -1, 1))
    if ang < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = np.array([R_res[2, 1] - R_res[1, 2],
                         R_res[0, 2] - R_res[2, 0],
                         R_res[1, 0] - R_res[0, 1]]) / (2 * np.sin(ang) + 1e-12)
    # transverse component = how much of |axis| is in the X-Z plane (not Y).
    transverse_frac = float(np.hypot(axis[0], axis[2]))
    best_t = "X" if abs(axis[0]) >= abs(axis[2]) else "Z"
    return res_deg, best_t, transverse_frac


# ── GT visib_fract index (aligned with scene_gt instance order) ──────────────
def load_visib_index(bop_root):
    """(scene, im, inst_idx) -> visib_fract, from scene_gt_info.json."""
    import glob
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


def gt_R_for_inst(bop_root):
    """(scene, im, inst_idx) -> (obj_id, R_gt_m2c, t_gt_m2c) from scene_gt.json."""
    import glob
    idx = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        gp = os.path.join(sc, "scene_gt.json")
        gt = json.load(open(gp))
        for im, insts in gt.items():
            for i, g in enumerate(insts):
                idx[(sid, int(im), i)] = (
                    int(g["obj_id"]),
                    np.array(g["cam_R_m2c"], float).reshape(3, 3),
                    np.array(g["cam_t_m2c"], float),
                )
    return idx


def find_preds(preds_dir):
    """Locate the per-object preds_best.csv for the Anker parts."""
    cand = {
        1: os.path.join(preds_dir, "anker_kurz", "preds_best.csv"),
        2: os.path.join(preds_dir, "anker_lang", "preds_best.csv"),
    }
    return {o: p for o, p in cand.items() if os.path.isfile(p)}


# ── metric forgiveness check (no GDRNPP) ─────────────────────────────────────
def metric_check(gt_index, syms, max_per_obj=200):
    """Take GT poses, apply EXACT 180deg transverse flip, measure naive vs
    sym-resolved rot err. If sym-resolved stays high the metric CATCHES the flip."""
    out = {}
    for axis_name, axis in TRANSVERSE.items():
        Rflip = A.axis_angle_matrix(axis, np.pi)
        for oid in ANKER_OBJS:
            rows = [(k, v) for k, v in gt_index.items() if v[0] == oid][:max_per_obj]
            naive, symv = [], []
            for _, (_, R_gt, _t) in rows:
                R_est = R_gt @ Rflip            # flip about transverse axis (model frame)
                naive.append(rot_err_naive(R_est, R_gt))
                symv.append(rot_err_sym_contY(R_est, R_gt, syms))
            if not naive:
                continue
            out.setdefault(ANKER_NAME[oid], {})[f"flip_{axis_name}"] = {
                "n": len(naive),
                "rot_naive_mean_deg": round(float(np.mean(naive)), 2),
                "rot_sym_mean_deg": round(float(np.mean(symv)), 2),
                "rot_sym_min_deg": round(float(np.min(symv)), 2),
                "rot_sym_max_deg": round(float(np.max(symv)), 2),
                "metric_catches_flip": bool(np.mean(symv) > FLIP_THR_DEG),
            }
    return out


def main():
    ap = argparse.ArgumentParser(description="Anker partial-vis flip repro (T-083)")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds-dir", required=True,
                    help="dir containing anker_kurz/ anker_lang/ preds_best.csv")
    ap.add_argument("--full-thr", type=float, default=0.95,
                    help="visib_fract >= this = full visibility")
    ap.add_argument("--partial-lo", type=float, default=0.20)
    ap.add_argument("--partial-hi", type=float, default=0.60)
    ap.add_argument("--flip-thr-deg", type=float, default=FLIP_THR_DEG)
    ap.add_argument("--metric-check", action="store_true")
    ap.add_argument("--rc-probe", type=int, default=0,
                    help="run refine_rc(cpu_edge) on up to N full + N partial flipped")
    ap.add_argument("--visaware-compare", type=int, default=0,
                    help="BEFORE vs AFTER visibility-aware refine_rc on the REAL "
                         "GDRNPP coarse, up to N partial + N full instances per "
                         "stratum. Measures flip-rate before/after (ADR-020/T-085).")
    ap.add_argument("--visaware-min-vf", type=float, default=0.25,
                    help="min_visib_fract for the visibility-conditioned gate")
    ap.add_argument("--visaware-min-margin", type=float, default=0.15,
                    help="min_margin (hard lower bound, ADR-020: >=0.15)")
    ap.add_argument("--visaware-all", action="store_true",
                    help="visaware-compare over ALL matched instances (ignore N cap)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    syms = cont_y_syms(180)
    visib = load_visib_index(args.bop_root)
    gt_inst = gt_R_for_inst(args.bop_root)
    gt_index = RR.load_gt_index(args.bop_root)      # for match (sid,im,obj)->[(i,t)]
    preds = find_preds(args.preds_dir)
    sys.stderr.write(f"[repro] preds found: {sorted(preds)}\n")

    # inst_idx lookup keyed by GT translation -> back to inst_idx for visib join.
    # RR.match_gt_inst returns inst_idx; we reuse it.
    per_inst = []     # collected measurements per matched Anker instance
    for oid, path in preds.items():
        rows = RR.read_preds(path)
        used = set()
        for r in rows:
            if r["obj_id"] != oid:
                continue
            sid, im = r["scene_id"], r["im_id"]
            inst_idx = RR.match_gt_inst(gt_index, sid, im, oid, r["t"], used)
            if inst_idx is None:
                continue
            vf = visib.get((sid, im, inst_idx), -1.0)
            gobj, R_gt, t_gt = gt_inst.get((sid, im, inst_idx), (None, None, None))
            if R_gt is None or gobj != oid:
                continue
            R_est = r["R"]
            rn = rot_err_naive(R_est, R_gt)
            rs = rot_err_sym_contY(R_est, R_gt, syms)
            res_deg, t_axis, t_frac = flip_residual_deg(R_est, R_gt, syms)
            per_inst.append({
                "obj_id": oid, "scene_id": sid, "im_id": im, "inst": inst_idx,
                "visib_fract": vf, "rot_naive": rn, "rot_sym": rs,
                "flip_resid_deg": res_deg, "flip_axis": t_axis,
                "transverse_frac": t_frac,
                "is_flip": bool(rs > args.flip_thr_deg),
                "R_est": R_est,         # REAL GDRNPP coarse (model->cam) for compare
            })

    sys.stderr.write(f"[repro] matched {len(per_inst)} Anker instances\n")

    def stratum(lo, hi):
        return [d for d in per_inst if lo <= d["visib_fract"] < hi]

    full = [d for d in per_inst if d["visib_fract"] >= args.full_thr]
    partial = stratum(args.partial_lo, args.partial_hi)

    def summarise(group, label):
        if not group:
            return {"label": label, "n": 0}
        rs = np.array([d["rot_sym"] for d in group])
        rn = np.array([d["rot_naive"] for d in group])
        nflip = sum(d["is_flip"] for d in group)
        # of the flipped ones, how many are genuine cross-flips (transverse)?
        flipped = [d for d in group if d["is_flip"]]
        tfrac = np.mean([d["transverse_frac"] for d in flipped]) if flipped else 0.0
        return {
            "label": label, "n": len(group),
            "flip_rate": round(nflip / len(group), 4),
            "n_flip": int(nflip),
            "rot_sym_median_deg": round(float(np.median(rs)), 2),
            "rot_sym_mean_deg": round(float(np.mean(rs)), 2),
            "rot_naive_median_deg": round(float(np.median(rn)), 2),
            "flipped_transverse_frac_mean": round(float(tfrac), 3),
        }

    result = {
        "preds_dir": args.preds_dir,
        "thresholds": {"full": args.full_thr, "partial_lo": args.partial_lo,
                       "partial_hi": args.partial_hi, "flip_deg": args.flip_thr_deg},
        "n_matched": len(per_inst),
        "full_vis": summarise(full, f"visib>={args.full_thr}"),
        "partial_vis": summarise(partial, f"{args.partial_lo}<=visib<{args.partial_hi}"),
        "per_obj": {},
        # mid band for completeness
        "mid_vis": summarise(stratum(args.partial_hi, args.full_thr),
                             f"{args.partial_hi}<=visib<{args.full_thr}"),
    }
    for oid in ANKER_OBJS:
        g = [d for d in per_inst if d["obj_id"] == oid]
        gf = [d for d in g if d["visib_fract"] >= args.full_thr]
        gp = [d for d in g if args.partial_lo <= d["visib_fract"] < args.partial_hi]
        result["per_obj"][ANKER_NAME[oid]] = {
            "full": summarise(gf, "full"), "partial": summarise(gp, "partial"),
        }

    if args.metric_check:
        result["metric_check"] = metric_check(gt_inst, syms)

    if args.rc_probe > 0:
        result["rc_probe"] = rc_probe(per_inst, args, syms)

    if args.visaware_compare > 0 or args.visaware_all:
        result["visaware_compare"] = visaware_compare(per_inst, args, syms)

    # human-readable to stderr
    sys.stderr.write("\n==================== ANKER FLIP REPRO ====================\n")
    for key in ("full_vis", "mid_vis", "partial_vis"):
        s = result[key]
        sys.stderr.write(
            f"  {s['label']:28s} n={s.get('n',0):4d}  flip_rate="
            f"{s.get('flip_rate','-'):>6}  flip={s.get('n_flip','-'):>3}  "
            f"rot_sym_med={s.get('rot_sym_median_deg','-'):>7}  "
            f"rot_naive_med={s.get('rot_naive_median_deg','-'):>7}\n")
    if "metric_check" in result:
        sys.stderr.write("\n  --- metric forgiveness (GT + exact 180deg transverse flip) ---\n")
        for part, axes in result["metric_check"].items():
            for ax, m in axes.items():
                sys.stderr.write(
                    f"  {part:12s} {ax:8s} naive={m['rot_naive_mean_deg']:6.1f}deg "
                    f"sym={m['rot_sym_mean_deg']:6.1f}deg  "
                    f"-> metric {'CATCHES' if m['metric_catches_flip'] else 'FORGIVES'} the flip\n")
    if "rc_probe" in result:
        rp = result["rc_probe"]
        sys.stderr.write("\n  --- refine_rc(cpu_edge) probe (does it break the flip?) ---\n")
        for grp in ("full", "partial"):
            b = rp.get(grp, {})
            sys.stderr.write(
                f"  {grp:8s} probed={b.get('n',0)}  "
                f"fixed(flip->ok)={b.get('n_fixed',0)}  "
                f"correct_hyp_wins={b.get('n_correct_wins',0)}  "
                f"switched={b.get('n_switched',0)}\n")
    if "visaware_compare" in result:
        vc = result["visaware_compare"]
        sys.stderr.write(
            "\n  --- visibility-aware refine_rc: BEFORE vs AFTER (real GDRNPP coarse) ---\n")
        sys.stderr.write(
            f"  gate: min_margin={vc['params']['min_margin']} "
            f"min_visib_fract={vc['params']['min_visib_fract']}\n")
        for grp in ("partial", "full"):
            b = vc.get(grp, {})
            if not b.get("n"):
                continue
            sys.stderr.write(
                f"  {grp:8s} n={b['n']:4d}  "
                f"flip_rate BEFORE={b['flip_rate_before']:.4f} ({b['n_flip_before']})  "
                f"AFTER={b['flip_rate_after']:.4f} ({b['n_flip_after']})  "
                f"fixed={b['n_fixed']}  broke={b['n_broke']}  "
                f"switched={b['n_switched_after']}\n")
    sys.stderr.write("==========================================================\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        sys.stderr.write(f"[repro] wrote {args.out}\n")
    print(json.dumps(result, indent=2))


def rc_probe(per_inst, args, syms):
    """Run refine_rc(cpu_edge) on a few flipped full-vis + partial-vis instances
    and report whether the refiner picks the correct (non-flipped) hypothesis."""
    import rc_refine_eval as RR
    cams = RR.load_cams(args.bop_root)
    meshes, downs = RR.load_meshes(args.bop_root)
    gt_inst = gt_R_for_inst(args.bop_root)

    flipped_full = [d for d in per_inst
                    if d["is_flip"] and d["visib_fract"] >= args.full_thr][:args.rc_probe]
    flipped_part = [d for d in per_inst
                    if d["is_flip"] and args.partial_lo <= d["visib_fract"] < args.partial_hi][:args.rc_probe]

    def probe(group):
        n = nfix = ncorrect = nsw = 0
        details = []
        for d in group:
            sid, im, oid, inst = d["scene_id"], d["im_id"], d["obj_id"], d["inst"]
            cam = cams[sid][str(im)]
            R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
            t_w2c = np.array(cam["cam_t_w2c"], float)
            K = np.array(cam["cam_K"], float).reshape(3, 3)
            # rebuild the GDRNPP coarse world pose from the matched pred.
            # we don't have R_est here directly; re-read preds is heavy, so use GT-
            # flipped as the coarse stand-in for the probe of separability (the
            # refiner sees a FLIPPED coarse and must un-flip it). This isolates the
            # scorer's separability, which is the mechanism question.
            _, R_gt, t_gt = gt_inst[(sid, im, inst)]
            mask = RR.load_mask(args.bop_root, sid, im, inst)
            if mask is None:
                continue
            rgb = RR.load_rgb(args.bop_root, sid, im)
            bbox = RR.mask_bbox(mask)
            ie = None
            if rgb is not None and bbox is not None:
                crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                ie = RC.image_edges(crop, bbox=bbox, full_hw=mask.shape)
            # coarse = a flipped GT (transverse flip) in WORLD frame.
            R_gt_world, t_world = A.bop_pose_to_world(R_gt, t_gt, R_w2c, t_w2c,
                                                      RR.TABLE_ORIGIN)
            Rflip = A.axis_angle_matrix(TRANSVERSE[d["flip_axis"]], np.pi)
            R_coarse_world = R_gt_world @ Rflip
            verts_mm = np.asarray(meshes[oid].vertices, float)
            R_new, info = RC.refine_detection(
                R_coarse_world, verts_mm=verts_mm, t_world_m=t_world,
                table_origin_m=RR.TABLE_ORIGIN, R_w2c=R_w2c, t_w2c_mm=t_w2c,
                K=K, hw=mask.shape, target_mask=mask, image_edge_mask=ie,
                sym_axis=tuple(SYM_AXIS), n_fold=None, stable_downs=downs[oid],
                scorer="cpu_edge", min_margin=0.0)   # OPEN gate to test separability
            n += 1
            if info.get("switched"):
                nsw += 1
            # did the refined world rotation land near the correct (GT) pose?
            R_new_cam = R_w2c @ R_new
            rs_after = rot_err_sym_contY(R_new_cam, R_gt, syms)
            if rs_after <= args.flip_thr_deg:
                nfix += 1
            # per-hypothesis: which hyp tag won + its score vs coarse
            scores = info.get("scores", [])
            tags = info.get("tags", [])
            best_tag = info.get("best_tag")
            # the 'correct' hypothesis is the one that un-flips (flip180_x/z)
            correct_tags = {"flip180_x", "flip180_z"}
            if best_tag in correct_tags:
                ncorrect += 1
            details.append({
                "scene": sid, "im": im, "inst": inst, "obj": oid,
                "visib": round(d["visib_fract"], 3),
                "best_tag": best_tag, "switched": info.get("switched"),
                "rs_after_deg": round(rs_after, 1),
                "coarse_score": round(info.get("coarse_score", 0.0), 4),
                "best_score": round(info.get("best_score", 0.0), 4),
                "scores": {t: round(s, 4) for t, s in zip(tags, scores)},
            })
        return {"n": n, "n_fixed": nfix, "n_correct_wins": ncorrect,
                "n_switched": nsw, "details": details}

    return {"full": probe(flipped_full), "partial": probe(flipped_part)}


def visaware_compare(per_inst, args, syms):
    """THE S-003/T-085 measurement: run refine_rc on the REAL GDRNPP coarse pose,
    BEFORE (today: scorer over the full crop, visib_mask=None) vs AFTER (ADR-020:
    scorer restricted to the visible region mask_visib + visibility-conditioned
    gate). Measure flip-rate (rot_sym>90deg) before/after, per stratum.

    The 'coarse' is the genuine GDRNPP R_est (model->cam) transformed to world —
    NOT a GT-flipped stand-in. So this measures the actual production flip, with
    the actual refiner, under the actual partial visibility. Both before and after
    use the SAME hard min_margin (ADR-020: never globally loosened); only AFTER
    adds the visibility restriction + visibility-conditioned gate.
    """
    cams = RR.load_cams(args.bop_root)
    meshes, downs = RR.load_meshes(args.bop_root)
    gt_inst = gt_R_for_inst(args.bop_root)

    def stratum_insts(lo, hi):
        g = [d for d in per_inst if lo <= d["visib_fract"] < hi]
        return g if args.visaware_all else g[:args.visaware_compare]

    def full_insts():
        g = [d for d in per_inst if d["visib_fract"] >= args.full_thr]
        return g if args.visaware_all else g[:args.visaware_compare]

    def run_group(group):
        n = nfix = nbroke = nsw_after = 0
        n_before = n_after = 0
        details = []
        for d in group:
            sid, im, oid, inst = d["scene_id"], d["im_id"], d["obj_id"], d["inst"]
            cam = cams[sid][str(im)]
            R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
            t_w2c = np.array(cam["cam_t_w2c"], float)
            K = np.array(cam["cam_K"], float).reshape(3, 3)
            visib = RR.load_mask(args.bop_root, sid, im, inst)        # mask_visib
            full = RR.load_full_mask(args.bop_root, sid, im, inst)    # full mask/
            if visib is None:
                continue
            if full is None:
                full = visib
            _, R_gt, t_gt = gt_inst[(sid, im, inst)]
            rgb = RR.load_rgb(args.bop_root, sid, im)
            # image edges from the FULL-mask bbox (the real crop the pipeline sees).
            bbox = RR.mask_bbox(full)
            ie = None
            if rgb is not None and bbox is not None:
                crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                ie = RC.image_edges(crop, bbox=bbox, full_hw=full.shape)
            # REAL GDRNPP coarse -> world.
            R_world, t_world = A.bop_pose_to_world(d["R_est"], t_gt, R_w2c, t_w2c,
                                                   RR.TABLE_ORIGIN)
            verts_mm = np.asarray(meshes[oid].vertices, float)
            vf = float(d["visib_fract"])
            common = dict(
                verts_mm=verts_mm, t_world_m=t_world, table_origin_m=RR.TABLE_ORIGIN,
                R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K, hw=full.shape, image_edge_mask=ie,
                sym_axis=tuple(SYM_AXIS), n_fold=None, stable_downs=downs[oid],
                scorer="cpu_edge", min_margin=args.visaware_min_margin)
            # BEFORE: today's path — full mask as target, no visibility restriction.
            R_before, info_b = RC.refine_detection(
                R_world, target_mask=full, visib_mask=None, **common)
            # AFTER: visibility-aware — target=mask_visib, score clipped to visible.
            R_after, info_a = RC.refine_detection(
                R_world, target_mask=visib, visib_mask=visib, visib_fract=vf,
                min_visib_fract=args.visaware_min_vf, **common)

            rs_coarse = rot_err_sym_contY(R_w2c @ R_world, R_gt, syms)
            rs_before = rot_err_sym_contY(R_w2c @ R_before, R_gt, syms)
            rs_after = rot_err_sym_contY(R_w2c @ R_after, R_gt, syms)
            flip_before = rs_before > args.flip_thr_deg
            flip_after = rs_after > args.flip_thr_deg
            n += 1
            n_before += int(flip_before)
            n_after += int(flip_after)
            if flip_before and not flip_after:
                nfix += 1
            if (not flip_before) and flip_after:
                nbroke += 1
            if info_a.get("switched"):
                nsw_after += 1
            details.append({
                "scene": sid, "im": im, "inst": inst, "obj": oid,
                "visib": round(vf, 3),
                "rs_coarse": round(rs_coarse, 1),
                "rs_before": round(rs_before, 1), "rs_after": round(rs_after, 1),
                "flip_before": bool(flip_before), "flip_after": bool(flip_after),
                "best_before": info_b.get("best_tag"),
                "best_after": info_a.get("best_tag"),
                "switched_after": bool(info_a.get("switched")),
                "visib_gated_out_after": bool(info_a.get("visib_gated_out", False)),
            })
        out = {
            "n": n,
            "n_flip_before": n_before, "n_flip_after": n_after,
            "flip_rate_before": round(n_before / n, 4) if n else 0.0,
            "flip_rate_after": round(n_after / n, 4) if n else 0.0,
            "n_fixed": nfix, "n_broke": nbroke, "n_switched_after": nsw_after,
            "details": details,
        }
        return out

    return {
        "params": {"min_margin": args.visaware_min_margin,
                   "min_visib_fract": args.visaware_min_vf,
                   "partial_band": [args.partial_lo, args.partial_hi],
                   "full_thr": args.full_thr,
                   "all": bool(args.visaware_all)},
        "partial": run_group(stratum_insts(args.partial_lo, args.partial_hi)),
        "full": run_group(full_insts()),
    }


if __name__ == "__main__":
    main()
