#!/usr/bin/env python3
"""eval_bop.py - Symmetry-aware BOP eval harness for the POSE project (Story S-501 / T-070).

Measures 6D pose predictions against a synthetic BOP ground-truth dataset using the
official ``bop_toolkit_lib`` metrics:

  * AR  = mean(AR_VSD, AR_MSSD, AR_MSPD)   (BOP19+ standard)
  * ADD / ADI                              (ADI for symmetric parts)
  * Translation error  [mm]                (human-friendly)
  * Rotation error     [deg]               (symmetry-resolved -> the 120/91 killer)

Symmetry is read straight from ``models_eval/models_info.json`` (continuous / discrete),
so a pose rotated about a symmetry axis is *not* punished -- this is the analytic fix for
the eigenbau 120/91-degree problem (Viktor ADR section 2).

This module is self-contained on the box: it imports ``bop_toolkit_lib`` directly and
does NOT touch any training. It runs read-only against the BOP dataset.

------------------------------------------------------------------------------
USAGE (on the GPU box, inside bop-venv)

  # Score real predictions (BOP results CSV) against the val split:
  /mnt/data/bop/bop-venv/bin/python eval_bop.py \
      --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
      --split val \
      --preds /path/to/predictions.csv \
      --out /mnt/data/bop/results/run_X

  # Self-validation (no predictions / no GDRNPP needed): synthesises preds from GT.
  /mnt/data/bop/bop-venv/bin/python eval_bop.py \
      --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
      --split val --self-test \
      --out /mnt/data/bop/results/selftest

From the laptop, wrap with ``box_src/eval_bop.sh`` (see EVAL_BOP.md).

------------------------------------------------------------------------------
PREDICTIONS INPUT FORMAT  (BOP results CSV, the de-facto BOP19 standard)

  scene_id,im_id,obj_id,score,R,t,time

  scene_id : int  (matches the train_pbr/val scene folder, e.g. 0 for 000000)
  im_id    : int  (image index within the scene)
  obj_id   : int  (1..6, the BOP obj_id -- detector class + 1, see ADR section 1.2)
  score    : float (detection/pose confidence; 1.0 if unknown)
  R        : 9 space-separated floats, row-major model->camera rotation
  t        : 3 space-separated floats, model->camera translation in MILLIMETRES
  time     : float, seconds for the whole image (-1 if unknown)

A header line is optional. This is exactly what ``inout.save_bop_results`` writes and
what GDRNPP / GigaPose / MegaPose all emit. See EVAL_BOP.md for the GDRNPP -> CSV bridge.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# bop_toolkit_lib is installed editable in the bop-venv on the box.
try:
    from bop_toolkit_lib import inout, misc, pose_error
except Exception as exc:  # pragma: no cover - only on a mis-set-up box
    sys.stderr.write(
        "FATAL: cannot import bop_toolkit_lib. Run me with the bop-venv python:\n"
        "  /mnt/data/bop/bop-venv/bin/python eval_bop.py ...\n"
        f"  underlying error: {exc!r}\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# BOP19 thresholds (verbatim from scripts/eval_bop19_pose.py)
# ---------------------------------------------------------------------------
VSD_DELTA = 15.0                                  # mm
VSD_TAUS = list(np.arange(0.05, 0.51, 0.05))      # 10 taus, normalised by diameter
VSD_THS = list(np.arange(0.05, 0.51, 0.05))       # 10 correctness thresholds
MSSD_THS = list(np.arange(0.05, 0.51, 0.05))      # x diameter
MSPD_THS = list(np.arange(5, 51, 5))              # px (scaled by r = 640/width)
MAX_SYM_DISC_STEP = 0.01
MSPD_REF_WIDTH = 640.0                            # BOP reference image width


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_models(dataset_dir, n_points=0):
    """Load eval meshes (points) + models_info (symmetry/diameter) per obj_id.

    n_points > 0 subsamples each mesh to at most n_points vertices for the
    point-based metrics (ADD/ADI/MSSD/MSPD). Continuous symmetry expands to ~315
    transforms, so capping points keeps MSSD/MSPD tractable; the effect on the
    distances is negligible (BOP ships decimated models_eval for the same reason).

    Returns dict: obj_id -> {pts, diameter, model_info, syms, ply_path, ...}
    """
    models_dir = os.path.join(dataset_dir, "models_eval")
    if not os.path.isdir(models_dir):
        models_dir = os.path.join(dataset_dir, "models")
    info_path = os.path.join(models_dir, "models_info.json")
    models_info = inout.load_json(info_path, keys_to_int=True)

    rng = np.random.default_rng(0)
    models = {}
    for obj_id, info in models_info.items():
        ply_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(ply_path)
        pts = model["pts"]
        if n_points and pts.shape[0] > n_points:
            idx = rng.choice(pts.shape[0], n_points, replace=False)
            pts = pts[idx]
        syms = misc.get_symmetry_transformations(info, MAX_SYM_DISC_STEP)
        models[obj_id] = {
            "pts": pts,
            "diameter": float(info["diameter"]),
            "model_info": info,
            "syms": syms,
            "n_syms": len(syms),
            "ply_path": ply_path,
            "is_symmetric": ("symmetries_continuous" in info)
            or ("symmetries_discrete" in info),
        }
    return models, models_dir


def load_scene_data(dataset_dir, split, max_images=0, visib_band=None):
    """Load per-scene GT + camera for the requested split.

    max_images > 0 keeps only the first max_images image-ids per scene (fast smoke).
    visib_band = (lo, hi): keep only GT instances with lo <= visib_fract < hi (read
    from scene_gt_info.json). This restricts the AR denom+numerator to a visibility
    stratum (e.g. the partial-vis subset where the flip-fix concentrates, T-089).

    Returns dict: scene_id -> {gt: {im_id: [insts]}, cam: {im_id: cam}}
    """
    split_dir = os.path.join(dataset_dir, split)
    scenes = {}
    for name in sorted(os.listdir(split_dir)):
        scene_path = os.path.join(split_dir, name)
        gt_path = os.path.join(scene_path, "scene_gt.json")
        cam_path = os.path.join(scene_path, "scene_camera.json")
        if not (os.path.isfile(gt_path) and os.path.isfile(cam_path)):
            continue
        scene_id = int(name)
        gt = inout.load_scene_gt(gt_path)       # {im_id: [ {obj_id, cam_R_m2c, cam_t_m2c} ]}
        cam = inout.load_scene_camera(cam_path)  # {im_id: {cam_K, depth_scale, ...}}
        if visib_band is not None:
            lo, hi = visib_band
            info_path = os.path.join(scene_path, "scene_gt_info.json")
            info = json.load(open(info_path)) if os.path.isfile(info_path) else {}
            gt2 = {}
            for im_id, insts in gt.items():
                vinfo = info.get(str(im_id)) or info.get(im_id) or []
                kept = [inst for i, inst in enumerate(insts)
                        if i < len(vinfo)
                        and lo <= float(vinfo[i].get("visib_fract", -1.0)) < hi]
                if kept:
                    gt2[im_id] = kept
            gt = gt2
        if max_images:
            keep = sorted(gt.keys())[:max_images]
            gt = {k: gt[k] for k in keep}
            cam = {k: cam[k] for k in keep if k in cam}
        scenes[scene_id] = {"id": scene_id, "path": scene_path, "gt": gt, "cam": cam}
    if not scenes:
        raise FileNotFoundError(f"no scenes with scene_gt.json under {split_dir}")
    return scenes


# ---------------------------------------------------------------------------
# Per-instance error computation
# ---------------------------------------------------------------------------
def rot_err_deg_naive(R_est, R_gt):
    """Raw geodesic rotation error in degrees (NOT symmetry aware).

    pose_error.re() already returns degrees (it does 180*acos/pi internally).
    """
    return float(pose_error.re(R_est, R_gt))


def rot_err_deg_sym(R_est, R_gt, syms):
    """Symmetry-RESOLVED rotation error in degrees.

    Picks the symmetry representative of the GT that is closest in rotation to the
    estimate, then measures the residual geodesic angle. For a continuous/discrete
    symmetric part a pose rotated about the symmetry axis collapses to ~0 deg --
    this is the analytic fix for the eigenbau 120/91-degree punishment.
    pose_error.re() already returns degrees.
    """
    best = None
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        e = pose_error.re(R_est, R_gt_sym)
        if best is None or e < best:
            best = e
    return float(best)


def compute_pair_errors(R_e, t_e, R_g, t_g, K, model, renderer, depth_test):
    """All BOP + human errors for one matched (est, gt) pair.

    t_e, t_g are 3x1 mm. Returns a dict of scalar errors.
    """
    pts = model["pts"]
    syms = model["syms"]
    diam = model["diameter"]

    out = {
        "add": float(pose_error.add(R_e, t_e, R_g, t_g, pts)),
        "adi": float(pose_error.adi(R_e, t_e, R_g, t_g, pts)),
        "mssd": float(pose_error.mssd(R_e, t_e, R_g, t_g, pts, syms)),
        "mspd_raw": float(pose_error.mspd(R_e, t_e, R_g, t_g, K, pts, syms)),
        "te_mm": float(pose_error.te(t_e, t_g)),
        "rot_deg": rot_err_deg_sym(R_e, R_g, syms),
        "rot_deg_naive": rot_err_deg_naive(R_e, R_g),
    }
    # ADD/ADI: use ADI for symmetric parts (the metric that respects symmetry).
    out["add_or_adi"] = out["adi"] if model["is_symmetric"] else out["add"]

    # VSD: optional (needs a depth renderer + the test-image depth map).
    if renderer is not None and depth_test is not None:
        vsd_errs = pose_error.vsd(
            R_e, t_e, R_g, t_g, depth_test, K,
            VSD_DELTA, VSD_TAUS, True, diam, renderer, model["obj_id"],
        )
        out["vsd"] = [float(v) for v in vsd_errs]  # one per tau
    else:
        out["vsd"] = None
    return out


# ---------------------------------------------------------------------------
# Greedy GT<->est matching per (scene, im, obj)
# ---------------------------------------------------------------------------
def match_greedy(ests, gts):
    """Greedy match estimates to GT by translation distance (mm).

    BOP localisation: for each estimate (highest score first), claim the nearest
    unclaimed GT instance of the same obj_id. Returns list of (est_idx, gt_idx).
    Unmatched estimates -> gt_idx None (false positive); unmatched GT -> miss.
    """
    order = sorted(range(len(ests)), key=lambda i: -ests[i]["score"])
    used_gt = set()
    pairs = []
    for ei in order:
        te = ests[ei]["t"]
        best_gi, best_d = None, None
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            d = float(np.linalg.norm(te.reshape(3) - g["t"].reshape(3)))
            if best_d is None or d < best_d:
                best_d, best_gi = d, gi
        if best_gi is not None:
            used_gt.add(best_gi)
            pairs.append((ei, best_gi))
        else:
            pairs.append((ei, None))
    return pairs, used_gt


# ---------------------------------------------------------------------------
# AR recall aggregation
# ---------------------------------------------------------------------------
def recall_from_errors(errs, ths, n_gt):
    """Mean recall over a list of thresholds.

    errs : list of per-matched-instance error values (matched only).
    n_gt : total number of GT instances (denominator -> misses count as 0).
    """
    if n_gt == 0:
        return 0.0, [0.0 for _ in ths]
    per_th = []
    for th in ths:
        correct = sum(1 for e in errs if e <= th)
        per_th.append(correct / n_gt)
    return float(np.mean(per_th)), per_th


def vsd_recall(vsd_lists, n_gt):
    """AR_VSD: recall averaged over (10 taus x 10 thresholds) per BOP19."""
    if n_gt == 0 or not vsd_lists:
        return None
    recalls = []
    n_taus = len(VSD_TAUS)
    for ti in range(n_taus):
        errs_ti = [v[ti] for v in vsd_lists]
        for th in VSD_THS:
            correct = sum(1 for e in errs_ti if e <= th)
            recalls.append(correct / n_gt)
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# Prediction sources
# ---------------------------------------------------------------------------
def preds_from_csv(path):
    """Load BOP-results CSV -> {(scene,im): [ {obj_id, score, R, t, time} ]}."""
    raw = inout.load_bop_results(path, version="bop19")
    by_img = defaultdict(list)
    for r in raw:
        by_img[(r["scene_id"], r["im_id"])].append(
            {
                "obj_id": r["obj_id"],
                "score": r["score"],
                "R": np.array(r["R"], dtype=np.float64).reshape(3, 3),
                "t": np.array(r["t"], dtype=np.float64).reshape(3, 1),
                "time": r.get("time", -1),
            }
        )
    return by_img


def preds_from_gt(scenes, noise=None, rng=None, models=None):
    """Synthesise predictions from GT (self-test).

    noise=None -> perfect predictions (GT-as-pred).
    noise={'rot_deg':d,'trans_mm':m,'axis':'random'|'sym'} -> perturb each pose.

    'axis'='sym' rotates about the part's SYMMETRY axis (model Y), by an amount the
    symmetry forgives: an arbitrary angle for continuous parts, exactly one discrete
    step (360/N) for discrete parts. Symmetric parts then shrug it off (resolved
    rot ~0) while asymmetric parts get punished -- proving the symmetry handling.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    by_img = defaultdict(list)
    for sid, sc in scenes.items():
        for im_id, insts in sc["gt"].items():
            for inst in insts:
                R = np.array(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                t = np.array(inst["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
                if noise is not None:
                    model = (models or {}).get(inst["obj_id"])
                    R, t = _perturb(R, t, noise, rng, model)
                by_img[(sid, im_id)].append(
                    {"obj_id": inst["obj_id"], "score": 1.0, "R": R, "t": t, "time": -1}
                )
    return by_img


def _sym_forgiven_angle_deg(model, default_deg):
    """Y-axis twist angle that the part's symmetry forgives.

    Continuous part -> default_deg (any angle is forgiven).
    Discrete C_N part -> exactly 360/N (one tooth step).
    Asymmetric part -> default_deg (will be a real error, as intended).
    """
    if model is None:
        return default_deg
    info = model["model_info"]
    if "symmetries_continuous" in info:
        return default_deg
    if "symmetries_discrete" in info:
        n = len(info["symmetries_discrete"]) + 1  # +1 for identity
        return 360.0 / n
    return default_deg


def _axis_angle_R(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s, C = np.cos(angle_rad), np.sin(angle_rad), 1.0 - np.cos(angle_rad)
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _perturb(R, t, noise, rng, model=None):
    if noise.get("axis") == "sym":
        # Rotate about the MODEL Y axis (symmetry axis), applied in model frame:
        # R_pred = R_gt @ R_y(ang). For a part symmetric about Y by this amount it
        # is a no-op for the sym-resolved error (continuous: any angle; discrete: 360/N).
        ang_deg = _sym_forgiven_angle_deg(model, noise.get("rot_deg", 0.0))
        dR = _axis_angle_R([0.0, 1.0, 0.0], np.radians(ang_deg))
        R_new = R.dot(dR)
    else:
        ang = np.radians(noise.get("rot_deg", 0.0))
        rand_axis = rng.normal(size=3)
        dR = _axis_angle_R(rand_axis, ang)
        R_new = dR.dot(R)  # world-frame perturbation
    dt = rng.normal(size=(3, 1))
    dt = dt / (np.linalg.norm(dt) + 1e-12) * noise.get("trans_mm", 0.0)
    return R_new, t + dt


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate(scenes, models, preds_by_img, use_vsd=False, obj_names=None):
    """Run the full eval. Returns a results dict."""
    renderer = None
    if use_vsd:
        renderer = _make_renderer(scenes, models)

    # accumulators per obj_id
    acc = defaultdict(
        lambda: {
            "n_gt": 0,
            "n_matched": 0,
            "add_or_adi": [],
            "add": [],
            "adi": [],
            "mssd": [],
            "mspd": [],   # scaled
            "vsd": [],
            "te_mm": [],
            "rot_deg": [],
            "rot_deg_naive": [],
        }
    )

    for sid, sc in scenes.items():
        for im_id, gt_insts in sc["gt"].items():
            cam = sc["cam"][im_id]
            K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
            width = _img_width(K)
            mspd_scale = MSPD_REF_WIDTH / width if width else 1.0

            depth_test = None
            if use_vsd:
                depth_test = _load_depth(sc, im_id, cam)

            ests = preds_by_img.get((sid, im_id), [])

            # group by obj_id
            gt_by_obj = defaultdict(list)
            for inst in gt_insts:
                gt_by_obj[inst["obj_id"]].append(
                    {
                        "R": np.array(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                        "t": np.array(inst["cam_t_m2c"], dtype=np.float64).reshape(3, 1),
                    }
                )
            est_by_obj = defaultdict(list)
            for e in ests:
                est_by_obj[e["obj_id"]].append(e)

            for obj_id, gts in gt_by_obj.items():
                model = models[obj_id]
                model["obj_id"] = obj_id
                a = acc[obj_id]
                a["n_gt"] += len(gts)

                ests_o = est_by_obj.get(obj_id, [])
                pairs, _ = match_greedy(ests_o, gts)
                for ei, gi in pairs:
                    if gi is None:
                        continue  # false positive, ignored for recall denom
                    a["n_matched"] += 1
                    R_e, t_e = ests_o[ei]["R"], ests_o[ei]["t"]
                    R_g, t_g = gts[gi]["R"], gts[gi]["t"]
                    errs = compute_pair_errors(
                        R_e, t_e, R_g, t_g, K, model, renderer, depth_test
                    )
                    a["add_or_adi"].append(errs["add_or_adi"])
                    a["add"].append(errs["add"])
                    a["adi"].append(errs["adi"])
                    a["mssd"].append(errs["mssd"])
                    a["mspd"].append(errs["mspd_raw"] * mspd_scale)
                    a["te_mm"].append(errs["te_mm"])
                    a["rot_deg"].append(errs["rot_deg"])
                    a["rot_deg_naive"].append(errs["rot_deg_naive"])
                    if errs["vsd"] is not None:
                        a["vsd"].append(errs["vsd"])

    return _aggregate(acc, models, obj_names, use_vsd)


def _aggregate(acc, models, obj_names, use_vsd):
    per_obj = {}
    ar_list, vsd_list, mssd_list, mspd_list = [], [], [], []
    for obj_id in sorted(acc.keys()):
        a = acc[obj_id]
        n_gt = a["n_gt"]
        diam = models[obj_id]["diameter"]

        mssd_ths = [th * diam for th in MSSD_THS]
        ar_mssd, _ = recall_from_errors(a["mssd"], mssd_ths, n_gt)
        ar_mspd, _ = recall_from_errors(a["mspd"], MSPD_THS, n_gt)
        ar_vsd = vsd_recall(a["vsd"], n_gt) if use_vsd else None

        components = [ar_mssd, ar_mspd] + ([ar_vsd] if ar_vsd is not None else [])
        ar = float(np.mean(components)) if components else 0.0

        per_obj[obj_id] = {
            "name": (obj_names or {}).get(obj_id, str(obj_id)),
            "symmetric": models[obj_id]["is_symmetric"],
            "n_sym_transforms": models[obj_id]["n_syms"],
            "diameter_mm": diam,
            "n_gt": n_gt,
            "n_matched": a["n_matched"],
            "AR": ar,
            "AR_VSD": ar_vsd,
            "AR_MSSD": ar_mssd,
            "AR_MSPD": ar_mspd,
            "ADD_mean_mm": _mean(a["add"]),
            "ADI_mean_mm": _mean(a["adi"]),
            "ADD_or_ADI_mean_mm": _mean(a["add_or_adi"]),
            "trans_err_mean_mm": _mean(a["te_mm"]),
            "trans_err_median_mm": _median(a["te_mm"]),
            "rot_err_mean_deg": _mean(a["rot_deg"]),
            "rot_err_median_deg": _median(a["rot_deg"]),
            "rot_err_naive_mean_deg": _mean(a["rot_deg_naive"]),
        }
        ar_list.append(ar)
        mssd_list.append(ar_mssd)
        mspd_list.append(ar_mspd)
        if ar_vsd is not None:
            vsd_list.append(ar_vsd)

    overall = {
        "AR": float(np.mean(ar_list)) if ar_list else 0.0,
        "AR_VSD": float(np.mean(vsd_list)) if vsd_list else None,
        "AR_MSSD": float(np.mean(mssd_list)) if mssd_list else 0.0,
        "AR_MSPD": float(np.mean(mspd_list)) if mspd_list else 0.0,
        "n_gt": sum(per_obj[o]["n_gt"] for o in per_obj),
        "n_matched": sum(per_obj[o]["n_matched"] for o in per_obj),
        "vsd_enabled": use_vsd,
    }
    return {"per_object": per_obj, "overall": overall}


def _mean(x):
    return float(np.mean(x)) if x else None


def _median(x):
    return float(np.median(x)) if x else None


# ---------------------------------------------------------------------------
# VSD support (renderer + depth)
# ---------------------------------------------------------------------------
def _img_width(K):
    return int(round(2.0 * K[0, 2]))  # cx ~ width/2 for centred Isaac renders


def _make_renderer(scenes, models):
    from bop_toolkit_lib.rendering.renderer import create_renderer

    # use the largest image dims seen
    w = h = 0
    for sc in scenes.values():
        for cam in sc["cam"].values():
            K = np.array(cam["cam_K"]).reshape(3, 3)
            w = max(w, int(round(2 * K[0, 2])))
            h = max(h, int(round(2 * K[1, 2])))
    ren = create_renderer(w, h, renderer_type="vispy", mode="depth")
    models_dir = models["_models_dir"] if "_models_dir" in models else None
    for obj_id, m in models.items():
        if not isinstance(obj_id, int):
            continue
        ply = m.get("ply_path")
        ren.add_object(obj_id, ply)
    return ren


def _load_depth(sc, im_id, cam):
    depth_path = os.path.join(sc["path"], "depth", f"{im_id:06d}.png")
    if not os.path.isfile(depth_path):
        return None
    depth = inout.load_depth(depth_path)
    return depth * cam.get("depth_scale", 1.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_table(results, title=""):
    po = results["per_object"]
    ov = results["overall"]
    cols = [
        ("obj", 4), ("name", 22), ("sym", 4), ("n", 5), ("AR", 7),
        ("AR_VSD", 7), ("AR_MSSD", 8), ("AR_MSPD", 8),
        ("ADD/ADI_mm", 11), ("trans_mm", 9), ("rot_deg", 8), ("rot_naive", 10),
    ]
    lines = []
    if title:
        lines.append(title)
    header = " ".join(f"{c:>{w}}" if c != "name" else f"{c:<{w}}" for c, w in cols)
    lines.append(header)
    lines.append("-" * len(header))

    def fmt(v, nd=3):
        return "  -  " if v is None else f"{v:.{nd}f}"

    for obj_id in sorted(po.keys()):
        r = po[obj_id]
        row = [
            f"{obj_id:>4}",
            f"{r['name'][:22]:<22}",
            f"{'Y' if r['symmetric'] else 'n':>4}",
            f"{r['n_gt']:>5}",
            f"{fmt(r['AR']):>7}",
            f"{fmt(r['AR_VSD']):>7}",
            f"{fmt(r['AR_MSSD']):>8}",
            f"{fmt(r['AR_MSPD']):>8}",
            f"{fmt(r['ADD_or_ADI_mean_mm'], 2):>11}",
            f"{fmt(r['trans_err_mean_mm'], 2):>9}",
            f"{fmt(r['rot_err_mean_deg'], 2):>8}",
            f"{fmt(r['rot_err_naive_mean_deg'], 2):>10}",
        ]
        lines.append(" ".join(row))
    lines.append("-" * len(header))
    lines.append(
        f"{'ALL':>4} {'(mean over objects)':<22} {'':>4} {ov['n_gt']:>5} "
        f"{fmt(ov['AR']):>7} {fmt(ov['AR_VSD']):>7} {fmt(ov['AR_MSSD']):>8} "
        f"{fmt(ov['AR_MSPD']):>8}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-validation driver
# ---------------------------------------------------------------------------
def run_self_test(scenes, models, obj_names, use_vsd):
    """Three scenarios proving the harness + the symmetry handling."""
    rng = np.random.default_rng(42)
    blocks = {}

    # (a) GT-as-prediction -> AR ~ 1.0, ADD ~ 0, rot ~ 0
    preds = preds_from_gt(scenes, noise=None)
    blocks["gt_vs_gt"] = evaluate(scenes, models, preds, use_vsd, obj_names)

    # (b) Noisy GT, RANDOM-axis rotation 10deg + 5mm -> errors rise for ALL parts
    preds = preds_from_gt(
        scenes, noise={"rot_deg": 10.0, "trans_mm": 5.0, "axis": "random"},
        rng=np.random.default_rng(7),
    )
    blocks["noise_random_10deg_5mm"] = evaluate(scenes, models, preds, use_vsd, obj_names)

    # (c) Noisy GT, SYMMETRY-axis rotation + 5mm. The twist amount is what the
    #     symmetry forgives: 30deg about Y for continuous parts, exactly 360/N for
    #     discrete (Zahnrad C_7 -> ~51.4deg). Symmetric parts then collapse the
    #     sym-resolved rot to ~0 while naive shows the full twist; asymmetric parts
    #     are punished (resolved == naive). This is the 120/91-degree fix proof.
    preds = preds_from_gt(
        scenes, noise={"rot_deg": 30.0, "trans_mm": 5.0, "axis": "sym"},
        rng=np.random.default_rng(7), models=models,
    )
    blocks["noise_symaxis_5mm"] = evaluate(scenes, models, preds, use_vsd, obj_names)

    return blocks


def print_self_test(blocks):
    out = []
    out.append("\n================ SELF-VALIDATION ================\n")
    out.append(render_table(blocks["gt_vs_gt"],
               "[a] GT-as-prediction (sanity: AR~1.0, ADD~0, rot~0)"))
    out.append("\n")
    out.append(render_table(blocks["noise_random_10deg_5mm"],
               "[b] GT + 10deg RANDOM-axis + 5mm noise (errors rise for ALL)"))
    out.append("\n")
    out.append(render_table(blocks["noise_symaxis_5mm"],
               "[c] GT + SYMMETRY-axis twist (cont:30deg, discrete:360/N) + 5mm noise "
               "(sym parts: rot_deg<<rot_naive => symmetry handling works)"))
    out.append("\n")

    # Explicit pass/fail verdicts
    verdicts = {}
    out.append("---------------- VERDICTS ----------------")
    a = blocks["gt_vs_gt"]["overall"]
    a_obj = blocks["gt_vs_gt"]["per_object"]
    max_add = max((v["ADD_or_ADI_mean_mm"] or 0) for v in a_obj.values())
    max_rot = max((v["rot_err_mean_deg"] or 0) for v in a_obj.values())
    ok_a = (a["AR"] is not None and a["AR"] > 0.999 and max_add < 0.5 and max_rot < 0.5)
    verdicts["a_gt_vs_gt"] = ok_a
    out.append(f"[a] GT-vs-GT  AR={a['AR']:.4f} (>0.999? {a['AR'] > 0.999})  "
               f"max ADD/ADI={max_add:.4f}mm  max rot={max_rot:.4f}deg  "
               f"-> {'PASS' if ok_a else 'FAIL'}")

    b = blocks["noise_random_10deg_5mm"]["overall"]
    ok_b = b["AR"] < a["AR"]
    verdicts["b_noise_raises_error"] = ok_b
    out.append(f"[b] random-noise AR={b['AR']:.4f} (should drop below [a]={a['AR']:.4f}) "
               f"-> {'PASS' if ok_b else 'FAIL'}")

    # [c] symmetry effect: for symmetric objects sym-resolved rot << naive rot.
    c_obj = blocks["noise_symaxis_5mm"]["per_object"]
    out.append("[c] symmetry effect per object (twist about symmetry axis):")
    sym_proof = True
    for oid in sorted(c_obj):
        r = c_obj[oid]
        rd, rn = r["rot_err_mean_deg"], r["rot_err_naive_mean_deg"]
        if r["symmetric"]:
            shrinks = rd < 2.0 and rd < rn - 5.0
            sym_proof = sym_proof and shrinks
            tag = "SYM  -> resolved rot collapses" if shrinks else "SYM  -> NOT collapsing(!)"
        else:
            tag = "asym -> resolved == naive (expected)"
        out.append(f"    obj {oid} {r['name'][:18]:<18} "
                   f"rot_resolved={rd:6.2f}deg  rot_naive={rn:6.2f}deg  [{tag}]")
    verdicts["c_symmetry_handling"] = sym_proof
    out.append(f"  symmetry handling verdict -> {'PASS' if sym_proof else 'FAIL'}")
    out.append("")
    overall_pass = all(verdicts.values())
    out.append(f"OVERALL SELF-VALIDATION -> {'PASS' if overall_pass else 'FAIL'}  {verdicts}")
    out.append("==========================================\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Symmetry-aware BOP eval harness (POSE / T-070)")
    ap.add_argument("--dataset-dir", required=True, help="BOP dataset root (pose_isaac)")
    ap.add_argument("--split", default="val", help="split folder (val / train_pbr / test)")
    ap.add_argument("--preds", help="BOP-results CSV with predictions")
    ap.add_argument("--self-test", action="store_true",
                    help="synthesise predictions from GT (no GDRNPP needed)")
    ap.add_argument("--vsd", action="store_true",
                    help="also compute VSD (needs vispy/EGL renderer + depth maps)")
    ap.add_argument("--n-points", type=int, default=2000,
                    help="subsample mesh to N points for point metrics "
                         "(0 = full mesh; default 2000 keeps MSSD/MSPD tractable)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap images per scene (0 = all; for fast smoke runs)")
    ap.add_argument("--visib-band", default="",
                    help="lo,hi : restrict AR to GT instances with "
                         "lo<=visib_fract<hi (e.g. 0.20,0.50 = partial-vis subset, "
                         "T-089). Reads scene_gt_info.json. Empty = full set.")
    ap.add_argument("--out", help="output dir for report.json + report.txt")
    args = ap.parse_args()

    visib_band = None
    if args.visib_band:
        lo, hi = (float(x) for x in args.visib_band.split(","))
        visib_band = (lo, hi)

    if not args.self_test and not args.preds:
        ap.error("provide --preds <csv> or --self-test")

    info = inout.load_json(os.path.join(args.dataset_dir, "dataset_info.json")) \
        if os.path.isfile(os.path.join(args.dataset_dir, "dataset_info.json")) else {}
    obj_names = {int(k): v for k, v in info.get("obj_names", {}).items()}

    sys.stderr.write(f"[eval_bop] loading models from {args.dataset_dir} "
                     f"(n_points={args.n_points or 'full'})\n")
    models, _ = load_models(args.dataset_dir, n_points=args.n_points)
    sym_ids = [o for o in models if models[o]["is_symmetric"]]
    nsyms = ", ".join("{}:{}".format(o, models[o]["n_syms"]) for o in sorted(models))
    sys.stderr.write(f"[eval_bop] loaded {len(models)} models; "
                     f"symmetric: {sym_ids}; n_syms: {{{nsyms}}}\n")
    scenes = load_scene_data(args.dataset_dir, args.split,
                             max_images=args.max_images, visib_band=visib_band)
    n_imgs = sum(len(sc['gt']) for sc in scenes.values())
    n_gt = sum(len(insts) for sc in scenes.values() for insts in sc['gt'].values())
    band_s = f" visib_band={visib_band}" if visib_band else ""
    sys.stderr.write(f"[eval_bop] split '{args.split}': {len(scenes)} scenes, "
                     f"{n_imgs} images, {n_gt} GT instances{band_s}\n")

    report_txt, report_json = "", {}

    if args.self_test:
        blocks = run_self_test(scenes, models, obj_names, args.vsd)
        report_txt = print_self_test(blocks)
        report_json = {"mode": "self_test", "split": args.split, "blocks": blocks}
        print(report_txt)
    else:
        preds = preds_from_csv(args.preds)
        n_preds = sum(len(v) for v in preds.values())
        sys.stderr.write(f"[eval_bop] loaded {n_preds} predictions from {args.preds}\n")
        results = evaluate(scenes, models, preds, args.vsd, obj_names)
        report_txt = render_table(results, f"BOP eval - split={args.split} preds={args.preds}")
        report_json = {"mode": "eval", "split": args.split,
                       "preds": args.preds, "results": results}
        print(report_txt)
        ov = results["overall"]
        print(f"\nOVERALL  AR={ov['AR']:.4f}  "
              f"AR_VSD={ov['AR_VSD'] if ov['AR_VSD'] is None else round(ov['AR_VSD'],4)}  "
              f"AR_MSSD={ov['AR_MSSD']:.4f}  AR_MSPD={ov['AR_MSPD']:.4f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report_json, f, indent=2)
        with open(os.path.join(args.out, "report.txt"), "w") as f:
            f.write(report_txt + "\n")
        sys.stderr.write(f"[eval_bop] wrote {args.out}/report.json + report.txt\n")


if __name__ == "__main__":
    main()
