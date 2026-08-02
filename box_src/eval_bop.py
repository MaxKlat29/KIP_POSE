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
the eigenbau 120/91-degree problem (ADR section 2).

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
    # pose_matching + score implement the OFFICIAL BOP19 / IC-BIN localisation
    # matching + recall aggregation. Imported lazily-but-here so the --icbin mode
    # uses the exact same code path as scripts/eval_bop19_pose.py (the bin-picking
    # protocol: per-threshold error-based greedy matching + inst_count denominator).
    from bop_toolkit_lib import pose_matching, score
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


def load_scene_data(dataset_dir, split, max_images=0):
    """Load per-scene GT + camera for the requested split.

    max_images > 0 keeps only the first max_images image-ids per scene (fast smoke).

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
# IC-BIN / official BOP19 localisation protocol (bin-picking)
# ---------------------------------------------------------------------------
# Unlike evaluate() above (which matches estimates to GT by raw translation once
# and reuses those pairs for every threshold), this path reproduces the OFFICIAL
# BOP19 localisation protocol used for bin-picking datasets like IC-BIN:
#
#   * matching is RE-DONE per error type AND per correctness threshold, greedily
#     by score, using the error metric itself (MSSD/MSPD/VSD) as the match cost
#     (bop_toolkit_lib.pose_matching.match_poses);
#   * the recall denominator = number of VALID GT instances per image
#     (visib_fract >= visib_gt_min, BOP19 = 0.1), taken from scene_gt_info.json
#     and the inst_count targets file (this is the multi-instance / bin-picking
#     part: several instances of the same object per image are each a target);
#   * AR_<err> = mean over thresholds of the over-objects mean recall
#     (bop_toolkit_lib.score.calc_localization_scores), AR = mean(AR_VSD,AR_MSSD,
#     AR_MSPD) (without --vsd: mean(AR_MSSD, AR_MSPD)).
#
# We call the official match/score functions directly, so this is byte-for-byte
# the same logic as scripts/eval_bop19_pose.py -- we only build the inputs from
# our (non-registered) pose_isaac dataset instead of via dataset_params.
VISIB_GT_MIN = 0.1                                # BOP19 default
N_TOP = -1                                        # BOP19: all estimates considered


def load_scene_gt_info(dataset_dir, split, scene_ids):
    """Load scene_gt_info.json (visib_fract per GT instance) per scene."""
    info = {}
    for sid in scene_ids:
        path = os.path.join(dataset_dir, split, f"{sid:06d}", "scene_gt_info.json")
        if os.path.isfile(path):
            info[sid] = inout.load_json(path, keys_to_int=True)
    return info


def load_targets(path):
    """Load BOP targets file -> {(scene_id, im_id): {obj_id: inst_count}} or None."""
    if not path or not os.path.isfile(path):
        return None
    raw = inout.load_json(path)
    targets = defaultdict(dict)
    for t in raw:
        targets[(t["scene_id"], t["im_id"])][t["obj_id"]] = t["inst_count"]
    return targets


def evaluate_icbin(scenes, models, preds_by_img, scene_gt_info, targets,
                   use_vsd=False, obj_names=None):
    """Official BOP19/IC-BIN localisation AR via bop_toolkit pose_matching+score.

    Returns the same shape as evaluate() so reporting is shared, but the AR /
    AR_MSSD / AR_MSPD / AR_VSD numbers are the multi-instance, per-threshold,
    error-based-matching, inst_count-denominator protocol.
    """
    renderer = None
    if use_vsd:
        renderer = _make_renderer(scenes, models)

    err_types = ["mssd", "mspd"] + (["vsd"] if use_vsd else [])
    # Per error type: list of correctness thresholds (BOP19).
    th_lists = {
        "mssd": list(MSSD_THS),                   # x diameter (per-object)
        "mspd": list(MSPD_THS),                   # px (already image-scaled below)
        "vsd": list(VSD_THS),                     # one match per (tau, theta) pair
    }

    obj_ids = sorted(models.keys())
    scene_ids = sorted(scenes.keys())

    # --- 1) compute per (im, obj) the errors of every est vs every GT ----------
    # scene_errs[err_type] : list of dicts compatible with pose_matching.match_poses
    #   {im_id, obj_id, est_id, score, errors:{gt_id:[err]}, gt_visib_fracts:{gt_id:f}}
    # scene_gt[err_type-agnostic] : {im_id: [{obj_id}, ...]} (GT order = gt_id)
    # scene_gt_valid : {im_id: [bool,...]}  (visib + targets gate)
    per_scene = {}  # scene_id -> dict(scene_gt, scene_gt_info, scene_gt_valid, errs{type})
    n_gt_total = defaultdict(int)
    n_matched_dbg = defaultdict(int)

    for sid in scene_ids:
        sc = scenes[sid]
        gi_scene = scene_gt_info.get(sid, {})
        sgt, sgt_valid, sgt_info = {}, {}, {}
        errs = {et: [] for et in err_types}

        for im_id, gt_insts in sc["gt"].items():
            cam = sc["cam"][im_id]
            K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
            width = _img_width(K)
            mspd_scale = MSPD_REF_WIDTH / width if width else 1.0

            depth_test = _load_depth(sc, im_id, cam) if use_vsd else None

            # GT list in stable gt_id order
            gt_list = [{"obj_id": g["obj_id"]} for g in gt_insts]
            gt_R = [np.array(g["cam_R_m2c"], dtype=np.float64).reshape(3, 3) for g in gt_insts]
            gt_t = [np.array(g["cam_t_m2c"], dtype=np.float64).reshape(3, 1) for g in gt_insts]

            info_list = gi_scene.get(im_id, [{} for _ in gt_insts])
            visib = [float(info_list[j].get("visib_fract", 1.0)) if j < len(info_list)
                     else 1.0 for j in range(len(gt_insts))]
            info_norm = [{"visib_fract": visib[j]} for j in range(len(gt_insts))]

            # valid GT = visible enough AND (if targets given) within inst_count.
            # OFFICIAL BOP semantics: a targets file IS the list of (scene,im) pairs
            # to evaluate — an image NOT in the targets file contributes NO valid GT
            # (it is not part of this eval's denominator). This lets a batch-eval that
            # only predicts a fixed subset of images (e.g. im=0 per scene) be scored
            # over exactly that subset instead of the whole split (else recall ~0).
            # `targets` empty/None (no file) → unrestricted (every visible GT counts).
            tgt = targets.get((sid, im_id)) if targets else None
            im_in_targets = (not targets) or ((sid, im_id) in targets)
            valid = []
            obj_seen = defaultdict(int)
            for j, g in enumerate(gt_insts):
                ok = (visib[j] >= VISIB_GT_MIN) and im_in_targets
                if ok and tgt is not None:
                    oid = g["obj_id"]
                    obj_seen[oid] += 1
                    if oid not in tgt or obj_seen[oid] > tgt[oid]:
                        ok = False
                valid.append(bool(ok))
                if ok:
                    n_gt_total[g["obj_id"]] += 1

            sgt[im_id] = gt_list
            sgt_valid[im_id] = valid
            sgt_info[im_id] = info_norm

            # estimates for this image
            ests = preds_by_img.get((sid, im_id), [])
            # group GT indices by obj for fast per-obj error dict
            gt_idx_by_obj = defaultdict(list)
            for j, g in enumerate(gt_insts):
                gt_idx_by_obj[g["obj_id"]].append(j)

            for est_id, e in enumerate(ests):
                oid = e["obj_id"]
                if oid not in gt_idx_by_obj:
                    continue  # est of an object with no GT in this image -> ignored (FP)
                model = models.get(oid)
                if model is None:
                    continue
                model["obj_id"] = oid
                R_e, t_e = e["R"], e["t"]
                err_per_type = {et: {} for et in err_types}
                visib_per_gt = {}
                for gj in gt_idx_by_obj[oid]:
                    R_g, t_g = gt_R[gj], gt_t[gj]
                    visib_per_gt[gj] = visib[gj]
                    if "mssd" in err_types:
                        err_per_type["mssd"][gj] = [
                            float(pose_error.mssd(R_e, t_e, R_g, t_g,
                                                  model["pts"], model["syms"]))
                        ]
                    if "mspd" in err_types:
                        err_per_type["mspd"][gj] = [
                            float(pose_error.mspd(R_e, t_e, R_g, t_g, K,
                                                  model["pts"], model["syms"])) * mspd_scale
                        ]
                    if "vsd" in err_types and depth_test is not None:
                        vs = pose_error.vsd(
                            R_e, t_e, R_g, t_g, depth_test, K, VSD_DELTA,
                            VSD_TAUS, True, model["diameter"], renderer, oid)
                        err_per_type["vsd"][gj] = [float(v) for v in vs]
                for et in err_types:
                    if not err_per_type[et]:
                        continue
                    errs[et].append({
                        "im_id": im_id, "obj_id": oid, "est_id": est_id,
                        "score": float(e["score"]),
                        "errors": err_per_type[et],
                        "gt_visib_fracts": visib_per_gt,
                    })

        per_scene[sid] = {"scene_gt": sgt, "scene_gt_valid": sgt_valid,
                          "scene_gt_info": sgt_info, "errs": errs}

    # --- 2) per error type: match per threshold, aggregate recall --------------
    # AR_<et> = mean over thresholds of mean_obj_recall; also keep per-object.
    ar_per_type = {}                              # et -> overall AR
    ar_per_type_obj = {et: defaultdict(list) for et in err_types}  # et -> obj -> [recall_per_th]
    for et in err_types:
        if et == "vsd":
            # VSD: matching uses (theta) thresholds; one error vector per tau.
            # Reproduce eval_bop19: for each tau index, build a 1-elem error and
            # sweep theta thresholds. We average over taus x thetas.
            ths_iter = [("vsd", ti, th) for ti in range(len(VSD_TAUS)) for th in VSD_THS]
        else:
            ths_iter = [(et, None, th) for th in th_lists[et]]

        recalls_overall = []
        for (_, tau_i, th) in ths_iter:
            all_matches = []
            for sid in scene_ids:
                ps = per_scene[sid]
                # per-object correctness threshold (MSSD scales by diameter)
                errs_th = []
                for e in ps["errs"][et]:
                    if et == "mssd":
                        diam = models[e["obj_id"]]["diameter"]
                        corr = [th * diam]
                        ee = e
                    elif et == "mspd":
                        corr = [th]
                        ee = e
                    else:  # vsd: pick the tau slice
                        corr = [th]
                        ee = {**e, "errors": {g: [v[tau_i]] for g, v in e["errors"].items()}}
                    errs_th.append((ee, corr))
                # match_poses_scene wants ONE correct_th per object class; MSSD's
                # per-object diameter scaling means we match per object class.
                ms = _match_scene_icbin(sid, ps, errs_th, et)
                all_matches += ms
            sc_res = score.calc_localization_scores(
                scene_ids, obj_ids, all_matches, N_TOP, do_print=False)
            recalls_overall.append(sc_res["mean_obj_recall"])
            for oid in obj_ids:
                ar_per_type_obj[et][oid].append(sc_res["obj_recalls"].get(oid, 0.0))
        ar_per_type[et] = float(np.mean(recalls_overall)) if recalls_overall else 0.0

    # --- 3) assemble report in the evaluate() shape ----------------------------
    per_obj = {}
    ar_list, vsd_list, mssd_list, mspd_list = [], [], [], []
    for oid in obj_ids:
        ar_mssd = float(np.mean(ar_per_type_obj["mssd"][oid])) if "mssd" in err_types else 0.0
        ar_mspd = float(np.mean(ar_per_type_obj["mspd"][oid])) if "mspd" in err_types else 0.0
        ar_vsd = (float(np.mean(ar_per_type_obj["vsd"][oid]))
                  if "vsd" in err_types else None)
        components = [ar_mssd, ar_mspd] + ([ar_vsd] if ar_vsd is not None else [])
        ar = float(np.mean(components)) if components else 0.0
        per_obj[oid] = {
            "name": (obj_names or {}).get(oid, str(oid)),
            "symmetric": models[oid]["is_symmetric"],
            "n_sym_transforms": models[oid]["n_syms"],
            "diameter_mm": models[oid]["diameter"],
            "n_gt": n_gt_total[oid],
            "n_matched": None,
            "AR": ar, "AR_VSD": ar_vsd, "AR_MSSD": ar_mssd, "AR_MSPD": ar_mspd,
            "ADD_mean_mm": None, "ADI_mean_mm": None, "ADD_or_ADI_mean_mm": None,
            "trans_err_mean_mm": None, "trans_err_median_mm": None,
            "rot_err_mean_deg": None, "rot_err_median_deg": None,
            "rot_err_naive_mean_deg": None,
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
        "n_gt": sum(n_gt_total.values()),
        "n_matched": None,
        "vsd_enabled": use_vsd,
        "protocol": "ic_bin_bop19_localisation",
        "visib_gt_min": VISIB_GT_MIN,
        "n_top": N_TOP,
        "targets_used": targets is not None,
    }
    return {"per_object": per_obj, "overall": overall}


def _match_scene_icbin(scene_id, ps, errs_th, et):
    """Run the official per-image greedy match for one threshold.

    errs_th : list of (err_dict, [correct_th]) tuples. MSSD's correct_th is per
    object (diameter-scaled), so we group by obj and call match_poses per
    (im, obj) exactly like match_poses_scene does internally.
    """
    sgt = ps["scene_gt"]
    sgt_valid = ps["scene_gt_valid"]
    sgt_info = ps["scene_gt_info"]

    # seed matches: every GT instance, est_id=-1 (unmatched) — same as official.
    im_matches = {}
    for im_id, im_gts in sgt.items():
        im_matches[im_id] = [
            {"scene_id": scene_id, "im_id": im_id, "obj_id": g["obj_id"],
             "gt_id": gid, "est_id": -1, "score": -1, "error": -1, "error_norm": -1,
             "valid": sgt_valid[im_id][gid],
             "gt_visib_fract": sgt_info[im_id][gid]["visib_fract"]}
            for gid, g in enumerate(im_gts)
        ]

    # organise estimate errors by (im, obj)
    by_im_obj = defaultdict(list)
    th_by_im_obj = {}
    for e, corr in errs_th:
        by_im_obj[(e["im_id"], e["obj_id"])].append(e)
        th_by_im_obj[(e["im_id"], e["obj_id"])] = corr

    for (im_id, obj_id), e_list in by_im_obj.items():
        corr = th_by_im_obj[(im_id, obj_id)]
        ms = pose_matching.match_poses(e_list, corr, N_TOP if N_TOP > 0 else 0,
                                       sgt_valid[im_id])
        for m in ms:
            g = im_matches[im_id][m["gt_id"]]
            g["est_id"] = m["est_id"]
            g["score"] = m["score"]
            g["error"] = m["error"]
            g["error_norm"] = m["error_norm"]

    out = []
    for im_id in im_matches:
        out += im_matches[im_id]
    return out


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
    ap.add_argument("--icbin", action="store_true",
                    help="use the OFFICIAL BOP19/IC-BIN localisation protocol "
                         "(per-threshold error-based greedy match via "
                         "bop_toolkit pose_matching + inst_count denominator). "
                         "This is the multi-instance bin-picking AR.")
    ap.add_argument("--targets",
                    help="BOP targets file (inst_count per scene/im/obj). "
                         "Default for --icbin: "
                         "<dataset-dir>/test_targets_<dataset>_<split>.json if present.")
    ap.add_argument("--n-points", type=int, default=2000,
                    help="subsample mesh to N points for point metrics "
                         "(0 = full mesh; default 2000 keeps MSSD/MSPD tractable)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap images per scene (0 = all; for fast smoke runs)")
    ap.add_argument("--out", help="output dir for report.json + report.txt")
    args = ap.parse_args()

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
    scenes = load_scene_data(args.dataset_dir, args.split, max_images=args.max_images)
    n_imgs = sum(len(sc['gt']) for sc in scenes.values())
    sys.stderr.write(f"[eval_bop] split '{args.split}': {len(scenes)} scenes, "
                     f"{n_imgs} images\n")

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
        if args.icbin:
            # locate the targets file (inst_count) for the multi-instance denominator
            targets_path = args.targets
            if not targets_path:
                ds_name = os.path.basename(os.path.normpath(args.dataset_dir))
                cand = os.path.join(args.dataset_dir,
                                    f"test_targets_{ds_name}_{args.split}.json")
                targets_path = cand if os.path.isfile(cand) else None
            targets = load_targets(targets_path)
            sys.stderr.write(
                f"[eval_bop] IC-BIN protocol: targets="
                f"{targets_path if targets else 'NONE (denominator = all visible GT)'}"
                f" visib_gt_min={VISIB_GT_MIN} n_top={N_TOP}\n")
            scene_gt_info = load_scene_gt_info(args.dataset_dir, args.split,
                                               sorted(scenes.keys()))
            results = evaluate_icbin(scenes, models, preds, scene_gt_info, targets,
                                     args.vsd, obj_names)
            title = (f"BOP IC-BIN eval (official localisation) - split={args.split} "
                     f"preds={args.preds}")
        else:
            results = evaluate(scenes, models, preds, args.vsd, obj_names)
            title = f"BOP eval - split={args.split} preds={args.preds}"
        report_txt = render_table(results, title)
        report_json = {"mode": "eval",
                       "protocol": "ic_bin" if args.icbin else "translation_match",
                       "split": args.split,
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
