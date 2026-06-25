#!/usr/bin/env python3
"""T-175 probe: forensic recheck of the RGB-D 3D conversion chain.

Re-derives eval_bop's AR_MSSD / AR_MSPD *by hand* from the FINAL-RUN CSV poses,
using the SAME bop_toolkit metrics, the SAME models_eval PLY + syms, the SAME
greedy-by-translation match, the SAME thresholds, and -- crucially -- the SAME
per-scene cam_K read from scene_camera.json. If my hand numbers reproduce the
official report.json AR_MSSD/AR_MSPD, the conversion chain (gateway T_cam_obj ->
instances_to_doc -> world_to_bop_cam -> CSV) introduces NO error: the CSV pose
== the raw service pose, K is consistent inference<->eval, no double transform.

Also decomposes the pred->GT translation in CAMERA and OBJECT frame
(|mean|/std per axis: constant offset >>1 vs random scatter <<1) to re-confirm
T-166 on the FINAL run, and runs a K-sanity (the CSV t is K-independent).

Read-only.
"""
import csv, json, os, sys
import numpy as np

RUN  = "/mnt/data/kip_pose/project/temp/batch_eval/run-20260608T201857Z"
VAL  = "/mnt/data/kip_pose/project/bop/pose_isaac/val"
DSET = "/mnt/data/kip_pose/project/bop/pose_isaac"
BT   = "/mnt/data/bop/repos/bop_toolkit"
sys.path.insert(0, BT)
from bop_toolkit_lib import inout, misc, pose_error

CFG_NAME = "yolo_seg__foundationpose"
CSV_PATH = f"{RUN}/csv/{CFG_NAME}.csv"
REPORT   = f"{RUN}/eval/{CFG_NAME}/report.json"

ANCHORS = [1, 2]                      # the 2 anchors (primary 2-class scope)
MAX_SYM_DISC_STEP = 0.01
MSSD_THS = list(np.arange(0.05, 0.51, 0.05))   # x diameter (eval_bop)
MSPD_THS = list(np.arange(5, 51, 5))           # px (eval_bop, scaled by 640/width)
MSPD_REF_WIDTH = 640.0

# ---- load models_eval (pts + syms + diameter), exactly like eval_bop.load_models
models_dir = os.path.join(DSET, "models_eval")
if not os.path.isdir(models_dir):
    models_dir = os.path.join(DSET, "models")
# Replicate eval_bop.load_models EXACTLY: load_json(keys_to_int) -> iterate ALL objs
# in that dict order with ONE shared rng(0), subsample to n_points=2000. The shared
# seeded stream means obj 3/4/5 (even with 0 GT) consume draws -> byte-faithful pts.
N_POINTS = 2000        # eval_bop subprocess_eval default
models_info = inout.load_json(os.path.join(models_dir, "models_info.json"), keys_to_int=True)
_rng = np.random.default_rng(0)
MODELS = {}
for oid, info in models_info.items():
    ply = inout.load_ply(os.path.join(models_dir, f"obj_{oid:06d}.ply"))
    pts = ply["pts"]
    if N_POINTS and pts.shape[0] > N_POINTS:
        idx = _rng.choice(pts.shape[0], N_POINTS, replace=False)
        pts = pts[idx]
    if oid not in ANCHORS:
        continue
    MODELS[oid] = {
        "pts": pts,
        "syms": misc.get_symmetry_transformations(info, MAX_SYM_DISC_STEP),
        "diameter": float(info["diameter"]),
        "is_symmetric": bool(info.get("symmetries_continuous") or info.get("symmetries_discrete")),
    }
minfo = {str(k): v for k, v in models_info.items()}

# ---- which (scene,im) are actually scored (eval scopes to targets_scoped.json)
tgt_path = f"{RUN}/eval/{CFG_NAME}/targets_scoped.json"
SCOPED = None
if os.path.exists(tgt_path):
    SCOPED = set()
    for t in json.load(open(tgt_path)):
        SCOPED.add((int(t["scene_id"]), int(t["im_id"])))

def parse_mat(s): return np.array([float(x) for x in s.split()][:9]).reshape(3, 3)
def parse_vec(s): return np.array([float(x) for x in s.split()][:3]).reshape(3, 1)

def load_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            sid, iid, oid = int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])
            if oid not in ANCHORS: continue
            if SCOPED is not None and (sid, iid) not in SCOPED: continue
            out.setdefault((sid, iid, oid), []).append(
                {"R": parse_mat(r["R"]), "t": parse_vec(r["t"]), "score": float(r["score"])})
    return out

def load_gt_and_cam():
    gt, cam = {}, {}
    for scene in sorted(os.listdir(VAL)):
        sd = os.path.join(VAL, scene)
        sgt = os.path.join(sd, "scene_gt.json")
        scam = os.path.join(sd, "scene_camera.json")
        if not (os.path.exists(sgt) and os.path.exists(scam)): continue
        sid = int(scene)
        cam_all = json.load(open(scam))
        for k, c in cam_all.items():
            cam[(sid, int(k))] = np.array(c["cam_K"], float).reshape(3, 3)
        for im_str, insts in json.load(open(sgt)).items():
            iid = int(im_str)
            for g in insts:
                oid = int(g["obj_id"])
                if oid not in ANCHORS: continue
                gt.setdefault((sid, iid, oid), []).append({
                    "R": np.array(g["cam_R_m2c"], float).reshape(3, 3),
                    "t": np.array(g["cam_t_m2c"], float).reshape(3, 1)})
    return gt, cam

def match_greedy(ests, gts):
    order = sorted(range(len(ests)), key=lambda i: -ests[i]["score"])
    used, pairs = set(), []
    for ei in order:
        te = ests[ei]["t"].reshape(3)
        best, bd = None, None
        for gi, g in enumerate(gts):
            if gi in used: continue
            d = float(np.linalg.norm(te - g["t"].reshape(3)))
            if bd is None or d < bd: bd, best = d, gi
        if best is not None: used.add(best); pairs.append((ei, best))
    return pairs

preds = load_csv(CSV_PATH)
gt, cam = load_gt_and_cam()

# per-object accumulators (mssd/mspd lists + translation decomposition)
acc = {oid: {"mssd": [], "mspd": [], "n_gt": 0, "dt_cam": [], "dt_obj": [],
             "rot_sym": [], "te": []} for oid in ANCHORS}

# count n_gt over ALL scoped GT (denominator = misses count as 0)
for key, gts in gt.items():
    sid, iid, oid = key
    if SCOPED is not None and (sid, iid) not in SCOPED: continue
    acc[oid]["n_gt"] += len(gts)

for key, ests in preds.items():
    sid, iid, oid = key
    gts = gt.get(key, [])
    if not gts: continue
    K = cam[(sid, iid)]
    width = int(round(2.0 * K[0, 2]))
    mspd_scale = MSPD_REF_WIDTH / width if width else 1.0
    m = MODELS[oid]
    for ei, gi in match_greedy(ests, gts):
        R_e, t_e = ests[ei]["R"], ests[ei]["t"]
        R_g, t_g = gts[gi]["R"], gts[gi]["t"]
        mssd = float(pose_error.mssd(R_e, t_e, R_g, t_g, m["pts"], m["syms"]))
        mspd = float(pose_error.mspd(R_e, t_e, R_g, t_g, K, m["pts"], m["syms"])) * mspd_scale
        acc[oid]["mssd"].append(mssd)
        acc[oid]["mspd"].append(mspd)
        dt_cam = (t_e - t_g).reshape(3)            # camera-frame mm
        dt_obj = R_g.T @ dt_cam                    # GT-object-frame mm
        rot_sym = min(float(pose_error.re(R_e, R_g @ s["R"])) for s in m["syms"])
        acc[oid]["dt_cam"].append(dt_cam)
        acc[oid]["dt_obj"].append(dt_obj)
        acc[oid]["rot_sym"].append(rot_sym)
        acc[oid]["te"].append(float(np.linalg.norm(dt_cam)))

def recall(errs, ths, n_gt):
    if n_gt == 0: return 0.0
    return float(np.mean([sum(1 for e in errs if e <= th) / n_gt for th in ths]))

# ---- official numbers for the apples-to-apples comparison
rep = json.load(open(REPORT))["results"]["per_object"]

print(f"=== T-175 hand-recompute vs eval_bop  ({CFG_NAME}, final run) ===")
print(f"scoped (scene,im) pairs: {len(SCOPED) if SCOPED else 'ALL'}")
for oid in ANCHORS:
    a = acc[oid]
    diam = MODELS[oid]["diameter"]
    ar_mssd = recall(a["mssd"], [t * diam for t in MSSD_THS], a["n_gt"])
    ar_mspd = recall(a["mspd"], MSPD_THS, a["n_gt"])
    ar = 0.5 * (ar_mssd + ar_mspd)
    r = rep[str(oid)]
    print(f"\nobj {oid} {MODELS[oid]['diameter']:.1f}mm  n_matched={len(a['mssd'])}  n_gt(hand)={a['n_gt']} / report n_gt={r['n_gt']}")
    print(f"  HAND   AR_MSSD={ar_mssd:.4f}  AR_MSPD={ar_mspd:.4f}  AR={ar:.4f}")
    print(f"  REPORT AR_MSSD={r['AR_MSSD']:.4f}  AR_MSPD={r['AR_MSPD']:.4f}  AR={r['AR']:.4f}")
    d_mssd = abs(ar_mssd - r["AR_MSSD"]); d_mspd = abs(ar_mspd - r["AR_MSPD"])
    verdict = "MATCH (conversion lossless)" if (d_mssd < 0.02 and d_mspd < 0.02) else "MISMATCH -> bug suspect"
    print(f"  delta  MSSD={d_mssd:.4f}  MSPD={d_mspd:.4f}  -> {verdict}")

    # translation decomposition (constant offset vs random scatter)
    dtc = np.array(a["dt_cam"]); dto = np.array(a["dt_obj"]); te = np.array(a["te"])
    print(f"  |t_err| mm: mean {te.mean():6.1f}  med {np.median(te):6.1f}  max {te.max():6.1f}   (MSSD th0.1*diam={0.1*diam:.1f}mm)")
    print(f"  dt_CAM mean[x,y,z]={dtc.mean(0).round(1)}  std={dtc.std(0).round(1)}")
    print(f"  dt_OBJ mean[x,y,z]={dto.mean(0).round(1)}  std={dto.std(0).round(1)}")
    for ax, lab in enumerate("XYZ"):
        mm, ss = abs(dto[:, ax].mean()), dto[:, ax].std() + 1e-9
        tag = "CONSTANT" if mm/ss > 1.5 else "random" if mm/ss < 0.7 else "mixed"
        print(f"    OBJ-{lab}: |mean|/std={mm/ss:5.2f} ({tag})")

# ---- K-sanity: the CSV t is metric and K-independent. Prove the per-scene K is
# the SAME object eval uses, and that a deliberately wrong K would change MSPD but
# NOT te (the translation), confirming K cannot inflate the metric translation.
print("\n=== K-consistency / K-independence check (obj 1, first matched) ===")
for key, ests in list(preds.items())[:1]:
    sid, iid, oid = key
    if oid != 1:  # find an obj1 entry
        cand = [k for k in preds if k[2] == 1]
        if not cand: break
        key = cand[0]; sid, iid, oid = key; ests = preds[key]
    K = cam[(sid, iid)]
    gts = gt.get(key, [])
    if not gts: break
    pr = match_greedy(ests, gts)
    if not pr: break
    ei, gi = pr[0]
    R_e, t_e = ests[ei]["R"], ests[ei]["t"]; R_g, t_g = gts[gi]["R"], gts[gi]["t"]
    m = MODELS[1]
    te_mm = float(np.linalg.norm((t_e - t_g).reshape(3)))
    mspd_true = float(pose_error.mspd(R_e, t_e, R_g, t_g, K, m["pts"], m["syms"]))
    K_bad = K.copy(); K_bad[0, 0] *= 1.5; K_bad[1, 1] *= 1.5  # deliberately wrong fx/fy
    mspd_bad = float(pose_error.mspd(R_e, t_e, R_g, t_g, K_bad, m["pts"], m["syms"]))
    print(f"  scene{sid} im{iid} obj1: per-scene K fx={K[0,0]:.1f} cx={K[0,2]:.1f} (from scene_camera.json)")
    print(f"  te (metric translation) = {te_mm:.1f} mm  -- INDEPENDENT of K")
    print(f"  MSPD(correct K) = {mspd_true:.2f} px   MSPD(K*1.5) = {mspd_bad:.2f} px  (only MSPD moves, te fixed)")
