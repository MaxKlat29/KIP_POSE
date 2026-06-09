#!/usr/bin/env python3
"""T-166 probe: is RGB-D (FoundationPose / GigaPose-3D) unfairly penalised?

For both anchors (obj 1, obj 2), match each prediction to the nearest GT
instance (BOP greedy-by-translation, highest score first, same as eval_bop),
then for the matched pairs compute:
  (Test 1) sym-resolved rotation error  -> proves whether symmetry/rotation is fine
  (Test 2) pred->GT translation offset in CAMERA frame AND OBJECT(GT) frame
           -> constant offset (mesh/convention, fixable) vs random scatter (genuinely worse)
Read-only. Uses the bop_toolkit + models_eval syms from the actual eval inputs.
"""
import csv, json, os, sys, math
import numpy as np

RUN = "/mnt/data/kip_pose/project/temp/batch_eval/run-20260608T113628Z"
VAL = "/mnt/data/kip_pose/project/bop/pose_isaac/val"
MODELS_INFO = "/mnt/data/kip_pose/project/bop/pose_isaac/models_eval/models_info.json"
BT = "/mnt/data/bop/repos/bop_toolkit"
sys.path.insert(0, BT)
from bop_toolkit_lib import misc, pose_error

CFGS = {
    "FoundationPose(RGBD)": f"{RUN}/csv/yolo_obb__foundationpose.csv",
    "GigaPose-3D(RGBD)":    f"{RUN}/csv/yolo_obb__gigapose_rgbd.csv",
    "GDRNPP(RGB)":          f"{RUN}/csv/gdrnpp.csv",
}
ANCHORS = [1, 2]
MAX_SYM_DISC_STEP = 0.01

minfo = json.load(open(MODELS_INFO))
syms_by_obj = {}
for oid in ANCHORS:
    syms_by_obj[oid] = misc.get_symmetry_transformations(minfo[str(oid)], MAX_SYM_DISC_STEP)

def parse_mat(s):
    v = [float(x) for x in s.split()]
    return np.array(v[:9]).reshape(3, 3)

def parse_vec(s):
    return np.array([float(x) for x in s.split()][:3])

def load_csv(path):
    out = {}  # (scene,im,obj) -> list of {R,t,score}
    with open(path) as f:
        for r in csv.DictReader(f):
            sid, iid, oid = int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])
            if oid not in ANCHORS:
                continue
            out.setdefault((sid, iid, oid), []).append({
                "R": parse_mat(r["R"]), "t": parse_vec(r["t"]), "score": float(r["score"])})
    return out

def load_gt():
    out = {}  # (scene,im,obj) -> list of {R,t}
    for scene in sorted(os.listdir(VAL)):
        sgt = os.path.join(VAL, scene, "scene_gt.json")
        if not os.path.exists(sgt):
            continue
        sid = int(scene)
        gt = json.load(open(sgt))
        for im_str, insts in gt.items():
            iid = int(im_str)
            for g in insts:
                oid = int(g["obj_id"])
                if oid not in ANCHORS:
                    continue
                R = np.array(g["cam_R_m2c"]).reshape(3, 3)
                t = np.array(g["cam_t_m2c"]).astype(float)
                out.setdefault((sid, iid, oid), []).append({"R": R, "t": t})
    return out

def greedy_match(ests, gts):
    """est->gt by translation, highest score first (eval_bop.match_greedy)."""
    order = sorted(range(len(ests)), key=lambda i: -ests[i]["score"])
    used = set()
    pairs = []
    for ei in order:
        best, bd = None, 1e18
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            d = np.linalg.norm(ests[ei]["t"] - g["t"])
            if d < bd:
                bd, best = d, gi
        if best is not None:
            used.add(best)
            pairs.append((ei, best))
    return pairs

gt = load_gt()
results = {}
for name, path in CFGS.items():
    preds = load_csv(path)
    rows = []  # per matched instance
    for key, ests in preds.items():
        gts = gt.get(key, [])
        if not gts:
            continue
        oid = key[2]
        for ei, gi in greedy_match(ests, gts):
            e, g = ests[ei], gts[gi]
            syms = syms_by_obj[oid]
            rot_sym = min(float(pose_error.re(e["R"], g["R"].dot(s["R"]))) for s in syms)
            rot_naive = float(pose_error.re(e["R"], g["R"]))
            dt_cam = e["t"] - g["t"]                       # camera-frame offset (mm)
            dt_obj = g["R"].T.dot(dt_cam)                  # GT-object-frame offset (mm)
            rows.append({"obj": oid, "rot_sym": rot_sym, "rot_naive": rot_naive,
                         "dt_cam": dt_cam, "dt_obj": dt_obj,
                         "te": float(np.linalg.norm(dt_cam))})
    results[name] = rows

def stat(name, rows):
    print(f"\n========== {name}  (n_matched={len(rows)}) ==========")
    for oid in ANCHORS:
        sub = [r for r in rows if r["obj"] == oid]
        if not sub:
            continue
        rs = np.array([r["rot_sym"] for r in sub])
        rn = np.array([r["rot_naive"] for r in sub])
        te = np.array([r["te"] for r in sub])
        dtc = np.array([r["dt_cam"] for r in sub])
        dto = np.array([r["dt_obj"] for r in sub])
        diam = minfo[str(oid)]["diameter"]
        print(f"  obj {oid} (diam={diam:.1f}mm, n={len(sub)}):")
        print(f"    rot_sym   deg: mean {rs.mean():6.1f}  med {np.median(rs):6.1f}  max {rs.max():6.1f}")
        print(f"    rot_naive deg: mean {rn.mean():6.1f}  med {np.median(rn):6.1f}")
        print(f"    |t_err|   mm : mean {te.mean():7.1f}  med {np.median(te):7.1f}  max {te.max():7.1f}  (MSSD<0.1*diam={0.1*diam:.1f}mm)")
        print(f"    dt_CAM mean [x,y,z] mm: [{dtc[:,0].mean():7.1f} {dtc[:,1].mean():7.1f} {dtc[:,2].mean():7.1f}]")
        print(f"    dt_CAM std  [x,y,z] mm: [{dtc[:,0].std():7.1f} {dtc[:,1].std():7.1f} {dtc[:,2].std():7.1f}]")
        print(f"    dt_OBJ mean [x,y,z] mm: [{dto[:,0].mean():7.1f} {dto[:,1].mean():7.1f} {dto[:,2].mean():7.1f}]  (constant-offset signature)")
        print(f"    dt_OBJ std  [x,y,z] mm: [{dto[:,0].std():7.1f} {dto[:,1].std():7.1f} {dto[:,2].std():7.1f}]  (scatter)")
        # constant-vs-random: |mean|/std per object-axis. >>1 => constant, <<1 => random
        for ax, lab in enumerate("XYZ"):
            m, s = abs(dto[:, ax].mean()), dto[:, ax].std() + 1e-9
            print(f"      OBJ-{lab}: |mean|/std = {m/s:5.2f}  ({'CONSTANT' if m/s > 1.5 else 'random' if m/s < 0.7 else 'mixed'})")

for name, rows in results.items():
    stat(name, rows)

# Per-instance dump for the worst (FoundationPose obj1) to eyeball scatter
print("\n--- FoundationPose obj1 per-instance dt_OBJ (mm) ---")
for r in [r for r in results["FoundationPose(RGBD)"] if r["obj"] == 1]:
    d = r["dt_obj"]
    print(f"   te={r['te']:6.1f}  rot_sym={r['rot_sym']:5.1f}  dt_obj=[{d[0]:7.1f} {d[1]:7.1f} {d[2]:7.1f}]")

print("\n--- FoundationPose obj2 per-instance (te, rot_sym) sorted by te ---")
_o2 = sorted([r for r in results["FoundationPose(RGBD)"] if r["obj"] == 2], key=lambda x: -x["te"])
for r in _o2:
    print("   te=%7.1f  rot_sym=%6.1f  dt_obj=[%7.1f %7.1f %7.1f]" % (
        r["te"], r["rot_sym"], r["dt_obj"][0], r["dt_obj"][1], r["dt_obj"][2]))
