#!/usr/bin/env python3
"""T-175 overlay renderer (runs ON the box).

For chosen (scene, im, obj) instances of yolo_seg__foundationpose (final run),
draw the object's 3D bounding box + a few mesh keypoints projected with the GT
pose (BLUE) and the predicted pose (RED) onto the RGB image, using the real
per-scene cam_K. Annotate each with translation error (mm) and sym-aware
rotation error (deg). Pure 2D projection (K @ (R@v + t)) + cv2 -- no renderer.

Saves PNGs to OUT_DIR on the box; they are pulled back to recon/ locally.
"""
import sys, json, csv, os
import numpy as np
import cv2
sys.path.insert(0, "/mnt/data/bop/repos/bop_toolkit")
from bop_toolkit_lib import inout, misc, pose_error

RUN  = "/mnt/data/kip_pose/project/temp/batch_eval/run-20260608T201857Z"
VAL  = "/mnt/data/kip_pose/project/bop/pose_isaac/val"
DSET = "/mnt/data/kip_pose/project/bop/pose_isaac"
CFG  = "yolo_seg__foundationpose"
MD   = os.path.join(DSET, "models_eval")
OUT  = "/tmp/t175_overlays"
os.makedirs(OUT, exist_ok=True)

minfo = json.load(open(os.path.join(MD, "models_info.json")))
NAMES = {1: "Anker_Kurz", 2: "Anker_Lang"}

# chosen cases: (scene, im, obj, tag, filename)
# (scene, im, obj, tag, filename, target_te_mm) -- target_te selects the specific
# matched instance (a scene can hold several anchors of the same obj_id).
CASES = [
    (9, 40, 1, "typical (~40mm off, rot ok)",               "t175_1_typical_40mm",   40.2),
    (1,  0, 2, "high-error 180deg cross-flip (T-166 obj2)", "t175_2_highflip_525mm", 524.9),
    (2, 60, 1, "good (GT ~= Pred)",                         "t175_3_good_4mm",         3.7),
    (7, 70, 1, "high-error cross-flip 58deg / 456mm",       "t175_4_flip_456mm",     456.2),
]

BLUE = (255, 80, 0)    # BGR -> blue   (GT)
RED  = (0, 60, 255)    # BGR -> red    (Pred)

def load_model(oid):
    ply = inout.load_ply(os.path.join(MD, f"obj_{oid:06d}.ply"))
    pts = ply["pts"]                       # mm, model frame
    syms = misc.get_symmetry_transformations(minfo[str(oid)], 0.01)
    return pts, syms

def bbox3d(pts):
    lo, hi = pts.min(0), pts.max(0)
    c = [[lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]]
    corners = np.array([[c[0][i], c[1][j], c[2][k]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    # edges between the 8 corners (index encodes i*4+j*2+k)
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:   # differ in exactly one axis
                edges.append((a, b))
    return corners, edges

def project(K, R, t_mm, V):
    X = (R @ V.T + t_mm.reshape(3, 1))      # 3xN, camera frame mm
    uv = (K @ X)
    uv = uv[:2] / uv[2]
    return uv.T                              # Nx2

def draw_box(img, K, R, t, corners, edges, color):
    uv = project(K, R, t, corners).astype(int)
    for a, b in edges:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)
    # mark the model-origin axes (small) for orientation legibility
    return uv

def draw_axes(img, K, R, t, color, length_mm=40):
    O = project(K, R, t, np.zeros((1, 3)))[0]
    for ax in np.eye(3) * length_mm:
        p = project(K, R, t, ax.reshape(1, 3))[0]
        cv2.arrowedLine(img, tuple(O.astype(int)), tuple(p.astype(int)), color, 2,
                        cv2.LINE_AA, tipLength=0.25)

def parse_mat(s): return np.array([float(x) for x in s.split()][:9]).reshape(3, 3)
def parse_vec(s): return np.array([float(x) for x in s.split()][:3])

# index CSV preds by (scene,im,obj)
preds = {}
for r in csv.DictReader(open(f"{RUN}/csv/{CFG}.csv")):
    sid, iid, oid = int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])
    preds.setdefault((sid, iid, oid), []).append(
        {"R": parse_mat(r["R"]), "t": parse_vec(r["t"]), "score": float(r["score"])})

def gt_for(sid, iid, oid):
    g = json.load(open(f"{VAL}/{sid:06d}/scene_gt.json"))[str(iid)]
    out = [x for x in g if int(x["obj_id"]) == oid]
    return [{"R": np.array(x["cam_R_m2c"], float).reshape(3, 3),
             "t": np.array(x["cam_t_m2c"], float)} for x in out]

def cam_for(sid, iid):
    c = json.load(open(f"{VAL}/{sid:06d}/scene_camera.json"))[str(iid)]
    return np.array(c["cam_K"], float).reshape(3, 3)

for sid, iid, oid, tag, fname, target_te in CASES:
    pts, syms = load_model(oid)
    corners, edges = bbox3d(pts)
    K = cam_for(sid, iid)
    rgb = cv2.imread(f"{VAL}/{sid:06d}/rgb/{iid:06d}.png")
    gts = gt_for(sid, iid, oid)
    ests = preds.get((sid, iid, oid), [])
    if not gts or not ests:
        print(f"SKIP {fname}: gts={len(gts)} ests={len(ests)}")
        continue
    # match exactly like eval_bop.match_greedy: highest-score est first claims the
    # nearest unclaimed GT. Take the est/gt pair that maximises the error so the
    # rendered case matches the picked (worst) instance for this (scene,im,obj).
    order = sorted(range(len(ests)), key=lambda i: -ests[i]["score"])
    used, pairs = [], []
    for ei in order:
        te_e = ests[ei]["t"]
        cand = [gi for gi in range(len(gts)) if gi not in used]
        if not cand:
            continue
        gi = min(cand, key=lambda gi: np.linalg.norm(te_e - gts[gi]["t"]))
        used.append(gi)
        pairs.append((ei, gi))
    # pick the matched pair whose translation error is closest to the curated
    # target_te (a scene may carry several anchors of the same obj_id).
    ei, gi = min(pairs,
                 key=lambda p: abs(np.linalg.norm(ests[p[0]]["t"] - gts[p[1]]["t"]) - target_te))
    e, g = ests[ei], gts[gi]

    te = float(np.linalg.norm(e["t"] - g["t"]))
    rsym = min(float(pose_error.re(e["R"], g["R"] @ s["R"])) for s in syms)
    rnaive = float(pose_error.re(e["R"], g["R"]))

    img = rgb.copy()
    draw_box(img, K, g["R"], g["t"], corners, edges, BLUE)   # GT blue
    draw_box(img, K, e["R"], e["t"], corners, edges, RED)    # Pred red
    draw_axes(img, K, g["R"], g["t"], BLUE)
    draw_axes(img, K, e["R"], e["t"], RED)

    # annotation panel
    lines = [
        f"{CFG}  scene{sid} im{iid}  {NAMES[oid]} (obj{oid})",
        f"GT (blue)   vs   Pred (red)",
        f"trans err = {te:6.1f} mm   (MSSD thr 0.1*diam = {0.1*minfo[str(oid)]['diameter']:.0f} mm)",
        f"rot err (sym) = {rsym:5.1f} deg   (naive {rnaive:5.1f} deg)",
        f"case: {tag}",
    ]
    y = 26
    for i, ln in enumerate(lines):
        col = (255, 255, 255)
        if i == 2: col = (0, 200, 255)       # te in amber
        if i == 3: col = (0, 255, 180)       # rot in green
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        y += 26
    out_path = os.path.join(OUT, fname + ".png")
    cv2.imwrite(out_path, img)
    print(f"WROTE {out_path}  te={te:.1f}mm rot_sym={rsym:.1f}deg rot_naive={rnaive:.1f}deg")
