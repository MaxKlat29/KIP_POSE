#!/usr/bin/env python3
"""anker_c2_check.py — decide the Anker 180° end-to-end flip symmetry from the CAD
(PHASE2_PLAN.md §2.1 / Fix-3, T-050).

THE QUESTION
  Our eval shows a 13-19% ≥90° "flip tail" on Anker_Kurz/Anker_Lang that the
  declared continuous-Y symmetry does NOT absorb: the cont-Y group canonicalises a
  TWIST about the long (Y) axis, but the catastrophic tail is an end-to-end
  180° FLIP about a TRANSVERSE axis (X or Z) — head swapped with tail. If the
  stab is genuinely 180°-flip-symmetric, that flip is a REAL symmetry and we must
  declare a discrete C_2 about the transverse axis so BOTH the PM loss and the
  metric forgive it (collapses the tail analytically, zero extra training cost,
  exactly like cont-Y collapses the twist). If the flip is NOT a true symmetry
  (some feature breaks it — a notch, an asymmetric head), declaring C_2 would make
  a CORRECT pose score as flip-equivalent → WRONG. So we must NOT declare blind:
  we measure the geometry.

METHOD (geometric, not a guess)
  For each candidate transverse axis a ∈ {X=[1,0,0], Z=[0,0,1]} (the long axis is
  Y per part_meta extents — long dimension is Y), rotate the mesh 180° about an
  axis through its centroid and compare the rotated point set to the original:
    1. symmetric Chamfer distance  D_chamfer = mean over both directions of the
       nearest-neighbour distance between the original surface samples and the
       180°-rotated surface samples (mm). Normalised by the object diameter →
       D_rel = D_chamfer / diameter.
    2. volume / occupancy overlap via a voxelised IoU of the two point clouds
       (coarse voxel grid, IoU of occupied cells). High IoU = the flipped shape
       occupies (nearly) the same volume.
  A flip is a TRUE C_2 symmetry about axis `a` iff D_rel ≤ CHAMFER_REL_THR AND
  voxel_IoU ≥ IOU_THR. We pick the BEST-scoring transverse axis; cont-Y already
  covers the twist, so the new discrete C_2 (if any) is about the transverse axis.

  Surface sampling: trimesh.sample.sample_surface (area-weighted, N points) — more
  robust than raw vertices (which can be non-uniform), and a KD-tree gives the
  Chamfer NN distance. Deterministic seed.

OUTPUT (stdout, machine-parseable + a JSON verdict file)
  Per part: chamfer_rel, voxel_iou, per-axis numbers, and a verdict:
    ADD_C2  axis=[..]   → the data step should add a discrete C_2 about that axis
    KEEP_CONT_Y          → flip is NOT a true symmetry; leave cont-Y, rely on res+DR
  Final line:  ANKER_C2_VERDICT <json>   (consumed by phase2_chain.sh)

  --apply  : if verdict is ADD_C2, also PATCH models_info.json in --bop-root:
             append the C_2 discrete matrix to the part's symmetries_discrete
             (kept ALONGSIDE symmetries_continuous — the group is cont-Y ⊕ C_2,
             both valid). Idempotent (won't double-add). The chain re-runs deploy
             afterwards to regenerate fps_points/keypoints_3d from the new sym.

USAGE (bop-venv on the box)
  /mnt/data/bop/bop-venv/bin/python box_src/gdrnpp/anker_c2_check.py \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
      --apply --json-out /mnt/data/bop/logs/anker_c2_verdict.json
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

# Anker parts to test (obj_id, label). The long axis is Y (part_meta: extent_m
# long dim = index 1); the candidate flip axes are the two TRANSVERSE axes X,Z.
ANKER = [(1, "Anker_Kurz"), (2, "Anker_Lang")]
TRANSVERSE_AXES = {"X": np.array([1.0, 0.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}

# Decision thresholds (relative to object diameter so they generalise to any size).
# A genuine 180° flip symmetry of a near-prismatic stab gives a very small
# normalised Chamfer (well under a few % of the diameter) and a high voxel IoU.
CHAMFER_REL_THR = 0.06   # ≤6% of the diameter mean-surface displacement
IOU_THR = 0.80           # ≥80% voxel-occupancy overlap
N_SAMPLE = 20000         # surface samples for Chamfer/IoU
VOXEL_DIV = 48           # voxel grid resolution along the longest extent
SEED = 0


def rot_about_axis_180(axis):
    """4x4? no — 3x3 rotation by 180° about a unit axis (Rodrigues, θ=π):
    R = 2 a aᵀ - I  for a unit vector a."""
    a = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(a, a) - np.eye(3)


def symmetric_chamfer(P, Q):
    """Mean symmetric nearest-neighbour distance between point sets P and Q (mm)."""
    try:
        from scipy.spatial import cKDTree
        d_pq = cKDTree(Q).query(P, k=1)[0]
        d_qp = cKDTree(P).query(Q, k=1)[0]
    except Exception:
        # pure-numpy fallback (slower) — chunked to bound memory
        def nn(A, B):
            out = np.empty(len(A))
            for i in range(0, len(A), 2000):
                chunk = A[i:i + 2000]
                dd = np.linalg.norm(chunk[:, None, :] - B[None, :, :], axis=2)
                out[i:i + 2000] = dd.min(axis=1)
            return out
        d_pq = nn(P, Q)
        d_qp = nn(Q, P)
    return 0.5 * (float(d_pq.mean()) + float(d_qp.mean()))


def voxel_iou(P, Q, div=VOXEL_DIV):
    """IoU of the occupied voxel sets of two point clouds on a shared grid."""
    allpts = np.vstack([P, Q])
    mn = allpts.min(0)
    mx = allpts.max(0)
    extent = (mx - mn)
    vox = extent.max() / div
    if vox <= 0:
        return 0.0

    def occ(X):
        idx = np.floor((X - mn) / vox).astype(np.int64)
        return set(map(tuple, idx))
    sp, sq = occ(P), occ(Q)
    inter = len(sp & sq)
    union = len(sp | sq)
    return inter / union if union else 0.0


def load_mesh(bop_root, oid):
    p = os.path.join(bop_root, "models", f"obj_{oid:06d}.ply")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return trimesh.load(p, force="mesh")


def diameter_of(mesh):
    """BOP diameter ≈ max pairwise vertex distance (convex-hull subsample)."""
    try:
        hull = mesh.convex_hull.vertices
    except Exception:
        hull = mesh.vertices
    if len(hull) > 1500:
        idx = np.random.default_rng(SEED).choice(len(hull), 1500, replace=False)
        hull = hull[idx]
    d = 0.0
    for i in range(len(hull)):
        dd = np.linalg.norm(hull[i + 1:] - hull[i], axis=1)
        if dd.size:
            d = max(d, float(dd.max()))
    return d


def check_part(mesh, label):
    """Return the per-axis Chamfer/IoU and the best transverse-axis verdict."""
    rng = np.random.default_rng(SEED)
    # area-weighted surface samples (deterministic)
    P, _ = trimesh.sample.sample_surface(mesh, N_SAMPLE, seed=SEED) \
        if "seed" in trimesh.sample.sample_surface.__code__.co_varnames \
        else (mesh.sample(N_SAMPLE), None)
    P = np.asarray(P, float)
    c = P.mean(0)            # centroid of the sampled surface (flip pivot)
    diam = diameter_of(mesh)

    per_axis = {}
    best = None
    for name, axis in TRANSVERSE_AXES.items():
        R = rot_about_axis_180(axis)
        Q = (R @ (P - c).T).T + c          # 180°-flipped about transverse axis thru centroid
        cham = symmetric_chamfer(P, Q)
        cham_rel = cham / diam if diam else 1.0
        iou = voxel_iou(P, Q)
        is_sym = (cham_rel <= CHAMFER_REL_THR) and (iou >= IOU_THR)
        per_axis[name] = {
            "axis": axis.tolist(),
            "chamfer_mm": round(cham, 4),
            "chamfer_rel": round(cham_rel, 5),
            "voxel_iou": round(iou, 4),
            "is_c2": bool(is_sym),
        }
        # rank: lowest chamfer_rel among symmetric ones; else lowest chamfer_rel
        score = (0 if is_sym else 1, cham_rel)
        if best is None or score < best[0]:
            best = (score, name)

    best_name = best[1]
    best_ax = per_axis[best_name]
    verdict = "ADD_C2" if best_ax["is_c2"] else "KEEP_CONT_Y"
    return {
        "label": label,
        "diameter_mm": round(diam, 3),
        "per_axis": per_axis,
        "best_axis": best_name,
        "best_axis_vec": per_axis[best_name]["axis"],
        "verdict": verdict,
        "thresholds": {"chamfer_rel": CHAMFER_REL_THR, "voxel_iou": IOU_THR},
    }


def c2_matrix_flat(axis, center_mm):
    """Row-major 4x4 flat for a 180° rotation about `axis` through center_mm.
    t = (I - R) @ c so the part rotates about its real centre (BOP convention,
    matches gen_models_info.discrete_matrix_flat)."""
    R = rot_about_axis_180(np.asarray(axis, float))
    t = (np.eye(3) - R) @ np.asarray(center_mm, float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return [float(x) for x in M.reshape(-1)]


def apply_to_models_info(bop_root, results):
    """Patch models_info.json: append the C_2 discrete matrix for any ADD_C2 part.
    Kept ALONGSIDE symmetries_continuous (the symmetry group is cont-Y ⊕ C_2_x/z).
    Idempotent: tagged with `_c2_added_by` so a re-run won't double-add. Also mirror
    to models_eval/ if present. Returns the list of changed obj_ids."""
    mi_path = os.path.join(bop_root, "models", "models_info.json")
    mi = json.load(open(mi_path))
    changed = []
    for oid, label in ANKER:
        res = results[label]
        if res["verdict"] != "ADD_C2":
            continue
        info = mi.get(str(oid))
        if info is None:
            continue
        # centre in mm from min_*/size_* (BOP models_info convention)
        center = [info["min_x"] + info["size_x"] / 2.0,
                  info["min_y"] + info["size_y"] / 2.0,
                  info["min_z"] + info["size_z"] / 2.0]
        if info.get("_c2_added_by") == "anker_c2_check":
            print(f"[anker_c2] obj {oid} already has C_2 (idempotent skip)", flush=True)
            continue
        mat = c2_matrix_flat(res["best_axis_vec"], center)
        info.setdefault("symmetries_discrete", [])
        info["symmetries_discrete"].append(mat)
        info["_c2_added_by"] = "anker_c2_check"
        info["_c2_axis"] = res["best_axis"]
        info["_c2_note"] = (f"C_2 180° about {res['best_axis']} (transverse), "
                            f"chamfer_rel={res['per_axis'][res['best_axis']]['chamfer_rel']}, "
                            f"voxel_iou={res['per_axis'][res['best_axis']]['voxel_iou']} "
                            f"— end-to-end flip is a true CAD symmetry")
        changed.append(oid)
        print(f"[anker_c2] obj {oid} ({label}): appended C_2 about {res['best_axis']}", flush=True)
    if changed:
        json.dump(mi, open(mi_path, "w"), indent=2)
        # mirror to models_eval/ if it exists (eval reads models_eval)
        me = os.path.join(bop_root, "models_eval", "models_info.json")
        if os.path.exists(me):
            json.dump(mi, open(me, "w"), indent=2)
            print(f"[anker_c2] mirrored to {me}", flush=True)
        print(f"[anker_c2] patched {mi_path} (obj {changed})", flush=True)
    else:
        print("[anker_c2] no changes to models_info.json (no ADD_C2 verdict)", flush=True)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="patch models_info.json with C_2 for ADD_C2 parts")
    ap.add_argument("--json-out", default="",
                    help="write the full verdict json here")
    args = ap.parse_args()

    results = {}
    for oid, label in ANKER:
        try:
            mesh = load_mesh(args.bop_root, oid)
        except FileNotFoundError as e:
            print(f"[anker_c2] MISSING mesh for {label}: {e}", file=sys.stderr)
            sys.exit(2)
        res = check_part(mesh, label)
        results[label] = res
        ax = res["per_axis"][res["best_axis"]]
        print(f"[anker_c2] {label}: diam={res['diameter_mm']}mm  "
              f"best_axis={res['best_axis']}  "
              f"chamfer_rel={ax['chamfer_rel']} (thr≤{CHAMFER_REL_THR})  "
              f"voxel_iou={ax['voxel_iou']} (thr≥{IOU_THR})  "
              f"=> {res['verdict']}", flush=True)
        # also show the loser axis for transparency
        for nm, d in res["per_axis"].items():
            if nm != res["best_axis"]:
                print(f"           ({nm}-axis: chamfer_rel={d['chamfer_rel']} "
                      f"iou={d['voxel_iou']} c2={d['is_c2']})", flush=True)

    changed = []
    if args.apply:
        changed = apply_to_models_info(args.bop_root, results)

    out = {"results": results, "applied": bool(args.apply), "changed_obj_ids": changed,
           "any_c2": any(r["verdict"] == "ADD_C2" for r in results.values())}
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"[anker_c2] verdict json -> {args.json_out}", flush=True)

    # single machine-parseable summary line for phase2_chain.sh
    print("ANKER_C2_VERDICT " + json.dumps(
        {lbl: {"verdict": r["verdict"], "axis": r["best_axis"],
               "chamfer_rel": r["per_axis"][r["best_axis"]]["chamfer_rel"],
               "voxel_iou": r["per_axis"][r["best_axis"]]["voxel_iou"]}
         for lbl, r in results.items()}), flush=True)


if __name__ == "__main__":
    main()
