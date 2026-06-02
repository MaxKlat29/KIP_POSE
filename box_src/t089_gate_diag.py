#!/usr/bin/env python3
"""t089_gate_diag.py — INERTNESS DIAGNOSIS for the SAM-proxy gate (T-089).

NO refine, NO scoring, NO sim, NO retrain. For every Anker (obj 1,2) coarse
pred whose GT visib_fract falls in the occlusion damage zone [0.20, 0.50)
(plus a little context band [0.10, 0.60)) it computes the three quantities
that decide whether the staggered gate can EVER fire live:

  gt_vf        = BOP GT visib_fract (scene_gt_info.json) — the eval upper-bound
                 gate input. Distribution of gt_vf tells us where the GT-fix lives.
  proxy_vf_csil= SAM-px / rendered-coarse-silhouette-px  (= TODAY's live gate_vf,
                 rc_refine_eval line ~313). THIS is what makes the gate inert.
  proxy_vf_full= SAM-px / full-GT-mask-px (diagnostic reference: what the proxy
                 WOULD read if the denominator were the true full silhouette).
  sam_over_box = SAM-px / detector-box-area (the pose-independent proxy_visib_fract
                 from sam_proxy_mask.py — the *other* candidate live scale).

Goal: show WHY proxy_vf_csil drops Anker instances under vf_occ_lo=0.20 / under
min_visib_fract=0.25 so the staggered gate never reaches the [0.20,0.50) band,
and which proxy scale + recalibrated band WOULD let the fix trigger live.

USAGE (box, bop-venv with CUDA for SAM):
  /mnt/data/bop/bop-venv/bin/python box_src/t089_gate_diag.py \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
      --preds /mnt/data/bop/results/coarse_combined.csv \
      --out /mnt/data/bop/results/t089_gate_diag.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "project"))
sys.path.insert(0, HERE)

import rc_refine_eval as RR  # noqa: E402
import refine_rc as RC       # noqa: E402
import bop_adapter as A      # noqa: E402
import sam_proxy_mask as SP  # noqa: E402

ANKER = {1, 2}
CONTEXT_BAND = (0.10, 0.60)   # measure a bit wider than the [0.20,0.50) damage zone


def pctl(a, ps):
    a = np.asarray(a, float)
    if len(a) == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=list(CONTEXT_BAND))
    args = ap.parse_args()

    lo, hi = args.band
    cams = RR.load_cams(args.bop_root)
    meshes, _downs = RR.load_meshes(args.bop_root)
    gt_index = RR.load_gt_index(args.bop_root)
    visib_index = RR.load_visib_index(args.bop_root)
    rows = RR.read_preds(args.preds)

    used = set()
    recs = []
    sys.stderr.write(f"[diag] {len(rows)} coarse preds; Anker {ANKER}; "
                     f"GT-vf context band [{lo},{hi})\n")
    n_seen = 0
    for r in rows:
        oid = r["obj_id"]
        if oid not in ANKER:
            continue
        sid, im = r["scene_id"], r["im_id"]
        inst_idx = RR.match_gt_inst(gt_index, sid, im, oid, r["t"], used)
        if inst_idx is None:
            continue
        gt_vf = visib_index.get((sid, im, inst_idx))
        if gt_vf is None or gt_vf < 0:
            continue
        # only diagnose the context band (cheap: skip well-vis majority)
        if not (lo <= gt_vf < hi):
            continue
        full = RR.load_full_mask(args.bop_root, sid, im, inst_idx)
        vis = RR.load_mask(args.bop_root, sid, im, inst_idx)
        if full is None or vis is None:
            continue
        rgb = RR.load_rgb(args.bop_root, sid, im)
        if rgb is None:
            continue
        bbox = RR.mask_bbox(full)
        # SAM proxy (pose-independent: box = full-mask AABB)
        pm = SP.sam_visible_mask(rgb, [bbox[0], bbox[1], bbox[2], bbox[3]])
        if pm is None or pm.shape != full.shape:
            continue
        sam_px = int(pm.sum())
        full_px = int(full.sum())
        vis_px = int(vis.sum())
        box_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        # rendered coarse silhouette at coarse pose (the TODAY denominator)
        cam = cams[sid][str(im)]
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        R_world, t_world = A.bop_pose_to_world(r["R"], r["t"], R_w2c, t_w2c,
                                               RR.TABLE_ORIGIN)
        verts_mm = np.asarray(meshes[oid].vertices, float)
        csil = RC.render_silhouette(verts_mm, R_world, t_world, RR.TABLE_ORIGIN,
                                    R_w2c, t_w2c, K, full.shape)
        csil_px = int(np.asarray(csil, bool).sum())

        proxy_vf_csil = (sam_px / csil_px) if csil_px > 0 else None
        proxy_vf_full = (sam_px / full_px) if full_px > 0 else None
        sam_over_box = sam_px / box_area
        # how well does SAM track the actual visible region?
        inter_vis = int(np.logical_and(pm, vis).sum())
        iou_vis = inter_vis / max(1, int(np.logical_or(pm, vis).sum()))
        recs.append({
            "scene": sid, "im": im, "obj": oid, "inst": inst_idx,
            "gt_vf": gt_vf, "proxy_vf_csil": proxy_vf_csil,
            "proxy_vf_full": proxy_vf_full, "sam_over_box": sam_over_box,
            "sam_px": sam_px, "full_px": full_px, "vis_px": vis_px,
            "csil_px": csil_px, "iou_sam_vs_gtvis": iou_vis,
        })
        n_seen += 1
        if n_seen % 25 == 0:
            sys.stderr.write(f"[diag] {n_seen} band-instances done\n")
        SP.free_image()

    # ── aggregate per part + overall ──
    def agg(subset):
        gt = [d["gt_vf"] for d in subset]
        pc = [d["proxy_vf_csil"] for d in subset if d["proxy_vf_csil"] is not None]
        pf = [d["proxy_vf_full"] for d in subset if d["proxy_vf_full"] is not None]
        sb = [d["sam_over_box"] for d in subset]
        iou = [d["iou_sam_vs_gtvis"] for d in subset]
        ps = [10, 25, 50, 75, 90]
        # fraction that PASSES each candidate gate input (>= threshold)
        def frac_ge(vals, thr):
            vals = [v for v in vals if v is not None]
            return (float(np.mean([v >= thr for v in vals])) if vals else None)
        return {
            "n": len(subset),
            "gt_vf": pctl(gt, ps),
            "proxy_vf_csil": pctl(pc, ps),
            "proxy_vf_full": pctl(pf, ps),
            "sam_over_box": pctl(sb, ps),
            "iou_sam_vs_gtvis": pctl(iou, ps),
            # gate-trigger diagnostics: how many land in occluded band [0.20,0.50)
            # vs get pushed under vf_occ_lo=0.20 vs under min_visib_fract=0.25
            "in_band_020_050": {
                "gt_vf": frac_in(gt, 0.20, 0.50),
                "proxy_vf_csil": frac_in([d["proxy_vf_csil"] for d in subset], 0.20, 0.50),
                "proxy_vf_full": frac_in([d["proxy_vf_full"] for d in subset], 0.20, 0.50),
                "sam_over_box": frac_in(sb, 0.20, 0.50),
            },
            "under_vf_occ_lo_020": {
                "gt_vf": frac_lt(gt, 0.20),
                "proxy_vf_csil": frac_lt([d["proxy_vf_csil"] for d in subset], 0.20),
                "proxy_vf_full": frac_lt([d["proxy_vf_full"] for d in subset], 0.20),
                "sam_over_box": frac_lt(sb, 0.20),
            },
            "under_min_visib_fract_025": {
                "gt_vf": frac_lt(gt, 0.25),
                "proxy_vf_csil": frac_lt([d["proxy_vf_csil"] for d in subset], 0.25),
                "proxy_vf_full": frac_lt([d["proxy_vf_full"] for d in subset], 0.25),
                "sam_over_box": frac_lt(sb, 0.25),
            },
        }

    report = {
        "task": "T-089 gate inertness diagnosis",
        "context_band": [lo, hi],
        "n_anker_band_instances": len(recs),
        "schedule": dict(RC.DEFAULT_MARGIN_SCHEDULE),
        "min_visib_fract": 0.25,
        "overall": agg(recs) if recs else {"n": 0},
        "by_part": {str(oid): agg([d for d in recs if d["obj"] == oid])
                    for oid in sorted(ANKER)},
        "records": recs,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    sys.stderr.write(f"[diag] wrote {args.out} ({len(recs)} records)\n")

    # human summary
    o = report["overall"]
    print(f"\n==== T-089 GATE INERTNESS DIAG  ({len(recs)} Anker in GT-vf [{lo},{hi})) ====")
    print(f"median  gt_vf={o['gt_vf']['p50']}  proxy_vf_csil={o['proxy_vf_csil']['p50']}  "
          f"proxy_vf_full={o['proxy_vf_full']['p50']}  sam_over_box={o['sam_over_box']['p50']}")
    print(f"SAM-vs-GTvis IoU median={o['iou_sam_vs_gtvis']['p50']}")
    print("\n-- fraction landing in occluded band [0.20,0.50) (where fix triggers) --")
    for k, v in o["in_band_020_050"].items():
        print(f"   {k:16}: {v}")
    print("-- fraction pushed under vf_occ_lo=0.20 (gate -> +inf, NEVER fires) --")
    for k, v in o["under_vf_occ_lo_020"].items():
        print(f"   {k:16}: {v}")
    print("-- fraction under min_visib_fract=0.25 (visib-gated out) --")
    for k, v in o["under_min_visib_fract_025"].items():
        print(f"   {k:16}: {v}")


def frac_in(vals, lo, hi):
    vals = [v for v in vals if v is not None]
    return (float(np.mean([(lo <= v < hi) for v in vals])) if vals else None)


def frac_lt(vals, thr):
    vals = [v for v in vals if v is not None]
    return (float(np.mean([v < thr for v in vals])) if vals else None)


if __name__ == "__main__":
    main()
