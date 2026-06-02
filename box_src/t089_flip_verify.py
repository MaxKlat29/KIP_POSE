#!/usr/bin/env python3
"""t089_flip_verify.py — FLIP-RATE verifier for the calibrated SAM proxy (T-089).

NO scoring (no eval_bop), NO sim, NO retrain. Runs refine_rows (refine_rc staggered
gate) on the Anker coarse preds whose GT visib_fract is in the partial-vis band
[0.20,0.50) (the occlusion damage zone) and reports the SWITCH count (= flip
signal) for each gate variant:

  gt          = proxy_mask=gt   (BOP mask_visib, GT visib_fract)   -> UPPER BOUND
  sam_raw     = proxy_mask=sam, proxy_vf_scale=1.0                 -> TODAY (inert)
  sam_calib   = proxy_mask=sam, proxy_vf_scale=S, clip            -> CALIBRATED

"switched" rises means the calibrated gate actually reaches the aggressive
occluded band and flips poses; if sam_calib stays ~0 the proxy cannot trigger
the fix live regardless of band (-> honest "no live proxy separates the flip").

USAGE (box, bop-venv):
  /mnt/data/bop/bop-venv/bin/python box_src/t089_flip_verify.py \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds /mnt/data/bop/results/coarse_combined.csv \
    --scale 0.384 --out /mnt/data/bop/results/t089_flip_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "project"))
sys.path.insert(0, HERE)

import rc_refine_eval as RR  # noqa: E402

ANKER = {1, 2}
BAND = (0.20, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--scale", type=float, default=0.384,
                    help="calibrated proxy_vf scale (GT-median/proxy-median)")
    ap.add_argument("--cfg", default="rc_anker_occ",
                    help="rc_anker_occ (=staggered+free-space, shipped) or "
                         "rc_anker_vis (=T-085 staggered ONLY, no free-space refutation)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = RR.load_cams(args.bop_root)
    meshes, downs = RR.load_meshes(args.bop_root)
    gt_index = RR.load_gt_index(args.bop_root)
    visib_index = RR.load_visib_index(args.bop_root)
    all_rows = RR.read_preds(args.preds)

    # restrict to Anker preds whose GT visib_fract is in the partial-vis band
    # (match against GT once, reusing rc_refine_eval's matcher semantics).
    used = set()
    rows = []
    for r in all_rows:
        if r["obj_id"] not in ANKER:
            continue
        inst = RR.match_gt_inst(gt_index, r["scene_id"], r["im_id"], r["obj_id"],
                                r["t"], used)
        if inst is None:
            continue
        vf = visib_index.get((r["scene_id"], r["im_id"], inst))
        if vf is None or not (BAND[0] <= vf < BAND[1]):
            continue
        rows.append(r)
    sys.stderr.write(f"[flip] {len(rows)} Anker preds in GT-vf band {BAND}\n")

    cfg = RR.CONFIGS[args.cfg]   # rc_anker_occ=+free-space, rc_anker_vis=staggered only
    variants = [
        ("gt_upperbound", dict(proxy_mask="gt",  proxy_vf_scale=1.0, proxy_vf_clip=False)),
        ("sam_raw_today", dict(proxy_mask="sam", proxy_vf_scale=1.0, proxy_vf_clip=False)),
        ("sam_calibrated", dict(proxy_mask="sam", proxy_vf_scale=args.scale, proxy_vf_clip=True)),
    ]
    report = {"task": "T-089 flip-rate verify", "band": list(BAND), "cfg": args.cfg,
              "n_anker_band": len(rows), "scale": args.scale, "variants": {}}
    for vname, kw in variants:
        t0 = time.time()
        _ref, st = RR.refine_rows(
            rows, cfg, cams, meshes, downs, gt_index, args.bop_root,
            scorer="cpu_edge", visib_index=visib_index, min_visib_fract=0.25,
            occ_max_violation=0.30, occ_dilate=2, occ_vf_hi=0.0,
            use_schedule=True, **kw)
        rc = max(1, st["rc"])
        st["switch_rate"] = st["switched"] / rc
        st["runtime_s"] = round(time.time() - t0, 1)
        report["variants"][vname] = st
        sys.stderr.write(f"[flip] {vname:16} rc={st['rc']} switched={st['switched']} "
                         f"({100*st['switch_rate']:.2f}%) refuted={st['refuted']} "
                         f"proxy_fail={st['proxy_fail']} {st['runtime_s']}s\n")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n==== T-089 FLIP-RATE (Anker, GT-vf band {BAND}, n={len(rows)}) ====")
    print(f"{'variant':16} {'rc':>5} {'switched':>9} {'switch%':>8} {'refuted':>8} {'proxy_fail':>11}")
    for vname, _ in variants:
        s = report["variants"][vname]
        print(f"{vname:16} {s['rc']:>5} {s['switched']:>9} "
              f"{100*s['switch_rate']:>7.2f}% {s['refuted']:>8} {s['proxy_fail']:>11}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
