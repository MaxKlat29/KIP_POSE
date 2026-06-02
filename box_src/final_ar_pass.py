#!/usr/bin/env python3
"""final_ar_pass.py - THE single (Max-mandated ONE-shot) AR measurement for T-089.

Runs the complete final pass on the box, in one go, against the EXISTING val set
+ CACHED GDRNPP coarse preds. NO sim, NO retrain, NO new generation.

For each (config, proxy) it:
  1. refines coarse_combined.csv via rc_refine_eval.refine_rows (refine_rc staggered
     gate = SHIPPED DEFAULT_MARGIN_SCHEDULE), writing a refined BOP-results CSV
  2. scores that CSV with eval_bop (sym-aware, BOP19 AR = mean(MSSD,MSPD)) on
       - the FULL val set
       - the partial-vis band [0.20, 0.50)  (the occlusion damage zone)
     collecting per-part AR (Anker_Kurz / Anker_Lang / Zahnrad).

Variants:
  baseline   = raw coarse preds (+ planar z-snap), no RC refine        -> "raw"
  fix_gt     = rc_anker_occ, proxy_mask=gt   (GT mask_visib = UPPER BOUND)
  fix_sam    = rc_anker_occ, proxy_mask=sam  (SAM seg of detector box = LIVE proxy)

Also reports the FLIP-RATE proxy (switched/refuted counts from refine_rows stats)
for fix_gt vs fix_sam so we can compare the live proxy against the GT upper bound.

USAGE (box, bop-venv):
  /mnt/data/bop/bop-venv/bin/python box_src/final_ar_pass.py \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
      --preds /mnt/data/bop/results/coarse_combined.csv \
      --out-dir /mnt/data/bop/results/t089_final
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
import eval_bop as EB        # noqa: E402

PART_IDS = {1: "Anker_Kurz", 2: "Anker_Lang", 6: "Zahnrad"}
PARTIAL_BAND = (0.20, 0.50)


def score_csv(csv_path, bop_root, models, obj_names, visib_band=None):
    """Run eval_bop on one CSV; return per-obj AR dict {obj_id: {AR, n_gt, n_matched}}."""
    scenes = EB.load_scene_data(bop_root, "val", max_images=0, visib_band=visib_band)
    preds = EB.preds_from_csv(csv_path)
    results = EB.evaluate(scenes, models, preds, use_vsd=False, obj_names=obj_names)
    po = results["per_object"]
    out = {}
    for oid in PART_IDS:
        if oid in po:
            out[oid] = {"AR": po[oid]["AR"], "n_gt": po[oid]["n_gt"],
                        "n_matched": po[oid]["n_matched"]}
        else:
            out[oid] = {"AR": None, "n_gt": 0, "n_matched": 0}
    out["_overall"] = {"AR": results["overall"]["AR"],
                       "n_gt": results["overall"]["n_gt"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-visib-fract", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # --- shared dataset state (load once) ---
    sys.stderr.write("[final] loading cams/meshes/gt/visib indices ...\n")
    cams = RR.load_cams(args.bop_root)
    meshes, downs = RR.load_meshes(args.bop_root)
    gt_index = RR.load_gt_index(args.bop_root)
    visib_index = RR.load_visib_index(args.bop_root)
    rows = RR.read_preds(args.preds)
    models, _models_dir = EB.load_models(args.bop_root, n_points=2000)
    di = os.path.join(args.bop_root, "dataset_info.json")
    obj_names = {}
    if os.path.isfile(di):
        obj_names = {int(k): v for k, v in
                     json.load(open(di)).get("obj_names", {}).items()}

    # --- the three variants (one refine each) ---
    variants = [
        ("baseline", "raw",           "gt"),   # no RC refine, just z-snap
        ("fix_gt",   "rc_anker_occ",  "gt"),    # UPPER BOUND
        ("fix_sam",  "rc_anker_occ",  "sam"),   # LIVE PROXY
    ]

    flip = {}     # variant -> refine stats (switched / refuted = proxy flip signal)
    ar_full = {}  # variant -> per-part AR (full set)
    ar_part = {}  # variant -> per-part AR (partial-vis band)

    for vname, cname, proxy in variants:
        cfg = RR.CONFIGS[cname]
        use_sched = bool(cfg.get("visaware"))
        sys.stderr.write(f"\n[final] === {vname}  (cfg={cname} proxy={proxy} "
                         f"sched={use_sched}) ===\n")
        t0 = time.time()
        ref, st = RR.refine_rows(
            rows, cfg, cams, meshes, downs, gt_index, args.bop_root,
            scorer="cpu_edge", visib_index=visib_index,
            min_visib_fract=args.min_visib_fract,
            occ_max_violation=0.30, occ_dilate=2, occ_vf_hi=0.0,
            proxy_mask=proxy, use_schedule=use_sched)
        flip[vname] = st
        csv_path = os.path.join(args.out_dir, f"preds_{vname}.csv")
        RR.write_csv(ref, csv_path)
        sys.stderr.write(f"[final] {vname} refine {time.time()-t0:.1f}s  "
                         f"stats={st}\n")

        ar_full[vname] = score_csv(csv_path, args.bop_root, models, obj_names)
        ar_part[vname] = score_csv(csv_path, args.bop_root, models, obj_names,
                                   visib_band=PARTIAL_BAND)
        sys.stderr.write(f"[final] {vname} scored (full+partial) "
                         f"{time.time()-t0:.1f}s total\n")

    report = {
        "task": "T-089 final AR pass",
        "preds": args.preds,
        "bop_root": args.bop_root,
        "n_coarse_preds": len(rows),
        "partial_band": PARTIAL_BAND,
        "schedule": (dict(RR.RC.DEFAULT_MARGIN_SCHEDULE)
                     if hasattr(RR.RC, "DEFAULT_MARGIN_SCHEDULE") else None),
        "flip_stats": flip,
        "ar_full": ar_full,
        "ar_partial_vis": ar_part,
        "runtime_s": round(time.time() - t_start, 1),
    }
    out_json = os.path.join(args.out_dir, "t089_final_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    sys.stderr.write(f"\n[final] wrote {out_json}\n")

    # --- human table to stdout ---
    def cell(d, oid):
        v = d.get(oid, {}).get("AR")
        return "  n/a " if v is None else f"{v:6.4f}"

    print("\n================ T-089 FINAL AR (sym-aware BOP19 AR=mean(MSSD,MSPD)) ================")
    print(f"coarse preds: {len(rows)} | full val + partial-vis band {PARTIAL_BAND}\n")
    for scope, table in (("FULL VAL SET", ar_full),
                         (f"PARTIAL-VIS {PARTIAL_BAND}", ar_part)):
        print(f"--- {scope} ---")
        hdr = f"{'part':12} {'n_gt':>6} | {'baseline':>9} {'fix_gt(UB)':>11} {'fix_sam(live)':>13}"
        print(hdr)
        for oid, name in PART_IDS.items():
            ngt = table["baseline"].get(oid, {}).get("n_gt", 0)
            print(f"{name:12} {ngt:>6} | {cell(table['baseline'],oid):>9} "
                  f"{cell(table['fix_gt'],oid):>11} {cell(table['fix_sam'],oid):>13}")
        ob = table["baseline"]["_overall"]["AR"]
        og = table["fix_gt"]["_overall"]["AR"]
        os_ = table["fix_sam"]["_overall"]["AR"]
        ngt = table["baseline"]["_overall"]["n_gt"]
        print(f"{'OVERALL':12} {ngt:>6} | {ob:9.4f} {og:11.4f} {os_:13.4f}\n")

    print("--- FLIP signal (refine stats) ---")
    print(f"{'variant':10} {'rc':>5} {'switched':>9} {'refuted':>8} {'proxy_fail':>11} {'no_mask':>8}")
    for v in ("baseline", "fix_gt", "fix_sam"):
        s = flip[v]
        print(f"{v:10} {s['rc']:>5} {s['switched']:>9} {s['refuted']:>8} "
              f"{s['proxy_fail']:>11} {s['no_mask']:>8}")
    print(f"\nruntime {report['runtime_s']}s")


if __name__ == "__main__":
    main()
