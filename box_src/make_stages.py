#!/usr/bin/env python3
"""make_stages.py — produce per-stage pose_result.json files for the viewer.

Runs AFTER e2e_finish has staged the A/B variant CSVs (raw / zsnap / m1_zsnap /
rc_anker / +tta / auto-best). For ONE chosen val scene+image, this wraps
real_pose_result.py once per CSV and writes the result to:

    project/temp/pose_result_<stage>.json

The Three.js viewer's stage-selector (frontend/index.html) loads these via
?file= so the user can step through the pipeline visually.

This script is a thin orchestrator — the heavy lifting (CSV row -> world pose
via bop_adapter, schema validation) lives in real_pose_result.py + bop_adapter.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Map stage-name -> CSV filename suffix produced by e2e_finish A/B.
# Order is pipeline-order (left-to-right in the viewer toolbar).
STAGES = [
    ("raw",       "raw"),           # GDRNPP output, no refinement
    ("zsnap",     "zsnap"),         # + planar Z-Snap (translation locked)
    ("m1_zsnap",  "m1_zsnap"),      # + M1 stable-pose rot-snap
    ("m2",        "rc_anker"),      # + M2 render-and-compare (cpu_edge)
    ("tta",       "tta_m1_zsnap"),  # + TTA rotation vote on planar (optional)
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ab-dir", required=True,
                    help="dir with the A/B variant CSVs (e2e_finish out)")
    ap.add_argument("--final-csv", required=True,
                    help="final auto-best CSV (-> pose_result.json)")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--im", type=int, required=True)
    ap.add_argument("--dataset-dir", required=True,
                    help="BOP dataset root (pose_isaac)")
    ap.add_argument("--temp-dir", required=True,
                    help="project/temp — where pose_result_<stage>.json go")
    ap.add_argument("--real-pose-result", default=None,
                    help="path to box_src/real_pose_result.py (default: alongside)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rpr = Path(args.real_pose_result) if args.real_pose_result else here / "real_pose_result.py"
    if not rpr.is_file():
        sys.exit(f"FATAL: real_pose_result.py not found at {rpr}")

    ab = Path(args.ab_dir)
    temp = Path(args.temp_dir)
    temp.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for stage_name, csv_suffix in STAGES:
        csv = ab / f"preds_{csv_suffix}.csv"
        if not csv.is_file():
            print(f"[stages] skip {stage_name}: {csv} missing", flush=True)
            n_skip += 1
            continue
        out_json = temp / f"pose_result_{stage_name}.json"
        out_overlay = temp / f"overlay_{stage_name}.png"
        cmd = [
            sys.executable, str(rpr),
            "--dataset-dir", args.dataset_dir,
            "--split", "val",
            "--scene", str(args.scene),
            "--im", str(args.im),
            "--preds", str(csv),
            "--out-json", str(out_json),
            "--out-overlay", str(out_overlay),
        ]
        print(f"[stages] {stage_name:>10} <- {csv.name}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            n_ok += 1
            print(f"  -> {out_json.name}", flush=True)
        else:
            print(f"  -> FAILED rc={rc}", flush=True)
            n_skip += 1

    # Final = auto-best. Goes to the canonical pose_result.json.
    final_csv = Path(args.final_csv)
    if final_csv.is_file():
        out_json = temp / "pose_result.json"
        out_overlay = temp / "det_overlay.png"   # the canonical name the viewer expects
        cmd = [
            sys.executable, str(rpr),
            "--dataset-dir", args.dataset_dir,
            "--split", "val",
            "--scene", str(args.scene),
            "--im", str(args.im),
            "--preds", str(final_csv),
            "--out-json", str(out_json),
            "--out-overlay", str(out_overlay),
        ]
        print(f"[stages]      final <- {final_csv.name}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            n_ok += 1
            print(f"  -> {out_json.name}  (canonical)", flush=True)
        else:
            print(f"  -> FAILED rc={rc}", flush=True)
            n_skip += 1
    else:
        print(f"[stages] WARN: final csv {final_csv} missing", flush=True)
        n_skip += 1

    print(f"\n[stages] DONE — wrote {n_ok} stage(s), skipped {n_skip}", flush=True)
    print(f"          temp/: {sorted(p.name for p in temp.glob('pose_result_*.json'))}")


if __name__ == "__main__":
    main()
