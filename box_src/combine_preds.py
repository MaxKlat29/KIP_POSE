#!/usr/bin/env python3
"""combine_preds.py - concatenate per-object GDRNPP BOP-results CSVs into one.

GDRNPP trains one model per object (single-object SO configs), so it writes one
predictions CSV per object under
  output/gdrn/poseIsaacPbrSO/<obj>/inference_/*/<obj>-iter0_pose_isaac-val.csv

eval_bop.py wants ONE BOP-results CSV covering all objects on the val split.
This just stacks them (keeping a single header) so the matcher in eval_bop.py
sees every obj_id's predictions per image.

Usage:
  python combine_preds.py out.csv in1.csv in2.csv [in3.csv ...]
"""
import sys


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: combine_preds.py OUT.csv IN1.csv [IN2.csv ...]\n")
        sys.exit(2)
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]
    rows = []
    header = "scene_id,im_id,obj_id,score,R,t,time"
    for p in in_paths:
        with open(p) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not lines:
            continue
        start = 1 if lines[0].startswith("scene_id") else 0
        rows.extend(lines[start:])
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")
    sys.stderr.write(
        f"[combine_preds] wrote {len(rows)} prediction rows from "
        f"{len(in_paths)} files -> {out_path}\n"
    )


if __name__ == "__main__":
    main()
