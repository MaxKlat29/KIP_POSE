#!/usr/bin/env python3
"""
Post-process SDG labels: drop the 2D bounding boxes of parts that are (almost)
fully occluded — e.g. lying completely behind the robot arm.

Replicator's bounding_box_2d_tight still emits a box for an object as long as a
single pixel is visible; the box then spans the occluder. Training a detector on
such a label teaches it to fire on the robot arm. This step removes any box
whose occlusionRatio (6th field) >= --max-occlusion.

    python sim_code/postprocess_labels.py --dir data/output/physvar2 --max-occlusion 0.9

By default it rewrites the bbox_2d_*.json in place (keeping a *.raw.json backup).
"""
import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-occlusion", type=float, default=0.9)
    ap.add_argument("--no-backup", action="store_true",
                    help="don't keep a .raw.json copy of the original labels")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "bbox_2d_*.json"))
                   if not f.endswith(".raw.json"))
    total = kept = dropped = 0
    for f in files:
        d = json.load(open(f))
        rows = d.get("data", [])
        total += len(rows)
        keep = [r for r in rows if (r[5] if len(r) > 5 else 0.0) < args.max_occlusion]
        dropped += len(rows) - len(keep)
        kept += len(keep)
        d["data"] = keep
        # keep the info bookkeeping consistent with the surviving boxes
        info = d.get("info", {})
        surviving_ids = {r[0] for r in keep}
        if "bboxIds" in info:
            info["bboxIds"] = [i for i in info["bboxIds"] if i in surviving_ids]
        if not args.no_backup and not os.path.exists(f.replace(".json", ".raw.json")):
            shutil.copy(f, f.replace(".json", ".raw.json"))
        json.dump(d, open(f, "w"), indent=2)

    print(f"post-process {len(files)} file(s), occ>={args.max_occlusion}: "
          f"kept {kept}, dropped {dropped} of {total}")


if __name__ == "__main__":
    main()
