#!/usr/bin/env python3
"""Extract labelled, deskewed face snippets for classifier training.

Re-clusters each rendered faceset into its physical face-views (the very same
``cluster_views`` partition the registry is built from, so labels never drift)
and writes one square, uniform-size snippet per view, foldered by face:

    data/snippets/<part>/<face_id>/<idx>.png      # label == face_id

Snippets are square, depth-height-map crops (visible even when the RGB frame is
featureless grey), resized to ``--size`` (default 96) on a neutral-grey canvas
so aspect ratio is preserved. Fully generic — no per-part special-casing; every
faceset under ``--output-root`` that has an ``EXPECTED`` entry (or, with
``--all``, any faceset present) is processed identically.

Each part also gets ``snippets_meta.json`` recording the face_id -> {count,
prob, R} mapping read straight from the clustering, so the downstream manifest
and classifier know the class set and the resting-pose prior.

    python faces/extract_snippets.py                       # all known parts
    python faces/extract_snippets.py --only Zahnrad        # one part
    python faces/extract_snippets.py --all --size 96       # every faceset found
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_views as cv


def square_resize(img, size):
    """Pad an RGB crop to square on neutral grey, then resize to size×size."""
    h, w = img.shape[:2]
    sd = max(h, w)
    sq = np.full((sd, sd, 3), 128, np.uint8)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = img
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.BILINEAR))


def face_id(k):
    """Stable folder/label name for the k-th kept face (0-based) -> 'face_1'."""
    return f"face_{k + 1}"


def extract_part(ds, out_root, size, min_prob):
    """Cluster one faceset and dump labelled square snippets. Returns meta dict."""
    name = os.path.basename(ds.rstrip("/")).replace("faceset_", "")
    rows, deps, msks, g = cv.load_faceset(ds)
    n = len(rows)
    keep, rare_n, descs, n_gc, n_faces = cv.cluster_faces(
        rows, deps, msks, g, min_prob=min_prob)

    part_dir = os.path.join(out_root, name)
    meta = {"part": name, "n_views": n, "snippet_size": size,
            "convention": "world = R @ body (column)", "faces": []}
    total = 0
    for k, f in enumerate(keep):
        fid = face_id(k)
        fdir = os.path.join(part_dir, fid)
        os.makedirs(fdir, exist_ok=True)
        # clean the folder so re-runs never mix stale snippets from old thresholds
        for old in os.listdir(fdir):
            if old.endswith(".png"):
                os.remove(os.path.join(fdir, old))
        med = f[int(np.argmin(np.linalg.norm(descs[f] - descs[f].mean(0), axis=1)))]
        n_ok = 0
        for j in f:
            crop, ok = cv.deskew(None, deps[j], msks[j])
            n_ok += int(ok)
            Image.fromarray(square_resize(crop, size)).save(
                os.path.join(fdir, f"{rows[j]['i']:04d}.png"))
        total += len(f)
        meta["faces"].append({"face_id": fid, "count": int(len(f)),
                              "prob": round(len(f) / n, 4), "crops_ok": n_ok,
                              "R": [round(v, 6) for v in rows[med]["R"]]})
        print(f"[snip] {name}/{fid}: {len(f)} snippets ({size}x{size}) "
              f"crops_ok={n_ok}/{len(f)}")
    os.makedirs(part_dir, exist_ok=True)
    json.dump(meta, open(os.path.join(part_dir, "snippets_meta.json"), "w"), indent=2)
    print(f"[snip] {name}: {len(keep)} faces, {total} snippets -> {part_dir}"
          + (f"  ({rare_n} rare views dropped)" if rare_n else ""))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="data/output",
                    help="dir holding faceset_<part>/ subdirs")
    ap.add_argument("--snippets-root", default="data/snippets",
                    help="where to write data/snippets/<part>/<face_id>/*.png")
    ap.add_argument("--size", type=int, default=96, help="square snippet edge px")
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these part names")
    ap.add_argument("--all", action="store_true",
                    help="process every faceset_* found, not just the known set")
    args = ap.parse_args()

    from check_face_counts import EXPECTED
    if args.only:
        parts = args.only
    elif args.all:
        parts = sorted(d.replace("faceset_", "")
                       for d in os.listdir(args.output_root)
                       if d.startswith("faceset_"))
    else:
        parts = list(EXPECTED)

    done = 0
    for part in parts:
        ds = os.path.join(args.output_root, f"faceset_{part}")
        if not os.path.isdir(ds):
            print(f"[snip] SKIP {part}: {ds} not found")
            continue
        extract_part(ds, args.snippets_root, args.size, args.min_prob)
        done += 1
    print(f"[snip] done — {done}/{len(parts)} parts -> {args.snippets_root}")


if __name__ == "__main__":
    main()
