#!/usr/bin/env python3
"""Regression check for face-view discovery — expected vs actual face counts.

Runs the appearance-clustering of ``cluster_views`` over a fixed set of rendered
facesets and asserts that each part collapses to its *expected* number of
distinct face-views. This is the guard that keeps the two clustering fixes
honest as the thresholds / renderer evolve:

  * T-013 — no part may yield a blank (all-grey) template  -> per-face tmpl_ok
  * T-014 — a symmetric rod must not over-split            -> Anker_Lang == 1
  *         a genuinely two-sided part must not collapse    -> Zahnrad   == 2

Each part is PASS iff (a) the kept face count (prob >= min_prob) equals the
expected count AND (b) every kept face has a non-blank template. Prints a table
and exits non-zero if any part fails, so it drops straight into CI.

    python faces/check_face_counts.py [--output-root data/output] [--min-prob 0.02]
    python faces/check_face_counts.py --only Anker_Lang Zahnrad      # subset
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

# Soll-counts — the physically correct number of camera-distinguishable rest
# faces per part. Tied to the rendered facesets under <output-root>/faceset_<k>.
EXPECTED = {
    "Anker_Lang": 1,            # symmetric rod: one lying face (T-014 anchor)
    "Zahnrad": 2,              # flat-on-disc vs on-edge (T-014 anchor)
    "Poltopf_kurz_centered": 1,  # dome, lies one way (T-013 anchor)
    "Getriebegehaeuse_typ4": 1,  # lies one way (T-013 anchor)
    "Buerstenhalter_2polig": 2,  # two distinguishable rests
}


def template_is_blank(png_path):
    """A template is *blank* iff it carries essentially no visible structure.

    Not the same as ``tmpl_ok`` (which only records whether the deskew PCA path
    was taken). A correct depth height-map of a round part legitimately skips
    deskew yet is full of structure. Blank = nearly-uniform grey (the old
    failure: the whole frame at the cval=128 fill). We flag it when the per-pixel
    standard deviation is tiny AND almost every pixel sits at the neutral grey.
    """
    import numpy as np
    from PIL import Image
    if not os.path.exists(png_path):
        return True
    im = np.asarray(Image.open(png_path).convert("RGB")).astype(float)
    near_grey = np.mean(np.all(np.abs(im - 128) <= 6, axis=2))
    return float(im.std()) < 4.0 or near_grey > 0.97


def run_one(ds_dir, min_prob):
    """Cluster one faceset and return (kept_faces, blank_face_names)."""
    import cluster_views as cv

    # Drive cluster_views.main() with a synthetic argv, capturing its stdout so
    # the run stays quiet; the registration JSON it writes is the source of truth.
    argv = sys.argv
    sys.argv = ["cluster_views.py", ds_dir, "--min-prob", str(min_prob)]
    try:
        with redirect_stdout(io.StringIO()):
            cv.main()
    finally:
        sys.argv = argv

    name = os.path.basename(ds_dir.rstrip("/")).replace("faceset_", "")
    fdir = os.path.join(ds_dir, "faces")
    reg = json.load(open(os.path.join(fdir, f"faces_{name}.json")))
    faces = reg["faces"]                      # already filtered to >= min_prob
    blank = [f["name"] for f in faces
             if template_is_blank(os.path.join(fdir, f"tmpl_{f['name'].replace(' ', '')}.png"))]
    return faces, blank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="data/output",
                    help="dir holding faceset_<part>/ subdirs")
    ap.add_argument("--min-prob", type=float, default=0.02)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these part names (default: all known)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parts = args.only or list(EXPECTED)
    rows = []
    n_fail = 0
    for part in parts:
        exp = EXPECTED.get(part)
        ds = os.path.join(args.output_root, f"faceset_{part}")
        if exp is None:
            rows.append((part, "?", "?", "SKIP", "no expected count")); continue
        if not os.path.isdir(ds):
            rows.append((part, exp, "-", "MISS", "faceset dir not found")); n_fail += 1; continue
        try:
            faces, blank = run_one(ds, args.min_prob)
        except Exception as e:
            rows.append((part, exp, "ERR", "FAIL", str(e)[:48])); n_fail += 1; continue
        got = len(faces)
        ok = got == exp and not blank
        note = ""
        if got != exp:
            note = f"count {got}!={exp}"
        elif blank:
            note = f"blank tmpl: {','.join(blank)}"
        else:
            note = " · ".join(f'{f["name"]} {f["prob"]*100:.0f}%' for f in faces)
        rows.append((part, exp, got, "PASS" if ok else "FAIL", note))
        if not ok:
            n_fail += 1

    w = max(len(p) for p, *_ in rows)
    print(f"\n{'PART'.ljust(w)}  EXP  GOT  RESULT  DETAIL")
    print("-" * (w + 40))
    for part, exp, got, res, note in rows:
        mark = {"PASS": "✓", "FAIL": "✗", "MISS": "✗", "SKIP": "·"}.get(res, " ")
        print(f"{part.ljust(w)}  {str(exp).center(3)}  {str(got).center(3)}  "
              f"{mark} {res.ljust(4)}  {note}")
    total = len([r for r in rows if r[3] != "SKIP"])
    print("-" * (w + 40))
    print(f"{total - n_fail}/{total} PASS"
          + (f"  ·  {n_fail} FAILED" if n_fail else "  ·  all green"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
