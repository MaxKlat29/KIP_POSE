#!/usr/bin/env python3
"""filter_visibility.py — >20% visibility PIPELINE FILTER  (Accuracy-Push P1 / T-038).

THE single, uniform visibility gate for the whole POSE pipeline. Max's rule:

  The data stays REALISTIC. The arm occludes, parts fall where they fall (some
  partly under the arm). BUT any instance whose visib_fract <= THRESHOLD (0.20)
  is CUT FROM THE PIPELINE — it is removed from scene_gt, scene_gt_info, the
  per-instance masks, AND from the detector training labels. A part that is
  <=20% visible is fairly not pose-able, so it must not count against (or for)
  the metrics, the training target, or the eval denominator.

  This is NOT keeping parts away from the arm. Occlusion stays in the image.
  Only the LABEL / TARGET / EVAL ignores <=20% instances.

ONE threshold (0.20) everywhere — replacing the old split of detector~0.15
(max_occ 0.85), GDRNPP 0.30 (FILTER_VISIB_THR) and eval 0.0 (no filter at all).

------------------------------------------------------------------------------
WHAT IT TOUCHES

  1. BOP GT path  (--bop-root, splits train_pbr + val):
       scene_gt.json        : drop instances with visib_fract <= THR, renumber gt_ids 0..k-1
       scene_gt_info.json   : same drop + renumber (kept aligned with scene_gt)
       mask/, mask_visib/   : delete the orphaned <im>_<gt>.png of dropped instances,
                              rename the survivors to the new contiguous gt_ids
     IDEMPOTENT: the first run snapshots scene_gt.json -> scene_gt.json.full and
     scene_gt_info.json -> scene_gt_info.json.full (full = pre-filter truth).
     Every run reads from the .full snapshot, so re-running with a different THR
     re-derives from the unfiltered source (never compounds).

  2. Detector label path  (--sdg-dir, the arm-visible SDG bundle obb_2d_*.json):
     reports how many OBB labels have visib (= 1 - occlusion) <= THR and would be
     dropped by the detector retrain. The retrain (train_detector_armvis.py) reads
     occlusion live, so the operative lever there is --max-occ = 1 - THR (0.80);
     this script only AUDITS the drop so the report has the train-label number.

------------------------------------------------------------------------------
USAGE  (on the box, bop-venv — pure json/os, no heavy deps)

  # GT filter on both splits + detector-label audit:
  /mnt/data/bop/bop-venv/bin/python box_src/filter_visibility.py \
      --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
      --splits train_pbr,val \
      --sdg-dir /mnt/data/kip_pose/data/sdg_armvis_full \
      --thr 0.20

  # restore the unfiltered GT (from the .full snapshots) and exit:
  ... --restore

  # dry-run (report only, write nothing):
  ... --dry-run
"""
import argparse
import glob
import json
import os
import shutil
import sys

# THE uniform pipeline visibility threshold. visib_fract <= THR -> cut.
VISIB_THR = 0.20


def _scene_dirs(bop_root, split):
    sdir = os.path.join(bop_root, split)
    if not os.path.isdir(sdir):
        return []
    out = []
    for name in sorted(os.listdir(sdir)):
        p = os.path.join(sdir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "scene_gt.json")):
            out.append(p)
    return out


def _snapshot(path):
    """Return the pristine (pre-filter) path: <path>.full, creating it on first run.
    All filtering derives from this snapshot so re-runs never compound."""
    full = path + ".full"
    if not os.path.exists(full):
        shutil.copy2(path, full)
    return full


def _restore_scene(scene_dir):
    n = 0
    for base in ("scene_gt.json", "scene_gt_info.json"):
        full = os.path.join(scene_dir, base + ".full")
        if os.path.exists(full):
            shutil.copy2(full, os.path.join(scene_dir, base))
            n += 1
    # masks: regenerate is expensive; we kept the originals only via rename, so a
    # true restore of masks needs a re-convert. We warn rather than silently lie.
    return n


def filter_scene(scene_dir, thr, dry_run=False, prune_masks=True):
    """Filter one BOP scene. Returns (n_total, n_dropped, n_kept)."""
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    info_path = os.path.join(scene_dir, "scene_gt_info.json")
    gt_full = _snapshot(gt_path) if not dry_run else (
        gt_path + ".full" if os.path.exists(gt_path + ".full") else gt_path)
    info_full = _snapshot(info_path) if not dry_run else (
        info_path + ".full" if os.path.exists(info_path + ".full") else info_path)

    gt = json.load(open(gt_full))
    info = json.load(open(info_full))

    new_gt, new_info = {}, {}
    n_total = n_dropped = n_kept = 0
    # per-image (kept_idx_in_full -> new_gt_id) for mask rename
    rename_plan = {}   # im_key -> list of (old_gt_id, new_gt_id)
    drop_plan = {}     # im_key -> list of old_gt_id dropped

    for im_key in gt.keys():
        insts_gt = gt[im_key]
        insts_info = info.get(im_key, [])
        keep_gt, keep_info = [], []
        renames, drops = [], []
        new_id = 0
        for old_id, (g, finfo) in enumerate(zip(insts_gt, insts_info)):
            n_total += 1
            vf = float(finfo.get("visib_fract", 1.0))
            if vf <= thr:
                n_dropped += 1
                drops.append(old_id)
                continue
            keep_gt.append(g)
            keep_info.append(finfo)
            renames.append((old_id, new_id))
            new_id += 1
            n_kept += 1
        new_gt[im_key] = keep_gt
        new_info[im_key] = keep_info
        rename_plan[im_key] = renames
        drop_plan[im_key] = drops

    if not dry_run:
        json.dump(new_gt, open(gt_path, "w"), indent=2)
        json.dump(new_info, open(info_path, "w"), indent=2)
        if prune_masks:
            _apply_mask_plan(scene_dir, rename_plan, drop_plan)

    return n_total, n_dropped, n_kept


def _apply_mask_plan(scene_dir, rename_plan, drop_plan):
    """Delete masks of dropped instances and renumber survivors to contiguous ids.

    Mask files are <im_id:06d>_<gt_id:06d>.png in mask/ and mask_visib/. We use a
    two-phase rename (-> temp -> final) so a survivor moving into a slot freed by a
    dropped/earlier instance never clobbers an unprocessed file.
    """
    for sub in ("mask", "mask_visib"):
        mdir = os.path.join(scene_dir, sub)
        if not os.path.isdir(mdir):
            continue
        for im_key, drops in drop_plan.items():
            im_id = int(im_key)
            for gid in drops:
                f = os.path.join(mdir, f"{im_id:06d}_{gid:06d}.png")
                if os.path.exists(f):
                    os.remove(f)
        # phase 1: survivors that change id -> .tmp
        for im_key, renames in rename_plan.items():
            im_id = int(im_key)
            for old_id, new_id in renames:
                if old_id == new_id:
                    continue
                src = os.path.join(mdir, f"{im_id:06d}_{old_id:06d}.png")
                if os.path.exists(src):
                    os.rename(src, src + ".tmp")
        # phase 2: .tmp -> final new id
        for im_key, renames in rename_plan.items():
            im_id = int(im_key)
            for old_id, new_id in renames:
                if old_id == new_id:
                    continue
                tmp = os.path.join(mdir, f"{im_id:06d}_{old_id:06d}.png.tmp")
                if os.path.exists(tmp):
                    os.rename(tmp, os.path.join(mdir, f"{im_id:06d}_{new_id:06d}.png"))


def audit_detector_labels(sdg_dir, thr):
    """Audit obb_2d_*.json: how many detector training boxes have
    visib (= 1 - occlusion) <= thr and would be cut. occlusion == -1 -> 0
    (fully visible). Returns (n_total, n_dropped)."""
    n_total = n_dropped = 0
    files = sorted(glob.glob(os.path.join(sdg_dir, "obb_2d_*.json")))
    for f in files:
        for b in json.load(open(f)).get("boxes", []):
            occ = float(b.get("occlusion", 0.0))
            if occ < 0:
                occ = 0.0
            visib = 1.0 - occ
            n_total += 1
            if visib <= thr:
                n_dropped += 1
    return n_total, n_dropped, len(files)


def main():
    ap = argparse.ArgumentParser(description=">20% visibility pipeline filter (T-038)")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--splits", default="train_pbr,val")
    ap.add_argument("--thr", type=float, default=VISIB_THR,
                    help="drop instances with visib_fract <= THR (default 0.20)")
    ap.add_argument("--sdg-dir", default="",
                    help="arm-visible SDG bundle dir for detector-label audit")
    ap.add_argument("--no-masks", action="store_true", help="skip mask prune/rename")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--restore", action="store_true",
                    help="restore scene_gt(+info) from .full snapshots and exit")
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s.strip()]

    if args.restore:
        for split in splits:
            for sd in _scene_dirs(args.bop_root, split):
                n = _restore_scene(sd)
                print(f"[restore] {sd}: restored {n} json (masks NOT restored — "
                      f"re-convert if masks were pruned)")
        print("[restore] done. NOTE masks were pruned/renamed in-place; if you need "
              "the original masks, re-run isaac_to_bop / convert_full_to_bop.")
        return

    print(f"[filter_visibility] THR={args.thr}  drop instances with visib_fract <= {args.thr}")
    grand = {}
    for split in splits:
        sds = _scene_dirs(args.bop_root, split)
        tot = drp = kpt = 0
        for sd in sds:
            t, d, k = filter_scene(sd, args.thr, dry_run=args.dry_run,
                                   prune_masks=not args.no_masks)
            tot += t; drp += d; kpt += k
        pct = 100.0 * drp / tot if tot else 0.0
        grand[split] = (tot, drp, kpt, pct)
        tag = "(DRY-RUN, nothing written)" if args.dry_run else "(written)"
        print(f"[GT {split}] {len(sds)} scenes  total={tot}  "
              f"dropped(<= {args.thr})={drp} ({pct:.1f}%)  kept={kpt}  {tag}")

    if args.sdg_dir and os.path.isdir(args.sdg_dir):
        nt, nd, nf = audit_detector_labels(args.sdg_dir, args.thr)
        pct = 100.0 * nd / nt if nt else 0.0
        print(f"[DET labels] {nf} obb_2d files  total_boxes={nt}  "
              f"would-drop(visib<= {args.thr} i.e. occ>= {1-args.thr:.2f})={nd} ({pct:.1f}%)  "
              f"-> set train_detector_armvis --max-occ {1-args.thr:.2f}")
    elif args.sdg_dir:
        print(f"[DET labels] WARN sdg-dir not found: {args.sdg_dir}", file=sys.stderr)

    print("\n[SUMMARY]")
    for split, (tot, drp, kpt, pct) in grand.items():
        print(f"  {split:10s}  dropped {drp}/{tot} = {pct:.1f}%  (kept {kpt})")


if __name__ == "__main__":
    main()
