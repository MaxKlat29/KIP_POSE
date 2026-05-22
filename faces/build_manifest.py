#!/usr/bin/env python3
"""Build per-part train/val manifests from the extracted face snippets.

Scans ``data/snippets/<part>/<face_id>/*.png`` (the output of
``extract_snippets.py``) and writes ``data/snippets/<part>/manifest.json``:

    {
      "part": "Zahnrad",
      "seed": 1234,
      "val_frac": 0.2,
      "size": 96,
      "classes": ["face_1", "face_2"],
      "augment": { ...hooks, all disabled by default... },
      "counts": {"face_1": {"train": 118, "val": 30}, ...},
      "items": [ {"path": "face_1/0007.png", "face_id": "face_1", "split": "train"}, ... ]
    }

The split is **stratified per face** (each class is split independently) and
**deterministic** (fixed seed -> identical assignment every run), so training is
reproducible and a rare face is never starved of val examples. Paths are stored
relative to the part directory so the manifest is location-independent.

Augmentation is declared here as **hooks**, all OFF by default. The training code
reads ``manifest["augment"]`` and applies only what is enabled; nothing is baked
into the snippets, so flipping a flag re-trains with augmentation without
re-extracting. Documented transforms:
  * rotate     — random in-plane rotation (deg range). Safe: yaw is a free DOF
                 of every resting face, so rotated snippets are valid samples.
  * flip       — horizontal mirror. Safe ONLY for parts whose faces are mirror-
                 symmetric from above (e.g. Anker); off by default to avoid
                 corrupting a chiral face. Per-part override possible.
  * brightness — multiplicative jitter, models lighting variation on the table.

    python faces/build_manifest.py                       # all parts found
    python faces/build_manifest.py --only Zahnrad --val-frac 0.2 --seed 1234
    python faces/build_manifest.py --enable-augment rotate brightness
"""
from __future__ import annotations

import argparse
import glob
import json
import os


# Augmentation hook defaults — all disabled. Training reads these; changing a
# flag re-trains with augmentation, no re-extraction needed. Documented, tunable.
DEFAULT_AUGMENT = {
    "rotate": {"enabled": False, "max_deg": 180},      # yaw is free -> always valid
    "flip": {"enabled": False, "horizontal": True},    # only for mirror-sym faces
    "brightness": {"enabled": False, "delta": 0.15},   # +-15% multiplicative
}


def split_part(part_dir, seed, val_frac):
    """Stratified, deterministic train/val split of one part's snippets.

    Returns (items, counts, classes). ``items`` is a list of dicts with a
    part-relative path, face_id label, and split.
    """
    import random

    classes = sorted(
        d for d in os.listdir(part_dir)
        if os.path.isdir(os.path.join(part_dir, d)) and d.startswith("face_"))
    items, counts = [], {}
    for cls in classes:
        files = sorted(os.path.basename(p)
                       for p in glob.glob(os.path.join(part_dir, cls, "*.png")))
        # deterministic shuffle seeded per (seed, class) so adding a new class
        # never reshuffles existing ones
        rng = random.Random(f"{seed}:{cls}")
        rng.shuffle(files)
        n_val = max(1, round(len(files) * val_frac)) if len(files) > 1 else 0
        val_set = set(files[:n_val])
        for f in files:
            split = "val" if f in val_set else "train"
            items.append({"path": f"{cls}/{f}", "face_id": cls, "split": split})
        counts[cls] = {"train": len(files) - len(val_set), "val": len(val_set)}
    return items, counts, classes


def build(part_dir, seed, val_frac, augment):
    name = os.path.basename(part_dir.rstrip("/"))
    items, counts, classes = split_part(part_dir, seed, val_frac)
    size = 96
    meta_path = os.path.join(part_dir, "snippets_meta.json")
    if os.path.exists(meta_path):
        size = json.load(open(meta_path)).get("snippet_size", 96)
    manifest = {
        "part": name, "seed": seed, "val_frac": val_frac, "size": size,
        "classes": classes, "augment": augment, "counts": counts,
        "items": items,
    }
    out = os.path.join(part_dir, "manifest.json")
    json.dump(manifest, open(out, "w"), indent=2)
    tr = sum(c["train"] for c in counts.values())
    va = sum(c["val"] for c in counts.values())
    print(f"[manifest] {name}: {len(classes)} classes, {tr} train / {va} val "
          f"(seed={seed}, val_frac={val_frac}) -> {out}")
    for cls in classes:
        print(f"             {cls}: {counts[cls]['train']} train / {counts[cls]['val']} val")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-root", default="data/snippets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--enable-augment", nargs="*", default=[],
                    choices=list(DEFAULT_AUGMENT),
                    help="turn on augmentation hooks (default: all off)")
    args = ap.parse_args()

    augment = json.loads(json.dumps(DEFAULT_AUGMENT))   # deep copy
    for k in args.enable_augment:
        augment[k]["enabled"] = True

    if args.only:
        parts = args.only
    else:
        parts = sorted(d for d in os.listdir(args.snippets_root)
                       if os.path.isdir(os.path.join(args.snippets_root, d)))
    done = 0
    for part in parts:
        part_dir = os.path.join(args.snippets_root, part)
        if not any(d.startswith("face_") for d in os.listdir(part_dir)):
            print(f"[manifest] SKIP {part}: no face_* folders")
            continue
        build(part_dir, args.seed, args.val_frac, augment)
        done += 1
    print(f"[manifest] done — {done} parts")


if __name__ == "__main__":
    main()
