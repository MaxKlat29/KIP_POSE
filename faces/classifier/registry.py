#!/usr/bin/env python3
"""Part -> model mapping, checkpoint-path convention, and face-template loading.

Pure stdlib + numpy/PIL — imports without torch, so the inference fallback works
on any machine. This module is the single place that knows:

  * which trained model a part uses          (PART_TO_MODEL)
  * where a model's checkpoint lives          (checkpoint_path)
  * a model's face class set + templates      (load_face_registry)

Model granularity
-----------------
One model per *part family*. A family is a set of parts that share the same
geometry and therefore the same face class set, so one network serves them all.
Anker_Lang and Anker_Kurz are different lengths of the same motor armature but
geometrically distinct enough to warrant their own model each:

    Anker_Lang  -> model "lang"   (long armature)
    Anker_Kurz  -> model "kurz"   (short armature)

Every other rendered part maps to its own model named after the part. Extend
``PART_TO_MODEL`` to point two parts at one shared model.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

# Repo-root-relative roots. Override via env for a different layout.
SNIPPETS_ROOT = os.environ.get("POSE_SNIPPETS_ROOT", "data/snippets")
OUTPUT_ROOT = os.environ.get("POSE_OUTPUT_ROOT", "data/output")
CKPT_ROOT = os.environ.get("POSE_CKPT_ROOT", "faces/classifier/checkpoints")

# ── part -> model mapping ─────────────────────────────────────────────────────
# Anker is the prompted two-model case; the rest default to one model per part.
PART_TO_MODEL = {
    "Anker_Lang": "lang",
    "Anker_Kurz": "kurz",
    "Zahnrad": "Zahnrad",
    "Poltopf_kurz_centered": "Poltopf_kurz_centered",
    "Getriebegehaeuse_typ4": "Getriebegehaeuse_typ4",
    "Buerstenhalter_2polig": "Buerstenhalter_2polig",
}


def model_name(part):
    """Model serving ``part``. Defaults to the part name if not mapped."""
    return PART_TO_MODEL.get(part, part)


def checkpoint_path(part, ckpt_root=None):
    """Canonical checkpoint path for the model serving ``part``.

        <CKPT_ROOT>/<model_name>.pt

    e.g. Anker_Lang -> faces/classifier/checkpoints/lang.pt
         Anker_Kurz -> faces/classifier/checkpoints/kurz.pt
    Drop a trained .pt here and inference picks it up automatically.
    """
    root = ckpt_root or CKPT_ROOT
    return os.path.join(root, f"{model_name(part)}.pt")


def has_checkpoint(part, ckpt_root=None):
    return os.path.exists(checkpoint_path(part, ckpt_root))


def part_dir(part, snippets_root=None):
    return os.path.join(snippets_root or SNIPPETS_ROOT, part)


def classes_for(part, snippets_root=None):
    """Ordered face class list for a part (from its manifest, else its folders).

    This order IS the model's output-index order — train and infer must agree,
    so it is read from one place.
    """
    pdir = part_dir(part, snippets_root)
    man = os.path.join(pdir, "manifest.json")
    if os.path.exists(man):
        return json.load(open(man))["classes"]
    if os.path.isdir(pdir):
        return sorted(d for d in os.listdir(pdir)
                      if os.path.isdir(os.path.join(pdir, d)) and d.startswith("face_"))
    return []


def load_templates(part, size=96, output_root=None):
    """Load the per-face reference templates for the nearest-template fallback.

    Returns ``{face_id: HxWx3 uint8 array}`` resized to ``size``. Templates are
    the ``tmpl_Face<k>.png`` written by cluster_views; face_id ``face_<k>`` maps
    to template ``Face<k>`` (1-based, matching the kept-face order).
    """
    name = part
    fdir = os.path.join(output_root or OUTPUT_ROOT, f"faceset_{part}", "faces")
    out = {}
    for k, cls in enumerate(classes_for(part), start=1):
        tp = os.path.join(fdir, f"tmpl_Face{k}.png")
        if os.path.exists(tp):
            img = np.asarray(Image.open(tp).convert("RGB").resize((size, size)))
            out[cls] = img
    return out
