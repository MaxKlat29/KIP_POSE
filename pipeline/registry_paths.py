#!/usr/bin/env python3
"""Read the committed face registry (``registry/<part>/``) for the pipeline.

The face registry is the persistent artefact the pipeline aligns against. It is
the per-part output of ``faces/cluster_views.py`` — ``faces_<part>.json`` plus
``tmpl_Face<k>.png`` — copied (rsync) from the GPU box into a committed folder:

    registry/<part>/faces_<part>.json
    registry/<part>/tmpl_Face1.png
    registry/<part>/tmpl_Face2.png
    ...

``faces_<part>.json`` schema (written by cluster_views):

    {"part": str, "n_views": int, "convention": "world = R @ body (column)",
     "faces": [{"name": "Face 1", "prob": float, "count": int,
                "tilt_deg": float, "tmpl_ok": bool, "R": [9 floats]}]}

``R`` is the canonical rest rotation ``R_face`` in **column convention
world = R @ body** — the same convention the pose_result contract freezes, so it
is composed onto, never transposed.

This module is the pipeline's single source of truth for face names, rest
rotations and yaw templates. It is intentionally separate from
``faces/classifier/registry.py`` (Kai's model-routing + checkpoint convention),
which we leave untouched.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Repo-root-relative; override for a different layout.
REGISTRY_ROOT = os.environ.get(
    "POSE_REGISTRY_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry"),
)


@dataclass
class Face:
    """One resting face from ``faces_<part>.json``."""

    name: str                       # canonical contract face name, e.g. "Face 1"
    prob: float
    tilt_deg: float
    R_face: np.ndarray              # 3x3, column convention world = R @ body
    template_path: str | None       # tmpl_Face<k>.png if present
    index: int                      # 1-based, matches tmpl_Face<index>.png


@dataclass
class PartRegistry:
    part: str
    convention: str
    faces: list[Face]

    def face_by_name(self, name: str) -> Face | None:
        for f in self.faces:
            if f.name == name:
                return f
        return None

    def resolve_face(self, classifier_face: str) -> Face | None:
        """Map a classifier face id to a registry :class:`Face`.

        Accepts the canonical contract name verbatim ("Face 1"), the folder-style
        id the nearest-template fallback emits ("face_1" / "Face_1"), or a bare
        1-based index. Falls back to the most probable face when nothing matches,
        so the contract always carries a valid registry face name.
        """
        if not self.faces:
            return None
        # exact contract name
        f = self.face_by_name(classifier_face)
        if f is not None:
            return f
        # folder-style "face_1" / "Face_1" / "Face 1" / bare index -> index
        norm = classifier_face.strip().lower().replace("face", "").replace("_", "").replace(" ", "")
        if norm.isdigit():
            idx = int(norm)
            for fc in self.faces:
                if fc.index == idx:
                    return fc
        # last resort: most probable face (faces are largest-first from clustering)
        return max(self.faces, key=lambda fc: fc.prob)


def registry_dir(part: str, root: str | None = None) -> str:
    return os.path.join(root or REGISTRY_ROOT, part)


def has_registry(part: str, root: str | None = None) -> bool:
    return os.path.exists(os.path.join(registry_dir(part, root), f"faces_{part}.json"))


def load_part_registry(part: str, root: str | None = None) -> PartRegistry | None:
    """Load ``registry/<part>/faces_<part>.json`` into a :class:`PartRegistry`.

    Returns ``None`` if no registry exists for ``part`` (the caller then falls
    back to identity rotation + a default face, keeping the contract valid).
    """
    pdir = registry_dir(part, root)
    jp = os.path.join(pdir, f"faces_{part}.json")
    if not os.path.exists(jp):
        return None
    with open(jp) as f:
        data = json.load(f)
    faces: list[Face] = []
    for k, fc in enumerate(data.get("faces", []), start=1):
        R = np.asarray(fc.get("R", []), dtype=float)
        if R.size == 9:
            R = R.reshape(3, 3)
        else:
            R = np.eye(3)
        tp = os.path.join(pdir, f"tmpl_Face{k}.png")
        faces.append(Face(
            name=str(fc.get("name", f"Face {k}")),
            prob=float(fc.get("prob", 0.0)),
            tilt_deg=float(fc.get("tilt_deg", 0.0)),
            R_face=R,
            template_path=tp if os.path.exists(tp) else None,
            index=k,
        ))
    return PartRegistry(
        part=data.get("part", part),
        convention=data.get("convention", "world = R @ body (column)"),
        faces=faces,
    )


def load_template(face: Face, size: int = 96) -> np.ndarray | None:
    """Load a face's ``tmpl_Face<k>.png`` as a greyscale float [0,1] ``size×size``.

    Returns ``None`` when the part has no template (yaw search then degrades to a
    documented identity yaw).
    """
    if not face.template_path or not os.path.exists(face.template_path):
        return None
    img = np.asarray(Image.open(face.template_path).convert("L").resize((size, size)))
    return img.astype(np.float32) / 255.0


def available_parts(root: str | None = None) -> list[str]:
    """List parts that have a committed registry (a ``faces_<part>.json``)."""
    rt = root or REGISTRY_ROOT
    if not os.path.isdir(rt):
        return []
    out = []
    for p in sorted(os.listdir(rt)):
        if has_registry(p, rt):
            out.append(p)
    return out
