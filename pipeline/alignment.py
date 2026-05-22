#!/usr/bin/env python3
"""S-009 — CAD alignment: face + crop -> full 6D rotation.

Given a detection's resting face (from the classifier, S-008) and its image crop,
recover the free in-plane **yaw** and compose it onto the registry rest rotation
to get ``R_world`` in the frozen column convention ``world = R @ body``.

Yaw search
----------
The two out-of-plane rotations and the rest height are fixed by the resting
face; only the in-plane position ``(x, y)`` and yaw are free. We find yaw by
rotating the face's reference template in-plane over a fixed angular grid,
overlaying it on the crop's silhouette, and taking the **MSE minimum**:

    yaw* = argmin_θ  MSE( rotate(template, θ), crop_silhouette )

with a documented step (default 5°). The match is on a normalised greyscale
height/silhouette so it is brightness-robust. When a part has no template (no
registry / missing tmpl) the yaw degrades to 0° (documented identity yaw) and
``R_world`` is just the registry ``R_face`` — still contract-valid.

Rotation composition
---------------------
``R_world = Rz(yaw) · R_face`` — a world-frame yaw about the up (Z) axis applied
to the registry rest rotation. ``R_face`` is the canonical ``world = R @ body``
column rotation from ``faces_<part>.json``; we **never transpose** it, matching
the contract and the registry. ``Rz`` is the same column-convention world-frame
rotation, so the product stays in convention.

Upright flag
------------
``upright = tilt_deg >= UPRIGHT_TILT_DEG`` — a resting face whose body up-axis is
tilted far from the table normal is a head-standing / edge rest ("Kopf steht
hochkant"). ``tilt_deg`` comes straight from the registry face (the angle between
mean body-gravity and world up).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from pipeline.registry_paths import Face, PartRegistry, load_template

# ── tunables (documented constants) ───────────────────────────────────────────
YAW_STEP_DEG = 5.0          # angular step of the in-plane yaw search
TMPL_SIZE = 96              # square size both crop + template are resampled to
UPRIGHT_TILT_DEG = 60.0     # face tilt at/above which the rest pose counts as upright


@dataclass
class Aligned:
    """Full 6D rotation result for one detection (translation added downstream)."""

    R_world: list[float]        # 9 floats, row-major, world = R @ body
    yaw_deg: float
    upright: bool
    face_name: str
    rest_height: float          # metres, body origin height above the table (0 by default)


def _rz(theta_rad: float) -> np.ndarray:
    """World-frame rotation about +Z (up), column convention world = R @ body."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _silhouette(crop: np.ndarray, size: int = TMPL_SIZE) -> np.ndarray:
    """Normalised greyscale silhouette of a crop, square-padded to ``size``.

    Background-masked crops are neutral grey (128); we centre the part on a black
    field and normalise to [0,1] so the MSE compares shape, not absolute level.
    """
    a = np.asarray(crop)
    if a.ndim == 3:
        g = a.astype(np.float32).mean(2)
    else:
        g = a.astype(np.float32)
    # square-pad to keep aspect (the renderer/templates are square)
    h, w = g.shape[:2]
    sd = max(h, w)
    sq = np.full((sd, sd), 128.0, np.float32)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = g
    img = np.asarray(Image.fromarray(sq.astype(np.uint8)).resize((size, size), Image.BILINEAR))
    img = img.astype(np.float32)
    # deviation from the grey background, normalised -> shape signal in [0,1]
    sig = np.abs(img - 128.0)
    m = sig.max()
    return sig / m if m > 1e-6 else sig


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def estimate_yaw(crop: np.ndarray, template: np.ndarray | None,
                 step_deg: float = YAW_STEP_DEG) -> float:
    """In-plane yaw (degrees, [0,360)) of ``crop`` vs ``template`` by MSE-min.

    Rotates the template over a ``step_deg`` grid and returns the angle of least
    MSE against the crop silhouette. Returns ``0.0`` (documented identity yaw)
    when no template is available.
    """
    if template is None:
        return 0.0
    q = _silhouette(crop)
    best_ang, best_mse = 0.0, float("inf")
    for ang in np.arange(0.0, 360.0, step_deg):
        rt = ndimage.rotate(template, ang, reshape=False, order=1, mode="constant", cval=0.0)
        m = _mse(rt, q)
        if m < best_mse:
            best_mse, best_ang = m, float(ang)
    return best_ang


def align_detection(part: str, classifier_face: str, crop: np.ndarray,
                    registry: PartRegistry | None,
                    step_deg: float = YAW_STEP_DEG) -> Aligned:
    """Compute the 6D rotation for one detection.

    ``registry`` is the part's :class:`PartRegistry` (or ``None`` when the part
    has no committed registry — then identity ``R_face`` + the classifier face id
    are used, keeping the contract valid).
    """
    face: Face | None = registry.resolve_face(classifier_face) if registry else None
    if face is None:
        # no registry: identity rest rotation, yaw 0, flat. Contract stays valid.
        R_world = np.eye(3)
        return Aligned(
            R_world=[float(v) for v in R_world.flatten()],
            yaw_deg=0.0,
            upright=False,
            face_name=classifier_face or "Face 1",
            rest_height=0.0,
        )

    template = load_template(face, size=TMPL_SIZE)
    yaw = estimate_yaw(crop, template, step_deg=step_deg)
    R_world = _rz(np.radians(yaw)) @ face.R_face       # world-frame yaw onto R_face
    upright = face.tilt_deg >= UPRIGHT_TILT_DEG
    return Aligned(
        R_world=[float(v) for v in R_world.flatten()],
        yaw_deg=yaw,
        upright=bool(upright),
        face_name=face.name,
        rest_height=0.0,
    )
