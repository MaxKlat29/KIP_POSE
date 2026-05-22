#!/usr/bin/env python3
"""S-009 — back-project a 2D bbox centre onto the world table plane.

The world frame (pose_result contract): Z-up, origin = table-plane null-point,
units = metres. The dataset camera (``sim_code/render_dataset.py``) looks
straight down the world ``-Z`` axis from height ``cam_h``:

    camera position  = (0, 0, cam_h)        # AddTranslateOp Set(Gf.Vec3d(0,0,cam_h))
    focal length     = focal mm             # CreateFocalLengthAttr
    image            = res × res px (square top-down render)

For a top-down pinhole, a table point ``(X, Y, 0)`` projects to pixel

    u = cx + fx * ( X / cam_h)
    v = cy - fy * ( Y / cam_h)              # image-v grows downward, world-Y up

with ``fx = fy = focal_mm / sensor_mm * res`` (pixels). Inverting for the table
plane (``Z = 0``, so depth from the camera is exactly ``cam_h``):

    X =  (u - cx) * cam_h / fx
    Y = -(v - cy) * cam_h / fy
    Z =  0  (+ the face rest height, added by the alignment stage)

These are the **documented default intrinsics** so the E2E run works without a
real calibration. Pass a calibrated :class:`Intrinsics` for a true Zivid scene.

Note on the dummy scene: ``data/examples/dummy_scene`` is a Zivid multi-object
render (1280×720, a tilted custom camera), not the canonical top-down
``render_dataset`` view. We still back-project it with the documented top-down
model — the resulting ``(x, y)`` are a plausible table layout for wiring/preview,
not metric ground truth. The contract and the convention are exact; the metric
accuracy needs the real camera pose, which Max plugs in via ``Intrinsics``.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── documented default intrinsics (from sim_code/render_dataset.py) ───────────
DEFAULT_CAM_H = 0.16        # camera height above the table plane, metres
DEFAULT_FOCAL_MM = 24.0     # focal length, mm
DEFAULT_SENSOR_MM = 20.955  # Isaac/USD default horizontal aperture, mm
DEFAULT_RES = 900           # square render resolution, px


@dataclass
class Intrinsics:
    """Top-down pinhole intrinsics + table geometry for back-projection.

    ``width``/``height`` default to a square ``res×res`` render; the principal
    point defaults to the image centre. ``fx``/``fy`` are derived from the focal
    length, sensor size and resolution unless given explicitly.
    """

    cam_h: float = DEFAULT_CAM_H
    focal_mm: float = DEFAULT_FOCAL_MM
    sensor_mm: float = DEFAULT_SENSOR_MM
    width: int = DEFAULT_RES
    height: int = DEFAULT_RES
    cx: float | None = None
    cy: float | None = None
    fx: float | None = None
    fy: float | None = None
    table_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0
        # px-per-mm-of-sensor * res = focal in px, applied to the longer/shorter
        # axis via the (square-pixel) assumption fx == fy.
        if self.fx is None:
            self.fx = self.focal_mm / self.sensor_mm * self.width
        if self.fy is None:
            self.fy = self.focal_mm / self.sensor_mm * self.height

    @classmethod
    def for_image(cls, width: int, height: int, **kw) -> "Intrinsics":
        """Build intrinsics whose principal point is the centre of a W×H image."""
        return cls(width=width, height=height, **kw)


def pixel_to_table(u: float, v: float, intr: Intrinsics) -> tuple[float, float]:
    """Back-project pixel ``(u, v)`` onto the world table plane ``Z = 0``.

    Returns ``(X, Y)`` in metres relative to the table-plane null-point.
    """
    x = (u - intr.cx) * intr.cam_h / intr.fx + intr.table_origin[0]
    y = -(v - intr.cy) * intr.cam_h / intr.fy + intr.table_origin[1]
    return float(x), float(y)


def bbox_center_to_world(bbox_2d, intr: Intrinsics, rest_height: float = 0.0):
    """Back-project a bbox centre to world ``(X, Y, Z)``.

    ``Z = table_origin.z + rest_height`` — the part body origin sits at the face's
    rest height above the table (supplied by the alignment stage from the registry
    face tilt / geometry; 0 keeps it on the plane).
    """
    x0, y0, x1, y1 = bbox_2d
    u = (x0 + x1) / 2.0
    v = (y0 + y1) / 2.0
    X, Y = pixel_to_table(u, v, intr)
    Z = intr.table_origin[2] + rest_height
    return [float(X), float(Y), float(Z)]
