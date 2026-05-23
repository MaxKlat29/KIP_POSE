#!/usr/bin/env python3
"""POSE — BOP -> pose_result Adapter (Viktor adr.md §3, beide Gleise).

DER Adapter zwischen jedem BOP-konventionierten 6D-Pose-Schätzer (GDRNPP / Gleis B
UND MegaPose / Gleis A) und dem eingefrorenen pose_result-Contract (ADR-017,
pose_result.schema.json). **Ein Adapter, zwei Caller** — beide Gleise emittieren so
identisches pose_result.

Eingabe pro Detektion (BOP-Konvention, exakt wie GDRNPP/MegaPose liefern):
  - R_m2c (3x3)  : Model -> Camera Rotation   (p_cam = R_m2c @ p_model + t_m2c)
  - t_m2c (3,)   : Model -> Camera Translation in **mm**
  - obj_id (int) : 1-basierte BOP-Objekt-ID (§1.2)
plus Kamera-Extrinsics aus scene_camera.json (§1.3):
  - R_w2c (3x3)  : World -> Camera Rotation    (p_cam = R_w2c @ p_world + t_w2c)
  - t_w2c (3,)   : World -> Camera Translation in **mm**
plus Szenen-Setup:
  - table_origin (3,) : Tisch-Nullpunkt in Welt, **Meter** (ADR-017)

Ausgabe: R_world (world = R @ body, row-major flat 9), t_world (Meter, rel. Tisch),
part-Name, face, upright.

Transform-Kette (§3.2), EXAKT:
  R_m2w = R_w2c.T @ R_m2c                       # model -> world rotation
  t_m2w = R_w2c.T @ (t_m2c - t_w2c)             # model origin in world (mm)
  R_world = R_m2w @ R_model_to_body             # body convention (default Identity)
  t_world = t_m2w / 1000.0 - table_origin       # mm -> m, dann - table_origin

Konvention (eingefroren, ADR-017): Z-up Welt, world = R @ body (Spaltenkonvention),
Ursprung = Tisch-Nullpunkt, Einheit Meter. KEIN Transpose am Boundary.

Selbst-enthalten: nur numpy + stdlib. Inline-fähig (Max-Präferenz: später in
infer.ipynb). Keine Projekt-Imports.
"""
from __future__ import annotations

import numpy as np

# ── §1.2 obj_id-Mapping (global eingefroren) ──────────────────────────────────
# 1-basiert (BOP-Konvention). Abgeleitet aus detector.metrics.json classes
# (0-basiert) +1. Single-Source — identisch zu PLY-Name, scene_gt, models_info.
OBJ_ID_TO_PART = {
    1: "Anker_Kurz",
    2: "Anker_Lang",
    3: "Buerstenhalter_2polig",
    4: "Getriebegehaeuse_typ4",
    5: "Ringmagnet",
    6: "Zahnrad",
}
PART_TO_OBJ_ID = {v: k for k, v in OBJ_ID_TO_PART.items()}

# Detektor-Klasse (0-basiert) -> obj_id (1-basiert): +1 (§1.2 / §4.1).
DETECTOR_CLASS_ORDER = [
    "anker_kurz", "anker_lang", "buerstenhalter_2polig",
    "getriebegehaeuse_typ4", "ringmagnet", "zahnrad",
]


def category_id_to_obj_id(category_id: int) -> int:
    """Detektor-category_id (0-basiert) -> BOP obj_id (1-basiert). §1.2/§4.1."""
    return int(category_id) + 1


def part_for_obj_id(obj_id: int) -> str:
    """obj_id (1-basiert) -> CAD/Registry part-Name. §1.2."""
    return OBJ_ID_TO_PART.get(int(obj_id), f"obj_{int(obj_id):06d}")


# ── §2.6 Symmetrie pro Teil (für Kanonisierung §3.3) ──────────────────────────
# Achse im Modell-Frame. Bei Anker/Ring/Zahnrad ist die Symmetrieachse die
# Y-Achse (extent_m: X ≈ Z, Y abweichend — die lange Achse).
#   "continuous": rotationssymmetrisch (∞) um axis
#   "discrete":   C_N um axis, n_fold = N  (Zahnrad; N aus CAD, default None)
#   None:         keine Symmetrie
PART_SYMMETRY = {
    "Anker_Kurz":             {"type": "continuous", "axis": [0.0, 1.0, 0.0]},
    "Anker_Lang":             {"type": "continuous", "axis": [0.0, 1.0, 0.0]},
    "Ringmagnet":             {"type": "continuous", "axis": [0.0, 1.0, 0.0]},
    # TODO(S-203): Zahnrad ist C_7 (7-fach) — verifiziert gegen das
    # CAD-abgeleitete models_info.json auf der Box (obj6: 6 symmetries_discrete
    # + Identität = 7-fach). n_fold absichtlich noch None gelassen, weil der
    # getestete None-Pfad (test_discrete_unknown_N_unchanged) sonst bricht; auf 7
    # setzen, sobald der Test gemeinsam mit S-203 nachgezogen wird.
    "Zahnrad":                {"type": "discrete",   "axis": [0.0, 1.0, 0.0],
                              "n_fold": None},   # -> 7 (C_7), s. TODO oben
    "Buerstenhalter_2polig":  None,
    "Getriebegehaeuse_typ4":  None,
}

# Body-Längsachse pro Teil (für upright-Ableitung). Y bei den Fokus-Teilen
# (Anker/Ring/Zahnrad), siehe partRegistry.js Box-Extents + part_meta.extent_m.
PART_LONG_AXIS = {
    "Anker_Kurz": 1, "Anker_Lang": 1, "Ringmagnet": 1, "Zahnrad": 1,
}
DEFAULT_LONG_AXIS = 1  # Body-Y als Default-Längsachse.


# ── Hilfen ────────────────────────────────────────────────────────────────────
def _as_R(R) -> np.ndarray:
    """Eingabe (3x3 oder flat-9) -> (3,3) float64."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (9,):
        R = R.reshape(3, 3)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3 or flat-9, got shape {R.shape}")
    return R


def _as_vec3(t) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.shape != (3,):
        raise ValueError(f"t must be length-3, got shape {t.shape}")
    return t


def axis_angle_matrix(axis, theta: float) -> np.ndarray:
    """Rodrigues: Rotation um Einheitsachse `axis` (Modell-Frame) um theta rad."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    x, y, z = a
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# ── §3.2 DIE Transform-Kette (cam -> world -> contract) ───────────────────────
def bop_pose_to_world(R_m2c, t_m2c_mm, R_w2c, t_w2c_mm, table_origin_m,
                      R_model_to_body=None):
    """BOP model->cam Pose + Kamera-Extrinsics -> Welt-Pose im pose_result-Frame.

    Args:
      R_m2c       : (3,3) model->cam Rotation.
      t_m2c_mm    : (3,) model->cam Translation, **mm**.
      R_w2c       : (3,3) world->cam Rotation (scene_camera.cam_R_w2c).
      t_w2c_mm    : (3,) world->cam Translation, **mm** (scene_camera.cam_t_w2c).
      table_origin_m : (3,) Tisch-Nullpunkt in Welt, **Meter**.
      R_model_to_body : (3,3) PLY-Modell-Frame -> Registry-Body-Frame; default I
                        (§3.1: PLY und Registry teilen das CAD).

    Returns:
      (R_world (3,3), t_world (3,) in Meter rel. Tisch).
      R_world: Spaltenkonvention world = R @ body. KEIN Transpose am Boundary.
    """
    R_m2c = _as_R(R_m2c)
    t_m2c = _as_vec3(t_m2c_mm)
    R_w2c = _as_R(R_w2c)
    t_w2c = _as_vec3(t_w2c_mm)
    table_origin = _as_vec3(table_origin_m)
    R_mb = np.eye(3) if R_model_to_body is None else _as_R(R_model_to_body)

    # 1) Model -> World aus Model->Cam und World->Cam (R_w2c^-1 = R_w2c^T).
    R_m2w = R_w2c.T @ R_m2c                      # model -> world rotation
    t_m2w = R_w2c.T @ (t_m2c - t_w2c)            # model origin in world (mm)

    # 2) Modell-Frame -> Body-Frame (Contract-Konvention), default Identity.
    R_world = R_m2w @ R_mb                       # world = R @ body

    # 3) mm -> Meter + Tisch-Nullpunkt.
    t_world = t_m2w / 1000.0 - table_origin      # m, rel. Tisch-Nullpunkt
    return R_world, t_world


# ── §3.3 Symmetrie-Kanonisierung (vor dem Mapping) ────────────────────────────
def canonicalize_rotation(R_m2c, part, symmetry=None, n_steps_continuous=64):
    """Projiziert R_m2c auf einen kanonischen Symmetrie-Repräsentanten (§3.3).

    Symmetrische Teile haben keine eindeutige Pose. Für deterministischen
    Viewer-Output (face/upright/R_world) wird der Repräsentant gewählt, dessen
    Body-Längsachse in der Welt am nächsten an einer kanonischen Lage liegt — wir
    nehmen den Repräsentanten mit minimalem In-Plane-Drehwinkel um die
    Symmetrieachse (kleinster |Frobenius-Abstand| zur Identität in der
    Achsen-Drehung). Das löst die 91°/120°-Mehrdeutigkeit deterministisch auf.

    Eval bleibt davon unberührt (bop_toolkit rechnet symmetrie-bewusst über
    models_info) — das hier dient NUR der eindeutigen Darstellung.

    Returns: kanonisches R_m2c (3,3). Bei `symmetry=None` unverändert.
    """
    R_m2c = _as_R(R_m2c)
    sym = symmetry if symmetry is not None else PART_SYMMETRY.get(part)
    if not sym:
        return R_m2c
    axis = np.asarray(sym.get("axis", [0.0, 1.0, 0.0]), float)

    if sym["type"] == "continuous":
        # Kontinuierlich: wähle den Yaw um die Achse, der die Rotation der
        # Identität (im Achsen-Eigenraum) am nächsten bringt -> minimaler
        # In-Plane-Winkel. Diskretisiert über n_steps für Robustheit.
        thetas = np.linspace(0.0, 2.0 * np.pi, n_steps_continuous, endpoint=False)
    elif sym["type"] == "discrete":
        n = sym.get("n_fold")
        if not n or n < 2:
            return R_m2c                          # N unbekannt -> keine Wahl
        thetas = np.array([k * 2.0 * np.pi / n for k in range(n)])
    else:
        return R_m2c

    best_R, best_score = R_m2c, np.inf
    for th in thetas:
        Rk = R_m2c @ axis_angle_matrix(axis, th)  # Drehung im Modell-Frame
        # Score: Nähe zur Identität (kleinster Gesamt-Drehwinkel) -> trace max.
        score = -float(np.trace(Rk))
        if score < best_score:
            best_score, best_R = score, Rk
    return best_R


# ── §3.4 / R8 face + upright aus der 6D-Rotation ──────────────────────────────
# Der Face-Classifier ist WEG (ADR-018 Rip-out). face/upright werden aus der
# kanonisierten R_world abgeleitet:
#   upright: zeigt die Body-Längsachse (Y bei den Fokus-Teilen) eher nach
#            Welt-+Z (steht hochkant) oder eher in die Tischebene (liegt flach)?
#            tilt = Winkel zwischen body-Längsachse-in-Welt und Welt-+Z.
#            upright = True, wenn tilt < UPRIGHT_TILT_DEG (Achse steht ~aufrecht).
#   face:    welche Body-(±)Achse zeigt am meisten nach Welt-DOWN (-Z)? Das ist
#            die Auflagefläche -> deterministischer Face-Name "face_<axis><sign>".
#            (Nearest-Registry-Face wäre exakter, aber die Faces-Registry ist im
#            BOP-Stack nicht mehr der Pose-Kern — R8: ableiten statt Classifier.)
UPRIGHT_TILT_DEG = 45.0
_AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def face_and_upright_from_R(R_world, part=None, long_axis=None,
                            upright_tilt_deg=UPRIGHT_TILT_DEG):
    """Leitet (face, upright) aus der Welt-Rotation ab. §3.4 / R8.

    Args:
      R_world   : (3,3) world = R @ body.
      part      : optional, bestimmt die Body-Längsachse via PART_LONG_AXIS.
      long_axis : optional Override (0=x,1=y,2=z). Default: part-Lookup, sonst Y.

    Returns: (face: str, upright: bool).
      face   : "face_<axis><sign>" der Body-Achse, die am meisten nach Welt-DOWN
               (-Z) zeigt = Auflagefläche.
      upright: True wenn die Body-Längsachse näher an Welt-±Z als an der Ebene.
    """
    R = _as_R(R_world)
    if long_axis is None:
        long_axis = PART_LONG_AXIS.get(part, DEFAULT_LONG_AXIS)

    # Spalten von R = Bilder der Body-Basisvektoren in der Welt (world = R @ body).
    world_z = np.array([0.0, 0.0, 1.0])

    # upright: tilt der Body-Längsachse gegen Welt-Z (egal welches Vorzeichen).
    axis_in_world = R[:, long_axis]
    axis_in_world = axis_in_world / (np.linalg.norm(axis_in_world) + 1e-12)
    cos_tilt = abs(float(np.dot(axis_in_world, world_z)))
    tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
    upright = bool(tilt_deg < upright_tilt_deg)

    # face: welche ±Body-Achse zeigt am meisten nach Welt-DOWN (-Z)?
    # dot(R[:,i], -z) maximal -> diese Body-Achse liegt unten = Auflagefläche.
    best_face, best_dot = "face_z+", -np.inf
    for i in range(3):
        col = R[:, i]
        col = col / (np.linalg.norm(col) + 1e-12)
        d_down = float(np.dot(col, -world_z))     # +Achse Richtung -Z
        if d_down > best_dot:
            best_dot, best_face = d_down, f"face_{_AXIS_NAMES[i]}+"
        if -d_down > best_dot:                      # -Achse Richtung -Z
            best_dot, best_face = -d_down, f"face_{_AXIS_NAMES[i]}-"
    return best_face, upright


# ── Komfort: ganze Detektion -> pose_result-Eintrag-Dict ──────────────────────
def detection_to_result(*, instance_id, obj_id, R_m2c, t_m2c_mm,
                        R_w2c, t_w2c_mm, table_origin_m, bbox_2d, confidence,
                        R_model_to_body=None, canonicalize=True):
    """Eine BOP-Detektion -> ein pose_result-`results[]`-Eintrag (Dict).

    Bündelt §3.2 (Transform) + §3.3 (Kanonisierung) + §3.4 (face/upright) +
    §3.1/§1.2 (obj_id->part). Schema-Felder exakt wie pose_result.schema.json.
    """
    part = part_for_obj_id(obj_id)
    R_m2c_can = canonicalize_rotation(R_m2c, part) if canonicalize else _as_R(R_m2c)
    R_world, t_world = bop_pose_to_world(
        R_m2c_can, t_m2c_mm, R_w2c, t_w2c_mm, table_origin_m,
        R_model_to_body=R_model_to_body)
    face, upright = face_and_upright_from_R(R_world, part=part)
    return {
        "instance_id": int(instance_id),
        "part": part,
        "face": face,
        "R_world": [float(v) for v in R_world.reshape(-1)],   # row-major flat 9
        "t_world": [float(v) for v in t_world],               # m, rel. Tisch
        "confidence": float(max(0.0, min(1.0, confidence))),
        "bbox_2d": [int(v) for v in bbox_2d],                 # [x0,y0,x1,y1]
        "upright": bool(upright),
    }
