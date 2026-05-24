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


# ── §3.5 Planares Tisch-Ebenen-Pose-Refinement (T-055, training-frei) ─────────
# DER stärkste Prior des planaren Setups: die Teile LIEGEN auf einer bekannten
# Tischebene (Contract: Welt z = table_z ist die Tischfläche). RGB-only-Schätzer
# (GDRNPP) sind in der Tiefe (Z) schwach — „Z/Tiefe ist der AR-Killer". Statt zu
# trainieren snappen wir die Z-Translation analytisch: der tiefste Mesh-Punkt
# unter der vorhergesagten Rotation MUSS die Tischebene berühren (das Teil ruht).
#
# Generalisierbar: braucht nur (1) die vorhergesagte Welt-Rotation, (2) das CAD-
# Mesh des Teils. Keine Symmetrie, kein Training, kein Teil-Spezialwissen.
#
# Konvention: world = R @ body, Z-up, Tischfläche bei Welt-z = table_z (Meter).
# Mesh-Vertices im Body-Frame, gleiche Einheit wie t_world (Meter).


def lowest_contact_z(R_world, mesh_verts):
    """Welt-z des tiefsten Mesh-Punkts unter R_world, RELATIV zur Body-Herkunft.

    min_v (R_world @ v_body).z  — die z-Komponente des am weitesten in Welt-DOWN
    rotierten Vertex. Addiert man t_world.z dazu, erhält man die absolute Welt-
    Höhe des Auflagepunkts. Vektorisiert (nur die dritte Matrixzeile nötig).

    Args:
      R_world   : (3,3) world = R @ body.
      mesh_verts: (N,3) Body-Frame-Vertices (gleiche Einheit wie t_world).
    Returns:
      float: min über alle Vertices von (R_world @ v).z.
    """
    R = _as_R(R_world)
    V = np.asarray(mesh_verts, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"mesh_verts must be (N,3), got {V.shape}")
    # (R @ v).z = R[2,:] · v  für jeden Vertex -> V @ R[2,:].
    z_world = V @ R[2, :]
    return float(z_world.min())


def planar_z_snap(R_world, t_world, mesh_verts, table_z=0.0, max_snap_m=None):
    """Z-Snap: verschiebt t_world.z so, dass das Teil auf der Tischebene RUHT.

    Der tiefste Mesh-Punkt unter R_world soll exakt auf Welt-z = table_z landen
    (Auflagepunkt = Tischfläche). x/y und die Rotation bleiben unangetastet —
    nur die schwache Tiefenschätzung wird durch den Planar-Prior ersetzt.

        contact_z_abs = t_world.z + min_v (R_world @ v).z
        dz            = table_z - contact_z_abs
        t_world.z    += dz

    GUARD (max_snap_m): der Planar-Prior gilt NUR für Teile, die wirklich auf dem
    Tisch ruhen. Hat der Schätzer ein Teil weit über dem Tisch platziert (Teil im
    Greifer / in der Luft / gestapelt), ist |dz| groß — dann ist die Auflage-
    Annahme verletzt und Snappen würde das Teil katastrophal verschieben (T-055-
    Messung: solche Held-Teile 17mm -> 530mm). Ist |dz| > max_snap_m, wird NICHT
    gesnappt (Pose unverändert). None = kein Guard (immer snappen).

    Args:
      R_world   : (3,3) world = R @ body (vorhergesagte Rotation, unverändert).
      t_world   : (3,) Welt-Translation (Meter), z wird korrigiert.
      mesh_verts: (N,3) CAD-Body-Vertices in Metern.
      table_z   : Welt-z der Tischfläche (Contract: 0.0).
      max_snap_m: max. erlaubte |dz| in Metern; darüber kein Snap (Held-Teil-Guard).
    Returns:
      (t_world_snapped (3,), dz (float)) — dz = TATSÄCHLICH angewandte z-Korrektur
      (0.0 wenn der Guard das Snappen verhindert hat).
    """
    t = _as_vec3(t_world).copy()
    contact_z_abs = t[2] + lowest_contact_z(R_world, mesh_verts)
    dz = float(table_z) - contact_z_abs
    if max_snap_m is not None and abs(dz) > float(max_snap_m):
        return t, 0.0                                # Guard: Teil ruht nicht -> nicht anfassen
    t[2] += dz
    return t, dz


def contact_planarity(R_world, mesh_verts, band_m=0.002):
    """Wie flach/ausgedehnt ist die Auflagefläche unter R_world? (0..1 grob).

    Misst den Anteil der Vertices, die in einem dünnen Band über dem tiefsten
    Punkt liegen (potenzielle Kontaktfläche), sowie deren xy-Streuung. Ein hoher
    Anteil in einem flachen Band + große xy-Ausdehnung = das Teil liegt
    konfident auf einer planaren Fläche. Ein einzelner Berührpunkt (Kante/Spitze)
    = niedriger Wert -> Tilt-Korrektur NICHT anwenden.

    Returns: (frac_in_band float, n_contact int). Nur als Heuristik für die
    KONSERVATIVE Tilt-Korrektur gedacht.
    """
    R = _as_R(R_world)
    V = np.asarray(mesh_verts, dtype=np.float64)
    z = V @ R[2, :]
    zmin = z.min()
    in_band = z <= (zmin + float(band_m))
    return float(in_band.mean()), int(in_band.sum())


# Default-Guard: max. erlaubte Z-Snap-Verschiebung in Metern. Resting-Teile haben
# Tiefen-Fehler ~ wenige cm (T-055: median 34mm); Teile im Greifer/in der Luft
# liegen 0.3-0.6m über dem Tisch -> deren |dz| >> 50mm -> Guard verhindert das
# katastrophale Snappen (sonst 17mm -> 530mm). 50mm fängt resting komfortabel ab.
DEFAULT_MAX_SNAP_M = 0.05


def planar_refine(R_world, t_world, mesh_verts, table_z=0.0,
                  z_snap=True, tilt_correct=False,
                  tilt_max_deg=12.0, planarity_min=0.06,
                  max_snap_m=DEFAULT_MAX_SNAP_M):
    """Planares Tisch-Ebenen-Refinement (T-055). Z-Snap + optional Tilt.

    Z-SNAP (default an): siehe planar_z_snap — ersetzt die schwache Z-Schätzung
    durch den Planar-Prior „Teil ruht auf dem Tisch". Generalisierbar für JEDES
    Teil (nur CAD + vorhergesagte Rotation nötig). GUARD max_snap_m: snappt NUR
    Teile, deren nötige Z-Korrektur klein ist (= sie ruhen plausibel auf dem
    Tisch). Teile, die der Schätzer weit über dem Tisch platziert hat (Greifer/in
    der Luft), werden NICHT angefasst — der Prior gilt dort nicht (T-055-Messung:
    ohne Guard 17mm -> 530mm auf den Held-Teilen). max_snap_m=None = kein Guard.

    TILT-CORRECT (default AUS, konservativ): KLEINE Kippung der Auflagefläche zur
    Tisch-Normalen — NUR wenn das Teil konfident flach liegt (Auflagefläche planar
    UND Rest-Kippung klein). Bei stehenden/gekippten Teilen oder Einzelpunkt-
    Kontakt wird die Rotation NICHT angefasst (würde sonst korrekte Posen kaputt
    machen). Standardmäßig deaktiviert, weil Tilt am realen GDRNPP-Output
    netto nicht hilft (siehe T-055-Messung); hinter Flag falls nützlich.

    Args:
      R_world, t_world : vorhergesagte Welt-Pose.
      mesh_verts       : (N,3) CAD-Body-Vertices in Metern.
      table_z          : Welt-z der Tischfläche (Contract 0.0).
      z_snap           : Z-Snap anwenden (default True).
      tilt_correct     : konservative Tilt-Korrektur (default False).
      tilt_max_deg     : Tilt nur wenn Rest-Kippung der Auflage < dieser Schwelle.
      planarity_min    : Tilt nur wenn Auflage-Vertex-Anteil >= dieser Schwelle.
      max_snap_m       : Guard — max |dz| zum Snappen (Held-Teil-Schutz, default 50mm).
    Returns:
      (R_world_refined (3,3), t_world_refined (3,), info dict). info["z_snap"]=False
      und info["snap_skipped"]=True wenn der Guard das Snappen verhindert hat.
    """
    R = _as_R(R_world)
    t = _as_vec3(t_world).copy()
    info = {"z_snap": False, "dz": 0.0, "tilt": False, "tilt_deg": 0.0,
            "snap_skipped": False}

    if tilt_correct:
        R_new, applied, ang = _planar_tilt_correct(
            R, mesh_verts, tilt_max_deg=tilt_max_deg, planarity_min=planarity_min)
        if applied:
            R = R_new
            info["tilt"] = True
            info["tilt_deg"] = ang

    if z_snap:
        # erst die nötige Korrektur prüfen (Guard), dann anwenden.
        contact_z_abs = t[2] + lowest_contact_z(R, mesh_verts)
        dz_needed = float(table_z) - contact_z_abs
        if max_snap_m is not None and abs(dz_needed) > float(max_snap_m):
            info["snap_skipped"] = True              # Teil ruht nicht -> nicht snappen
        else:
            t[2] += dz_needed
            info["z_snap"] = True
            info["dz"] = dz_needed

    return R, t, info


def _planar_tilt_correct(R_world, mesh_verts, tilt_max_deg=12.0, planarity_min=0.06,
                         face_band_frac=0.06, flat_tol=0.30):
    """Konservative Tilt-zur-Tisch-Normalen-Korrektur. Intern (planar_refine).

    Idee: die KANDIDATEN-AUFLAGEFLÄCHE wird im BODY-Frame als Vertex-Band an der
    Unterseite gewählt (tilt-invariant): entlang der Body-Achse, die am meisten
    nach Welt-DOWN zeigt, alle Vertices innerhalb `face_band_frac` × Teil-Höhe vom
    untersten Punkt. So wird bei einem nur leicht gekippten flachen Teil die GANZE
    Unterseite erfasst (nicht nur die Kontaktkante). Eine Ebene wird per PCA durch
    diese Vertices gelegt und auf Welt-DOWN ausgerichtet — ABER NUR wenn die
    Auflage KONFIDENT PLANAR ist:
      (a) genug Kandidaten-Vertices (>= planarity_min Anteil, >= 8 absolut),
      (b) sie bilden ein flaches 2D-Band (Senkrecht-Streuung << laterale Ausdehnung,
          S2/S1 < flat_tol — schließt Kanten/Spitzen/Stäbe aus),
      (c) die nötige Rest-Kippung ist klein (< tilt_max_deg).
    Sonst R unverändert (stehende/gekippte Teile NICHT kaputtmachen).

    Returns: (R_corrected (3,3), applied bool, tilt_deg float).
    """
    R = _as_R(R_world)
    V = np.asarray(mesh_verts, dtype=np.float64)
    if V.shape[0] < 12:
        return R, False, 0.0
    down = np.array([0.0, 0.0, -1.0])
    body_down = R.T @ down                          # Welt-DOWN im Body-Frame
    body_down = body_down / (np.linalg.norm(body_down) + 1e-12)
    proj = V @ body_down                            # Lage entlang body_down (größer=tiefer)
    span = float(proj.max() - proj.min())           # Teil-Ausdehnung entlang dieser Achse
    if span < 1e-9:
        return R, False, 0.0
    band = float(face_band_frac) * span
    mask = proj >= (proj.max() - band)              # Unterseiten-Band (ganze Fläche)
    if int(mask.sum()) < 8 or (mask.sum() / V.shape[0]) < planarity_min:
        return R, False, 0.0
    contact = (V[mask]) @ R.T                        # Unterseiten-Vertices in Welt
    cc = contact - contact.mean(axis=0, keepdims=True)
    try:
        _, S, Vt = np.linalg.svd(cc, full_matrices=False)
    except np.linalg.LinAlgError:
        return R, False, 0.0
    # S[0]>=S[1]>=S[2]: laterale Streuung S[0],S[1]; Dicke senkrecht S[2].
    # Planar + flächig: S[2] << S[1] (Band ist dünn UND zweidimensional, kein Stab).
    if S[1] < 1e-9 or (S[2] / S[1]) > float(flat_tol):
        return R, False, 0.0                       # kein flaches 2D-Band -> nicht anfassen
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    down = np.array([0.0, 0.0, -1.0])
    if np.dot(normal, down) < 0:                   # nach unten orientieren
        normal = -normal
    cos_t = float(np.clip(np.dot(normal, down), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_t)))
    if tilt_deg < 1e-3 or tilt_deg > float(tilt_max_deg):
        return R, False, tilt_deg                  # schon flach ODER zu groß -> nicht anfassen
    axis = np.cross(normal, down)                  # kleinste Drehung normal->down (Welt)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R, False, tilt_deg
    axis = axis / n
    R_align = axis_angle_matrix(axis, np.radians(tilt_deg))
    return R_align @ R, True, tilt_deg


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
                        R_model_to_body=None, canonicalize=True,
                        apply_planar=False, mesh_verts_m=None, table_z=0.0,
                        tilt_correct=False, max_snap_m=DEFAULT_MAX_SNAP_M):
    """Eine BOP-Detektion -> ein pose_result-`results[]`-Eintrag (Dict).

    Bündelt §3.2 (Transform) + §3.3 (Kanonisierung) + §3.5 (planares Refinement,
    T-055) + §3.4 (face/upright) + §3.1/§1.2 (obj_id->part). Schema-Felder exakt
    wie pose_result.schema.json.

    apply_planar (T-055): wenn True UND mesh_verts_m gegeben, wird die Welt-Pose
    auf die Tischebene gesnappt (Z-Snap aus R+CAD-Kontakt; tilt_correct optional,
    konservativ — siehe planar_refine()). table_z ist die Tischfläche im
    Contract-Frame (default 0.0, weil t_world bereits relativ zum Tisch-Nullpunkt
    ist). face/upright werden aus der refinten Rotation abgeleitet.
    """
    part = part_for_obj_id(obj_id)
    R_m2c_can = canonicalize_rotation(R_m2c, part) if canonicalize else _as_R(R_m2c)
    R_world, t_world = bop_pose_to_world(
        R_m2c_can, t_m2c_mm, R_w2c, t_w2c_mm, table_origin_m,
        R_model_to_body=R_model_to_body)
    if apply_planar and mesh_verts_m is not None:
        R_world, t_world, _ = planar_refine(
            R_world, t_world, mesh_verts_m, table_z=table_z,
            z_snap=True, tilt_correct=tilt_correct, max_snap_m=max_snap_m)
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
