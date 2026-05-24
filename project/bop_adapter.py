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

# trimesh ist nur für die training-freien Rotations-Refinements (M1 Stable-Pose-
# Snap, M4 Contour-Yaw-Lock) nötig — und auch dort nur, wenn man die Stable-Poses
# / Silhouetten aus einem Mesh ableiten lässt. Der Kern-Adapter (§3.2 Transform,
# §3.3 Kanonisierung, §3.4 face/upright, §3.5 Z-Snap) bleibt rein numpy+stdlib und
# importierbar OHNE trimesh. Darum optional — fehlt es, wirft erst der konkrete
# Aufruf von compute_stable_poses/silhouette eine klare Meldung.
try:  # pragma: no cover - Umgebungs-abhängig
    import trimesh as _trimesh
except Exception:  # pragma: no cover
    _trimesh = None

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

# Default-Guard für M1 (Stable-Pose-Snap, T-041): Resting-Teile haben Tilt-Fehler
# von typ. <30–40°; ein um >55° „gekipptes" Teil ist plausibel nicht ruhend
# (steht/Greifer) → nicht snappen. Großzügiger als der Z-Snap-Guard, weil der
# Tilt-Flip selbst bis ~90° gehen kann und genau DEN wollen wir einfangen; >55°
# trennt resting (auch geflippt) von stehend. (Definition früh, weil
# planar_refine() ihn als Default-Argument referenziert.)
DEFAULT_MAX_TILT_SNAP_DEG = 55.0


def planar_refine(R_world, t_world, mesh_verts, table_z=0.0,
                  z_snap=True, tilt_correct=False,
                  tilt_max_deg=12.0, planarity_min=0.06,
                  max_snap_m=DEFAULT_MAX_SNAP_M,
                  stable_pose_snap=False, stable_downs=None, stable_probs=None,
                  max_tilt_snap_deg=DEFAULT_MAX_TILT_SNAP_DEG):
    """Planares Tisch-Ebenen-Refinement (T-055/T-041). Z-Snap + optional Tilt + M1.

    M1 STABLE-POSE-SNAP (default AUS, hinter Flag): snappt die Tilt-Komponente der
    Rotation auf die nächste Ruhelage (welche Fläche unten), behält den In-Plane-
    Yaw — siehe stable_pose_snap(). Braucht `stable_downs` (aus
    stable_pose_body_downs). Wird VOR dem Z-Snap angewandt (erst flach legen, dann
    auf die Tischhöhe snappen → konsistente Auflage). Kombinierbar mit Z-Snap; der
    DEFAULT-Pfad (planar_refine ohne Flag) bleibt unverändert (nur Z-Snap).


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
            "snap_skipped": False, "m1_snapped": False, "m1_tilt_deg": 0.0,
            "m1_skipped": False, "m1_pose_idx": -1}

    # M1 zuerst (flach legen), dann konservatives Tilt (falls separat gewünscht),
    # dann Z-Snap auf die korrigierte Auflage.
    if stable_pose_snap:
        if stable_downs is None:
            raise ValueError("stable_pose_snap=True braucht stable_downs "
                             "(siehe stable_pose_body_downs).")
        R, m1info = _apply_stable_pose_snap(
            R, stable_downs, stable_probs, max_tilt_snap_deg)
        info["m1_snapped"] = m1info["snapped"]
        info["m1_tilt_deg"] = m1info["tilt_deg"]
        info["m1_skipped"] = m1info["snap_skipped"]
        info["m1_pose_idx"] = m1info["pose_idx"]

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


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Training-freie ROTATIONS-Refinements (T-041 / S-047)
# Der Planar-Prior „Teil ruht auf bekannter Tischebene" beschränkt nicht nur die
# Tiefe (Z-Snap, oben), sondern auch die ROTATION: ein ruhendes Teil kann nur in
# einer von K diskreten Ruhe-Orientierungen (= welche Fläche unten) × In-Plane-Yaw
# liegen. M1/M3/M4 nutzen das — alles geometrisch, kein Training, keine GPU.
# ══════════════════════════════════════════════════════════════════════════════


def _min_rot_align(a, b):
    """Kleinste Rotation R mit R @ â = b̂ (Einheitsvektoren, Rodrigues-Form).

    Robuster Sonderfall bei a ≈ ±b. Liefert immer eine gültige SO(3)-Matrix.
    """
    a = np.asarray(a, float); a = a / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:                                     # parallel oder antiparallel
        if c > 0:
            return np.eye(3)
        # 180°: drehe um eine beliebige zu a senkrechte Achse.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        return axis_angle_matrix(perp, np.pi)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ── M1 — Stable-Pose-ROTATIONS-Snap ──────────────────────────────────────────
# Cache der Stable-Pose-„Body-Down"-Vektoren pro Mesh (compute_stable_poses ist
# teuer; pro Teil einmal reicht). Key = id(mesh_verts)+shape ist fragil → der
# Caller gibt besser einen stabilen `cache_key` (z.B. obj_id/part-Name).
_STABLE_DOWN_CACHE = {}


def stable_pose_body_downs(mesh=None, mesh_verts_mm=None, faces=None,
                           prob_min=0.02, max_k=None, cache_key=None):
    """K Ruhe-Orientierungen eines Teils → deren „Body-Down"-Achsen (+Wahrsch.).

    `trimesh.compute_stable_poses` liefert die statisch stabilen Ruhe-Transforms
    T (Mesh→Welt, sortiert nach Landewahrscheinlichkeit). Für jede Ruhelage ist die
    relevante Größe die **Body-Achse, die nach Welt-DOWN zeigt** (= welche Fläche
    unten liegt): b_down = R_stableᵀ · (0,0,-1). Genau diese Tilt-Information snappt
    M1 — der In-Plane-Yaw bleibt der Vorhersage überlassen.

    prob_min filtert degenerierte Kanten-Ruhelagen (z.B. Zahnrad hat 260 stabile
    Posen, aber nur die flache Lage p≈0.37 ist real; alles <0.02 ist Rauschen) →
    flach-liegende Teile snappen verlässlich auf ihre echte(n) Auflagefläche(n).

    Args:
      mesh          : trimesh.Trimesh (Einheit egal; nur Rotationen zählen) ODER
      mesh_verts_mm : (N,3) Vertices + optional `faces` (M,3) → baut ein Trimesh.
      prob_min      : min. Landewahrscheinlichkeit einer behaltenen Ruhelage.
      max_k         : optional Obergrenze der behaltenen Ruhelagen (nach Prob).
      cache_key     : stabiler Schlüssel zum Cachen (z.B. obj_id). None = kein Cache.

    Returns: (downs (K,3) Einheitsvektoren im Body-Frame, probs (K,)).
    """
    if cache_key is not None and cache_key in _STABLE_DOWN_CACHE:
        return _STABLE_DOWN_CACHE[cache_key]
    if _trimesh is None:
        raise RuntimeError(
            "M1 Stable-Pose-Snap braucht trimesh (pip install trimesh). "
            "Der Kern-Adapter läuft ohne; nur stable_pose_* benötigt es.")
    if mesh is None:
        if mesh_verts_mm is None:
            raise ValueError("stable_pose_body_downs: mesh ODER mesh_verts_mm nötig")
        V = np.asarray(mesh_verts_mm, float)
        mesh = _trimesh.Trimesh(vertices=V, faces=faces, process=False)
    Ts, probs = _trimesh.poses.compute_stable_poses(mesh, n_samples=1, threshold=0.0)
    down_w = np.array([0.0, 0.0, -1.0])
    downs, ps = [], []
    for T, p in zip(Ts, probs):
        if p < prob_min:
            continue
        R = np.asarray(T)[:3, :3]
        bd = R.T @ down_w
        bd = bd / (np.linalg.norm(bd) + 1e-12)
        downs.append(bd); ps.append(float(p))
        if max_k is not None and len(downs) >= max_k:
            break
    if not downs:                                    # prob_min zu hart → nimm die top-1
        R = np.asarray(Ts[0])[:3, :3]
        bd = R.T @ down_w; bd = bd / (np.linalg.norm(bd) + 1e-12)
        downs, ps = [bd], [float(probs[0])]
    out = (np.array(downs), np.array(ps))
    if cache_key is not None:
        _STABLE_DOWN_CACHE[cache_key] = out
    return out


def stable_pose_snap(R_world, stable_downs, max_tilt_snap_deg=None,
                     stable_probs=None):
    """M1: Snappe die Tilt-Komponente von R_world auf die nächste Ruhelage.

    Idee (geometrisch exakt, training-frei): R_world hat eine „Body-Down"-Achse
    b_pred = R_worldᵀ·(0,0,-1) (welche Body-Richtung gerade nach unten zeigt).
    Wir wählen unter den Ruhelagen-Body-Downs den nächsten b* und drehen R_world
    MINIMAL, sodass b* exakt nach Welt-DOWN zeigt:

        R_new = R_world · Align(b* → b_pred)

    Das ändert NUR die Auflage-Tilt (welche Fläche unten), der In-Plane-Yaw um die
    Welt-Z-Achse bleibt erhalten (die gute Netz-Vorhersage). Effekt:
      • Anker: der katastrophale ≥90°-Tilt-Flip (Stab kippt aus der Ebene) kollabiert
        auf eine gültige liegende Ruhelage; der freie Yaw-um-Y wird von cont-Y eh
        vergeben.
      • Zahnrad: legt das (evtl. gekippt/stehend vorhergesagte) Zahnrad flach auf
        seine Fläche — Voraussetzung für M4 (1-D-Yaw-Lock).

    GUARD (max_tilt_snap_deg): wie der Z-Snap-Guard. Ist die nötige Tilt-Korrektur
    sehr groß (Teil steht/wird gehalten/ist gekippt im Greifer), gilt der Ruhe-Prior
    NICHT → kein Snap (R unverändert). None = immer snappen.

    Args:
      R_world          : (3,3) world = R @ body (Vorhersage).
      stable_downs     : (K,3) Body-Down-Achsen der Ruhelagen (stable_pose_body_downs).
      max_tilt_snap_deg: max. erlaubte Tilt-Korrektur in Grad; darüber kein Snap.
      stable_probs     : optional (K,) — bei (fast) gleichweitem Match wird die
                         wahrscheinlichere Ruhelage bevorzugt (kleiner Tie-Break).
    Returns:
      (R_new (3,3), info dict). info["snapped"], info["tilt_deg"] (angewandt),
      info["pose_idx"] (gewählte Ruhelage), info["snap_skipped"].
    """
    R = _as_R(R_world)
    downs = np.asarray(stable_downs, float)
    if downs.ndim != 2 or downs.shape[1] != 3 or downs.shape[0] == 0:
        raise ValueError(f"stable_downs must be (K,3), got {downs.shape}")
    info = {"snapped": False, "tilt_deg": 0.0, "pose_idx": -1, "snap_skipped": False}

    b_pred = R.T @ np.array([0.0, 0.0, -1.0])
    b_pred = b_pred / (np.linalg.norm(b_pred) + 1e-12)

    cos = downs @ b_pred                              # Nähe jeder Ruhelage zur Vorhersage
    if stable_probs is not None:
        # winziger Prob-Bias (≤0.5°-Äquivalent): bricht Gleichstände zugunsten der
        # wahrscheinlicheren Lage, ohne eine klar nähere Lage zu überstimmen.
        p = np.asarray(stable_probs, float)
        cos = cos + 1e-4 * (p / (p.max() + 1e-12))
    k = int(np.argmax(cos))
    b_star = downs[k]
    tilt = float(np.degrees(np.arccos(np.clip(float(np.dot(b_star, b_pred)), -1.0, 1.0))))
    info["pose_idx"] = k

    if tilt < 1e-4:                                  # schon in einer Ruhelage
        return R, info
    if max_tilt_snap_deg is not None and tilt > float(max_tilt_snap_deg):
        info["snap_skipped"] = True                  # Teil ruht nicht → nicht anfassen
        return R, info

    R_align = _min_rot_align(b_star, b_pred)         # body: b* → b_pred
    R_new = R @ R_align                              # world = R @ body, Yaw-erhaltend
    info["snapped"] = True
    info["tilt_deg"] = tilt
    return R_new, info


def _apply_stable_pose_snap(R_world, stable_downs, stable_probs, max_tilt_snap_deg):
    """Interner Wrapper um stable_pose_snap (umgeht das Parameter-Shadowing in
    planar_refine, wo `stable_pose_snap` der Flag-Name ist). Gibt (R, info)."""
    return stable_pose_snap(R_world, stable_downs,
                            max_tilt_snap_deg=max_tilt_snap_deg,
                            stable_probs=stable_probs)


# ── M3 — Anker Top-Down-C_2-Check (view-abhängige Symmetrie) ──────────────────
def topdown_c2_flip_identical(mesh=None, mesh_verts_mm=None, faces=None,
                              R_world=None, n_px=128, iou_thresh=0.97,
                              cache_key=None):
    """Ist die Top-Down-Silhouette eines liegenden Teils unter 180°-In-Plane-Flip
    deckungsgleich? (M3 — entscheidet, ob der 180°-Flip view-äquivalent ist.)

    BOP-Distrib: der Anker-Flip ist eine VIEW-abhängige Near-Symmetrie (von oben
    sieht der Stab unter 180°-Drehung gleich aus, auch wenn die 3D-Enden minimal
    differ). Wir prüfen das EMPIRISCH: rendere die Top-Down-Silhouette der
    gegebenen (liegenden) Welt-Rotation, rotiere sie um 180° in der Bildebene und
    miss die IoU. Hohe IoU ⇒ der 180°-Flip ist top-down nicht unterscheidbar ⇒
    behandelbar als äquivalent (NUR für die Pose-Wahl, NICHT blind als models_info-
    C_2 — das würde echte 3D-Fehler gutschreiben).

    Args:
      R_world : (3,3) liegende Welt-Rotation (z.B. nach M1). Default: flach via
                erste Ruhelage, falls None.
      n_px    : Auflösung des orthografischen Silhouette-Rasters.
      iou_thresh: ab dieser Self-IoU gilt der Flip als top-down-identisch.
    Returns: (is_flip_identical bool, iou float).
    """
    verts = _resolve_verts_mm(mesh, mesh_verts_mm)
    if R_world is None:
        downs, _ = stable_pose_body_downs(mesh=mesh, mesh_verts_mm=mesh_verts_mm,
                                          faces=faces, cache_key=cache_key)
        # baue eine flache Welt-Rotation, deren Body-Down = downs[0].
        R_world = _min_rot_align(downs[0], np.array([0.0, 0.0, -1.0]))
    R = _as_R(R_world)
    # Top-Down-xy der gedrehten Vertices, ZENTRIERT auf den Schwerpunkt (der CAD-
    # Ursprung sitzt nicht zwingend im geometrischen Zentrum — der 180°-Flip muss um
    # das Silhouette-Zentrum erfolgen, sonst misst man nur die Schwerpunkts-Drift).
    P = (R @ verts.T).T[:, :2]
    c = P.mean(axis=0)
    Pc = P - c
    Pflip = -Pc                                      # 180° In-Plane-Rotation um das Zentrum
    # gemeinsame bbox für faire IoU beider Silhouetten.
    mn = np.minimum(Pc.min(axis=0), Pflip.min(axis=0))
    mx = np.maximum(Pc.max(axis=0), Pflip.max(axis=0))
    sil = _rasterize_points(Pc, n_px=n_px, bbox=(mn, mx))
    sil180 = _rasterize_points(Pflip, n_px=n_px, bbox=(mn, mx))
    iou = _mask_iou(sil, sil180)
    return bool(iou >= float(iou_thresh)), float(iou)


# ── M4 — Contour-Yaw-Lock für diskret-symmetrische Teile (Zahnrad C_N) ────────
def contour_yaw_lock(R_world, target_mask, K, t_world_m, table_origin_m,
                     R_w2c, t_w2c_mm, mesh=None, mesh_verts_mm=None, faces=None,
                     n_fold=7, sym_axis=(0.0, 1.0, 0.0), cache_key=None,
                     n_surface_samples=6000):
    """M4: wähle unter den N tooth-aligned Yaws den, dessen Silhouette am besten
    zur Detektor-Maske passt (1-D-Suche, CPU, kein GPU).

    Voraussetzung: M1 hat das Teil flach gelegt → der einzig offene Freiheitsgrad
    ist der In-Plane-Yaw um die Symmetrieachse. Für ein C_N-Teil sind die GDRNPP-
    nicht-auflösbaren Kandidaten {yaw + k·2π/N}. Wir rendern für jeden Kandidaten
    die Teil-Silhouette in den Kamera-Frame (orthografisch genähert über K +
    Bounding-Box-Fit) und scoren per IoU gegen die gegebene Detektor-Maske.

    Generalisierbar: n_fold aus models_info (Zahnrad N=7); für N=1/None No-Op.

    Args:
      R_world      : (3,3) Welt-Rotation NACH M1 (flach).
      target_mask  : (H,W) bool/0-1 Detektor-Maske des Teils im Bild.
      K            : (3,3) Kamera-Intrinsik.
      t_world_m,table_origin_m,R_w2c,t_w2c_mm : zum Rückprojizieren ins Kamera-Bild.
      n_fold       : N der C_N-Symmetrie (7 für Zahnrad). <2 → R unverändert.
      sym_axis     : Symmetrieachse im Body-Frame (Y).
    Returns:
      (R_best (3,3), info dict). info["k"], info["iou"], info["ious"] (alle N),
      info["applied"].
    """
    R = _as_R(R_world)
    info = {"applied": False, "k": 0, "iou": 0.0, "ious": []}
    if not n_fold or n_fold < 2:
        return R, info
    if target_mask is None:
        return R, info
    verts = _resolve_verts_mm(mesh, mesh_verts_mm)
    # Für eine GEFÜLLTE Silhouette dichte Oberflächen-Punkte samplen (Vertices allein
    # sind bei groben CAD-Meshes zu spärlich → löchrige Maske, keine Tooth-
    # Diskriminierung). Nur wenn ein trimesh-Mesh + sample verfügbar ist.
    if mesh is not None and _trimesh is not None and n_surface_samples:
        try:
            pts, _ = _trimesh.sample.sample_surface(mesh, int(n_surface_samples))
            verts = np.vstack([verts, np.asarray(pts, float)])
        except Exception:  # pragma: no cover - Fallback auf Vertices
            pass
    axis = np.asarray(sym_axis, float)
    tgt = np.asarray(target_mask).astype(bool)

    # Kamera-Pose des Body-Frames: R_m2c = R_w2c @ R_world (R_model_to_body=I).
    R_w2c = _as_R(R_w2c)
    t_world = _as_vec3(t_world_m)
    table_origin = _as_vec3(table_origin_m)
    t_w2c = _as_vec3(t_w2c_mm)
    # body-Origin in Welt (mm) → Kamera (mm): invertiert die Adapter-Translationskette.
    t_m2w_mm = (t_world + table_origin) * 1000.0
    t_m2c = R_w2c @ t_m2w_mm + t_w2c

    ious = []
    for k in range(int(n_fold)):
        Rk = R @ axis_angle_matrix(axis, k * 2.0 * np.pi / n_fold)
        R_m2c = R_w2c @ Rk
        sil = _camera_silhouette(verts, R_m2c, t_m2c, _as_R(K), tgt.shape)
        ious.append(_mask_iou(sil, tgt))
    ious = np.asarray(ious)
    kbest = int(np.argmax(ious))
    R_best = R @ axis_angle_matrix(axis, kbest * 2.0 * np.pi / n_fold)
    info.update(applied=True, k=kbest, iou=float(ious[kbest]),
                ious=[float(v) for v in ious])
    return R_best, info


# ── Silhouette-/Masken-Hilfen (CPU, kein GPU) ─────────────────────────────────
def _resolve_verts_mm(mesh, mesh_verts_mm):
    """Vertices (N,3) holen — aus trimesh-Mesh oder direkt aus Array."""
    if mesh is not None:
        return np.asarray(mesh.vertices, float)
    if mesh_verts_mm is None:
        raise ValueError("mesh ODER mesh_verts_mm nötig")
    return np.asarray(mesh_verts_mm, float)


def _camera_silhouette(verts, R_m2c, t_m2c_mm, K, hw):
    """Perspektivische Silhouette der Vertices im Kamera-Bild (Maske der bbox).

    Projiziert die Body-Vertices via R_m2c/t_m2c (mm) + K ins Bild und rastert die
    belegten Pixel. Gibt eine (H,W)-bool-Maske zur IoU gegen die Detektor-Maske.
    """
    H, W = hw
    P = (R_m2c @ verts.T).T + t_m2c_mm.reshape(1, 3)  # Kamera-Frame (mm)
    z = P[:, 2]
    valid = z > 1e-6
    P = P[valid]
    if P.shape[0] == 0:
        return np.zeros((H, W), bool)
    u = (K[0, 0] * P[:, 0] / P[:, 2]) + K[0, 2]
    v = (K[1, 1] * P[:, 1] / P[:, 2]) + K[1, 2]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    mask = np.zeros((H, W), bool)
    mask[vi[m], ui[m]] = True
    return _dilate(mask, 1)                           # dünne Punktwolke → kleine Dilation


def _rasterize_points(P2, n_px=128, pad=0.06, bbox=None):
    """(N,2)-Punkte → n_px×n_px-bool-Belegung (bbox-normiert, mit Rand).

    bbox=(mn,mx): optionale GEMEINSAME Bounding-Box (für faire IoU zweier
    Silhouetten im selben Raster). Default: jede Silhouette eigene bbox.
    """
    if P2.shape[0] == 0:
        return np.zeros((n_px, n_px), bool)
    if bbox is None:
        mn = P2.min(axis=0); mx = P2.max(axis=0)
    else:
        mn = np.asarray(bbox[0], float).copy(); mx = np.asarray(bbox[1], float).copy()
    span = (mx - mn)
    span[span < 1e-12] = 1e-12
    m = pad * span.max()
    mn = mn - m; mx = mx + m
    span = (mx - mn)
    span[span < 1e-12] = 1e-12
    ij = ((P2 - mn) / span * (n_px - 1)).astype(np.int64)
    ij = np.clip(ij, 0, n_px - 1)
    img = np.zeros((n_px, n_px), bool)
    img[ij[:, 1], ij[:, 0]] = True
    return _dilate(img, 1)


def _dilate(mask, it=1):
    """Billige 4-Nachbar-Dilation (füllt Punktwolken-Lücken, kein scipy nötig)."""
    m = mask.copy()
    for _ in range(int(it)):
        d = m.copy()
        d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        m = d
    return m


def _mask_iou(a, b):
    """IoU zweier bool-Masken (0..1). Beide leer → 1.0."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


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
                        tilt_correct=False, max_snap_m=DEFAULT_MAX_SNAP_M,
                        stable_pose_snap=False, stable_downs=None, stable_probs=None,
                        max_tilt_snap_deg=DEFAULT_MAX_TILT_SNAP_DEG,
                        contour_yaw_lock=False, target_mask=None, K=None,
                        n_fold=None, sym_axis=(0.0, 1.0, 0.0), mesh=None):
    """Eine BOP-Detektion -> ein pose_result-`results[]`-Eintrag (Dict).

    Bündelt §3.2 (Transform) + §3.3 (Kanonisierung) + §3.5 (planares Refinement,
    T-055) + Phase-3 Rotations-Refinements (T-041: M1 Stable-Pose-Snap, M4 Contour-
    Yaw-Lock) + §3.4 (face/upright) + §3.1/§1.2 (obj_id->part). Schema-Felder exakt
    wie pose_result.schema.json.

    apply_planar (T-055): wenn True UND mesh_verts_m gegeben, wird die Welt-Pose
    auf die Tischebene gesnappt (Z-Snap aus R+CAD-Kontakt; tilt_correct optional,
    konservativ). table_z ist die Tischfläche im Contract-Frame.

    DEFAULTS (T-041-Messung an realen GDRNPP-val-Predictions, s. REFINE_T041.md):
      • Z-Snap (apply_planar)   = der einzige training-freie AR-Gewinn (Anker +0.04..
        0.06) → in der Pipeline AN.
      • stable_pose_snap (M1)   = DEFAULT AUS: an diesem val-Set netto neutral bis
        leicht negativ (Anker −0.002; Zahnrad ±0). Hinter Flag, generalisierbar.
      • contour_yaw_lock (M4)   = DEFAULT AUS: 0 AR-Wirkung auf das C_7-Zahnrad
        (die AR-Metrik ist bereits symmetrie-bewusst → die 7 Zahn-Yaws sind
        äquivalent; zudem ist die Top-Down-Silhouette eines C_7-Teils rotations-
        invariant). NUR nützlich für deterministische Viewer-Yaw-Wahl ODER für
        diskret-symmetrische Teile mit SILHOUETTE-brechendem Merkmal.
      • M3 (Anker-C_2)          = NICHT angewandt: Top-Down-Self-IoU 0.58/0.64 ≪ 1
        → der Anker-Flip ist KEINE view-Symmetrie (Kopf vs. dünner Schaft) → ein
        blinder C_2 würde echte Fehler gutschreiben. Entscheidung dokumentiert.
    """
    part = part_for_obj_id(obj_id)
    R_m2c_can = canonicalize_rotation(R_m2c, part) if canonicalize else _as_R(R_m2c)
    R_world, t_world = bop_pose_to_world(
        R_m2c_can, t_m2c_mm, R_w2c, t_w2c_mm, table_origin_m,
        R_model_to_body=R_model_to_body)
    # M1 + Z-Snap (+ optional Tilt) via planar_refine.
    if (apply_planar or stable_pose_snap) and mesh_verts_m is not None:
        R_world, t_world, _ = planar_refine(
            R_world, t_world, mesh_verts_m, table_z=table_z,
            z_snap=apply_planar, tilt_correct=tilt_correct, max_snap_m=max_snap_m,
            stable_pose_snap=stable_pose_snap, stable_downs=stable_downs,
            stable_probs=stable_probs, max_tilt_snap_deg=max_tilt_snap_deg)
    # M4 contour yaw-lock (nur diskret-symmetrische Teile mit Maske + K).
    if contour_yaw_lock and target_mask is not None and K is not None and n_fold:
        R_world, _ = globals()["contour_yaw_lock"](
            R_world, target_mask, K, t_world, table_origin_m, R_w2c, t_w2c_mm,
            mesh=mesh, mesh_verts_mm=(mesh_verts_m * 1000.0
                                      if mesh is None and mesh_verts_m is not None
                                      else None),
            n_fold=n_fold, sym_axis=sym_axis)
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
