#!/usr/bin/env python3
"""POSE — standalone end-to-end inference: ein Bild -> pose_result.json.

    python project/e2e_infer.py --image project/input/scene_0000.png
    python project/e2e_infer.py --image project/input/scene_0000.png --out out.json
    python project/e2e_infer.py --image project/input/scene_0000.png --serve

Selbst-enthalten: die Pipeline-Logik ist INLINE (kein Import auf Projektmodule),
damit dieses eine Skript für sich allein lauffähig und weitergebbar ist.

Pipeline:
  1. Teile finden   — echter OBB-Detektor (models/detector.pt, YOLOv8-OBB via
                      ultralytics) liefert orientierte Boxen + Klasse. Fallback:
                      SDG-Annotator-Boxen (bbox_2d_<idx>.json), dann Dummy-Box.
  2. Crop           — achsenparalleler Ausschnitt der OBB-Hülle pro Detection.
  3. 6D-Pose        — GDRNPP (BOP-SOTA, siehe ADR-018): liefert pro Detektion
                      R_world + t_world. Aktuell STUB (Anbindung in W2/W3).
  4. pose_result    — gegen den Contract validiert (stdlib immer; jsonschema wenn
                      verfügbar), dann geschrieben. Schema-valide oder es wird
                      nichts geschrieben.

Mit --serve startet das Skript danach einen localhost-Server (ab project/) und
öffnet den 3D-Viewer (frontend/) auf das erzeugte pose_result.

Hinweis (ADR-018, BOP-Pivot): Der alte Eigenbau-Pose-Mittelteil (Face-Atlas /
faces_<part>.json-Registry / Template-Bank-Render-and-Compare / Face-Classifier /
template-MSE-Yaw / Eigenbau-Backprojection) ist BEWUSST entfernt ("für die
Tonne"). An seiner Stelle steht GDRNPP — bis dessen Inferenz angebunden ist
(W2/W3), ist Stufe 3 ein Stub, der keine Posen liefert; der Contract bleibt
unverändert und schema-valide.

Konvention (eingefroren): Z-up Welt, world = R @ body (Spaltenkonvention),
Ursprung = Tisch-Nullpunkt, Einheit Meter.

Abhängigkeiten: numpy, PIL (Pflicht). torch + ultralytics nur für den Detektor-
Checkpoint (sonst Fallback). jsonschema optional (Bonus-Gate).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

# BOP -> pose_result Adapter (Viktor §3) — eigenständiges Geschwister-Modul,
# inline-fähig (Max-Präferenz). Beide Gleise (GDRNPP/MegaPose) gehen hier durch.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bop_adapter as BOP

# M2 Multi-Hypothesen-Render-and-Compare-Refiner (T-058). Optional importiert —
# nur nötig, wenn refine_rc aktiv ist. Fehlt das Modul, läuft die Kette ohne RC.
try:
    import refine_rc as RC
except Exception:  # pragma: no cover
    RC = None

# TTA — Test-Time-Augmentation für die GDRNPP-Rotation (T-058, S-049). Optional,
# inferenzseitig, hinter Flag. Fehlt das Modul, läuft die Kette ohne TTA.
try:
    import tta_pose as TTA
except Exception:  # pragma: no cover
    TTA = None

# ── Pfade (relativ zu diesem Skript, CWD-unabhängig) ──────────────────────────
HERE = pathlib.Path(__file__).resolve().parent          # project/
MODELS = HERE / "models"
# pose_result-Contract (in project/ committet; ältere docs/-Lage als Fallback).
SCHEMA_FILE = next((p for p in (HERE / "pose_result.schema.json",
                                HERE.parent / "docs" / "pose_result.schema.json")
                    if p.exists()), HERE / "pose_result.schema.json")

SCHEMA_VERSION = "1.0.0"
COORD_CONVENTION = ("Z-up world; column rotation world = R @ body; "
                    "origin = table-plane null-point")

# Tisch-Nullpunkt = Tray-Arbeitsfläche (Welt, Meter), x/y = GST-Welt-Ursprung,
# damit die Teile mit dem cell.glb (echtes CAD, Welt-Frame) fluchten.
TABLE_ORIGIN_SCENE = (0.0, 0.0, 0.08)

# Planares Tisch-Ebenen-Refinement (T-055): Z-Snap der Welt-Pose auf die
# Tischebene. Default AN für den planaren Setup (alle Teile ruhen auf dem Tisch).
# Im Contract-Frame ist die Tischfläche bei Welt-z = 0.0 (t_world ist bereits
# relativ zum Tisch-Nullpunkt). Tilt-Korrektur default AUS (konservativ, hilft
# am realen Schätzer netto nicht — siehe T-055-Messung).
PLANAR_REFINE_DEFAULT = True
PLANAR_TABLE_Z = 0.0
PLANAR_TILT_CORRECT = False

# M2 Render-and-Compare-Refiner (T-058). DEFAULT AUS bis validiert (GPU-Pfad beim
# Finish, CPU-Kanten-Scorer als training-freier Fallback). Hinter Flag refine_rc.
# scorer: "cpu_edge" (kein GPU, default) | "megapose" (GPU, finish-time).
REFINE_RC_DEFAULT = False
REFINE_RC_SCORER = "cpu_edge"
# Visibility-aware Render-Compare (ADR-020, S-003, T-085). DEFAULT AUS. Eng
# gescopter Pfad NUR für Eval/Repro: wenn AN und eine Detektion eine sichtbare
# Region (`d["visib_mask"]`, offline aus BOP `mask_visib/`) trägt, scoret refine_rc
# NUR über die sichtbare Region + ein visibility-konditioniertes Gate. Löst den
# Anker-180°-Quer-Flip bei partieller Sicht (Kopf sichtbar, Schaft occludiert).
# LIVE bleibt unberührt: der OBB-Detektor liefert keine `mask_visib` -> visib_mask
# ist im Live-Pfad None -> exakt heutiges Verhalten (kein stiller Drift). Live-
# Proxy-Visibility = dokumentierter Follow-up (S-005), NICHT hier.
REFINE_RC_VISAWARE_DEFAULT = False
REFINE_RC_MIN_VISIB_FRACT = 0.25
# Visibility-GESTAFFELTES Margin-Gate (ADR-020, S-003 v2, T-085). Wenn das visaware-
# Flag AN ist UND eine Detektion eine sichtbare Region trägt, staffelt dieses
# Schedule den effektiven Margin nach visib_fract: aggressiv (niedrig) im
# occludierten Band, konservativ (= shipped 0.15) im well-vis Band, gar kein Switch
# unterhalb der Mindest-Sichtbarkeit. Das ist der Hebel, der den vis-aware Score
# operativ wirksam macht OHNE well-vis-Regression (statisches Gate kann beides nicht).
# None -> kein Schedule (reines statisches Margin-Gate, rückwärtskompatibel). Die
# konkreten Schwellen sind die T-085-Sweep-Kalibrierung (siehe eval_s003.md).
REFINE_RC_MARGIN_SCHEDULE = None    # z.B. RC.DEFAULT_MARGIN_SCHEDULE wenn enabled
# TTA (T-058, S-049). DEFAULT AUS. Inferenzseitig, RGB-only, kein Retrain. Härtet
# die View-Empfindlichkeit der per-View-Einzel-Rotation von GDRNPP ab (in-plane
# 90°-Rotationen des Crops, exakte Pixel-Ops, Rück-Transform + Aggregation auf
# SO(3)). Erwartung: kleiner unimodaler Gewinn (+1..3 AR Literatur), LÖST NICHT
# das falsche Becken / den 3D-Flip (dafür M2-MegaPose / SO(3)-Head Phase-3).
# Validierbar erst GPU-frei am echten Checkpoint.
TTA_DEFAULT = False
TTA_N_ROT = 4               # 1=aus, 2={0,180}, 4=alle vier in-plane-90°-views
TTA_HFLIP = False           # Spiegelung — ändert Chiralität, default AUS
TTA_AGG = "medoid"          # medoid (score-frei, bimodal-robust) | chordal | score
# C_N pro obj_id (aus models_info): Anker continuous-Y -> kein discrete N; Zahnrad
# C_7 (-> Yaw-Hypothesen). Generalisierbar via PART_SYMMETRY im Adapter.
OBJ_NFOLD = {1: None, 2: None, 6: 7}
# CAD-Meshes (BOP models, mm) für den Z-Snap. Erst project/cad_input, dann die
# BOP-models-Lage. Lazy geladen, in Metern gecacht. Fehlt das Mesh -> Skip
# (kein Snap, Pose unverändert) statt hart zu brechen.
_MESH_CACHE: dict = {}
_CAD_DIRS = [HERE / "cad_input" / "models", HERE / "bop" / "pose_isaac" / "models_eval",
             HERE / "bop" / "pose_isaac" / "models", HERE / "models" / "cad"]


def _load_mesh_verts_m(obj_id, warn=print):
    """CAD-Mesh-Vertices (Body-Frame, **Meter**) für obj_id. None wenn keins da.

    Liest die BOP-PLY (Vertices in mm) und skaliert auf Meter (Contract-Einheit).
    Generalisierbar: jedes Teil mit obj_<id>.ply funktioniert. Stdlib-PLY-Parser
    (ASCII + binär-little-endian), kein extra Dependency — der Adapter braucht nur
    die Vertex-Wolke (Faces egal für den tiefsten-Punkt-Snap)."""
    if obj_id in _MESH_CACHE:
        return _MESH_CACHE[obj_id]
    verts = None
    for d in _CAD_DIRS:
        p = d / f"obj_{int(obj_id):06d}.ply"
        if p.exists():
            try:
                verts = _read_ply_verts(p) / 1000.0      # mm -> m
                warn(f"[planar] CAD-Mesh {p.name} ({verts.shape[0]} Vertices) geladen")
                break
            except Exception as exc:
                warn(f"[planar] CAD-Mesh {p} unlesbar ({exc!r})")
    _MESH_CACHE[obj_id] = verts
    return verts


def _read_ply_verts(path):
    """Minimaler PLY-Vertex-Reader (ASCII + binary_little_endian). -> (N,3) float64."""
    import struct
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("kein PLY")
        fmt = None; n_vert = 0; props = []; in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY-Header unvollständig")
            s = line.strip().split()
            if not s:
                continue
            if s[0] == b"format":
                fmt = s[1]
            elif s[0] == b"element":
                in_vertex = (s[1] == b"vertex")
                if in_vertex:
                    n_vert = int(s[2])
            elif s[0] == b"property" and in_vertex and s[1] != b"list":
                props.append((s[2].decode(), s[1].decode()))
            elif s[0] == b"end_header":
                break
        names = [n for n, _ in props]
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        if fmt == b"ascii":
            verts = []
            for _ in range(n_vert):
                vals = f.readline().split()
                verts.append((float(vals[xi]), float(vals[yi]), float(vals[zi])))
            return np.asarray(verts, dtype=np.float64)
        # binary_little_endian
        type_map = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                    "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
                    "ushort": "H", "uint16": "H", "short": "h", "int16": "h",
                    "uint": "I", "uint32": "I", "int": "i", "int32": "i"}
        offsets = []; size = 0; xyz_off = {}
        for n, t in props:
            c = type_map[t]; sz = struct.calcsize("<" + c)
            if n in ("x", "y", "z"):
                xyz_off[n] = (size, c)
            size += sz
        buf = f.read(size * n_vert)
        verts = np.empty((n_vert, 3), dtype=np.float64)
        for i in range(n_vert):
            base = i * size
            for j, ax in enumerate(("x", "y", "z")):
                off, c = xyz_off[ax]
                verts[i, j] = struct.unpack_from("<" + c, buf, base + off)[0]
        return verts


def available_parts():
    """Teile-Namen — Single-Source = bop_adapter.OBJ_ID_TO_PART (§1.2, eingefroren).

    Optionaler Override via models/part_meta.json (falls ein Cleanup sie wieder
    hinlegt). Fehlt die Datei (Default seit ADR-018-Cleanup), wird die kanonische
    obj_id->part-Map des Adapters benutzt — KEINE separate, driftende Liste mehr.
    """
    pm = MODELS / "part_meta.json"
    if pm.exists():
        try:
            data = json.load(open(pm))
            names = list(data.get("parts", data)) if isinstance(data, (dict, list)) else []
            if names:
                return sorted(str(n) for n in names)
        except Exception:
            pass
    # Kanonisch aus dem Adapter (gleiche Quelle wie obj_id-Mapping, scene_gt, PLY).
    return sorted(BOP.OBJ_ID_TO_PART.values())


def _canonical_parts():
    return {p.lower(): p for p in available_parts()}


def canonical_part(raw: str, canon: dict[str, str]) -> str:
    """Mappt ein rohes Label (beliebige Schreibweise) auf den kanonischen
    part-Namen. Unbekannte Labels werden unverändert durchgereicht."""
    return canon.get(raw.strip().lower(), raw)


# ── Stufe 1: Detections (OBB-Detektor, sonst SDG-BBoxes, sonst Dummy) ─────────
# Drei Quellen, in dieser Priorität, jede ein sauberer Fallback der vorigen:
#   1. models/detector.pt  — echt trainierter YOLOv8-OBB-Detektor (ultralytics).
#   2. bbox_2d_<idx>.json   — SDG-Annotator-Boxen neben dem Bild (Sim-Ground-Truth).
#   3. Dummy                — eine ganze-Bild-Box, damit die Kette nie hart bricht.
DETECTOR_FILE = MODELS / "detector.pt"
DETECTOR_CONF = 0.40
DETECTOR_IMGSZ = 1280
_DETECTOR_CACHE: dict = {}


def _load_detector():
    """YOLOv8-OBB-Checkpoint lazy laden. None wenn kein .pt / kein ultralytics."""
    if "m" in _DETECTOR_CACHE:
        return _DETECTOR_CACHE["m"]
    m = None
    if DETECTOR_FILE.exists():
        try:
            from ultralytics import YOLO
            m = YOLO(str(DETECTOR_FILE))
        except Exception:
            m = None
    _DETECTOR_CACHE["m"] = m
    return m


def _obb_to_aabb(corners, W: int, H: int) -> list[int]:
    """Orientierte 4-Eck-Box -> achsenparallele BBox [x0,y0,x1,y1] (geklemmt)."""
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    x0 = max(0, min(int(round(min(xs))), W - 1)); x1 = max(0, min(int(round(max(xs))), W))
    y0 = max(0, min(int(round(min(ys))), H - 1)); y1 = max(0, min(int(round(max(ys))), H))
    return [x0, y0, x1, y1]


def _obb_angle_deg(corners):
    """In-plane Winkel der langen OBB-Achse (Grad), als Yaw-Prior fürs Pose-Backend."""
    p = np.asarray(corners, float)
    e01 = p[1] - p[0]; e12 = p[2] - p[1]
    long_edge = e01 if np.hypot(*e01) >= np.hypot(*e12) else e12
    return float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))


def detect_with_model(img_path, canon, warn=print):
    """Echte OBB-Detektionen via models/detector.pt. [] wenn nicht verfügbar."""
    m = _load_detector()
    if m is None:
        return []
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]
    try:
        r = m.predict(str(img_path), imgsz=DETECTOR_IMGSZ, conf=DETECTOR_CONF, verbose=False)[0]
    except Exception as exc:
        warn(f"[detect] Detektor-Inferenz fehlgeschlagen ({exc!r}) — Fallback")
        return []
    if r.obb is None or len(r.obb) == 0:
        return []
    polys = r.obb.xyxyxyxy.cpu().numpy()          # (N,4,2)
    cls = r.obb.cls.cpu().numpy().astype(int)
    conf = r.obb.conf.cpu().numpy()
    names = r.names
    dets = []
    for inst, (poly, c, cf) in enumerate(zip(polys, cls, conf)):
        corners = [[float(x), float(y)] for x, y in poly]
        bbox = _obb_to_aabb(corners, W, H)
        if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
            continue
        raw = names[int(c)]
        dets.append({"instance_id": inst, "part": canonical_part(raw, canon),
                     "bbox_2d": bbox, "raw_label": raw, "occlusion": 0.0,
                     "obb_corners": corners, "obb_angle_deg": _obb_angle_deg(corners),
                     "det_conf": float(cf)})
    warn(f"[detect] {len(dets)} orientierte Boxen vom OBB-Detektor (detector.pt)")
    return dets


def find_bbox_json(img_path) -> pathlib.Path | None:
    """Sucht die zum Bild gehörende SDG-Annotator-BBox-JSON (bbox_2d_<stem>.json,
    sonst die erste bbox_2d_*.json im selben Ordner). None, wenn keine da."""
    img_path = pathlib.Path(img_path)
    stem = img_path.stem.replace("rgb_", "").replace("scene_", "")
    cands = [img_path.parent / f"bbox_2d_{stem}.json"]
    cands += sorted(img_path.parent.glob("bbox_2d_*.json"))
    for c in cands:
        if c.exists():
            return c
    return None


def detections_for(img_path, warn=print):
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]
    canon = _canonical_parts()
    # 1) echter OBB-Detektor (bevorzugt)
    dets = detect_with_model(img_path, canon, warn=warn)
    if dets:
        return rgb, dets
    # 2) SDG-Annotator-Boxen neben dem Bild
    dets = []
    bj = find_bbox_json(img_path)
    if bj is not None:
        data = json.load(open(bj))
        rows = data.get("data", [])
        id2 = (data.get("info", {}) or {}).get("idToLabels", {})
        for inst, row in enumerate(rows):
            if len(row) < 5:
                continue
            raw = (id2.get(str(int(row[0])), {}) or {}).get("class", "")
            x0, x1 = sorted((int(round(row[1])), int(round(row[3]))))
            y0, y1 = sorted((int(round(row[2])), int(round(row[4]))))
            x0, x1 = max(0, min(x0, W - 1)), max(0, min(x1, W))
            y0, y1 = max(0, min(y0, H - 1)), max(0, min(y1, H))
            if (x1 - x0) < 4 or (y1 - y0) < 4:
                continue
            dets.append({"instance_id": inst, "part": canonical_part(raw, canon),
                         "bbox_2d": [x0, y0, x1, y1], "raw_label": raw,
                         "occlusion": float(row[5]) if len(row) > 5 else 0.0})
        warn(f"[detect] {len(dets)} BBoxes aus SDG-Annotator {bj.name}")
    if not dets:
        part = available_parts()[0] if available_parts() else "Anker_Lang"
        dets = [{"instance_id": 0, "part": part, "bbox_2d": [0, 0, W, H],
                 "raw_label": part, "occlusion": 0.0}]
        warn(f"[detect] DUMMY: eine ganze-Bild-BBox als '{part}'")
    return rgb, dets


# === BOP-6D-Pose: GDRNPP-Inferenz + Adapter (Viktor §3, ADR-018) ==============
# Stufe 3-Kette (pro Detektion): Crop+bbox -> call_gdrnpp(crop,K,bbox,obj_id) ->
# (R_m2c, t_m2c) [BOP-Konvention, mm, Kamera-Frame] -> BOP-Adapter (§3.2/§3.3/
# §3.4) -> pose_result-Eintrag. **Eine Adapter-Funktion, beide Gleise.**
#
# GDRNPP-Checkpoint existiert noch NICHT (Kai trainiert parallel, S-402). Daher:
#   - call_gdrnpp() hat einen MOCK-Modus, der pro Detektion plausible (R_m2c,
#     t_m2c) liefert, sodass die GANZE Kette JETZT grün durchläuft.
#   - der Checkpoint-Pfad ist Arg/Config (GdrnppConfig.checkpoint). Liegt ein
#     echtes .pth da, schaltet die Kette automatisch auf den echten Call.
#   >>> Kais echter GDRNPP-Call wird in _gdrnpp_real() eingehängt (DIE Stelle). <<<

# ── Default-Kamera (BOP §1.3): nur wenn keine scene_camera.json neben dem Bild ──
# Plausibler Zivid-artiger Top-Down-Blick auf den Tray, RGB-only. K aus den
# bestehenden Default-Intrinsics (sim_code/render_dataset.py: focal 24mm,
# sensor 20.955mm). Extrinsics: Kamera ~0.5m über Tisch, blickt nach -Z.
DEFAULT_CAM_HEIGHT_M = 0.5
DEFAULT_FOCAL_MM = 24.0
DEFAULT_SENSOR_MM = 20.955


def _default_K(W, H):
    """Pinhole-K (3x3 row-major) aus Default-Intrinsics für ein WxH-Bild."""
    f_px = DEFAULT_FOCAL_MM / DEFAULT_SENSOR_MM * max(W, H)
    return np.array([[f_px, 0.0, W / 2.0],
                     [0.0, f_px, H / 2.0],
                     [0.0, 0.0, 1.0]])


def _default_extrinsics():
    """World->Cam (BOP), mm. Kamera über dem Tisch, Blick nach Welt-DOWN (-Z).

    Welt ist Z-up; die Kamera schaut von oben auf den Tisch. Kamera-Blickachse
    (+Z_cam) zeigt nach Welt-(-Z). Eine 180°-Drehung um X bildet Welt-Z auf
    Cam-(-Z) ab und Welt-Y auf Cam-(-Y) (Bild steht richtig herum)."""
    R_w2c = BOP.axis_angle_matrix([1.0, 0.0, 0.0], np.pi)   # 180° um X
    # Kamera-Ursprung in Welt: (0,0,height). t_w2c = -R_w2c @ cam_origin, in mm.
    cam_origin_m = np.array([0.0, 0.0, DEFAULT_CAM_HEIGHT_M])
    t_w2c_mm = -(R_w2c @ cam_origin_m) * 1000.0
    return R_w2c, t_w2c_mm


def load_scene_camera(img_path, W, H, warn=print):
    """Kamera-Intrinsics + Extrinsics (BOP §1.3). scene_camera.json neben dem
    Bild, sonst Default. Liefert (K 3x3, R_w2c 3x3, t_w2c_mm 3,)."""
    img_path = pathlib.Path(img_path)
    cand = img_path.parent / "scene_camera.json"
    if cand.exists():
        try:
            data = json.load(open(cand))
            # BOP: dict keyed per im_id; nimm den passenden oder den ersten.
            stem = img_path.stem.replace("rgb_", "").replace("scene_", "")
            key = str(int(stem)) if stem.isdigit() and str(int(stem)) in data \
                else next(iter(data))
            cam = data[key]
            K = np.asarray(cam["cam_K"], float).reshape(3, 3)
            R_w2c = np.asarray(cam["cam_R_w2c"], float).reshape(3, 3)
            t_w2c = np.asarray(cam["cam_t_w2c"], float).reshape(3)
            warn(f"[cam] scene_camera.json[{key}] geladen (BOP §1.3)")
            return K, R_w2c, t_w2c
        except Exception as exc:
            warn(f"[cam] scene_camera.json unlesbar ({exc!r}) — Default-Kamera")
    R_w2c, t_w2c = _default_extrinsics()
    warn("[cam] Default-Kamera (kein scene_camera.json) — Top-Down ~0.5m, RGB-only")
    return _default_K(W, H), R_w2c, t_w2c


class GdrnppConfig:
    """Konfiguration für den GDRNPP-Call. Checkpoint-Pfad als Config-Feld.

    - checkpoint: Pfad zum GDRNPP-Checkpoint (.pth). None/fehlend -> MOCK-Modus.
    - mock:       erzwingt MOCK auch wenn ein Checkpoint da ist (für Tests).
    """
    def __init__(self, checkpoint=None, mock=None):
        self.checkpoint = pathlib.Path(checkpoint) if checkpoint else None
        # Auto: Mock wenn kein Checkpoint existiert; explizit überschreibbar.
        if mock is None:
            mock = not (self.checkpoint and self.checkpoint.exists())
        self.mock = bool(mock)


def _crop(rgb, bbox):
    """Achsenparalleler Crop [x0,y0,x1,y1] (geklemmt)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return rgb
    return rgb[y0:y1, x0:x1]


def _gdrnpp_mock(crop, K, bbox, obj_id, table_z_m=None):
    """Deterministischer MOCK: plausible (R_m2c, t_m2c) pro Detektion.

    Baut eine ABSOLUTE Welt-Pose, die garantiert auf dem Tisch sitzt (Teil flach,
    leichter Yaw aus der bbox-Position, z = Tisch-Höhe), projiziert sie via der
    Default-Extrinsics in den Kamera-Frame (BOP) und gibt (R_m2c, t_m2c_mm)
    zurück. So sieht der Viewer plausible Posen JETZT — ohne echten Checkpoint.

    table_z_m: absolute Welt-Höhe der Tischebene (= TABLE_ORIGIN_SCENE[2]); nach
    der Adapter-Subtraktion von table_origin landet t_world.z ~ 0.
    """
    if table_z_m is None:
        table_z_m = TABLE_ORIGIN_SCENE[2]
    R_w2c, t_w2c = _default_extrinsics()
    # Welt-Pose: Teil liegt flach (R_face ~ Identität), Yaw aus bbox-x-Mitte.
    cx = (bbox[0] + bbox[2]) / 2.0
    yaw = (cx % 360) * np.pi / 180.0 + obj_id * 0.3      # deterministisch, variabel
    R_world = BOP.axis_angle_matrix([0.0, 0.0, 1.0], yaw)
    # t_world (absolut): bbox-Mitte grob auf eine 0.2x0.2m Tray-Fläche gemappt,
    # z = Tisch-Höhe (so dass t_world.z nach -table_origin ~ 0 ist).
    fx, cx_k = K[0, 0], K[0, 2]
    fy, cy_k = K[1, 1], K[1, 2]
    bx = (bbox[0] + bbox[2]) / 2.0
    by = (bbox[1] + bbox[3]) / 2.0
    tx = (bx - cx_k) / fx * 0.2
    ty = -(by - cy_k) / fy * 0.2
    t_world = np.array([tx, ty, table_z_m])              # absolute Welt-Koordinaten
    # Welt -> BOP cam-Frame (Inverse der Adapter-Kette, absolute Welt -> cam):
    R_m2c = R_w2c @ R_world
    t_m2c = R_w2c @ (t_world * 1000.0) + t_w2c            # mm
    return R_m2c, t_m2c


def _gdrnpp_real(crop, K, bbox, obj_id, cfg):
    """>>> EINHÄNGE-PUNKT für Kais echten GDRNPP-Call. <<<

    Hier wird der trainierte GDRNPP-Checkpoint (cfg.checkpoint) auf den Crop
    angewendet. Erwartete Rückgabe (BOP-Konvention, EXAKT wie der Adapter sie
    will): (R_m2c (3x3), t_m2c (3,) in **mm**, Kamera-Frame).

    Vertrag des Calls:
      Eingabe: crop (HxWx3 RGB ndarray), K (3x3 Pinhole-Intrinsics des
               *ungecroppten* Bildes), bbox ([x0,y0,x1,y1] im Vollbild),
               obj_id (1-basiert, §1.2).
      Ausgabe: (R_m2c, t_m2c_mm).

    Kai hängt hier den GDRNPP-Inferenz-Aufruf ein (Modell-Load gecached, AMP,
    Crop->Netz->PnP/Direct-Regression). Bis dahin: NotImplementedError ->
    call_gdrnpp() fällt auf den Mock zurück, falls cfg.mock nicht gesetzt ist."""
    raise NotImplementedError(
        "GDRNPP-Checkpoint-Inferenz noch nicht angebunden (Kai, S-402). "
        "Checkpoint-Pfad: %s" % cfg.checkpoint)


def call_gdrnpp(crop, K, bbox, obj_id, cfg=None):
    """6D-Pose-Schätzung für EINE Detektion (BOP-Konvention).

    Saubere, gleichbleibende Schnittstelle für beide Modi:
      - cfg.mock=True (oder kein Checkpoint): plausibler MOCK.
      - sonst: Kais echter GDRNPP-Call (_gdrnpp_real).
    Returns: (R_m2c (3x3), t_m2c (3,) in mm).
    """
    cfg = cfg or GdrnppConfig()
    if cfg.mock:
        return _gdrnpp_mock(crop, K, bbox, obj_id)
    try:
        return _gdrnpp_real(crop, K, bbox, obj_id, cfg)
    except NotImplementedError:
        return _gdrnpp_mock(crop, K, bbox, obj_id)      # graceful bis Checkpoint da


def _make_rc_refiner(obj_id, mesh_m, crop, bbox, K, R_w2c, t_w2c, table_origin,
                     full_hw, target_mask=None, scorer=REFINE_RC_SCORER,
                     visib_mask=None, visib_fract=None,
                     min_visib_fract=REFINE_RC_MIN_VISIB_FRACT,
                     margin_schedule=REFINE_RC_MARGIN_SCHEDULE, warn=print):
    """Baut den per-Detektion-RC-Refiner-Callback (refine_rc.refine_detection).

    Der Callback bekommt (R_world, t_world) der Coarse-Pose und liefert die
    verfeinerte Welt-Rotation. None, wenn RC nicht verfügbar / kein Mesh. RGB-only:
    Kanten aus dem Crop, Silhouetten-Render aus dem CAD (mm).

    VISIBILITY-AWARE (ADR-020, S-003): wenn `visib_mask` (sichtbare Region, offline
    aus BOP `mask_visib/`) gegeben ist, scoret refine_rc NUR über die sichtbare
    Region und gated visibility-konditioniert (`visib_fract`/`min_visib_fract`).
    `visib_mask=None` (Live-Pfad) -> exakt heutiges Verhalten.
    """
    if RC is None or mesh_m is None:
        return None
    verts_mm = np.asarray(mesh_m, float) * 1000.0           # m -> mm (Renderer-Einheit)
    # Bildkanten des Crops, ins Vollbild an die bbox eingebettet (passt zum
    # Vollbild-Silhouetten-Render). Metall -> starke Kanten (ContourPose-Idee).
    image_edge_mask = RC.image_edges(crop, bbox=bbox, full_hw=full_hw)
    vis = None if visib_mask is None else np.asarray(visib_mask, bool)
    # visib_fract aus der Maske ableiten, falls nicht separat gegeben (sichtbare
    # Pixel / Gesamt-Silhouette via target_mask, sonst absolut über visible_px-Gate).
    vf = visib_fract
    if vis is not None and vf is None and target_mask is not None:
        tgt = np.asarray(target_mask, bool)
        denom = float(max(tgt.sum(), vis.sum(), 1))
        vf = float(vis.sum()) / denom
    # Ruhelagen-Body-Downs (für die Ruhelagen-Hypothesen) — optional, via trimesh.
    stable_downs = None
    try:
        stable_downs, _ = BOP.stable_pose_body_downs(
            mesh_verts_mm=verts_mm, prob_min=0.02, max_k=6, cache_key=obj_id)
    except Exception as exc:  # pragma: no cover
        warn(f"[rc] keine Ruhelagen für obj{obj_id} ({exc!r}) — ohne rest-Hyps")
    n_fold = OBJ_NFOLD.get(obj_id)
    mk = None
    if scorer == "megapose":
        mk = {"crop_rgb": crop, "K_full": K, "bbox": bbox,
              "obj_label": f"obj_{int(obj_id):06d}"}

    def _refine(R_world, t_world):
        R_ref, info = RC.refine_detection(
            R_world, verts_mm=verts_mm, t_world_m=t_world,
            table_origin_m=table_origin, R_w2c=R_w2c, t_w2c_mm=t_w2c, K=K,
            hw=full_hw, target_mask=target_mask, image_edge_mask=image_edge_mask,
            visib_mask=vis, visib_fract=vf, min_visib_fract=min_visib_fract,
            margin_schedule=margin_schedule,
            sym_axis=(0.0, 1.0, 0.0), n_fold=n_fold, stable_downs=stable_downs,
            scorer=scorer, megapose_kwargs=mk)
        va = " visaware" if info.get("visib_aware") else ""
        warn(f"[rc] obj{obj_id}: {info['n_hyps']} Hyps, scorer={info['scorer']}{va}, "
             f"best={info.get('best_tag')} switched={info.get('switched')}")
        return R_ref

    return _refine


def estimate_poses(rgb, dets, table_origin=None, cfg=None, warn=print,
                   planar_refine=PLANAR_REFINE_DEFAULT,
                   tilt_correct=PLANAR_TILT_CORRECT,
                   refine_rc=REFINE_RC_DEFAULT, rc_scorer=REFINE_RC_SCORER,
                   refine_rc_visaware=REFINE_RC_VISAWARE_DEFAULT,
                   rc_min_visib_fract=REFINE_RC_MIN_VISIB_FRACT,
                   rc_margin_schedule=REFINE_RC_MARGIN_SCHEDULE,
                   tta=TTA_DEFAULT, tta_n_rot=TTA_N_ROT, tta_hflip=TTA_HFLIP,
                   tta_agg=TTA_AGG):
    """6D-Pose pro Detektion: GDRNPP-Inferenz -> BOP-Adapter (Viktor §3) ->
    [M2 Render-and-Compare-Refiner, optional] -> planares Z-Snap (T-055).

    Kette pro Detektion:
      Crop+bbox -> call_gdrnpp -> (R_m2c, t_m2c) -> BOP.detection_to_result
      (M2-RC-Refine wenn refine_rc, dann Planar-Z-Snap aus R+CAD wenn planar_refine).
    Liefert die 'aligned'-Liste (Format wie build_pose_result erwartet:
      {instance_id, part, face_name, R_world(9), t_world(3), confidence,
       bbox_2d(4), upright}).
    """
    cfg = cfg or GdrnppConfig()
    if table_origin is None:
        table_origin = TABLE_ORIGIN_SCENE
    H, W = rgb.shape[:2]
    K, R_w2c, t_w2c = load_scene_camera_state(rgb, warn=warn)
    mode = "MOCK" if cfg.mock else "GDRNPP"
    n_snapped = 0
    n_rc = 0
    n_tta = 0
    aligned = []
    for d in dets:
        bbox = d["bbox_2d"]
        obj_id = BOP.PART_TO_OBJ_ID.get(d["part"])
        if obj_id is None:                               # unbekanntes Teil -> skip
            warn(f"[pose] '{d['part']}' nicht in obj_id-Map (§1.2) — übersprungen")
            continue
        crop = _crop(rgb, bbox)
        if tta and TTA is not None and tta_n_rot > 1:
            R_m2c, t_m2c, _ = TTA.tta_call_gdrnpp(
                call_gdrnpp, crop, K, bbox, obj_id, cfg=cfg,
                n_rot=tta_n_rot, use_hflip=tta_hflip, agg=tta_agg, warn=warn)
            n_tta += 1
        else:
            R_m2c, t_m2c = call_gdrnpp(crop, K, bbox, obj_id, cfg=cfg)
        # Mesh wird für Z-Snap UND den RC-Refiner gebraucht.
        need_mesh = planar_refine or refine_rc
        mesh_m = _load_mesh_verts_m(obj_id, warn=warn) if need_mesh else None
        rc_refiner = None
        if refine_rc:
            # visibility-aware NUR wenn Flag AN UND die Detektion eine sichtbare
            # Region trägt (offline aus BOP mask_visib/). Live-Pfad: None -> No-Op.
            visib_mask = d.get("visib_mask") if refine_rc_visaware else None
            visib_fract = d.get("visib_fract") if refine_rc_visaware else None
            # gestaffeltes Margin-Gate NUR im visaware-Pfad (sonst kein visib_fract).
            margin_schedule = rc_margin_schedule if refine_rc_visaware else None
            rc_refiner = _make_rc_refiner(
                obj_id, mesh_m, crop, bbox, K, R_w2c, t_w2c, table_origin,
                full_hw=(H, W), target_mask=d.get("mask"), scorer=rc_scorer,
                visib_mask=visib_mask, visib_fract=visib_fract,
                min_visib_fract=rc_min_visib_fract,
                margin_schedule=margin_schedule, warn=warn)
            if rc_refiner is not None:
                n_rc += 1
        r = BOP.detection_to_result(
            instance_id=d.get("instance_id", len(aligned)),
            obj_id=obj_id, R_m2c=R_m2c, t_m2c_mm=t_m2c,
            R_w2c=R_w2c, t_w2c_mm=t_w2c, table_origin_m=table_origin,
            bbox_2d=bbox, confidence=d.get("det_conf", 1.0),
            apply_planar=(planar_refine and mesh_m is not None),
            mesh_verts_m=mesh_m, table_z=PLANAR_TABLE_Z, tilt_correct=tilt_correct,
            rc_refiner=rc_refiner)
        if planar_refine and mesh_m is not None:
            n_snapped += 1
        # build_pose_result erwartet 'face_name'; detection_to_result liefert 'face'.
        r["face_name"] = r.pop("face")
        aligned.append(r)
    snap = (f", Planar-Z-Snap auf {n_snapped}/{len(aligned)}"
            if planar_refine else " (Planar-Refine AUS)")
    rc = f", RC-Refine ({rc_scorer}) auf {n_rc}/{len(aligned)}" if refine_rc else ""
    tt = (f", TTA ({tta_n_rot}-rot/{tta_agg}) auf {n_tta}/{len(aligned)}"
          if tta and TTA is not None and tta_n_rot > 1 else "")
    warn(f"[pose] {mode}: {len(aligned)} Pose(n) via BOP-Adapter (Viktor §3){snap}{rc}{tt}")
    return aligned


# Modul-State für die Kamera (gesetzt in run(), genutzt in estimate_poses).
_SCENE_CAM_STATE: dict = {}


def load_scene_camera_state(rgb, warn=print):
    """Holt die in run() gesetzte Kamera, sonst Default aus der RGB-Größe."""
    if "K" in _SCENE_CAM_STATE:
        s = _SCENE_CAM_STATE
        return s["K"], s["R_w2c"], s["t_w2c"]
    H, W = rgb.shape[:2]
    R_w2c, t_w2c = _default_extrinsics()
    return _default_K(W, H), R_w2c, t_w2c


# ── Stufe 4: pose_result + Contract-Gate ──────────────────────────────────────
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def check_pose_result(doc):
    """Pure-stdlib Contract-Gate -> Liste von Verstößen (leer = gültig)."""
    e = []
    if not isinstance(doc, dict):
        return ["top level must be object"]
    meta = doc.get("meta", {})
    if not isinstance(meta.get("source_image"), str) or not meta.get("source_image"):
        e.append("meta.source_image")
    to = meta.get("table_origin")
    if not (isinstance(to, list) and len(to) == 3 and all(_is_num(v) for v in to)):
        e.append("meta.table_origin")
    if meta.get("units") != "m":
        e.append("meta.units must be 'm'")
    if not isinstance(meta.get("coordinate_convention"), str) or not meta.get("coordinate_convention"):
        e.append("meta.coordinate_convention")
    sv = meta.get("schema_version")
    if not (isinstance(sv, str) and sv.count(".") == 2 and all(p.isdigit() for p in sv.split("."))):
        e.append("meta.schema_version")
    res = doc.get("results")
    if not isinstance(res, list):
        return e + ["results must be array"]
    for i, r in enumerate(res):
        p = f"results[{i}]"
        if not (_is_int(r.get("instance_id")) and r["instance_id"] >= 0):
            e.append(f"{p}.instance_id")
        if not (isinstance(r.get("part"), str) and r["part"]):
            e.append(f"{p}.part")
        if not (isinstance(r.get("face"), str) and r["face"]):
            e.append(f"{p}.face")
        R = r.get("R_world")
        if not (isinstance(R, list) and len(R) == 9 and all(_is_num(v) for v in R)):
            e.append(f"{p}.R_world")
        t = r.get("t_world")
        if not (isinstance(t, list) and len(t) == 3 and all(_is_num(v) for v in t)):
            e.append(f"{p}.t_world")
        c = r.get("confidence")
        if not (_is_num(c) and 0.0 <= c <= 1.0):
            e.append(f"{p}.confidence")
        b = r.get("bbox_2d")
        if not (isinstance(b, list) and len(b) == 4 and all(_is_int(v) and v >= 0 for v in b)):
            e.append(f"{p}.bbox_2d")
        elif not (b[0] <= b[2] and b[1] <= b[3]):
            e.append(f"{p}.bbox_2d order")
        if not isinstance(r.get("upright"), bool):
            e.append(f"{p}.upright")
    return e


def _check_with_jsonschema(doc):
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return []
    if not SCHEMA_FILE.exists():
        return []
    v = Draft202012Validator(json.load(open(SCHEMA_FILE)))
    return [f"[jsonschema] {'/'.join(str(x) for x in e.path)}: {e.message}"
            for e in v.iter_errors(doc)]


def jsonschema_available():
    try:
        import jsonschema  # noqa: F401
        return True
    except Exception:
        return False


def build_pose_result(img_path, aligned, table_origin=TABLE_ORIGIN_SCENE):
    return {
        "meta": {"source_image": str(img_path),
                 "table_origin": [float(v) for v in table_origin],
                 "units": "m", "coordinate_convention": COORD_CONVENTION,
                 "schema_version": SCHEMA_VERSION},
        "results": [{"instance_id": int(a["instance_id"]), "part": a["part"],
                     "face": a["face_name"],
                     "R_world": [float(v) for v in a["R_world"]],
                     "t_world": [float(v) for v in a["t_world"]],
                     "confidence": float(max(0.0, min(1.0, a["confidence"]))),
                     "bbox_2d": [int(v) for v in a["bbox_2d"]],
                     "upright": bool(a["upright"])} for a in aligned],
    }


# ── Orchestrierung ────────────────────────────────────────────────────────────
def run(image, out_path, cfg=None, warn=print,
        planar_refine=PLANAR_REFINE_DEFAULT, tilt_correct=PLANAR_TILT_CORRECT,
        refine_rc=REFINE_RC_DEFAULT, rc_scorer=REFINE_RC_SCORER,
        refine_rc_visaware=REFINE_RC_VISAWARE_DEFAULT,
        tta=TTA_DEFAULT, tta_n_rot=TTA_N_ROT, tta_hflip=TTA_HFLIP,
        tta_agg=TTA_AGG):
    """Ganze Pipeline für ein Bild. Schreibt schema-valides pose_result.json.

    Kette (ADR-018, Viktor §3/§4 + T-055 §3.5 + T-058 M2):
      Detektor (Stufe 1, OBB->AABB §4) -> pro Detektion Crop+bbox ->
      GDRNPP-Inferenz call_gdrnpp (Stufe 3; MOCK bis Kais Checkpoint da) ->
      (R_m2c, t_m2c) -> BOP-Adapter (§3) -> [M2 Render-and-Compare-Refiner, wenn
      refine_rc] -> Planar-Z-Snap auf Tischebene (§3.5, wenn planar_refine) ->
      pose_result (Stufe 4, schema-valide).
    """
    cfg = cfg or GdrnppConfig()
    if not pathlib.Path(image).exists():
        raise FileNotFoundError(
            f"Eingabebild nicht gefunden: {image}. "
            f"Pfad pruefen (z.B. project/input/<scene>.png).")
    rgb, dets = detections_for(image, warn=warn)
    # Kamera (BOP §1.3) einmal laden + für estimate_poses bereitstellen.
    H, W = rgb.shape[:2]
    K, R_w2c, t_w2c = load_scene_camera(image, W, H, warn=warn)
    _SCENE_CAM_STATE.update({"K": K, "R_w2c": R_w2c, "t_w2c": t_w2c})
    aligned = estimate_poses(rgb, dets, table_origin=TABLE_ORIGIN_SCENE,
                             cfg=cfg, warn=warn, planar_refine=planar_refine,
                             tilt_correct=tilt_correct, refine_rc=refine_rc,
                             rc_scorer=rc_scorer,
                             refine_rc_visaware=refine_rc_visaware,
                             tta=tta, tta_n_rot=tta_n_rot,
                             tta_hflip=tta_hflip, tta_agg=tta_agg)
    doc = build_pose_result(image, aligned, table_origin=TABLE_ORIGIN_SCENE)
    errors = check_pose_result(doc) + _check_with_jsonschema(doc)
    if errors:
        raise ValueError("pose_result NICHT contract-valid:\n  - " + "\n  - ".join(errors))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out_path, "w"), indent=2)
    gate = "stdlib + jsonschema" if jsonschema_available() else "stdlib"
    mode = "MOCK" if cfg.mock else "GDRNPP"
    warn(f"[e2e] {len(doc['results'])} Teile, Pose-Backend={mode}, "
         f"Schema-Gate PASS ({gate}) -> {out_path}")
    return doc


def serve_viewer(out_path, port=8000, open_browser=True):
    """localhost-Server ab project/ starten + den Viewer aufs pose_result zeigen.
    Serviert von project/, damit der Viewer (frontend/) und das Ergebnis (temp/)
    über relative Pfade erreichbar sind. Blockiert bis Ctrl-C."""
    import functools, http.server, socketserver, threading, time, webbrowser
    out_path = pathlib.Path(out_path).resolve()
    try:
        rel = out_path.relative_to(HERE)            # z.B. temp/pose_result.json
    except ValueError:
        rel = pathlib.Path("..") / out_path.name
    file_arg = "../" + str(rel).replace(os.sep, "/")   # relativ zu frontend/
    url = f"http://127.0.0.1:{port}/frontend/?file={file_arg}"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    print(f"\n[serve] Viewer: {url}\n[serve] Ctrl-C zum Beenden.")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] gestoppt.")
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="POSE E2E: ein Bild -> pose_result.json")
    ap.add_argument("--image", required=True, help="Eingabebild (project/input/<x>)")
    ap.add_argument("--out", default=None, help="Ausgabe (default: project/temp/pose_result.json)")
    ap.add_argument("--serve", action="store_true",
                    help="nach der Inferenz localhost-Server + 3D-Viewer öffnen")
    ap.add_argument("--port", type=int, default=8000, help="Server-Port für --serve")
    ap.add_argument("--checkpoint", default=None,
                    help="GDRNPP-Checkpoint (.pth). Fehlt -> MOCK-Modus.")
    ap.add_argument("--mock", action="store_true",
                    help="MOCK-Pose-Backend erzwingen (auch wenn Checkpoint da ist)")
    ap.add_argument("--no-planar-refine", action="store_true",
                    help="planares Tisch-Ebenen-Refinement (Z-Snap) AUS (default an)")
    ap.add_argument("--tilt-correct", action="store_true",
                    help="konservative Tilt-zur-Tischnormalen-Korrektur an (default aus)")
    ap.add_argument("--refine-rc", action="store_true",
                    help="M2 Multi-Hypothesen-Render-and-Compare-Refiner an (default aus)")
    ap.add_argument("--rc-scorer", default=REFINE_RC_SCORER,
                    choices=["cpu_edge", "megapose"],
                    help="RC-Scorer: cpu_edge (kein GPU, default) | megapose (GPU)")
    ap.add_argument("--refine-rc-visaware", action="store_true",
                    help="RC visibility-aware: scoren NUR über die sichtbare Region "
                         "(BOP mask_visib) + vis-konditioniertes Gate (Eval/Repro; "
                         "Live ohne mask_visib = No-Op). ADR-020/T-085. Default aus.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-Time-Augmentation (in-plane-Rotationen) an (default aus)")
    ap.add_argument("--tta-n-rot", type=int, default=TTA_N_ROT, choices=[1, 2, 4],
                    help="TTA in-plane-90°-Views: 1=aus, 2={0,180}, 4=alle (default 4)")
    ap.add_argument("--tta-hflip", action="store_true",
                    help="TTA H-Flip-View zusätzlich (ändert Chiralität, default aus)")
    ap.add_argument("--tta-agg", default=TTA_AGG,
                    choices=["medoid", "chordal", "score"],
                    help="TTA-Aggregation: medoid (bimodal-robust) | chordal | score")
    a = ap.parse_args(argv)
    cfg = GdrnppConfig(checkpoint=a.checkpoint, mock=(True if a.mock else None))
    out = a.out or str(HERE / "temp" / "pose_result.json")
    doc = run(a.image, out, cfg=cfg, planar_refine=not a.no_planar_refine,
              tilt_correct=a.tilt_correct, refine_rc=a.refine_rc,
              rc_scorer=a.rc_scorer, refine_rc_visaware=a.refine_rc_visaware,
              tta=a.tta, tta_n_rot=a.tta_n_rot,
              tta_hflip=a.tta_hflip, tta_agg=a.tta_agg)
    print(f"\n[e2e] {len(doc['results'])} Teile -> {out}")
    for r in doc["results"]:
        print(f"  #{r['instance_id']:>2} {r['part']:<22} {r['face']:<10} "
              f"conf={r['confidence']:.2f} t={[round(v, 3) for v in r['t_world']]} "
              f"upright={r['upright']}")
    if a.serve:
        serve_viewer(out, port=a.port)
    else:
        print(f"\nIm 3D-Viewer öffnen:\n"
              f"  python3 project/e2e_infer.py --image {a.image} --serve\n"
              f"  oder manuell: cd {HERE} && python3 -m http.server {a.port}\n"
              f"  dann http://127.0.0.1:{a.port}/frontend/?file=../temp/pose_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
