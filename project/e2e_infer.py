#!/usr/bin/env python3
"""POSE — standalone end-to-end inference: ein Bild -> pose_result.json.

    python project/e2e_infer.py --image project/input/scene_0000.png
    python project/e2e_infer.py --image project/input/scene_0000.png --out out.json
    python project/e2e_infer.py --image project/input/scene_0000.png --serve

Selbst-enthalten: die gesamte Pipeline-Logik ist INLINE (kein Import auf
Projektmodule). Bewusst dupliziert mit setup.ipynb / infer.ipynb — damit dieses
eine Skript für sich allein lauffähig und weitergebbar ist.

Pipeline:
  1. Teile finden        — echter OBB-Detektor (models/detector.pt, YOLOv8-OBB via
                           ultralytics) liefert orientierte Boxen + Klasse. Fallback:
                           SDG-Annotator-Boxen (bbox_2d_<idx>.json), dann Dummy-Box.
  2. Crop                — achsenparalleler Ausschnitt der OBB-Hülle pro Detection.
  3. Face klassifizieren — models/<part>.pt (CNN, lazy torch), sonst
                           Nearest-Template-Fallback gegen die Registry-Templates.
  4. 6D-Alignment        — Yaw aus dem OBB-Winkel (180°-Flip per Template-MSE
                           aufgelöst; ohne OBB Vollsuche), R_world = Rz(yaw)·R_face,
                           Backprojection des BBox-Zentrums auf die Tisch-Ebene.
  5. pose_result         — gegen den Contract validiert (stdlib immer; jsonschema
                           wenn verfügbar), dann geschrieben. Schema-valide oder
                           es wird nichts geschrieben.

Mit --serve startet das Skript danach einen localhost-Server (ab project/) und
öffnet den 3D-Viewer (frontend/) auf das erzeugte pose_result.

Konvention (eingefroren): Z-up Welt, world = R @ body (Spaltenkonvention),
Ursprung = Tisch-Nullpunkt, Einheit Meter.

Abhängigkeiten: numpy, scipy, PIL (Pflicht). torch + ultralytics nur für die
trainierten Checkpoints (sonst Fallback). jsonschema optional (Bonus-Gate).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage

# ── Pfade (relativ zu diesem Skript, CWD-unabhängig) ──────────────────────────
HERE = pathlib.Path(__file__).resolve().parent          # project/
MODELS = HERE / "models"
REGISTRY = MODELS / "registry"
# pose_result-Contract (in project/ committet; ältere docs/-Lage als Fallback).
SCHEMA_FILE = next((p for p in (HERE / "pose_result.schema.json",
                                HERE.parent / "docs" / "pose_result.schema.json")
                    if p.exists()), HERE / "pose_result.schema.json")

SCHEMA_VERSION = "1.0.0"
COORD_CONVENTION = ("Z-up world; column rotation world = R @ body (same as "
                    "faces_<part>.json registry); origin = table-plane null-point")
IMG_SIZE = TMPL_SIZE = 96
YAW_STEP_DEG = 5.0
UPRIGHT_TILT_DEG = 60.0

# top-down Default-Intrinsics (entspricht render_dataset.py)
DEFAULT_CAM_H, DEFAULT_FOCAL_MM, DEFAULT_SENSOR_MM = 0.16, 24.0, 20.955

# Szenen-Backprojection (Multi-Part-Zellen-View, GST_Scene + Zivid/_DRCam):
# Die Kamera schaut leicht schräg von ~0.9 m auf das Tray. Statt einer exakten
# (nicht verfügbaren) metrischen Inversen mappen wir die sichtbare Tray-Fläche auf
# die kalibrierten realen Tray-Grenzen — Teile streuen so realistisch über den
# Tisch (für den Viewer) statt in einem Pixel-Klumpen zu kollabieren. Die Werte
# stammen aus den SPAWN_BOUNDS der SDG-Pipeline (datagenerationscript.py).
SCENE_TRAY_BOUNDS = {"x": (0.057, 0.783), "y": (0.042, 0.537)}  # Meter, Welt
# Pixel-Region des Trays im 1280x720-Render (grob kalibriert am Zivid/_DRCam-Frame).
SCENE_PIX_REGION = {"u": (250, 1080), "v": (40, 560)}


# ── Registry-Loader ───────────────────────────────────────────────────────────
class Face:
    def __init__(self, name, prob, tilt_deg, R_face, template_path, index):
        self.name = name; self.prob = prob; self.tilt_deg = tilt_deg
        self.R_face = R_face; self.template_path = template_path; self.index = index


class PartRegistry:
    def __init__(self, part, convention, faces):
        self.part = part; self.convention = convention; self.faces = faces

    def resolve_face(self, classifier_face):
        if not self.faces:
            return None
        for f in self.faces:
            if f.name == classifier_face:
                return f
        norm = classifier_face.strip().lower().replace("face", "").replace("_", "").replace(" ", "")
        if norm.isdigit():
            for f in self.faces:
                if f.index == int(norm):
                    return f
        return max(self.faces, key=lambda f: f.prob)


def load_part_registry(part, root=REGISTRY):
    pdir = pathlib.Path(root) / part
    jp = pdir / f"faces_{part}.json"
    if not jp.exists():
        return None
    data = json.load(open(jp))
    faces = []
    for k, fc in enumerate(data.get("faces", []), start=1):
        R = np.asarray(fc.get("R", []), float)
        R = R.reshape(3, 3) if R.size == 9 else np.eye(3)
        tp = pdir / f"tmpl_Face{k}.png"
        faces.append(Face(str(fc.get("name", f"Face {k}")), float(fc.get("prob", 0.0)),
                          float(fc.get("tilt_deg", 0.0)), R, str(tp) if tp.exists() else None, k))
    return PartRegistry(data.get("part", part),
                        data.get("convention", "world = R @ body (column)"), faces)


def load_template(face, size=TMPL_SIZE):
    if not face or not face.template_path or not os.path.exists(face.template_path):
        return None
    img = np.asarray(Image.open(face.template_path).convert("L").resize((size, size)))
    return img.astype(np.float32) / 255.0


def available_parts(root=REGISTRY):
    root = pathlib.Path(root)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / f"faces_{d.name}.json").exists())


# ── Stufe 1: Detections (OBB-Detektor, sonst SDG-BBoxes, sonst Dummy) ─────────
# Drei Quellen, in dieser Priorität, jede ein sauberer Fallback der vorigen:
#   1. models/detector.pt  — echt trainierter YOLOv8-OBB-Detektor (ultralytics).
#      Findet Teile in beliebigen Szenen als orientierte Boxen + Klasse.
#   2. bbox_2d_<idx>.json   — SDG-Annotator-Boxen neben dem Bild (Sim-Ground-Truth).
#   3. Dummy                — eine ganze-Bild-Box, damit die Kette nie hart bricht.
DETECTOR_FILE = MODELS / "detector.pt"
DETECTOR_CONF = 0.25
DETECTOR_IMGSZ = 960
_DETECTOR_CACHE: dict = {}


def _canonical_parts():
    parts = available_parts()
    if parts:
        return {p.lower(): p for p in parts}
    return {n.lower(): n for n in ["Anker_Lang", "Anker_Kurz", "Zahnrad",
            "Poltopf_kurz_centered", "Getriebegehaeuse_typ4", "Buerstenhalter_2polig"]}


def canonical_part(raw, canon):
    return canon.get(raw.strip().lower(), raw)


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


def _obb_to_aabb(corners, W, H):
    """Orientierte 4-Eck-Box -> achsenparallele BBox [x0,y0,x1,y1] (geklemmt)."""
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    x0 = max(0, min(int(round(min(xs))), W - 1)); x1 = max(0, min(int(round(max(xs))), W))
    y0 = max(0, min(int(round(min(ys))), H - 1)); y1 = max(0, min(int(round(max(ys))), H))
    return [x0, y0, x1, y1]


def _obb_angle_deg(corners):
    """In-plane Winkel der langen OBB-Achse (Grad), zum Seeden der Yaw-Suche."""
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


def find_bbox_json(img_path):
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
        warn(f"[detect] DUMMY: eine ganze-Bild-BBox als '{part}' (OBB-Detektor TODO)")
    return rgb, dets


# ── Stufe 3: Face-Classifier (Checkpoint, sonst Nearest-Template) ─────────────
_MODEL_CACHE: dict = {}
_REF_CACHE: dict = {}


def _prep(crop, size=IMG_SIZE):
    a = np.asarray(crop)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    sd = max(h, w)
    sq = np.full((sd, sd, 3), 128, np.uint8)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = a[:, :, :3]
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.BILINEAR))


def _build_net(n_classes):
    import torch.nn as nn

    def block(c, o):
        return nn.Sequential(nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.MaxPool2d(2))

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.f = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
            self.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Dropout(0.2), nn.Linear(64, n))

        def forward(self, x):
            return self.h(self.f(x))

    return Net(n_classes)


def _load_model(part):
    if part in _MODEL_CACHE:
        return _MODEL_CACHE[part]
    ckpt = MODELS / f"{part}.pt"
    if not ckpt.exists():
        _MODEL_CACHE[part] = None
        return None
    try:
        import torch
        state = torch.load(ckpt, map_location="cpu")
        classes = state["classes"]
        net = _build_net(len(classes))
        net.load_state_dict(state["model"])
        net.eval()
        _MODEL_CACHE[part] = (net, classes)
    except Exception:
        _MODEL_CACHE[part] = None
    return _MODEL_CACHE[part]


def _infer_model(crop, part):
    loaded = _load_model(part)
    if loaded is None:
        return None
    net, classes = loaded
    import torch
    x = _prep(crop).astype(np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        probs = torch.softmax(net(x), 1)[0].numpy()
    k = int(np.argmax(probs))
    return {"face": classes[k], "confidence": float(probs[k])}


def _grey(img):
    a = np.asarray(img).astype(np.float32)
    return a.mean(2) / 255.0 if a.ndim == 3 else a / 255.0


def _ncc_rot_mirror(a, b, step=10):
    best = -1.0
    for src in (a, a[:, ::-1]):
        for ang in range(0, 360, step):
            ar = ndimage.rotate(src, ang, reshape=False, order=1)
            m = (ar > 0.02) | (b > 0.02)
            if m.sum() < 20:
                continue
            x = ar[m] - ar[m].mean(); y = b[m] - b[m].mean()
            d = np.linalg.norm(x) * np.linalg.norm(y) + 1e-9
            best = max(best, float((x @ y) / d))
    return best


def _load_refs(part):
    if part in _REF_CACHE:
        return _REF_CACHE[part]
    reg = load_part_registry(part)
    refs = {}
    if reg:
        for f in reg.faces:
            if f.template_path:
                refs[f"face_{f.index}"] = _grey(_prep(np.asarray(
                    Image.open(f.template_path).convert("RGB"))))
    _REF_CACHE[part] = refs
    return refs


def _infer_fallback(crop, part):
    refs = _load_refs(part)
    reg = load_part_registry(part)
    classes = [f"face_{f.index}" for f in reg.faces] if reg else []
    if not refs:
        return {"face": classes[0] if classes else "face_1", "confidence": 0.0}
    q = _grey(_prep(crop))
    scores = {c: _ncc_rot_mirror(q, r) for c, r in refs.items()}
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_cls, best = items[0]
    conf = (best - items[1][1]) / (abs(best) + 1e-6) if len(items) > 1 else best
    return {"face": best_cls, "confidence": float(np.clip(conf, 0.0, 1.0))}


def infer(crop, part):
    """Face-Klassifikation. Checkpoint wenn vorhanden, sonst Nearest-Template."""
    out = _infer_model(crop, part)
    return out if out is not None else _infer_fallback(crop, part)


def backend(part):
    return "checkpoint" if _load_model(part) is not None else "fallback"


# ── Stufe 4: 6D-Alignment + Backprojection ────────────────────────────────────
def _rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _silhouette(crop, size=TMPL_SIZE):
    a = np.asarray(crop)
    g = a.astype(np.float32).mean(2) if a.ndim == 3 else a.astype(np.float32)
    h, w = g.shape[:2]
    sd = max(h, w)
    sq = np.full((sd, sd), 128.0, np.float32)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = g
    img = np.asarray(Image.fromarray(sq.astype(np.uint8)).resize((size, size),
                     Image.BILINEAR)).astype(np.float32)
    sig = np.abs(img - 128.0)
    m = sig.max()
    return sig / m if m > 1e-6 else sig


def estimate_yaw(crop, template, step_deg=YAW_STEP_DEG):
    if template is None:
        return 0.0
    q = _silhouette(crop)
    best_ang, best_mse = 0.0, float("inf")
    for ang in np.arange(0.0, 360.0, step_deg):
        rt = ndimage.rotate(template, ang, reshape=False, order=1, mode="constant", cval=0.0)
        m = float(np.mean((rt - q) ** 2))
        if m < best_mse:
            best_mse, best_ang = m, float(ang)
    return best_ang


def refine_yaw_with_obb(crop, template, obb_angle_deg, step_deg=YAW_STEP_DEG):
    """Yaw aus dem OBB-Winkel ableiten + die 180°-Ambiguität per Template-MSE
    auflösen. Der OBB-Detektor liefert die Lang-Achse bis auf Flip — das Template
    entscheidet, welches Ende vorn ist. Robuster und schneller als die Vollsuche."""
    if template is None:
        return float(obb_angle_deg % 360.0)
    q = _silhouette(crop)
    best_ang, best_mse = float(obb_angle_deg % 360.0), float("inf")
    # Feinraster um den OBB-Winkel + den 180°-Flip
    for base in (obb_angle_deg, obb_angle_deg + 180.0):
        for d in np.arange(-step_deg, step_deg + 1e-6, step_deg / 2.0):
            ang = float((base + d) % 360.0)
            rt = ndimage.rotate(template, ang, reshape=False, order=1, mode="constant", cval=0.0)
            m = float(np.mean((rt - q) ** 2))
            if m < best_mse:
                best_mse, best_ang = m, ang
    return best_ang


def align_detection(part, classifier_face, crop, registry, step_deg=YAW_STEP_DEG,
                    obb_angle_deg=None):
    face = registry.resolve_face(classifier_face) if registry else None
    if face is None:
        return {"R_world": list(np.eye(3).flatten()), "yaw_deg": 0.0, "upright": False,
                "face_name": classifier_face or "Face 1", "rest_height": 0.0}
    template = load_template(face, size=TMPL_SIZE)
    if obb_angle_deg is not None:
        yaw = refine_yaw_with_obb(crop, template, obb_angle_deg, step_deg)   # Detektor-Prior
    else:
        yaw = estimate_yaw(crop, template, step_deg)                          # Vollsuche
    R_world = _rz(np.radians(yaw)) @ face.R_face
    return {"R_world": [float(v) for v in R_world.flatten()], "yaw_deg": yaw,
            "upright": bool(face.tilt_deg >= UPRIGHT_TILT_DEG),
            "face_name": face.name, "rest_height": 0.0}


class Intrinsics:
    def __init__(self, width, height, cam_h=DEFAULT_CAM_H, focal_mm=DEFAULT_FOCAL_MM,
                 sensor_mm=DEFAULT_SENSOR_MM, table_origin=(0.0, 0.0, 0.0)):
        self.width = width; self.height = height; self.cam_h = cam_h
        self.table_origin = table_origin
        self.cx = width / 2.0; self.cy = height / 2.0
        self.fx = focal_mm / sensor_mm * width
        self.fy = focal_mm / sensor_mm * height


class SceneIntrinsics:
    """Backprojection für die Multi-Part-Zellen-View: linear vom Tray-Pixel-Bereich
    auf die realen Tray-Grenzen, zentriert auf den Tisch-Nullpunkt. Nicht exakt
    metrisch (Kamera schräg, keine echten Intrinsics), aber gibt eine realistische
    Streuung über den Tisch — genau was der Viewer für die Rekonstruktion braucht."""

    def __init__(self, width, height, table_origin=(0.0, 0.0, 0.0),
                 tray=SCENE_TRAY_BOUNDS, pix=SCENE_PIX_REGION):
        self.width = width; self.height = height; self.table_origin = table_origin
        # Tray-Mittelpunkt in Welt -> auf den Nullpunkt legen, damit Teile um den
        # Tisch-Ursprung zentriert erscheinen.
        self.wx0, self.wx1 = tray["x"]; self.wy0, self.wy1 = tray["y"]
        self.u0, self.u1 = pix["u"]; self.v0, self.v1 = pix["v"]
        self.wcx = (self.wx0 + self.wx1) / 2.0; self.wcy = (self.wy0 + self.wy1) / 2.0

    def to_world(self, bbox, rest_height=0.0):
        x0, y0, x1, y1 = bbox
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        fu = (u - self.u0) / max(1e-6, (self.u1 - self.u0))   # 0..1 über Tray-Breite
        fv = (v - self.v0) / max(1e-6, (self.v1 - self.v0))   # 0..1 über Tray-Höhe
        wx = self.wx0 + fu * (self.wx1 - self.wx0)
        wy = self.wy0 + fv * (self.wy1 - self.wy0)
        # auf Tray-Mitte zentrieren + Tisch-Ursprung
        x = (wx - self.wcx) + self.table_origin[0]
        y = -(wy - self.wcy) + self.table_origin[1]            # Bild-v wächst nach unten
        return [float(x), float(y), float(self.table_origin[2] + rest_height)]


def bbox_center_to_world(bbox, intr, rest_height=0.0):
    if isinstance(intr, SceneIntrinsics):
        return intr.to_world(bbox, rest_height)
    x0, y0, x1, y1 = bbox
    u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x = (u - intr.cx) * intr.cam_h / intr.fx + intr.table_origin[0]
    y = -(v - intr.cy) * intr.cam_h / intr.fy + intr.table_origin[1]
    return [float(x), float(y), float(intr.table_origin[2] + rest_height)]


# ── Stufe 5: pose_result + Contract-Gate ──────────────────────────────────────
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


def build_pose_result(img_path, aligned, intr):
    return {
        "meta": {"source_image": str(img_path),
                 "table_origin": [float(v) for v in intr.table_origin],
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
def run(image, out_path, warn=print):
    """Ganze Pipeline für ein Bild. Schreibt schema-valides pose_result.json."""
    rgb, dets = detections_for(image, warn=warn)
    H, W = rgb.shape[:2]
    # Zellen-View (Multi-Part-Szene) -> Tray-kalibrierte Backprojection; sonst die
    # top-down Faceset-Intrinsics. Heuristik: breite Szene (>=1000px) = Zellen-View.
    intr = SceneIntrinsics(W, H) if W >= 1000 else Intrinsics(W, H)
    known = set(available_parts())
    reg_cache: dict = {}
    aligned = []
    for d in dets:
        if known and d["part"] not in known:
            warn(f"[infer] unbekanntes Teil '{d['raw_label']}' (canonical "
                 f"'{d['part']}') — kein Modell/Registry, skip #{d['instance_id']}")
            continue
        x0, y0, x1, y1 = d["bbox_2d"]
        crop = rgb[y0:y1, x0:x1].copy()
        res = infer(crop, d["part"])
        if d["part"] not in reg_cache:
            reg_cache[d["part"]] = load_part_registry(d["part"])
        a = align_detection(d["part"], res["face"], crop, reg_cache[d["part"]],
                            obb_angle_deg=d.get("obb_angle_deg"))
        t_world = bbox_center_to_world(d["bbox_2d"], intr, rest_height=a["rest_height"])
        aligned.append({"instance_id": d["instance_id"], "part": d["part"],
                        "confidence": res["confidence"], "bbox_2d": d["bbox_2d"],
                        "t_world": t_world, **a})
        warn(f"[infer] #{d['instance_id']:>2} {d['part']:<22} {a['face_name']:<8} "
             f"conf={res['confidence']:.2f} ({backend(d['part'])}) "
             f"yaw={a['yaw_deg']:5.1f}° upright={a['upright']}")
    doc = build_pose_result(image, aligned, intr)
    errors = check_pose_result(doc) + _check_with_jsonschema(doc)
    if errors:
        raise ValueError("pose_result NICHT contract-valid:\n  - " + "\n  - ".join(errors))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out_path, "w"), indent=2)
    gate = "stdlib + jsonschema" if jsonschema_available() else "stdlib"
    warn(f"[e2e] {len(doc['results'])} Teile, Schema-Gate PASS ({gate}) -> {out_path}")
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
    a = ap.parse_args(argv)
    out = a.out or str(HERE / "temp" / "pose_result.json")
    doc = run(a.image, out)
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
