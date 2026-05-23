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
# Template-Banken (bank.npz pro Teil). Über POSE_TEMPLATES_DIR überschreibbar
# (A/B-Test Isaac- vs trimesh-Banken, ohne die produktive Lage anzufassen).
TEMPLATES = pathlib.Path(os.environ.get("POSE_TEMPLATES_DIR", str(MODELS / "templates")))
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

# ── ECHTE Zivid-Kamera der GST_Scene (aus GST_Scene.usd ausgelesen) ───────────
# Die Multi-Part-Zellen-Renders kommen von dieser Kamera. Mit echten Intrinsics +
# Extrinsics wird die Backprojection metrisch korrekt: jeder Pixel-Strahl wird auf
# die Tisch-Ebene (z=0, Welt) geschnitten -> Teile landen an der richtigen Stelle
# statt zentrumsnah geclustert.
#   focalLength=15.5mm, horiz/vertAperture=15.0mm, Render 1280x720
#   Kamera-Welt-Pose: t=(0.45, 0.08807, 1.1), leicht nach -Y/-Z gekippt.
ZIVID = {
    "focal_mm": 15.5, "aperture_h_mm": 15.0, "aperture_v_mm": 15.0,
    "render_w": 1280, "render_h": 720,
    "cam_t": (0.45, 0.08807, 1.1),
    # Welt-aus-Kamera Rotationsmatrix (Zeilen), aus ComputeLocalToWorldTransform.
    "R_wc": ((-0.99939, -0.00666, 0.03426),
             (0.0,      -0.98163, -0.19081),
             (0.0349,   -0.19069, 0.98103)),
}
# P0 TOP-DOWN-Kamera der Inferenz-Renders (gen_*_topdown.py): senkrecht über der
# Tray-Mitte, Blick gerade nach -Z, +Y oben. Die Top-Down-Eingabe wird mit DIESEN
# Intrinsics backprojiziert (statt der gekippten Zivid-Cam) -> metrisch korrekt.
ZIVID_TOPDOWN = {
    "focal_mm": 18.0, "aperture_h_mm": 20.955, "aperture_v_mm": 20.955 * 720 / 1280,
    "render_w": 1280, "render_h": 720,
    "cam_t": (0.46, 0.27, 0.95),
    "R_wc": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}
# Tray-/Spawn-Grenzen (datagenerationscript.py SPAWN_BOUNDS) — als Plausi-Klemme
# falls ein Strahl knapp daneben trifft.
SCENE_TRAY_BOUNDS = {"x": (0.057, 0.783), "y": (0.042, 0.537)}  # Meter, Welt


# ── Registry-Loader ───────────────────────────────────────────────────────────
class Face:
    def __init__(self, name, prob, tilt_deg, R_face, template_path, index, rest_height=0.0):
        self.name = name; self.prob = prob; self.tilt_deg = tilt_deg
        self.R_face = R_face; self.template_path = template_path; self.index = index
        # rest_height = halbe z-Ausdehnung der Mesh in DIESER Ruhelage (m): wie hoch
        # der geometrische Mittelpunkt (= t_world-Bezugspunkt) über der Tischfläche
        # liegt, wenn das Teil auf diesem Face liegt. Stehende Faces -> größer.
        self.rest_height = float(rest_height)


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
                          float(fc.get("tilt_deg", 0.0)), R, str(tp) if tp.exists() else None, k,
                          rest_height=float(fc.get("rest_height", 0.0))))
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


# ── Part-Meta: prim-origin -> Schwerpunkt-Offset + Mesh-Ausdehnung (Meter) ─────
# Aus export_part_glbs.py. KRITISCH für korrekte Translation: die Isaac-GT-t ist
# der PRIM-URSPRUNG des Teils; viele USD-Prims haben ihren Ursprung WEIT weg vom
# geometrischen Schwerpunkt (Getriebe: +243mm in Y, -191mm in Z!). Die Pipeline
# platziert Teile am Schwerpunkt (Backprojection des BBox-Zentrums = Bildmitte des
# Teils ≈ Schwerpunkt-Projektion). Für einen fairen, metrisch korrekten Vergleich
# (und für die Mesh-Platzierung im Viewer) muss der Offset rotiert addiert werden:
#   t_centroid = t_prim + R_world @ centroid_offset_body
_PART_META_CACHE: dict = {}


def load_part_meta():
    """centroid_offset_m + extent_m pro Teil (gecached). {} wenn keine Datei."""
    if _PART_META_CACHE:
        return _PART_META_CACHE
    for cand in (MODELS / "part_meta.json", HERE / "frontend" / "assets" / "part_meta.json"):
        if cand.exists():
            try:
                _PART_META_CACHE.update(json.load(open(cand)))
                break
            except Exception:
                pass
    return _PART_META_CACHE


def centroid_offset(part):
    """Body-frame Vektor prim-origin -> Schwerpunkt (Meter). 0 wenn unbekannt."""
    m = load_part_meta().get(part)
    if m and "centroid_offset_m" in m:
        return np.asarray(m["centroid_offset_m"], float)
    return np.zeros(3, float)


# ── Stufe 1: Detections (OBB-Detektor, sonst SDG-BBoxes, sonst Dummy) ─────────
# Drei Quellen, in dieser Priorität, jede ein sauberer Fallback der vorigen:
#   1. models/detector.pt  — echt trainierter YOLOv8-OBB-Detektor (ultralytics).
#      Findet Teile in beliebigen Szenen als orientierte Boxen + Klasse.
#   2. bbox_2d_<idx>.json   — SDG-Annotator-Boxen neben dem Bild (Sim-Ground-Truth).
#   3. Dummy                — eine ganze-Bild-Box, damit die Kette nie hart bricht.
DETECTOR_FILE = MODELS / "detector.pt"
# Der neue YOLOv8s-OBB (5 Klassen, mAP50≈0.99) ist deutlich konfidenter — höhere
# conf filtert die Doppel-/Geister-Boxen, die der alte 2-Klassen-Detektor lieferte.
DETECTOR_CONF = 0.40
DETECTOR_IMGSZ = 1280
# Rand-Detektionen: am Bildrand abgeschnittene Teile haben einen falschen 2D-
# Schwerpunkt -> der Backprojection-Strahl trifft die Tisch-Ebene weit daneben
# (typisch geklemmt auf x≈0.89). NUR echte Geister/stark-abgeschnittene Boxen
# verwerfen: Box-ZENTRUM praktisch außerhalb des Sichtfelds (>96% Rand). Teile, die
# den Rand nur berühren aber mit echtem Schwerpunkt im Bild liegen, bleiben — ihre
# x/y kommen aus dem TIEFEN-Schwerpunkt der sichtbaren Pixel (robust gegen den Rand).
EDGE_MARGIN_PX = 3
EDGE_CENTER_FRAC = 0.06
_DETECTOR_CACHE: dict = {}


def _is_edge_detection(bbox, W, H, margin=EDGE_MARGIN_PX, cfrac=EDGE_CENTER_FRAC):
    """Geister-/Rand-Box verwerfen: Box BERÜHRT den Bildrand UND ihr Zentrum liegt im
    äußeren cfrac-Band. Solche Boxen sind entweder am Rand abgeschnittene Teile (mit
    falschem 2D-Schwerpunkt -> Strahl trifft daneben) oder wiederkehrende Annotator-
    Artefakte (z.B. der konstante Rand-Geist bei (1206,670)). Mittig liegende Teile
    bleiben immer erhalten (deren Zentrum ist nicht im Außenband)."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    touches = (x0 <= margin or y0 <= margin or x1 >= W - margin or y1 >= H - margin)
    if not touches:
        return False
    return (cx < cfrac * W or cx > (1 - cfrac) * W or
            cy < cfrac * H or cy > (1 - cfrac) * H)


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
    n_edge = 0
    for inst, (poly, c, cf) in enumerate(zip(polys, cls, conf)):
        corners = [[float(x), float(y)] for x, y in poly]
        bbox = _obb_to_aabb(corners, W, H)
        if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
            continue
        if _is_edge_detection(bbox, W, H):
            n_edge += 1
            continue
        raw = names[int(c)]
        dets.append({"instance_id": len(dets), "part": canonical_part(raw, canon),
                     "bbox_2d": bbox, "raw_label": raw, "occlusion": 0.0,
                     "obb_corners": corners, "obb_angle_deg": _obb_angle_deg(corners),
                     "det_conf": float(cf)})
    warn(f"[detect] {len(dets)} orientierte Boxen vom OBB-Detektor (detector.pt)"
         + (f" — {n_edge} Rand-Boxen verworfen" if n_edge else ""))
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


# ── Template-Bank Render-and-Compare (DIE Pose-Quelle) ────────────────────────
# Statt aus einem CNN zu *raten*, wird der Crop deterministisch gegen eine pro
# Teil aus dem ECHTEN CAD gerenderte Bank gematcht: {stabile Ruhelagen} x {Yaw
# 0..360 in 5°}. Jedes Template = top-down Tiefen-/Silhouetten-Bild der Mesh in
# Lage Rz(yaw)·R_face. Bester Match -> exakte Ruhelage (Face) + exakter Yaw ->
# volle R_world direkt aus der Bank. Planar-eingeschränktes Render-and-Compare
# (CosyPose/MegaPose-Idee). Die OBB liefert (x,y) + groben Yaw-Seed (Suchfenster).
_BANK_CACHE: dict = {}

# ── LERNBASIERTES Pose-Refinement (GigaPose/MegaPose-Stil) ────────────────────
# Ein gelerntes Embedding (models/refiner_<part>.pt) bettet den Query-Tiefen-Crop UND
# die Bank-Tiefen-Templates in denselben Raum, so dass das embedding-nächste Template
# das pose-nächste ist — es schliesst den Domain-Gap, an dem die Handmetrik (NCC/IoU/
# Gradient) scheitert (Oracle ~3°, Handmetrik real ~50-95°). Trainiert auf voll-
# synthetischen Top-Down-Tiefen-Crops mit GT-Rotation + Domain-Randomization, sym-
# aware (unbeobachtbare DoF werden nicht als Negative bestraft). Drop-in: ist KEIN
# Checkpoint da, läuft die Handmetrik wie bisher (sauberer Fallback).
_REFINER_CACHE: dict = {}


def _build_refiner_enc(emb):
    import torch.nn as nn
    import torch.nn.functional as Fnn

    def conv(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                             nn.ReLU(True), nn.Conv2d(co, co, 3, padding=1),
                             nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))

    class Enc(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.f = nn.Sequential(conv(1, 32), conv(32, 64), conv(64, 96), conv(96, 128))
            self.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, e))

        def forward(self, x):
            return Fnn.normalize(self.h(self.f(x)), dim=1)

    return Enc(emb)


def load_refiner(part):
    """refiner_<part>.pt laden (gecached): {net, bank_emb (T,emb), size}. None ohne
    Checkpoint / ohne torch."""
    if part in _REFINER_CACHE:
        return _REFINER_CACHE[part]
    ckpt = MODELS / f"refiner_{part}.pt"
    out = None
    if ckpt.exists():
        try:
            import torch
            st = torch.load(ckpt, map_location="cpu")
            net = _build_refiner_enc(int(st["emb"]))
            net.load_state_dict(st["state"]); net.eval()
            out = {"net": net, "bank_emb": np.asarray(st["bank_emb"], np.float32),
                   "size": int(st["size"])}
        except Exception:
            out = None
    _REFINER_CACHE[part] = out
    return out


def _refiner_scores(part, query_depth):
    """Lernbasierte Re-Rank-Scores (T,) in [-1,1] (Cosinus Query-Embedding vs Bank-
    Embeddings). None ohne Refiner. query_depth = normiertes Tiefen-Relief (SxS)."""
    rf = load_refiner(part)
    if rf is None or query_depth is None:
        return None
    try:
        import torch
        size = rf["size"]
        q = np.asarray(Image.fromarray((np.clip(query_depth, 0, 1) * 255).astype(np.uint8))
                       .resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
        with torch.no_grad():
            zq = rf["net"](torch.from_numpy(q)[None, None]).numpy()[0]   # (emb,)
        return rf["bank_emb"] @ zq                                       # (T,)
    except Exception:
        return None


def refiner_backend(part):
    return "learned" if load_refiner(part) is not None else "handcrafted"


def load_template_bank(part):
    """bank.npz für ein Teil laden (gecached). None wenn keine Bank da."""
    if part in _BANK_CACHE:
        return _BANK_CACHE[part]
    bp = TEMPLATES / part / "bank.npz"
    bank = None
    if bp.exists():
        try:
            d = np.load(bp)
            n = len(d["face_idx"])
            # rest_h pro Template (m): halbe z-Ausdehnung in der Ruhelage. Ältere
            # Banken ohne rest_h -> 0.0 (Teil liegt auf der Tischebene, alt-Verhalten).
            rest_h = (d["rest_h"].astype(np.float32) if "rest_h" in d.files
                      else np.zeros(n, np.float32))
            prob = (d["prob"].astype(np.float32) if "prob" in d.files
                    else np.ones(n, np.float32))
            # centroid_depth (m) = metrische Höhe des Schwerpunkts über der Tisch-
            # ebene in dieser Lage. Für die Isaac-Banken exakt aus der Geometrie; ältere
            # Banken haben es nicht -> auf rest_h zurückfallen (Schwerpunkt ≈ Mitte).
            cdepth = (d["centroid_depth"].astype(np.float32)
                      if "centroid_depth" in d.files else rest_h)
            bank = {"depth": d["depth"].astype(np.float32), "sil": d["sil"].astype(np.float32),
                    "face_idx": d["face_idx"], "yaw": d["yaw"].astype(np.float32),
                    "R_world": d["R_world"].astype(np.float32), "rest_h": rest_h,
                    "centroid_depth": cdepth, "prob": prob,
                    "size": int(d["size"]), "yaw_step": float(d["yaw_step"])}
        except Exception:
            bank = None
    _BANK_CACHE[part] = bank
    return bank


def _query_relief(crop, size):
    """Crop -> Form-Signal (0..1), das mit der Template-Tiefe vergleichbar ist:
    quadrat-gepaddete, normierte Abweichung vom Neutralgrau. Trägt die Silhouette
    + grobe Binnenstruktur des Teils auch aus reinem RGB."""
    a = np.asarray(crop)
    g = a.astype(np.float32).mean(2) if a.ndim == 3 else a.astype(np.float32)
    h, w = g.shape[:2]
    sd = max(h, w, 1)
    sq = np.full((sd, sd), 128.0, np.float32)
    sq[(sd - h) // 2:(sd - h) // 2 + h, (sd - w) // 2:(sd - w) // 2 + w] = g
    img = np.asarray(Image.fromarray(sq.astype(np.uint8)).resize((size, size),
                     Image.BILINEAR)).astype(np.float32)
    sig = np.abs(img - 128.0)
    m = sig.max()
    return sig / m if m > 1e-6 else sig


def _query_relief_depth(depth_crop, size):
    """Tiefen-Crop -> Höhen-Relief (0..1), direkt vergleichbar mit den Bank-Tiefen-
    Templates. Eine metrische Top-Down-Tiefenkarte (distance_to_camera) ist VIEL
    diskriminativer als das RGB-Helligkeits-Relief: sie kodiert die echte Höhe des
    Teils und löst Ruhelagen auf, die im RGB mehrdeutig aussehen. Hintergrund =
    Tischebene (große Distanz), Teil = nähere Pixel. Wir invertieren zu Höhe-über-
    Tisch, normieren wie der Bank-Renderer (0 = Hintergrund/tiefster Punkt, 1 =
    höchster). None wenn der Crop keine brauchbare Tiefe hat."""
    a = np.asarray(depth_crop, np.float32)
    finite = np.isfinite(a) & (a > 0)
    if int(finite.sum()) < 8:
        return None
    far = float(np.nanmax(a[finite]))
    d = a.copy(); d[~finite] = far
    # Höhe über dem fernsten (= Tisch-)Punkt: nähere Pixel sind höher.
    h = far - d
    # Teil-Maske: deutlich über der Tischebene (robust gegen Tisch-Tiefenrauschen).
    span = float(h[finite].max())
    if span < 1e-6:
        return None
    part_mask = h > 0.10 * span
    h[~part_mask] = 0.0
    H, W = h.shape
    sd = max(H, W)
    sq = np.zeros((sd, sd), np.float32)
    sq[(sd - H) // 2:(sd - H) // 2 + H, (sd - W) // 2:(sd - W) // 2 + W] = h
    img = np.asarray(Image.fromarray((sq / (sq.max() + 1e-9) * 255).astype(np.uint8))
                     .resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    return img


def _yaw_window_mask(yaws, obb_angle_deg, half_window=25.0):
    """Boolean-Maske: nur Yaws im Fenster um den OBB-Winkel + 180°-Flip. Verkleinert
    die Suche und nutzt den Detektor-Prior. None -> ganze Bank durchsuchen."""
    if obb_angle_deg is None:
        return np.ones(len(yaws), bool)
    base = obb_angle_deg % 360.0
    mask = np.zeros(len(yaws), bool)
    for b in (base, (base + 180.0) % 360.0):
        dd = np.abs((yaws - b + 180.0) % 360.0 - 180.0)
        mask |= dd <= half_window
    return mask if mask.any() else np.ones(len(yaws), bool)


def _grad_mag(img):
    """Sobel-Gradientenbetrag (0..1), normiert. Kanten = diskriminatives Form-Signal."""
    a = np.asarray(img, np.float32)
    gx = ndimage.sobel(a, axis=1, mode="constant")
    gy = ndimage.sobel(a, axis=0, mode="constant")
    g = np.hypot(gx, gy)
    m = g.max()
    return g / m if m > 1e-6 else g


def _refine_yaw_subdeg(q, base_template, yaw0, R0, Rstable=None, bank=None,
                       best_i=None, span=None, step=1.0):
    """Yaw unter das Raster verfeinern: das beste Tiefen-Template um feine Winkel in
    2D rotieren (Top-Down: Welt-Z-Yaw == Bilddrehung) und gegen die Query maximieren.
    Liefert (verfeinerter_yaw, verfeinertes_R). R wird um die Welt-Z-Differenz nach-
    gedreht (R = Rz(Δ) @ R0). Konservativ: nur innerhalb ±yaw_step."""
    if span is None:
        span = float(bank["yaw_step"]) if bank is not None else YAW_STEP_DEG
    qf = q.ravel() - q.ravel().mean()
    qn = np.linalg.norm(qf) + 1e-9
    best_d, best_ncc = 0.0, -np.inf
    for dd in np.arange(-span, span + 1e-6, step):
        if abs(dd) < 1e-6:
            rt = base_template
        else:
            rt = ndimage.rotate(base_template, dd, reshape=False, order=1,
                                mode="constant", cval=0.0)
        tf = rt.ravel() - rt.ravel().mean()
        ncc = float((qf @ tf) / (qn * (np.linalg.norm(tf) + 1e-9)))
        if ncc > best_ncc:
            best_ncc, best_d = ncc, float(dd)
    if abs(best_d) < 1e-6:
        return yaw0, R0
    th = np.radians(best_d)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    return (yaw0 + best_d) % 360.0, Rz @ R0


def match_template_bank(crop, part, obb_angle_deg=None, registry=None, depth_crop=None):
    """Crop gegen die Template-Bank matchen -> (face_idx, yaw, R_world, score, ...).

    Score = Korrelation (NCC) zwischen Query-Relief und Template-Tiefe, gewichtet
    mit Silhouetten-IoU + (wenn vorhanden) einem schwachen Stable-Pose-Prior. Sucht
    über alle Ruhelagen; pro Lage nur Yaws im OBB-Fenster. Wenn ein metrischer
    Tiefen-Crop (depth_crop) vorliegt, wird das DEPTH-Relief als Query genutzt — das
    löst Ruhelagen viel zuverlässiger auf als reines RGB. Liefert None ohne Bank."""
    bank = load_template_bank(part)
    if bank is None:
        return None
    size = bank["size"]
    q = _query_relief_depth(depth_crop, size) if depth_crop is not None else None
    if q is None:
        q = _query_relief(crop, size)
    qf = q.ravel(); qf = qf - qf.mean()
    qn = np.linalg.norm(qf) + 1e-9
    qsil = (q > 0.10).astype(np.float32)

    # Gradient-Betrag der Query (Kanten) — diskriminativer als reine Höhe für die
    # Yaw-/Lagen-Auflösung, robust gegen globale Tiefen-Offsets (Domain-Gap).
    qg = _grad_mag(q); qgf = qg.ravel(); qgf = qgf - qgf.mean()
    qgn = np.linalg.norm(qgf) + 1e-9

    depth = bank["depth"]; sil = bank["sil"]; yaws = bank["yaw"]
    prob = bank.get("prob")
    cdepth = bank.get("centroid_depth")
    pmax = float(prob.max()) if prob is not None and prob.size else 1.0
    win = _yaw_window_mask(yaws, obb_angle_deg)
    idxs = np.nonzero(win)[0]

    def _score(i):
        t = depth[i].ravel(); t = t - t.mean()
        ncc = float((qf @ t) / (qn * (np.linalg.norm(t) + 1e-9)))
        si = sil[i]
        inter = float((qsil * si).sum()); union = float(((qsil + si) > 0).sum()) + 1e-9
        iou = inter / union
        tg = _grad_mag(depth[i]).ravel(); tg = tg - tg.mean()
        gncc = float((qgf @ tg) / (qgn * (np.linalg.norm(tg) + 1e-9)))
        # depth-NCC + Silhouetten-IoU + Kanten-NCC (multi-cue, je robust gegen eine
        # andere Domain-Gap-Komponente).
        sc = 0.45 * ncc + 0.30 * iou + 0.25 * gncc
        if prob is not None and prob.size:
            sc += 0.05 * float(prob[i]) / (pmax + 1e-9)
        return sc

    scores = np.array([_score(i) for i in idxs])
    # LERNBASIERTES Refinement: wenn ein refiner_<part>.pt da ist, die Bank-Kandidaten
    # mit dem gelernten Embedding re-ranken (Cosinus Query<->Bank-Template-Embedding).
    # Das Embedding schliesst den Domain-Gap, an dem die Handmetrik die richtige Lage
    # verfehlt. Blend (gewichtete Summe) ist robuster als hartes Ersetzen: die Hand-
    # metrik bleibt ein sanity-Anker. learned_w hoch, weil die gelernte Cue die
    # diskriminative ist. Über POSE_REFINER_WEIGHT justierbar (0 = aus).
    rsc = _refiner_scores(part, q)
    used_refiner = False
    if rsc is not None and rsc.shape[0] == len(yaws):
        lw = float(os.environ.get("POSE_REFINER_WEIGHT", "0.75"))
        # Hand- und Lern-Scores je auf 0..1 robust normieren (rangbasiert stabil),
        # dann blenden — beide leben auf unterschiedlichen Skalen.
        rl = rsc[idxs]
        def _nrm(a):
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo + 1e-9)
        scores = (1.0 - lw) * _nrm(scores) + lw * _nrm(rl)
        used_refiner = True
    order = np.argsort(-scores)
    best_i = int(idxs[order[0]]); best_score = float(scores[order[0]])
    face_k = int(bank["face_idx"][best_i])
    R = bank["R_world"][best_i].reshape(3, 3)
    yaw = float(yaws[best_i])
    rest_h = float(bank["rest_h"][best_i])         # metrische Ruhe-Höhe der Lage
    centroid_depth = float(cdepth[best_i]) if cdepth is not None else rest_h
    # Sub-Grad-Yaw-Verfeinerung um den besten Kandidaten (±yaw_step, 1°) -> Yaw
    # schärfer als das 5°-Raster, ohne die Lagen-/Flip-Wahl zu verfälschen.
    yaw, R = _refine_yaw_subdeg(q, depth[best_i], yaw, R, bank=bank, best_i=best_i)
    # Face-Name + Aufrecht-Flag aus der Registry (sonst generisch).
    face_name = f"Face {face_k}"; upright = False
    if registry and registry.faces:
        for f in registry.faces:
            if f.index == face_k:
                face_name = f.name
                upright = bool(f.tilt_deg >= UPRIGHT_TILT_DEG)
                if rest_h <= 0.0 and f.rest_height > 0.0:
                    rest_h = f.rest_height          # Bank ohne rest_h -> Registry
                break
    # Match-Confidence: Score auf 0..1 geklemmt (NCC kann negativ werden).
    conf = float(np.clip((best_score + 0.2) / 1.0, 0.0, 1.0))
    return {"face_idx": face_k, "face_name": face_name, "yaw_deg": yaw,
            "R_world": R, "score": best_score, "confidence": conf, "upright": upright,
            "rest_height": rest_h, "centroid_depth": centroid_depth,
            "template_idx": int(best_i), "refined": bool(used_refiner)}


def estimate_yaw(crop, template, step_deg=YAW_STEP_DEG):
    """Fallback (keine Bank): Yaw via Silhouetten-MSE über ein einzelnes Template."""
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


def align_detection(part, classifier_face, crop, registry, step_deg=YAW_STEP_DEG,
                    obb_angle_deg=None, depth_crop=None):
    """6D-Pose eines Crops. PRIMÄR: Template-Bank Render-and-Compare (Face + Yaw +
    R_world deterministisch aus der Bank). Mit depth_crop wird die Ruhelage über das
    metrische Tiefen-Relief aufgelöst (zuverlässiger). FALLBACK (keine Bank):
    Registry-R_face + Yaw-Suche über das Face-Template."""
    # 1) Template-Bank-Match — die Pose-Quelle.
    m = match_template_bank(crop, part, obb_angle_deg=obb_angle_deg, registry=registry,
                            depth_crop=depth_crop)
    if m is not None:
        return {"R_world": [float(v) for v in m["R_world"].flatten()],
                "yaw_deg": m["yaw_deg"], "upright": m["upright"],
                "face_name": m["face_name"], "rest_height": float(m["rest_height"]),
                "centroid_depth": float(m.get("centroid_depth", m["rest_height"])),
                "match_score": float(m["score"]), "match_conf": m["confidence"],
                "template_idx": m["template_idx"],
                "pose_source": "template_bank+refiner" if m.get("refined") else "template_bank"}
    # 2) Fallback: Registry-R_face + Yaw-Suche (kein Bank-NPZ vorhanden).
    face = registry.resolve_face(classifier_face) if registry else None
    if face is None:
        return {"R_world": list(np.eye(3).flatten()), "yaw_deg": 0.0, "upright": False,
                "face_name": classifier_face or "Face 1", "rest_height": 0.0,
                "pose_source": "identity"}
    template = load_template(face, size=TMPL_SIZE)
    yaw = estimate_yaw(crop, template, step_deg)
    R_world = _rz(np.radians(yaw)) @ face.R_face
    return {"R_world": [float(v) for v in R_world.flatten()], "yaw_deg": yaw,
            "upright": bool(face.tilt_deg >= UPRIGHT_TILT_DEG),
            "face_name": face.name, "rest_height": float(face.rest_height),
            "pose_source": "registry_yaw"}


class Intrinsics:
    def __init__(self, width, height, cam_h=DEFAULT_CAM_H, focal_mm=DEFAULT_FOCAL_MM,
                 sensor_mm=DEFAULT_SENSOR_MM, table_origin=(0.0, 0.0, 0.0)):
        self.width = width; self.height = height; self.cam_h = cam_h
        self.table_origin = table_origin
        self.cx = width / 2.0; self.cy = height / 2.0
        self.fx = focal_mm / sensor_mm * width
        self.fy = focal_mm / sensor_mm * height


class SceneIntrinsics:
    """Metrische Backprojection für die Multi-Part-Zellen-View mit den ECHTEN
    Zivid-Intrinsics + Extrinsics aus GST_Scene.usd. Jeder BBox-Zentrums-Pixel
    wird zu einem Welt-Strahl (Kamera-Ursprung + Richtung) und gegen die Tisch-
    Ebene z = table_origin[2] + rest_height geschnitten. Ergebnis ist metrisch in
    Welt-Koordinaten — Teile liegen an der RICHTIGEN Stelle (nicht geclustert).

    USD-Kamera-Konvention: blickt entlang lokal -Z, +Y oben, +X rechts. fx/fy aus
    focalLength/aperture. Pixel (u,v) (v wächst nach unten) -> lokaler Strahl ->
    mit R_wc (Welt-aus-Kamera) nach Welt gedreht."""

    def __init__(self, width, height, table_origin=(0.0, 0.0, 0.0), zivid=ZIVID,
                 tray=SCENE_TRAY_BOUNDS):
        self.width = width; self.height = height; self.table_origin = table_origin
        self.tray = tray
        # Intrinsics auf die TATSÄCHLICHE Render-Auflösung skalieren.
        sx = width / zivid["render_w"]; sy = height / zivid["render_h"]
        self.fx = zivid["focal_mm"] / zivid["aperture_h_mm"] * zivid["render_w"] * sx
        self.fy = zivid["focal_mm"] / zivid["aperture_v_mm"] * zivid["render_h"] * sy
        self.cx = width / 2.0; self.cy = height / 2.0
        self.cam_t = np.asarray(zivid["cam_t"], float)
        self.R_wc = np.asarray(zivid["R_wc"], float)            # Zeilen-Konvention

    def _ray_dir_world(self, u, v):
        # Pixel -> lokale Kamera-Richtung (USD: -Z vorwärts, +Y oben).
        dx = (u - self.cx) / self.fx
        dy = -(v - self.cy) / self.fy        # v wächst nach unten -> +Y oben
        d_cam = np.array([dx, dy, -1.0])
        # Welt-Richtung: USD row-vector -> d_world = d_cam @ R_wc.
        d_world = d_cam @ self.R_wc
        n = np.linalg.norm(d_world)
        return d_world / n if n > 1e-9 else d_world

    def to_world(self, bbox, rest_height=0.0, uv_override=None):
        x0, y0, x1, y1 = bbox
        if uv_override is not None:
            u, v = float(uv_override[0]), float(uv_override[1])
        else:
            u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        d = self._ray_dir_world(u, v)
        z_plane = self.table_origin[2] + rest_height
        # Strahl cam_t + s*d schneidet z = z_plane.
        if abs(d[2]) < 1e-9:
            s = 0.0
        else:
            s = (z_plane - self.cam_t[2]) / d[2]
        p = self.cam_t + max(s, 0.0) * d
        x = p[0] - self.table_origin[0]; y = p[1] - self.table_origin[1]
        # Plausi-Klemme auf die (leicht erweiterten) Tray-Grenzen.
        tx, ty = self.tray["x"], self.tray["y"]
        mx = (tx[1] - tx[0]) * 0.15; my = (ty[1] - ty[0]) * 0.15
        x = float(np.clip(x, tx[0] - mx - self.table_origin[0], tx[1] + mx - self.table_origin[0]))
        y = float(np.clip(y, ty[0] - my - self.table_origin[1], ty[1] + my - self.table_origin[1]))
        return [x, y, float(z_plane)]


def _part_pixel_centroid(depth_crop, bbox):
    """(u,v) Bild-Schwerpunkt der Teil-Pixel (über der Tischebene) im BBox-Fenster.
    None ohne brauchbare Tiefe -> Aufrufer nutzt das BBox-Zentrum. Robust: nimmt nur
    Pixel deutlich über dem fernsten (Tisch-)Punkt, gewichtet mit der Höhe."""
    if depth_crop is None:
        return None
    a = np.asarray(depth_crop, np.float32)
    finite = np.isfinite(a) & (a > 0)
    if int(finite.sum()) < 12:
        return None
    far = float(np.nanmax(a[finite]))
    h = far - a; h[~finite] = 0.0
    span = float(h[finite].max())
    if span < 1e-6:
        return None
    mask = h > 0.20 * span
    if int(mask.sum()) < 8:
        return None
    ys, xs = np.nonzero(mask)
    w = h[mask]
    cu = float((xs * w).sum() / (w.sum() + 1e-9)) + bbox[0]
    cv = float((ys * w).sum() / (w.sum() + 1e-9)) + bbox[1]
    return (cu, cv)


def bbox_center_to_world(bbox, intr, rest_height=0.0, uv_override=None):
    if isinstance(intr, SceneIntrinsics):
        return intr.to_world(bbox, rest_height, uv_override=uv_override)
    x0, y0, x1, y1 = bbox
    if uv_override is not None:
        u, v = float(uv_override[0]), float(uv_override[1])
    else:
        u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x = (u - intr.cx) * intr.cam_h / intr.fx + intr.table_origin[0]
    y = -(v - intr.cy) * intr.cam_h / intr.fy + intr.table_origin[1]
    return [float(x), float(y), float(intr.table_origin[2] + rest_height)]


def camera_position(intr):
    """Welt-Position der Kamera (für die Tiefen-/Stapel-Ordnung). SceneIntrinsics
    kennt sie exakt (cam_t); Top-down-Intrinsics: senkrecht über dem Nullpunkt."""
    if isinstance(intr, SceneIntrinsics):
        return np.asarray(intr.cam_t, float)
    return np.array([intr.table_origin[0], intr.table_origin[1],
                     intr.table_origin[2] + intr.cam_h], float)


# ── P4: Overlap/Occlusion — Tiefen-Stapelung ──────────────────────────────────
def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _mean_crop_depth(depth_map, bbox):
    """Mittlere Tiefe (Kamera-Distanz) im BBox-Fenster aus einer echten Tiefenkarte.
    None wenn keine Karte / leeres Fenster."""
    if depth_map is None:
        return None
    H, W = depth_map.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, x1 = max(0, x0), min(W, x1); y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = depth_map[y0:y1, x0:x1].astype(np.float32)
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return float(np.median(sub)) if sub.size else None


def load_depth_map(img_path, shape_hw, warn=print):
    """Tiefenkarte neben dem Bild laden (für die Stapel-Ordnung, P4). Bevorzugt
    metrisch (depth_<idx>.npy = distance_to_camera in m), sonst relativ
    (depth_<idx>.png 16-bit, per-Bild normalisiert — Ordnung bleibt erhalten).
    None wenn keine Karte da -> Fallback Kamera-Distanz aus t_world."""
    img_path = pathlib.Path(img_path)
    stem = img_path.stem.replace("rgb_", "").replace("scene_", "")
    H, W = shape_hw
    for cand in (img_path.parent / f"depth_{stem}.npy",
                 img_path.parent / f"depth_{img_path.stem}.npy"):
        if cand.exists():
            try:
                dm = np.load(cand).astype(np.float32)
                warn(f"[occlusion] metrische Tiefenkarte {cand.name}")
                return dm
            except Exception:
                pass
    for cand in (img_path.parent / f"depth_{stem}.png",
                 img_path.parent / f"depth_{img_path.stem}.png"):
        if cand.exists():
            try:
                dm = np.asarray(Image.open(cand)).astype(np.float32)
                if dm.ndim == 3:
                    dm = dm.mean(2)
                warn(f"[occlusion] relative Tiefenkarte {cand.name} (Ordnung erhalten)")
                return dm
            except Exception:
                pass
    return None


def assign_occlusion_order(aligned, intr, depth_map=None, iou_thresh=0.10, warn=print):
    """P4 — Stapel-Reihenfolge bei überlappenden 2D-BBoxes.

    Pro Detection eine Kamera-Tiefe bestimmen (echte Tiefenkarte falls vorhanden,
    sonst ||cam - t_world||). Kleinere Tiefe = näher an der Kamera = liegt OBEN.
    `depth_order` = globaler Rang (0 = unterste/fernste Lage zuerst legen). Pro
    überlappendem Paar das nähere Teil als 'on top' markieren (`occluded_by`).
    Die Render-/Platzier-Reihenfolge folgt depth_order (fern -> nah, unten -> oben).
    """
    cam = camera_position(intr)
    for a in aligned:
        d = _mean_crop_depth(depth_map, a["bbox_2d"])
        if d is None:
            d = float(np.linalg.norm(cam - np.asarray(a["t_world"], float)))
        a["cam_depth"] = d
        a["occluded_by"] = []
        a["on_top_of"] = []
    # paarweise Überlappung -> wer ist näher (oben)
    pairs = 0
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            if _bbox_iou(aligned[i]["bbox_2d"], aligned[j]["bbox_2d"]) < iou_thresh:
                continue
            pairs += 1
            near, far = (i, j) if aligned[i]["cam_depth"] <= aligned[j]["cam_depth"] else (j, i)
            aligned[far]["occluded_by"].append(aligned[near]["instance_id"])
            aligned[near]["on_top_of"].append(aligned[far]["instance_id"])
    # globaler Stapel-Rang: fern (groß) zuerst = 0, nah (klein) zuletzt = oben
    order = sorted(range(len(aligned)), key=lambda k: -aligned[k]["cam_depth"])
    for rank, k in enumerate(order):
        aligned[k]["depth_order"] = rank
    n_top = sum(1 for a in aligned if a["occluded_by"])
    warn(f"[occlusion] {pairs} überlappende Paare, {n_top} Teile (teil-)verdeckt "
         f"(Tiefe: {'Karte' if depth_map is not None else 'Kamera-Distanz'})")
    return aligned


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
        # optionale P2/P4-Felder (nur prüfen wenn vorhanden)
        if "rest_height" in r and not (_is_num(r["rest_height"]) and r["rest_height"] >= 0):
            e.append(f"{p}.rest_height")
        if "depth_order" in r and not (_is_int(r["depth_order"]) and r["depth_order"] >= 0):
            e.append(f"{p}.depth_order")
        for key in ("occluded_by", "on_top_of"):
            if key in r and not (isinstance(r[key], list)
                                 and all(_is_int(v) and v >= 0 for v in r[key])):
                e.append(f"{p}.{key}")
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
                     "upright": bool(a["upright"]),
                     "rest_height": float(max(0.0, a.get("rest_height", 0.0))),
                     "depth_order": int(a.get("depth_order", 0)),
                     "occluded_by": [int(v) for v in a.get("occluded_by", [])],
                     "on_top_of": [int(v) for v in a.get("on_top_of", [])]}
                    for a in aligned],
    }


# Tisch-Nullpunkt in Welt: die ECHTE Arbeitsfläche der Zelle, auf der die Teile
# ruhen. Aus GST_Scene.usd gemessen: Anker_Tray/Poltopf_Tray zmin = -0.0070 m
# (= Wagen-Tischplatte). x/y = Welt-Ursprung der GST_Scene, damit die platzierten
# Teile mit dem cell.glb (echtes CAD, Welt-Frame) fluchten. t_world.z =
# TABLE_Z + rest_height(face) -> jedes Teil ruht mit seiner Unterseite auf -0.007.
TABLE_Z_SCENE = -0.007
TABLE_ORIGIN_SCENE = (0.0, 0.0, TABLE_Z_SCENE)


def scene_intrinsics_from_params(width, height, cam_params, table_origin=TABLE_ORIGIN_SCENE):
    """SceneIntrinsics aus aufgezeichneten Kamera-Parametern (P5-Eval: die GT-Szene
    speichert die echte _DRCam-Pose, damit die Backprojection konsistent ist)."""
    zivid = {"focal_mm": cam_params["focal_mm"], "aperture_h_mm": cam_params["aperture_h_mm"],
             "aperture_v_mm": cam_params["aperture_v_mm"], "render_w": cam_params["render_w"],
             "render_h": cam_params["render_h"], "cam_t": tuple(cam_params["cam_t"]),
             "R_wc": tuple(tuple(r) for r in cam_params["R_wc"])}
    return SceneIntrinsics(width, height, table_origin=table_origin, zivid=zivid)


# ── Orchestrierung ────────────────────────────────────────────────────────────
def run(image, out_path, warn=print, intr_override=None):
    """Ganze Pipeline für ein Bild. Schreibt schema-valides pose_result.json.

    Pose-Quelle ist die Template-Bank (Render-and-Compare): Face + Yaw + R_world
    kommen aus dem Bank-Match, NICHT aus dem Face-Classifier. Der Classifier läuft
    nur noch als optionaler Schnell-Vorfilter (Log/Vis). intr_override erlaubt der
    Eval, mit den aufgezeichneten GT-Kamera-Intrinsics zu backprojizieren."""
    rgb, dets = detections_for(image, warn=warn)
    H, W = rgb.shape[:2]
    # Tiefenkarte früh laden: dient sowohl der Ruhelagen-Auflösung (Bank-Match über
    # Depth-Relief) als auch später der Stapel-Ordnung (P4).
    depth_map = load_depth_map(image, (H, W), warn=warn)
    # Zellen-View (Multi-Part-Szene) -> echte Zivid-Intrinsics, metrische Backprojection;
    # sonst top-down Faceset-Intrinsics. Heuristik: breite Szene (>=1000px) = Zellen-View.
    if intr_override is not None:
        intr = intr_override
    elif W >= 1000:
        # P0: die Zellen-/Inferenz-View ist jetzt TOP-DOWN -> Top-Down-Intrinsics.
        intr = SceneIntrinsics(W, H, table_origin=TABLE_ORIGIN_SCENE, zivid=ZIVID_TOPDOWN)
    else:
        intr = Intrinsics(W, H)
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
        depth_crop = (depth_map[y0:y1, x0:x1].copy()
                      if depth_map is not None else None)
        res = infer(crop, d["part"])                      # optionaler Vorfilter
        if d["part"] not in reg_cache:
            reg_cache[d["part"]] = load_part_registry(d["part"])
        a = align_detection(d["part"], res["face"], crop, reg_cache[d["part"]],
                            obb_angle_deg=d.get("obb_angle_deg"), depth_crop=depth_crop)
        # Translation: z = Tisch + Schwerpunkt-Höhe (centroid_depth, exakt aus der
        # Isaac-Bank-Geometrie statt halber z-Spanne). x/y per Backprojection des
        # TEIL-PIXEL-SCHWERPUNKTS (aus der Tiefen-Maske) statt des bloßen BBox-
        # Zentrums — bei am Rand abgeschnittenen oder asymmetrischen Crops sitzt das
        # BBox-Zentrum daneben, der Tiefen-Schwerpunkt trifft den echten Mittelpunkt.
        z_h = a.get("centroid_depth", a["rest_height"])
        uv = _part_pixel_centroid(depth_crop, d["bbox_2d"])
        t_world = bbox_center_to_world(d["bbox_2d"], intr, rest_height=z_h, uv_override=uv)
        # Confidence aus der Pose-Quelle: Template-Bank-Match-Score wenn vorhanden,
        # sonst die (schwächere) Face-Classifier-Confidence.
        conf = a.get("match_conf", res["confidence"])
        aligned.append({"instance_id": d["instance_id"], "part": d["part"],
                        "confidence": conf, "bbox_2d": d["bbox_2d"],
                        "t_world": t_world, "clf_face": res["face"],
                        "clf_conf": res["confidence"], **a})
        src = a.get("pose_source", "?")
        sc = a.get("match_score")
        sc_s = f"score={sc:+.2f} " if sc is not None else ""
        warn(f"[infer] #{d['instance_id']:>2} {d['part']:<22} {a['face_name']:<8} "
             f"conf={conf:.2f} [{src}] {sc_s}"
             f"yaw={a['yaw_deg']:5.1f}° upright={a['upright']} "
             f"rest_h={a.get('rest_height', 0.0):.3f} "
             f"(clf {res['face']}/{res['confidence']:.2f})")
    # P4: Tiefen-/Stapel-Reihenfolge bei überlappenden BBoxes auflösen
    # (Tiefenkarte oben bereits geladen).
    assign_occlusion_order(aligned, intr, depth_map=depth_map, warn=warn)
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


# ── P5: Eval gegen Ground-Truth-6D-Pose ───────────────────────────────────────
def _rot_err_deg(Ra, Rb):
    """Geodätischer Rotationsfehler (Grad) zwischen zwei Rotationsmatrizen.
    Robust gegen Teil-Symmetrien NICHT — misst die rohe Pose-Abweichung."""
    Ra = np.asarray(Ra, float).reshape(3, 3); Rb = np.asarray(Rb, float).reshape(3, 3)
    Rd = Ra.T @ Rb
    c = (np.trace(Rd) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# Pro-Teil Symmetrie-Spezifikation für die sym-bewusste Metrik. Eine unbeobachtbare
# DoF (Rotation um eine Körperachse) DARF nicht bestraft werden — sonst misst man
# physikalisch nicht auflösbare Mehrdeutigkeit als "Fehler".
#   axis      = Körperachse der kontinuierlichen Rotationssymmetrie (None = keine)
#   discrete  = zusätzliche diskrete In-Plane-Flips um Welt-Z (Grad-Liste)
# Anker/Ringmagnet sind Drehkörper -> kontinuierlich um ihre lange Körperachse (Y).
PART_SYMMETRY = {
    "Anker_Lang": {"axis": (0, 1, 0), "discrete": [0, 180]},
    "Anker_Kurz": {"axis": (0, 1, 0), "discrete": [0, 180]},
    "Ringmagnet": {"axis": (0, 0, 1), "discrete": [0, 90, 180, 270]},
    "Zahnrad": {"axis": None, "discrete": [k * 360.0 / 14 for k in range(14)]},
}
_DEFAULT_SYM = {"axis": None, "discrete": [0, 90, 180, 270]}


def _sym_rot_err_deg(Ra, Rb, part=None):
    """Symmetrie-bewusster Rotationsfehler. Für Drehkörper (axis gesetzt) wird über
    die KONTINUIERLICHE Rotation um die Körperachse minimiert (die unbeobachtbare
    DoF wird nicht bestraft); zusätzlich diskrete In-Plane-Flips. Default: 4 Welt-Z-
    Flips (planare Teile). Ra=GT, Rb=Pred (beide world=R@body)."""
    Ra = np.asarray(Ra, float).reshape(3, 3); Rb = np.asarray(Rb, float).reshape(3, 3)
    spec = PART_SYMMETRY.get(part, _DEFAULT_SYM) if part else _DEFAULT_SYM
    best = 1e9
    # diskrete In-Plane-Flips um Welt-Z
    for dz in spec.get("discrete", [0]):
        th = np.radians(dz)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        Rb_z = Rz @ Rb
        axis = spec.get("axis")
        if axis is None:
            best = min(best, _rot_err_deg(Ra, Rb_z))
        else:
            # kontinuierliche Symmetrie um die Körperachse: feines Sampling (2°)
            ax = np.asarray(axis, float); ax = ax / (np.linalg.norm(ax) + 1e-9)
            for ang in np.arange(0.0, 360.0, 2.0):
                best = min(best, _rot_err_deg(Ra, Rb_z @ _axis_angle(ax, np.radians(ang))))
    return best


def _axis_angle(ax, theta):
    """Rodrigues-Rotationsmatrix um Achse ax (Einheit) mit Winkel theta (rad)."""
    x, y, z = ax; c, s = np.cos(theta), np.sin(theta); C = 1 - c
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def eval_against_gt(image, gt_path, warn=print):
    """P5 — volle Pipeline auf eine Eval-Szene mit bekannter GT-6D-Pose laufen
    lassen + den Fehler messen. Matcht Predictions zu GT-Instanzen per 2D-BBox-
    IoU. Liefert pro-Instanz + aggregierte Translation-(mm)/Rotation-(deg)/Face-/
    Klassen-Metriken. gt_path = eval_gt_<idx>.json (vom Isaac-Eval-Generator)."""
    gt = json.load(open(gt_path))
    table_z = float(gt.get("table_z", TABLE_ORIGIN_SCENE[2]))
    gt_inst = [g for g in gt.get("instances", []) if g.get("bbox_2d")]
    # Pipeline laufen lassen (schreibt pose_result; wir nehmen das doc direkt).
    # Mit den aufgezeichneten GT-Kamera-Intrinsics backprojizieren -> kamera-konsistent.
    intr_override = None
    cp = gt.get("camera_params")
    if cp:
        from PIL import Image as _Im
        _w, _h = _Im.open(image).size
        intr_override = scene_intrinsics_from_params(_w, _h, cp,
                            table_origin=(0.0, 0.0, table_z))
    tmp_out = pathlib.Path(image).parent / f"_eval_pred_{pathlib.Path(image).stem}.json"
    doc = run(image, tmp_out, warn=lambda *a, **k: None, intr_override=intr_override)
    preds = doc["results"]
    # Pred-t_world ist bereits Welt: x/y weil table_origin x/y = 0, z weil die
    # Pipeline z = table_origin.z + rest_height (absolut) schreibt — dieselbe
    # Konvention, mit der der Viewer die Boxen platziert. GT-t ist ebenfalls Welt.
    def pred_world_t(r):
        return list(r["t_world"])
    # Greedy-Matching per BBox-IoU (höchste IoU zuerst).
    pairs = []
    for gi, g in enumerate(gt_inst):
        for pi, p in enumerate(preds):
            iou = _bbox_iou(g["bbox_2d"], p["bbox_2d"])
            if iou >= 0.30:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); matches.append((gi, pi, iou))
    rows = []
    for gi, pi, iou in matches:
        g, p = gt_inst[gi], preds[pi]
        # GT-t ist der PRIM-URSPRUNG des Teils — bei vielen USD-Prims weit vom
        # Schwerpunkt entfernt (Getriebe: prim-origin schwebt 26cm über der Geometrie).
        # Die Pipeline platziert am Schwerpunkt. Für einen fairen Vergleich GT auf den
        # Schwerpunkt umrechnen: t_centroid = t_prim + R_gt @ centroid_offset_body.
        off = centroid_offset(g["part"])
        Rg = np.asarray(g["R"], float).reshape(3, 3)
        gt_t_centroid = np.asarray(g["t"], float) + Rg @ off
        te = float(np.linalg.norm(np.asarray(pred_world_t(p)) - gt_t_centroid)) * 1000.0
        re = _rot_err_deg(g["R"], p["R_world"])
        re_sym = _sym_rot_err_deg(g["R"], p["R_world"], part=g["part"])
        cls_ok = (g["part"].lower() == p["part"].lower())
        rows.append({"gt_part": g["part"], "pred_part": p["part"], "class_ok": cls_ok,
                     "iou": iou, "trans_err_mm": te, "rot_err_deg": re,
                     "rot_err_sym_deg": re_sym, "pred_face": p.get("face")})
        warn(f"[eval] {g['part']:<22} -> {p['part']:<22} "
             f"cls={'OK ' if cls_ok else 'XX '} t_err={te:6.1f}mm "
             f"R_err={re:5.1f}° (sym {re_sym:5.1f}°) IoU={iou:.2f}")
    n_gt, n_pred, n_match = len(gt_inst), len(preds), len(matches)
    cls_correct = sum(1 for r in rows if r["class_ok"])
    agg = {
        "n_gt": n_gt, "n_pred": n_pred, "n_matched": n_match,
        "detect_recall": n_match / n_gt if n_gt else 0.0,
        "class_acc": cls_correct / n_match if n_match else 0.0,
        "trans_err_mm_mean": float(np.mean([r["trans_err_mm"] for r in rows])) if rows else None,
        "trans_err_mm_median": float(np.median([r["trans_err_mm"] for r in rows])) if rows else None,
        "rot_err_deg_mean": float(np.mean([r["rot_err_deg"] for r in rows])) if rows else None,
        "rot_err_deg_median": float(np.median([r["rot_err_deg"] for r in rows])) if rows else None,
        "rot_err_sym_deg_median": float(np.median([r["rot_err_sym_deg"] for r in rows])) if rows else None,
    }
    return {"image": str(image), "gt": str(gt_path), "rows": rows, "agg": agg}


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
    ap.add_argument("--eval-gt", default=None,
                    help="P5: gegen eine GT-Datei (eval_gt_<idx>.json) evaluieren statt nur Inferenz")
    a = ap.parse_args(argv)
    out = a.out or str(HERE / "temp" / "pose_result.json")
    if a.eval_gt:
        res = eval_against_gt(a.image, a.eval_gt)
        ag = res["agg"]
        print(f"\n[eval] {ag['n_matched']}/{ag['n_gt']} gematcht | recall={ag['detect_recall']:.2f} "
              f"| class_acc={ag['class_acc']:.2f}")
        print(f"[eval] Translation-Error: median={ag['trans_err_mm_median']} mm  mean={ag['trans_err_mm_mean']} mm")
        print(f"[eval] Rotation-Error:    median={ag['rot_err_deg_median']}°  "
              f"(symmetrie-bewusst median={ag['rot_err_sym_deg_median']}°)")
        return 0
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
