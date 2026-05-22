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

import numpy as np
from PIL import Image

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


def available_parts():
    """Teile-Namen aus models/part_meta.json (post-ADR-018). Fallback: Liste."""
    pm = MODELS / "part_meta.json"
    if pm.exists():
        try:
            data = json.load(open(pm))
            names = list(data.get("parts", data)) if isinstance(data, (dict, list)) else []
            if names:
                return sorted(str(n) for n in names)
        except Exception:
            pass
    return ["Anker_Lang", "Anker_Kurz", "Zahnrad", "Poltopf_kurz_centered",
            "Getriebegehaeuse_typ4", "Buerstenhalter_2polig"]


def _canonical_parts():
    return {p.lower(): p for p in available_parts()}


def canonical_part(raw, canon):
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


def _obb_to_aabb(corners, W, H):
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
        warn(f"[detect] DUMMY: eine ganze-Bild-BBox als '{part}'")
    return rgb, dets


# === BOP pose pipeline (GDRNPP — siehe ADR-018), wird in W2/W3 eingesetzt ======
# Stufe 3 (6D-Pose) ist hier ein STUB. Der alte Eigenbau-Mittelteil (Face-Atlas,
# Template-Bank, Face-Classifier, template-MSE-Yaw, Eigenbau-Backprojection) ist
# bewusst entfernt. Sobald GDRNPP angebunden ist, liefert estimate_poses() pro
# Detektion einen Eintrag mit:
#   {instance_id, part, face_name, R_world(9 floats), t_world(3 floats),
#    confidence(0..1), bbox_2d(4 ints), upright(bool)}
# und build_pose_result() giesst das schema-valide in pose_result.json.
def estimate_poses(rgb, dets, warn=print):
    """6D-Pose pro Detektion via GDRNPP (BOP-SOTA). STUB bis W2/W3-Anbindung.

    Erwartete Eingabe: das RGB + die OBB-Detektionen aus Stufe 1 (Crop + OBB-
    Winkel als Prior). Erwartete Ausgabe: Liste 'aligned' (s. Format oben).
    Bis GDRNPP läuft -> leere Liste (keine Posen)."""
    warn(f"[pose] GDRNPP-Stub: {len(dets)} Detektion(en), 0 Posen "
         f"(BOP-Pipeline wird in W2/W3 angebunden — ADR-018)")
    return []


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
def run(image, out_path, warn=print):
    """Ganze Pipeline für ein Bild. Schreibt schema-valides pose_result.json.

    Detektor (Stufe 1) -> GDRNPP (Stufe 3, Stub) -> pose_result (Stufe 4). Solange
    GDRNPP nicht angebunden ist, sind 'results' leer — der Contract bleibt valide."""
    rgb, dets = detections_for(image, warn=warn)
    aligned = estimate_poses(rgb, dets, warn=warn)      # GDRNPP (Stub bis W2/W3)
    doc = build_pose_result(image, aligned, table_origin=TABLE_ORIGIN_SCENE)
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
