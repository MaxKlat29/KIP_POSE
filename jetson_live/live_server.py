#!/usr/bin/env python3
"""live_server.py — lightweight On-Demand Live-Feed + Capture+Infer fuer den
Jetson-Zellen-Controller (lara5, Adresse via LIVE_JETSON_URL / JETSON_IP).

ZWECK: serviert der KIP-Web-App (ueber die Workstation, scoped KIT-VPN) einen
on-demand Zivid-Vorschau-Feed + auf Knopfdruck eine Aufnahme der aktuellen Szene
+ Inferenz. KEIN Dauer-Stream — laeuft nur solange dieser Server laeuft, und der
wird MANUELL gestartet (von Max), nicht automatisch.

DESIGN-PRINZIP (Max-Direktive): die bestehende ~/DetectionPipeline wird NICHT
veraendert. Dieser Server liegt in einem EIGENEN Ordner (~/kip_live/ oder wo immer),
nutzt nur (a) das `zivid`-SDK wie ~/DetectionPipeline/CapturePicture.py es tut und
(b) ruft die bestehenden Pipeline-Skripte read-only auf. stdlib-HTTP, keine neuen
schweren Deps (zivid + ultralytics + opencv sind auf dem Jetson bereits da).

ENDPUNKTE (binden an 0.0.0.0:8090 — nur via KIT-Netz/VPN erreichbar):
  GET  /health         -> {"ok":true,"camera_connected":bool}
  GET  /preview        -> JPEG eines frischen 2D-Zivid-Frames (fuer den Live-Loop)
  POST /capture_infer  -> nimmt 2D+3D auf, speichert die Szene, inferiert (YOLO 2D;
                          3D-CAD-Match optional/Hook), liefert JSON-Ergebnis
  GET  /frame/<name>   -> liefert ein gespeichertes Ergebnis-/Annotations-Bild

START (manuell auf dem Jetson, Kamera angeschlossen + an):
  python3 ~/kip_live/live_server.py            # Port 8090
  # Stoppen: Ctrl-C bzw. den Prozess killen. NICHT als Dauerdienst einrichten.

⚠️ ZIVID-CAPTURE-AUFRUF: die `_capture_*`-Funktionen spiegeln das Muster aus
   ~/DetectionPipeline/CapturePicture.py. Falls die SDK-2.17-Methodennamen dort
   minimal abweichen (capture vs capture_2d), 1:1 an CapturePicture.py angleichen —
   das ist der einzige Punkt, der gegen die echte Kamera verifiziert werden muss.
"""
from __future__ import annotations

import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "0.0.0.0", 8090

# Pfade der bestehenden Pipeline (NUR lesen / Capture-Targets schreiben).
PIPELINE_DIR = Path.home() / "DetectionPipeline"
INFERENZ_IMG = PIPELINE_DIR / "inferenz" / "img" / "inferenzbild.png"
INFERENZ_ZDF = PIPELINE_DIR / "inferenz" / "zdf" / "inferenz.zdf"
YOLO_WEIGHTS = PIPELINE_DIR / "run" / "train1" / "weights" / "best.pt"
RESULT_DIR = Path.home() / "kip_live" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()   # Zivid-Kamera ist single-access -> Zugriffe serialisieren


# ── Zivid-Helfer (Muster aus CapturePicture.py) ──────────────────────────────
def _camera_connected() -> bool:
    """Lightweight-Check ob eine Zivid-Kamera da ist — OHNE Capture."""
    try:
        import zivid
        app = zivid.Application()
        cams = app.cameras()
        return any(str(c.state.status).lower().find("available") >= 0 or c.state.connected for c in cams) if cams else False
    except Exception:
        return False


def _capture_2d_jpeg() -> bytes:
    """Schnelles 2D-Frame -> JPEG-Bytes (fuer die Live-Vorschau)."""
    import zivid
    import cv2
    import numpy as np
    with _LOCK:
        app = zivid.Application()
        camera = app.connect_camera()
        settings_2d = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
        # SDK-2.17: 2D-Capture. Bei Abweichung an CapturePicture.py angleichen.
        frame_2d = camera.capture(settings_2d)
        rgba = frame_2d.image_rgba().copy_data()        # HxWx4 (numpy)
        bgr = cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG-Encode fehlgeschlagen")
        return buf.tobytes()


def _capture_scene():
    """Nimmt 2D+3D auf, speichert PNG + ZDF in die Pipeline-Inferenz-Pfade
    (genau wie CapturePicture.py) -> bereit fuer die bestehende Pipeline."""
    import zivid
    import cv2
    import numpy as np
    INFERENZ_IMG.parent.mkdir(parents=True, exist_ok=True)
    INFERENZ_ZDF.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        app = zivid.Application()
        camera = app.connect_camera()
        settings = zivid.Settings(
            acquisitions=[zivid.Settings.Acquisition()],
            color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
        )
        frame = camera.capture(settings)
        frame.save(str(INFERENZ_ZDF))                     # 3D-Punktwolke (.zdf)
        rgba = frame.frame_2d().image_rgba().copy_data()
        bgr = cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(INFERENZ_IMG), bgr)               # 2D-Farbbild (.png)
    return INFERENZ_IMG, INFERENZ_ZDF


def _infer_2d(img_path: Path):
    """YOLO-2D-Detektion auf dem aufgenommenen Bild (wie predict.py). Liefert
    (n_detections, results[], annotated_path)."""
    from ultralytics import YOLO
    import cv2
    model = YOLO(str(YOLO_WEIGHTS))
    res = model.predict(source=str(img_path), conf=0.55, imgsz=800, verbose=False)[0]
    results = []
    for b in res.boxes:
        results.append({
            "cls": int(b.cls[0]), "conf": float(b.conf[0]),
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            "label": res.names.get(int(b.cls[0]), str(int(b.cls[0]))),
        })
    annotated = RESULT_DIR / "annotated.png"
    cv2.imwrite(str(annotated), res.plot())
    return len(results), results, annotated

# ── 3D-CAD-Match (Hook): hier die bestehende matching_3d/main.py bzw.
#    output_csv.py einhaengen, sobald gegen die echte Kamera verifiziert.
#    Bewusst NICHT blind verdrahtet (CLI/CAD-Pfade muessen am Jetson bestaetigt
#    werden) — siehe jetson_live/README.md.


# ── HTTP-Handler ─────────────────────────────────────────────────────────────
class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def _bin(self, code, data, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # leiser
        pass

    def do_GET(self):
        try:
            if self.path == "/health":
                self._json(200, {"ok": True, "camera_connected": _camera_connected()})
            elif self.path.startswith("/preview"):
                self._bin(200, _capture_2d_jpeg(), "image/jpeg")
            elif self.path.startswith("/frame/"):
                name = self.path.split("/frame/", 1)[1].split("?")[0]
                p = (RESULT_DIR / name).resolve()
                if RESULT_DIR.resolve() not in p.parents or not p.exists():
                    self._json(404, {"error": "not found"}); return
                self._bin(200, p.read_bytes(), "image/png")
            else:
                self._json(404, {"error": "unknown path"})
        except Exception as e:
            traceback.print_exc()
            self._json(503, {"error": str(e)})

    def do_POST(self):
        try:
            if self.path.startswith("/capture_infer"):
                img, zdf = _capture_scene()
                n, results, annotated = _infer_2d(img)
                self._json(200, {
                    "ok": True, "n_detections": n, "results": results,
                    "saved_as": str(img.name), "zdf": str(zdf.name),
                    "image_url": f"live/frame/{annotated.name}",
                    "note": "2D-Detektion. 3D-CAD-Match-Hook siehe README (noch nicht verdrahtet).",
                })
            else:
                self._json(404, {"error": "unknown path"})
        except Exception as e:
            traceback.print_exc()
            self._json(503, {"error": str(e)})


def main():
    srv = ThreadingHTTPServer((HOST, PORT), H)
    print(f"[live_server] on-demand auf http://{HOST}:{PORT} — Ctrl-C zum Stoppen.")
    print(f"[live_server] Kamera erkannt: {_camera_connected()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[live_server] gestoppt.")


if __name__ == "__main__":
    main()
