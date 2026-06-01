# `jetson_live/` — On-Demand Live-Feed + Capture+Infer auf dem Zellen-Jetson

Lightweight HTTP-Server, der der KIP-Web-App (über die Workstation, scoped KIT-VPN)
einen **on-demand** Zivid-Live-Vorschau-Feed + Aufnahme+Inferenz der echten Anlage liefert.

> **Status: DELIVERABLE / NICHT DEPLOYT.** Wird **manuell** auf dem Jetson gestartet
> (von Max, wenn die Zivid-Kamera dran + an ist). Die bestehende `~/DetectionPipeline`
> wird **nicht** verändert.

## Was es tut
- `GET /health` → `{ok, camera_connected}` (Kamera-Check ohne Capture)
- `GET /preview` → JPEG eines frischen Zivid-2D-Frames (für den Live-Loop; Zivid ist
  3D-Struct-Light = kein Video → „live" = schnelle wiederholte 2D-Captures)
- `POST /capture_infer` → nimmt 2D+3D auf, speichert in die bestehenden
  `~/DetectionPipeline/inferenz/{img,zdf}`-Pfade (wie `CapturePicture.py`), YOLO-2D-Detektion
  (`run/train1/weights/best.pt`), liefert JSON + annotiertes Bild
- `GET /frame/<name>` → gespeichertes Ergebnisbild

## Start (auf dem Jetson, manuell)
```bash
mkdir -p ~/kip_live && cp live_server.py ~/kip_live/
python3 ~/kip_live/live_server.py          # bindet 0.0.0.0:8090 (nur via KIT-Netz/VPN)
# Stoppen: Ctrl-C. NICHT als Dauerdienst (systemd) einrichten — on-demand only.
```

## ⚠️ Vor dem ersten echten Lauf zu verifizieren (am Jetson, gegen die Kamera)
1. **Zivid-Capture-Aufruf:** `_capture_2d_jpeg` / `_capture_scene` spiegeln das Muster aus
   `~/DetectionPipeline/CapturePicture.py`. Falls SDK 2.17 dort `capture_2d(...)` statt
   `capture(settings_2d)` o.ä. nutzt → die 1–2 Zeilen 1:1 an `CapturePicture.py` angleichen.
2. **3D-CAD-Match-Hook:** `capture_infer` macht aktuell die **2D-Detektion** sicher. Der
   **3D-Pose-Schritt** (`matching_3d/main.py` bzw. `output_csv.py`) ist bewusst **nicht blind
   verdrahtet** — dessen CLI/CAD-`.stl`-Pfade müssen am Jetson bestätigt werden. Dann dort als
   Subprocess/Import einhängen und ins JSON-Ergebnis mergen.
3. **Kamera-Singleton:** Zugriffe sind via `_LOCK` serialisiert. Falls die Pipeline/ein
   anderer Prozess die Kamera schon hält → Capture schlägt fehl (sauber als 503 gemeldet).

## Netzwerk-Pfad (wie die Web-App hier landet)
```
Browser → max-utils.com/KIP → kip_server (Workstation) → /api/live/* Proxy
        → [scoped KIT-VPN, NUR Route zu 172.22.192.166] → Jetson live_server:8090
```
Die Workstation baut das scoped VPN mit `box_src/kit_vpn_scoped_connect.sh` (nur Jetson-Route,
kein Full-Tunnel). Voller Plan: `project/docs/LIVE_TAB_PLAN.md`.
