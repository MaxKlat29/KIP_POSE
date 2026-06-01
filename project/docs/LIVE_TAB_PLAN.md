# /KIP Live-Tab — Architektur & Plan

On-demand Live-Feed der **echten** Zellen-Anlage (Zivid-Kamera am Jetson `lara5`,
172.22.192.166, KIT-wbk) + Aufnahme+Inferenz der aktuellen Szene, im /KIP-Frontend.

> **Leitplanke (Max):** Am Jetson wird **erstmal NICHTS angefasst/zerschossen**.
> Live-Feed braucht zwingend Capture-Code **auf** dem Jetson (nur er hat die Kamera) →
> dieser Teil ist als **Deliverable geplant + geschrieben**, aber **du** deployst/startest
> ihn (mit Kamera dran). Alles andere (Workstation + FE) ist gebaut.

## Datenfluss
```
Browser ──(https)──► max-utils.com/KIP ──► kip_server (Workstation, :8077)
                                              │  /api/live/{status,preview,capture_infer,frame}
                                              ▼
                                   [scoped KIT-VPN auf der Workstation]
                                   NUR Route 172.22.192.166/32 durch den Tunnel
                                   (kein Full-Tunnel, kein DNS-Hijack)
                                              ▼
                              Jetson live_server :8090  (jetson_live/live_server.py)
                                   /preview        → Zivid 2D-Frame (JPEG)
                                   /capture_infer  → Zivid 2D+3D + YOLO (+3D-Match-Hook)
                                              ▼
                                        Zivid-Kamera + ~/DetectionPipeline
```

## Schichten & Status

### C · Unsere Seite — ✅ GEBAUT (kein Jetson-Touch)
- **FE 3. Tab „Live (Anlage)"** (`frontend/kip.html` + `src/live.js` + Hook in `src/kip.js`):
  - „Live verbinden" → `/api/live/status`; wenn erreichbar: Vorschau-Polling (~0.9 s) in die
    **Vorschau-Box unten** mit **● LIVE**-Badge, „Aufnehmen+Inferieren" wird aktiv. On-demand —
    stoppt beim Trennen / Tab-Wechsel. **Kein Dauer-Stream.**
  - „Aufnehmen + Inferieren" → `/api/live/capture_infer` → aktuelle Szene gespeichert + inferiert,
    Ergebnis (Bild + Detektionen) in der Vorschau.
- **`kip_server.py` `/api/live/*`** — Proxy zur Jetson-`LIVE_JETSON_URL` (default
  `http://172.22.192.166:8090`). Nicht erreichbar → saubere **503** (kein Crash, gdrnpp-Pfad unberührt).

### B · Bridge — Workstation joins KIT-VPN (scoped), ⏳ SETUP-SCRIPTS GEBAUT
- `box_src/kit_vpn_scoped_connect.sh` / `kit_vpn_scoped_disconnect.sh`:
  on-demand KIT-VPN auf der Workstation, **`--route-nopull` + route-up-Hook → nur die eine
  Jetson-Host-Route**. Restlicher Box-Traffic (Civion, Tailscale, kip) bleibt unangetastet.
- Einmalig auf der Box: `sudo apt-get install -y openvpn`; `kit.ovpn` (von scc.kit.edu) +
  `auth.txt` (**Zeile1 = U-Kürzel `ulumu`, NICHT die E-Mail** — sonst AUTH_FAILED) nach
  `/mnt/data/kitvpn/`.

### A · Jetson — 📦 DELIVERABLE, NICHT DEPLOYT (braucht dein Go + Kamera)
- `jetson_live/live_server.py` (+ `jetson_live/README.md`): lightweight stdlib-HTTP,
  spiegelt `CapturePicture.py`, nutzt vorhandenes `zivid`+`ultralytics`. Liegt in eigenem
  Ordner, **ändert `~/DetectionPipeline` nicht**. Start: `python3 ~/kip_live/live_server.py`.
- **Vor echtem Lauf zu verifizieren (am Jetson):** exakter Zivid-2.17-Capture-Aufruf (an
  `CapturePicture.py` angleichen) + 3D-CAD-Match-Hook (`matching_3d/main.py`) einhängen.

## Offene physische Voraussetzung
- ⚠️ **Zivid-Kamera war beim Recon NICHT in `lsusb`** → muss physisch angeschlossen + an sein,
  sonst kein Feed. `/api/live/status` meldet `camera_connected:false` entsprechend.

## Inbetriebnahme-Reihenfolge (wenn Kamera dran)
1. Workstation: openvpn + `kit.ovpn`/`auth.txt` ablegen → `sudo box_src/kit_vpn_scoped_connect.sh` → Jetson pingbar.
2. Jetson: `python3 ~/kip_live/live_server.py` starten (Zivid-Call + 3D-Hook ggf. angleichen).
3. Browser: /KIP → Tab „Live" → „Live verbinden" → Vorschau + „Aufnehmen+Inferieren".
4. Danach: `kit_vpn_scoped_disconnect.sh` + live_server stoppen (on-demand, nichts bleibt laufen).
