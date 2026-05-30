# POSE — 6D-Pose-Pipeline (BOP-SOTA-Stack)

Aus einem 2D-Top-Down-Bild einer Montagezelle die **6D-Pose** der Metallteile
schätzen und sie als echtes CAD im **3D-Viewer** an ihrer Pose rendern.

## Webservice (live: <https://max-utils.com/KIP/>)

Production-Webservice, on-demand auf der GPU-Workstation. Public über
Cloudflare-Tunnel an [`max-utils.com/KIP`](https://max-utils.com/KIP/), zwei Ansichten:

| Ansicht | Funktion |
|---|---|
| **Reales Foto** | Zivid-Bild hochladen → YOLOv8-OBB-Detektor → warmer GDRNPP-Worker → 3D-Pose-Render |
| **Simulation** | Klick → **echte Live-Isaac-Generation** (Render + GT + Boxen + GDRNPP, ~80&nbsp;s pro Klick) |

**Features:**
- Modellauswahl-Dropdown in der Kopfzeile (GDRNPP aktiv, 3 Slots für späteren Architektur-Vergleich)
- Animierte Ladebalken mit **echten Prozenten + Phasen-Flags** (Real: 5 Phasen,
  Sim: 6 Phasen inkl. Isaac-Booten und -Render)
- Bounding-Boxen-Toggle (PiP unten rechts, klassen-farbig: Anker_Kurz Orange,
  Anker_Lang Magenta, Zahnrad Cyan, mit Confidence-Label)
- PiP-Fullscreen (⛶) — Vorschau auf ca. 4-fache Fläche aufziehen, Esc bricht ab
- Foto-View (Kamera springt auf die Aufnahmeperspektive, danach frei beweglich)
- **View+Zoom persistieren** in localStorage — Reload landet exakt dort wo man war
- **Ansicht zurücksetzen**-Button in der Top-Bar (fittet neu auf die Szene)
- **Boden-Physik in zwei Stufen**: Backend `planar_z_snap` zieht auf Tisch-Ebene
  (Lift bei dz>0 immer, Senken nur bis 10 cm — Held-Teile schweben lassen),
  Frontend `groundClamp` macht final per-Teil Raycast gegen die echte Cell-
  Geometrie (Tray-Plateaus + Maschinen-Blöcke) — keine Clipping-Restdrift mehr
- Sim ist **immer live**: kein Pool-Cache, jeder Klick triggert echte Isaac-
  Generation auf der GPU mit neuem Seed, neuem Render, neuen Labels und
  neuem GDRNPP-Inferenz-Lauf
- High-Resolution-CAD-Meshes (Original-BOP-PLYs für alle 6 Teile, 1:1 die
  Geometrie auf der GDRNPP trainiert wurde)
- Metriken-Panel zeigt die Final-AR aller trainierten Modelle

### Modell-AR (Final, best-by-val)

| Objekt | Final AR | Checkpoint |
|---|---|---|
| Anker_Kurz | **0.870** | `model_0112949.pth` (ep ~150) |
| Anker_Lang | **0.907** | `model_0120959.pth` (ep ~160) |
| Zahnrad | **0.838** | `model_0130399.pth` (ep ~165) |
| **Durchschnitt** | **0.872** | — |

### Architektur

```
Browser ─https─▶ Cloudflare-Tunnel ─▶ cloudflared (ai-desk Raspi)
                ─▶ Caddy :8000 @kip ─▶ reverse_proxy 100.85.216.95:8077
                ─▶ kip_server.py (FastAPI, port 8077, systemd-Service "kip-server")
                    ├─ /api/health, /api/metrics
                    ├─ /api/sim/generate_async      ← echte Live-Isaac-Pipeline
                    │   ├─ subprocess gen_sdg_arm_visible.py (isaacsim-venv)
                    │   ├─ subprocess isaac_to_bop.py (gdrnpp-venv)
                    │   ├─ Detektor (YOLOv8-OBB)
                    │   └─ Worker-Call → Pose-Inferenz
                    ├─ /api/sim/live_rgb/<id>, /api/sim/live_boxes/<id>
                    ├─ /api/sim/job/<id>            ← Phase-Poll-Endpoint
                    └─ /api/real/{infer_async, job/<id>, result/<id>, rgb/<id>, boxes/<id>}
                       └─ ruft den Multi-Objekt-Worker:
                           kip_infer_worker.py (port 8078, systemd-Service "kip-worker")
                           hält alle 3 GDRNPP-Modelle warm im VRAM (~2.3 GB)
```

Async-Job-Endpoints liefern Phasen-Fortschritt — der Browser pollt
`/api/{real,sim}/job/<id>` ca. 3× pro Sekunde und aktualisiert den Ladebalken.

### Quick-Start (CLI)

```bash
# Real-Upload (Zivid-Format) — async + Phase-Poll
JOB=$(curl -sF "image=@scene.png" https://max-utils.com/KIP/api/real/infer_async | jq -r .job)
while :; do
  ST=$(curl -s https://max-utils.com/KIP/api/real/job/$JOB)
  echo "$ST"
  [ $(echo "$ST" | jq -r .pct) -ge 100 ] && break
  sleep 1
done
curl -s "https://max-utils.com/KIP/api/real/result/$JOB"

# Sim-Szene LIVE generieren (Isaac + Detektor + GDRNPP, ~80 s)
JOB=$(curl -s https://max-utils.com/KIP/api/sim/generate_async | jq -r .job)
# ... pollen wie oben, dann GET /api/sim/job_result/$JOB
```

Aus dem `infer.ipynb` heraus genauso — das Notebook ruft den Webservice
(POST `/api/real/infer` + GET `/api/real/result/<job>`) und zeigt die Posen-Tabelle
inline an. Kein lokales GDRNPP-Setup nötig.

### Persistenz

Beide Komponenten laufen als systemd-Services auf der Workstation:
- `kip-server.service` — Webserver, Auto-Restart bei Crash
- `kip-worker.service` — Multi-Objekt-Worker, lädt alle drei Checkpoints beim Start (~4 min)

Start/Stop/Status (auf der Box):
```bash
sudo systemctl restart kip-server.service kip-worker.service
sudo systemctl status  kip-server.service kip-worker.service
journalctl -u kip-server -f          # Logs live
```

---

## Reproduktion (Notebooks)

Zwei Notebooks, beide **top-to-bottom in einem Rutsch** lauffähig:

| Notebook | Zweck |
|----------|-------|
| **`setup.ipynb`** | „Training reproduzieren" — CAD + Isaac-SDG → BOP-Daten → trainierte Modelle (auf der GPU-Box) |
| **`infer.ipynb`** | „Inferenz + 3D-Viewer" — Bild rein → 3D-Pose via Webservice |

## Local-Setup (für die nächste Person)

Das Projekt läuft an drei Stellen:

1. **GPU-Workstation** (`max@100.85.216.95`, RTX 3090, Tailscale):
   - `kip-server.service` + `kip-worker.service` (systemd)
   - Isaac Sim 5.1 in `/mnt/data/isaacsim-venv/`
   - GDRNPP in `/mnt/data/bop/repos/gdrnpp/` mit `gdrnpp-venv`
   - YOLOv8-OBB-Detektor in `/mnt/data/bop/repos/...` mit `train-venv`
   - BOP-Konverter + SDG-Skripte unter `/mnt/data/kip_pose/box_src/`
   - Datasets unter `/mnt/data/kip_pose/project/bop/pose_isaac/`
2. **ai-desk Raspberry Pi** (Caddy + Cloudflared für public Routing zu max-utils.com/KIP)
3. **Laptop / Dev-Rechner** (dieses Repo) — Notebooks + Standalone-Tests, kein lokales GPU/Training nötig

### Vom Laptop loslegen

```bash
# Repo klonen
git clone <repo> POSE && cd POSE

# Webservice-Health prüfen (Tailscale: brauchst Workstation-Zugang via VPN)
curl http://100.85.216.95:8077/api/health
# oder public (geht ohne VPN):
curl https://max-utils.com/KIP/api/health

# Eingabebild + Pipeline test
cd project
python -m venv .posevenv && source .posevenv/bin/activate
pip install requests Pillow numpy
jupyter notebook infer.ipynb       # ruft den Webservice, zeigt Posen-Tabelle
```

Das `infer.ipynb` braucht **keinen lokalen GDRNPP-Stack** — alle schweren Sachen
laufen auf der Workstation. Wer lokal die Pipeline reproduzieren will, geht über
`setup.ipynb` (lädt SSH + Isaac + Training; mehrere Tage GPU-Zeit).

### Auf der Workstation deployen

```bash
ssh max@100.85.216.95
# Backend-Update (kip_server.py)
sudo systemctl restart kip-server.service
# Worker-Update (kip_infer_worker.py) — lädt alle 3 Checkpoints, ~4 min Warm-Load
sudo systemctl restart kip-worker.service
# Logs live
journalctl -u kip-server -f
journalctl -u kip-worker -f
```

Frontend-Files (`project/frontend/{kip.html, src/*, assets/*}`) werden direkt von
Caddy statisch ausgeliefert; ein `scp` reicht, Cache-Control ist `no-store` →
keine CF-Purge nötig.

### Konventionen

- **Welt-Frame:** Z-up, Ursprung = Tisch-Nullpunkt, Einheit Meter
- **Rotation:** `world = R @ body`, R im pose_result row-major 9-flat
- **BOP-Boundary:** `cam_t_m2c` in **Millimeter** (Worker konvertiert intern)
- **Snap:** `planar_z_snap` liftet immer (negative dz nur bis Guard `max_snap_m`)
- **Frontend-Clamp:** finaler Raycast pro Teil gegen `cellGroup` für lokale
  Tisch-Höhe (Tray-Plateaus, Maschinen-Blöcke), nie senken

## Architektur (ADR-018 — Pivot auf BOP-SOTA)

Der hausgemachte Mittelteil (Face-Atlas / Template-Bank) wurde verworfen
(unbrauchbar auf realen Metallteilen) und durch den **BOP-Benchmark-SOTA-Stack**
ersetzt. **RGB-only, hart** — keine Depth in den Pose-Netzen.

```
CAD (GLB) ─gen_models_info─▶ models/*.ply (mm) + models_info.json (Symmetrie)
Isaac-Zelle ─gen_sdg_arm_visible─▶ Top-Down-RGB MIT LARA5-Arm + gedroppte Teile
            ─isaac_to_bop / convert_full_to_bop─▶ BOP-Datensatz (train_pbr / val)
            ─train_detector_armvis─▶ detector.pt (YOLOv8-OBB, arm-sichtbar)
            ─obb_to_aabb_dets─▶ BOP-Detektionen (OBB→AABB-Bridge)
            ─train_chain (GDRNPP)─▶ per-Objekt GDRNPP-Checkpoints (RGB-only)

Inferenz:  Bild ─Detektor(OBB)─▶ OBB→AABB-Crops ─GDRNPP─▶ (R_m2c, t_m2c) [BOP, mm]
                ─bop_adapter §3─▶ Welt-Pose ─▶ pose_result.json ─▶ Three.js-Viewer
```

**Eingefrorene Konventionen:**
- Z-up Welt · `world = R @ body` (Spaltenkonvention) · Ursprung = Tisch-Nullpunkt · Einheit Meter.
- `obj_id` (1-basiert, Single-Source `bop_adapter.OBJ_ID_TO_PART`):
  `1=Anker_Kurz 2=Anker_Lang 3=Buerstenhalter_2polig 4=Getriebegehaeuse_typ4 5=Ringmagnet 6=Zahnrad`.
- Detektor-Klasse (0-basiert) `+1` = `obj_id`.
- Symmetrie: Anker/Ring = continuous um Y, Zahnrad = discrete C_7 (analytischer Fix des 120°/91°-Problems).
- Contract: `pose_result.schema.json` (ADR-017, unverändert → Viewer entkoppelt).

## Zwei Workflows

### 1 · Training reproduzieren → `setup.ipynb`

Notebook oben→unten durchlaufen. Es **ruft die `box_src/`-Skripte über die
GPU-Box auf** (kein Code-Duplikat) und erklärt jede Stufe:
Config + Box-Verbindung → Isaac arm-sichtbare SDG → Isaac→BOP + train/val-Split +
`bop_toolkit`-Validierung + GT-Overlay → `models_info` Symmetrie → Detektor-Retrain
mit Arm → GDRNPP-RGB-Training. Schwere Jobs laufen als `nohup` auf der Box; das
Notebook startet + pollt sie (eine RTX 3090 → Stufen sequenziell, via `train_chain.sh`).

### 2 · Inferenz + 3D-Viewer → `infer.ipynb`

Notebook oben→unten durchlaufen. Lädt ein Bild aus `input/` (oder erzeugt ein
synthetisches Demo, wenn leer), läuft Detektor → GDRNPP → `bop_adapter` →
schema-valides `pose_result.json`, zeigt Detektions-Overlay + Pose-Tabelle und
**startet in der letzten Zelle den 3D-Viewer** (Server + Link + IFrame, ein Klick).

**Produktion:** Alle drei trainierten GDRNPP-Modelle (`anker_kurz`, `anker_lang`,
`zahnrad`) liegen warm im `kip-worker.service` auf der Workstation. Das Notebook
schickt das Bild an den Webservice (`/api/real/infer`) — Detektor + Worker +
Welt-Transform + Boden-Snap laufen in ~4 s end-to-end. Kein lokaler Checkpoint-
Pfad noetig.

## Voraussetzungen

| Wo | Was |
|---|---|
| **Laptop (Inferenz via Webservice)** | `requests`, `numpy`, `Pillow`. Kein lokaler GDRNPP-Stack noetig. |
| **GPU-Workstation (`max@100.85.216.95`, Tailscale, RTX 3090)** | `isaacsim-venv` (SDG-Render), `train-venv` (Detektor), `bop-venv` (Konverter/Eval), `gdrnpp-venv` (Worker). Konfig in `.env`. |

Box-Setup im Detail: `../box_src/BOP_SETUP.md`. Eval: `../box_src/EVAL_BOP.md`.

## Datei-Übersicht

| Pfad | Inhalt |
|------|--------|
| `setup.ipynb` | Training reproduzieren (ruft `box_src/`-Skripte über die Box auf) |
| `infer.ipynb` | Inferenz + 3D-Viewer (importiert `e2e_infer`/`bop_adapter`) |
| `kip_server.py` | FastAPI-Webservice (port 8077) — alle `/api/`-Endpoints fuer Real + Sim |
| `kip_infer_worker.py` | stdlib-http.server-Daemon (port 8078) — alle 3 GDRNPP-Modelle warm im VRAM |
| `e2e_infer.py` | Standalone-Pipeline (alt) — Produktion laeuft ueber den Webservice |
| `bop_adapter.py` | BOP(R/t cam, mm) → `pose_result` (Welt, m). Symmetrie-Kanonisierung + face/upright + `planar_z_snap`. Getestet |
| `pose_result.schema.json` | Eingefrorener Output-Contract (ADR-017) |
| `frontend/` | Three.js-3D-Viewer + `assets/cell_web.glb` (web-leichte Anlage) + `assets/parts/ply/*.ply` (alle 6 hi-res Original-BOP-Meshes) |
| `cad_input/` | CAD-Eingang (Teile-USDs + Zellen-Szene) |
| `input/` | Eingabe-Szenen (RGB + optional `bbox_2d_*.json` / `scene_camera.json`) |
| `temp/` | Scratch: `pose_result.json`, Overlays (gitignored) |
| `tests/` | `test_bop_adapter.py` + `test_e2e_mock.py` (21 Tests) |
| `../box_src/` | GPU-Box-Pipeline: SDG-Gen, Isaac→BOP-Konverter, Detektor-/GDRNPP-Training, Eval, `gpu_run.sh`-Harness |

## Tests

```bash
cd project && python3 -m pytest tests/ -q      # 21 passed
```

Deckt den BOP→Welt-Adapter (Transform-Kette, Symmetrie-Kanonisierung,
face/upright) und die E2E-Kette ueber einen deterministischen Stub-Estimator ab
(schema-valides `pose_result`).

## Status & Optimierungen

**Angewandt (sicher):** Teile-Mapping zentralisiert (`available_parts()` zieht
aus `bop_adapter.OBJ_ID_TO_PART` statt driftender Hardcoded-Liste); robuste
Server-/Box-Helfer in den Notebooks; synthetischer Demo-Fallback für sofortige
Lauffähigkeit.

**Empfohlen (nächste Iterationen):** mehr SDG-Daten + stärkere Domain-Randomization
(wirksamster Sim2Real-Hebel auf texturlosem Metall); längere GDRNPP-Schedules;
RGB-vs-RGB-D-Ablation mit confidence-gefilterter Zivid-Depth (+10–15 AR laut
BOP-Industrial-Evidenz, RGB bleibt Default); Eval-Automatisierung (nach jedem
Checkpoint `eval_bop.sh` triggern + Report archivieren); Zahnrad-N (C_7) gegen
das CAD gegenchecken.

## Dokumentation (`docs/`)

| Doc | Inhalt |
|-----|--------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline-Stages, Datenfluss, Methodenwahl |
| [`docs/ADD_NEW_PART.md`](docs/ADD_NEW_PART.md) | **Neues CAD-Teil aufnehmen** — Generalisierung für Weiterverwendung |
| [`docs/EVAL.md`](docs/EVAL.md) | BOP-Metriken (AR / ADD / ADI) + `eval_bop` nutzen |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Von Null reproduzieren (lokal + Box-venvs, pip-Pins) |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | Methoden-Zitate (GDRNPP, CNOS, GigaPose, MegaPose, BOP) |
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Akademischer Kurzbericht (Uni-tauglich) |

## Lizenz

Eigener Code: **MIT** ([`../LICENSE`](../LICENSE)). Dritt-Komponenten behalten ihre eigene
Lizenz — siehe [`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md). ⚠️ Der **YOLOv8-Detektor
ist AGPL-3.0** (relevant nur für geschlossene Weitergabe des Detektors; der BOP-Pose-Stack
ist MIT/Apache und akzeptiert beliebige Detektoren).
