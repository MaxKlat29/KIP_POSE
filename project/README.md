# POSE — 6D-Pose-Pipeline (BOP-SOTA-Stack)

Aus einem 2D-Top-Down-Bild einer Montagezelle die **6D-Pose** der Metallteile
schätzen und sie als echtes CAD im **3D-Viewer** an ihrer Pose rendern.

## Webservice (live: <https://max-utils.com/KIP/>)

Production-Webservice, on-demand auf der GPU-Workstation. Public über
Cloudflare-Tunnel an [`max-utils.com/KIP`](https://max-utils.com/KIP/), zwei Ansichten:

| Ansicht | Funktion |
|---|---|
| **Reales Foto** | Zivid-Bild hochladen → Detektor + GDRNPP-Worker → 3D-Pose-Render aller erkannten Teile |
| **Simulation** | Val-Szene wählen → on-demand Live-Inferenz → Ground-Truth (blau) gegen Schätzung (rot) |

**Features:**
- Modellauswahl-Dropdown in der Kopfzeile (GDRNPP aktiv, 3 Slots für späteren Architektur-Vergleich)
- Animierte Ladebalken mit **echten Prozenten + Phasen-Flags** („Detektor 20 %",
  „GDRNPP-Inferenz 55 %", „BOP→Welt + Snap 85 %", „Fertig 100 %")
- Bounding-Boxen-Toggle (PiP unten rechts, schaltet zwischen Roh-RGB und Detektor-Boxen)
- Foto-View (Kamera springt auf die Aufnahmeperspektive, danach frei beweglich)
- Boden-Physik (`planar_z_snap` zieht inferred Posen auf die Tischebene)
- Szenenwechsel per Dropdown lädt direkt
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
                    ├─ /api/health, /api/metrics, /api/sim/{scenes,rgb,boxes,infer_async,job/<id>}
                    └─ /api/real/{infer_async, job/<id>, result/<id>, rgb/<id>}
                       └─ ruft den Multi-Objekt-Worker:
                           kip_infer_worker.py (port 8078, systemd-Service "kip-worker")
                           hält alle 3 GDRNPP-Modelle warm im VRAM (~2.3 GB)
```

Async-Job-Endpoints liefern Phasen-Fortschritt — der Browser pollt
`/api/{real,sim}/job/<id>` ca. 3× pro Sekunde und aktualisiert den Ladebalken.

### Quick-Start (CLI)

```bash
# Real-Upload (Zivid-Format) -> 3D-Posen
curl -F "image=@scene.png" https://max-utils.com/KIP/api/real/infer

# Sim-Szene live inferieren
curl https://max-utils.com/KIP/api/sim/infer?scene=7
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

### Multi-Stage Viewer (Pipeline-Zwischenergebnisse)

Der 3D-Viewer hat oben eine **Stage-Leiste**: `Raw | +Z-Snap | +M1 | +M2 | +TTA |
Final (Auto-Best)`. Pro Stufe liegt ein eigenes `temp/pose_result_<stage>.json`
(erzeugt von `box_src/make_stages.py` nach `e2e_finish`). Click → Viewer lädt
diese Stufe → man sieht direkt, was jeder Refinement-Lever am Endergebnis
verändert (Translation-Sprung durch Z-Snap, Rotation durch M2, etc.). Fehlt
eine Stufen-Datei (Pipeline nicht durchgelaufen), schaltet die Stage einfach
auf einen leeren Scene-Fallback — der Viewer bricht nicht.

## Viewer-Start — der EINE Befehl

```bash
python project/e2e_infer.py --image project/input/scene_0000.png --serve
```

Erzeugt `pose_result.json` **und** öffnet den 3D-Viewer (`http://127.0.0.1:8000/frontend/`).
Mit echtem Checkpoint: `--checkpoint /pfad/zu/gdrnpp.pth` ergänzen. (Im
`infer.ipynb` macht das die letzte Zelle.)

## Voraussetzungen

| | |
|---|---|
| **Lokal (Inferenz + Viewer)** | `numpy`, `Pillow` (Pflicht). `jsonschema` optional (Bonus-Schema-Gate). `torch`+`ultralytics` nur für echten Detektor-Checkpoint (sonst Fallback). |
| **GPU-Box (Training)** | `max@100.85.216.95` (Tailscale), RTX 3090. venvs: `isaacsim-venv` (SDG), `train-venv` (Detektor), `bop-venv` (Konverter/Eval), `gdrnpp-venv` (GDRNPP). Konfig in `.env`. |

Box-Setup im Detail: `../box_src/BOP_SETUP.md`. Eval: `../box_src/EVAL_BOP.md`.

## Datei-Übersicht

| Pfad | Inhalt |
|------|--------|
| `setup.ipynb` | Training reproduzieren (ruft `box_src/`-Skripte über die Box auf) |
| `infer.ipynb` | Inferenz + 3D-Viewer (importiert `e2e_infer`/`bop_adapter`) |
| `e2e_infer.py` | Standalone-Pipeline (alt): Bild → `pose_result.json`. Produktion laeuft jetzt ueber den Webservice (`kip_server.py` + `kip_infer_worker.py`) |
| `bop_adapter.py` | BOP(R/t cam, mm) → `pose_result` (Welt, m). Symmetrie-Kanonisierung + face/upright. Single-Source `OBJ_ID_TO_PART`. Getestet |
| `pose_result.schema.json` | Eingefrorener Output-Contract (ADR-017) |
| `frontend/` | Three.js-3D-Viewer + `assets/cell.glb` (echte Anlage) + `assets/parts/*.glb` (echte CAD-Meshes) |
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
