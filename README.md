# KIP POSE — 6D-Pose-Estimation für Metallteile in einer Montagezelle

<p align="center">
  <img src="https://img.shields.io/badge/Live-max--utils.com%2FKIP-34d399?style=for-the-badge" alt="Live"/>
  <img src="https://img.shields.io/badge/AR%20Anker__Kurz-0.870-2878ff?style=for-the-badge" alt="AR Anker Kurz"/>
  <img src="https://img.shields.io/badge/AR%20Anker__Lang-0.907-2878ff?style=for-the-badge" alt="AR Anker Lang"/>
  <img src="https://img.shields.io/badge/AR%20Zahnrad-0.838-2878ff?style=for-the-badge" alt="AR Zahnrad"/>
  <img src="https://img.shields.io/badge/AR%20%C3%98-0.872-ff2d2d?style=for-the-badge" alt="AR Mittel"/>
</p>

<p align="center">
  <b>Aus einem 2D-Foto einer Montagezelle die 6D-Pose aller Metallteile schätzen</b><br/>
  <i>Detektor (YOLOv8-OBB) → GDRNPP (RGB-only, per Objekt) → Welt-Transform + Boden-Snap → 3D-Render</i>
</p>

---

## 1 · TL;DR

| | |
|---|---|
| **Live-Demo** | <https://max-utils.com/KIP/> |
| **Stack** | Isaac-Sim 5.1 (synthetisch) + YOLOv8-OBB (Detektor) + **GDRNPP** (6D-Pose) + Three.js (3D-Viewer) |
| **3 trainierte Teile** | Anker_Kurz · Anker_Lang · Zahnrad |
| **Metrik** | BOP `AR = mean(AR_MSSD, AR_MSPD)`, symmetrie-bewusst |
| **Final-AR (best-by-val)** | **0.872** (Ø über 3 Teile) |
| **Latenz Real-Foto** | ~4 s end-to-end (Detektor + Worker + Snap + Render) |
| **Latenz Sim-Live** | ~80 s (Isaac-Boot + Render + BOP + Detektor + GDRNPP) |

---

## 2 · Pipeline (von Bild zu 3D-Pose)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│    [Real-Foto]                          [Live-Isaac-Sim]                      │
│        │                                       │                              │
│        │                                       ▼                              │
│        │                          gen_sdg_arm_visible.py                      │
│        │                          (Spawn + Physics-Settle + Render)           │
│        │                                       │                              │
│        │                                       ▼                              │
│        │                          isaac_to_bop.py                             │
│        │                          (raw → scene_camera/gt/masks)               │
│        │                                       │                              │
│        ▼                                       ▼                              │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ YOLOv8-OBB Detektor (train-venv subprocess)      │                         │
│  │   → det.json {scene/im: [{obj_id, bbox_est, …}]} │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ GDRNPP Worker  (persistent daemon, Port 8078)    │                         │
│  │   - alle 3 model_best.pth warm im VRAM (~2.3 GB) │                         │
│  │   - per-Objekt single-obj-Datasets registrieren  │                         │
│  │   - Output: (R_m2c, t_m2c)  BOP cam-frame, mm    │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ bop_adapter.bop_pose_to_world                    │                         │
│  │   R_world = R_w2c.T @ R_m2c                      │                         │
│  │   t_world = R_w2c.T (t_m2c - t_w2c)/1000 − Θ     │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ planar_z_snap  (Lift bei dz>0 IMMER, Sink mit    │                         │
│  │                 Guard max_snap_m=0.10)           │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ pose_result.json  (frozen Contract ADR-017)      │                         │
│  │   Z-up world, world = R @ body, Meter            │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────┐                         │
│  │ Three.js Viewer  (kip.html, scene.js)            │                         │
│  │   - cell.glb (Maschinen-CAD) + part GLB/PLY      │                         │
│  │   - groundClamp: per-Teil Raycast gegen cellGroup│                         │
│  │   - View+Zoom persist via localStorage           │                         │
│  │   - PiP-Fullscreen, Boxen-Overlay                │                         │
│  └──────────────────────────────────────────────────┘                         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3 · Architektur (Live-Deployment)

```
                                   ┌────────────────────────┐
   max-utils.com/KIP  ─https─▶     │   Cloudflare CDN/TLS   │
                                   └───────────┬────────────┘
                                               │
                                               ▼
                                   ┌────────────────────────┐
                                   │  Cloudflare Tunnel     │
                                   │  (cloudflared)         │
                                   └───────────┬────────────┘
                                               │
                                               ▼
                                   ┌────────────────────────┐
                                   │  ai-desk Raspberry Pi  │
                                   │  Caddy Reverse-Proxy   │
                                   │  @kip /KIP/* → :8077   │
                                   │  Cache-Control no-store│
                                   └───────────┬────────────┘
                                               │ Tailscale-LAN
                                               ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  GPU-Workstation  (max@100.85.216.95, RTX 3090, Ubuntu 24)          │
   │  ─────────────────────────────────────────────────────────────────  │
   │                                                                     │
   │   systemd-Services (Auto-Start, Auto-Restart):                      │
   │   ├─ kip-server.service   FastAPI :8077  (Web-API + Frontend)       │
   │   └─ kip-worker.service   stdlib HTTP :8078  (GDRNPP Inference)     │
   │                                                                     │
   │   venvs (per Stage isoliert):                                       │
   │   ├─ /mnt/data/isaacsim-venv/      Isaac Sim 5.1 (SDG-Render)       │
   │   ├─ /mnt/data/bop/train-venv/     YOLOv8-OBB Detektor              │
   │   ├─ /mnt/data/bop/bop-venv/       BOP-Konverter + Eval-Harness     │
   │   └─ /mnt/data/bop/gdrnpp-venv/    GDRNPP (Worker)                  │
   │                                                                     │
   │   Datasets:                                                         │
   │   └─ /mnt/data/kip_pose/project/bop/pose_isaac/                     │
   │      ├─ train_pbr/    (synthetisch generierte Trainings-Szenen)     │
   │      ├─ val/          (10 Szenen × ~100 Frames für Eval)            │
   │      └─ models/       (obj_000001.ply … obj_000006.ply, 6 Teile)    │
   │                                                                     │
   │   Live-Sim-Output:                                                  │
   │   └─ /mnt/data/kip_pose/project/temp/kip_live/<job>/                │
   │      ├─ rgb_0000.png + gt_raw + instance + …  (Isaac-Bundle)        │
   │      └─ bop/test/000000/  (nach isaac_to_bop Konvertierung)         │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 4 · Web-Viewer-Features (`max-utils.com/KIP`)

### Tab "Reales Foto"
- **Upload Zivid-Format** → Detektor + GDRNPP-Worker → 3D-Render
- **Phasen-Bar (5 Phasen):** Upload → Detektor → "N Box(en) gefunden" → GDRNPP → Welt+Snap → Fertig
- **PiP** (Picture-in-Picture) unten rechts: Original-RGB mit klassen-farbigen Detektor-Boxen + Confidence
- **Fullscreen-Toggle** (⛶) — PiP auf 4-fache Fläche; Esc bricht ab

### Tab "Simulation"
- **Echte Live-Isaac-Generation** pro Klick (kein Cache, kein Pool)
- **Phasen-Bar (6 Phasen):** Isaac booting → rendering → BOP-Convert → Detektor → GDRNPP → Fertig (~80 s gesamt)
- **GT (blau) gegen Schätzung (rot)** im selben 3D-Modell
- **Auto-Filter** instabiler Posen (rotationssymmetrische Teile mit aufrechter body-Y-Achse — Zahnrad auf Zähnen, Anker hochkant)

### Globale Top-Bar
- **Modell-Dropdown** (GDRNPP aktiv, 3 Platzhalter für späteren Architektur-Vergleich)
- **Workstation-Status** (Bereit / Training läuft / Offline) per `/api/health`-Poll alle 30 s
- **Ansicht zurücksetzen** — fittet die Kamera neu auf die Szene

### 3D-Viewer (Three.js)
- **cell.glb** (Maschinen-CAD, 27 MB / 1.12 M tris) + **CAD-Meshes pro Teil** (BOP-PLYs, full Tessellation)
- **View+Zoom persistieren** in localStorage (`kip.viewer.view.v1`) — Reload landet wo man war
- **groundClamp** per-Teil Raycast gegen cellGroup (3×3 Sample-Grid über Teile-Bbox) — Posen sitzen auf echter lokaler Tisch-Geometrie (Tray-Plateaus, Maschinen-Blöcke), nie clipping
- **Boden-Physik:** `planar_z_snap` zieht inferred Posen auf Tisch-Ebene; Lift IMMER, Sink mit Guard `max_snap_m=0.10` (Held-im-Greifer-Schutz)

---

## 5 · API-Referenz (FastAPI auf Port 8077)

### Health + Metadaten
```
GET  /api/health                  → {status, gpu_training_active, trained_objects, ts}
GET  /api/metrics                 → {objects: {slug: {best_full_ar, best_ckpt, status}}}
```

### Real-Foto-Inferenz
```
POST /api/real/infer_async        body: image=@<file>
                                  → {job}
GET  /api/real/job/<job>          → {phase, pct, [result_url, rgb_url, boxes_url, counts, n_det, n_parts]}
GET  /api/real/result/<job>       → pose_result.json (Contract)
GET  /api/real/rgb/<job>          → image/png  (unverändertes Foto)
GET  /api/real/boxes/<job>        → image/png  (mit klassen-farbigen Detektor-Boxen)
```

### Sim-Live-Generation (Isaac → Detektor → GDRNPP)
```
GET  /api/sim/generate_async      → {job}
GET  /api/sim/job/<job>           → {phase, pct, …}
GET  /api/sim/job_result/<job>    → pose_result.json (GT blau + Pred rot)
GET  /api/sim/live_rgb/<job>      → image/png  (frisches Isaac-RGB)
GET  /api/sim/live_boxes/<job>    → image/png  (mit Detektor-Boxen)
```

### pose_result Contract (ADR-017, frozen)
```jsonc
{
  "meta": {
    "source_image": "live/abc123",
    "table_origin": [0.0, 0.0, 0.08],         // Welt-Pos. Tisch-Nullpunkt, Meter
    "units": "m",
    "scene": 99, "im": 0,
    "source": "isaac-live",                   // oder "worker", "preds_best"
    "camera": {                               // Aufnahme-Kamera im Welt-Frame
      "cam_pos": [...], "look_at": [...], "up": [...], "fov_y": 30.45
    },
    "n_gt": 4, "n_pred": 3,
    "seed": 12345, "n_obj": 5                 // nur bei Live-Sim
  },
  "results": [
    {
      "instance_id": 1,
      "part": "Anker_Kurz",
      "face": "—",
      "confidence": 0.93,
      "t_world": [0.453, 0.281, -0.068],      // pose-frame = world − table_origin
      "R_world": [r00, r01, r02, r10, …],     // row-major 9-flat, world = R @ body
      "upright": false,
      "color": "gt" | "pred"
    }
  ]
}
```

**Konventionen:**
- **Welt-Frame:** Z-up, Ursprung = Tisch-Nullpunkt, Einheit Meter
- **Rotation:** `world = R @ body` (Spaltenkonvention), row-major flat 9
- **BOP-Boundary:** `cam_t_m2c` in **Millimeter** (Worker konvertiert intern)

---

## 6 · Quick-Start

### Lokal (Inferenz via Webservice — kein GPU/Setup nötig)

```bash
git clone <repo> POSE && cd POSE/project
python3 -m venv .posevenv && source .posevenv/bin/activate
pip install requests Pillow numpy

# Real-Upload
curl -F "image=@scene.png" https://max-utils.com/KIP/api/real/infer_async   # → {"job": "abc123"}
curl https://max-utils.com/KIP/api/real/job/abc123                          # poll bis pct=100
curl https://max-utils.com/KIP/api/real/result/abc123                       # pose_result.json

# Oder per Notebook
jupyter notebook infer.ipynb
```

### Lokal (volle Pipeline reproduzieren)

```bash
# 1. SSH + Isaac-Setup auf Workstation
ssh max@100.85.216.95

# 2. SDG-Daten neu rendern (Tausende Frames, mehrere Stunden GPU)
/mnt/data/isaacsim-venv/bin/python /mnt/data/kip_pose/box_src/gen_sdg_arm_visible.py \
  --scene /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd \
  --usd-dir /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files \
  --output /mnt/data/sdg-fresh --num-scenes 2000 --dr-strong

# 3. Isaac → BOP konvertieren
/mnt/data/bop/gdrnpp-venv/bin/python /mnt/data/kip_pose/box_src/isaac_to_bop.py \
  --raw-dir /mnt/data/sdg-fresh --bop-root /mnt/data/bop-fresh --split train_pbr

# 4. Detektor-Retrain (YOLOv8-OBB, ~6 h auf 3090)
bash /mnt/data/kip_pose/box_src/train_detector.sh

# 5. GDRNPP-Training (per Objekt, ~24 h pro Teil)
bash /mnt/data/kip_pose/box_src/phase2_chain.sh

# 6. Eval
bash /mnt/data/kip_pose/box_src/eval_bop.sh
```

Details: [`project/setup.ipynb`](project/setup.ipynb) führt Schritt-für-Schritt.

### Web-Viewer lokal entwickeln

```bash
cd project/frontend
python3 -m http.server 8000
# http://localhost:8000/kip.html  — API-Calls gehen relativ ./api/ ans Tailscale-Backend
```

---

## 7 · Auf der Workstation deployen

```bash
# Code-Update (Backend)
scp project/kip_server.py max@100.85.216.95:/mnt/data/kip_pose/project/
ssh max@100.85.216.95 'sudo systemctl restart kip-server.service'

# Worker-Update
scp box_src/kip_infer_worker.py max@100.85.216.95:/mnt/data/kip_pose/box_src/
ssh max@100.85.216.95 'sudo systemctl restart kip-worker.service'
# → Worker lädt alle 3 Checkpoints, ~4 min Warm-Load. journalctl -u kip-worker -f

# Frontend (statisch, Cache-Control no-store → keine CF-Purge nötig)
scp project/frontend/{kip.html,src/*,assets/*} max@100.85.216.95:/mnt/data/kip_pose/project/frontend/
```

---

## 8 · Datei-Übersicht

```
POSE/
├── README.md                                this file
├── project/
│   ├── README.md                            project-specific extended notes
│   ├── kip_server.py                        FastAPI Web-Service (Port 8077)
│   ├── e2e_infer.py                         Standalone-Pipeline (legacy)
│   ├── bop_adapter.py                       BOP cam → World transform + planar_z_snap
│   ├── pose_result.schema.json              Output-Contract (ADR-017)
│   ├── setup.ipynb                          "Training reproduzieren" (auf Workstation)
│   ├── infer.ipynb                          "Inferenz via Webservice" (lokal)
│   ├── tests/                               pytest — 106 passed, 6 skipped
│   ├── frontend/
│   │   ├── kip.html                         2-Screen Viewer
│   │   ├── src/{kip.js,scene.js,…}          Three.js + UI-Wiring
│   │   ├── vendor/three/                    Three.js r160 + Add-ons local
│   │   └── assets/
│   │       ├── cell.glb                     Fallback (8 MB / 0.44 M tris)
│   │       ├── cell_web.glb                 Default (27 MB / 1.12 M tris)
│   │       ├── cell_hq.glb                  Offline (189 MB / 8 M tris, gitignored)
│   │       └── parts/ply/obj_00000{1..6}.ply  hi-res BOP-CAD-Meshes
│   └── models/                              SELECT_BEST per Teil (gitignored, 4.8 GB)
├── box_src/
│   ├── kip_infer_worker.py                  Multi-Obj GDRNPP daemon (Port 8078)
│   ├── gen_sdg_arm_visible.py               Isaac SDG-Render-Skript
│   ├── isaac_to_bop.py                      raw → BOP-Konverter
│   ├── obb_to_aabb_dets.py                  Detektor → BOP-det-json
│   ├── eval_bop.{py,sh}                     BOP-Toolkit Eval-Harness
│   ├── phase2_chain.sh                      Detektor → 3× GDRNPP-Train Chain
│   └── BOP_SETUP.md, EVAL_BOP.md            Setup-Docs
└── docs/
    ├── ARCHITECTURE.md
    ├── ADD_NEW_PART.md
    ├── EVAL.md
    ├── REPRODUCE.md
    └── PROJECT_REPORT.md                    akademischer Kurzbericht
```

---

## 9 · Final-AR (best-by-val, BOP-Toolkit)

| Objekt | Final AR | Final Checkpoint | n_ckpts evaluiert |
|---|---:|---|---:|
| Anker_Kurz | **0.870** | `model_0112949.pth` (ep ~150) | 17 |
| Anker_Lang | **0.907** | `model_0120959.pth` (ep ~160) | 17 |
| Zahnrad | **0.838** | `model_0130399.pth` (~ep 165) | 17 |
| **Mittel** | **0.872** | — | — |

`AR = mean(AR_MSSD, AR_MSPD)`, symmetrie-bewusst (Zahnrad C_7, Anker rotationssym. um Y).

---

## 10 · Lessons (was hat dieser Stack uns gelehrt)

| | |
|---|---|
| **GDRNPP > Eigenbau** | Eigenbau-Pipeline (Face-Atlas + Template-Bank) verworfen nach 2 Wochen. GDRNPP RGB-only liefert 0.87 AR direkt aus dem Schlauch. Siehe [ADR-018](project/docs/decisions/adr-018-pose-bop-sota-pivot.md). |
| **Isaac SDG single-scene-Bug** | `gen_sdg_arm_visible.py --num-scenes 1` ohne `--force-counts` registriert keine semantic-Labels (nur BACKGROUND / UNLABELLED). Mit force-counts greift der add_labels-Patch zuverlässig. |
| **USD-Transform-Scaling** | `ComputeLocalToWorldTransform` enthält `OBJECT_SCALE=1e-3` in der 3×3-Block. Filter, die direkt Komponenten gegen geometrische Schwellwerte vergleichen, müssen erst per Spalte normalisieren. |
| **Worker-Unit-Konvention** | GDRNPP liefert `trans` in **Meter**. BOP-Boundary erwartet **mm**. Worker muss `*1000.0` in **beiden** Output-Codepfaden machen — sonst pose schwebt 1 m über dem Tisch. |
| **Mesh-Center-Drift** | Backend `planar_z_snap` braucht AABB-zentrierte Verts — sonst snapped es den PLY-Origin-Tiefpunkt, der Viewer rendert aber den Mesh-Center → 1-2 cm Schwebedrift. |
| **`gltfpack -cc` ist NICHT lossless** | Trotz Marketing verzerrt EXT_meshopt-Compression Normalen/UVs für Vendor-CADs sichtbar. Für Geometric-Fidelity lieber unkomprimiert in lower poly. |
| **pgrep Self-Match** | `pgrep -af pattern` matcht Monitor-Loops die `pgrep pattern` als Argument enthalten. Immer mit Skript-Suffix (`.sh`) oder Negativ-Filter. |
| **Browser-View-Persist + async loadCell** | Persistierte View überleben nur wenn fitView in einem Zeitfenster (3 s) nach Viewer-Init priorisiert — sonst überschreibt der asynchrone GLTF-Load die restored Kamera. |

---

## 11 · Konventionen

| Was | Wert |
|---|---|
| **Welt-Frame** | Z-up, Ursprung = Tisch-Nullpunkt, Einheit Meter |
| **Rotation** | `world = R @ body` (Spaltenkonvention), row-major flat 9 in JSON |
| **BOP-Boundary** | `cam_t_m2c` in **mm**, alles andere in m |
| **obj_id-Mapping** | `{1: Anker_Kurz, 2: Anker_Lang, 3: Buerstenhalter_2polig, 4: Getriebegehaeuse_typ4, 5: Ringmagnet, 6: Zahnrad}` (1-basiert, BOP-Standard) |
| **trainierte Klassen** | obj_id ∈ {1, 2, 6} — Anker_Kurz, Anker_Lang, Zahnrad. Restliche 3 spawnen wir als Distractor in SDG, predicten sie aber nicht |
| **Symmetrie** | Anker_{Kurz,Lang} + Ringmagnet = `continuous` um body-Y, Zahnrad = `discrete C_7` um body-Y |
| **Snap** | Lift (dz>0) IMMER; Sink (dz<0) nur bis `max_snap_m=0.10` (Held-im-Greifer-Schutz) |

---

## 12 · Tests

```bash
cd project && python3 -m pytest tests/ -q
# 106 passed, 6 skipped
```

Deckt:
- `bop_adapter.bop_pose_to_world` — Transform-Kette + Symmetrie-Kanonisierung
- `bop_adapter.planar_z_snap` — Lift/Sink-Verhalten + Guards
- E2E-Kette mit deterministischem Stub-Estimator

---

## 13 · Roadmap / Open Topics

- [ ] **Sim Re-Training mit fixed-Filter** (optional Polish, ~30 h GPU) — würde das marginal verbessern; aktuell nicht nötig weil real-world Bias ohnehin liegend ist
- [ ] **Live-Isaac-Daemon** statt subprocess-Spawn pro Klick (würde Sim von ~80 s auf ~10 s drücken)
- [ ] **Mid-poly cell-Variante** (~3-4 M tris, ~80 MB) zwischen `cell_web` und `cell_hq` — falls Detail-Anforderungen steigen
- [ ] **Architektur-Vergleich** (MegaPose / FoundPose / GigaPose Slots im Modell-Dropdown vorbereitet)

---

## 14 · Kontakt + Lizenz

Privat / WIP. Workstation: `max@100.85.216.95` (Tailscale).
Brain-Notes für Setup + Bugs unter `/Users/Admin/Documents/CLAUDE_BRAIN/`.

---

<p align="center"><i>Built with way too many late-night SSH sessions to the GPU-box.</i></p>
