# POSE — 6D-Pose-Pipeline (BOP-SOTA-Stack)

Aus einem 2D-Top-Down-Bild einer Montagezelle die **6D-Pose** der Metallteile
schätzen und sie als echtes CAD im **3D-Viewer** an ihrer Pose rendern.

Zwei Notebooks, beide **top-to-bottom in einem Rutsch** lauffähig:

| Notebook | Zweck |
|----------|-------|
| **`setup.ipynb`** | „Training reproduzieren" — CAD + Isaac-SDG → BOP-Daten → trainierte Modelle (auf der GPU-Box) |
| **`infer.ipynb`** | „Inferenz + 3D-Viewer" — Bild rein → `pose_result.json` → localhost-3D-Viewer |

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

**Bis GDRNPP fertig trainiert ist** läuft Stufe 3 automatisch im **MOCK-Modus**
(deterministische, plausible Posen) — die ganze Kette inkl. Viewer ist **jetzt
schon grün**. Sobald ein Checkpoint da ist: `GDRNPP_CHECKPOINT` im Notebook
setzen, der echte Call schaltet automatisch zu.

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
| `e2e_infer.py` | Standalone-Pipeline: Bild → `pose_result.json` (`--serve` öffnet Viewer). GDRNPP MOCK-Fallback bis Checkpoint da |
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
face/upright) und die volle E2E-Kette im MOCK ab (schema-valides `pose_result`).

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
