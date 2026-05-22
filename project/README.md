# POSE — End-to-End 6D-Pose-Pipeline

Selbst-enthaltenes Projekt: **CAD rein → Modelle → 2D-Szene → 6D-Pose → 3D-Viewer.**
Alles Wesentliche steckt INLINE in den beiden Notebooks plus `e2e_infer.py` —
keine Projekt-Module ausser `e2e_infer.py`. Datengenerierung + Training laufen
über die GPU-Box (Isaac Sim / torch); Inferenz + Viewer laufen lokal.

## Ablauf

```
setup.ipynb   CAD (cad_input/) ─Isaac-SDG─▶ Daten ─train─▶ models/
                  ├ 1 Multi-Part-DR-Szenen (GPU-Box) ──▶ data/output/big
                  ├ 2 Face-Discovery + Registry        ──▶ models/registry/<part>/
                  ├ 3 Snippet-Dataset + Manifest
                  ├ 4 Face-Classifier pro Teil (CNN)   ──▶ models/<part>.pt
                  └ 5 OBB-Detektor (YOLOv8-OBB)        ──▶ models/detector.pt

infer.ipynb / e2e_infer.py
                  2D-Szene (input/) ─Detektor─▶ Crops ─Face─▶ Alignment
                  ──▶ temp/pose_result.json  (Contract: pose_result.schema.json)

frontend/         pose_result.json ──▶ Three.js-Viewer (Tisch + Teile an 6D-Pose)
```

## Verzeichnisse

| Pfad | Inhalt |
|------|--------|
| `setup.ipynb` | Daten-Generierung + Training (CAD → Modelle), alles inline |
| `infer.ipynb` | Inferenz-Pipeline (2D-Bild → pose_result), alles inline |
| `e2e_infer.py` | Standalone-Variante derselben Pipeline (ein Skript, weitergebbar) |
| `cad_input/enviroment/` | **CAD-Eingang** — Teile-USDs (`parts/`) + Zellen-/Anlagen-Szenen |
| `models/` | trainierte Checkpoints: `detector.pt`, `<part>.pt`, `registry/<part>/` |
| `input/` | Eingabe-Szenen (RGB + optional bbox/semantic-JSON) |
| `frontend/` | Three.js-3D-Viewer (kein Build-Step, Three via CDN-Importmap) |
| `temp/` | Scratch: `pose_result.json`, Detektor-Overlays, finales Render |
| `training_data/` | generierte Trainingsdaten (gross, gitignored) |

## Schnellstart

```bash
# Inferenz auf einer Szene + Viewer
python project/e2e_infer.py --image project/input/scene_0000.png --serve

# nur pose_result erzeugen
python project/e2e_infer.py --image project/input/scene_0000.png --out project/temp/pose_result.json
```

`setup.ipynb` braucht eine erreichbare GPU-Box (`.env`: `GPU_HOST`, `BOX_REPO`,
venvs). Die Inferenz braucht nur `numpy`, `scipy`, `PIL` (Fallback-Pfad); `torch`
+ `ultralytics` werden lazy für die trainierten Checkpoints geladen.

## Konvention (eingefroren)

Z-up Welt · `world = R @ body` (Spaltenkonvention) · Ursprung = Tisch-Nullpunkt ·
Einheit Meter. Contract: `pose_result.schema.json`.
