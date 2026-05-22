# POSE — End-to-End 6D-Pose-Pipeline

Selbst-enthaltenes Projekt: **CAD rein → Modelle → 2D-Szene → 6D-Pose → 3D-Viewer.**
Alles Wesentliche steckt INLINE in den beiden Notebooks plus `e2e_infer.py` —
keine Projekt-Module ausser `e2e_infer.py`. Datengenerierung + Training laufen
über die GPU-Box (Isaac Sim / torch); Inferenz + Viewer laufen lokal.

## Ablauf

```
setup.ipynb   CAD (cad_input/) ─Isaac-SDG─▶ Daten ─train─▶ models/
                  ├ 1  Multi-Part-DR-Szenen (GPU-Box)  ──▶ data/output/big
                  ├ 2  Face-Discovery + Registry        ──▶ models/registry/<part>/
                  ├ 2b Template-Bank Render-and-Compare ──▶ models/templates/<part>/bank.npz
                  ├ 3  Snippet-Dataset + Manifest
                  ├ 4  Face-Classifier pro Teil (CNN)   ──▶ models/<part>.pt  (Vorfilter)
                  ├ 5  OBB-Detektor (YOLOv8-OBB)        ──▶ models/detector.pt
                  └ 6  GST_Scene -> cell.glb            ──▶ frontend/assets/cell.glb

infer.ipynb / e2e_infer.py
                  2D-Szene (input/) ─Detektor(OBB)─▶ Crops
                  ─Template-Bank-Match─▶ exakte Ruhelage + Yaw -> R_world
                  ─metrische Backprojection (echte Zivid-Intrinsics)─▶ t_world
                  ──▶ temp/pose_result.json  (Contract: pose_result.schema.json)

frontend/         pose_result.json + cell.glb ──▶ Three.js-Viewer
                  (echtes Anlagen-CAD: Tisch + Roboterarm + Teile an 6D-Pose)
```

## Pose-Methode — Template-Bank Render-and-Compare

Die Orientierung kommt **nicht** aus einem CNN-Rateschritt, sondern aus einem
deterministischen Match gegen eine pro Teil aus dem ECHTEN CAD gerenderte
Template-Bank: **{stabile Ruhelagen aus `faces_<part>.json`} × {Yaw 0–360° in
5°}**, top-down Tiefen-/Silhouetten-Templates (`models/templates/<part>/bank.npz`).
Bei der Inferenz wird der detektierte Crop gegen die Bank gematcht → exakte
Ruhelage (Face) + exakter Yaw → volle `R_world` direkt aus der Bank. Die OBB
liefert (x,y) + groben Yaw-Seed (verkleinert das Suchfenster). Planar-
eingeschränktes Render-and-Compare (CosyPose/MegaPose-Idee). Der Face-Classifier
bleibt als optionaler Schnell-Vorfilter, ist aber nicht mehr die Pose-Quelle.

## Verzeichnisse

| Pfad | Inhalt |
|------|--------|
| `setup.ipynb` | Daten-Generierung + Training (CAD → Modelle), alles inline |
| `infer.ipynb` | Inferenz-Pipeline (2D-Bild → pose_result), alles inline |
| `e2e_infer.py` | Standalone-Variante derselben Pipeline (ein Skript, weitergebbar) |
| `cad_input/enviroment/` | **CAD-Eingang** — Teile-USDs (`parts/`) + Zellen-/Anlagen-Szenen |
| `models/` | `detector.pt`, `<part>.pt`, `registry/<part>/`, `templates/<part>/bank.npz` |
| `input/` | Eingabe-Szenen (RGB + optional bbox/semantic-JSON) |
| `frontend/` | Three.js-3D-Viewer + `assets/cell.glb` (echtes Anlagen-CAD) |
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

Z-up Welt · `world = R @ body` (Spaltenkonvention) · Ursprung = Tisch-Nullpunkt
(Tray-Arbeitsfläche der GST-Welt) · Einheit Meter. Contract:
`pose_result.schema.json`.

## Finales Abnahme-Bild

```bash
python project/e2e_infer.py --image project/input/scene_0000.png   # pose_result
python project/temp/make_split_render.py                            # 2D | 3D Split
open project/temp/final_2d_vs_3d.png
```

Links das 2D-Input (mit Detektionen), rechts das 3D-Rendering der echten Anlage
(`cell.glb`) mit den platzierten Teilen.
