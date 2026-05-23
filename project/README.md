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

## Schnellstart — EIN Befehl startet die Website

```bash
# Inferenz auf einer Szene + interaktiver localhost-CAD-Viewer (öffnet den Browser)
python project/e2e_infer.py --image project/input/scene_0000.png --serve
```

Das erzeugt `temp/pose_result.json`, startet einen localhost-Server ab `project/`
und öffnet `http://127.0.0.1:8000/frontend/?file=../temp/pose_result.json` — den
frei interaktiven 3D-Viewer mit den **echten CAD-Teil-Meshes** an der predicted
6D-Pose, auf dem echten Zellen-CAD (`cell.glb`): Orbit/Zoom/Pan, Klick→Info-Panel
(Teil/Face/Position rel. Nullpunkt/Confidence). Three.js ist lokal **vendored**
(`frontend/vendor/three/`) → läuft komplett offline, kein CDN.

```bash
# nur pose_result erzeugen (ohne Viewer)
python project/e2e_infer.py --image project/input/scene_0000.png --out project/temp/pose_result.json
```

`setup.ipynb` braucht eine erreichbare GPU-Box (`.env`: `GPU_HOST`, `BOX_REPO`,
venvs). Die Inferenz braucht nur `numpy`, `scipy`, `PIL` (Fallback-Pfad); `torch`
+ `ultralytics` werden lazy für die trainierten Checkpoints geladen.

## Lernbasiertes Pose-Refinement (GigaPose/MegaPose-Stil)

Nach dem Bank-Match verfeinert ein gelerntes Embedding (`models/refiner_<part>.pt`)
die Rotation: ein CNN bettet Query-Tiefen-Crop UND alle Bank-Tiefen-Templates in
denselben Raum, so dass das embedding-nächste Template das pose-nächste ist — das
schliesst den Domain-Gap synthetisch↔Isaac, an dem die Handmetrik (depth-NCC +
Silhouetten-IoU + Gradient) scheitert. Trainiert vollsynthetisch (Top-Down-Tiefen-
Crops + GT, Domain-Randomization), sym-aware (unbeobachtbare DoF nicht bestraft).
`match_template_bank` re-rankt die Kandidaten geblendet mit der Handmetrik
(`POSE_REFINER_WEIGHT`, default 0.75; 0 = Refiner aus). **Drop-in:** ohne
Checkpoint läuft die Handmetrik weiter. Training: `train_refiner.py` (Box).

## Pipeline-Verbesserungen (P1–P7)

| # | Verbesserung | Stand |
|---|---|---|
| **P1** | **Tisch-Kollision** — beim SDG settlen die Teile jetzt sauber AUF der echten Wagen-Tischplatte (z = -0.007 m, aus `GST_Scene.usd` gemessen), nie hinein. Fix: statische Collider auf Wagen+Trays, kalibrierte Spawn-Region über der Tischplatte, Settle 200. | echt, verifiziert (8-11/10-12 Teile on-surface, 0 clippen rein) |
| **P2** | **Höhen-/Stehende-Teile** — `t_world.z = TabletischZ + rest_height(face)`. `rest_height` = halbe z-Ausdehnung der Mesh in der Ruhelage, pro Face in `bank.npz` + `faces_<part>.json`. Stehende Faces -> höher. | echt |
| **P3** | **Mehr Teile** — Zahnrad (Zahnrad_Typ7) + **Ringmagnet** (neu): Drops + Face-Registry + Template-Bank + in Multi-Part-Szenen + Detektor (7 Klassen). | echt |
| **P4** | **Overlap/Occlusion** — überlappende 2D-BBoxes: Tiefe (metr. `depth_<idx>.npy` bevorzugt, sonst Kamera-Distanz) entscheidet oben/unten. Felder `depth_order`/`occluded_by`/`on_top_of`; Viewer + Recon rendern in der Stapel-Reihenfolge. | echt |
| **P5** | **Eval vs Ground-Truth** — Isaac-Eval-Szene mit GT-6D-Pose (Settle-Readback), volle Pipeline drauf, Fehler gemessen: Translation (mm), Rotation (deg, +symmetrie-bewusst), Klassen-/Recall. `setup.ipynb` Stufe 7 + `e2e_infer.py --eval-gt`. | echt |
| **P6** | **Isaac-Recon-Render** — `pose_result.json` → Isaac platziert die ECHTEN CAD-Teile an den predicted 6D-Posen in die Zelle → RGB-Render `temp/recon_render.png` (+ optional USD). `setup.ipynb` Stufe 8. | echt (Box) |
| **P7** | **3D-Viewer frei drehbar** — OrbitControls (orbit/zoom), echtes Zellen-CAD (`cell.glb`) + Teile als korrekt hohe Boxen an 6D-Pose, in Stapel-Reihenfolge. | echt, verifiziert |

## Konvention (eingefroren)

Z-up Welt · `world = R @ body` (Spaltenkonvention) · Ursprung = Tisch-Nullpunkt
= echte Wagen-Tischplatte der GST-Welt (z = -0.007 m) · Einheit Meter. `t_world`
ist faktisch Welt (x/y, da Tisch-x/y = 0; z = -0.007 + rest_height). Contract:
`pose_result.schema.json` (+ optionale Felder rest_height/depth_order/occluded_by).

## Finales Abnahme-Bild

`infer.ipynb` (letzte Zelle) bzw. `setup.ipynb` Stufe 9 bauen das Split-Screen
nach `temp/final_2d_vs_3d.png` und öffnen es:

- **links** das 2D-Input mit Detektionen (BBox + part·face),
- **rechts** der 3D-Recon: bevorzugt der **Isaac-Render mit den echten CAD-Teilen**
  an den predicted 6D-Posen (`temp/recon_render.png`, P6), sonst ein
  Web-Viewer-Screenshot (echtes Zellen-CAD + Teile, frei drehbar).

Stapelung (P4) + Höhe (P2) sind in beiden 3D-Darstellungen sichtbar.
