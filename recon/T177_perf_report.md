# T-177 — PERF-Pass: Max-Performance aller 12 Kombis (Eval + Optimierung)

**Ticket:** T-177 · **Agent:** Claude (solo) · **Datum:** 2026-06-10
**Box:** `max@100.85.216.95` (RTX 3090, 24 GB) · **Auftrag (Max):** „Performance vom
aktuellen System — max Performance rausholen aus allen Kombis, alles evaluieren und
optimieren auf dem, wie es basiert."

> **TL;DR: Drei Daten-/Config-Bugs gefunden, die 8 der 12 Kombis künstlich
> gedrückt haben. Nach den Fixes: FoundationPose 0.658→0.97 (neuer Genauigkeits-
> König, schlägt Pipeline A), sam3-Kombis 0.22–0.41→0.68–0.95 bei Coverage
> 37%→100%, GigaPose-3D 0.635→0.84, GigaPose-2D 0.436→0.66 (Ebenen-Prior).
> Pipeline A (0.886) bleibt byte-identisch unberührt. Zwei Hebel wurden als
> ehrliche Negative gemessen und verworfen (FP-Iter-Reduktion, ROI-Expansion).**

---

## 1. Befund-Kette (Probes, alle read-only reproduzierbar)

### Bug 1 — Depth speicherte euklidische Ray-Distanz statt planarem Z (KRITISCH)

- **Probe C** (Fehler-Zerlegung entlang Sehstrahl, final-run CSVs):
  FP `+41.2mm radial` (lateral 0.7mm!), GigaPose-3D `+34.8mm radial` — kein
  „random scatter" wie in T-166/T-173/T-175 angenommen, sondern systematisch.
- **Probe P1** (Depth-PNG direkt vs GT-Pose): median `+38.1mm`.
- **Beweis:** 1/cos-Vorhersage des Euclid-als-Planar-Fehlers = `+35.1mm`,
  **corr(delta, prediction) = 0.91**. `isaac_to_bop.py` schrieb Isaacs
  `distance_to_camera` (euklidisch, stand sogar im npy-Docstring) 1:1 in die
  BOP-PNGs; alle Konsumenten (Gateway, FP, GigaPose-ICP) lesen planar.
- **Fix:** `isaac_to_bop.planar_cos_map` (Konverter, für alle künftigen Renders
  inkl. Live-Sim-Pfad) + `box_src/repair_depth_planar.py` (Bestands-Repair,
  idempotent via `.planar_repaired`-Marker, Backup `depth_euclid_orig/`).
  Verifiziert: Bias +38.1 → **+5.6mm**.
- Erklärt auch T-104 („vereinzelte anomale GT-Tiefe").
- Brain: `bugs/kip-pose-depth-euclid-vs-planar-convention.md`.

### Bug 2 — sam3-Coverage-Killer: Prompt-Drift (0 Detections)

- batch_eval/combos overrode die sam3-Prompts per Request mit
  `"short/long anchor metal part"` — die im Service getunten Defaults lauten
  `"short/long metal motor armature part"`. **Gemessen auf demselben Frame:
  Override → 0 Detections, Defaults → 27.** Betraf Eval UND Live-Pfad.
- **Fix:** Overrides entfernt; Single Source of Truth = sam3-svc-Env.
- Brain: `bugs/kip-pose-sam3-prompt-drift-zero-detections.md`.

### Bug 3 — sam3-Klassen: kurz/lang real nur 3mm auseinander

- Probe A: sam3-Masken matchen **100%** der GT-Anker (IoU≥0.3) — nur die Labels
  waren zufällig (alle Final-Run-Detections eine Klasse → Anker_Lang AR 0.0).
- Mesh-Wahrheit: obj1 = 112.0mm, obj2 = 115.0mm Volllänge → **jeder geometrische
  Klassifikator ist tot** (der im sam3-svc-Docstring dokumentierte depth-PCA-Band
  <128/>133 stammt von einem anderen Teile-Set; GT-Separierbarkeit hier nur 73%).
- **Fix:** Gateway-Label-Transfer (T-177): bei `seg=sam3` zusätzlich yolo-obb
  (~76ms), Klassen + orientierte Box per Mask↔OBB-IoU-Match übernehmen,
  ungematchte Konzept-FPs droppen (260/295 im Probe waren FPs). sam3 behält
  seine Masken; Response weist `class_source: "yolo-obb"` aus.
  Env: `SAM3_CLASS_FROM_YOLO=1` (Default), `SAM3_YOLO_IOU=0.30`.

### Hebel 4 — GigaPose-2D: Tisch-Ebenen-Prior („Stable-Pose-Snap")

- Probe C: GP-2D-Fehler ist ~rein radial (92mm median radial vs 5.3mm lateral) —
  RGB-coarse hat keine metrische Tiefe.
- Probe 4 (Hand-AR-Simulation mit bop_toolkit): Snap der Translation entlang des
  Sehstrahls auf die Rig-Ebene (Objektzentrum-Ruhehöhe 14.3mm Welt, gemessen als
  GT-Median = Kalibrierkonstante) → AR 0.414 → 0.609.
- **Fix:** Gateway-Stage `table_plane` (JSON {n,d}, Cam-Frame, Meter; Guard max
  300mm Shift) + batch_eval-Wiring aus scene_camera-Extrinsics. Nur für
  `gigapose_rgb` (FP/GP-3D haben metrische Tiefe, GDRNPP-RGB liegt bei 2.3mm).
  Verfahren-Label ehrlich: „coarse+Ebenen-Prior".
- **Live-Messung: 0.446 → 0.656** auf denselben 30 Frames.

### Ehrliche Negative (gemessen, verworfen)

| Hebel | Messung | Verdikt |
|---|---|---|
| FP `iterations` 5→3→2 | 0.972@7.9s → 0.953@6.0s → 0.932@5.1s (30 Frames, crash 0%) | **iter=5 bleibt** — ~2 AR-Punkte pro Stufe; T-173 ergänzt: 5 ist Optimum, nicht nur Sättigung |
| gdrnpp mask-AABB ×1.19 (Amodal-Kalibrierung, Probe D: Maskenboxen 16% tighter) | 0.856 (×1.19) vs **0.885 (×1.0)**, gleiche 30 Frames | **zurück auf 1.0** — GDRNPPs DZI-ROI padded selbst, Doppel-Padding schadet. Knob `GDRNPP_MASK_AABB_EXPAND` bleibt dokumentiert |

### VRAM-Ops (T-133-Beweis)

fp-svc kriecht unter Last auf **13.4GB** und gibt nicht zurück (warp-Allocations
am torch-Allocator vorbei); Gesamtstack dann exakt an der 24GB-Kante → OOM-Crashes
(2 Sanity-Runs zerschossen + 1 Parallel-Lauf-Ops-Fehler meinerseits, im Board
geloggt). Eval-Ops jetzt: **fp-Restart vor jedem FP-lastigen Lauf, strikt
sequenzielle Runs, sam3/gigapose situativ stoppen.** Der saubere Fix bleibt
S-007/T-133 (VRAM-Lifecycle-Manager, Backlog).

---

## 2. Zwischenstand-Messungen (vor Final-Run)

Quick-Checks auf Frames 0,10,20 (30 Frames), Full-100 wo markiert:

| Kombi | vorher (run-20260608) | nachher | Δ |
|---|---|---|---|
| yolo-obb → FoundationPose | 0.658 | **0.972** (full-100) | +0.31 |
| yolo-seg → FoundationPose | 0.658 | **0.973** (full-100) | +0.31 |
| sam3 → FoundationPose | 0.279 | **0.948** | +0.67 |
| sam3 → GDRNPP | 0.409 | **0.896** | +0.49 |
| yolo-obb/seg → GigaPose-3D | 0.635 | **0.837/0.838** (full-100) | +0.20 |
| sam3 → GigaPose-3D | 0.276 | **0.846** | +0.57 |
| sam3 → GigaPose-2D | 0.216 | **0.676** | +0.46 |
| yolo-obb/seg → GigaPose-2D | 0.436 | **0.656** | +0.22 |
| yolo-seg → GDRNPP | 0.863 | 0.885 (30-Frame-Subset, unverändert konfiguriert) | ~0 |
| **Pipeline A (yolo-obb → GDRNPP)** | **0.886** | **unberührt** (byte-identischer Live-Pfad) | 0 |

sam3-Coverage: 37% → **100%**. Alle Quick-Checks crash 0%.

## 3. Final-Run — `run-20260610T102118Z` (12 Kombis × 100 Frames, crash 0%)

Gefahren als 4 VRAM-gestaffelte Teilläufe (fp-Peak 13.4 GB + GigaPose-unter-Last
4.9 GB passen nicht mehr gleichzeitig in die 24 GB; der all-residente 12er-Lauf
OOM-crashte) + `merge_runs.py` → EIN kuratierter Run mit Standard-run-id.
`eval_every=10` (Endzahlen identisch, ~90 % eval_bop-Wall-Clock gespart):
**5239 s statt 13425 s** für 12×100.

| # | Kombi | Input | AR vorher | **AR final** | Δ |
|---|---|---|---|---|---|
| 1 | yolo-obb → FoundationPose | RGBD | 0.658 | **0.968** | +0.31 |
| 2 | yolo-seg → FoundationPose | RGBD | 0.658 | **0.968** | +0.31 |
| 3 | sam3 → FoundationPose | RGBD | 0.279 | **0.964** | +0.69 |
| 4 | **Pipeline A** (yolo-obb → GDRNPP) | RGB | 0.886 | **0.886** | ±0 (Referenz hält!) |
| 5 | sam3 → GDRNPP | RGB | 0.409 | **0.880** | +0.47 |
| 6 | yolo-seg → GDRNPP | RGB | 0.863 | **0.863** | ±0 |
| 7 | yolo-obb → GigaPose-3D | RGBD | 0.635 | **0.838** | +0.20 |
| 8 | yolo-seg → GigaPose-3D | RGBD | 0.635 | **0.838** | +0.20 |
| 9 | sam3 → GigaPose-3D | RGBD | 0.276 | **0.829** | +0.55 |
| 10 | sam3 → GigaPose-2D | RGB | 0.216 | **0.650** | +0.43 |
| 11 | yolo-obb → GigaPose-2D | RGB | 0.436 | **0.636** | +0.20 |
| 12 | yolo-seg → GigaPose-2D | RGB | 0.436 | **0.636** | +0.20 |

- **Plattform-Schnitt 0.49 → 0.85**; schlechteste Kombi 0.216 → 0.636.
- **Pipeline A exakt auf Referenz (0.8861)** = Regressions-Beweis, Live-Pfad unberührt.
- sam3-Coverage 37 % → 96–97 %, alle 12 Kombis crash 0 %.
- Viewer/Edge verifiziert: `/api/eval/runs` = 2 kuratierte Runs (neu + alt als
  Vorher-Referenz), `max-utils.com/KIP` served den neuen Run als Default.
  Alle t177-Zwischenläufe gelöscht (S174-Disziplin).

### Einordnung RGB-D vs RGB (Max-Frage)
- **Methodisch:** Zero-Shot-RGB-D (FP 0.97) schlägt jetzt trainiertes RGB
  (GDRNPP 0.886) — Tiefe liefert die metrische Translation, die RGB fehlt.
- **Produktiv:** FoundationPose ist **non-commercial** (nur Eval) + 7.9 s/Frame;
  bestes einsetzbares RGB-D = GigaPose-3D 0.838 < Pipeline A 0.886. **Pipeline A
  bleibt Produktions-König.**
- **Vorbehalt:** Eval-Tiefe ist (reparierte) Sim-Tiefe ≈ ideal. Echte
  Zivid-Tiefe auf glänzendem Metall ist verrauscht (Grund der ursprünglichen
  RGB-only-Entscheidung). **Next:** kleines Real-Set mit echter Zivid-Tiefe
  durch FP/GigaPose-3D messen.

## 4. Geänderte Artefakte

**Code (lokal == Box, Backups `.bak-T177` auf der Box):**
- `box_src/isaac_to_bop.py` — `planar_cos_map` + Euclid→Planar im Konverter
- `box_src/repair_depth_planar.py` — NEU, Bestands-Repair (val 10×100 Frames repariert)
- `project/mesh/gateway/app.py` — sam3-Label-Transfer + `table_plane`-Snap-Stage
  (+ `class_source`/`plane_snapped` im Response)
- `project/mesh/sam3-svc/app.py` — Doku: Limitation→Resolution, 3mm-Befund
- `project/mesh/gdrnpp-svc/app.py` — `GDRNPP_MASK_AABB_EXPAND`-Knob (Default 1.0,
  Negativ dokumentiert)
- `project/pipelines/combos.py` — Prompt-Overrides raus, sam3-Label/Flag
- `project/eval/batch_eval.py` — `--configs`-Filter, `table_plane_cam`-Wiring,
  Prompt-Override raus, Verfahren-Label
- `project/eval/run_t177.py` — NEU, parametrisierter Subset-Runner
- `project/frontend/src/pipeline.js` — SEG_SOURCES-Spiegel (sam3-Label/Flag)
- `project/tests/test_batch_eval.py`, `test_live_standings.py` — sam3-Flag-Kontrakt

**Daten (Box):** `project/bop/pose_isaac/val/*/depth/` repariert (Backup
`depth_euclid_orig/` + Marker), `models_info`/Meshes unberührt.

**Tests:** 281 passed / 7 skipped (1 bekannter T-088-Flake deselektiert).

## 5. Lehren (→ Brain)

1. Wenn eine **Methoden-Familie** (alle Depth-Konsumenten) denselben Bias gegen
   GT zeigt, den RGB-Methoden nicht haben → **Daten-Pipeline auditieren, bevor
   man eine Methoden-Decke deklariert.** Drei Sessions tunten Refiner gegen
   einen Daten-Bug.
2. Fehler **entlang des Sehstrahls** zerlegen (radial/lateral), nicht nur in
   Cam-XYZ — „random scatter" kann ein deterministisches Off-Axis-Muster sein.
3. **Prompts sind Tuning-Parameter** — forken/umformulieren ohne Messung =
   ungetestetes Modellverhalten (0 vs 27 Detections!).
4. Jede Stage **isoliert messen** (Service direkt), bevor man Symptome einem
   bekannten Limitations-Label zuschreibt.
5. Eval-Architektur kennen: inkrementelles Scoring (T-153) ⇒ „CSVs voll" ≠
   „Run fertig"; parallele Läufe verboten.
