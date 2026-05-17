# Roadmap Rationale — ML / CV / 6D-Pose-Estimation (POSE)

_Generated at 2026-05-17T13:19:12Z by step `roadmap_plan`._

## Why this phase-set

Vision-Pipeline mit Sim2Real-Training. Reihenfolge: Foundation → Clickdummy-v0-mit-Mock-Daten (frühe End-Visualisierung) → Synth-Data-Pipeline → Detection → Pose-Estimation → Refinement → Stable-Pose-Snap → Eval → Clickdummy-v1-mit-echten-Daten → Release. Clickdummy zuerst mit Mock-Daten, damit das Endprodukt früh sichtbar ist und das Datenformat (JSON-Pose-Schema) sich frühzeitig zementiert.

## Detected profile

- `profile_id`: **ml-cv-pose**
- source: `/users/admin/pose/.claude-project/pose-phase-set.json`

## Concept signals that influenced this choice

_(no concept answers available — fallback default profile selected)_

## Gap-analysis influence

# Gap-Analysis — pose

**Generated:** 2026-05-17

## Status
**Greenfield Project** — nothing to compare yet. This file is a placeholder; re-run `gap_analysis` once the project has actual code.

> Marker: `greenfield: true`

## Phases

- **P1** (Foundation + Repo-Setup) — target `0.1` — Python-Env (CUDA, PyTorch), Repo-Struktur, Test-Fixtures, CAD-Loader, JSON-Pose-Schema, Coord-System-Konvention.
  - hint: Repo-Setup mit Python-Env + CUDA + PyTorch + uv
  - hint: CAD-Loader (STL/OBJ/PLY → trimesh)
  - hint: JSON-Output-Schema-Spec ({object_id, pose_4x4, confidence})
  - hint: Coord-System + Camera-Intrinsics-Datentypen
  - hint: Test-Fixtures (Demo-CADs + Beispiel-Images)
- **P2** (Clickdummy v0 (Mock-Daten)) — target `0.1` — Three.js-Viewer mit selbst-gebauten Beispieldaten. Frühe End-Visualisierung. Side-by-Side Image-links + 3D-rechts. Zementiert das Datenformat.
  - hint: Three.js Bootstrap mit Vite + CAD-Loader (STL)
  - hint: Side-by-Side Layout: Image links, 3D-Scene rechts
  - hint: Mock-JSON Loader (Image + Pose-Liste)
  - hint: OrbitControls + Zoom + Grid + Light-Rig
  - hint: Mock-Data-Generator: 3-5 Demo-Frames mit handgesetzten Posen
  - hint: Snap-to-Stable-Pose Animation (Demo-Effekt: Pose snapped sichtbar zur Lage)
- **P3** (Synth-Data-Pipeline (BlenderProc4BOP)) — target `0.1` — CAD-Modelle auf simulierten Tisch fallen lassen, PBR-Rendering, Domain-Randomization, BOP-Format-Output. Ersetzt/wrappet bestehende Physics-Sim.
  - hint: BlenderProc4BOP Installation + Hello-World-Render
  - hint: CAD-Drop-Sim mit PyBullet-Physics (N=200 pro CAD)
  - hint: PBR-Material- + HDRI-Background-Randomization
  - hint: BOP-Format-Export (Pose, Mask, RGB, Camera-Intrinsics)
  - hint: Dataset-Generation-CLI (--n-frames, --cads, --out)
  - hint: Stable-Lagen-Cluster-Vorberechnung pro CAD (Output: <cad-id>-stable-poses.json)
- **P4** (Detection-Stage (CNOS)) — target `0.1` — CAD-template-driven Detection + Segmentation, RGB-only, unseen-objects-fähig.
  - hint: CNOS Repo + Dependencies + DINOv2-Weights
  - hint: CAD-Template-Rendering pro Bauteil (multi-view)
  - hint: CNOS Inference auf Synth-Test-Set
  - hint: BOP-conformer Detection-Output (BBox + Mask + Class + Conf)
  - hint: Detection-Eval-Script: Recall@k, Precision@IoU
- **P5** (Pose-Estimation (GigaPose + MegaPose)) — target `0.1` — RGB-only Coarse-Pose mit GigaPose, gefolgt von MegaPose-Refinement. Stärkste RGB-only-Combo laut SOTA-Brief.
  - hint: GigaPose Repo + Pretrained-Weights
  - hint: GigaPose-Inference auf CNOS-Detections (Crops)
  - hint: MegaPose-Refinement-Integration (Render-and-Compare)
  - hint: End-to-End-Pipeline-Verkettung (Detect → Coarse → Refine)
  - hint: Pose-Eval auf Synth-Test-Set (Translation/Rotation-Error)
- **P6** (Stable-Pose-Snap (Planar-Postprocessing)) — target `0.1` — Predicted Pose auf nächste stabile Liegelage auf bekannter Tischebene projizieren. Stabilizer für RGB-only-Pipeline.
  - hint: Stable-Lagen-Cluster aus P3 laden (top-K diskrete Lagen)
  - hint: Tisch-Ebenen-Calibration aus Camera-Intrinsics + Setup
  - hint: Snap-Algorithmus: predicted Pose → argmin(dist) zur Stable-Lage
  - hint: Yaw-only-Refinement um Z-Achse nach Snap
  - hint: Ablation-Study: mit/ohne Snap auf Eval-Set (AR-Vergleich)
- **P7** (Eval-Harness + Reports) — target `0.1` — BOP-Metriken (AR, ADD, MSPD, MSSD), Translation/Rotation-Error, Reports, Worst-Examples-Viewer.
  - hint: BOP-Eval-Toolkit-Integration
  - hint: Test-Set-Aufbau (held-out aus Sim, optional 50 echte Frames)
  - hint: Metrics-CSV + JSON-Report Generator
  - hint: Error-Analysis: worst-50-examples HTML-Report
  - hint: Sim-vs-Real-Performance-Vergleich (falls Real-Daten verfügbar)
- **P8** (Clickdummy v1 (Echte Daten)) — target `0.1` — Pipeline-Outputs an Clickdummy gepluggt. File-Drop → Inference → Live-3D-Rekonstruktion.
  - hint: Backend-API: Image-Upload → Pipeline-Inference → JSON-Response
  - hint: Frontend: Drag-Drop-Upload → Trigger Pipeline → Live-Render
  - hint: Loading-State + Error-Handling + Confidence-Coloring
  - hint: Demo-Set mit 10 vorgefertigten Test-Frames (One-Click-Demos)
  - hint: End-to-End-Walkthrough-Recording (Screencast für README)
- **P9** (Integration + Docs + v0.1-Release) — target `0.1` — CLI-Wrapper, ADRs finalisieren, README, Docs, v0.1.0-Tag + Brain-Snapshot.
  - hint: End-to-End-CLI: 'pose infer <image> --cads <dir>'
  - hint: ADRs finalisieren (Methoden-Wahl, RGB-only, Stable-Snap, BlenderProc)
  - hint: README mit Quickstart + Architektur-Diagramm + Demo-GIF
  - hint: Docs: Pipeline-Stages, Eval-Reproduktion, Bauteil-Onboarding
  - hint: v0.1.0-Release-Tag + Brain-Snapshot

---

_You can edit the roadmap any time with the `gantt` CLI; this file is a one-shot record of the bootstrap choice and is not kept in sync afterwards._