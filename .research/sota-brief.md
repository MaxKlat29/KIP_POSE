# 6D-Object-Pose-Estimation — SOTA-Brief 2025/2026

**Kontext:** Roboterarm pickt Bauteile (Schrauben, Muttern, Kleinteile) von einer flachen Tischplatte.
Top-Down-Kamera, CAD-Modelle aller Teile vorhanden, Physics-Sim für Trainingsdaten.
Ziel: 2D-Top-Down-Bild → klassifiziere Objekte + schätze 6D-Pose → CAD-Modelle in 3D-Szene platzieren.
Stand: 2026-05-17.

---

## 1. BOP-Leaderboard — wer führt?

Das **BOP-Benchmark** (Brno) ist die einzige ernstzunehmende Vergleichsbasis. Relevant für uns: Task **"Model-based 6D Localization of Unseen Objects"** (CAD bekannt, Objekt nicht im Training gesehen — passt zu "neue Bauteile später hinzufügen").

**Top der Leaderboard ([core-datasets-Tabelle](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/), Stand 2026-03):**

| Rang | Methode | Score (AR) | Input | Datum |
|---|---|---|---|---|
| 1 | WAPR.v2 (Multi-Det) | 0.845 | RGB-D | 2026-03 |
| 2 | FRTPose-WAPR.v2 | 0.844 | RGB-D | 2025-10 |
| 3 | FRTPose-WAPR.v2 (Default) | 0.837 | RGB-D | 2025-10 |
| ~ | FreeZeV2.1 | ~0.833 | RGB-D | 2024 (BOP'24 Winner) |
| ~ | FoundationPose | ~0.720 | RGB-D | 2024 |

**Take-aways:**
- Die Top-Methoden nutzen **alle RGB-D**. RGB-only ist 10-15 AR-Punkte schwächer.
- **FreeZeV2** ([Paper](https://arxiv.org/abs/2506.09784), [Repo](https://github.com/andreacaraffa/freeze)) gewann BOP'24 (Best Overall) — training-free, nutzt nur frozen DINOv2 + 3D-Geo-Foundation-Modelle + RANSAC-Registrierung. Stark, aber rechenintensiv.
- **WAPR / FRTPose** dominieren BOP'25 — beides relativ neue Methoden (kaum Awareness außerhalb Benchmark), entstanden im BOP-Challenge-Ökosystem.

**Relevante BOP-Datasets für unseren Use-Case:** T-LESS (texturearme Industrieteile, Tisch-Setup), ITODD (industrielle, dunkle/metallische Teile), IC-BIN (Bin-Picking, dichte Cluster), LM-O (Occlusion-Variante). T-LESS und ITODD sind die nächsten Analoga zu "Schrauben/Muttern auf Tisch".

---

## 2. SOTA-Methoden 2024-2026 im Überblick

| Methode | Jahr | Input | Trainings-Modus | Stärke | Einsatzempfehlung |
|---|---|---|---|---|---|
| **FoundationPose** ([CVPR'24 Highlight](https://nvlabs.github.io/FoundationPose/), [Repo](https://github.com/NVlabs/FoundationPose)) | 2024 | RGB-D | "Trained once" + CAD | Generalisiert auf unseen objects, Top-Pick für Robotik (Isaac ROS-Integration) | **PRAGMATISCHER START** |
| **SAM-6D** ([CVPR'24](https://arxiv.org/html/2311.15707v2), [Repo](https://github.com/JiehongLin/SAM-6D)) | 2024 | RGB-D | Zero-shot | Koppelt SAM-Segmentation mit 2-Stage Point-Matching, 69.9% AR auf LM-O vs. MegaPose 49.9% | Solide RGB-D-Alternative |
| **MegaPose** ([CoRL'22](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf)) | 2022 | RGB(-D) | Render-and-Compare | Erprobt, gute Refinement-Stufe | Refinement-Add-On |
| **GigaPose** ([CVPR'24](https://nv-nguyen.github.io/gigapose/)) | 2024 | RGB | Template+Patch | 38× schneller als alternative coarse stages, +3.2 AP | **RGB-only Pick** |
| **GenFlow** | 2024 | RGB | Refinement | BOP'23 RGB Winner | Combo mit GigaPose |
| **GDRNPP / FAST-GDRNPP** ([arxiv 2024](https://arxiv.org/html/2409.12720v1)) | 2022/24 | RGB(-D) | Per-Objekt trainiert | Hohe Präzision bei bekannten Objekten | Wenn Objekt-Set fix |
| **FreeZeV2** ([arxiv'25](https://arxiv.org/abs/2506.09784)) | 2025 | RGB-D | **Training-free!** | BOP'24-Winner, kein Training nötig | Ambitionierte Variante |
| **Any6D** ([CVPR'25](https://github.com/taeyeopl/Any6D)) | 2025 | RGB-D | Model-free | Braucht nur **eine** Referenz-RGB-D statt CAD | Wenn CAD fehlt |
| **Pos3R** ([CVPR'25](https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Pos3R_6D_Pose_Estimation_for_Unseen_Objects_Made_Easy_CVPR_2025_paper.pdf)) | 2025 | RGB | Diffusion-prior | Einfache, neue Baseline | Experimentell |
| **UnPose** ([arxiv'25](https://arxiv.org/abs/2508.15972)) | 2025 | RGB-D | Diffusion + 3DGS | Zero-shot ohne CAD | Forschungsstadium |
| **PVNet** | 2019 | RGB | Voting | Klassiker, aber abgehängt | Nicht mehr empfehlenswert |
| **OnePose++** | 2022 | RGB | CAD-frei | Nice falls kein CAD | Wir haben CAD → skip |

**Trend 2025/2026:** Drei Linien dominieren —
1. **Foundation-Model-based** (FreeZe, FoundationPose) — generalisieren auf unseen objects ohne pro-Objekt-Training.
2. **Diffusion-Priors** (UnPose, Pos3R, Any6D) — generieren 3D-Repräsentation aus wenigen Views, noch experimentell.
3. **CAD + Render-and-Compare** (MegaPose, GigaPose) — robust, gut etabliert.

**RGB vs. RGB-D:** RGB-only ist machbar (GigaPose+MegaPose-Refinement ist die stärkste RGB-Combo), kostet aber ~10-15 AR-Punkte gegenüber RGB-D. Für Robotik-Greifen sehr empfohlen: Tiefenkamera dazu (Realsense D435 / D455 reicht).

---

## 3. Sim2Real-Pipeline

**Tool-Empfehlung:**
- **BlenderProc2** ([Repo](https://github.com/DLR-RM/BlenderProc), speziell [BlenderProc4BOP](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md)) ist **De-facto-Standard** für BOP-style Trainingsdaten. PBR-Rendering, PyBullet-Physics-Drop, BOP-kompatibles Annotations-Format. Genau auf unseren Use-Case zugeschnitten.
- **NVIDIA Isaac Sim / Replicator** — stärker, wenn man Roboterarm und Sim2Real-Closing-the-Loop in einer Engine will. Mehr Setup-Overhead. Sinnvoll später, sobald wir auch den Greif-Planner trainieren.
- **Cosmos-Predict2** (NVIDIA, 2025) kann BlenderProc-Outputs in video-konsistente Sequenzen umwandeln — für statisches Pose-Training nicht relevant.

**Sample-Volumen (Best-Practice):**
- BOP-Standard: **50.000 PBR-Bilder pro Dataset** ([Quelle](https://github.com/DLR-RM/BlenderProc/issues/478))
- Für 5-10 Bauteile auf einem Tisch reichen erfahrungsgemäß **25-50k synthetische Bilder**.
- Domain-Randomization-PFLICHT: PBR-Materialien randomisieren (nicht versuchen, das echte Material exakt zu modellieren — randomisieren funktioniert besser), Lichter, HDRI-Backgrounds, Kameraposition leicht variieren (auch wenn nominal Top-Down).
- **PhysiSim-Drops:** PyBullet-Drops in BlenderProc liefern realistische Lagen — perfekt für "Schrauben fallen auf Tisch".

**Synthetic-only vs. Real-Fine-Tuning:**
- FoundationPose, FreeZe, SAM-6D, GigaPose generalisieren **ohne** Real-Daten — synthetic-only reicht für sie meist.
- GDRNPP profitiert deutlich von ~500-2000 echten Real-Bildern mit Annotations (15% Improvement-Bereich gemäß [Synthetic-Sim2Real-Paper 2023](https://arxiv.org/html/2311.11039v2)).
- Pragmatisch: synthetic-only starten, erst bei Bedarf 200-500 reale Labels nachschießen.

---

## 4. Architektur-Empfehlung (Standard-Stack)

```
[Top-Down-RGB(-D)-Image]
        │
        ▼
Stage 1: Detection + Segmentation
   ├─ CNOS (CAD-basiert, SAM+DINOv2) — unseen-objects-ready
   └─ Alternativ: YOLOv8/9 per Objekt trainiert (schneller, fixe Klassen)
        │
        ▼  (Crops + Masken pro Detection)
Stage 2: Coarse 6D-Pose Estimation
   ├─ FoundationPose (RGB-D)
   └─ Alternativ RGB-only: GigaPose
        │
        ▼  (Pose-Hypothesen)
Stage 3: Refinement
   ├─ FoundationPose-Refinement (eingebaut)
   ├─ MegaPose-Refiner
   └─ ICP auf Depth (klassisch, fast geschenkt bei RGB-D)
        │
        ▼
[Liste: {object_id, pose_4x4, confidence}]
        │
        ▼
Szenen-Rekonstruktion → CAD-Platzierung → Greifplaner
```

**Single-Stage-Alternativen:** GDR-Net, PVNet — schwächer auf unseen objects, nur bei festem Bauteil-Set sinnvoll. Für unseren Fall (Bauteil-Sets erweiterbar) ist die **Two-Stage-Pipeline mit CNOS+FoundationPose** klar überlegen.

---

## 5. Planar/Top-Down-Spezifika

Der Constraint "Objekt liegt stabil auf bekannter Ebene" reduziert effektiv von 6DoF auf **3DoF (x, y, yaw)**, weil:
- z = Tischplatte + halbe Objekthöhe (aus CAD bekannt, abhängig von Liegelage)
- roll, pitch = diskret aus "stabilen Liegelagen" des Objekts (CAD-Analyse: meist 1-6 stabile Lagen pro Bauteil)

**Praktische Konsequenz:**
- Es gibt keine SOTA-Methode, die diesen Constraint explizit nativ ausnutzt — alle modernen Pipelines lernen full-6D.
- ABER: man kann den Constraint nachgelagert **als Postprocessing-Refinement** anwenden ("snap to nearest stable resting pose"). Das ist robust und stabilisiert wackelige Predictions drastisch.
- Vor-Berechnung pro Bauteil: stabile Liegelagen via BlenderProc-Drops aus N=200 Simulationen → diskrete Set von erlaubten (roll, pitch, z). Bei Inference: predicted Pose auf nächste stabile Lage projizieren.
- Klassisches "2D planar grasp"-Framing (siehe [vision-based-robotic-grasping survey](https://github.com/GeorgeDu/vision-based-robotic-grasping)) bleibt also als Output-Layer relevant, auch wenn intern 6D geschätzt wird.

---

## 6. Konkrete Empfehlung

### Top-Pick (pragmatischer Start)
**FoundationPose** (NVIDIA) + **CNOS** (Detection) + **BlenderProc2** (Synth-Data)

**Warum:**
- Code Open-Source, gut dokumentiert, Isaac-ROS-Integration vorhanden.
- Generalisiert auf unseen objects mit CAD — perfekt für "Bauteile später dazu".
- RGB-D-Setup mit Realsense D435 (~250 EUR) hebt Genauigkeit signifikant.
- Keine Per-Objekt-Trainings nötig — Onboarding eines neuen Bauteils = CAD reinwerfen, fertig.
- CNOS macht den Detection-Step ebenfalls CAD-driven und unseen-object-fähig.

**Training-Zeit auf RTX 3090 (24GB):**
- FoundationPose-Model selbst ist **vortrainiert** — wir trainieren es **nicht neu**, nur Onboarding pro CAD (Sekunden).
- BlenderProc-Dataset-Generation für 10 Bauteile, 30k Bilder: **~8-12 Stunden RTX 3090**.
- Optional: CNOS-Templates rendern (einmalig pro CAD, ~5 min).
- Falls Fine-Tuning später nötig: ~6-10h pro Run.

### Ambitionierte Alternative
**FreeZeV2** + **CNOS** — wenn FoundationPose zu schwach oder zu stark NVIDIA-gebunden.

**Warum:**
- BOP'24-Winner, höchste Accuracy am Markt.
- **Training-free** — keine GPU-Hours für Pose-Modell, nur Inference-Compute.
- Modular ensemble-fähig.
- Nachteil: langsamer in Inference (sekunden-skaliert pro Frame), für reine Pick-Planung aber OK.

### Kompletter RGB-only-Pfad (Fallback ohne Tiefenkamera)
**GigaPose + MegaPose-Refinement** + **CNOS** — die stärkste RGB-only-Kombination laut [GigaPose-Paper](https://arxiv.org/html/2311.14155v2).

### Was wir NICHT empfehlen
- **PVNet, OnePose** — abgehängt.
- **Per-Objekt-spezifische Methoden (GDR-Net pur)** — skaliert nicht mit wachsender Bauteilbibliothek.
- **Diffusion-basierte (UnPose, Pos3R)** — zu experimentell für Produktion 2026.

---

## Konkreter Next-Step-Vorschlag

1. **Realsense D435 oder D455** beschaffen (RGB-D, Top-Down über Tisch montieren).
2. **BlenderProc4BOP-Pipeline** aufsetzen, 5 Test-Bauteile als CAD reinwerfen, 30k Bilder rendern (~10h auf 3090).
3. **CNOS** lokal aufsetzen (Detection auf CAD-Templates, BOP-Format-kompatibel).
4. **FoundationPose** via Isaac-ROS oder pure-PyTorch-Repo lokal, mit CNOS-Detections gepipelined.
5. **Stable-Pose-Snapping** als Postprocessing-Layer für Planar-Constraint.
6. Evaluation auf 100 echten Tisch-Aufnahmen mit manuellen Pose-Labels — wenn AR > 0.6, Pipeline reicht für Greifplanung.

**Erwartete Endperformance:** AR ≈ 0.70-0.85 auf eigenem Datensatz (T-LESS-ähnliche Industrieteile sind FoundationPose's Stärke). Real-World-Pose-Genauigkeit: meist <5mm Translation, <5° Rotation — mehr als ausreichend für Greifplanung mit Toleranzen.

---

## Quellen

- [BOP Leaderboard](https://bop.felk.cvut.cz/leaderboards/) | [BOP Challenge 2024](https://bop.felk.cvut.cz/challenges/)
- [FoundationPose (CVPR'24)](https://nvlabs.github.io/FoundationPose/) | [Repo](https://github.com/NVlabs/FoundationPose)
- [SAM-6D (CVPR'24)](https://arxiv.org/html/2311.15707v2) | [Repo](https://github.com/JiehongLin/SAM-6D)
- [GigaPose (CVPR'24)](https://nv-nguyen.github.io/gigapose/) | [Repo](https://github.com/nv-nguyen/gigapose)
- [MegaPose (CoRL'22)](https://proceedings.mlr.press/v205/labbe23a/labbe23a.pdf)
- [FreeZeV2 (2025)](https://arxiv.org/abs/2506.09784) | [Repo](https://github.com/andreacaraffa/freeze)
- [CNOS (ICCV'23 R6D)](https://nv-nguyen.github.io/cnos/) | [Repo](https://github.com/nv-nguyen/cnos)
- [GDRNPP / FAST-GDRNPP](https://arxiv.org/html/2409.12720v1)
- [Any6D (CVPR'25)](https://github.com/taeyeopl/Any6D)
- [Pos3R (CVPR'25)](https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Pos3R_6D_Pose_Estimation_for_Unseen_Objects_Made_Easy_CVPR_2025_paper.pdf)
- [UnPose (2025)](https://arxiv.org/abs/2508.15972)
- [BlenderProc4BOP](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md)
- [Awesome Object Pose Estimation Survey (IJCV'26)](https://github.com/CNJianLiu/Awesome-Object-Pose-Estimation)
- [Vision-based Robotic Grasping Survey](https://github.com/GeorgeDu/vision-based-robotic-grasping)
- [Sim2Real Production Paper](https://arxiv.org/html/2311.11039v2)
