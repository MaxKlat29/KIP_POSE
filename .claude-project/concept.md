# Concept — pose

**Profile:** ml  
**Created:** 2026-05-17  
**Schema:** v1  
**Initial pitch:** ML-Pipeline für 6D-Pose-Estimation von Bauteilen auf einer Tischplatte aus Top-Down-Bildern, mit Sim2Real-Training über Physics-Simulation und CAD-Modellen, für Roboterarm-Pick-and-Place. **Methoden-Stack (RGB-only): CNOS (Detection) + GigaPose (Coarse-Pose) + MegaPose (Refinement) + BlenderProc4BOP (Synth-Data) + Stable-Pose-Snap (Planar-Postprocessing).** Endprodukt: Top-Down-Bild → klassifizierte 3D-Modelle mit 6D-Pose auf simulierter Tischplatte (Three.js-Clickdummy).

> **Aktive Methoden-Entscheidung (2026-05-17):** RGB-only statt RGB-D. Hintergrund: vorhandene RGB-D-Daten sind nicht zuverlässig genug (Reflexionen, Metallteile, Depth-Noise) — wir wollen die Pipeline nicht auf instabile Depth-Maps stützen. Trade-off: ~10-15 AR-Punkte gegenüber RGB-D-SOTA (FoundationPose), kompensiert teilweise durch das starke Planar-Constraint (Objekt liegt auf bekannter Tischebene → effektiv 3DoF statt 6DoF). Stable-Pose-Snapping wird dadurch zum kritischen Stabilizer, nicht zum nice-to-have.

## TL;DR
Eine ML-Pipeline die aus einem Top-Down-Foto einer Tischplatte erkennt, welche bekannten Bauteile dort liegen und in welcher 6D-Pose — sodass ein Roboterarm die Teile greifen kann.

## Vision
> Eine ML-Pipeline die aus einem Top-Down-Foto einer Tischplatte erkennt, welche bekannten Bauteile dort liegen und in welcher 6D-Pose — sodass ein Roboterarm die Teile greifen kann.

## Problem
> Aktuell kann der Roboterarm Bauteile auf einer Tischplatte nicht autonom aufnehmen, weil ihm fehlt zu wissen WAS, WO und WIE GEDREHT die Teile liegen. CAD-Modelle aller Bauteile sind vorhanden, aber es gibt keinen Vision-Layer der diese mit einer Top-Down-Kamera-Aufnahme verheiratet. Klassische 2D-Detection reicht nicht — der Roboter braucht die volle 6D-Pose (3D-Position + 3D-Rotation) jedes Teils, um sicher greifen zu koennen.

## Erfolgsbild (6 Monate)
> Der Roboter sieht eine Tischplatte voller Schrauben, Muttern und Kleinteile, weiss innerhalb von <1s pro Frame welches Teil wo liegt, und kann sie eins nach dem anderen sicher aufnehmen. Endprodukt-Demo: ein Three.js-Clickdummy zeigt links das Kamerabild, rechts die rekonstruierte 3D-Szene mit CAD-Modellen exakt platziert — Top-Down-Bild rein, 3D-Welt-Rekonstruktion raus.

## Warum jetzt?
> Wir haben aktuell die Hardware (Roboterarm + 3D-Kamera + Tisch), die CAD-Bibliothek aller Bauteile, und bereits eine Physics-Simulation die uns Trainingsdaten mit Ground-Truth-Labels generiert. Was fehlt ist der ML-Layer dazwischen. SOTA-Methoden (FoundationPose, CNOS) sind 2024/2025 robust genug geworden, dass synthetic-only Training realistisch funktioniert.

## Zielgruppe
**Primärer User:** Roboterarm-Pipeline (downstream-Konsument), entwickelt von uns selbst. Im v0.1 ein Clickdummy-Operator der die Visualisierung anschaut.  
**Häufigster Use-Case:** Pro Pick-Zyklus ein Top-Down-Frame aufnehmen, Pipeline schickt {object_id, pose_4x4, confidence}-Liste an Greifplaner. Frequenz: einmal pro Pick-Zyklus, typischerweise alle 5-15s waehrend aktiver Arbeit.

**Sekundäre Use-Cases (post-v0.1):**
> Spaeter: kontinuierliches Multi-Frame-Tracking. Spaeter: Sim-to-Sim-Eval-Harness fuer neue CAD-Modelle. Spaeter: ROS2-Node-Wrapper. v0.1 ist Single-Shot pro Frame.

## Tech-Stack
- **Sprache:** python
- **Deployment:** local_cli
- **Privacy:** strictly_local
- **Bestehender Code:** Physics-Simulation in Blender/Python existiert bereits separat (CAD-Modelle auf Tischplatte fallen lassen, Ground-Truth-Pose-Labels generieren). Diese wird ueber BlenderProc4BOP-Standard ersetzt/angepasst.
- **Woher kommt das Modell:** use_open_source_as_is
- **Wo läuft die Inferenz:** local_gpu_workstation
- **Wie viel Daten:** medium_under_10m_rows
- **Wo liegen / sollen die:** parquet

## Must-Have-Features (v0.1)
> - **BlenderProc4BOP-Synth-Data-Pipeline:** CAD-Modelle auf simulierten Tisch fallen lassen, Top-Down-PBR-Rendering mit Domain-Randomization, BOP-Format-Output (Pose, Mask, RGB). Depth optional gerendert für spätere Erweiterung, aber **RGB ist Primary**.
> - **CNOS Detection + Segmentation** (CAD-template-driven, unseen-objects-fähig, RGB-only-fähig).
> - **GigaPose Coarse-Pose-Estimation** (RGB-only, 38× schneller als Alternative, +3.2 AP vs Baseline).
> - **MegaPose Refinement-Stage** (Render-and-Compare, hebt RGB-only-Predictions deutlich).
> - **Stable-Pose-Snapping-Postprocessing** (CAD-Analyse pro Bauteil: stabile Liegelagen via BlenderProc-Drop-Sim vorberechnen; predicted Pose auf nächste stabile Lage projizieren). **Wird zum kritischen Stabilizer für RGB-only-Pipeline.**
> - **Three.js-Clickdummy mit Mock-Daten (Phase X — früh):** funktionierende End-Visualisierung mit von Hand gebauten Demo-Daten (gerenderte Bilder + JSON-Poses), damit Endprodukt früh sichtbar. Echte Pipeline-Outputs werden später dran-gepluggt.
> - **Inference-CLI:** nimmt RGB-Image + Liste der CAD-Modelle, gibt JSON `{object_id, pose_4x4, confidence}` zurück.
> - **Eval-Harness:** vergleicht Predictions vs Ground-Truth aus Simulation (BOP-Metriken: AR, ADD, Translation-Error, Rotation-Error).

## Backlog (v0.2+)
> - Multi-Frame-Tracking über Zeit.
> - ROS2-Node-Wrapper.
> - Echte Live-Capture-Integration (Kamera-Treiber).
> - **FoundationPose RGB-D-Pfad als optionaler Upgrade-Path**, sobald Depth-Daten zuverlässig sind (besser für reflektierende Metallteile).
> - **FreeZeV2 als training-free Alternative** (BOP'24-Winner, höhere Accuracy, langsamere Inference).
> - WebGL-Live-Preview während Pipeline läuft.
> - Eigene Pose-Fine-Tuning falls synthetic-only zu schwach.

## Constraints
- **Zeit/Woche:** 5_to_10h
- **Deadline:** Keine harte Deadline. Konzept jetzt, Clickdummy in den naechsten Wochen, Production-Pipeline danach.
- **Wie strict muss Reproducibility sein:** seed_pinned_only

## Anti-Scope
> - Kein Multi-Object-Tracking über Zeit (v0.1 nur Single-Frame).
> - Keine eigene Pose-Refinement-Erfindung (wir nehmen MegaPose-Refiner).
> - Kein Greif-Planung-Algorithmus (out of scope).
> - Keine Cloud-Inference.
> - Keine Realtime-30fps-Anforderung (1s pro Frame reicht).
> - Kein eigenes Pose-Modell-Training v0.1 (wir nehmen vortrainiertes GigaPose+MegaPose).
> - **Keine RGB-D-Pipeline v0.1** (bewusste Entscheidung wegen unzuverlässiger Depth-Daten — bleibt aber Upgrade-Pfad v0.2+).

## Risks & Unknowns
> - **Sim2Real-Gap könnte größer sein als erwartet** (synthetic-only reicht eventuell nicht für reale Bilder). Mitigation: 200-500 reale Labels nachschießen wenn nötig.
> - **RGB-only ist 10-15 AR-Punkte schwächer als RGB-D-SOTA.** Mitigation: Planar-Constraint-Postprocessing (Stable-Pose-Snap) als Stabilizer.
> - **Texturarme / metallisch-spiegelnde Bauteile** sind klassischer RGB-Failure-Case. Mitigation: Domain-Randomization über Materialien, evtl. später Polarisations- oder Multi-View-Erweiterung.
> - **Stable-Pose-Snap braucht stabile-Liegelagen-Vorberechnung pro CAD-Modell** (one-time CAD-Analysis via N=200 BlenderProc-Drops).
> - **Kamera-Intrinsics & Tisch-Ebenen-Calibration** müssen sauber bestimmt werden (sonst projiziert Stable-Pose-Snap auf falsche Ebene).
> - **GigaPose/MegaPose haben CUDA-Abhängigkeiten** — RTX 3090 Workstation als Inference-Target (ist verfügbar).

## Success Metrics
- **Quant:** BOP Average Recall >= 0.70 auf eigenem Test-Set. Translation-Error Median < 5mm. Rotation-Error Median < 5 Grad. Inference < 1s pro Frame. Detection-Recall >= 0.90 fuer Teile >= 50px.
- **Qual:** Im Clickdummy lade ich ein Top-Down-Bild und sehe rechts eine 3D-Szene wo alle CAD-Modelle sichtbar korrekt platziert sind — ich kann das Three.js-Modell rotieren und visuell pruefen ob die Pose stimmt. Bei 10 Test-Bildern sind 8+ visuell ueberzeugend.

## Evaluation
- **Woher kommt das Eval-Dataset:** holdout_split_of_train
- **Brauchst du Human-Review-Loops (Annotation, Spot-Check):** occasional_spot_check

## Open Questions
- **CAD-Lieferung:** Max liefert echte Industrieteile-CADs nach. v0.1-Clickdummy nutzt Demo-Daten (T-LESS-Subset oder selbst-gerenderte Beispieldaten).
- **Tisch-Material:** matt/glänzend? Beeinflusst Domain-Randomization-Strategie.
- **Kamera-Modell+Intrinsics:** konkretes Kamera-Setup noch zu spezifizieren (Position über Tisch, Brennweite, Auflösung).
- **Anzahl Bauteile pro Frame:** typisch 1-5 oder eher 10-50 (Bin-Picking-artig)?
- **Stable-Lagen-Schwelle:** wie viele diskrete Lagen pro Bauteil zulassen (top 3? top 6?) — beeinflusst Snap-Aggressivität.

## Glossary
- **6D-Pose:** 3D-Position (x,y,z) + 3D-Rotation (3 Achsen) = 6 Freiheitsgrade.
- **BOP:** Benchmark for 6D Object Pose Estimation (Brno), Standard-Metrik AR (Average Recall).
- **CNOS:** CAD-based Novel Object Segmentation, kombiniert SAM + DINOv2 für Detection ohne pro-Objekt-Training.
- **GigaPose:** RGB-only Coarse-Pose-Estimator (CVPR'24, NV-LIONS).
- **MegaPose:** RGB(-D) Render-and-Compare Pose-Refiner (CoRL'22).
- **BlenderProc4BOP:** PBR-Rendering-Toolkit mit PyBullet-Physics für BOP-konforme Synth-Data.
- **Stable-Pose-Snap:** Postprocessing das eine predicted Pose auf die nächste stabile Liegelage auf der Tischebene projiziert (CAD-vorberechnete Set diskreter Lagen).
- **Sim2Real-Gap:** Performance-Lücke zwischen synthetic-trained Model und realer Inference, klassisches Problem in der Robotik-Vision.
