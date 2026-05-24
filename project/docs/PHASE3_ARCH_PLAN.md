# PHASE 3 — Architektur-Hebel für mehr Pose-Performance (T-058 / S-049)

> **Ticket:** T-058 · **Worktree:** `.worktrees/S-049` · **Branch:**
> `team/2026-05-22-pose-bop-pivot/S-049` · **Date:** 2026-05-24
> **Frage (Max):** Bekommen wir *architektur-wise* — über Daten/Config + über den
> M2-Cascade hinaus — noch mehr Performance, parallel zum laufenden Training?
> **Geltungsbereich:** GDRNPP per-Objekt, **RGB-only** (Zivid-Depth auf Metall =
> hartes Nein), top-down, planarer Prior, texturloses Metall, eine RTX 3090.
>
> Begleiter: [`PHASE3_PEAK_PLAN.md`](PHASE3_PEAK_PLAN.md) (M1–M7-Recherche) ·
> [`REFINE_T041.md`](REFINE_T041.md) (M1/M3/M4 gebaut + gemessen = ausgereizt) ·
> [`M2_REFINER.md`](M2_REFINER.md) (Multi-Hyp-Render-Compare gebaut) ·
> [`PROJECT_REPORT.md`](PROJECT_REPORT.md) §4 (Fehler-Verteilung).

---

## 0. TL;DR — die ehrliche Antwort zuerst

**JA, es gibt einen echten Architektur-Hebel — aber er ist ein Retrain und
gehört NACH den laufenden Lauf, nicht davor.** Der training-freie Raum ist
nachweislich leer (T-041: nur Z-Snap hilft, M1/M3/M4 = 0). Der gebaute M2-Cascade
(Multi-Hyp + MegaPose-Score) ist der nächste validierbare Schritt und braucht nur
GPU-Inferenz. Was **architektur-wise** darüber hinaus geht und unsere Fehler an
der Wurzel trifft, ist **ein SO(3)-Rotations-Klassifikations-Kopf (SC6D-Stil)**
statt der aktuellen Einzel-Rotations-Regression — gegen das **Zahnrad im falschen
Becken**. Der ist **gescaffolded + getestet + ready-to-train**, aber **NICHT
trainiert** (GPU belegt).

**Was JETZT (ohne GPU-Job) gebaut + grün ist:**
- **TTA-Wrapper** (`project/tta_pose.py`) — inferenzseitig, RGB-only, hinter Flag.
  Validierbar erst am echten Checkpoint (GPU-frei). Erwartung: **klein** (+1..3 AR
  unimodal), löst NICHT Becken/Flip. 14 Unit-Tests grün.
- **SO(3)-Klassifikations-Kopf-Scaffold** (`box_src/so3_rotation_head.py` +
  Config + Train-Kommando + Integrations-Doku) — der EINE stärkste Architektur-
  Upgrade, Phase-3, **queued**. 11 Unit-Tests grün, Self-Check grün.

**Empfehlung in einem Satz:** *Erst die Ergebnisse des laufenden Retrains + den
M2-MegaPose-Cascade messen; PARALLEL den SO(3)-cls-Kopf bereithalten und ihn als
ERSTE GPU-Aktion fürs Zahnrad fahren, wenn der Retrain das Becken-Problem (wie
erwartet) nicht allein löst.* Ein voller Architektur-WECHSEL (SurfEmb/ZebraPose,
anderes Modell-Family) lohnt den Aufwand **derzeit nicht** — der SO(3)-Kopf holt
denselben Gewinn chirurgisch in GDRNPP.

---

## 1. Unsere Fehler (die Architektur muss DIESE treffen, nicht allgemeine)

Aus `PROJECT_REPORT.md` §4 + den val-Messungen (sym-bewusst, >20%-visib, n=1077):

| Teil | AR | Restfehler | Natur | training-frei lösbar? |
|---|---|---|---|---|
| Anker_Kurz | 0.645* | 13–19% ≥90°-Flips (Median 5–8° gut) | **bimodal**: echter 180°-3D-Flip (kein view-C_2, M3 self-IoU 0.58/0.64) | **NEIN** (T-041) |
| Anker_Lang | 0.650* | dito | dito | **NEIN** |
| Zahnrad | 0.360 | naive ~138°, 1% <5° | **falsches Becken**: C_7 in-plane ungelernt, Regression committet nicht auf EINEN Zahn | **NEIN** |

\* nach Z-Snap (der einzige training-freie Gewinn). Übergreifend: ~23% GT
unmatched (Occlusion-Tail) — das ist Recall, Sache des laufenden DR-Retrains (M6).

**Kern-Diagnose:** Beide Restfehler sind **bimodale/multimodale Rotations-
Verteilungen, in eine unimodale Regression gequetscht.** GDRNPP gibt pro View
EINE Rotation aus (`allo_rot6d`) und landet beim mehrdeutigen Teil zwischen den
Modi. **Das ist ein Repräsentations-Problem, kein Daten- oder Auflösungs-Problem**
(REFINE_T041: mehr Pixel/Stable-Snap/Contour-Yaw alle = 0 AR).

---

## 2. Recherchierte Architektur-Hebel (über Daten/Config + über M2 hinaus)

Jeder Hebel: Wirkung auf UNSERE Fehler · GPU/Retrain? · generalisierbar? ·
Integrations-Aufwand mit GDRNPP · Quelle. Geerdet an **ITODD** (BOP-Dataset, das
uns am nächsten ist: texturloses *Metall*, CAD-only) — RGB-only-Ceiling dort 46 AR
(GPose2023), niemand knackt 50 ohne Depth.

### A) Rotations-VERTEILUNGS-/Klassifikations-Köpfe (für Ambiguität)

| Methode | Mechanik | ITODD RGB AR | Sym-agnostisch? | Retrain? | GDRNPP-Integration |
|---|---|---|---|---|---|
| **SC6D** (3DV'22) | SO(3)-Einbettung + **Cosinus-Klassifikation** über N Anker; argmax statt Regression | **30.3** | **JA, ohne CAD-Sym** | ja | **M** (neuer Kopf-Zweig, Backbone bleibt) |
| **Implicit-PDF** (ICML'21) | nicht-parametr. **Dichte** auf SO(3), HEALPix-Grid; echte Multimodalität | — (Pascal3D/ModelNet-SOTA) | JA | ja | M–L (Dichte-Eval teurer) |
| **SymCode/SymNet** (2024) | one-to-**many** Korrespondenzen für diskrete Sym | T-LESS recall 38.5→**78.0** | für discrete-sym | ja | L (Korrespondenz-Decoder) |
| **ZebraPose** (CVPR'22) | hierarchischer **binärer Oberflächen-Code** | stark auf T-LESS | implizit | ja | L (anderer Geo-Head + PnP) |
| Wigner-D / I2S (2024) | äquivariante Harmonische auf SO(3) | — | JA | ja | L (äquivariantes Backbone) |

**Bester für UNS:** **SC6D-Kopf.** Begründung: (a) genau die Krankheit — *picks
the best of N sampled rotations* statt EINE zu regredieren -> löst „falsches
Becken"; (b) **symmetrie-agnostisch ohne CAD-Symmetrie** (das Embedding lernt die
C_7-Mehrdeutigkeit selbst — kein fragiles models_info-Hand-Tuning); (c) ITODD-Metall
30.3 AR ist die höchste sym-agnostische RGB-Zahl auf unserem Analog; (d) **kleinster
Eingriff** — er ersetzt nur den Rot-Zweig des PnP-Net, Backbone/Geo-Head/Translation
(der gelöste Teil) bleiben. **Erwarteter Impact:** Zahnrad 0.36 -> plausibel
**0.55–0.75** (Hypothese, geerdet an SC6D-ITODD; bis gemessen NICHT versprochen).
**Kosten:** ein per-Objekt-Retrain (~2.5h/Objekt auf der 3090). **→ GESCAFFOLDED.**

> Implicit-PDF ist die „reinere" Verteilungs-Formulierung, aber teurer und ohne
> Mehrwert gegenüber SC6D auf unserem diskreten Fall. Wir borgen daraus nur das
> **HEALPix-Anker-Grid** (im Scaffold drin) und halten SC6Ds Cosinus-Klassifikation.

### B) TTA — Test-Time-Augmentation (inferenzseitig, billig)

- **Mechanik:** den Crop in-plane um 90°/180°/270° drehen (exakte Pixel-Ops),
  je ein GDRNPP-Forward, die Augmentierung an der vorhergesagten Rotation
  rück-transformieren, auf SO(3) aggregieren.
- **Impact (Literatur):** typisch **+1..3 AR** für gut-gelöste Teile (härtet die
  View-Empfindlichkeit einer per-View-Regression). **LÖST NICHT** Becken/Flip —
  ein Mittelwert zwischen zwei Becken ist physikalisch falsch; deshalb bietet der
  Wrapper **Medoid**/**Score**-Aggregation (kein Verschmieren), nicht nur Mittel.
- **GPU/Retrain:** kein Retrain; nur N× Inferenz des bereits geladenen Netzes.
- **Generalisierbar:** ja, jedes Teil. **Integration:** **S** — Wrapper um
  `call_gdrnpp`, hinter Flag. **→ GEBAUT.** Quelle: Better Aggregation in TTA
  (arXiv:2011.11156); 6D-Aug (EfficientPose).

### C) Dense-Correspondence / symmetrie-native Alternativen

| Methode | ITODD RGB AR | vs GDRNPP | Aufwand |
|---|---|---|---|
| **SurfEmb** (CVPR'22) | **38.7** (+79% rel. ggü. Vor-RGB-SOTA) | **anderes Modell** (gelernte dichte Surface-Embeddings + Korrespondenz-Verteilung) | **L–XL** (eigenes Training + Inferenz-Stack) |
| **ZebraPose** | stark | anderer Geo-Head + PnP | L |

**Befund:** SurfEmb ist auf ITODD die stärkste RGB-Zahl (38.7 > SC6D 30.3) und
modelliert Symmetrie/Ambiguität nativ über Korrespondenz-Verteilungen. **Aber:**
es ist ein **Modell-Wechsel** (eigenes Training, eigener Inferenz-Pfad, eigene
Pipeline) — der Integrations-Aufwand ist um Größenordnungen höher als ein
Kopf-Tausch in GDRNPP, für +8 AR auf einem Benchmark, der nicht 1:1 unser Setup
ist. **Lohnt JETZT nicht** — der SC6D-Kopf holt den Großteil des Becken-Gewinns
in unserer bestehenden GDRNPP-Pipeline. SurfEmb bleibt die **Fallback-Option**,
falls der SC6D-Kopf plateaut.

### D) Ensemble / Backbone-Kapazität / instance-mask-crop

| Hebel | Impact auf unsere Fehler | Retrain? | Lohnt? |
|---|---|---|---|
| **Multi-Seed-Ensemble** | mittelt Varianz, **nicht** den systematischen Becken-Fehler | ja (N× train) | **Nein** — N× Kosten, falscher Fehlertyp |
| **convnext_base→large** | +Kapazität, aber Engpass ist Repräsentation, nicht Kapazität | ja, +VRAM | **eher nicht** — erst Kopf, dann ggf. Backbone |
| **instance-mask- statt bbox-Crop** | tighterer Crop -> stabilere Coarse-Z/Skala; Detektor mAP50 0.991 ist NICHT der Engpass | ja (Eingabe ändert sich) | **klein** (+Recall-Tail), nicht der Rot-Hebel |

Keiner davon trifft den Kern (bimodale Rotation). Multi-Seed mittelt sogar in die
falsche Richtung (wie der TTA-Mittelwert). Backbone-large erst, wenn der Kopf
sitzt und Kapazität nachweislich limitiert.

---

## 3. Gerankte Architektur-Hebel (Impact / Kosten / wann)

| Rang | Hebel | Impact auf UNSERE Fehler | Retrain/GPU | wann | Status |
|---|---|---|---|---|---|
| **1** | **SC6D SO(3)-cls-Kopf** (Zahnrad) | **Becken-Fix**, Zahnrad 0.36→~0.55–0.75 (Hyp.) | **Retrain** | **nach Retrain, erste GPU-Aktion** | **SCAFFOLD ✓** |
| **2** | **M2 MegaPose-Multi-Hyp** (gebaut, S-048) | Anker-Flip + Becken via gelerntem Score | GPU-Inferenz | beim Finish (GPU frei) | gebaut, finish-validierbar |
| **3** | **TTA-Wrapper** | unimodale View-Härtung (+1..3 AR) | nur Inferenz | beim Finish (GPU frei) | **GEBAUT ✓** |
| 4 | DR-Retrain (M6, in flight) | Occlusion/Recall-Tail | läuft | jetzt | in flight |
| 5 | SurfEmb (Modell-Wechsel) | nativ sym, ITODD 38.7 | Retrain+Stack | NUR wenn 1+2 plateauen | Fallback |
| 6 | Backbone large / instance-crop | Kapazität / Recall-Rand | Retrain | spät, wenn Kopf sitzt | nicht queued |
| — | Multi-Seed-Ensemble | mittelt Becken FALSCH | N× train | nicht empfohlen | verworfen |
| — | Depth (M7) | +24 AR ITODD | — | **hartes Nein (Regel)** | out of scope |

---

## 4. Empfehlung — Architektur-Wechsel? Ja/Nein, welcher, wann

**Architektur-WECHSEL (anderes Modell, SurfEmb/ZebraPose): NEIN — derzeit nicht.**
Der Aufwand (eigenes Training + Inferenz-Stack + Pipeline) steht in keinem
Verhältnis zu +8 AR auf einem nicht-identischen Benchmark, solange der billigere,
chirurgische Kopf-Tausch denselben Becken-Gewinn in GDRNPP verspricht.

**Architektur-UPGRADE (Kopf-Tausch in GDRNPP): JA — der SC6D-SO(3)-cls-Kopf.**
Aber mit klarer Reihenfolge, **nicht blind noch einen Retrain davorschieben**:

1. **JETZT:** laufenden Retrain (320/80, 8k DR, 160ep) zu Ende laufen lassen.
   Parallel (kein GPU): TTA + SO(3)-Kopf gebaut/gescaffolded (= dieser Ticket).
2. **Retrain fertig → messen:** neue val-AR gegen die 0.31-Baseline. Wenn das
   Zahnrad — wie erwartet (REFINE_T041, Template-Studie) — weiter im falschen
   Becken steht, ist die Diagnose bestätigt.
3. **Dann GPU-frei, in dieser Reihenfolge:**
   a. **M2 MegaPose-Multi-Hyp** validieren (gebaut, kein Retrain) — schnellster
      möglicher Becken-/Flip-Gewinn ohne Training.
   b. **TTA** A/B am echten Checkpoint (billig, kein Retrain).
   c. **Wenn Zahnrad-AR immer noch < Ziel:** den **SC6D-SO(3)-cls-Kopf fürs
      Zahnrad** trainieren (`train_so3cls_phase3.sh`), A/B gegen 0.36. Bei
      Erfolg auf Anker ausrollen.
4. **Fallback:** plateaut auch der SO(3)-Kopf → SurfEmb (Modell-Wechsel) ODER
   confidence-gefilterte Depth-als-Refiner-Constraint (M7, respektiert RGB-Regel).

**Ehrliche Einschätzung „lohnt der Aufwand vs. erst abwarten":** Der SO(3)-Kopf
ist der einzige Hebel, der das Zahnrad-Becken (den Teil, der das *overall* >0.80
entscheidet) an der Wurzel trifft — alles training-freie ist gemessen leer, M2 ist
ein gelernter Re-Scorer (gut, aber abhängig von einem Becken in der Hypothesen-
Menge), TTA ist klein. Also: **das Scaffolden lohnt sich (jetzt, ohne GPU-Kosten),
das Trainieren erst NACH dem Retrain + M2-Messung** — damit wir nicht einen teuren
Kopf-Retrain fahren, bevor wir wissen, ob M2 das Becken schon billig löst.

---

## 5. Was JETZT validierbar ist (wenig) vs. was queued wird

**JETZT validierbar (lokal, kein GPU):**
- TTA-Transform-Algebra (Round-Trip), Aggregation, Integration am MOCK — 14 Tests.
- SO(3)-Kopf: Anker-Mathematik, Forward/Decode/Loss, Self-Check — 11 Tests.
- Gesamtsuite grün (`pytest project/tests/`).

**Queued (braucht GPU-frei + echten Checkpoint):**
- TTA-A/B am realen GDRNPP-Checkpoint (echter AR-Delta — die Literatur-„+1..3" am
  echten Netz bestätigen/widerlegen). **Aktion:** `--tta --tta-agg medoid` im
  Inferenz-/Eval-Pfad, AR gegen Baseline.
- M2-MegaPose-Multi-Hyp-Finish (`e2e_finish.sh --refine-rc --rc-scorer megapose`).
- **SO(3)-Kopf-Training** (NACH M2-Messung): GDRNPP-Integration (3 Punkte,
  `box_src/SO3_INTEGRATION.md`) → `train_so3cls_phase3.sh --self-check → --smoke →
  --train` → A/B gegen Zahnrad-0.36.

**Nichts davon schiebt sich vor den laufenden Retrain** — alle GPU-Aktionen sind
*nach* ihm geplant; `train_so3cls_phase3.sh` bricht sogar hart ab, solange die GPU
belegt ist (kein versehentliches Davor-Schieben).

---

## 6. Gebaute Artefakte (dieser Ticket, S-049)

| Artefakt | Pfad | Status |
|---|---|---|
| TTA-Wrapper | `project/tta_pose.py` | gebaut, 14 Tests grün |
| TTA-Integration | `project/e2e_infer.py` (`--tta`, `estimate_poses(tta=…)`) | verdrahtet, Default AUS |
| TTA-Tests | `project/tests/test_tta_pose.py` | grün |
| SO(3)-cls-Kopf (Scaffold) | `box_src/so3_rotation_head.py` | gebaut, Self-Check grün |
| SO(3)-Kopf-Tests | `project/tests/test_so3_head.py` | 11 grün |
| Phase-3-Config-Stub | `box_src/configs_phase3/zahnrad_so3cls.py` | ready-to-deploy |
| Train-Kommando (GPU-guarded) | `box_src/train_so3cls_phase3.sh` | ready, bricht bei GPU-busy ab |
| GDRNPP-Integrations-Doku | `box_src/SO3_INTEGRATION.md` | 3-Punkte-Anleitung |

---

## Quellen (dieser Recherche-Runde, mit den oben genutzten Zahlen)
- **SC6D** (SO(3)-Einbettung + Cosinus-Klassifikation, sym-agnostisch; T-LESS 78.0,
  ITODD **30.3**, 5k train / 480k infer Anker, τ=0.1, ResNet34-UNet):
  https://arxiv.org/pdf/2208.02129 · Code: https://github.com/dingdingcai/SC6D-pose
- **Implicit-PDF** (HEALPix-äquivolumetrisches SO(3)-Grid, nicht-parametr. Dichte):
  https://arxiv.org/pdf/2106.05965 · https://implicit-pdf.github.io/
- **SurfEmb** (dichte Korrespondenz-Verteilungen; ITODD RGB **38.7**, +79% rel.):
  https://arxiv.org/pdf/2111.13489
- **SymCode/SymNet** (one-to-many CE; T-LESS discrete recall 38.5→78.0):
  https://arxiv.org/html/2405.10557v1
- **ZebraPose** (hierarchischer binärer Oberflächen-Code):
  https://openaccess.thecvf.com/content/CVPR2022/papers/Su_ZebraPose_Coarse_To_Fine_Surface_Encoding_for_6DoF_Object_Pose_CVPR_2022_paper.pdf
- **Wigner-D / Equivariant Pose Regression** (SO(3)-Harmonische):
  https://arxiv.org/html/2411.00543v1
- **Better Aggregation in TTA** (Mittel vs. gelernt; Aggregations-Caveats):
  https://arxiv.org/pdf/2011.11156
- **6D-Augmentation** (Rotation/Scale-Aug-Wirkung, EfficientPose):
  https://arxiv.org/pdf/2011.04307
- **Rotation Averaging** (chordaler/geodätischer Mittelwert auf SO(3), Hartley/Trumpf):
  https://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
- **BOP Challenge 2023** (RGB vs RGB-D Ceilings; ITODD RGB 46): https://arxiv.org/html/2403.09799v2
- **GDRNPP** (Architektur, allo_rot6d, get_rot_mat, detector→pose-Kopplung): https://arxiv.org/html/2102.12145v5

## Related
- [`PHASE3_PEAK_PLAN.md`](PHASE3_PEAK_PLAN.md) · [`REFINE_T041.md`](REFINE_T041.md) ·
  [`M2_REFINER.md`](M2_REFINER.md) · [`PROJECT_REPORT.md`](PROJECT_REPORT.md) ·
  `project/tta_pose.py` · `box_src/so3_rotation_head.py` ·
  `box_src/SO3_INTEGRATION.md` · `box_src/train_so3cls_phase3.sh` ·
  ADR-018 (BOP pivot) · ADR-017 (pose_result contract)
