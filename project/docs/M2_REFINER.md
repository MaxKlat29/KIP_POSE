# M2 — Multi-Hypothesis Render-and-Compare-Refiner (T-058) — gebaut, ehrlich gemessen

> **Date:** 2026-05-24 · **Ticket:** T-058 · **Worktree:** `.worktrees/S-048` ·
> **Module:** `project/refine_rc.py` (Generator + CPU-Scorer + MegaPose-Contract) ·
> **Integration:** `project/e2e_infer.py` + `bop_adapter.detection_to_result(rc_refiner=…)` ·
> **Box-Harness:** `box_src/rc_refine_eval.py` (CPU-Validierung) ·
> `box_src/real_pose_result.py`
> (`--refine-rc`) · `box_src/e2e_finish.sh` (`--refine-rc --rc-scorer …`)

---

## Warum M2 (Kontext)

Die training-freien Geometrie-Priors (M1 Stable-Pose-Snap, M3 C_2-Check, M4
Contour-Yaw-Lock — T-041) sind **ausgereizt**: nur der Z-Snap bringt AR. Die zwei
harten Restfehler sind **lokal nicht snapbar**, weil sie GLOBAL-mehrdeutig sind:

- **Anker:** echte 180°-3D-Flips (Stab Kopf-über-Schaft). Kein view-Symmetrie-C_2
  (M3: Top-Down-Self-IoU 0.58/0.64 ≪ 1 — Kopf und Schaft sind top-down
  unterscheidbar, der Flip ist ein echter 3D-Fehler).
- **Zahnrad:** falsches Rotations-Becken (~138° daneben), keine Tilt-/Yaw-Drift.

Ein lokaler Refiner kommt aus einem falschen Becken nicht heraus. Ein
**Multi-Hypothesen-Render-and-Compare** schon: erzeuge mehrere Kandidaten-Posen,
**rendere** die CAD an jeder, **vergleiche** mit dem RGB-Crop, nimm die beste.

---

## Mechanik (was gebaut wurde)

### 1) Hypothesen-Generator (`generate_hypotheses`)
Pro Detektion aus der GDRNPP-Coarse-Welt-Rotation R0 (nach `bop_adapter`,
**vor** dem Z-Snap):

| Kandidat | adressiert | Quelle |
|---|---|---|
| Coarse (Index 0, Gate-Referenz) | Fallback | R0 |
| 180°-Flips um die Body-Hauptachsen | Anker End-über-End-Flip | `flip_axes` |
| C_N-Yaw-Varianten (k·2π/N) | Zahnrad-Becken | `n_fold` aus models_info |
| K stabile Ruhelagen | richtige Auflagefläche | `stable_pose_body_downs` (M1) |
| Tilt-Varianten (±°) | Becken-Nachbarn | `tilt_degs`/`tilt_axes` |

Generalisierbar (Achsen/N/Ruhelagen aus models_info + CAD), dedupliziert
(geodätischer SO(3)-Winkel < `dedup_tol_deg`), R0 bleibt immer Index 0.
Unit-getestet: korrekte Flip/Yaw/Tilt/Ruhelagen-Kandidaten, Dedup, max-Grenze.

### 2a) CPU-Kanten/Silhouetten-Scorer (`cpu_edge_score`, kein GPU)
Pro Hypothese: rendere die CAD-Silhouette in den Kamera-Frame (re-nutzt
`bop_adapter._camera_silhouette`), score = `w_iou·IoU(Silhouette, Detektor-Maske)`
+ `w_chamfer·Chamfer(Silhouetten-Kontur, Bildkanten des RGB-Crops)`. Metall hat
starke Kanten (ContourPose-Idee). `select_best_hypothesis` gated gegen die Coarse
(`min_margin`): nur wechseln, wenn die beste Hypothese die Coarse klar schlägt.

### 2b) MegaPose-RGB-Scorer (`megapose_score`, GPU — NotImplemented post-Phase-2)
MegaPose (`/mnt/data/bop/repos/megapose6d`, megapose-1.0-RGB) nimmt jede Hypothese
als `TCO_init`, refined sie `forward_refiner(n_iterations=N)` (render-vs-RGB),
liefert refinte Posen + einen **gelernten Score** pro Hypothese; beste gewinnt,
zurück in den Welt-Frame (`bop_adapter.bop_pose_to_world`). Der **gelernte** Score
ist (anders als der CPU-Score) in der Lage, den Flip/das Becken zu trennen.

### 3) Integration
Optionale Stufe **NACH GDRNPP-Coarse + Welt-Transform**, **VOR dem Z-Snap**,
hinter Flag `refine_rc` (Default AUS). Im Adapter via `detection_to_result(
rc_refiner=callback)`; in `e2e_infer.py` via `--refine-rc [--rc-scorer cpu_edge|
megapose]`; im Finish via `e2e_finish.sh --refine-rc [--rc-scorer …]`. RGB-only,
Contract unverändert.

---

## Ehrliche Messung — CPU-Kanten-Scorer auf den BESTEHENDEN val-Predictions

Gemessen an `val_preds_combined.csv` (dieselben echten GDRNPP-Predictions), mit
dem realen CAD + den realen `mask_visib`-Masken + den realen RGB-Crop-Kanten,
symmetrie-bewusst + >20%-gefiltert (`eval_bop.py`, GT n=1077). Z-Snap immer an
(geshippte Baseline). Harness: `box_src/rc_refine_eval.py`.

| Config | Anker_Kurz AR | Anker_Lang AR | Zahnrad AR | overall AR | switched |
|---|---|---|---|---|---|
| **raw (Z-Snap, kein RC) = BASELINE** | **0.645** | **0.650** | **0.360** | **0.331** | — |
| rc_anker (offenes Gate, margin 0) | 0.533 | 0.577 | 0.360 | 0.294 | 460/519 |
| rc_all (offenes Gate, margin 0) | 0.533 | 0.577 | ~0.36 | 0.293 | 724/796 |
| rc_anker_strict (margin **0.15**) | 0.645 | 0.647 | 0.360 | **0.330** | 2/519 |

Anker rot-median: raw 30.7°/22.5° → rc_anker (offen) **57.0°/41.6°** (schlechter!).

**Befund, ehrlich:** Der CPU-Kanten-Scorer **korrigiert den Anker-Flip NICHT** —
mit offenem Gate **verschlechtert** er die AR (−0.04 overall), weil er Flips
**einführt**: die Top-Down-Silhouette/Kante trennt den 180°-Anker-Flip auf realem
RGB nicht (deckt sich exakt mit M3: Self-IoU 0.58/0.64, und T-041). Mit striktem
Gate (0.15) wechseln nur 2/519 → AR **neutral** (0.330 ≈ raw), also kein Schaden,
aber **auch kein Gewinn**. Für den C_7-Zahnrad ist die Top-Down-Silhouette
rotations-invariant (T-041) → der CPU-Scorer kann (und muss metrisch) den Yaw
nicht auflösen.

**→ KEIN training-freier Teilgewinn aus M2-CPU.** Der Default-Gate ist deshalb
konservativ auf **0.15** gesetzt (`DEFAULT_CPU_MIN_MARGIN`): aktivierbar ohne
Schaden, aber der echte Hebel ist der **gelernte** MegaPose-Score.

---

## Wie M2 beim Finish validiert/aktiviert wird (GPU-Schritt)

GPU aktuell belegt (rendert/trainiert) → der MegaPose-Pfad ist **verdrahtet, aber
nicht jetzt ausgeführt**. Beim Finish (GPU frei):

1. **Self-Check (schnell):**
   `(removed — MegaPose-M2 path measured to ceiling, scoped out)`
   → prüft MegaPose-Import + Hypothesen-Generator-Wiring ohne schweren Refine.
2. **Voller Pfad:**
   `box_src/e2e_finish.sh --refine-rc --rc-scorer megapose` (oder direkt
   `real_pose_result.py --refine-rc --rc-scorer megapose`) → GDRNPP-Coarse →
   M2-Hypothesen → MegaPose-`forward_refiner` über alle Hypothesen → bester
   gelernter Score → verfeinerte Pose → pose_result (Contract unverändert).
3. **A/B gegen Baseline:** dieselbe val-Predictions-Menge mit/ohne `--refine-rc
   --rc-scorer megapose` über `eval_bop.py` scoren, AR-Diff dokumentieren.

Bei MegaPose-Unavailable fällt `refine_detection` automatisch auf den CPU-Scorer
zurück (dokumentiert in `info["megapose_fallback"]`).

---

## Erwarteter Impact (Research) + Einschränkung

- **MegaPose-RGB-Refiner:** +~23.7 ARC im Mittel (arXiv:2212.06870), RGB-only-
  Refiner schlägt CosyPose. **GenFlow:** coarse 23.5 → 67.4 ARC (+43.9).
- **ContourPose** (reflektierendes texturloses METALL): Kontur-Refiner schlägt
  Keypoint-Baselines — die Begründung, warum **Kanten** für unser Metall der
  richtige Refiner-Term sind. ABER: das gilt für den **gelernten** Kontur-Decoder,
  nicht für einen rohen Silhouetten-IoU/Chamfer (T-058-Befund oben).
- **Einschränkung (zentral):** Der Refiner fixt das falsche Becken / den Flip NUR
  über die **Multi-Hypothese** (mehrere Becken testen + gelernt scoren), **nicht
  lokal**. Ein lokaler render-vs-RGB-Schritt aus der geflippten Coarse bleibt im
  falschen Becken. Genau dafür liefert `generate_hypotheses` die Becken-Sprünge
  (Flips + C_N-Yaws + Ruhelagen), die der MegaPose-Score dann auswählt.

---

## Reproduzieren

```bash
# Box (bop-venv), CPU-RC-Validierung — kein GPU:
/mnt/data/bop/bop-venv/bin/python box_src/rc_refine_eval.py \
  --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
  --preds /mnt/data/bop/results/val_preds_combined.csv \
  --out-dir /tmp/rc_out --config raw rc_anker rc_all
for c in raw rc_anker rc_all; do
  /mnt/data/bop/bop-venv/bin/python box_src/eval_bop.py \
    --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac --split val \
    --preds /tmp/rc_out/preds_$c.csv --n-points 2000
done
# Finish (GPU frei) — MegaPose-RGB-Pfad:
# MegaPose-M2 path removed post-Phase-2
box_src/e2e_finish.sh --refine-rc --rc-scorer megapose
```

## Related
- `project/refine_rc.py` · `project/e2e_infer.py` · `project/bop_adapter.py`
  (`detection_to_result(rc_refiner=…)`) · `project/tests/test_refine_rc.py`
- `box_src/rc_refine_eval.py` (operative CPU-Scorer) ·
  `box_src/real_pose_result.py` · `box_src/e2e_finish.sh`
- `docs/PHASE3_PEAK_PLAN.md` (M2-Spec) · `docs/REFINE_T041.md` (M1/M3/M4-Befund) ·
  `docs/PROJECT_REPORT.md` · ADR-018 (BOP pivot) · ADR-017 (pose_result contract)
