# T-041 — Training-freie Rotations-Refinements (M1 / M3 / M4) — gemessen, ehrlich

**Date:** 2026-05-24 · **Worktree:** `.worktrees/S-047` · **Branch:**
`team/2026-05-22-pose-bop-pivot/S-047` · **No retrain, no GPU-inference** —
gemessen an den BESTEHENDEN GDRNPP-val-Predictions (`*-iter0_pose_isaac-val.csv`),
symmetrie-bewusster, >20%-visib-gefilterter `eval_bop.py` (val GT = 1077 Instanzen).

Alle drei Maßnahmen leben in `project/bop_adapter.py` (hinter Flags, generalisierbar,
rein numpy + trimesh-CPU). Harness: `box_src/refine_eval.py` (cam→world→refine→cam,
exakt invertierbar; `raw`-Config reproduziert die Eingabe-CSV bit-nah).

---

## TL;DR — was im Default landet

| Maßnahme | AR-Wirkung (gemessen) | Default | Warum |
|---|---|---|---|
| **Z-Snap** (bestand schon) | **Anker +0.04..0.06**, Zahnrad +0.004 | **AN** | einziger echter training-freier Gewinn |
| **M1** Stable-Pose-Snap | Anker **−0.002..−0.003**, Zahnrad ±0 | **AUS** | netto neutral/leicht negativ an diesem Set |
| **M4** Contour-Yaw-Lock | Zahnrad **+0.000..+0.003** (= Rauschen) | **AUS** | 0 AR-Wirkung (Metrik ist C_7-bewusst) |
| **M3** Anker-C_2 | n/a | **NICHT angewandt** | Flip ist KEINE Top-Down-Symmetrie |

**Ehrliche Gesamtaussage:** Die training-freien Rotations-Fixes bringen das Ziel
(Zahnrad → 0.7+, Anker → 0.8+) **NICHT**. Der einzige training-freie Hebel bleibt
der bestehende Z-Snap. Anker stehen nach Z-Snap bei **0.645 / 0.650**, Zahnrad bei
**0.360** — unverändert. Der Rest braucht das, was der PHASE3-Plan als M2 (Render-/
Contour-Refiner) und M5 (SC6D-Rotations-Klassifikation, Retrain) listet.

---

## Die Zahlen (val, gefiltert, sym-aware, an denselben Predictions)

AR / rot-median(sym,°) / rot_naive_mean(°) pro Objekt, **vorher → nachher**:

| Config | AnkerKurz AR | AnkerLang AR | Zahnrad AR | gear rot-med | gear rot_naive |
|---|---|---|---|---|---|
| raw (kein Snap) | 0.589 | 0.606 | 0.356 | 81.5 | 123.96 |
| **zsnap = BASELINE** | **0.645** | **0.650** | **0.360** | 81.5 | 123.96 |
| m1 (ohne zsnap) | 0.587 | 0.602 | 0.355 | 85.9 | 126.82 |
| m1 + zsnap | 0.643 | 0.647 | 0.359 | 85.9 | 126.82 |
| m4 (ohne zsnap) | 0.589 | 0.606 | 0.357 | 81.5 | 123.15 |
| **m1 + m4 + zsnap** | 0.643 | 0.647 | 0.362 | 85.9 | 123.41 |

Anker rot-med (sym) bleibt 4.9–6.4°; rot_naive ~96–101° (der Flip-Tail). Deltas
gegen die zsnap-Baseline: **M1 −0.002/−0.003 (Anker), +0.000 (gear); M4 +0.002
(gear); kombiniert −0.003/−0.003/+0.002.** Alles ≤ Rausch-/negativ.

---

## M1 — Stable-Pose-Rotations-Snap: korrekt gebaut, hilft hier aber nicht

`trimesh.compute_stable_poses` → K Ruhe-Body-Down-Achsen; snappe die **Tilt**-
Komponente der Vorhersage auf die nächste, behalte den In-Plane-Yaw; Guard
`max_tilt_snap_deg=55°` (schützt stehende/gehaltene Teile, wie der Z-Snap-Guard).
Unit-getestet: snappt exakt, idempotent, Yaw-erhaltend, Guard greift.

**Warum kein Gewinn — zwei gemessene Gründe:**

1. **Anker:** der ≥90°-Flip-Tail (11 % der Instanzen, sym-resolved) ist ein ECHTER
   End-zu-End-Flip des Stabs, KEINE kleine Tilt-Drift. M1 findet zwar eine gültige
   Ruhelage, aber bei einem flippten Stab die *falsche* (der Anker hat 12–22 fast
   gleichwertige Seiten-Ruhelagen). cont-Y vergibt den In-Plane-Teil ohnehin; der
   Rest ist ein 3D-Flip, den ein Tilt-Snap nicht konsistent rückgängig macht. Netto
   verkippt M1 gelegentlich eine *korrekte* Pose minimal auf eine Nachbar-Ruhelage →
   −0.002.
2. **Zahnrad:** die GT-Zahnräder im val-Set ruhen **NICHT flach** — Median-Tilt zur
   nächsten *flachen* Ruhelage = **41°**; nur 34 % liegen flach (das dicke Zahnrad,
   34 mm hoch / 50 mm Ø, ruht oft auf der Zahn-Kante). Gegen ALLE 260 Ruhelagen ist
   der GT-Tilt zwar 1°, aber GDRNPP's Zahnrad-Rotation ist im *falschen Becken*
   (naive ~138°) — ein Tilt-Snap auf irgendeine Ruhelage bringt das nicht in die
   richtige Orientierung (sym-rot 94.9° → 91.9°, marginal).

**Verbleibt im Code, hinter `stable_pose_snap`-Flag**, generalisierbar (jedes Teil +
CAD). Default AUS.

## M3 — Anker Top-Down-C_2-Check: ENTSCHEIDUNG = C_2 NICHT valide

`topdown_c2_flip_identical` rendert die Top-Down-Silhouette der liegenden Pose,
dreht sie 180° um das Silhouette-Zentrum, misst die Self-IoU. **Validiert:** ein
symmetrischer Zylinder → IoU 1.000 (flip-identisch=True); ein L-Teil → <0.90.

**Befund am echten CAD:** AnkerKurz IoU **0.64**, AnkerLang IoU **0.58** — **deutlich
unter jeder Identitäts-Schwelle**. Grund (aus dem Mesh-Profil): der Anker ist
 entlang Y stark asymmetrisch — fetter Körper (r≈12 mm) an einem Ende, langer dünner
Schaft (r≈4.6 mm) am anderen. Von oben sind die beiden Enden klar unterscheidbar.

**Entscheidung:** Der 180°-Flip ist **KEINE view-abhängige Symmetrie**. Es wird
**KEIN diskreter C_2 in `models_info.json` gesetzt** — das würde den echten 3D-Flip-
Fehler fälschlich als korrekt gutschreiben (false credit, BOP-Distrib-Warnung). Der
Anker bleibt continuous-Y; der Flip-Tail muss über M2 (Refiner) / M5 (Multi-Hyp,
Retrain) gelöst werden, nicht analytisch.

## M4 — Zahnrad Contour-Yaw-Lock: 0 AR-Wirkung (zweifach begründet)

`contour_yaw_lock` rendert (CPU, Oberflächen-Sampling) die Silhouette an den N=7
Zahn-Yaws und wählt per IoU-zur-Maske. Mechanismus unit-getestet an einem Zahnrad
**mit markiertem Zahn** (silhouette-brechend) → wählt korrekt 1 von 7, idempotent.

**Warum am ECHTEN Zahnrad keine Wirkung — zwei harte Gründe:**

1. **Die AR-Metrik ist bereits C_7-bewusst.** MSSD/MSPD lösen die Symmetrie über
   `models_info` auf → die 7 Zahn-Yaws sind metrisch ÄQUIVALENT. Egal welchen M4
   wählt, der sym-resolved Fehler ändert sich NICHT (gemessen: m1 91.9° == m1+m4
   91.9°, identisch).
2. **Die Top-Down-Silhouette eines C_7-Zahnrads ist rotations-invariant.** Eine
   Drehung um 360/7 erzeugt eine quasi-identische Silhouette (IoUs der 7 Kandidaten
   am echten Zahnrad: alle 0.39–0.40, nicht trennbar). Genau die Mehrdeutigkeit, die
   M4 auflösen soll, lässt die Silhouette unverändert. (Deckt sich exakt mit der im
   PHASE3-Plan zitierten Template-Hashing-Studie: Pixel allein brechen Zahn-
   Mehrdeutigkeit nicht.)

**Verbleibt im Code, hinter `contour_yaw_lock`-Flag**, generalisierbar für diskret-
symmetrische Teile **mit silhouette-brechendem Merkmal** (oder für deterministische
Viewer-Yaw-Wahl). Default AUS. Für die AR-Metrik wertlos beim echten C_7-Zahnrad.

---

## Wo wir nach den training-freien Fixes stehen (ehrlich)

- **Anker_Kurz 0.645 / Anker_Lang 0.650** (Z-Snap-Baseline) — Ziel 0.8+ **nicht
  erreicht**. Median-Rotation ist bereits gut (5–6° sym); der ≥90°-Flip-Tail (11 %)
  ist training-frei nicht knackbar (kein echter C_2, M1 fixt ihn nicht).
- **Zahnrad 0.360** (unverändert) — Ziel 0.7+ **weit verfehlt**. Die C_7-Rotation
  ist im Netz ungelernt; weder Tilt-Snap (Teil ruht oft nicht flach) noch Silhouetten-
  Yaw-Lock (Metrik+Silhouette beide C_7-invariant) helfen.
- **Nächste, NICHT-training-freie Hebel** (PHASE3-Plan): M2 Render-/Contour-Refiner
  auf GDRNPP-coarse (GPU-Inferenz), M5 SC6D-Rotations-Klassifikations-Head fürs
  Zahnrad (Retrain). Der Z-Snap bleibt der einzige geschenkte Hebel.

## Reproduzieren

```bash
# lokal/Box: refinte CSVs erzeugen (CPU)
python box_src/refine_eval.py --bop-root <bop> \
  --preds anker_kurz=<csv> anker_lang=<csv> zahnrad=<csv> \
  --out-dir /tmp/refine_out --config raw zsnap m1 m1_zsnap m4 m1_m4_zsnap
# Box (bop-venv): jede Config scoren (GT bleibt im gefilterten Zustand = 1077)
for c in raw zsnap m1 m1_zsnap m4 m1_m4_zsnap; do
  python box_src/eval_bop.py --dataset-dir <bop> --split val \
    --preds /tmp/refine_out/preds_$c.csv --n-points 2000
done
```

## Related
- `bop_adapter.py` (M1 `stable_pose_snap`, M3 `topdown_c2_flip_identical`,
  M4 `contour_yaw_lock`; alle hinter Flags) · `box_src/refine_eval.py`
- `docs/PHASE3_PEAK_PLAN.md` (M1/M3/M4-Specs) · `docs/REEVAL_T038_visib20.md`
- ADR-018 (BOP pivot) · ADR-017 (pose_result contract)
