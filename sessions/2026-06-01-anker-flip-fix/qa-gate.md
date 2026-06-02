# QA-Gate — Ravi (QS)

## Wave Tune (2026-06-02)

### T-094: APPROVED — Detektor-Schwelle conf 0.3 -> 0.1
**Branch** `team/2026-06-02-pose-tune` · **Commit** `0d3aeeb` · Owner Kai

**Review (Korrektheit / Edge-Cases):**
- Diff ist WIRKLICH nur die conf-Schwelle: ein einziger Wert `conf=0.3 -> conf=0.1`
  in der inline-Code-String von `kip_server.py::_run_detector` (Z.301), plus
  ein erklaerender Docstring-Block. `imgsz=1280` und NMS-iou (ult-default 0.7,
  hier gar nicht gesetzt) unveraendert. **Kein Kollateral.** (9 LOC, +/- wie deklariert.)
- Callsite-Check: `_run_detector` wird an 3 Stellen gerufen (Z.354 real/infer,
  Z.431, Z.907) — alle gehen durch dieselbe Funktion, conf=0.1 gilt damit
  einheitlich fuer JEDEN Inferenz-Pfad. Konsistent.
- Downstream-Annahme-Check: die Detektor-`score` wird nachgelagert NUR als
  Anzeige-Label (`score*100%`, Z.987) und als Pass-Through-Metadatum an den
  GDRNPP-Worker weitergereicht. **Es gibt KEINEN nachgelagerten Score-Threshold,
  der eine `>=0.3`-Annahme machen wuerde.** Niedrigere conf = mehr Boxen an den
  Worker, sonst nichts. Bricht nichts am real-infer-Pfad.
- Plausibilitaet: Kais Sweep (val 4043 GT) rec_occ 0.971->0.979 (+0.8pp),
  prec 0.920->0.915 (-0.5pp, kein Tank), false_pos 345->369. Marginal aber sauber
  positiv; Detektor ist nicht der Cap (GDRNPP-AR ~0.87 ist es). Risikoarm.

**Tests:** Keine neuen Tests noetig — reine Konfig-Konstante, durch Kais
post-hoc Sweep gegen echtes val-Set bereits empirisch belegt; statische
Code-Verifikation deckt den Rest (keine gebrochenen Annahmen).

**Security:** -> Bruno. Diff beruehrt KEINE Auth/Input-Boundary/Deps/Secrets/Crypto
(nur eine numerische Inferenz-Schwelle). Kein Bruno-Review noetig.

**Gate: APPROVED** — nur-conf, risikoarm, kein gebrochener Pfad.
**Deploy:** Backend-Restart noetig (kip_server laedt die Konstante beim Start).

---

### T-095: APPROVED — groundClamp Mehrstufen-Fix
**Branch** `team/2026-06-02-groundclamp` · **Commit** `b07662e` · Owner Lena

**Review (Korrektheit / Edge-Cases — load-bearing, aendert Platzierung ALLER Teile):**
- Diff isoliert auf `scene.js::groundClamp()` (+85/-43, eine Datei). partsGroup
  ist Identity-Transform (Z.106f, kein pos/rot/scale) -> die lokale
  `holder.matrix.elements[14] += dz` == world-Z-Offset. Manipulation korrekt.
- **Center-Anker** (Z.241f): `_surfacesAt(cx,cy)` castet von z=5 runter, sortiert
  absteigend, nimmt hoechsten Treffer unterm Schwerpunkt = die Auflage-Ebene.
  Hat Vorrang -> ueber-Kante-haengendes Teil wird auf die Schwerpunkt-Flaeche
  gesetzt, NICHT von einer benachbarten hoeheren Empore-Kante hochgezogen. Korrekt.
- **Median-Fallback** (Z.249-260): greift NUR wenn Center ins Leere traf
  (`tableZ===null`, z.B. Ringmagnet ueber Bohrung). 3x3-Grid, MEDIAN statt MAX.
- **Beide Richtungen** (Z.264-271): dz kann +/- -> Schwebendes runter, Versunkenes hoch.
- **Sanity-Guard** (Z.267): `|dz|>0.6m -> return` faengt Teleport quer durch die ~1.4m Zelle.
- **Idempotenz-Guard** (Z.268): `|dz|<1e-5 -> return` -> gut-platziertes Teil
  kriegt KEINEN neuen Jitter/Versatz.
- **Kein-Treffer-Fallback** (Z.260): Footprint ohne jede Flaeche -> `return`,
  Teil bleibt liegen statt auf 0 gezwungen.
- Multi-Part: per-holder-Iteration, kein Cross-Contamination (verifiziert).

**Tests (selbst nachgelaufen — echter Pfad, echtes cell.glb):**
Ravi-Gate-Harness (Playwright/chromium, importiert echtes `createViewer`/`setParts`
welches intern `groundClamp` ruft, raycastet gegen echtes `assets/cell.glb`).
Reale Surface-Z-Range -0.006..1.2157 (Zelle ist nachweislich mehrstufig).
- `8/8 PASS`:
  - **REGRESSION: gut-platziertes Teil bleibt seated** — drift 0.0000m (KERN-GATE).
  - **Idempotenz** — zweiter setParts: dz=0, kein Drift.
  - **SCHWEBE 25cm** -> sitzt auf Flaeche (1.1098).
  - **VERSUNKEN 10cm** -> angehoben auf Flaeche.
  - **SANITY-GUARD >0.6m** -> NICHT gesnappt, bleibt oben (kein Teleport).
  - **KEIN-TREFFER** (Footprint ausserhalb Zelle) -> Teil bleibt (minZ~2.98, NICHT 0).
  - cell.glb laedt, keine Browser-Errors.
- **MULTI-PART PASS**: 2 Teile gleichzeitig auf Boden (0.02) + Plateau (1.111),
  beide korrekt seated, kein gegenseitiges Hochziehen.
(Harness lief extern via /tmp playwright-deps; NICHT in den Worktree committed,
um den static-FE-Merge sauber zu halten. Lenas eigene 8/8 reproduziert + um
Regressions-/Idempotenz-/Multi-Part-Faelle erweitert.)

**Security:** -> Bruno. Reiner clientseitiger Viewer-Geometrie-Code, keine
Auth/Input/Deps/Secrets/Crypto. Kein Bruno-Review noetig.

**Gate: APPROVED** — keine Regression auf gut-platzierte Teile (drift 0),
alle Guards (Sanity 0.6m / Kein-Treffer / Idempotenz) greifen wie spezifiziert.
**Deploy:** Static-Frontend (scene.js) — kein Restart, nur Asset-Auslieferung.

---

**Beide deploy-safe.** Sam kann mergen: T-094 (Backend, Restart) + T-095 (static FE).
