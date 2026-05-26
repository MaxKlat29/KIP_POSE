# Multi-Stage Viewer

Der 3D-Viewer (`project/frontend/`) zeigt die **Refinement-Pipeline** stufenweise.
Pro Stufe lädt er ein eigenes `temp/pose_result_<stage>.json`; eine Stage-Leiste
oben im Viewer schaltet zwischen ihnen um.

## Stufen (Pipeline-Reihenfolge)

| Stage   | Datei                              | Was passiert                          |
|---------|------------------------------------|---------------------------------------|
| Raw     | `pose_result_raw.json`             | GDRNPP-Output, kein Refinement        |
| +Z-Snap | `pose_result_zsnap.json`           | Planar-Z-Snap (Translation→Tisch)     |
| +M1     | `pose_result_m1_zsnap.json`        | + stable-pose Rotation-Snap           |
| +M2     | `pose_result_m2.json`              | + Render-and-Compare (cpu_edge)       |
| +TTA    | `pose_result_tta.json`             | + Rotation-Test-Time-Aug              |
| Final   | `pose_result.json` (kanonisch)     | Auto-Best (A/B-Gewinner)              |

## Wie die Stufen-Files entstehen

`box_src/e2e_finish.sh` läuft die A/B über alle Refinement-Lever und schreibt
Variant-CSVs nach `$REMOTE_AB/preds_*.csv`. `box_src/make_stages.py` ist ein
dünner Wrapper, der pro Variant-CSV einmal `box_src/real_pose_result.py` für
die gewählte Show-Case-Szene aufruft (`PRIMARY_SCENE` / `PRIMARY_IM` aus
e2e_finish) und ein eigenes pose_result.json pro Stufe schreibt.

```bash
# Auf der Box nach e2e_finish:
python box_src/make_stages.py \
  --ab-dir /mnt/data/bop/results/ab \
  --final-csv /mnt/data/bop/results/preds_all.csv \
  --scene 0 --im 92 \
  --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
  --temp-dir /mnt/data/kip_pose/project/temp
```

Lokal werden die Files via `scripts/pull_box_artifacts.sh` (oder rsync direkt)
nach `project/temp/` gezogen.

## UI

- **Stage-Leiste** (oben, mittig) — `Raw | +Z-Snap | +M1 | +M2 | +TTA | Final`
- Click auf Button → URL-Param `?file=...` ändert sich → Seite lädt neu mit der
  Stufe.
- Aktiver Button wird farblich hervorgehoben (`.stage-btn--active`).
- Fehlt eine Stufen-Datei (z.B. weil A/B-Variant nicht produziert), gibt der
  Viewer den leeren Default-Scene zurück (kein Fehler) — die anderen Stufen
  bleiben anklickbar.

## Source-of-Truth

- HTML: `frontend/index.html` (Stage-Leiste als `<nav id="stage-selector">`)
- CSS:  `frontend/src/style.css` (`.stages`, `.stage-btn`)
- JS:   `frontend/src/main.js` — `markActiveStage()` highlightet den aktuellen
        Button, der Rest läuft über Browser-Default-Navigation (`<a href="?file=...">`).
- Backend: `box_src/make_stages.py` — produziert die Pro-Stufe-JSONs.
