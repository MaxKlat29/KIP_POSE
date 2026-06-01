# Pipeline-Integration & -Vergleich

Wie eine **komplett andere, eigenständige** 6D-Pose-Pipeline in KIP_POSE angebunden
und gegen die gelieferte Hauptlinie (`gdrnpp`) verglichen wird.

> **Status (2026-06-01): SCAFFOLD.** Die Hülle (`project/pipelines/` +
> `project/compare_pipelines.py` + API-Hooks) steht und ist getestet. Die fremden
> Pipelines selbst sind **leer** — sie kommen als rohe Python-Projekte von Max und
> werden über je einen Adapter angebunden. **Noch nichts „reingesetzt".**

---

## 1 · Modell

Eine **Pipeline** ist ein vollständiges System *Szenen-Bild → 6D-Posen*. Wir fixieren
nur den **Contract an der Grenze**, nicht die Interna — eine fremde Pipeline darf einen
völlig anderen Detektor + Pose-Schätzer (oder gar keine Trennung) verwenden, in eigener
venv / eigenem Env.

```
                       ┌───────────────── PipelineAdapter (Seam) ─────────────────┐
 image_path  ─────────►│  infer(image_path, camera, table_origin) -> pose_result  │──► pose_result.json
 camera (K, w2c)       │    gdrnpp  → e2e_infer.run (Referenz/Baseline)            │    (frozen schema)
 table_origin (m)      │    pipeline_x → vendor/<rohes Projekt> + adapter-Mapping  │
                       └──────────────────────────────────────────────────────────┘
```

Registry (`pipelines/registry.py`) hält `id → Adapter`. Der Vergleichs-Harness
(`compare_pipelines.py`) und die API (`/api/pipelines`) zählen sie auf.

---

## 2 · Contract (Pflicht: exaktes bestehendes Schema)

**Max-Entscheidung 2026-06-01:** jede Pipeline liefert ein `pose_result`-Doc, das
**`project/pose_result.schema.json`** erfüllt — `additionalProperties:false`, alle
Felder Pflicht:

```
meta:     source_image, table_origin[3], units="m", coordinate_convention, schema_version
results[]: instance_id, part, face, R_world[9], t_world[3], confidence, bbox_2d[4], upright
```

Prüfung (stdlib immer + jsonschema wenn vorhanden):

```python
from pipelines import contract
errs = contract.validate(doc)        # [] = gültig
contract.assert_valid(doc)           # wirft sonst ValueError
```

Kennt die fremde Pipeline kein `face`/`upright`-Konzept → **`assemble_doc`** leitet
beide analytisch aus `R_world` ab (`bop_adapter.face_and_upright_from_R`):

```python
entries = [{
    "instance_id": 0,
    "part": "Anker_Lang",           # muss CAD/Registry-Namen treffen (bop_adapter.PART_TO_OBJ_ID)
    "R_world": [...9...],            # world = R @ body, Z-up
    "t_world": [x, y, z],           # Meter, rel. Tisch-Nullpunkt
    "confidence": 0.87,             # 0..1
    "bbox_2d": [x0, y0, x1, y1],    # Pixel, ints
}]
doc = contract.assemble_doc(image_path, entries, table_origin)   # face/upright + Validierung inklusive
```

> ⚠️ **Runtime-Abweichung:** der Live-Web-Viewer (`kip_server._bop_pose_to_result`)
> rendert eine Variante mit `color` (gt/pred) und **ohne** `bbox_2d`. Das ist die
> Overlay-Form, **nicht** der Contract. Adapter halten sich ans frozen Schema; die
> gt/pred-Einfärbung beim Vergleich macht der Harness/Viewer.

---

## 3 · Eine Pipeline anbinden (Schritt für Schritt)

1. **Kopieren:** `pipelines/_template/` → `pipelines/<id>/` (z. B. `pipelines/pipeline_x/`).
2. **Vendor:** das rohe externe Python-Projekt (von Max, unverändert) → `pipelines/<id>/vendor/`.
3. **Deps isolieren** (fremde Projekte kollidieren oft mit unseren/Box-Deps):
   ```bash
   python3 -m venv pipelines/<id>/.venv
   pipelines/<id>/.venv/bin/pip install -r pipelines/<id>/requirements.txt
   ```
4. **Adapter** (`pipelines/<id>/adapter.py`): `id/name/description` setzen,
   `available` auf echten Check (Vendor + Deps da), `infer()` implementieren:
   - bevorzugt **Subprocess** in die `<id>/.venv` (robust bei Dep-Konflikten),
   - rohen Output → `contract.assemble_doc(...)`.
5. **Registrieren:** `register(<Adapter>())` am Ende von `adapter.py` entkommentieren.

Danach erscheint die Pipeline automatisch in `/api/pipelines`, im Modell-Dropdown
und im Vergleich.

---

## 4 · Die vier Vergleichs-Achsen (`compare_pipelines.py`)

| # | Achse | Quelle | Braucht |
|---|---|---|---|
| 1 | **Genauigkeit vs GT** | `box_src/eval_bop.py` (BOP AR/ADD/MSSD/MSPD, sym-aware) | GT-Dataset + GPU-Box (`bop-venv`, `bop_toolkit`) |
| 2 | **Visueller Side-by-Side** | `<out>/sidebyside/<scene>.json` (Multi-Pipeline-Posen) → Viewer-Overlay | nur die `pose_result`-Docs |
| 3 | **Latenz/Laufzeit** | `PipelineAdapter.run_timed` (Wall-Clock pro Bild) | — |
| 4 | **Robustheit/Coverage** | `aggregate_telemetry` (n_results, Crash-Rate, leere Outputs) | — |

**Eval-Rückrechnung (Achse 1):** `pose_result` liegt im Welt-Frame, `eval_bop`
arbeitet im BOP-cam-Frame. `compare_pipelines.world_to_bop_cam(...)` ist die exakte
Inverse von `bop_adapter.bop_pose_to_world` (round-trip-getestet) und baut die
BOP-results-CSV, die `eval_bop` scort.

Trockenlauf (offline, ohne Bild/Box):
```bash
python3 project/compare_pipelines.py --dry-run --out /tmp/cmp && cat /tmp/cmp/COMPARE.md
```

Echter Lauf (auf der Box, mit GT):
```bash
python3 project/compare_pipelines.py \
    --pipelines gdrnpp,pipeline_x \
    --images <img1>,<img2>,... \
    --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac --split val \
    --out project/temp/compare_run
```

Nicht-verfügbare Pipelines werden mit **explizitem Log** übersprungen (kein stilles
Weglassen).

---

## 5 · API-Hooks (Backend / Frontend)

- `GET /api/pipelines` → `[{id, name, description, available}]` (für Dropdown + Harness).
- Infer-Endpoints akzeptieren optional `pipeline=<id>` (Default `gdrnpp` = unveränderter
  Live-Pfad; nicht-angebundene ids → `501`).
- `GET /api/compare?...` → Stub, bis ≥2 Pipelines verfügbar sind.
- Frontend: Modell-Dropdown wird aus `/api/pipelines` befüllt (Fallback = statisches
  `gdrnpp`); `frontend/src/compare.js` = dokumentierter Side-by-Side-Hook (noch nicht
  in den Default-Render-Pfad verdrahtet).

---

## 6 · Referenz (`gdrnpp`)

Die Baseline. `pipelines/gdrnpp_adapter.py` wrappt `e2e_infer.run` (mock-fähig →
lokal ausführbar) und **ändert den Live-Viewer-Pfad nicht**. Sie beweist, dass der
Seam end-to-end trägt, und ist die Vergleichs-Baseline für jede fremde Pipeline.
