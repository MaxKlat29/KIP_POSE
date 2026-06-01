# `pipelines/` — Multi-Pipeline-Vergleich (pluggable)

Hülle, um **komplett andere, eigenständige** 6D-Pose-Pipelines gegeneinander zu
vergleichen. Eine Pipeline = ein vollständiges System *Bild → 6D-Posen*. Die
gelieferte Hauptlinie (`gdrnpp`) ist nur die **Referenz/Baseline**.

> **Status (2026-06-01): SCAFFOLD.** Nur die Hülle. Fremde Pipelines sind leer
> (`NotImplementedError`) — sie kommen als rohe Python-Projekte von Max und werden
> hier sauber angebunden. **Noch nichts „reingesetzt".**

## Aufbau

| Datei | Rolle |
|---|---|
| `base.py` | `PipelineAdapter` (ABC) + `PipelineResult` (Telemetrie). Die Schnittstelle. |
| `contract.py` | Gate gegen das **eingefrorene** `pose_result.schema.json` (reuse von `e2e_infer`). `assemble_doc()` leitet `face`/`upright` analytisch ab. |
| `registry.py` | `id → Adapter`. Referenz registriert sich automatisch. `all_pipelines()` für `/api/pipelines` + Dropdown. |
| `gdrnpp_adapter.py` | Referenz-Adapter (wrappt `e2e_infer.run`, ändert den Live-Viewer **nicht**). |
| `_template/` | Kopiervorlage für eine neue Pipeline (`adapter.py` stub + `vendor/` + isolierte `requirements.txt`). |
| `../compare_pipelines.py` | Vergleichs-Harness über 4 Achsen. `--dry-run` läuft offline. |

## Contract (Max-Entscheidung: exaktes bestehendes Schema)

Jeder Adapter liefert ein `pose_result`-Doc, das **`project/pose_result.schema.json`**
erfüllt (`instance_id, part, face, R_world[9], t_world[3], confidence, bbox_2d[4],
upright`, `additionalProperties:false`). Prüfung: `pipelines.contract.validate(doc)`.

Kennt die fremde Pipeline kein `face`-Konzept → `contract.assemble_doc(...)` leitet
`face`/`upright` analytisch aus `R_world` ab (`bop_adapter.face_and_upright_from_R`).

⚠️ Der **Live-Web-Viewer** konsumiert eine Runtime-Variante mit `color` (gt/pred) und
ohne `bbox_2d` — das ist Overlay, **nicht** der Contract. Adapter halten sich ans
frozen Schema; die gt/pred-Einfärbung macht der Harness/Viewer separat.

## Neue Pipeline anbinden

Siehe [`_template/README.md`](_template/README.md) und die ausführliche
[`../docs/PIPELINE_INTEGRATION.md`](../docs/PIPELINE_INTEGRATION.md).

## Offline-Trockenlauf

```bash
python3 project/compare_pipelines.py --dry-run --out /tmp/cmp
cat /tmp/cmp/COMPARE.md
```
