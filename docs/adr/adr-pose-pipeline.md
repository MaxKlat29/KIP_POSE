# ADR — POSE planar 6D-Pose Pipeline & pose_result Contract

- **Status:** Accepted
- **Date:** 2026-05-22
- **Story:** S-006 / T-018 (P0 contract blocker)
- **Author:** Viktor (Tech Lead)
- **Decides for:** S-007 (Kai), S-008 / S-009 (Jonas), S-010 / S-011 (Lena)

## Context

Parts are dropped on a table and come to rest in a finite set of stable **faces**
(see `faces/README.md`). Per face the two out-of-plane rotations and the rest
height are fixed; only in-plane `(x, y)` and yaw stay free. We turn an SDG scene
image + 2D detections into a per-instance 6D pose, then render the CAD parts at
that pose in a 3D viewer.

The producer (inference, S-008/S-009) and the consumer (viewer, S-010/S-011) are
built by different people in parallel. To unblock both we **freeze the data
format between them** instead of letting it grow implicitly.

## Architecture

```
Face-Atlas         Registry              Classifier        Alignment            Viewer
(stable poses  ->  faces_<part>.json  -> infer(crop,part) -> CAD fit (yaw,t) -> Three.js
 per CAD part)     {faces:[name,R,..]}    -> face+conf        -> R_world,t_world   table+CAD@pose
```

1. **Face-Atlas** (`faces/`) discovers stable rest faces per CAD part.
2. **Registry** `faces_<part>.json` (written by `faces/cluster_views.py`) is the
   truth for the face set: `{part, n_views, convention, faces:[{name, prob,
   count, tilt_deg, R}]}`. `R` is the canonical body→world rest rotation
   `R_face`, column convention `world = R @ body`.
3. **Classifier** picks the resting face from an image crop (S-007).
4. **Alignment** fits in-plane yaw + translation against the CAD, composing the
   inferred yaw onto `R_face` → `R_world`, and sets `t_world` (S-009).
5. **Viewer** loads the table at the null-point and renders each CAD part at
   `(R_world, t_world)` (S-010/S-011).

## The Contract: `pose_result`

- Schema: [`docs/pose_result.schema.json`](../pose_result.schema.json) (JSON-Schema Draft 2020-12)
- Example: [`data/examples/pose_result.example.json`](../../data/examples/pose_result.example.json) — validates against the schema; serves as Lena's dummy input.
- Produced end-to-end by [`scripts/run_e2e.sh`](../../scripts/run_e2e.sh) (pipeline →
  `data/output/pose_result.json`, schema-gated before write).
- **One file per scene image.**

```
{
  meta:    { source_image, table_origin[3], units="m",
             coordinate_convention, schema_version },
  results: [ { instance_id, part, face, R_world[9], t_world[3],
               confidence, bbox_2d[4], upright } ]
}
```

**Frozen invariants** (the contract):

- **World frame:** Z-up. **Origin = table-plane null-point.** **Units = meters.**
- **Rotation:** `R_world` is 3×3 flattened to 9 floats, **column convention
  `world = R @ body`** — identical to `faces_<part>.json` `faces[].R`. The
  inferred yaw is composed onto the registry `R_face`. Producer and consumer use
  the same convention; no transpose at the boundary.
- **`face`** matches the registry name verbatim (`"Face 1"`, ...).
- **`part`** matches the CAD/registry name (`Anker_Lang`, `Zahnrad_Typ7`, ...) and
  drives both model routing and CAD load.
- **`bbox_2d`** = `[x0,y0,x1,y1]` in source-image pixels, integer, `x0<=x1`,
  `y0<=y1`.
- **`upright`** flags head-standing rest poses (Kopf hochkant), derived from the
  registry face tilt.
- **`schema_version`** is semver; compatibility is asserted on the major version.

### infer signature (S-007 Kai ↔ S-008 Jonas)

The face classifier exposes exactly:

```python
infer(crop: np.ndarray, part: str) -> {"face": str, "confidence": float}
```

- `crop`: the part's image crop (the producer cuts it from `bbox_2d`).
- `part`: the part name, so `infer` can route to the right model and clamp to
  that part's registry face set.
- returns `face` (a registry `name`, e.g. `"Face 1"`) and `confidence` (0..1,
  lands in `results[].confidence`).

This is part of the contract: Kai builds **to** this signature, Jonas builds
**against** it. `R_world` / `t_world` are produced by the downstream alignment
step (S-009), not by `infer`.

## Model routing

Routing is by `part` name, one classifier checkpoint per part family:

| part         | model         |
|--------------|---------------|
| `Anker_Lang` | model-lang    |
| `Anker_Kurz` | model-kurz    |
| other parts  | per-part checkpoint by name |

Routing lives behind `infer(...)`; callers pass `part` and never pick a model.

## Fallback strategy

When **no trained checkpoint exists** for a part, `infer` falls back to
**nearest-template** matching against the registry face templates
(`tmpl_Face*.png` written next to `faces_<part>.json`): pick the face whose
deskewed template is closest to the crop, return it with a deflated
`confidence`. This keeps the contract satisfied (always a valid `face` +
`confidence`) so the pipeline and viewer run end-to-end before every part is
trained.

## Consequences

- **+** D (inference) and E (viewer) build fully in parallel against a frozen
  JSON; the example is a working dummy input today.
- **+** Single rotation convention end-to-end, anchored to the existing registry
  — no silent transpose bugs at the boundary.
- **+** Fallback path means no checkpoint is a hard blocker for E2E wiring.
- **−** Changing any frozen invariant is a contract break → bump major
  `schema_version` and notify both sides.

## End-to-end wiring & going live

[`scripts/run_e2e.sh`](../../scripts/run_e2e.sh) runs the whole chain on the
bundled dummy scene with **no trained model**: pipeline → schema-gated
`data/output/pose_result.json` (12 instances) → viewer hint. Three steps take it
from the fallback to trained models, with no code change and the same
`run_e2e.sh` (full commands in the repo `README.md` § *So geht Max live* and
[`faces/classifier/README.md`](../../faces/classifier/README.md)):

1. **Simulate data** — `sim_code/render_dataset.py` per part → faceset;
   `faces/cluster_views.py` → `faces_<part>.json` + `tmpl_Face*.png` into
   `registry/<part>/`; `faces/extract_snippets.py` + `faces/build_manifest.py`
   → training data.
2. **Train** — `faces/classifier/train.py --part Anker_Lang` (→ `lang.pt`) and
   `--part Anker_Kurz` (→ `kurz.pt`), checkpoints at the canonical path
   `faces/classifier/checkpoints/<model>.pt`.
3. **Plug in** — drop the `.pt` at that path; `infer` auto-switches
   fallback → CNN on the next call. Nothing else to wire.

## Open points / known limitations

- **`registry/Anker_Kurz/` is missing.** Five parts have a committed registry
  (`Anker_Lang`, `Buerstenhalter_2polig`, `Getriebegehaeuse_typ4`,
  `Poltopf_kurz_centered`, `Zahnrad`); `Anker_Kurz` does not. In the E2E run its
  instances resolve via the fallback (identity `R_face`, default face) — contract
  stays valid, pose is not yet faithful. Closing it = step 1 above for
  `Anker_Kurz` (render faceset → cluster → copy into `registry/Anker_Kurz/`).
- **The dummy scene is a Zivid-camera capture, not the canonical top-down
  `render_dataset` view.** Back-projection (`pipeline/backproject.py`) uses the
  documented top-down pinhole intrinsics, so the resulting `(x, y)` are a
  *plausible* table layout for wiring/preview, **not metric ground truth**. The
  contract and the rotation convention are exact; metric accuracy needs the real
  camera pose/intrinsics, which plug into `backproject.Intrinsics`.
- **No trained checkpoints yet.** The E2E run is green on the fallback
  (`confidence = 0.0`); confidences become real once `lang.pt` / `kurz.pt` (and
  the multi-class parts) land at the canonical checkpoint path.

## Self-check

Validated `data/examples/pose_result.example.json` against
`docs/pose_result.schema.json` with `jsonschema` Draft202012Validator: schema
well-formed, example valid, all three `R_world` are proper rotations (det = 1),
all `bbox_2d` ordered. Reproduce:

```bash
uv run --with jsonschema python -c "import json;from jsonschema import Draft202012Validator as V; \
s=json.load(open('docs/pose_result.schema.json')); e=json.load(open('data/examples/pose_result.example.json')); \
V.check_schema(s); V(s).validate(e); print('VALID')"
```
