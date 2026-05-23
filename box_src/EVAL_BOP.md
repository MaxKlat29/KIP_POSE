# BOP Eval Harness (`eval_bop.py`)

> Ticket **T-070 / Story S-501**. Measures 6D-pose predictions against the synthetic
> BOP ground truth with the **official `bop_toolkit_lib` metrics**, symmetry-aware.
> Built + validated *before* GDRNPP training finishes, so the moment Kai's checkpoint
> lands we can score it. RGB-only, synthetic holdout (`val` split). **No training touched.**

## What it computes

Per object (`obj_id` 1..6) and as an over-objects mean:

| Metric | Meaning |
|---|---|
| **AR** | BOP19+ headline score = `mean(AR_VSD, AR_MSSD, AR_MSPD)`. Without `--vsd` it's `mean(AR_MSSD, AR_MSPD)`. |
| **AR_VSD** | Visible Surface Discrepancy recall (delta=15mm, taus 0.05..0.50, normalised by diameter). Needs the renderer + depth. |
| **AR_MSSD** | Max Symmetry-aware Surface Distance recall (thresholds 0.05..0.50 x diameter). |
| **AR_MSPD** | Max Symmetry-aware Projection Distance recall (thresholds 5..50 px, scaled by `640/img_width`). |
| **ADD / ADI** | Avg distance of model points. **ADI** (nearest-neighbour) is used for symmetric parts, **ADD** otherwise. |
| **trans_mm** | Plain translation error in mm (mean + median). Human-friendly. |
| **rot_deg** | **Symmetry-resolved** rotation error in degrees -- the analytic fix for the eigenbau 120/91-degree punishment. |
| **rot_naive** | Raw geodesic rotation error (no symmetry). Shown alongside `rot_deg` so the symmetry effect is visible. |

Symmetry comes straight from `models_eval/models_info.json` via
`bop_toolkit_lib.misc.get_symmetry_transformations` (Viktor ADR section 2):
obj 1/2 (Anker) + obj 5 (Ringmagnet) continuous about Y, obj 6 (Zahnrad) discrete C_7,
obj 3/4 none.

## How to run

### From the laptop (recommended) -- via the wrapper

```bash
# Self-validation (no predictions, no GDRNPP needed):
box_src/eval_bop.sh --self-test

# Score real predictions (a BOP-results CSV that lives ON the box):
box_src/eval_bop.sh --preds /mnt/data/bop/results/gdrnpp/preds.csv

# Add VSD (slower; uses the vispy/EGL renderer + depth maps):
box_src/eval_bop.sh --self-test --vsd
```

The wrapper ships `eval_bop.py` to the box, runs it in the `bop-venv`, and pulls the
report (`report.json` + `report.txt`) back to `./results/eval/`.

### Directly on the box -- via `gpu_run.sh`

```bash
.worktrees/S-204/box_src/gpu_run.sh -- '
/mnt/data/bop/bop-venv/bin/python /mnt/data/bop/eval_bop.py \
  --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
  --split val --self-test \
  --out /mnt/data/bop/results/selftest'
```

### Flags

| Flag | Meaning |
|---|---|
| `--dataset-dir` | BOP dataset root (`pose_isaac`). Reads `models_eval/`, `<split>/`. |
| `--split` | `val` (default), `train_pbr`, or `test`. |
| `--preds PATH` | BOP-results CSV with predictions (mutually inclusive-or with `--self-test`). |
| `--self-test` | Synthesise predictions from GT (3 scenarios, see below). No checkpoint needed. |
| `--vsd` | Also compute VSD (vispy/EGL renderer + depth). Off by default (heavy). |
| `--out DIR` | Write `report.json` + `report.txt`. |

## Predictions input format -- BOP-results CSV

One row per predicted instance, exactly the BOP19 standard
(`scene_id,im_id,obj_id,score,R,t,time`), which is what
`bop_toolkit_lib.inout.save_bop_results` writes and what GDRNPP / GigaPose / MegaPose emit:

```
scene_id,im_id,obj_id,score,R,t,time
0,0,2,0.97,0.075 -0.364 -0.928 0.190 -0.908 0.372 -0.979 -0.205 0.001,-443.2 313.7 893.4,0.21
0,0,4,0.95,0.241 0.119 0.963 -0.970 0.029 0.239 0.0 -0.992 0.123,-473.2 118.1 1152.7,0.21
```

| Column | Type | Notes |
|---|---|---|
| `scene_id` | int | matches the scene folder (`0` -> `val/000000/`). |
| `im_id` | int | image index within the scene. |
| `obj_id` | int | **1..6**, the BOP obj_id. Detector class is 0-based -> `obj_id = class + 1` (ADR section 1.2). |
| `score` | float | confidence; `1.0` if unknown. Used to order greedy GT matching. |
| `R` | 9 floats | space-separated, **row-major**, model->camera rotation. |
| `t` | 3 floats | space-separated, model->camera translation in **millimetres**. |
| `time` | float | seconds per image; `-1` if unknown. |

A header line is optional. The same `R_m2c` / `t_m2c` (camera frame, mm) convention as
`scene_gt.json` -- *before* the cam->world->contract mapping of ADR section 3. **Eval is done
in the camera frame** (BOP convention); the world mapping is only for the viewer.

## Getting GDRNPP output into the BOP-results CSV

GDRNPP's BOP-2022 eval pipeline already writes a BOP-results CSV directly. Two paths:

1. **GDRNPP native dump (preferred).** GDRNPP's `core/gdrn_modeling/test_gdrn.py` saves
   results via `bop_toolkit`'s `save_bop_results` to
   `output/.../inference_.../<dataset>-test.csv` (filename pattern
   `<method>_<dataset>-<split>.csv`). Point `--preds` at that file. obj_ids must be 1..6.

2. **Manual bridge** (if you have raw `R_m2c`/`t_m2c` per detection). Use the toolkit:

   ```python
   from bop_toolkit_lib import inout
   results = [{
       "scene_id": int(scene_id), "im_id": int(im_id), "obj_id": int(obj_id),
       "score": float(score),
       "R": R_m2c_3x3_numpy,          # model->cam, row-major
       "t": t_m2c_mm_3x1_numpy,       # mm
       "run_time": float(time_s),
   } for ... in detections]
   inout.save_bop_results("preds.csv", results, version="bop19")
   ```

   Watch the unit: GDRNPP works in **mm** internally for BOP datasets -- keep `t` in mm.
   If a stage emits metres, multiply by 1000 before writing.

> **Detection bridge (ADR section 4):** GDRNPP needs detections (boxes) as input. Use the
> arm-visible retrained detector (S-401) exported to AABB. The eval here only consumes the
> final pose CSV -- it does not run detection.

## Self-validation -- what `--self-test` proves

No GDRNPP needed: predictions are synthesised from the GT itself, three scenarios:

1. **`[a] GT-as-prediction`** -- sanity. Expect `AR ~= 1.0`, `ADD/ADI ~= 0`, `rot ~= 0`.
   Proves the loading, matching, and metric plumbing are correct.
2. **`[b] GT + 10deg random-axis + 5mm`** -- errors must rise for *all* parts and AR drop
   below `[a]`. Proves the metrics react to error.
3. **`[c] GT + 10deg symmetry-axis + 5mm`** -- the rotation is applied *about the model Y
   axis* (the symmetry axis). For symmetric parts the symmetry-resolved `rot_deg` must
   **collapse to ~0** while `rot_naive` stays ~10deg; asymmetric parts stay punished
   (`rot_deg == rot_naive`). **This is the proof the symmetry handling works** -- the same
   mechanism that stops a correct anker/zahnrad pose from being scored as a 91/120-degree miss.

The output is a per-object table plus explicit PASS/FAIL verdicts.

## Notes / caveats

- **Greedy matching** by translation distance per `(scene, im, obj_id)`, estimates ordered by
  score. Unmatched GT counts as a miss (recall denominator = number of GT instances). For the
  synthetic holdout (no clutter, one instance dominant) this matches BOP localisation semantics.
- **MSPD scaling**: BOP fixes the px thresholds (5..50) for a 640px-wide reference and scales
  the *error* by `640/img_width`. Our renders are 1280 wide -> factor 0.5, handled internally.
- **Continuous symmetry** is discretised at `max_sym_disc_step=0.01` (BOP19 default) ->
  ~315 transforms per continuous part; this makes MSSD/MSPD the slow path. VSD off by default.
- **Point subsampling**: `--n-points 2000` (default) caps each mesh for ADD/ADI/MSSD/MSPD so the
  ~315x continuous-symmetry expansion stays tractable; BOP ships decimated `models_eval` for the
  same reason and the effect on distances is negligible. `--n-points 0` uses the full mesh.

### VSD caveat (read before trusting AR_VSD on this dataset)

The vispy/EGL renderer **works** (`--vsd` runs end to end), but **VSD-of-GT-against-itself is
NOT ~1.0** on `pose_isaac` (GT-vs-GT AR_VSD ~0.49, varies per object). Root cause: VSD's
visibility mask compares a **single-object** render against the dataset's **full-scene** depth
map (table + LARA5 arm + all 6 parts). Where the table/arm/another part sits within `delta=15mm`
of the target surface, the single-object render disagrees with the stored depth, the visible mask
shrinks, and even the true pose scores < 1. This is inherent to single-object VSD against a
full-scene PBR depth -- not a bug in the harness, and a known BOP nuance.

**Consequence:** by default the **headline AR = mean(AR_MSSD, AR_MSPD)** (both are exact and
symmetry-aware, GT-vs-GT = 1.000). Only pass `--vsd` if you also render per-object depth maps that
match the dataset, or accept AR_VSD as a relative (A/B) signal rather than an absolute. For the
GDRNPP A/B comparison MSSD+MSPD are the trustworthy headline; VSD can be reported as supplementary.

### Renderer backends

The toolkit on the box has the renderer under `bop_toolkit_lib.rendering`; **vispy/EGL works**,
while `cpp` (`bop_renderer`) and `python` (`glumpy`) backends are not installed. `--vsd` uses vispy.
