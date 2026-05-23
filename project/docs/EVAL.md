# Evaluation — BOP Metrics

> How the pipeline is scored, what the numbers mean, and how to run the harness.
> Backed by the official `bop_toolkit_lib` metrics — symmetry-aware. The harness
> (`box_src/eval_bop.py`, wrapper `box_src/eval_bop.sh`) reads the BOP dataset
> read-only and **never touches training**.

For the box setup the harness runs on, see `box_src/BOP_SETUP.md`. For the full
caveat detail see `box_src/EVAL_BOP.md`.

---

## 1. What is computed

Per object (`obj_id` 1..6) and as an over-objects mean:

| Metric | Meaning |
|---|---|
| **AR** | The BOP19+ headline score. By default `mean(AR_MSSD, AR_MSPD)`; with `--vsd` it becomes `mean(AR_VSD, AR_MSSD, AR_MSPD)`. |
| **AR_MSSD** | Max Symmetry-aware Surface Distance recall (thresholds 0.05..0.50 × diameter). |
| **AR_MSPD** | Max Symmetry-aware Projection Distance recall (thresholds 5..50 px, scaled by `640 / img_width`). |
| **AR_VSD** | Visible Surface Discrepancy recall (δ = 15 mm, τ 0.05..0.50). Needs the renderer + depth. **See the VSD caveat below — off by default on this dataset.** |
| **ADD / ADI** | Average distance of model points. **ADI** (nearest-neighbour) for symmetric parts, **ADD** otherwise. |
| **trans_mm** | Plain translation error in mm (mean + median). Human-friendly. |
| **rot_deg** | **Symmetry-resolved** rotation error in degrees — the analytic fix for the eigenbau 120°/91° punishment. |
| **rot_naive** | Raw geodesic rotation error (no symmetry), shown next to `rot_deg` so the symmetry effect is visible. |

**MSSD and MSPD are the trustworthy headline** here — both are exact,
symmetry-aware, and GT-vs-GT = 1.000.

Symmetry comes straight from `models_eval/models_info.json` via
`bop_toolkit_lib.misc.get_symmetry_transformations`: obj 1/2 (Anker) + obj 5
(Ringmagnet) continuous about Y, obj 6 (Zahnrad) discrete C_7, obj 3/4 none.
This is what stops a correct anchor/gear pose from being scored as a 91°/120°
miss.

---

## 2. How to run

### From the laptop (recommended) — via the wrapper

```bash
# Self-validation (no predictions, no GDRNPP needed):
box_src/eval_bop.sh --self-test

# Score real predictions (a BOP-results CSV that lives ON the box):
box_src/eval_bop.sh --preds /mnt/data/bop/results/gdrnpp/preds.csv

# Add VSD (slower; uses the vispy/EGL renderer + depth maps):
box_src/eval_bop.sh --self-test --vsd
```

The wrapper ships `eval_bop.py` to the box, runs it in the `bop-venv`, and pulls
the report (`report.json` + `report.txt`) back to `./results/eval/`.

### Directly on the box — via `gpu_run.sh`

```bash
box_src/gpu_run.sh -- '
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
| `--preds PATH` | BOP-results CSV with predictions. |
| `--self-test` | Synthesise predictions from GT (3 scenarios). No checkpoint needed. |
| `--vsd` | Also compute VSD (heavy; off by default). |
| `--n-points N` | Cap mesh points for ADD/ADI/MSSD/MSPD (default 2000; `0` = full mesh). |
| `--out DIR` | Write `report.json` + `report.txt`. |

---

## 3. Predictions input — the BOP-results CSV

One row per predicted instance, exactly the BOP19 standard, which is what
`bop_toolkit_lib.inout.save_bop_results` writes and what GDRNPP / GigaPose /
MegaPose emit:

```
scene_id,im_id,obj_id,score,R,t,time
0,0,2,0.97,0.075 -0.364 -0.928 0.190 -0.908 0.372 -0.979 -0.205 0.001,-443.2 313.7 893.4,0.21
```

| Column | Type | Notes |
|---|---|---|
| `scene_id` | int | matches the scene folder (`0` → `val/000000/`). |
| `im_id` | int | image index within the scene. |
| `obj_id` | int | **1..6**. Detector class is 0-based → `obj_id = class + 1`. |
| `score` | float | confidence; `1.0` if unknown. Orders greedy GT matching. |
| `R` | 9 floats | space-separated, **row-major**, model→camera rotation. |
| `t` | 3 floats | space-separated, model→camera translation in **millimetres**. |
| `time` | float | seconds per image; `-1` if unknown. |

**Eval is done in the camera frame** (BOP convention), with the same `R_m2c` /
`t_m2c` (mm) as `scene_gt.json` — *before* the cam→world→contract mapping. The
world mapping in `bop_adapter.py` is only for the viewer.

**Getting GDRNPP output in:** GDRNPP's BOP-2022 test pipeline already writes a
BOP-results CSV via `save_bop_results`. Point `--preds` at it. If a stage emits
metres, multiply `t` by 1000 before writing (GDRNPP works in mm internally for
BOP datasets).

---

## 4. `--self-test` — proving the harness before any checkpoint exists

Predictions are synthesised from the GT itself, three scenarios:

1. **`[a] GT-as-prediction`** — sanity. Expect `AR ≈ 1.0`, `ADD/ADI ≈ 0`,
   `rot ≈ 0`. Proves loading, matching, and metric plumbing.
2. **`[b] GT + 10° random-axis + 5 mm`** — errors must rise for *all* parts and
   AR drop below `[a]`. Proves the metrics react to error.
3. **`[c] GT + 10° symmetry-axis + 5 mm`** — rotation applied *about the model Y
   axis* (the symmetry axis). For symmetric parts the symmetry-resolved `rot_deg`
   must **collapse to ≈ 0** while `rot_naive` stays ≈ 10°; asymmetric parts stay
   punished. **This is the proof the symmetry handling works** — the same
   mechanism that stops a correct anker/zahnrad pose from being scored as a
   91°/120° miss.

The output is a per-object table plus explicit PASS/FAIL verdicts.

---

## 5. Reading the report

`report.txt` is a per-object table; `report.json` is the machine-readable form.
Read it top-down:

1. **Per-object MSSD/MSPD AR** — high (→ 1.0) is good. Low on a *symmetric* part
   while `rot_deg` is small means MSSD/MSPD is fine and any "rotation error" is
   just the symmetry ambiguity (correct). Low with large `trans_mm` is a real
   translation problem.
2. **`rot_deg` vs `rot_naive`** — for symmetric parts `rot_deg ≪ rot_naive` is
   expected and *good*: the symmetry resolved the ambiguity. For asymmetric parts
   they should be ≈ equal.
3. **Over-objects mean AR** — the single headline number to compare across runs
   (A/B between Gleis A and Gleis B, or across GDRNPP schedules).

---

## 6. Caveats

- **VSD is off by default and that is intentional.** On `pose_isaac`,
  VSD-of-GT-against-itself is **not** ≈ 1.0 (~0.49, varies per object). Cause:
  VSD compares a *single-object* render against the dataset's *full-scene* depth
  map (table + LARA5 arm + all parts); where the arm/table sits within δ = 15 mm
  of the target surface the visible mask shrinks and even the true pose scores
  < 1. This is inherent to single-object VSD against full-scene PBR depth — a
  known BOP nuance, not a harness bug. Use VSD only as a *relative* (A/B) signal,
  or render per-object depth that matches the dataset. The default headline
  `AR = mean(AR_MSSD, AR_MSPD)` sidesteps it.
- **Greedy matching** by translation distance per `(scene, im, obj_id)`,
  estimates ordered by score; unmatched GT counts as a miss. Matches BOP
  localisation semantics for this near-single-instance synthetic holdout.
- **MSPD scaling** — renders are 1280 px wide → factor `640/1280 = 0.5` applied
  to the px error, handled internally.
- **Continuous symmetry** is discretised at `max_sym_disc_step = 0.01` (BOP19
  default) → ~315 transforms per continuous part; this makes MSSD/MSPD the slow
  path. `--n-points` caps each mesh to keep it tractable; the effect on distances
  is negligible (BOP ships decimated `models_eval` for the same reason).
- **Renderer backend** — the box's toolkit has vispy/EGL working; the `cpp`
  (`bop_renderer`) and `python` (glumpy) backends are not installed. `--vsd`
  uses vispy.

---

## Related
- [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`ADD_NEW_PART.md`](ADD_NEW_PART.md) ·
  [`REFERENCES.md`](REFERENCES.md)
- `box_src/EVAL_BOP.md` — the harness's own full notes.
