# Adding a New Part — Reuse Guide

> **The generalisation guide.** How to take *any* new CAD part through the POSE
> pipeline so it can be detected and 6D-pose-estimated like the existing six.
> This is what makes the project reusable for a new university or industry
> project: a different tray of parts, same pipeline.

There are two ways to bring a new part in. Pick by how much accuracy you need
*now*:

| Path | Training? | Accuracy | Use when |
|---|---|---|---|
| **A — Zero-shot (Gleis A)** | none | moderate (~0.55–0.6 AR) | you just want a pose *today* from the CAD, no GPU days |
| **B — Trained (Gleis B, GDRNPP)** | yes (multi-day) | high (~0.8 AR with good synth) | the part is a permanent fixture and you want max accuracy |

Both paths share **Steps 1–4** (get the CAD into the BOP world). Path A stops
there; Path B continues with **Steps 5–7** (generate data, retrain, eval).

All commands assume the box layout from `box_src/BOP_SETUP.md`:
`BOX=/mnt/data/kip_pose`, `BOP=$BOX/project/bop/pose_isaac`. Run them through
`box_src/gpu_run.sh` or directly on the box.

---

## Step 0 — Decide the `obj_id` (do this first, never change it later)

`obj_id` is the immutable identity of a part across the whole system: the PLY
filename, `scene_gt`, `models_info`, the detector class, and the viewer registry
must all agree. The current mapping is frozen in
`project/bop_adapter.py → OBJ_ID_TO_PART`:

```
1 Anker_Kurz · 2 Anker_Lang · 3 Buerstenhalter_2polig
4 Getriebegehaeuse_typ4 · 5 Ringmagnet · 6 Zahnrad
```

A new part takes the **next free integer** (`7`, `8`, …). **Never renumber or
reuse an existing id** — every dataset, checkpoint and result already on disk is
keyed by it. Add the new entry to `OBJ_ID_TO_PART` (and `PART_SYMMETRY` /
`PART_LONG_AXIS` if it is symmetric — see Step 3) plus the matching detector
class order in `box_src/gen_models_info.py → PARTS` and
`box_src/isaac_to_bop.py → OBJ_ID`.

> Detector class is 0-based; `obj_id = detector_class + 1`. Keep the detector
> class order identical to the `obj_id` order so the `+1` rule stays valid.

---

## Step 1 — Get a clean CAD mesh (GLB, metres)

The pipeline ingests **GLB in metres**. Put the part's GLB where
`gen_models_info.py` looks for it (default `--glb-dir
/mnt/data/kip_pose/tmpl_build/part_glbs`, file named `<PartName>.glb`), and the
part's USD into the Isaac USD folder (`$BOX/data/SDG/IsaacSim/USD-Files`) if you
will generate synthetic data (Path B).

**Sanity-check the mesh first.** A degenerate export will silently poison
everything (cf. `Poltopf_kurz_centered`, whose ~50 µm extent disqualified it).
Confirm the bounding box is in the right ballpark (centimetres of real metal,
i.e. ~0.01–0.2 m in GLB units) before continuing.

---

## Step 2 — CAD → BOP model (PLY in mm + `models_info.json`)

```bash
BOX=/mnt/data/kip_pose ; BOP=$BOX/project/bop/pose_isaac

/mnt/data/bop/bop-venv/bin/python box_src/gen_models_info.py \
  --glb-dir $BOX/tmpl_build/part_glbs \
  --bop-root $BOP \
  --also-eval
```

This writes `models/obj_<id>.ply` (GLB metres ×1000 → **mm**) and updates
`models/models_info.json` with `diameter`, bounding box, and the symmetry entry.
`--also-eval` duplicates it into `models_eval/` (the meshes the evaluation reads).

The script also **counts the rotational order N** for any part you mark
`discrete` — it rasterises the silhouette along the symmetry axis and reads the
periodicity of the rim/bore (vertex scatter is too sparse and gives noise). For
the Zahnrad this yields **C_7** (the inner 7-spline hub; the outer rim is a
smooth disc).

---

## Step 3 — Declare symmetry (the rotation-ambiguity fix)

This is the single most important correctness step for symmetric parts. Without
it, a *correct* pose of a rotationally symmetric part is scored as a large error
(the eigenbau's 120°/91° problem) and the viewer jitters between equivalent
orientations.

In `box_src/gen_models_info.py → PARTS`, give the new part one of:

| Symmetry | Declare as | Example |
|---|---|---|
| Rotationally symmetric (∞) about an axis (cylinder, ring, shaft) | `"continuous"` | Anker, Ringmagnet — `axis Y` |
| N-fold (gear, polygonal flange, bolt-circle) | `"discrete"` | Zahnrad — C_7, N counted from mesh |
| None (asymmetric body) | `"none"` | Buerstenhalter, Getriebegehaeuse |

The symmetry **axis is the model axis with the two near-equal orthogonal
extents** (for the current parts that is **Y**). The generator writes:
- `continuous` → `symmetries_continuous: [{axis, offset}]`, offset = the axis's
  (x, z) position so it passes through the part centre.
- `discrete` → `symmetries_discrete`: the N−1 rotation matrices `R_y(k·2π/N)`,
  each with `t = (I − R)·centre` so the part rotates about its real centre.

If the part is symmetric, also add it to `project/bop_adapter.py → PART_SYMMETRY`
(and `PART_LONG_AXIS` if its long axis is not Y) so inference canonicalises the
pose the same way the eval does.

**Acceptance gate:** render the part overlaid with the part-under-its-symmetry
transform — they must coincide. No overlay belief = symmetry unverified. The
`models_info.json` symmetry is later confirmed end-to-end by `eval_bop.py
--self-test` (scenario `[c]`, see [`EVAL.md`](EVAL.md)).

---

## Step 4 — Verify the model is BOP-clean

```bash
/mnt/data/bop/bop-venv/bin/python box_src/validate_bop_full.py --bop-root $BOP
# symmetry sanity through the official toolkit:
/mnt/data/bop/bop-venv/bin/python -c "
from bop_toolkit_lib import inout, misc
mi = inout.load_json('$BOP/models/models_info.json', keys_to_int=True)
for oid, info in mi.items():
    n = len(misc.get_symmetry_transformations(info, 0.01))
    print(oid, '->', n, 'symmetry transforms')
"
```

`get_symmetry_transformations` must return: `1` for an asymmetric part, `N` for a
C_N discrete part, ~hundreds for a continuous part (discretised at the BOP19
step). If those numbers match your intent, the model is correct.

> **Path A users stop here.** The new part now has a BOP model with correct
> symmetry. Run Gleis A (CNOS → GigaPose → MegaPose) against it for a zero-shot
> pose — CNOS detects/segments straight from the CAD templates, no detector
> retrain and no GDRNPP training needed (see ADR-018 / `box_src/BOP_SETUP.md`
> for the per-repo invocations).

---

## Step 5 — Generate arm-visible synthetic data (Path B)

```bash
USD=$BOX/data/SDG/IsaacSim/USD-Files

/mnt/data/isaacsim-venv/bin/python box_src/gen_sdg_arm_visible.py \
  --scene $USD/GST_Scene.usd --usd-dir $USD \
  --output $BOX/data/sdg_armvis \
  --num-scenes 2000 --min-obj 7 --max-obj 13 --settle 180 --focus-frac 0.6 \
  --spawn-x 0.18,0.52 --spawn-y 0.08,0.50

# Isaac → BOP (single scene shown; use convert_full_to_bop.py for the full set + split)
/mnt/data/bop/bop-venv/bin/python box_src/isaac_to_bop.py \
  --raw-dir $BOX/data/sdg_armvis --bop-root $BOP --split train_pbr --scene-id 0

# visual GT-pose belief (acceptance gate)
/mnt/data/bop/bop-venv/bin/python box_src/vis_bop_overlay.py \
  --bop-root $BOP --split train_pbr --scene-id 0 --out $BOX/project/temp/bop_check
```

Make sure the new part's USD is in `--usd-dir` so it actually gets dropped into
scenes, and that it carries a semantic label matching its name (so
`isaac_to_bop.py` maps it to the new `obj_id`). The arm stays visible and acts as
an occluder — that is the task and the strongest Sim2Real lever. **More scenes +
stronger domain randomisation is the highest-impact knob** on texture-less metal.

---

## Step 6 — Retrain the detector + GDRNPP (Path B)

The single RTX 3090 means detector and pose training run **sequentially**.
`train_chain.sh` chains them; add the new part to its per-object list.

```bash
# add the new object to the GDRNPP loop in box_src/train_chain.sh, e.g.:
#   run_gdrn <new_part_lowercase> "7/7"
# (and create configs/gdrn/poseIsaacPbrSO/<new_part>.py, copy an existing one)

cd /mnt/data/kip_pose
nohup bash box_src/train_chain.sh > /mnt/data/bop/logs/train_chain.log 2>&1 &
echo "CHAIN_PID=$!"

# poll (do NOT wait inline — GDRNPP per object is 1–2 days):
tail -40 /mnt/data/bop/logs/train_chain.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

The chain: detector retrain (arm-visible, ~1 h) → OBB→AABB val detections →
GDRNPP deploy → per-object GDRNPP training. The detector retrain must keep the
class order so `obj_id = class + 1` stays valid.

---

## Step 7 — Evaluate and wire into inference

```bash
# score the new checkpoint's predictions with the official symmetry-aware metric:
box_src/eval_bop.sh --preds /mnt/data/bop/results/gdrnpp/preds.csv
```

See [`EVAL.md`](EVAL.md) for reading the report. Then point inference at the new
checkpoint:

```bash
python project/e2e_infer.py --image project/input/<scene>.png \
  --checkpoint /path/to/gdrnpp_<new_part>.pth --serve
```

Finally, give the viewer the new part's CAD: drop `<PartName>.glb` into
`project/frontend/assets/parts/` and register it in
`project/frontend/src/partRegistry.js` (and `assets/part_meta.json`). The viewer
keys on `part` name from `pose_result`, which comes from `OBJ_ID_TO_PART` — keep
the names consistent end-to-end.

---

## Checklist

- [ ] **Step 0** — next free `obj_id` chosen; added to `OBJ_ID_TO_PART`,
      `gen_models_info.py PARTS`, `isaac_to_bop.py OBJ_ID` (and symmetry maps if
      symmetric). Existing ids untouched.
- [ ] **Step 1** — clean GLB (metres) in `--glb-dir`; bounding box sane (not
      degenerate).
- [ ] **Step 2** — `gen_models_info.py` ran; `obj_<id>.ply` (mm) + `models_info`
      entry in `models/` and `models_eval/`.
- [ ] **Step 3** — symmetry declared (`continuous` / `discrete` C_N / `none`),
      axis correct, overlay belief produced.
- [ ] **Step 4** — `validate_bop_full.py` clean; `get_symmetry_transformations`
      returns the expected count.
- [ ] **(Path A)** — Gleis A gives a zero-shot pose. *Done.*
- [ ] **Step 5** — USD in `--usd-dir`, arm-visible synth generated, Isaac→BOP
      converted, GT overlay sits on the parts.
- [ ] **Step 6** — part added to `train_chain.sh` + GDRNPP config; chain launched
      under `nohup` and polled.
- [ ] **Step 7** — `eval_bop.sh` report acceptable; checkpoint wired into
      `e2e_infer.py`; GLB + registry added to the viewer.

---

## Related
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the stages this guide instantiates.
- [`EVAL.md`](EVAL.md) — how to read the metric report.
- ADR-018 — the two-track design (zero-shot vs trained).
