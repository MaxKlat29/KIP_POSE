# POSE — Project Report

> Short academic report on the POSE 6D-pose-estimation pipeline for metal
> assembly parts. Companion to the architecture and reuse documentation in
> [`ARCHITECTURE.md`](ARCHITECTURE.md), [`ADD_NEW_PART.md`](ADD_NEW_PART.md),
> [`EVAL.md`](EVAL.md) and [`REFERENCES.md`](REFERENCES.md).

---

## 1. Problem statement

Given a single top-down RGB image of a robot assembly cell — with the **LARA5
robot arm visible in frame** and texture-less metal parts randomly dropped on a
tray — estimate the full **6D pose** (3D rotation + 3D translation) of each part
and render the corresponding CAD mesh at that pose in an interactive 3D viewer.

The setting is deliberately the hard one: texture-less, specular metal; strong
rotational symmetry (anchors, gears, ring magnets) that makes a single view
ambiguous; and self-/arm-occlusion. The visible arm and cluttered tray are part
of the task specification, not a nuisance to be removed.

An earlier in-house pose core (face-atlas + template-bank render-and-compare)
scored **120° / 91° median rotation error and 186 mm translation error** on the
real parts — unusable. This motivated a pivot to the BOP-benchmark state of the
art.

## 2. Method

The pipeline follows the BOP (Benchmark for 6D Object Pose) convention end to
end, which makes every BOP-benchmark method and the official evaluation directly
usable. Two interchangeable middle tracks sit behind one output adapter:

- **Track B (primary, maximum accuracy):** **GDRNPP** — a geometry-guided,
  fully-learning-based pose estimator from the BOP'22 winning line — trained
  per-object on synthetic PBR data. Chosen for its accuracy on ITODD-like
  texture-less metal and its reported synth-only AR of ~82.7 (Sim2Real
  practically closed with strong domain randomisation), all on a single
  RTX 3090.
- **Track A (generalisation + instant baseline):** **CNOS → GigaPose →
  MegaPose** — a zero-shot, RGB-only chain that estimates the pose of a *novel*
  CAD part with no training. It provides an immediate result while Track B
  trains for days, and is the path for "new part in, no training out".

Both tracks are **RGB-only by design**: commodity/Zivid depth degrades exactly on
shiny metal (reflection noise), so depth-mandatory methods were rejected. An
RGB-vs-RGB-D ablation with confidence-filtered depth is left as future work; RGB
remains the default.

The rotational-symmetry ambiguity — the root cause of the eigenbau's 120°/91°
failure — is solved **analytically**: each part declares its symmetry in
`models_info.json` (Anker/Ringmagnet continuous about Y, Zahnrad discrete C_7,
others none), and the BOP metric maps every pose to the nearest symmetric
representative before scoring. The same canonicalisation is applied at inference
for a deterministic viewer pose.

## 3. Pipeline

1. **CAD → BOP model** — GLB meshes exported to PLY in millimetres; symmetry
   flags and bounding geometry written to `models_info.json`.
2. **Synthetic data** — NVIDIA Isaac Sim renders arm-visible top-down PBR frames
   with physically-dropped parts and domain randomisation.
3. **Isaac → BOP** — a converter produces a spec-compliant BOP dataset (camera
   intrinsics/extrinsics, ground-truth poses in mm/row-major, full + visible
   masks); the arm is treated as an occluder, never an object.
4. **Detector** — a YOLOv8-OBB detector, retrained on arm-visible data, supplies
   oriented boxes; an OBB→AABB bridge feeds the BOP pose stage.
5. **Pose** — GDRNPP (Track B) or the CNOS/GigaPose/MegaPose chain (Track A)
   produces `(R_m2c, t_m2c)` in the camera frame.
6. **Adapter + contract** — a single adapter maps the camera-frame pose to the
   frozen world contract (`pose_result.json`); both tracks emit identical output.
7. **Viewer** — a Three.js viewer renders the real CAD at the estimated pose,
   fully decoupled from the pose method via the contract.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full data flow.

## 4. Results

> Pose accuracy figures are filled in once GDRNPP training completes; the eval
> harness and its self-validation are already in place and proven.

| Component | Metric | Value |
|---|---|---|
| Object detector (arm-visible) | mAP50 | **0.991** (measured) |
| Eval harness self-test `[a]` GT-as-prediction | AR / ADD / rot | **AR ≈ 1.0, errors ≈ 0** (validated — plumbing correct) |
| Eval harness self-test `[c]` symmetry-axis perturbation | `rot_deg` vs `rot_naive` | **`rot_deg` collapses to ≈ 0** for symmetric parts (validated — symmetry handling correct) |
| GDRNPP per-object pose | AR (mean MSSD/MSPD) | *to be filled in after training* |
| GDRNPP per-object pose | ADD / ADI | *to be filled in after training* |
| Track A (CNOS→GigaPose→MegaPose) zero-shot | AR | *baseline, to be measured* |

The detector reaching mAP50 0.991 and the symmetry-aware metric collapsing the
symmetry-axis error to ≈ 0 are the two concrete results to date. They establish
that (a) parts are reliably localised under the visible arm, and (b) the
analytic symmetry handling does what it must — a correct symmetric-part pose is
no longer punished, which directly addresses the original 120°/91° failure.

## 5. Limitations

- **Single-view rotational ambiguity** — fundamentally unresolvable from one
  image for symmetric parts. Addressed analytically via symmetry declarations
  (the pose is correct *up to the symmetry group*), which is the correct
  treatment, not a workaround. Parts with subtle symmetry-breaking features
  (a notch, a tag) would need their symmetry downgraded to discrete.
- **RGB-only** — no metric depth in the pose nets. A deliberate choice for shiny
  metal, but it forgoes the absolute-scale signal depth provides; translation
  along the optical axis is the hardest component.
- **Sim2Real gap** — training is on Isaac synthetic data. Synth-only accuracy is
  high *with* strong domain randomisation, but real performance is only provable
  against real Zivid scenes with ground-truth poses, which are required to close
  the loop.
- **Single GPU** — one RTX 3090 forces sequential training; per-object GDRNPP is
  multi-day, so the project ships with a MOCK pose backend that keeps the full
  chain (and viewer) runnable before any checkpoint exists.
- **Detector licensing** — the YOLOv8 backbone is AGPL-3.0; relevant only for
  closed redistribution of the detector, not for the MIT/Apache BOP pose stack
  (see [`REFERENCES.md`](REFERENCES.md) and `THIRD_PARTY_LICENSES.md`).

## 6. Future work

- Finish per-object GDRNPP training and record AR/ADD/ADI; A/B Track A vs
  Track B on an identical scene set.
- Collect real Zivid scenes with ground-truth poses to measure (and close) the
  Sim2Real gap on actual hardware.
- RGB-vs-RGB-D ablation with confidence-filtered Zivid depth (BOP industrial
  evidence suggests +10–15 AR on these parts), keeping RGB as the default.
- Scale synthetic data and strengthen domain randomisation — the highest-impact
  knob on texture-less metal.
- Automate evaluation after each checkpoint (trigger `eval_bop.sh`, archive the
  report) so accuracy is tracked continuously.
- Extend the part set via [`ADD_NEW_PART.md`](ADD_NEW_PART.md); add the excluded
  Poltopf as `obj_id 7` once a valid mesh is available.

---

## Related
- [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`ADD_NEW_PART.md`](ADD_NEW_PART.md) ·
  [`EVAL.md`](EVAL.md) · [`REFERENCES.md`](REFERENCES.md)
- ADR-018 (pivot to BOP SOTA) · ADR-017 (`pose_result` contract)
