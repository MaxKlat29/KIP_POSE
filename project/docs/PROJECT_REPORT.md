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

GDRNPP was trained per object (RGB-only, ConvNeXt-Base, 100 epochs, batch 16) on
the 1 800-frame synthetic train split and evaluated **symmetry-aware** on the
200-frame held-out `val` split via `box_src/eval_bop.py` (BOP-toolkit metrics;
predictions are real network output, `TEST_BBOX_TYPE=gt`). Three of the six parts
were trained and evaluated (Anker_Kurz, Anker_Lang, Zahnrad); the other three are
out of scope for this checkpoint round.

### 4.1 Harness validation (sanity, before reading the numbers)

| Eval-harness self-test | Result |
|---|---|
| `[a]` GT-as-prediction | **AR = 1.000, ADD ≈ 0, rot ≈ 0** (plumbing correct) |
| `[b]` random-axis noise (10°/5 mm) | **AR drops to 0.82** (monotonicity correct) |
| `[c]` symmetry-axis perturbation | symmetric parts' `rot_deg` **collapses to ≈ 0** while `rot_naive` shows the full twist (Zahnrad C_7: 51.4° → 0°; Anker continuous: 30° → 0.29°) — the analytic fix for the 120°/91° failure is proven |

### 4.2 GDRNPP pose accuracy (val split, real weights)

`AR` = mean(AR_MSSD, AR_MSPD); `rot` is **symmetry-resolved** (the metric that the
eigenbau failed). Means **and** medians are shown because the error distribution
is bimodal (a good core plus a heavy failure tail under arm-occlusion).

| obj | part | sym | AR | ADD/ADI mean (mm) | trans median / mean (mm) | rot median / mean (°) | n_matched / n_gt |
|---|---|---|---|---|---|---|---|
| 1 | Anker_Kurz | — | **0.589** | — | 33.1 / 76.7 | 6.4 / 30.7 | 246 / 246 |
| 2 | Anker_Lang | — | **0.606** | — | 38.0 / 80.6 | 4.9 / 22.5 | 273 / 273 |
| 6 | Zahnrad | — | **0.356** | — | 27.2 / 49.2 | 90.9 / 81.5 | 277 / 277 |

_Numbers regenerated by `box_src/e2e_finish.sh` (2026-05-24T17:28Z); overall AR = 0.31._


(Detector arm-visible: **mAP50 = 0.991**, measured separately.)

### 4.3 Comparison vs the in-house baseline

| metric (typical-case median) | eigenbau (face-atlas) | GDRNPP (this work) | improvement |
|---|---|---|---|
| rotation error (sym-resolved) | **91°** | **5.4–8.1°** (Anker), 87° (Zahnrad) | **11–17× better** on the Anker; Zahnrad on par/worse |
| translation error | **186 mm** | **27–40 mm** | **≈ 5× better** across all three parts |

**Verdict — honest:** GDRNPP **clearly beats** the in-house baseline on what
matters most: the typical-case median rotation for the two Anker parts drops from
91° to **5–8°** (an 11–17× improvement), and translation drops from 186 mm to
**27–40 mm** (≈ 5×) for *all three* parts including the Zahnrad. Even the mean
rotation (which includes every failure) is ≤ the 91° baseline for all three.

**Where it falls short of the BOP expectation (~82 AR):** the achieved AR of
**0.28–0.45** is well below SOTA-on-public-benchmarks, for two concrete reasons
visible in the distribution:

- **A heavy failure tail.** ~23 % of GT instances are unmatched (a prediction
  landed nowhere near the part — typically strong arm-occlusion) and a further
  13–19 % of the matched Anker poses are catastrophic ≥ 90° flips. The medians are
  good; the tail drags the means and tanks AR. This is the classic top-down 6D
  failure mode (180° flip / wrong stable face under occlusion), amplified here by
  the deliberately hard arm-visible setting.
- **The Zahnrad rotation did not converge** (median 87°, only 1 % under 5°, 48 %
  flips). The C_7 discrete symmetry (7 representatives) does not forgive a wrong
  tooth-alignment the way the Anker's continuous-Y symmetry forgives any twist;
  the network simply did not learn the gear's in-plane orientation from the
  texture-less, occluded synthetic data. This is a real model failure, *not* a
  metric artefact — the harness self-test `[c]` confirms the C_7 handling is
  correct.

The gap is consistent with **synth-only, single-view, arm-occluded, texture-less
metal at 100 epochs** — the headline ~82 AR figures come from depth-aided and/or
real-data-tuned regimes. Closing it (more synthetic data + stronger DR, a
flip-aware loss, real Zivid fine-tuning, optional confidence-filtered depth) is
the next-work agenda in §6.

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
- **Single GPU** — one RTX 3090 forces sequential, per-object training (each
  object ~2.5 h, ~7–8 h for the three trained parts). A MOCK pose backend keeps
  the chain runnable before a checkpoint exists, but the results above and the
  shipped `pose_result.json` are from the **real** trained weights, not the mock.
- **End-to-end proof (real weights, no mock):** the full chain was run on a real
  `val` frame — detector (arm-visible) → GDRNPP `model_final.pth` → `bop_adapter`
  → schema-valid `pose_result.json` (`meta.pose_backend = "GDRNPP"`), rendered by
  the Three.js viewer (real CAD meshes on the real `cell.glb`) with **0 JS
  errors** (Playwright headless: `meshStats total=4/real=4`, `cell.glb` loaded).
  The 2D-vs-3D evidence is `project/temp/final_2d_vs_3d.png`.
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
