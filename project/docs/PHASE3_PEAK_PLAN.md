# PHASE 3 — Peak-Performance Roadmap to >80% BOP-AR

> **Ticket:** T-058 · **Worktree:** `.worktrees/S-046` · **Date:** 2026-05-24
> **Goal (mandatory):** push overall BOP-AR from **0.31** (Anker ~0.59/0.61, Zahnrad **0.36**)
> to **> 0.80**, on OUR setup: GDRNPP per-object, **RGB-only** (depth on shiny metal is a hard
> no), top-down, robot-arm-occluded, planar table prior, texture-less metal, single RTX 3090,
> YOLOv8-OBB detector, >20%-visibility scope.
>
> Companions: [`PROJECT_REPORT.md`](PROJECT_REPORT.md) §4 · [`PHASE2_PLAN.md`](PHASE2_PLAN.md)
> (the retrain in flight: crop 320/80, 160 ep, DR-5k, COLOR_AUG 0.9) ·
> [`REEVAL_T038_visib20.md`](REEVAL_T038_visib20.md) · [`RESULTS_PHASE2.md`](RESULTS_PHASE2.md) ·
> `bop_adapter.py` (planar Z-snap + canonicalisation — where Phase-3 rotation snap hooks in).
> ADR-018 (BOP pivot) · ADR-017 (pose_result contract).

---

## 0. The honest ceiling first — is >80% even reachable RGB-only?

This is the single most important number in this document, so it leads. The BOP-2023 official
results give the **RGB-only ceiling on texture-less metal** directly:

| dataset (texture-less / metal) | best **RGB-only** AR | best **RGB-D** AR | RGB-D − RGB gap |
|---|---|---|---|
| **T-LESS** (texture-less, symmetric, plastic+metal) | **79.9** (GPose2023-RGB) | 91.4 (GPose2023) | **+11.5** |
| **ITODD** (texture-less **metal**, our closest analogue) | **46.0** (GPose2023-RGB) | 70.4 (GPose2023) | **+24.4** |

Source: BOP Challenge 2023, https://arxiv.org/html/2403.09799v2 (Tables for T-LESS / ITODD,
seen-object pose task).

**What this means for our target, stated plainly:**

- **T-LESS-like difficulty → ~80 AR is achievable RGB-only.** The very best public RGB-only
  method *just reaches* 79.9 on T-LESS. So **>0.80 overall is at the absolute edge of what
  RGB-only delivers even at SOTA**, and only on the *easier* (T-LESS-like) end of texture-less.
- **ITODD-like difficulty → ~80 AR is NOT reached by anyone RGB-only.** The ITODD ceiling is
  46 AR RGB-only. ITODD is the dataset that most resembles us (texture-less *metal*, industrial,
  CAD-only). Public SOTA cannot crack 50 there without depth.
- **Our reality is between the two**, but with two *advantages* ITODD lacks: (a) a **strong
  planar prior** (parts rest on a known plane — kills the Z error that tanks ITODD), and (b) we
  train **per-object** on **in-distribution synthetic data** (ITODD has almost no PBR training
  data, which is *why* its RGB number is so low). Those two advantages are exactly the levers
  this roadmap pulls.

**Verdict (carried to §6):** **>0.80 overall is reachable, but only if (i) the planar prior is
exploited for ROTATION as well as Z, (ii) a render-and-compare / contour refiner is added, and
(iii) the Zahnrad C_7 is fixed by representation, not by hoping more pixels do it.** Anker alone,
with the flip killed, lands ~0.80–0.88; the Zahnrad is the part that decides whether the *overall*
mean clears 0.80. A defensible honest target ladder: **Anker → 0.85+, Zahnrad → 0.70–0.80,
overall 0.78–0.85.** Hitting *strictly* >0.80 overall hinges almost entirely on the Zahnrad.

---

## 1. Our failures mapped to the literature (no general platitudes)

| our failure (from §0 of PHASE2_PLAN + eval JSON) | what it actually is | SOTA framing |
|---|---|---|
| Anker `rot_naive_mean` ~98–105°, 13–19% ≥90° flips, sym-median fine (5–8°) | **end-to-end 180° flip** of the rod — a **view-dependent** near-symmetry, NOT the declared cont-Y | BOP-Distrib: per-instance/view symmetry; flip looks identical *from this view* though not globally symmetric |
| Zahnrad MSSD **0.043**, rot-median 87–91°, 1% <5°, 48% flips | **C_7 in-plane orientation unlearned** — teeth are the only cue, a few px at 256/64 | correspondence/regression heads can't commit to a tooth → SymCode/ZebraPose/SC6D; resolution ceiling 256/64 |
| MSSD ≪ MSPD (0.26 vs 0.57) on Anker; 27mm trans on Zahnrad | **Z/optical-axis translation** — classic RGB-only weakness | ITODD RGB→RGB-D gap is +24 *because of Z*; our planar Z-snap already halves it |
| ~23% GT unmatched (pre-T038), occlusion tail | Sim2Real + arm-occlusion coverage | shape-bias DR (R3), material-recon (R2), refiners recover marginal matches |

Two of these (Anker flip, Zahnrad C_7) are **rotation** problems and are the whole AR story now
that planar Z-snap fixed translation. **Phase 3 is a rotation-disambiguation phase.**

---

## 2. The measures, researched (with sources + expected impact)

### M1 — Stable-pose ROTATION snapping (planar prior for rotation, not just Z) ★ TOP LEVER
**Idea.** We already snap Z to the table plane (`planar_z_snap`). The *same* planar prior
constrains **rotation**: a part at rest on a known plane can only be in one of **K discrete
stable resting poses** = (which face is down) × (in-plane yaw about world-Z). `trimesh`'s
`compute_stable_poses(mesh)` returns exactly these K 4×4 transforms **sorted by landing
probability** (it samples the CoM and computes which faces are statically stable).
Source: https://trimesh.org/trimesh.poses.html ; impl ref
https://github.com/mikedh/trimesh/issues/1620 ; theory StablePose (CVPR'21)
https://arxiv.org/pdf/2102.09334.

**How it kills our two rotation failures:**
- **Anker flip:** the head-to-tail 180° flip is *one of the K stable poses* (or its CoM-twin).
  Snapping the predicted rotation's **tilt** (out-of-plane part) to the nearest stable rest
  orientation removes the "floating between two near-equal modes" wobble. The remaining in-plane
  ambiguity for the rod is already absorbed by cont-Y. Net: the catastrophic ≥90° tilt-flips
  (the ones cont-Y does NOT forgive) collapse onto a valid resting tilt.
- **Zahnrad:** the gear has essentially **one** stable pose family (flat on a face) × yaw. Snapping
  the *tilt* to "lying flat" is almost free and removes any out-of-plane error; it does NOT fix
  the in-plane tooth yaw (that's M4), but it stops the gear from being predicted tilted/standing,
  which is part of the 87° error.

**Mechanism (training-free, drop-in to `bop_adapter.py`):** after `bop_pose_to_world`, decompose
R_world into (tilt-to-plane-normal) + (in-plane yaw). Replace the *tilt* component with the
nearest stable-pose tilt from `compute_stable_poses` (gated by a confidence/`max_tilt_snap_deg`
guard exactly like the existing `max_snap_m` Z-guard, so a genuinely-held/standing part is not
clobbered). Keep the predicted yaw (the network's good signal) for non-symmetric parts; for
cont-Y parts the yaw is free anyway.

- **Expected AR impact:** **+0.08 to +0.18 overall.** It directly attacks the Anker flip tail
  (the single biggest drag on Anker mean-rot, 98–105° naive) and removes the gear's out-of-plane
  error. This is the **same class of win** the existing Z-snap gave (translation halved) but on
  the rotation axis. Strong because it is *exact* (geometry, not learned) and free of Sim2Real.
- **Effort:** **S–M** (numpy + trimesh, lives next to `planar_refine`; ~1 file, guards + a unit
  test mirroring the Z-snap tests). **No training. No GPU.**
- **Generalizable:** **YES, fully** — any part + CAD + known plane. Same primitive as Z-snap.
- **Risk:** mis-snapping a part that is *not* resting (in gripper/standing). Mitigated by the
  tilt-snap guard (reuse the resting-confidence logic from `_planar_tilt_correct` /
  `contact_planarity`, which already exists in the adapter).
- **Dependencies:** none. Builds on existing `planar_refine`. Do this FIRST.

### M2 — Render-and-compare / contour refiner on top of GDRNPP coarse ★ TOP LEVER
**Idea.** GDRNPP gives a *coarse* pose; a refiner iterates render-vs-observed to sharpen it.
BOP-2023 numbers for how much a refiner adds on top of coarse:
- **GenFlow:** coarse 23.5 ARC → refined **67.4 ARC** = **+43.9** (RAFT-style recurrent flow,
  RGB). https://arxiv.org/html/2403.11510v1
- **MegaPose** refiner: **+23.7** average for the model variant; RGB-only refiner beats CosyPose's
  refiner. https://arxiv.org/abs/2212.06870
- **FoundPose** (DINOv2, training-free) + render-compare = RGB-only SOTA for unseen objects.
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03742.pdf

**Why a refiner is decisive for texture-less metal specifically:** metal has **strong, reliable
edges/contours** even when it has no texture. The right refiner objective for us is
**edge/silhouette/contour alignment**, not photometric. ContourPose (IEEE T-RO 2023) is built
for *exactly our object class* — "Monocular 6-D pose for reflective texture-less metal parts" —
using a contour decoder as a geometric prior to iteratively solve pose, reporting significant
gains over keypoint/PVNet baselines on reflective metal.
https://ieeexplore.ieee.org/document/10189174/ . Classic edge/chamfer refinement for
texture-less industrial parts: https://link.springer.com/chapter/10.1007/978-3-030-66645-3_35.

**Mechanism:** add a refinement stage after GDRNPP. Two build options:
- **(A) MegaPose RGB refiner** (off-the-shelf, render-and-compare, CAD-driven, no per-object
  train) as a coarse→fine pass on GDRNPP output. Fastest to stand up; already in our Track-A
  stack (CNOS→GigaPose→MegaPose exists).
- **(B) Edge/silhouette refiner** (chamfer / IoU on the rendered contour vs detected edges) —
  lighter, ICP-free, plays to metal's strong edges and our exact-CAD + known-plane setup. Pair
  with M1 so the refiner only searches the in-plane yaw + small residual (huge speed/robustness
  win because the stable-pose prior collapses the search space).

- **Expected AR impact:** **+0.10 to +0.25**, larger on the Zahnrad (a contour/edge refiner can
  lock the tooth ring's in-plane yaw that the coarse net misses) and on the occlusion tail. Even
  the conservative MegaPose-style +0.08–0.10 on already-good Anker poses + a big Zahnrad recovery
  is the difference between 0.31 and ~0.55–0.65 *before* M1/M4.
- **Effort:** **M** (A: wire MegaPose refiner to GDRNPP output — most plumbing exists) to **L**
  (B: build a contour refiner). **No retrain for option A**; GPU at inference only.
- **Generalizable:** **YES** — refiner is object-agnostic (CAD-driven).
- **Risk:** refiner can diverge from a bad coarse init → gate by render-vs-observed score (keep
  coarse if refine score worse), exactly MegaPose's design.
- **Dependencies:** best *after* M1 (stable-pose init makes the refiner converge in far fewer
  iterations and stops it walking into the flipped basin).

### M3 — Fix the Anker 180° flip the RIGHT way: per-instance symmetry, not a blind C_2 ★ TOP LEVER
**Idea.** The flip tail is the Anker's whole AR drag. PHASE2 §2.1 flagged the choice: add a
discrete C_2 to `models_info.json`, OR treat it as a learning problem. The new evidence resolves
*how* to decide: **BOP-Distrib** (arXiv:2408.17297, https://arxiv.org/html/2408.17297v2) shows
the flip is a **view-dependent (per-instance) ambiguity**, not necessarily a global symmetry —
the rod looks flip-identical *from a top-down view* even if a tiny feature breaks the symmetry
globally. Their fix: a **soft-intersection** symmetry pattern per image (tolerance τ over visible
points). The practical recommendation for a near-symmetric part is: **if the visible surface is
explained equally well by the flip, both poses are legitimate** — so declare the symmetry the
metric/loss uses to match what is *actually visible top-down*.

**Concrete decision rule (must look at the CAD — owned by data-strang):**
- If the Anker rod is **genuinely ≥99% flip-identical** (no symmetry-breaking notch/tag visible
  top-down) → **add discrete C_2 about the long axis** to `models_info.json`. This collapses the
  flip tail *analytically* in BOTH loss and metric — zero training cost — exactly as cont-Y
  collapses the twist. Regenerate `fps_points.pkl` / `keypoints_3d.pkl` after the edit.
- If a feature **does** break it → keep cont-Y, and let **M1 (stable-pose tilt snap) + M2
  (refiner) + M5 (SO(3) multi-hyp)** resolve it; do NOT fake a C_2 (a wrong C_2 makes a *correct*
  unflipped pose score as flip-equivalent — false credit).

- **Expected AR impact:** if C_2 is valid: **+0.05 to +0.12 on the Ankers** (median already fine;
  this collapses the mean-rot tail, ~98° → low). If not valid, the impact moves into M1/M2/M5.
- **Effort:** **S** if it is a sym-def edit (1 line in models_info.json + regen fps/keypoints,
  data-strang). **No new code in pose.** Possibly a short retrain only if the loss should also
  see C_2 (the metric alone helps immediately at eval).
- **Generalizable:** the *rule* (declare exactly the real symmetry group, view-aware) generalizes.
- **Risk:** **correctness** — wrong C_2 = false credit. Decide from CAD first (BOP-Distrib's
  soft-τ logic = "is the visible-from-top surface flip-invariant?").
- **Dependencies:** CAD inspection. Independent of M1/M2 otherwise.

### M4 — Zahnrad C_7: discrete-surface encoding + resolution + contour yaw-lock (the part that decides >80%)
**Idea.** The gear's C_7 in-plane yaw is genuinely *unlearned* (MSSD 0.043, 1% <5°), confirmed a
**model** failure not a metric artefact (self-test [c] passes). Three converging fixes, in order
of leverage:
1. **Discrete surface encoding instead of dense XYZ regression.** GDRNPP's region/XYZ head wants
   a one-to-one pixel→surface map, which is ill-posed for a 7-fold-symmetric ring. **ZebraPose**
   (CVPR'22, hierarchical binary surface code, https://arxiv.org/pdf/2203.09418) and **SymCode/
   SymNet** (one-to-**many** correspondences for discrete symmetry, https://arxiv.org/html/2405.10557v1
   — reports T-LESS discrete-sym recall **38.5%→78.0%**, ~2×) directly target this. **SC6D**
   (3DV'22, https://arxiv.org/pdf/2208.02129) is **correspondence-free**: it learns an SO(3)
   embedding and *classifies* rotation by cosine-similarity to sampled rotations — **symmetry-
   agnostic**, SOTA on T-LESS, no CAD-symmetry needed. SC6D's rotation-classification head is the
   most surgical add for a gear: it picks the best of N sampled SO(3) rotations rather than
   regressing a single one the net can't commit to.
2. **Resolution** (already in the in-flight retrain: 320/80). Necessary but **not sufficient** —
   the perceptual-hashing/template study shows even *exhaustive* 10°-step silhouette matching
   cannot break gear/teeth ambiguity by pixels alone (object-7 cup, C-sym, only 0.29 recall:
   https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1424036/full).
   If 320/80 still leaves MSSD low, push the **gear SO-config to 384/96** (per-object resolution
   scales with object size — generalizable, not a hack).
3. **Contour/template yaw-lock at refinement (M2-B).** After M1 fixes the tilt (gear flat), the
   *only* remaining DoF is the in-plane yaw — a **1-D search** over 7 tooth-aligned candidates.
   A contour/silhouette match (ContourPose-style) over those 7 yaws is cheap and exact and is the
   most reliable way to pick the tooth alignment the coarse net misses.

- **Expected AR impact:** **Zahnrad 0.36 → 0.65–0.80** if (3) lands (1-D yaw search is highly
  tractable); (1) is the deeper but heavier fix if the refiner is insufficient. This is the
  measure that moves the **overall** number across 0.80, since Anker is already near-target.
- **Effort:** (3) **M** (1-D contour search, builds on M2-B + M1) · (2) **S** (config, in flight)
  · (1) **L** (new head / model family, retrain).
- **Generalizable:** the 1-D yaw-lock generalizes to any single-DoF-residual part after M1; the
  discrete-encoding head generalizes to all discrete-symmetric parts.
- **Risk:** if 320/80 + M1 + contour-yaw still fails, only the heavy (1) remains → schedule risk.
- **Dependencies:** **M1 first** (collapses to 1-D), then M2-B. (1) is the fallback.

### M5 — SO(3) multi-hypothesis head (Implicit-PDF / EPOS) — deferred, principled
**Idea.** Emit a *distribution* over SO(3) (top-K rotations) instead of one, then resolve at test
time with the planar prior (M1). Principled cure for residual ambiguity (Anker flip if not a true
C_2; gear if discrete encoding not adopted). Implicit-PDF https://arxiv.org/pdf/2106.05965 ; EPOS
https://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf ;
SurfEmb (learns ambiguity unsupervised, **79% relative improvement over next-best RGB on ITODD**)
https://arxiv.org/abs/2111.13489.
- **Impact:** moderate-high but **redundant if M1+M3 work** (they resolve the same ambiguity more
  cheaply). Keep as the fallback if M1/M3/M4 plateau.
- **Effort:** **L** (arch change + retrain). **Generalizable:** yes. **Defer.**

### M6 — Shape-bias Sim2Real data (in flight) + detector→pose coupling
**Idea.** Texture-less metal must be learned shape-first. The in-flight retrain already does this
(DR-5k, COLOR_AUG 0.9). Evidence: randomized-texture/shape-bias **+18.6% ITODD pose, +60.3%
detect** (R3, https://arxiv.org/html/2402.04878); material-recon head (R2,
https://arxiv.org/pdf/2503.03655). **Data volume:** ITODD's low RGB number is largely a
*data-starvation* artefact — per-object PBR + DR at ~8–16k in-distribution frames is already in
the right ballpark; the lever is **DR quality** (per-object roughness/metallic, lighting 120–2200,
specular), not raw count. **Detector coupling:** GDRNPP confirms pose accuracy rises with detector
quality and dynamic-zoom handles crop-tightness variance
(https://arxiv.org/html/2102.12145v5). Our YOLOv8-OBB at mAP50 0.991 is **not** the bottleneck —
keep it; ensure the OBB→AABB crop is tight + dynamic-zoom-augmented so coarse Z (centroid-scale)
is stable.
- **Impact:** **+0.03–0.08** on the unmatched/occluded tail (recall → AR), across all parts.
- **Effort:** **M** (data-strang renders/converts; in flight). **Generalizable:** yes.

### M7 — Honest depth trade-off (RGB-only stays default)
Per BOP-Industrial: depth adds **+11.5 AR on T-LESS, +24.4 AR on ITODD**
(https://arxiv.org/html/2403.09799v2). That is the biggest single AR lever in existence — and it
is **off the table** by hard rule (Zivid depth on shiny metal is noise). **But** the question was
"any salvageable geometry signal?": the defensible compromise is **confidence-filtered depth used
ONLY where trustworthy** (matte sub-regions, edge-consistent pixels) as a *refiner constraint*,
not an input channel — i.e. M2-B's contour/edge term can ingest stereo/edge geometry where it is
reliable without making the net depth-dependent. **Recommendation:** keep RGB-only as default and
shipped; document depth-refine as the measured +10–24 AR option *if* the planar+contour route
plateaus below 0.80. The planar prior is our *substitute* for the depth Z-signal and is why we can
aim near the T-LESS (not ITODD) ceiling RGB-only.

---

## 3. Ranked roadmap (impact / effort / training / GPU / generalizable / deps)

| rank | measure | exp. AR impact | effort | retrain? | GPU? | general? | depends on |
|---|---|---|---|---|---|---|---|
| **1 ★** | **M1 stable-pose ROTATION snap** | **+0.08–0.18** | S–M | **no** | **no** | yes | — (extends Z-snap) |
| **2 ★** | **M2 render/contour refiner (A=MegaPose, B=edge)** | **+0.10–0.25** | M (A) / L (B) | no (A) | infer | yes | M1 (init) |
| **3 ★** | **M3 Anker per-instance C_2 (if CAD-valid)** | **+0.05–0.12** | S | no* | no | rule | CAD inspect |
| **4 ★** | **M4 Zahnrad: M1+contour yaw-lock (then 320/80→384/96)** | **Zahnrad→0.65–0.80** | M (+S) | partial | infer | yes | M1, M2-B |
| 5 | M6 shape-bias DR + tight crop (in flight) | +0.03–0.08 | M | **in flight** | yes | yes | render done |
| 6 | M5 SO(3) multi-hyp (Implicit-PDF/SC6D head) | +0.05–0.10 | L | yes | yes | yes | fallback |
| 7 | M7 confidence-filtered depth-as-refiner-constraint | +0.10–0.24 | L | no | infer | yes | last resort |

\* M3 helps at **eval immediately** via the metric; a short retrain only if the *loss* should also
see C_2.

### Top 3–5 highest-leverage **buildable** measures (the ones to build now)
1. **M1 — stable-pose rotation snap** (training-free, builds on existing planar Z-snap, kills the
   Anker tilt-flip + gear tilt). **Build first.**
2. **M2 — render-and-compare / contour refiner** (MegaPose RGB to stand up fast; edge-refiner for
   the metal-edge win). Biggest single learnable lever; +44 ARC precedent (GenFlow).
3. **M4 — Zahnrad 1-D yaw-lock** = M1 (gear→flat, collapses to 1 DoF) + contour search over 7
   tooth yaws. **This is the measure that decides whether overall clears 0.80.**
4. **M3 — Anker per-instance C_2** (1-line + regen, if CAD confirms top-down flip-identity). Free
   tail-collapse on the better-performing parts.
5. **M6 — finish the shape-bias DR retrain** (already in flight) for the occlusion/recall tail.

These five are mutually reinforcing: **M1 collapses the search space → M2/M4 converge cheaply →
M3 frees the Anker tail → M6 lifts recall.** None except M6 needs a new training run; M1+M3 need
**zero GPU**.

---

## 4. Can we hit >0.80? — realistic call

**Yes, but it is genuinely tight and rests on the Zahnrad.** Reasoning, grounded in the numbers:

- **RGB-only SOTA on T-LESS-difficulty = 79.9 AR** — so >0.80 is *at the very ceiling* of RGB-only
  even for easier texture-less. **ITODD-difficulty RGB-only = 46** — nobody is near 0.80 on metal
  without depth. We sit between, but with two ITODD-beating advantages: **per-object in-distribution
  PBR data** and a **planar prior** that substitutes for depth's Z-signal.
- **Projected per-part with this roadmap:**
  - **Anker (0.59/0.61 → ~0.82–0.88):** M1 (tilt-flip collapse) + M3 (C_2 if valid) + M2 refiner.
    The median is already 5–8°; only the tail is broken, and M1/M3 target exactly the tail.
  - **Zahnrad (0.36 → 0.65–0.80):** M1 (→flat) + M4 (1-D contour yaw-lock) + 320/80→384/96.
    Realistically **0.70–0.78** is the honest expectation; 0.80 needs the contour yaw-lock to be
    very clean.
  - **Overall (n-weighted over the 3 parts ≈ equal):** **0.75–0.85.** The point estimate lands
    **~0.80**, i.e. >0.80 is reachable but not guaranteed by margin — it is decided by the Zahrad.
- **What it takes to *secure* >0.80 (not just touch it):**
  1. M1 + M2 + M4 all built and working (the rotation trio).
  2. Either the Anker C_2 is CAD-valid (M3 free win) **or** the refiner is strong enough to cover
     the flip — having both de-risks it.
  3. Zahnrad contour yaw-lock genuinely locks ≥70% of tooth alignments. If it does not, fall to
     M5 (SC6D rotation-classification head, retrain) for the gear specifically.
  4. The in-flight DR retrain lands (recall tail) and Sim2Real on real Zivid is *measured* (the
     biggest unquantified risk — see §5).
- **If RGB-only plateaus at ~0.75:** the only remaining +10–24 AR lever is depth (M7), which the
  hard rule forbids as an input. The escape valve that respects the rule is **confidence-filtered
  depth/edge geometry as a refiner *constraint* only** (M7), kept out of the network — document
  as the contingency.

**Bottom line:** **>0.80 is a credible, buildable target — on the rotation-disambiguation route
(M1+M2+M4+M3), without depth, on the one 3090 — with the Zahnrad as the swing factor.** Plan for
0.78–0.85; the realistic central case is ~0.80.

---

## 5. Biggest risks on the **running Phase-2 training** (do NOT touch it; these are watch-items)

These are the concrete ways the in-flight retrain (320/80, 160 ep, DR-5k, COLOR_AUG 0.9) under-
delivers, with mitigations — verifiable only on the box.

1. **Smoke/OOM at INPUT_RES=320, batch 16.** The `--smoke` gate exists; if it OOMs → batch 12 or
   fall back to 256/64. **#1 launch blocker** (PHASE2 §4.3). *Watch:* `nvidia-smi` mem at probe.
2. **Too little >20%-visible GT for the Zahnrad.** T-038 dropped ~45% of instances at visib≤0.20;
   the gear is small and easily occluded → its kept-instance count may be thin, starving the C_7
   yaw signal further. *Mitigation:* verify per-object kept counts post-filter; if Zahnrad is
   under-represented, oversample it or lower its filter to 0.15 *for training only* (eval stays
   0.20). This is plausibly *why* 320/80 alone won't fix the gear → reinforces M4's yaw-lock.
3. **320/80 simply not enough for the teeth.** Confirmed risk from the template study (pixels
   alone can't break tooth ambiguity). *Mitigation:* M4 — bump gear SO-config to 384/96 and rely
   on the contour yaw-lock; do not expect resolution alone to fix MSSD 0.043.
4. **Sim2Real on real Zivid is UNMEASURED.** All numbers are synth-val. The DR-5k set may still
   leave a real-domain gap (specular highlights, real arm shadows) that synth-val cannot see.
   *Mitigation:* this is the **largest unquantified risk to >0.80** — collect a small real
   Zivid+GT set ASAP to measure it; otherwise the 0.80 claim is synth-only.
5. **DR-5k convert/wiring slips (data-strang).** If `isaac_to_bop --min-visib 0.20` / A-vs-B
   wiring isn't ready, train on train_pbr alone (partial M6) — still gets M1/M3 (no training).
6. **Calendar contention.** 160 ep × 3 objects sequential on one 3090 is multi-day; the render
   must finish before training so they never contend. Plan M1/M3 (zero-GPU) to land *during* the
   training so progress is decoupled from the GPU.

---

## 6. Build order (dependency-correct)

```
NOW (no GPU, parallel to training):
  M1  stable-pose rotation snap            → bop_adapter.py (next to planar_refine) + tests
  M3  CAD inspection → C_2 decision        → data-strang (models_info.json) if valid
THEN (inference GPU):
  M2A wire MegaPose RGB refiner on GDRNPP coarse (init from M1)
  M4  gear 1-D contour yaw-lock (M1→flat, search 7 tooth yaws)  [+ M2B edge refiner]
WHEN training lands:
  M6  re-eval with DR-5k weights; diff vs §0 table
IF plateau < 0.80:
  M5  SC6D rotation-classification head for the gear (retrain)  | M7 depth-as-constraint (contingency)
```

---

## Sources (all consulted this round)
- BOP Challenge 2023 (RGB vs RGB-D ceilings, refiner gains): https://arxiv.org/html/2403.09799v2
- MegaPose (render-compare refiner, +23.7): https://arxiv.org/abs/2212.06870 · https://megapose6d.github.io/
- GenFlow (coarse 23.5 → 67.4 ARC): https://arxiv.org/html/2403.11510v1
- FoundPose (DINOv2 RGB SOTA refiner): https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03742.pdf
- ContourPose (reflective texture-less METAL, contour refiner): https://ieeexplore.ieee.org/document/10189174/
- Edge/chamfer refinement for texture-less industrial parts: https://link.springer.com/chapter/10.1007/978-3-030-66645-3_35
- trimesh compute_stable_poses (K stable rest poses + probs): https://trimesh.org/trimesh.poses.html · https://github.com/mikedh/trimesh/issues/1620
- StablePose (geometrically stable patches): https://arxiv.org/pdf/2102.09334
- BOP-Distrib (per-instance / view-dependent symmetry, soft-τ): https://arxiv.org/html/2408.17297v2
- ZebraPose (discrete hierarchical surface code): https://arxiv.org/pdf/2203.09418
- SymCode/SymNet (one-to-many corr; T-LESS discrete recall 38.5→78.0): https://arxiv.org/html/2405.10557v1
- SC6D (correspondence-free SO(3) classification, symmetry-agnostic, T-LESS SOTA): https://arxiv.org/pdf/2208.02129
- SurfEmb (dense corr. distributions; +79% rel. on ITODD RGB): https://arxiv.org/abs/2111.13489
- Implicit-PDF (SO(3) distribution): https://arxiv.org/pdf/2106.05965
- EPOS (symmetry via many-to-one fragments): https://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf
- Shape-bias texture-agnostic (R3; +18.6% ITODD pose): https://arxiv.org/html/2402.04878
- Metallic-object GDRNPP heads (R2; material-recon): https://arxiv.org/pdf/2503.03655
- Template/perceptual-hash matching for texture-less + gears (silhouette can't break flips): https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1424036/full
- GDRNPP (256/64 ceiling, detector→pose coupling, dynamic zoom): https://arxiv.org/html/2102.12145v5
- XYZ-IBD industrial metal bin-picking dataset: https://xyz-ibd.github.io/

## Related
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) · [`PHASE2_PLAN.md`](PHASE2_PLAN.md) ·
  [`REEVAL_T038_visib20.md`](REEVAL_T038_visib20.md) · [`RESULTS_PHASE2.md`](RESULTS_PHASE2.md) ·
  `bop_adapter.py` (planar_refine / canonicalize_rotation — M1 hooks here) ·
  ADR-018 (BOP pivot) · ADR-017 (pose_result contract)
