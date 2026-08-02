# PHASE 2 — Research-Backed Generalizable Pose Fixes + Retrain Plan

> **Ticket:** T-050 · **Worktree:** `.worktrees/S-041` · **Date:** 2026-05-24
> **Scope:** analyse our real eval failures, find SOTA fixes via web research, design
> **generalizing** pipeline fixes (no per-part hacks), and make the Phase-2 GDRNPP
> retrain start-ready (config/code only — **NOT launched**, the GPU is rendering).
>
> Companion: [`PROJECT_REPORT.md`](PROJECT_REPORT.md) (§4 results),
> [`REEVAL_T038_visib20.md`](REEVAL_T038_visib20.md) (visibility filter re-eval),
> `eval_gdrnpp_val.json` (raw per-object numbers). ADR-018 (BOP pivot).

---

## 0. Our actual failures (the numbers we are fixing)

From `eval_gdrnpp_val.json` (unfiltered) + `REEVAL_T038_visib20.md` (visib>0.20 filtered),
real network output, symmetry-aware metric, GT boxes:

| obj | sym | AR (unfilt → filt) | trans med/mean (mm) | rot med (sym) | **rot_naive mean** | MSSD / MSPD | %<5° | n_matched/n_gt |
|---|---|---|---|---|---|---|---|---|
| 1 Anker_Kurz | cont-Y | 0.42 → **0.59** | 36.6 / 84.5 | **8.1°** | **105.2°** | 0.26 / 0.57 | — | 300/390 |
| 2 Anker_Lang | cont-Y | 0.45 → **0.61** | 39.8 / 90.8 | **5.4°** | **98.2°** | 0.27 / 0.63 | — | 308/397 |
| 6 Zahnrad | **C_7** | 0.28 → **0.36** | 27.2 / 47.4 | **87.1°** (failed) | 122.9° | **0.04** / 0.51 | **1%** | 297/392 |

Three diagnostic signals jump out of these numbers:

1. **Anker flip tail.** Median sym-rot is excellent (5–8°) but `rot_naive_mean` is ~98–105°
   and 13–19% of matched Anker poses are ≥90° catastrophic flips. The continuous-Y symmetry
   *does* collapse a twist about the long axis to ~0°, so the median is good — but the heavy
   90–180° tail is **not** absorbed by cont-Y, which means it is a *different* ambiguity (end-to-end
   180° flip of the stab), not the one we declared.
2. **Zahnrad rotation unlearned.** MSSD = **0.043** (catastrophic), median sym-rot 87°, only 1%
   under 5°, 48% flips. The C_7 metric self-test passes (51.4°→0°), so this is a genuine **learning**
   failure, not a metric artefact.
3. **Translation = AR ceiling.** For both Ankers MSSD (0.26) ≪ MSPD (0.57–0.63). MSPD is the
   2D-reprojection metric (in-plane is fine); MSSD is the full 3D-displacement metric. The gap means
   **depth / translation along the optical axis (Z)** is what tanks AR — the classic RGB-only weakness.

---

## 1. Research findings (SOTA, web)

| # | Source | Technique | Relevance to us | Reported gain |
|---|---|---|---|---|
| R1 | **SymCode / SymNet** — Resolving Symmetry Ambiguity in Correspondence-based Methods, arXiv:2405.10557 | One-to-**many** 2D-3D correspondences: a pixel maps to the *set* of symmetry-equivalent surface points; hierarchical binary partition of the correspondence sets; end-to-end (no PnP-RANSAC). | **Directly the Zahnrad failure.** Correspondence/region methods (GDRNPP has a region head) enforce one-to-one maps that are ill-posed for discrete-symmetric parts → the net cannot commit to a tooth alignment. | **T-LESS discrete-sym objects: recall 38.5% → 78.0% (2×)**; T-LESS AR 0.677 → 0.736 |
| R2 | **Improving 6D Pose of metallic Household/Industry Objects** (GDRNPP-based), arXiv:2503.03655 | Adds (a) a **keypoint heatmap head** (3D Harris salient points) and (b) a **material-estimation head** that reconstructs a "non-metallic / high-roughness" version of the crop before the geo head. BlenderProc PBR, 5 lighting × 3 bg. | Both heads are generalizable (use synth GT geometry/render params, no per-object tuning). Material-recon directly attacks specular metal. | Industry AR **6.97% → 14.24%** (keypoint+BAM); cans **31.8% → 40.8%** (material-recon) |
| R3 | **Shape-biased Texture-Agnostic Representations** (Thalhammer et al.), arXiv:2402.04878 | **Randomized-texture UV-mapping** at render time (1226 textures) → destroys texture cues → network becomes shape/geometry-biased. Pure data-generation change, no arch/loss change. | The single highest-leverage Sim2Real lever for texture-less metal; integrates in the render (our data-strang) + on-the-fly grayscale/colour aug. | **ITODD metal: detect mAP +60.3%, pose +18.6%**; T-LESS pose +7.4% |
| R4 | **Implicit-PDF** (arXiv:2106.05965) + **EPOS** (CVPR'20) | Represent a full **distribution over SO(3)** (multi-hypothesis) instead of a single rotation; reason about symmetry/uncertainty; pick the mode. EPOS handles symmetry via many-to-one fragment correspondences. | A principled way to *not* be punished for the Anker 180° flip — emit top-K hypotheses, resolve at test time with the planar prior. Heavier (arch change); deferred. | Qualitative on symmetric objects; SOTA on SO(3) density |
| R5 | **SurfEmb** (arXiv:2111.13489) | Learns **continuous dense correspondence distributions** with a contrastive loss, *unsupervised* w.r.t. symmetry — captures multimodal surface distributions automatically. | An alternative to declaring symmetry by hand; learns the ambiguity. Strong but a different model family (P3P + refine, slow). Reference, not adopted now. | Strong BOP AR on symmetric/texture-less |
| R6 | **GDRNPP** paper + repo (arXiv:2102.12145, shanice-l/gdrnpp_bop2022) | Confirms: symmetry-aware PM loss uses BOP symmetry labels; **crops resized to fixed 256×256**, output 64×64 (`INPUT_RES`/`OUTPUT_RES` in `gdrn_base.py`); stronger DR + ConvNeXt is the PP delta. | Pins the small-feature resolution ceiling (256/64) and confirms our PM_LOSS_SYM wiring is the right primitive. | BOP'22/'23 winner; synth-only ~82.7 AR (depth/real-tuned regimes) |

---

## 2. Gap analysis — which failure is which kind of problem

### 2.1 Symmetry-definition problem? — **Anker: YES (add discrete 180°). Zahnrad: NO.**

Verified on the box from `models_info.json`:

- **Anker_Kurz / Anker_Lang (obj 1/2):** declared **continuous-Y only**
  (`symmetries_continuous: axis [0,1,0]`), **no `symmetries_discrete`**.
  The 13–19% ≥90° flip tail is an **end-to-end 180° ambiguity** of the stab: flipped head-to-tail
  it looks nearly identical in a single top-down view, but that flip is **not** a rotation about the
  declared Y axis, so the cont-Y group never canonicalizes it and the PM loss never consistently
  punishes it. The network sits between two near-equivalent modes and the loss is happy with either
  → bimodal output, heavy tail. **→ This is a real symmetry-DEF gap, fixable at the data level.**
  - **Action:** if the CAD is genuinely 180°-flip-near-symmetric, add a discrete C_2 about the
    long axis to the Anker's `models_info.json` so BOTH the loss and the metric forgive the flip
    (collapses the tail analytically, exactly like cont-Y collapses the twist). **OWNED BY THE
    DATA-STRANG** (`models_info.json` / `isaac_to_bop.py`) — flagged to them, not changed here.
  - **If the flip is NOT a true symmetry** (a notch/feature breaks it), then it is a *learning*
    problem and the fix is resolution + DR (Fix-1/2) + multi-hypothesis (Fix-3), NOT a sym edit.
    **This must be decided by looking at the CAD** (see §Open questions).

- **Zahnrad (obj 6):** declared **C_7** — verified correct (6 discrete entries at 51.4°/102.9°/154.3°
  = multiples of 360/7, plus identity). The metric self-test collapses a 360/7 twist to 0°.
  **→ NOT a symmetry-def problem.** The 87°-median / 1%-<5° / MSSD-0.043 is a genuine learning failure.

### 2.2 Resolution / architecture problem? — **Zahnrad: YES (primary).**

`gdrn_base.py` defaults: `INPUT_RES=256, OUTPUT_RES=64`. The Zahnrad diameter is **49.9 mm** —
less than half the Anker's ~112 mm — and its discriminative cue is the **fine C_7 tooth ring**. In a
256px crop the teeth are a few pixels each, and the region/XYZ head only emits a **64px** grid, below
the spatial frequency of the tooth pattern that the C_7 in-plane orientation depends on. The net
literally cannot see which tooth is which. SymCode (R1) is the deeper fix (one-to-many correspondences),
but the cheap, generalizable first move is **more pixels**: raise to 320/80 (→384/96 for the gear if
VRAM allows). This generalizes to any small/fine-feature part.

### 2.3 Data problem? — **All three (the tail + Sim2Real).**

- ~23% of GT is unmatched (prediction landed nowhere) — strong arm-occlusion + texture-less metal,
  i.e. a Sim2Real / data-coverage gap. T-038 already scoped <20%-visible instances out of the AR
  denominator (honest); the remaining tail is real.
- Texture-less specular metal + synth-only is exactly the regime R3 (shape-bias) and R2 (material-recon)
  target. The DR-heavy `sdg_armvis_dr5k` render (per-object roughness/metallic/tint, lights 120–2200,
  cam jitter+roll, focal 14–24, clutter 8–16) is the right data lever — it just needs to be converted
  (`--min-visib 0.20`) and wired in.

### 2.4 Translation / RGB-only? — **structural, partially addressable.**

MSSD≪MSPD says Z is the killer. GDRNPP's `TRANS_TYPE="centroid_z"` regresses Z from the crop scale; on
texture-less metal that signal is weak. RGB-only is a hard constraint (Max). Levers: more/denser DR poses
(better scale prior), higher crop res (better centroid), and — as the documented future option — a
confidence-filtered Zivid depth-refine pass (BOP industrial evidence: +10–15 AR). Depth-refine stays
**out of scope** for Phase 2 (RGB-only hard rule); noted as the biggest remaining AR lever.

---

## 3. Generalizing fixes — prioritized by impact / effort

> All fixes are pipeline-wide (apply to every part), not per-part hacks. Ordered by
> (expected AR impact) / (effort). **Top-3 are the Phase-2 must-haves.**

### ★ FIX-1 — Higher crop/output resolution (256/64 → 320/80) · config-only · **GPU-GATED**
- **What:** `MODEL.POSE_NET.INPUT_RES=320, OUTPUT_RES=80` in `config_base_so.py` (done).
- **Why:** §2.2 — restores small-feature detail; the direct lever for the Zahnrad's unlearned C_7.
- **Generalizes:** any small/fine-feature part; the Ankers also benefit at the margin.
- **Source:** R6 (the 256/64 ceiling) + R1 (correspondence resolution matters for discrete sym).
- **Expected impact:** Zahnrad MSSD/rot the big mover (the gear is currently MSSD 0.04); modest on Anker.
- **Effort:** S (config). **Risk:** VRAM/shape — **needs the 1-iter GPU smoke-test** (now wired into
  `train_chain.sh --gdrnpp-only --smoke`). If 320 OOMs at batch 16 → drop to batch 12 or 256/64.

### ★ FIX-2 — Shape-bias Sim2Real: DR-8k data + stronger texture-destroying aug · data + config
- **What:** train on the DR-heavy `sdg_armvis_dr5k` set (converted `--min-visib 0.20`) **in addition to**
  the existing train_pbr; bump `COLOR_AUG_PROB 0.8→0.9` with the existing grayscale/contrast chain (done).
- **Why:** §2.3 — texture-less metal must be learned shape-first, not texture-first; cuts the unmatched
  tail and the Sim2Real gap.
- **Generalizes:** every part, every scene. Pure data/render + on-the-fly aug, no arch change.
- **Source:** R3 (randomized-texture → +18.6% ITODD pose), R2 (PBR material DR).
- **Expected impact:** highest on the unmatched/occluded tail → recall → AR across all parts.
- **Effort:** M (data-strang renders + converts; we wire the dataset name). **Risk:** the DR-5k render
  must finish and convert cleanly; wiring A vs B (see config + §Retrain validation).

### ★ FIX-3 — Resolve the Anker 180° flip the RIGHT way (sym-def OR multi-hypothesis) · data OR arch
- **What:** **First** decide from the CAD whether the Anker flip is a true symmetry. If YES →
  add a discrete C_2 about the long axis to `models_info.json` (data-strang) — collapses the tail
  analytically in both loss and metric, *zero* extra training cost, exactly like cont-Y collapses the
  twist. If NO (a feature breaks the flip) → keep cont-Y and rely on Fix-1/2 + (later) a multi-hypothesis
  head (Implicit-PDF/EPOS, R4) that emits top-K rotations resolved at test time by the planar prior.
- **Why:** §2.1 — the flip tail is the single biggest drag on Anker mean rot (98–105° naive) and AR tail.
- **Generalizes:** the C_2-if-true-symmetry rule is the general principle (declare the real symmetry
  group, no more no less); the multi-hypothesis fallback generalizes to any residual ambiguity.
- **Source:** R4 (Implicit-PDF/EPOS), R1/R6 (declare the correct symmetry group).
- **Expected impact:** if the C_2 is valid, the Anker mean-rot tail collapses (median already fine) → AR up.
- **Effort:** S if it is a sym-def edit (data-strang); L if it needs the multi-hypothesis head (deferred).
  **Risk:** mis-declaring a C_2 that the CAD does not actually have would make a *correct* non-flipped pose
  score as a flip-equivalent — **must be confirmed against the CAD before editing models_info.json.**

### FIX-4 (deferred, documented) — Material-reconstruction / keypoint head (R2)
- Architecture addition to GDRNPP (extra decoder heads). Big-ticket but invasive; not Phase-2.
- Keep as the next arch upgrade if Fix-1/2/3 plateau below target.

### FIX-5 (deferred, hard rule) — confidence-filtered depth-refine (RGB-D ablation)
- Biggest remaining AR lever per BOP industrial evidence (+10–15 AR), but **violates the RGB-only hard
  rule** (Max). Documented as future work only.

---

## 4. Phase-2 retrain — start-ready command (DO NOT LAUNCH — GPU rendering)

### 4.1 What is already prepared (this ticket)
- `box_src/gdrnpp/config_base_so.py` — INPUT_RES 320 / OUTPUT_RES 80 (Fix-1), COLOR_AUG_PROB 0.9
  (Fix-2), FILTER_VISIB_THR 0.20 (T-038 alignment), TOTAL_EPOCHS 160, DR-dataset wiring note (A/B).
- `box_src/train_chain.sh` — `--gdrnpp-only` now also refreshes the per-object so_configs and supports
  `--smoke` (few-iter GPU probe of the INPUT_RES change; aborts the chain on shape/OOM failure).
- Per-object configs (`so_configs/*.py`) unchanged — they inherit everything from base_so.

### 4.2 Launch command (when GPU is free)
```bash
# ON the box, AFTER the sdg_armvis_dr5k render finishes AND is converted to BOP
# (data-strang: isaac_to_bop.py --min-visib 0.20, wiring A=merge or B=separate split).
cd /mnt/data/kip_pose

# STEP 0 (MANDATORY first time): validate the INPUT_RES=320 change on the GPU.
#   --gdrnpp-only re-applies patches + refreshes configs, then --smoke runs a
#   ~20-iter anker_kurz probe and ABORTS if it OOMs / shape-mismatches.
nohup bash box_src/train_chain.sh --gdrnpp-only --smoke \
    > /mnt/data/bop/logs/train_chain_phase2.log 2>&1 &
echo "CHAIN_PID=$!"

# If the smoke passes, the SAME run continues straight into the full multi-day
# per-object training (anker_kurz → anker_lang → zahnrad), sequential on the one
# 3090. Poll:
tail -40 /mnt/data/bop/logs/train_chain_phase2.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

> `--gdrnpp-only` is correct: the detector (mAP50 0.99, S-401) must NOT be retrained, and the
> OBB→AABB bridge + GDRNPP deploy artifacts already exist. The chain verifies `detector.pt`,
> `fps_points.pkl`, `keypoints_3d.pkl` before skipping stages 1–3.

### 4.3 Retrain validation — what MUST be checked when the GPU frees up
> These are the items that **cannot** be verified offline (GPU busy rendering); do them before/at launch.

1. **GPU smoke-test of INPUT_RES=320** — the `--smoke` probe must pass (no shape mismatch into
   GEO_HEAD in_dim=1024, no OOM at batch 16). If it OOMs → batch 12, or fall back to 256/64.
   **This is the #1 launch blocker.**
2. **DR-5k render finished + converted** — `data/sdg_armvis_dr5k` complete; `isaac_to_bop.py
   --min-visib 0.20` produced valid BOP scenes; decide wiring **A (merge into train_pbr)** vs
   **B (separate `_train_dr` split)** and reflect it in `DATASETS.TRAIN`. Confirm the dataset
   loads via `bop_toolkit` and instance counts look right (≈2× after merge).
3. **Symmetry-def decision for the Anker flip (Fix-3)** — inspect the CAD: is the 180° flip a true
   symmetry? If yes, the data-strang adds discrete C_2 to `models_info.json` AND regenerates
   `fps_points.pkl`/`keypoints_3d.pkl` BEFORE training. If no, leave cont-Y and rely on Fix-1/2.
   **Do not edit models_info.json blind.**
4. **fps/keypoints regenerated** if any model/symmetry changed (deploy step does this; re-run
   `pose_isaac_compute_fps.py` + `pose_isaac_compute_keypoints_3d.py` if models_info.json changed).
5. **Epoch budget vs GPU calendar** — 160 epochs × 3 objects on one 3090 is multi-day; confirm the
   render is done so the two jobs never contend for the GPU.
6. **Post-train eval** — re-run `box_src/eval_bop.py` (filtered + unfiltered) and diff against the
   §0 table; the targets are: Zahnrad MSSD ≫ 0.04 and median rot ≪ 87°; Anker rot_naive_mean ≪ 98°
   (flip tail shrinks); overall AR up from 0.19.

---

## 5. Risks & open questions

- **R-1 (launch blocker):** INPUT_RES=320 may OOM or shape-mismatch at batch 16 → mitigated by the
  `--smoke` gate + batch-12 fallback. Verifiable only on the GPU.
- **R-2:** DR-5k convert/wiring is owned by the data-strang; if it slips, train on train_pbr alone
  (Fix-2 partially) — still gets Fix-1 + Fix-3.
- **R-3 (correctness):** adding a discrete C_2 to the Anker is only valid if the flip is a true CAD
  symmetry. Wrong call → a correct pose scores as flip-equivalent. **Decide from the CAD first.**
- **R-4:** raising INPUT_RES + dataset size + epochs together lengthens each per-object run; on one
  3090 the three are strictly sequential — plan the calendar.
- **Open Q-1:** Is the Anker genuinely 180°-flip symmetric, or is there a symmetry-breaking feature?
  (Determines Fix-3 path: sym-def edit vs multi-hypothesis.)
- **Open Q-2:** 320/80 enough for the gear, or push the Zahnrad SO config to 384/96 specifically?
  (Per-part *resolution* is still generalizable — it scales with object size, not a hack — but try
  the uniform 320/80 first and only bump the gear if it is still short.)

---

## Related
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md) · [`REEVAL_T038_visib20.md`](REEVAL_T038_visib20.md) ·
  `eval_gdrnpp_val.json` · ADR-018 (BOP pivot) · ADR-017 (pose_result contract)

## Sources (web research)
- SymCode/SymNet — Resolving Symmetry Ambiguity in Correspondence-based Methods: https://arxiv.org/html/2405.10557v1
- Improving 6D Pose of metallic Household/Industry Objects (GDRNPP-based): https://arxiv.org/html/2503.03655
- Shape-biased Texture-Agnostic Representations: https://arxiv.org/html/2402.04878
- Implicit-PDF (SO(3) distributions): https://arxiv.org/abs/2106.05965
- EPOS (estimating 6D pose of objects with symmetries): https://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf
- SurfEmb (dense continuous correspondence distributions): https://arxiv.org/abs/2111.13489
- GDRNPP paper: https://arxiv.org/abs/2102.12145 · repo: https://github.com/shanice-l/gdrnpp_bop2022
