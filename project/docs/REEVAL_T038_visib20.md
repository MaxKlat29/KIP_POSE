# T-038 — >20% Visibility Pipeline Filter + Re-Eval (Accuracy-Push Phase 1)

**Date:** 2026-05-24 · **Worktree:** `.worktrees/S-038` · **No retrain** (same checkpoints, same predictions).

## The filter (Max rule, uniform THR = 0.20)

A single uniform visibility gate, replacing the old split (detector ~0.15 / GDRNPP 0.30 / eval 0.0).
Instances with `visib_fract <= 0.20` are cut from the pipeline: `scene_gt.json`, `scene_gt_info.json`,
the per-instance masks, and the detector training labels. The occlusion stays in the RGB — only the
label / target / eval ignores <=20% instances (a part <=20% visible is fairly not pose-able).

Implemented as a post-hoc idempotent filter `box_src/filter_visibility.py` (snapshots `.full`, always
re-derives from it) plus a convert-time passthrough `isaac_to_bop.py --min-visib` /
`convert_full_to_bop.py --min-visib` and detector `train_detector_armvis.py --max-occ 0.80`.

### How many instances dropped (visib_fract <= 0.20)

| split | total | dropped | % dropped | kept |
|---|---|---|---|---|
| train_pbr | 18083 | 8224 | **45.5%** | 9859 |
| val | 1964 | 887 | **45.2%** | 1077 |

Of the dropped, the lion's share is `visib < 0.05` (41.6% train / 41.2% val = quasi-invisible).
Detector-label audit (SDG `obb_2d` occlusion, computed differently from BOP full-mask visib):
739/16383 boxes = 4.5% would-drop at occ>=0.80.

## Re-eval (val, same GDRNPP predictions, only the GT denominator changed)

Baseline (unfiltered) eval reproduced exactly the old W4 numbers (0.42/0.45/0.28), confirming
the harness is sound and the comparison is apples-to-apples.

| obj | AR old -> filtered | trans-med old -> filt (mm) | rot-med old -> filt (deg, sym) | n_gt old -> filt | n_matched |
|---|---|---|---|---|---|
| 1 Anker_Kurz | 0.42 -> **0.59** (+0.17) | 36.6 -> 33.1 | 8.1 -> 6.4 | 390 -> 246 | 246 |
| 2 Anker_Lang | 0.45 -> **0.61** (+0.16) | 39.8 -> 38.0 | 5.4 -> 4.9 | 397 -> 273 | 273 |
| 6 Zahnrad | 0.28 -> **0.36** (+0.08) | 27.2 -> 27.2 | 87.1 -> 90.9 (failed) | 392 -> 277 | 277 |

**Why the jump:** `n_matched` is unchanged (246/273/277) — the same predictions match the same poses.
The ~45% quasi-invisible GT that used to count as misses in the AR denominator are removed, so recall
rises. The medians also improve slightly (the very-low-visib instances that *were* predicted tended to
be worse poses). The jump is honest scoping, not a model change.

**Zahnrad rotation stays broken** (90.9deg median, ~48% flips): the C_7 symmetry does not forgive a
wrong tooth alignment. The eval self-test [c] (C_7 51.4deg->0deg) confirms this is a real model
convergence failure, not a metric artifact. -> Phase-2 target (flip-aware loss / DR-heavy retrain).

## Phase-2 prep

DR-heavy SDG render kicked off (5000 scenes, `--dr-strong`): per-object material roughness/metallic/tint,
lights 120-2200, camera jitter +-0.08 + roll +-12deg, focal 14-24, clutter 8-16 parts. PID 374952,
log `/mnt/data/bop/logs/sdg_dr5k.log`, output `data/sdg_armvis_dr5k`. Convert it with `--min-visib 0.20`.

## Reproduce

```bash
# on the box, bop-venv
python box_src/filter_visibility.py --bop-root <bop> --splits train_pbr,val --sdg-dir <sdg> --thr 0.20
python box_src/combine_preds.py /tmp/preds.csv <anker_kurz csv> <anker_lang csv> <zahnrad csv>
python box_src/eval_bop.py --dataset-dir <bop> --split val --preds /tmp/preds.csv --n-points 2000 --out <out>
# restore unfiltered GT: python box_src/filter_visibility.py --bop-root <bop> --restore
```
