# SCFlow2 × FoundationPose — integration spike (Phase 0–1)

Exploratory work toward using **SCFlow2** as a pose *refiner* on top of **FoundationPose**
for the KIP_POSE metal-parts cell. This folder documents what was built, what we found,
and what's blocking a fair accuracy measurement. **Status: paused after a negative
forward-pass result on clean-sim data (see Results).**

## What SCFlow2 is

> **SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow** (CVPR 2025)
> Qingyuan Wang, Rui Song, Jiaojiao Li, Kerui Cheng, David Ferstl, Yinlin Hu

- Paper: <https://arxiv.org/abs/2504.09160> (arXiv:2504.09160)
- Code: <https://github.com/W-QY/SCFlow2>
- Project page: <https://scflow2.github.io/>
- Pretrained weights (Google Drive): `1tUBKuc1TOam8lPJBxlvVf5AHf4-IByqZ` → `scflow2_pretrained.pth` (719 MB)

SCFlow2 is a **generalizable, mesh-based, RGB-D** 6D-pose refiner. Given an RGB-D image,
a target **3D mesh**, and an **initial pose** from any estimator, it renders a synthetic
reference, builds a 4D correlation volume (RGB + depth encoders), and iteratively predicts
a shape-constrained 3D scene flow → pose update. It generalizes to novel objects **without
retraining** (so retraining on our parts is not the intended path; training code is also not
fully released upstream).

## Why this matters for KIP_POSE

Our production pipeline (GDRNPP, RGB-only) discards depth even though the cell has a depth
sensor. FoundationPose (RGB-D, zero-shot) is a non-production combo. The idea: chain
`FoundationPose → SCFlow2` as a depth-aware refiner and measure whether it improves
grasp-relevant pose accuracy. Decisive metric must be **per-axis mm/deg error + failure rate
at grasp tolerance + tail**, not aggregate AR.

## What was done

**Phase 0 — environment + model (✅ complete).** Built an isolated env on the local
RTX 5000 Ada (sm_89, CUDA 12.1, torch 2.1.0+cu121) in `scflow2/.venv` and validated the
model: `scflow2_pretrained.pth` loads into the 101.4M-param `SCFlow2Refiner` with **0 missing
/ 0 unexpected** keys and runs on GPU. Every feared build blocker resolved:

| dependency | how it resolved |
|---|---|
| `mmcv-full==1.7.2` | prebuilt wheel exists for cu121/torch2.1 |
| `pytorch3d` | prebuilt wheel `py310_cu121_pyt210` |
| `pointnet2_ops` | source build; **must patch its hardcoded arch list to include sm_89** |
| `lietorch` | source build (`--recursive` for eigen), `--no-build-isolation` |
| `bop_toolkit` | non-editable install (old backend lacks PEP660) |
| `scikit-sparse` / CHOLMOD | no system SuiteSparse → **scipy/SuperLU shim** (`sksparse_cholmod_stub.py`) |

**Phase 1 — forward pass on our data (✅ runs, ❌ negative result).** Converted our 5
`Anker_Kurz` frames (`data/fp_input` rgb/depth/mask + `data/fp_output` FoundationPose init
poses + `data/meshes`) into a no-GT `RefineTestDataset` layout (mm units; crop by the FP-pose
bbox) and ran SCFlow2. It refined all 5 frames, but **moved away from FoundationPose's
already-good poses on every frame.**

## Results

GT-free quality check (does the refined pose fit the observed mask/depth better than FP?):

| frame | mask inlier % (FP → SCFlow2) | depth fit mm (FP → SCFlow2) |
|------:|:---------------------------:|:---------------------------:|
| 0 | 56 → 33 | 3.9 → 7.4 |
| 1 | 60 → 30 | 4.3 → 9.8 |
| 3 | 99 → 95 | 4.2 → 14.6 |
| 4 | 60 → 30 | 4.7 → 8.9 |

Refined was better in **0/5** frames. **This is not a data-prep bug** — the mesh is correctly
AABB-centered and FoundationPose's init already fits the depth to **~4 mm** (mask 56–99%), so
units/mesh/projection are sound. The degradation is SCFlow2's own refinement.

## Roadblocks & concerns

1. **No headroom on clean sim depth.** FP already fits the clean Isaac depth to ~4 mm; a
   depth-driven refiner has essentially nothing to correct and can only add noise. Clean sim
   *flatters* FP — the wrong substrate to judge a refiner.
2. **Textureless CAD starves SCFlow2's RGB branch.** SCFlow2 correlates a *rendered reference*
   against the observed RGB. Our CAD has no texture, so we render a **flat-gray** reference —
   meaningless against a *metallic* real/sim image. Specular metal appearance is also
   view/lighting-dependent, so a faithful reference render is genuinely hard.
3. **Tiny objects (~182 px).** Heavy upsampling to 256² makes the optical/scene flow
   unreliable.
4. **Symmetry.** `Anker` is continuous-symmetric about its long (body-Y) axis; part of the
   refiner's rotation change is harmless symmetry rotation (axis 0.6–0.87 aligned with Y), but
   genuine perpendicular error is also introduced.
5. **No trustworthy ground truth for scoring.** Our GT is Isaac **world-frame**
   (`data/output/anker_random/pose_*.json` + a single shared `T_cam_world.txt`). Converting to
   BOP camera frame is the conversion a prior FoundationPose eval got stuck on (~130° rotation
   error). Proper ADD/mm/deg scoring needs a clean BOP eval set (the `pose_isaac` set used by
   `box_src/eval_bop.sh`) rather than re-deriving that conversion.
6. **Env coupling.** SCFlow2 (mmcv 1.7.2 / torch 2.1) cannot share FoundationPose's container —
   a live integration needs its own service/container.

## Next steps (decided direction: real-metal data + textured meshes)

1. **Re-test on real specular-metal depth** where FP has real error (headroom), with a
   **textured / material-bearing mesh** so SCFlow2's RGB branch has something valid to match
   (or run a depth-only ablation to isolate the geometric contribution).
2. **Source trustworthy GT** via the BOP `pose_isaac` eval set; then score FP vs FP+SCFlow2
   with symmetry-aware ADD + per-axis mm/deg + grasp-tolerance failure rate.
3. If a real gain is shown: **Phase 2** — wrap `scflow2_refiner` as its own `scflow2-svc` mesh
   container, chain `fp-svc → scflow2-svc` in the gateway, and keep the pipeline order
   `FP → SCFlow2 → bop_adapter (cam→world) → Z-snap + canonicalize` (Z-snap LAST).

## Reproduce

The working tree lives in the gitignored `scflow2/` clone (upstream repo + `.venv` +
`_deps` + `checkpoints` ≈ 7 GB). The scripts here are the KIP-specific pieces that were placed
*inside* that clone and run from it.

```bash
# 1. clone upstream + build env (see scripts/_phase0_*.sh for the exact, working sequence)
git clone https://github.com/W-QY/SCFlow2 scflow2 && cd scflow2
python3 -m venv .venv
bash _phase0_build.sh        # torch2.1 + numpy<2 + mmcv-full 1.7.2
bash _phase0_compile.sh      # pointnet2 (arch-patched) + lietorch + pytorch3d
bash _phase0_bop_imports.sh  # bop_toolkit + albumentations/imgaug/pycocotools
bash _phase0_cleanup.sh      # pin numpy/opencv + install sksparse_cholmod_stub.py as sksparse/cholmod.py
# download scflow2_pretrained.pth into checkpoints/scflow2_files/

# 2. KIP forward pass (no GT): place scripts + config into the clone, then
.venv/bin/python _kip_prep.py            # data/fp_input + fp_output + meshes -> data/kip (mm)
.venv/bin/python _kip_infer.py           # refine; dumps results/kip_refined + init->refined deltas
.venv/bin/python _kip_geomcheck.py       # GT-free mask/depth fit: FP vs SCFlow2
.venv/bin/python _kip_diag.py            # mesh centering, mask size, rotation-axis vs symmetry
```

## File manifest (`scripts/`)

| file | purpose |
|---|---|
| `_phase0_build.sh` `_phase0_compile.sh` `_phase0_bop_imports.sh` `_phase0_cleanup.sh` | reproducible env build (the working sequence) |
| `sksparse_cholmod_stub.py` | scipy/SuperLU drop-in for `sksparse.cholmod` (install as `sksparse/cholmod.py`) |
| `scflow2_kip.py` | KIP config — `RefineTestDataset`, ref-pose-bbox crop, no GT (place at `configs/flow_refine/`) |
| `_kip_prep.py` | converts our frames → SCFlow2 layout (mm units, FP init poses, BOP-named PLY + models_info) |
| `_kip_infer.py` | no-GT runner (`single_gpu_test`), mmcv×torch2.1 `_get_stream` patch; dumps refined poses |
| `_kip_geomcheck.py` | GT-free quality signal (mask-inlier % + depth MAE, FP vs SCFlow2) |
| `_kip_diag.py` | degradation diagnostics (centering, mask size, symmetry-axis decomposition) |

See also project memory: `scflow2-env-build`, `scflow2-integration-plan`.
