# POSE — Architecture

> 6D-pose estimation for metal assembly parts from a single top-down RGB image,
> rendered back as real CAD in a 3D viewer.
> Built on the BOP-benchmark SOTA stack (ADR-018). **RGB-only.**

This document describes the *as-built* pipeline: the stages, the data flow, why
each method was chosen, the file/module layout, and the `pose_result` output
contract. For the day-to-day workflows see [`../README.md`](../README.md); to add
a new part see [`ADD_NEW_PART.md`](ADD_NEW_PART.md); for evaluation see
[`EVAL.md`](EVAL.md); for citations [`REFERENCES.md`](REFERENCES.md).

---

## 1. Problem

An overhead camera looks down on a tray inside a robot assembly cell. The
**LARA5 robot arm is visible in frame** and randomly-dropped metal parts lie on
the tray (this visible-arm, cluttered top-down view is the actual task
specification — the arm is *not* hidden). For each detected part we must recover
its full **6D pose** (3D rotation + 3D translation) and place the true CAD mesh
at that pose in a browser 3D viewer.

The parts are **texture-less, shiny metal** (anchors, gears, ring magnets) — the
hard regime for pose estimation: little surface texture, specular highlights,
and strong rotational symmetry that makes a single view ambiguous.

A previous in-house pose core (face-atlas + template-bank render-and-compare)
scored **120° / 91° median rotation error, 186 mm translation** on the real
parts — unusable. ADR-018 replaced that middle stage with the BOP SOTA stack.

---

## 2. Pipeline at a glance

```
                         ┌──────────────────────── TRAINING (GPU box) ───────────────────────────┐

  CAD (GLB, metres)  ──gen_models_info.py──▶  models/obj_*.ply (mm)  +  models_info.json (symmetry)
                                                                          │
  Isaac GST cell  ──gen_sdg_arm_visible.py──▶  top-down RGB  WITH  LARA5 arm  +  dropped parts
  (USD scene)                                  + depth + instance-seg + gt_raw (intrinsics, c2w, poses)
                                                                          │
                          ──isaac_to_bop.py / convert_full_to_bop.py──▶  BOP dataset  (train_pbr / val)
                                                  scene_camera.json · scene_gt.json · scene_gt_info.json
                                                  rgb/ · depth/ · mask/ · mask_visib/
                                                                          │
                          ──train_detector_armvis.py──▶  detector.pt  (YOLOv8-OBB, arm-visible)
                          ──obb_to_aabb_dets.py──▶  BOP detections JSON  (OBB→AABB bridge)
                          ──train_chain.sh (GDRNPP)──▶  per-object GDRNPP checkpoints (RGB-only)

                         └────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────── INFERENCE (laptop) ────────────────────────────┐

   image (input/)  ──detector (YOLOv8-OBB)──▶  oriented boxes  ──OBB→AABB──▶  per-detection crops
                   ──GDRNPP (RGB-only)──▶  (R_m2c, t_m2c)  [BOP convention, mm, camera frame]
                   ──bop_adapter.py §3──▶  world pose  (R_world, t_world, face, upright)
                   ──▶  pose_result.json  (frozen contract, schema-validated)
                   ──▶  Three.js viewer  (frontend/, renders real CAD at the pose)

                         └────────────────────────────────────────────────────────────────────────┘
```

```mermaid
flowchart TD
    CAD["CAD GLB (metres)"] -->|gen_models_info.py| MODELS["models/obj_*.ply (mm)<br/>models_info.json (symmetry)"]
    ISAAC["Isaac GST cell (USD)"] -->|gen_sdg_arm_visible.py| RAW["arm-visible top-down RGB<br/>+ depth + instance-seg + gt_raw"]
    RAW -->|isaac_to_bop.py| BOP["BOP dataset<br/>train_pbr / val"]
    MODELS --> BOP
    RAW -->|train_detector_armvis.py| DET["detector.pt (YOLOv8-OBB)"]
    BOP -->|train_chain.sh| GDRN["GDRNPP checkpoints (RGB-only)"]
    DET --> GDRN

    IMG["scene image (input/)"] --> DETECT["detector → OBB → AABB crops"]
    DETECT -->|GDRNPP RGB-only| RT["(R_m2c, t_m2c) BOP, mm, cam frame"]
    RT -->|bop_adapter.py §3| WORLD["world pose (R_world, t_world, face, upright)"]
    WORLD --> PR["pose_result.json (frozen contract)"]
    PR --> VIEW["Three.js viewer (real CAD)"]
    GDRN -.checkpoint.-> RT
    DET -.detector.pt.-> DETECT
```

---

## 3. Stages

### Stage A — CAD → BOP models
`box_src/gen_models_info.py` exports each part's GLB mesh to
`models/obj_<id>.ply` **in millimetres** (GLB is metres → ×1000) and writes
`models/models_info.json` with `diameter`, bounding box, and **symmetry flags**
per part. The symmetry flags are the analytic fix for the rotational-ambiguity
problem (see §6). This is fast and runs in the BOP venv (trimesh + numpy).

### Stage B — Synthetic data generation (Isaac SDG)
`box_src/gen_sdg_arm_visible.py` renders top-down PBR frames of the Isaac GST
cell **with the LARA5 arm visible** and 7–13 parts physically dropped onto the
tray (domain-randomised lighting/material/camera/background). Each frame yields
RGB, metric depth, a per-pixel instance map, and `gt_raw` (intrinsics,
camera-to-world, per-instance object-to-world transforms). Runs in
`isaacsim-venv`.

### Stage C — Isaac → BOP conversion
`box_src/isaac_to_bop.py` (single scene) and `box_src/convert_full_to_bop.py`
(full multi-scene + train/val split) turn the raw Isaac bundle into a
spec-compliant BOP dataset. This is **the central data contract** — once the
output matches the BOP layout, CNOS, GigaPose, MegaPose, GDRNPP and
`bop_toolkit` all read it without a single line of repo patching. Correctness
hot-spots, all handled in the converter:

- **w2c, not c2w** — Isaac stores camera-*to*-world; the converter inverts to
  world-to-camera and flips the OpenGL→OpenCV axis convention (Y, Z).
- **millimetres** — all translations ×1000; the `OBJECT_SCALE 1e-3` baked into
  the object matrices is stripped via column-norm + SVD orthonormalisation to
  recover a pure rotation.
- **row-major** — every 3×3 / 4×4 stored row-major flat (BOP/numpy native).
- **instance match by prim path** — `idToLabels` is matched on the exact prim
  path token (`SpawnedObject_NNN`), avoiding same-class ambiguity.
- **the arm is an occluder, never an instance** — it has no semantic label, so
  it never becomes an `obj_id`; it only lowers `visib_fract` of the parts
  behind it. This is the Sim2Real occlusion lever.

### Stage D — Detector retrain (arm-visible)
`box_src/train_detector_armvis.py` retrains the YOLOv8-OBB detector on the
arm-visible data. The detector emits **oriented** bounding boxes (cheap in-plane
orientation prior + viewer overlay). `box_src/obb_to_aabb_dets.py` converts the
OBB output to the axis-aligned `[x, y, w, h]` boxes the BOP pose stages consume.
The old `detector.pt` was arm-*hidden*; under a visible arm it degrades on
occlusion, hence the retrain.

### Stage E — GDRNPP training
`box_src/train_chain.sh` chains, on the single RTX 3090: detector retrain →
OBB→AABB val detections → GDRNPP deploy → per-object GDRNPP training for the
focus parts (`anker_kurz`, `anker_lang`, `zahnrad` first). Per-object training
keeps VRAM small and accuracy high. Long jobs run under `nohup` and are polled,
never waited on inline.

### Stage F — Inference
`project/e2e_infer.py` is the self-contained inference pipeline (one image →
`pose_result.json`):
1. **Detect** — `models/detector.pt` (YOLOv8-OBB via ultralytics). Fallbacks, in
   order: SDG annotator boxes (`bbox_2d_*.json`) → a whole-image dummy box, so
   the chain never hard-breaks.
2. **Crop** — axis-aligned crop of the OBB hull per detection.
3. **6D pose** — `call_gdrnpp()` per crop returns `(R_m2c, t_m2c)` in BOP
   convention (mm, camera frame). Until a GDRNPP checkpoint exists it runs a
   **deterministic MOCK** that emits plausible on-table poses, so the whole
   chain — viewer included — is green today. Drop in a checkpoint
   (`--checkpoint …`) and the real call (`_gdrnpp_real`, the single hook point)
   takes over.
4. **Map + emit** — `bop_adapter.py` maps the camera-frame pose to the world
   contract; the result is schema-validated before it is written.

### Stage G — Viewer
`project/frontend/` is a Three.js viewer. It loads `cell.glb` (the real cell)
and `assets/parts/*.glb` (the real CAD meshes) and places each part at its
`R_world` / `t_world` from `pose_result.json`. The viewer is **decoupled** from
the pose method via the frozen contract (ADR-017): swapping GDRNPP for any other
BOP estimator changes nothing downstream.

---

## 4. Why these methods

| Choice | Reason |
|---|---|
| **GDRNPP** (primary, Gleis B) | BOP'22 winning line; geometry-guided direct regression; best accuracy on ITODD-like texture-less metal; **synth-only reaches ~82.7 AR** (Sim2Real practically solved with good PBR randomisation); RGB-capable; trains + infers on a single RTX 3090. |
| **CNOS → GigaPose → MegaPose** (Gleis A) | Zero-shot "new CAD in, no training" baseline + immediate result while GDRNPP trains for days. Fully RGB, permissively licensed (MIT/Apache), 3090-friendly (run stages sequentially, free VRAM between them). |
| **RGB-only** | Depth on shiny metal is exactly where commodity/Zivid sensors add reflection noise; depth-mandatory methods (FoundationPose, SAM-6D, FreeZeV2) degrade there. RGB sidesteps the worst sensor failure mode. An RGB-vs-RGB-D ablation with confidence-filtered Zivid depth is the recommended follow-up experiment, but RGB stays the default. |
| **BOP format + metric** | The lingua franca of 6D pose. Conforming to it unlocks *every* BOP repo and the official, symmetry-aware evaluation (`bop_toolkit`) with zero custom glue. |
| **YOLOv8-OBB detector** | Oriented boxes are strictly more informative than AABB (in-plane orientation prior + viewer overlay); the AABB the pose stages need is a trivial adapter, not a separate model. |

**Rejected:** FoundationPose (NVIDIA non-commercial + depth-mandatory),
SAM-6D (depth-mandatory), FreeZeV2 (depth-dependent + 17–25 s/image). See ADR-018.

---

## 5. File & module map

### `project/` (laptop — inference + viewer)
| Path | Role |
|---|---|
| `e2e_infer.py` | Self-contained pipeline: image → `pose_result.json`; `--serve` opens the viewer. GDRNPP MOCK fallback until a checkpoint exists. |
| `bop_adapter.py` | BOP `(R_m2c, t_m2c)` (cam, mm) → `pose_result` (world, m). Symmetry canonicalisation + `face`/`upright`. Single-source `OBJ_ID_TO_PART`. Tested. |
| `pose_result.schema.json` | Frozen output contract (ADR-017). |
| `setup.ipynb` | "Reproduce training" — drives `box_src/` scripts on the GPU box; no logic duplicated. |
| `infer.ipynb` | "Inference + 3D viewer" — imports `e2e_infer` / `bop_adapter`. |
| `frontend/` | Three.js viewer + `assets/cell.glb` + `assets/parts/*.glb` + `assets/part_meta.json`. |
| `cad_input/` | CAD intake (parts + cell scene). |
| `input/` | Input scenes (RGB + optional `bbox_2d_*.json` / `scene_camera.json`). |
| `tests/` | `test_bop_adapter.py` + `test_e2e_mock.py` (21 tests). |

### `box_src/` (GPU box — data + training + eval)
| Path | Role |
|---|---|
| `gen_models_info.py` | GLB → `models/obj_*.ply` (mm) + `models_info.json` (symmetry). |
| `gen_sdg_arm_visible.py` | Isaac SDG: arm-visible top-down PBR frames. |
| `isaac_to_bop.py` | Single-scene Isaac → BOP converter (the data contract). |
| `convert_full_to_bop.py` | Full multi-scene convert + train/val split. |
| `validate_bop_full.py` | BOP-format validation gate. |
| `vis_bop_overlay.py` | GT-pose overlay (visual acceptance gate, ADR §1.6). |
| `train_detector_armvis.py` | YOLOv8-OBB detector retrain (arm-visible). |
| `obb_to_aabb_dets.py` | OBB → AABB BOP-detection bridge. |
| `train_chain.sh` | Sequential single-GPU training chain (detector → GDRNPP). |
| `gdrnpp/` | GDRNPP env prep + `pose_isaac` deploy scripts + configs. |
| `eval_bop.py` / `eval_bop.sh` | BOP-metric evaluation harness + laptop wrapper. |
| `gpu_run.sh` | SSH/Wake-on-LAN run harness for the box. |
| `BOP_SETUP.md` / `EVAL_BOP.md` / `README_BOP_DATA.md` | Box setup, eval, data notes. |

---

## 6. Frozen conventions

- **World frame:** Z-up; `world = R @ body` (column convention); origin = table
  null-point; unit = metres. **No transpose at the boundary.**
- **`obj_id`** (1-based, single-source `bop_adapter.OBJ_ID_TO_PART`):
  `1=Anker_Kurz 2=Anker_Lang 3=Buerstenhalter_2polig 4=Getriebegehaeuse_typ4
  5=Ringmagnet 6=Zahnrad`. Detector class (0-based) **+1** = `obj_id`.
  (`Poltopf_kurz_centered` is intentionally excluded — its mesh export is
  degenerate ~50 µm; add it as `obj_id 7` once a valid mesh exists, never
  renumber the existing IDs.)
- **Symmetry** (the analytic fix for the 120°/91° punishment): Anker_Kurz,
  Anker_Lang and Ringmagnet are **continuous** about the model Y axis; the
  Zahnrad is **discrete C_7** (tooth count measured from the mesh — outer rim is
  a smooth disc, the periodic feature is the inner 7-spline hub). Buerstenhalter
  and Getriebegehaeuse have no symmetry. The flags live in `models_info.json`;
  `bop_toolkit` maps each pose to the nearest symmetric representative *before*
  scoring it.

---

## 7. The `pose_result` contract (ADR-017, frozen)

One file per processed image. The producer (`e2e_infer.py` via `bop_adapter.py`)
and the consumer (the Three.js viewer) agree on exactly this shape — nothing else
crosses the boundary.

```json
{
  "meta": {
    "source_image": "input/scene_0000.png",
    "table_origin": [0.0, 0.0, 0.08],
    "units": "m",
    "coordinate_convention": "Z-up world; column rotation world = R @ body; origin = table-plane null-point",
    "schema_version": "1.0.0"
  },
  "results": [
    {
      "instance_id": 0,
      "part": "Zahnrad",
      "face": "face_y-",
      "R_world": [r11, r12, r13, r21, r22, r23, r31, r32, r33],
      "t_world": [x, y, z],
      "confidence": 0.91,
      "bbox_2d": [x0, y0, x1, y1],
      "upright": true
    }
  ]
}
```

- `R_world` — row-major flat 9, `world = R @ body`.
- `t_world` — metres, relative to `table_origin`.
- `bbox_2d` — `[x0, y0, x1, y1]` (the contract uses corner form; BOP detection
  JSON uses `[x, y, w, h]` — `bop_adapter` does the conversion).
- `face` / `upright` — derived from the canonicalised world rotation (the
  ex-face-classifier is gone; these are computed analytically in `bop_adapter`).

The full JSON Schema is `project/pose_result.schema.json`. `e2e_infer.py`
validates every output against it (pure-stdlib gate always; `jsonschema` as a
bonus when installed) and refuses to write a non-conforming file.

---

## 8. Two-track design (ADR-018)

The middle stage is two interchangeable tracks behind the same adapter:

- **Gleis B (primary, max accuracy):** GDRNPP, per-object, trained on Isaac PBR
  synth. Multi-day GPU job → trains in the background.
- **Gleis A (generalisation + instant baseline):** CNOS → GigaPose → MegaPose,
  zero-shot, RGB-only. No training; works the moment a new CAD exists.

Both tracks emit BOP-convention `(R_m2c, t_m2c)` and go through the **same**
`bop_adapter` function, so they produce **identical** `pose_result` files. That
is what keeps the viewer and the evaluation method-agnostic.

---

## Related
- ADR-018 — POSE: pivot to BOP SOTA stack (the architecture decision).
- ADR-017 — `pose_result` contract (the output boundary).
- [`ADD_NEW_PART.md`](ADD_NEW_PART.md) · [`EVAL.md`](EVAL.md) ·
  [`REFERENCES.md`](REFERENCES.md) · [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
