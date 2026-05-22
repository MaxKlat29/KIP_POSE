# BOP Data Foundation — Isaac SDG → BOP for GDRNPP (S-201/202/203)

Pipeline that turns the Isaac GST cell into a BOP dataset (`pose_isaac`) for
GDRNPP/CNOS/GigaPose/MegaPose. Implements Viktor `adr.md` §1 (layout), §2
(symmetry). Runs on the GPU box; Isaac in `isaacsim-venv`, the rest in `bop-venv`.

## obj_id mapping (frozen, adr.md §1.2 = detector class + 1)
1 Anker_Kurz · 2 Anker_Lang · 3 Buerstenhalter_2polig · 4 Getriebegehaeuse_typ4
· 5 Ringmagnet · 6 Zahnrad   (Poltopf excluded — degenerate mesh, R6)

## Steps

```bash
BOX=/mnt/data/kip_pose ; BOP=$BOX/project/bop/pose_isaac
USD=$BOX/data/SDG/IsaacSim/USD-Files

# 1) models/ PLY (mm) + models_info.json (symmetries)   [bop-venv, fast]
/mnt/data/bop/bop-venv/bin/python box_src/gen_models_info.py --bop-root $BOP --also-eval

# 2) arm-VISIBLE top-down SDG (LARA5 visible = occluder)  [isaacsim-venv]
/mnt/data/isaacsim-venv/bin/python box_src/gen_sdg_arm_visible.py \
  --scene $USD/GST_Scene.usd --usd-dir $USD --output $BOX/data/sdg_armvis \
  --num-scenes 2000 --min-obj 7 --max-obj 13 --settle 180 --focus-frac 0.6 \
  --spawn-x 0.18,0.52 --spawn-y 0.08,0.50

# 3) Isaac → BOP (scene_camera/gt/gt_info + mask + mask_visib)  [bop-venv]
/mnt/data/bop/bop-venv/bin/python box_src/isaac_to_bop.py \
  --raw-dir $BOX/data/sdg_armvis --bop-root $BOP --split train_pbr --scene-id 0

# 4) visual GT-pose belief (acceptance gate, adr.md §1.6)
/mnt/data/bop/bop-venv/bin/python box_src/vis_bop_overlay.py \
  --bop-root $BOP --split train_pbr --scene-id 0 --out $BOX/project/temp/bop_check
```

## Correctness notes (validated)
- **mm / row-major / w2c**: gt_raw stores `cam_c2w` (USD); converter inverts to
  w2c AND flips OpenGL→OpenCV (Y,Z). `cam_t` in mm.
- **OBJECT_SCALE 1e-3** baked into `obj2world` 3×3 → stripped via column-norm +
  SVD-orthonormalise to recover pure rotation. PLY is mm (= GLB metres ×1000).
- **instance-seg idToLabels = PRIM PATH** (not class) → exact `SpawnedObject_NNN`
  token match (no same-class ambiguity).
- **Zahnrad symmetry = C_7** (tooth count from rasterised inner 7-spline hub;
  outer rim is a smooth disc). Anker/Ring = continuous about Y.
- **Arm = occluder**: no semantic label → never an obj_id; lowers `visib_fract`.

Validated: BOP format check ALL PASS, `bop_toolkit` native loaders read it
unmodified, `get_symmetry_transformations` yields C_7/continuous/none, Zahnrad GT
origin projects 2.4px from its mask centroid.
