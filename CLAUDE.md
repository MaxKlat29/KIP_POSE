# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project does

KIP_POSE estimates the 6D pose (3D rotation + 3D translation) of metal assembly parts from a single top-down RGB image of a robot cell. The production pipeline is:

```
YOLOv8-OBB detector → GDRNPP (per-object RGB-only CNN) → BOP→World transform → planar Z-snap → pose_result.json → Three.js viewer
```

Live at `https://max-utils.com/KIP/`. Runs on a GPU workstation (`max@100.85.216.95`, RTX 3090, Tailscale). Final AR IC-BIN: Anker_Kurz 0.871 · Anker_Lang 0.907 · Zahnrad 0.838 (mean 0.872).

---

## Commands

### Tests (run locally, no GPU needed)
```bash
cd project && python3 -m pytest tests/ -q
# expected: 106 passed, 6 skipped
# to run a single test file:
python3 -m pytest tests/test_bop_adapter.py -q
```

### Standalone inference (local, uses live webservice)
```bash
cd project
python3 -m venv .posevenv && source .posevenv/bin/activate
pip install requests Pillow numpy

# via CLI
python e2e_infer.py --image input/scene_0000.png --out out.json
python e2e_infer.py --image input/scene_0000.png --serve   # opens 3D viewer

# via webservice
JOB=$(curl -sF "image=@scene.png" https://max-utils.com/KIP/api/real/infer_async | jq -r .job)
curl -s "https://max-utils.com/KIP/api/real/job/$JOB"      # poll until pct=100
curl -s "https://max-utils.com/KIP/api/real/result/$JOB"   # pose_result.json
```

### Frontend dev server (static, no GPU needed)
```bash
cd project/frontend
python3 -m http.server 8000
# open http://localhost:8000/kip.html
```

### Deploy to workstation
```bash
# Backend
scp project/kip_server.py max@100.85.216.95:/mnt/data/kip_pose/project/
ssh max@100.85.216.95 'sudo systemctl restart kip-server.service'

# Worker (reloads all 3 GDRNPP checkpoints, ~4 min warm-load)
scp box_src/kip_infer_worker.py max@100.85.216.95:/mnt/data/kip_pose/box_src/
ssh max@100.85.216.95 'sudo systemctl restart kip-worker.service'
# watch: journalctl -u kip-worker -f

# Frontend (Cache-Control no-store → no CDN purge needed)
scp -r project/frontend/{kip.html,src,assets} max@100.85.216.95:/mnt/data/kip_pose/project/frontend/
```

### Service status on workstation
```bash
ssh max@100.85.216.95
sudo systemctl status kip-server.service kip-worker.service
journalctl -u kip-server -f
journalctl -u kip-worker -f
```

### BOP evaluation (from laptop via wrapper)
```bash
box_src/eval_bop.sh --self-test                       # no predictions needed
box_src/eval_bop.sh --preds /mnt/data/bop/results/gdrnpp/preds.csv
```

### GPU box remote run
```bash
box_src/gpu_run.sh -- 'nvidia-smi'
box_src/gpu_run.sh -- 'nohup bash /mnt/data/kip_pose/box_src/phase2_chain.sh > /mnt/data/bop/logs/chain.log 2>&1 & echo PID=$!'
box_src/gpu_run.sh -- 'tail -40 /mnt/data/bop/logs/chain.log'
```

---

## Repository layout

```
KIP_POSE/
├── project/                    Main application layer
│   ├── kip_server.py           FastAPI web service (port 8077) — all /api/ endpoints
│   ├── bop_adapter.py          THE boundary: BOP cam-frame → world pose_result (also Z-snap, symmetry, face/upright)
│   ├── e2e_infer.py            Standalone CLI + frozen Contract validator + build_pose_result
│   ├── pose_result.schema.json Frozen output contract (ADR-017, additionalProperties:false)
│   ├── refine_rc.py            Optional M2 render-and-compare refiner (--refine-rc flag)
│   ├── tta_pose.py             Optional test-time augmentation (--tta flag)
│   ├── pipelines/              Multi-pipeline adapter framework (non-production combos 2–7)
│   │   ├── combos.py           7-combo whitelist + feasibility matrix
│   │   ├── composed.py         ComposedPipeline (Seg→Pose), tcamobj_to_world_entry
│   │   ├── contract.py         Re-exports e2e_infer validators — single source of truth
│   │   ├── gdrnpp_adapter.py   Pipeline A wrapper (production monolith)
│   │   ├── seg_base.py         SegAdapter base (POST /segment)
│   │   └── pose_base.py        PoseAdapter base (POST /pose)
│   ├── mesh/                   Docker micro-services (each has app.py + Dockerfile)
│   │   ├── gateway/            Orchestration gateway (fronts all mesh services for browser)
│   │   ├── gdrnpp-svc/         GDRNPP as a service
│   │   ├── fp-svc/             FoundationPose (RGB-D)
│   │   ├── gigapose-svc/       GigaPose (RGB and RGB-D modes)
│   │   ├── yolo-obb-svc/       YOLOv8-OBB detector
│   │   ├── yolo-svc/           YOLO segmentation masks
│   │   └── sam3-svc/           SAM3 promptable segmentation
│   ├── tests/                  pytest suite (106 tests)
│   └── frontend/               Three.js viewer (kip.html + src/ + assets/)
├── box_src/                    GPU-workstation scripts
│   ├── kip_infer_worker.py     Multi-object GDRNPP daemon (port 8078) — holds all 3 checkpoints in VRAM
│   ├── gen_sdg_arm_visible.py  Isaac Sim synthetic data generation
│   ├── isaac_to_bop.py         Raw Isaac bundle → BOP dataset format
│   ├── obb_to_aabb_dets.py     OBB detections → BOP det JSON bridge (needed for GDRNPP training)
│   ├── eval_bop.py / .sh       BOP-toolkit symmetry-aware evaluation harness
│   ├── phase2_chain.sh         Chained training: detector → 3× GDRNPP
│   ├── train_detector_armvis.py YOLOv8-OBB detector training
│   └── gdrnpp/                 Per-object GDRNPP configs (one .py per part)
├── data/                       Mesh data, CAD GLBs
├── foundationpose/             FoundationPose vendored venv (Python 3.10)
├── jetson_live/                Jetson edge-inference server
└── recon/                      Investigation reports (S006, T166, T177, T178)
```

---

## Architecture: how the production pipeline actually works

### The two services on the workstation

**`kip-server.service`** (`project/kip_server.py`, port 8077) is the FastAPI web server. It:
- Serves the static frontend from `project/frontend/`
- Handles async job lifecycle for real-photo and live-sim workflows
- Calls the worker for pose inference
- Orchestrates Isaac Sim subprocesses for the live-sim tab
- Exposes `/api/health`, `/api/metrics`, `/api/real/*`, `/api/sim/*`

**`kip-worker.service`** (`box_src/kip_infer_worker.py`, port 8078) is a persistent stdlib HTTP daemon. It loads all three GDRNPP checkpoints into VRAM at startup (~2.3 GB, ~4 min warm-load) and keeps them warm. The server calls it per-detection at `POST /infer/{obj_id}`.

### The BOP boundary (`bop_adapter.py`)

This is the single most important file. Every pose estimator (GDRNPP, GigaPose, FoundationPose) must deliver its result in BOP camera-frame convention, and `bop_adapter.bop_pose_to_world` converts it to the frozen world-frame contract:

```python
R_m2w = R_w2c.T @ R_m2c
t_m2w = R_w2c.T @ (t_m2c - t_w2c)   # t_m2c MUST be in mm
R_world = R_m2w                        # world = R @ body (column convention)
t_world = t_m2w / 1000.0 - table_origin
```

**Critical unit invariant:** `t_m2c` must be in **mm** at this boundary. GDRNPP's worker internally delivers metres and multiplies ×1000 before returning. The ComposedPipeline (`composed.py`) multiplies `T_cam_obj[:3,3] * 1000.0` before calling `bop_pose_to_world`. If a pose is 1 m above the table, this conversion is missing.

### The frozen output contract (ADR-017)

`pose_result.schema.json` is the frozen contract. Every pipeline adapter must emit a schema-valid document. The validator lives in `e2e_infer.check_pose_result` and `_check_with_jsonschema`; `pipelines/contract.py` re-exports them. Never soften `additionalProperties: false` or remove required fields (`face`, `upright`, `bbox_2d`) — the viewer and evaluation depend on them.

### The multi-pipeline framework (`project/pipelines/`)

Non-production combos (2–7) use `ComposedPipeline`: a `SegAdapter` (Stage 1, POST /segment → masks) paired with a `PoseAdapter` (Stage 2, POST /pose → T_cam_obj 4×4). The gateway at `project/mesh/gateway/app.py` routes browser requests to the right micro-service combo.

The class-mapping invariant (**§6 in the code comments**): mesh services use lowercase (`anker_kurz`), `bop_adapter` uses CamelCase (`Anker_Kurz`). Always go through `obj_id`:
```python
obj_id = CLASS_TO_OBJ_ID[cls]       # "anker_kurz" → 1
part = bop_adapter.part_for_obj_id(obj_id)  # 1 → "Anker_Kurz"
```
Never do `PART_SYMMETRY.get("anker_kurz")` — it returns None and silently skips symmetry canonicalization, poisoning eval.

### The Z-snap (`bop_adapter.planar_z_snap`)

The dominant training-free accuracy gain (+0.04–0.06 AR). Given the predicted rotation, it finds the lowest CAD vertex and snaps the part down to the table. Two guards:
- **Always lift** (dz > 0): a part below the table is physically impossible.
- **Sink guard** (dz < 0): only snap down if `|dz| ≤ max_snap_m` (default 0.10 m) to avoid snapping parts held by the gripper.

This requires AABB-centred mesh vertices — the PLY origin must be at the mesh centre, not some export-time artifact.

### Symmetry handling

Declared in `models/models_info.json` (written by `box_src/gen_models_info.py`):
- `Anker_Kurz`, `Anker_Lang`, `Ringmagnet`: continuous rotation about body-Y
- `Zahnrad`: discrete C_7 about body-Y

At eval time: `bop_toolkit_lib.pose_matching` uses these to score rotationally equivalent poses as correct. At inference time: `bop_adapter.canonicalize_rotation` collapses predictions to a canonical representative, preventing yaw-jitter in the viewer. Both must agree — if `models_info.json` says C_7 and `PART_SYMMETRY` in `bop_adapter.py` says something else, eval and viewer diverge.

### obj_id mapping (single source of truth)

`bop_adapter.OBJ_ID_TO_PART` is the single source. Detector class (0-based) + 1 = obj_id (1-based). Never maintain a separate list:
```
1=Anker_Kurz  2=Anker_Lang  3=Buerstenhalter_2polig
4=Getriebegehaeuse_typ4  5=Ringmagnet  6=Zahnrad
```
Only obj_ids 1, 2, 6 have trained GDRNPP checkpoints. The other three are distractors in synthetic scenes.

---

## Key invariants and gotchas

**Translation unit:** `t_m2c` is in **mm** everywhere BOP-adjacent (worker output, `bop_pose_to_world` input, `scene_camera.json`). World-frame `t_world` is in **metres**. The boundary conversion is `/ 1000.0`. Missing it causes poses to float 1 m above the table.

**Rotation convention:** `world = R @ body` (column convention, NOT row/transpose). `R_world` is stored row-major flat-9 in JSON. Do not transpose at the boundary.

**GDRNPP is OBB-native:** It requires oriented bounding box crops as input. `obb_to_aabb_dets.py` bridges OBB→AABB for GDRNPP's BOP detection format. Using axis-aligned crops from a mask-segmenter degrades accuracy (logged as `degraded: aabb_from_mask` in combo metadata).

**Worker unit bug history:** GDRNPP returns translation in metres; the worker must multiply ×1000 in **both** output code paths (success + fallback). This has been a recurring source of 1 m hover bugs.

**SAM3 class ambiguity:** SAM3 cannot reliably separate Anker_Kurz from Anker_Lang (3 mm size difference). Since T-177, the gateway transfers class labels from yolo-obb to SAM3 detections via IoU-matching. Do not feed SAM3 labels directly to `CLASS_TO_OBJ_ID` without this relabeling step.

**Isaac SDG single-scene bug:** `gen_sdg_arm_visible.py --num-scenes 1` without `--force-counts` registers no semantic labels (only BACKGROUND/UNLABELLED). Use `--force-counts` for single-scene runs.

**USD transform scaling:** `ComputeLocalToWorldTransform` includes `OBJECT_SCALE=1e-3` in the 3×3 block. Normalize column-by-column before comparing against geometric thresholds.

**MegaPose `CONDA_PREFIX`:** `megapose/config.py` reads `os.environ["CONDA_PREFIX"]`. In a plain venv: `export CONDA_PREFIX=/mnt/data/bop/bop-venv`.

**`gltfpack -cc` is lossy:** Despite marketing, EXT_meshopt compression visibly distorts normals/UVs on vendor CAD. Use uncompressed meshes for geometric fidelity.

---

## venv layout on the GPU workstation

| venv | Python | Purpose |
|---|---|---|
| `/mnt/data/isaacsim-venv/` | 3.10 | Isaac Sim 5.1 (SDG data generation) |
| `/mnt/data/bop/train-venv/` | 3.11 | YOLOv8-OBB detector training (ultralytics) |
| `/mnt/data/bop/bop-venv/` | 3.11 | bop_toolkit, CNOS, GigaPose, MegaPose |
| `/mnt/data/bop/gdrnpp-venv/` | 3.11 | GDRNPP (isolated — incompatible pins) |

Do not cross-invoke between venvs. GDRNPP is isolated because its pins (`onnx 1.8.1`, `mmcv-full`, old detectron2) conflict with the bop-venv torch 2.5 stack.

---

## Adding a new part

The full guide is in `project/docs/ADD_NEW_PART.md`. The short version:

1. **Assign `obj_id`**: next free integer after 6. Add to `bop_adapter.OBJ_ID_TO_PART`, `box_src/gen_models_info.py PARTS`, `box_src/isaac_to_bop.py OBJ_ID`. Never renumber existing ids.
2. **Declare symmetry** in `gen_models_info.py PARTS` (`continuous` / `discrete` / `none`) AND in `bop_adapter.PART_SYMMETRY`. They must match.
3. **GLB → BOP model**: run `gen_models_info.py` → `models/obj_<id>.ply` (mm) + `models_info.json`.
4. **Verify BOP-clean**: `validate_bop_full.py` + check `get_symmetry_transformations` count.
5. *(Zero-shot path stops here — use GigaPose/FoundationPose against the BOP model)*
6. **Generate synthetic data** via `gen_sdg_arm_visible.py` + `isaac_to_bop.py`
7. **Retrain**: add to `phase2_chain.sh`, add GDRNPP config in `box_src/gdrnpp/so_configs/`
8. **Wire into viewer**: drop `<PartName>.glb` in `project/frontend/assets/parts/`, register in `partRegistry.js` and `part_meta.json`

---

## API reference (port 8077)

```
GET  /api/health                  → {status, gpu_training_active, trained_objects, ts}
GET  /api/metrics                 → {metric:"ar_ic_bin", objects:{slug:{ar, ...}}}

POST /api/real/infer_async        body: image=@<file>  → {job}
GET  /api/real/job/<job>          → {phase, pct, [result_url, rgb_url, boxes_url]}
GET  /api/real/result/<job>       → pose_result.json
GET  /api/real/rgb/<job>          → image/png (unmodified)
GET  /api/real/boxes/<job>        → image/png (with coloured detector boxes)

GET  /api/sim/generate_async      → {job}  (triggers live Isaac generation, ~80 s)
GET  /api/sim/job/<job>           → {phase, pct, ...}
GET  /api/sim/job_result/<job>    → pose_result.json (GT blue + pred red)
```

---

## pose_result contract (ADR-017, frozen)

```jsonc
{
  "meta": {
    "source_image": "live/abc123",
    "table_origin": [0.0, 0.0, 0.08],   // world pos of table null-point, metres
    "units": "m",
    "source": "worker",                  // or "isaac-live", "preds_best"
    "camera": {"cam_pos": [...], "look_at": [...], "up": [...], "fov_y": 30.45}
  },
  "results": [{
    "instance_id": 1,
    "part": "Anker_Kurz",                // from OBJ_ID_TO_PART
    "face": "face_y-",                   // which body axis faces world-down
    "confidence": 0.93,
    "t_world": [0.453, 0.281, -0.068],  // metres, relative to table_origin
    "R_world": [r00, r01, ..., r22],    // row-major flat 9, world = R @ body
    "upright": false,
    "bbox_2d": [x0, y0, x1, y1]
  }]
}
```

**World frame:** Z-up, origin = table null-point, unit = metres. `world = R @ body` (column convention).
