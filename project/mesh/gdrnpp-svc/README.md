# gdrnpp-svc — Pipeline A pose service (combo 1: yolo-obb → GDRNPP)

GDRNPP is our accuracy main line. This service exposes the per-object GDRNPP
checkpoints (anker_kurz=obj_id 1, anker_lang=obj_id 2; **D1 = 2 classes, no
zahnrad**) behind the FROZEN mesh `/pose` contract (`../CONTRACT.md` §3 + §4).

It is the **only** non-freely-combinable pose source: it reads the oriented box
`obb` from each instance (not `mask_b64`) and is therefore hard-coupled to
`yolo-obb` (CONTRACT §4). That coupling is enforced by the gateway (`yolo-obb`
forced when `pose_source=gdrnpp`).

## Image decision — native venv + stdlib HTTP, NOT Docker

GDRNPP needs detectron2 + the gdrnpp checkout + the trained per-object
checkpoints. All three are already wired in the box-native venv
`/mnt/data/bop/gdrnpp-venv` — the **same environment the live `:8078` worker
runs in** (detectron2 0.6, torch 2.5.1+cu121). That venv has **no fastapi**, so —
exactly like `box_src/kip_infer_worker.py` — this service is a stdlib
`http.server`, not FastAPI.

Reasons against a Docker image here (unlike fp/gigapose):

- A detectron2 + GDRNPP image is ~20 GB and reproduces a notoriously painful
  build (custom CUDA ops, flow networks) for **zero gain** — the env already
  exists and works.
- The live `:8078` worker is itself a native-venv stdlib daemon; matching that
  keeps one consistent GDRNPP runtime pattern on the box.
- Checkpoints (`output/gdrn/poseIsaacPbrSO/<slug>/model_best.pth`) and the gdrnpp
  repo are large host artifacts already in place; no Docker mount juggling.

→ **gdrnpp-svc runs as a separate native process on a separate port (8012),
fully isolated from the live `:8078` worker** (different process, different VRAM
fraction). No `pkill`, no concurrency conflict with the live worker.

## Det-driven, not GT-dependent (CONTRACT / T-115)

The service reuses the live worker's `run_upload` machinery: it builds an
**isolated det-consistent dataset root** (`_build_det_consistent_root`, symlinked
RGB + dummy GT keyed off the detections) so every **detected** object gets posed,
even one with no `scene_gt` backing. It never reads or writes any real
`scene_gt`, never touches the live `:8078` worker.

## Pose convention

GDRNPP emits `R_m2c` (model→cam) + `t_m2c` in mm — already the OpenCV cam frame
(`project/bop_adapter.py:10-11`). So:

```
T_cam_obj = [[ R_m2c , t_m2c / 1000 ],
             [ 0 0 0 ,      1       ]]    # METRES, mesh→cam, OpenCV cam (CONTRACT §1)
```

No D_FLIP, no world transform — those only apply when going to the world frame
(`bop_adapter.bop_pose_to_world`) or three.js (FE flip).

## Run (on the box)

```bash
# separate instance, port 8012, 0.30 VRAM cap (coexists with live :8078 @ 0.40)
GDRN_ROOT=/mnt/data/bop/repos/gdrnpp GDRNPP_PORT=8012 GDRNPP_MEMFRAC=0.30 \
  /mnt/data/bop/gdrnpp-venv/bin/python project/mesh/gdrnpp-svc/app.py
# or: bash project/mesh/gdrnpp-svc/run_box.sh
curl -s localhost:8012/health   # {"ok":true,"state":"ready","classes":["anker_kurz","anker_lang"],...}
```

`run_box.sh` rsyncs `box_src/kip_infer_worker.py` (imported for the T-115
det-root builder) onto the box `PYTHONPATH` and launches the daemon.

## Contract surface

- `POST /pose` — `{rgb_b64, K[9], iterations?, instances:[{id,class,obb|mask_b64}]}`
  → `{poses:[{id,class,T_cam_obj,score}]}`. `obb=[cx,cy,w,h,theta(rad)]` preferred;
  `mask_b64` is an AABB fallback. Unknown class / no box → instance silently
  skipped (CONTRACT §3).
- `GET /health` → `{ok, state∈{loading,ready,error}, classes, loaded_obj_ids, cuda}`.
