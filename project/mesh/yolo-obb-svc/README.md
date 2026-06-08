# yolo-obb-svc — YOLOv8-OBB seg source (combo 1 / Pipeline A)

The **3rd segmentation source** of the pose mesh. Wraps our trained YOLOv8-**OBB**
detector (oriented bounding boxes) as a drop-in `/segment` service per
[`../CONTRACT.md`](../CONTRACT.md) §2.

Where `yolo-svc` (YOLO-seg) and `sam3-svc` produce **pixel masks**, this service
produces **oriented boxes** `[cx, cy, w, h, θ]` carried in the `obb` field. That
is the seg input GDRNPP needs (combo 1 = Pipeline A, the accuracy king). GDRNPP is
**hard-coupled** to yolo-obb (CONTRACT.md §4): it reads `obb`, not a pixel mask.

## Contract (`POST /segment`)

```jsonc
// request
{ "rgb_b64": "<base64 PNG, uint8 RGB>" }     // K / depth_b64 / prompts ignored here

// response
{ "detections": [
    { "id": 0,
      "class": "anker_kurz",                 // ∈ {anker_kurz, anker_lang}
      "conf": 0.91,
      "mask_b64": "<base64 PNG 0/255, full image>",   // rasterized OBB quad
      "obb": [cx, cy, w, h, theta] }          // theta in RADIANS (ultralytics xywhr)
] }
```

```jsonc
// GET /health
{ "ok": true }
```

### Design notes

- **OBB → `obb` field.** Ultralytics gives `r.obb.xywhr` = `[cx, cy, w, h, θ(rad)]`
  directly — that *is* the contract's `obb`. The 4 corners (`r.obb.xyxyxyxy`) are
  rasterized to the **mandatory** full-image `mask_b64` (the contract requires a
  mask; OBB has no pixel polygon, so the filled oriented rectangle stands in). The
  precise box travels in `obb`; downstream GDRNPP consumes `obb`, the mask only
  satisfies the drop-in contract.
- **2-class scope (D1).** The detector is 6-class with a FROZEN order
  (`anker_kurz=0, anker_lang=1, …, zahnrad=5`; `box_src/train_detector_armvis.py`).
  This service is scoped to the **two Anker classes** — class ids ≥ 2 (incl.
  zahnrad) are **filtered out**. Class names come out lowercase (`anker_kurz` /
  `anker_lang`) to match the rest of the mesh.
- **obj_id alignment.** `cls_id + 1 == obj_id` (anker_kurz→1, anker_lang→2),
  consistent with `gigapose_infer.CLASS_TO_OBJ_ID` and `bop_adapter` (CONTRACT.md §6).

## Image

Lightweight — base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` + ultralytics +
FastAPI. It does **not** need the heavy `foundationpose:ampere` / `gigapose:ampere`
images (no pytorch3d / nvdiffrast / xformers). The stock torch CUDA wheel already
ships sm_86 (Ampere / RTX 3090) kernels, so no recompile is needed.

## Env

| Var | Default | Meaning |
|---|---|---|
| `YOLO_OBB_WEIGHTS` | `/weights/detector.pt` | path to the `.pt` OBB detector |
| `YOLO_OBB_CONF` | `0.40` | confidence threshold (matches `e2e_infer`) |
| `YOLO_OBB_IMGSZ` | `1280` | inference image size (matches training) |

## Smoke (box)

`smoke_box.py` exercises the exact `segment()` code path against a real image and
validates the response against contract §2 (id ordering, 2-Anker classes, conf
range, full-image 0/255 mask, 5-tuple `obb`). It does **not** start a server.

```bash
# on the box (train-venv has ultralytics 8.4.x + torch cu121)
YOLO_OBB_WEIGHTS=/mnt/data/kip_pose/data/detector_armvis/detector.pt \
  /mnt/data/train-venv/bin/python smoke_box.py \
  --image /mnt/data/pose_eval/project/input/scene_0000.png [--device cpu]
```

### S-005 measured result (RTX 3090)

- `scene_0000.png` (1280×720) → **5 detections** (2× `anker_kurz`, 3× `anker_lang`),
  classes 2–5 (incl. zahnrad) correctly filtered. Contract §2 **PASS**.
- VRAM ≈ **456 MiB** process footprint (torch peak alloc 190 MB / reserved 222 MB);
  fully released after the call (no leak). **Cheapest service in the mesh** — input
  for the VRAM lifecycle / eviction work (S-007).
- CPU path ≈ 1.5 s / image; GPU steady-state sub-second (first call includes CUDA
  warmup).

## Wiring (next, S-004)

The gateway URL is set in `docker-compose.yml` (`YOLO_OBB_URL=http://yolo-obb-svc:8011`).
Adding `INFER_SOURCES["yolo-obb"]` + the GDRNPP-coupling gate to the gateway is the
S-004 gateway-extension story (CONTRACT.md §8).
