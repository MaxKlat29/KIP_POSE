"""
Orchestrator gateway. No GPU. Fronts yolo-svc (segmentation) and fp-svc (6-DoF
pose) and exposes a single multipart /predict endpoint for the browser frontend.

Flow for POST /predict:
  rgb (PNG/JPG file), depth (uint16 mm PNG file), fx/fy/cx/cy (form floats),
  iterations (form int=5), top_n (form int, optional), want_pointcloud (form bool),
  pose_source (form str, default 'foundationpose').
    1. read rgb bytes -> base64, POST to the chosen seg_source /segment
    2. optionally keep top_n detections by conf
    3. build instances, look up the chosen pose_source, POST to its /pose
       (forwarding depth only when that pose source needs it)
    4. merge conf back onto poses by id
    5. optionally back-project a decimated point cloud from depth via K
  -> {width,height,K,instances:[{id,class,conf,T_cam_obj,mask_b64}],pointcloud}

Two orthogonal selectors drive the pipeline: seg_source picks the MASK source
(yolo / gt) and pose_source picks the 6-DoF POSE estimator (FoundationPose or
GigaPose RGB-D / RGB-only). They are independent stages — GigaPose is a pose
estimator, not a segmenter, so it never appears in INFER_SOURCES.

Coordinate convention: poses and point cloud are in the OpenCV camera frame
(x right, y down, +z forward), metres. The frontend converts to three.js.

Env:
  YOLO_URL       default http://yolo-svc:8001
  FP_URL         default http://fp-svc:8002
  GIGAPOSE_URL   default http://gigapose-svc:8003
"""
import base64
import json
import os
import time

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

YOLO_URL = os.environ.get("YOLO_URL", "http://yolo-svc:8001")
FP_URL = os.environ.get("FP_URL", "http://fp-svc:8002")
GIGAPOSE_URL = os.environ.get("GIGAPOSE_URL", "http://gigapose-svc:8003")
SAM3_URL = os.environ.get("SAM3_URL", "http://sam3-svc:8004")
YOLO_OBB_URL = os.environ.get("YOLO_OBB_URL", "http://yolo-obb-svc:8011")
GDRNPP_URL = os.environ.get("GDRNPP_URL", "http://gdrnpp-svc:8012")

# ── Segmentation-source registry (the mask-source dropdown) ─────────────────
# A segmentation source produces per-instance masks. INFER sources are HTTP
# services exposing POST /segment {rgb_b64} -> {detections:[{id,class,conf,mask_b64}]}.
# To add a future model (e.g. Mask2Former): stand up a service with that contract,
# add its URL here, and add a matching option in the frontend dropdown.
INFER_SOURCES = {
    "yolo": {"label": "YOLO26n-seg (infer masks)", "url": YOLO_URL},
    "sam3": {"label": "SAM3 (promptable concept masks)", "url": SAM3_URL},
    # yolo-obb is the ORIENTED-box detector. It is the ONLY source that carries an
    # `obb` field, which GDRNPP (Pipeline A) consumes instead of a mask (CONTRACT
    # §4). It is hard-coupled to the gdrnpp pose source (see GDRNPP_COUPLED_SEG).
    "yolo-obb": {"label": "YOLO26-OBB (oriented boxes → GDRNPP)", "url": YOLO_OBB_URL},
}
# "gt" is a special non-inference source: the caller supplies ground-truth masks
# (the sim instance segmentation) directly in the request, so no model is run.
GT_SOURCE = {"id": "gt", "label": "Ground-truth masks (sim)"}

# ── Pose-source registry (the "pose estimator" dropdown) ────────────────────
# A pose source is an HTTP service exposing POST /pose
#   {rgb_b64, depth_b64?, K, iterations, instances:[{id,class,mask_b64}], ...}
#   -> {poses:[{id,class,T_cam_obj[,score]}]}  (T_cam_obj = OpenCV cam frame, m).
# This is a SEPARATE stage from segmentation: masks come from the seg_source
# above; pose_source only chooses which 6-DoF estimator consumes them.
#  - foundationpose : the original RGB-D path; payload byte-identical to before.
#  - gigapose_rgbd  : GigaPose coarse + GenFlow(5-hypo) + Kabsch depth refine.
#  - gigapose_rgb   : GigaPose coarse + GenFlow(5-hypo), no depth, no Kabsch.
# 'needs_depth' gates whether depth_b64 is forwarded; the two gigapose ids carry
# a 'pipeline' selector that the svc uses to derive its depth/Kabsch tail branch.
# To add a future estimator: stand up a /pose service and add an entry here.
POSE_SOURCES = {
    # gdrnpp = Pipeline A (combo 1, the accuracy main line). It is RGB (CONTRACT §5:
    # needs_depth=no; depth is only an optional refine we do not forward). It reads
    # the `obb` oriented box from each instance, NOT a mask, so it is hard-coupled to
    # the yolo-obb seg source (GDRNPP_COUPLED_SEG below). rgb_only=True relaxes the
    # FE depth guard: combo 1 is selectable without a depth upload.
    "gdrnpp": {
        "label": "GDRNPP (Pipeline A, RGB — yolo-obb only)",
        "url": GDRNPP_URL,
        "endpoint": "/pose",
        "needs_depth": False,
        "rgb_only": True,
    },
    "foundationpose": {
        "label": "FoundationPose (RGB-D)",
        "url": FP_URL,
        "endpoint": "/pose",
        "needs_depth": True,
        "rgb_only": False,
    },
    "gigapose_rgbd": {
        "label": "GigaPose RGB-D (GenFlow 5-hypo + Kabsch)",
        "url": GIGAPOSE_URL,
        "endpoint": "/pose",
        "needs_depth": True,
        "pipeline": "rgbd",
        "hypotheses": 5,
        "rgb_only": False,
    },
    "gigapose_rgb": {
        "label": "GigaPose RGB-only (GenFlow 5-hypo)",
        "url": GIGAPOSE_URL,
        "endpoint": "/pose",
        "needs_depth": False,
        "pipeline": "rgb",
        "hypotheses": 5,
        "rgb_only": True,
    },
}

# ── GDRNPP coupling (CONTRACT §4) ────────────────────────────────────────────
# GDRNPP (= Pipeline A) is the ONLY non-freely-combinable pose source: it consumes
# the oriented box `obb`, which only yolo-obb emits. So when pose_source=gdrnpp the
# seg_source is forced to GDRNPP_COUPLED_SEG and the obb field is forwarded into the
# /pose instances. This is the server side of the 7-combo whitelist (CONTRACT §5):
# the only valid gdrnpp combo is yolo-obb → gdrnpp.
GDRNPP_COUPLED_SEG = "yolo-obb"

# Pose estimation can take minutes for many instances; be generous.
POSE_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)
SEGMENT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0)

# Point cloud decimation parameters.
PC_STRIDE = 8
PC_MAX_POINTS = 20000

app = FastAPI(title="kip-pose-gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Raw SDG -> GT overlay conversion ────────────────────────────────────────
# The viewer only ever consumes T_cam_obj (OpenCV cam frame, metres). Raw Isaac
# Sim SDG output stores 6-DoF GT as a row-major native->world transform with a
# 0.001 asset scale baked in (bbox_3d_*.json). This converts it, mirroring
# kip-pose-detection/pose_eval/fp_build_scene.py exactly. Conversion lives server-side
# because the math is non-trivial/shared with the detection repo, and because the
# browser canvas is 8-bit and would corrupt instance ids > 255 in the PNG.
D_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera -> OpenCV camera

# Default camera extrinsic: the GST_Scene /World/Zivid world->camera "view"
# matrix (row-major, USD convention) that the bundled poc500 frames + most SDG
# runs against that scene share. Used when the /gt_overlay request omits a
# world2cam upload, so the common case "convert a poc500/GST_Scene frame" needs
# no extra file. Override by uploading a world2cam_view.txt for a different scene.
DEFAULT_WORLD2CAM = np.array([
    [-0.99939083, 0.00000000, 0.03489950, 0.00000000],
    [-0.00665914, -0.98162718, -0.19069276, 0.00000000],
    [0.03425829, -0.19080900, 0.98102920, 0.00000000],
    [0.41262822, 0.29634180, -1.07804259, 1.00000000],
], dtype=np.float64)


def _parse_mat(text: str) -> np.ndarray:
    rows = [[float(x) for x in ln.split()] for ln in text.strip().splitlines() if ln.strip()]
    m = np.asarray(rows, dtype=np.float64)
    if m.shape != (4, 4):
        raise HTTPException(status_code=400,
                            detail=f"world2cam must be a 4x4 matrix, got shape {m.shape}")
    return m


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _gt_pose_cam_obj(bbox_transform, T_world2cvcam: np.ndarray) -> np.ndarray:
    """bbox_3d transform (row-major native->world, 0.001 scale) -> T_cam_obj."""
    M = np.asarray(bbox_transform, dtype=np.float64)
    T_n2w = M.T
    R = _orthonormalize(T_n2w[:3, :3] * 1000.0)  # strip the 0.001 asset scale
    t = T_n2w[:3, 3]
    T_mesh2world = np.eye(4)
    T_mesh2world[:3, :3] = R
    T_mesh2world[:3, 3] = t
    return T_world2cvcam @ T_mesh2world


def _parse_instance_labels(info: dict) -> dict:
    m = info.get("idToLabels", info)
    out = {}
    for k, v in m.items():
        try:
            iid = int(k)
        except (ValueError, TypeError):
            continue
        prim = v if isinstance(v, str) else (
            v.get("instanceId") or v.get("prim") or v.get("class") or json.dumps(v)
        )
        out[iid] = prim
    return out


def _mask_b64(inst_img: np.ndarray, iid: int):
    arr = np.where(inst_img == iid, 255, 0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return None, 0
    return base64.b64encode(buf.tobytes()).decode("ascii"), int((arr > 0).sum())


def _convert_sdg_to_overlay(bbox, inst_img, id2prim, T_world2cvcam, K):
    """Return [{id,class,occlusion,T_cam_obj,mask_b64?}] from raw SDG data.

    With an instance image: iterate visible ids, link each to its bbox_3d row
    (via instance_labels, else 3D-centroid projection), extract its mask. Without
    one: emit every bbox_3d row as a pose (no masks, no visibility filter)."""
    data = bbox["data"]
    id2label = bbox["info"]["idToLabels"]
    prim_paths = bbox["info"].get("primPaths", [None] * len(data))
    by_prim = {}
    for i, row in enumerate(data):
        by_prim[prim_paths[i]] = (row, id2label[str(int(row[0]))]["class"].lower())

    out = []
    if inst_img is None:
        for i, row in enumerate(data):
            cls = id2label[str(int(row[0]))]["class"].lower()
            out.append({
                "id": i,
                "class": cls,
                "occlusion": float(row[8]) if len(row) > 8 else None,
                "T_cam_obj": _gt_pose_cam_obj(row[7], T_world2cvcam).tolist(),
            })
        return out

    vis_ids = [int(i) for i in np.unique(inst_img) if int(i) != 0]
    for iid in vis_ids:
        entry = None
        if id2prim is not None:
            prim = id2prim.get(iid)
            if prim in by_prim:
                entry = by_prim[prim]
        else:
            mask = inst_img == iid
            for row, cls in by_prim.values():
                M = np.asarray(row[7]).T
                c = np.array([(row[1] + row[4]) / 2, (row[2] + row[5]) / 2,
                              (row[3] + row[6]) / 2, 1.0])
                cc = T_world2cvcam @ (M @ c)
                if cc[2] <= 0:
                    continue
                u = int(round(K[0, 0] * cc[0] / cc[2] + K[0, 2]))
                v = int(round(K[1, 1] * cc[1] / cc[2] + K[1, 2]))
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                    entry = (row, cls)
                    break
        if entry is None:
            continue
        row, cls = entry
        b64, npx = _mask_b64(inst_img, iid)
        if npx == 0:
            continue
        out.append({
            "id": int(iid),
            "class": cls,
            "occlusion": float(row[8]) if len(row) > 8 else None,
            "T_cam_obj": _gt_pose_cam_obj(row[7], T_world2cvcam).tolist(),
            "mask_b64": b64,
        })
    return out


def _backproject_pointcloud(depth_bytes: bytes, rgb_bgr: np.ndarray, K: dict):
    """Decimated back-projection of a uint16-mm depth PNG to camera-frame metres."""
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_m = depth.astype(np.float32) / 1000.0  # mm -> m

    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    h, w = depth_m.shape[:2]

    # Subsample on a regular grid (every PC_STRIDE-th pixel).
    vs = np.arange(0, h, PC_STRIDE)
    us = np.arange(0, w, PC_STRIDE)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel()
    vv = vv.ravel()
    z = depth_m[vv, uu]

    valid = z > 0
    uu, vv, z = uu[valid], vv[valid], z[valid]

    # Cap point count by further random subsampling if needed.
    if uu.size > PC_MAX_POINTS:
        idx = np.random.choice(uu.size, PC_MAX_POINTS, replace=False)
        uu, vv, z = uu[idx], vv[idx], z[idx]

    x = (uu.astype(np.float32) - cx) * z / fx
    y = (vv.astype(np.float32) - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1)

    # Sample colour. rgb_bgr is BGR (cv2); convert sampled pixels to RGB 0-255.
    colors_bgr = rgb_bgr[vv, uu]  # (N, 3) BGR
    colors_rgb = colors_bgr[:, ::-1]

    return {"xyz": xyz.tolist(), "rgb": colors_rgb.astype(int).tolist()}


@app.get("/health")
async def health():
    async def ping(url):
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
                r = await client.get(url + "/health")
                r.raise_for_status()
                return r.json()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    yolo = await ping(YOLO_URL)
    fp = await ping(FP_URL)
    gigapose = await ping(GIGAPOSE_URL)
    sam3 = await ping(SAM3_URL)
    yolo_obb = await ping(YOLO_OBB_URL)
    gdrnpp = await ping(GDRNPP_URL)
    # fp + yolo must be up for the default pipeline; the others (gigapose, sam3,
    # yolo-obb, gdrnpp) are reported but do not gate overall health — any may be
    # intentionally not running to save VRAM (VRAM-lifecycle, ADR-021).
    ok = bool(yolo.get("ok")) and bool(fp.get("ok"))
    return {"ok": ok, "yolo": yolo, "fp": fp, "gigapose": gigapose, "sam3": sam3,
            "yolo_obb": yolo_obb, "gdrnpp": gdrnpp}


@app.get("/sources")
def sources():
    """Available segmentation (mask) sources for the frontend dropdown."""
    out = [{"id": k, "label": v["label"], "kind": "infer"} for k, v in INFER_SOURCES.items()]
    out.append({"id": GT_SOURCE["id"], "label": GT_SOURCE["label"], "kind": "ground_truth"})
    return {"sources": out}


@app.get("/pose_sources")
def pose_sources():
    """Available 6-DoF pose estimators for the frontend dropdown (a stage
    separate from segmentation). 'rgb_only' tells the frontend to relax its
    depth-required guard and skip the depth upload for that source."""
    out = [
        {"id": k, "label": v["label"], "rgb_only": v["rgb_only"]}
        for k, v in POSE_SOURCES.items()
    ]
    return {"pose_sources": out}


@app.post("/gt_overlay")
async def gt_overlay(
    bbox_3d: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    world2cam: UploadFile | None = File(None),
    instance: UploadFile | None = File(None),
    instance_labels: UploadFile | None = File(None),
):
    """Convert a raw Isaac-Sim SDG frame into the viewer's GT overlay bundle.

    Required: bbox_3d_*.json (6-DoF GT) + intrinsics (fx/fy/cx/cy). Optional:
    world2cam_view.txt (camera extrinsic — defaults to the GST_Scene Zivid
    extrinsic when omitted), instance_*.png (per-object masks + visibility
    filter), instance_labels_*.json (robust mask<->pose linkage; falls back to
    3D-centroid projection if absent).

    -> {"instances":[{id,class,occlusion,T_cam_obj[,mask_b64]}]}  (T_cam_obj =
       OpenCV cam frame, metres — exactly what the upload slot expects).
    """
    try:
        bbox = json.loads(await bbox_3d.read())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"bbox_3d is not valid JSON: {e}")
    if not (isinstance(bbox, dict) and "data" in bbox and "info" in bbox):
        raise HTTPException(
            status_code=400,
            detail="bbox_3d must be a raw Replicator bounding_box_3d file "
                   "(top-level 'data' + 'info'). For already-converted overlays, "
                   "use the direct upload instead.")

    view = DEFAULT_WORLD2CAM
    if world2cam is not None:
        raw_w2c = await world2cam.read()
        if raw_w2c.strip():  # treat an empty upload as "use default"
            view = _parse_mat(raw_w2c.decode("utf-8", "replace"))
    T_world2cvcam = D_FLIP @ view.T
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    inst_img = None
    if instance is not None:
        raw = await instance.read()
        if raw:
            inst_img = cv2.imdecode(
                np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
            )
            if inst_img is None:
                raise HTTPException(status_code=400, detail="could not decode instance PNG")
            if inst_img.ndim == 3:  # collapse a single-channel-as-RGB encoding
                inst_img = inst_img[..., 0]

    id2prim = None
    if instance_labels is not None:
        raw = await instance_labels.read()
        if raw:
            try:
                id2prim = _parse_instance_labels(json.loads(raw))
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"instance_labels is not valid JSON: {e}")

    try:
        instances = _convert_sdg_to_overlay(bbox, inst_img, id2prim, T_world2cvcam, K)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Most often a bbox_2d (or otherwise non-3D) file slipped through: it has
        # the same data/info shape but rows lack the 4x4 transform at row[7].
        raise HTTPException(
            status_code=422,
            detail=f"could not parse as a bounding_box_3d file ({type(e).__name__}: {e}). "
                   "Make sure you uploaded bbox_3d_*.json, not bbox_2d_*.json.")
    if not instances:
        raise HTTPException(
            status_code=422,
            detail="no instances converted — check that bbox_3d, instance and "
                   "instance_labels are from the same frame.")
    return {"instances": instances, "num_instances": len(instances)}


async def _segment(seg_source: str, rgb_b64: str, gt_masks: str | None,
                   seg_prompts: str | None = None) -> list[dict]:
    """Return detections [{id,class,conf,mask_b64}] from the chosen source.

    seg_prompts is an optional JSON object {class: concept prompt} forwarded to
    the segmentation service as 'prompts' (per-request override; sam3-svc merges
    it over its env defaults, yolo-svc ignores the extra field)."""
    if seg_source == GT_SOURCE["id"]:
        if not gt_masks:
            raise HTTPException(
                status_code=400,
                detail="seg_source='gt' requires a gt_masks bundle (ground-truth "
                       "masks only exist for the sim sample frame).")
        try:
            parsed = json.loads(gt_masks)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"gt_masks is not valid JSON: {e}")
        items = parsed.get("instances", parsed) if isinstance(parsed, dict) else parsed
        return [
            {"id": i, "class": it["class"], "conf": float(it.get("conf", 1.0)),
             "mask_b64": it["mask_b64"]}
            for i, it in enumerate(items)
        ]

    src = INFER_SOURCES.get(seg_source)
    if src is None:
        raise HTTPException(status_code=400, detail=f"unknown seg_source '{seg_source}'")
    payload: dict = {"rgb_b64": rgb_b64}
    if seg_prompts:
        try:
            prompts = json.loads(seg_prompts)
            assert isinstance(prompts, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in prompts.items())
        except (ValueError, AssertionError):
            raise HTTPException(
                status_code=400,
                detail="seg_prompts must be a JSON object {class: prompt string}")
        payload["prompts"] = prompts
    try:
        async with httpx.AsyncClient(timeout=SEGMENT_TIMEOUT) as client:
            r = await client.post(src["url"] + "/segment", json=payload)
            r.raise_for_status()
            return r.json().get("detections", [])
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"{seg_source}-svc error: {e}")


@app.post("/predict")
async def predict(
    rgb: UploadFile = File(...),
    depth: UploadFile | None = File(None),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    iterations: int = Form(5),
    top_n: int | None = Form(None),
    want_pointcloud: bool = Form(False),
    seg_source: str = Form("yolo"),
    pose_source: str = Form("foundationpose"),
    gt_masks: str | None = Form(None),
    seg_prompts: str | None = Form(None),
    degraded: bool = Form(False),
):
    pose = POSE_SOURCES.get(pose_source)
    if pose is None:
        raise HTTPException(status_code=400, detail=f"unknown pose_source '{pose_source}'")

    # GDRNPP coupling (CONTRACT §4/§5): gdrnpp consumes the obb, which only yolo-obb
    # emits, so it is hard-coupled to that seg source. If the caller picked gdrnpp
    # but a different seg_source (or none), force it to yolo-obb — this is the server
    # side of the 7-combo whitelist (yolo-obb → gdrnpp is the only valid gdrnpp combo).
    #
    # degraded=true (T-153) is the explicit opt-in for the eval's GDRNPP-degraded
    # combos (yolo-seg→gdrnpp, sam3→gdrnpp): keep the chosen mask-emitting seg_source
    # and let gdrnpp-svc derive an AABB from the mask (its documented fallback). This
    # is OFF by default — the live FE path never sets it, so Pipeline A behaviour and
    # the whitelist are unchanged. It only exists so the batch-eval can FILL the
    # degraded AR rows instead of leaving them as empty reference cells (CONTRACT §4
    # notes the degraded AABB-from-mask path is measured, not forbidden).
    if pose_source == "gdrnpp" and seg_source != GDRNPP_COUPLED_SEG and not degraded:
        seg_source = GDRNPP_COUPLED_SEG
    # And the inverse guard: yolo-obb's oriented boxes are only meaningful for gdrnpp;
    # any mask-consuming pose source would ignore the obb. Reject the off-whitelist mix
    # rather than silently produce a mask-based pose from an obb-only detector.
    if seg_source == GDRNPP_COUPLED_SEG and pose_source != "gdrnpp":
        raise HTTPException(
            status_code=400,
            detail=f"seg_source '{GDRNPP_COUPLED_SEG}' is only valid with "
                   f"pose_source 'gdrnpp' (CONTRACT §4 GDRNPP coupling).")

    # depth is required only when the chosen pose source consumes it (and for a
    # point cloud). The RGB-only GigaPose path runs with no depth upload at all.
    depth_bytes = await depth.read() if depth is not None else b""
    if pose["needs_depth"] and not depth_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"pose_source '{pose_source}' requires a depth image.")

    rgb_bytes = await rgb.read()

    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise HTTPException(status_code=400, detail="could not decode rgb image")
    height, width = rgb_bgr.shape[:2]

    rgb_b64 = base64.b64encode(rgb_bytes).decode("ascii")
    depth_b64 = base64.b64encode(depth_bytes).decode("ascii") if depth_bytes else None

    # 1. segmentation (chosen pipeline source: an inference model, or supplied GT)
    t_seg0 = time.perf_counter()
    detections = await _segment(seg_source, rgb_b64, gt_masks, seg_prompts)
    seg_ms = (time.perf_counter() - t_seg0) * 1000.0

    # 2. keep top_n by confidence
    if top_n is not None and top_n >= 0:
        detections = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)[:top_n]

    # 3. build instances + pose. Forward the `obb` field when the detection carries
    #    one (yolo-obb only); gdrnpp reads it instead of the mask (CONTRACT §4). Other
    #    pose sources ignore the extra key.
    instances = []
    for d in detections:
        inst = {"id": d["id"], "class": d["class"], "mask_b64": d.get("mask_b64")}
        if d.get("obb") is not None:
            inst["obb"] = d["obb"]
        instances.append(inst)
    conf_by_id = {d["id"]: d.get("conf") for d in detections}

    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    # Build the /pose payload for the chosen pose source. FoundationPose keeps the
    # exact wire contract it always had (rgb_b64/depth_b64/K/iterations/instances);
    # the gigapose sources add a 'pipeline' + 'hypotheses' selector and omit depth
    # for the RGB-only path. depth_b64 is forwarded only when needs_depth.
    pose_payload = {
        "rgb_b64": rgb_b64,
        "K": K,
        "iterations": iterations,
        "instances": instances,
    }
    if pose["needs_depth"]:
        pose_payload["depth_b64"] = depth_b64
    if "pipeline" in pose:
        pose_payload["pipeline"] = pose["pipeline"]
    if "hypotheses" in pose:
        pose_payload["hypotheses"] = pose["hypotheses"]

    poses = []
    pose_ms = 0.0
    if instances:
        t_pose0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=POSE_TIMEOUT) as client:
                r = await client.post(pose["url"] + pose["endpoint"], json=pose_payload)
                r.raise_for_status()
                poses = r.json().get("poses", [])
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"{pose_source}-svc error: {e}")
        pose_ms = (time.perf_counter() - t_pose0) * 1000.0

    # 4. merge conf + mask back by id (the mask lets the frontend's 2D mode draw
    #    the predicted segmentation; it round-trips from the detection stage).
    mask_by_id = {d["id"]: d.get("mask_b64") for d in detections}
    out_instances = [
        {
            "id": p.get("id"),
            "class": p.get("class"),
            "conf": conf_by_id.get(p.get("id")),
            "T_cam_obj": p.get("T_cam_obj"),
            "mask_b64": mask_by_id.get(p.get("id")),
        }
        for p in poses
    ]

    # 5. optional point cloud (needs depth; skipped on the RGB-only path)
    pointcloud = None
    if want_pointcloud and depth_bytes:
        pointcloud = _backproject_pointcloud(
            depth_bytes, rgb_bgr, {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        )

    return {
        "width": width,
        "height": height,
        "K": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "seg_source": seg_source,
        "pose_source": pose_source,
        "num_detections": len(detections),
        "instances": out_instances,
        "pointcloud": pointcloud,
        # Per-stage wall-clock (ms), measured at the gateway. pose_ms is the
        # whole pose-svc round-trip (sequential estimation over all instances).
        "timings": {
            "seg_ms": round(seg_ms, 1),
            "pose_ms": round(pose_ms, 1),
            "num_posed": len(poses),
        },
    }
