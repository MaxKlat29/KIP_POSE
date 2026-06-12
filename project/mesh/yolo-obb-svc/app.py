"""
YOLOv8-OBB segmentation micro-service — the 3rd seg source of the mesh.

Unlike yolo-svc (YOLO-seg, pixel polygons), this wraps our trained YOLOv8-**OBB**
detector (oriented boxes, project/models/detector.pt). It is the seg source for
combo 1 (yolo-obb -> GDRNPP, = Pipeline A). GDRNPP consumes the oriented box, not
a pixel mask, so each detection carries an `obb` field [cx,cy,w,h,theta(rad)] in
addition to a (rasterized-quad) `mask_b64` that keeps the frozen contract happy.

POST /segment   (CONTRACT.md §2 — drop-in with yolo-svc / sam3-svc)
  { "rgb_b64": <base64 PNG, uint8 RGB> [, "K", "depth_b64" ignored] }
->{ "detections": [ {"id":0, "class":"anker_kurz", "conf":0.91,
                     "mask_b64": <base64 PNG, 0/255 single channel, full image>,
                     "obb": [cx, cy, w, h, theta]}, ... ] }

GET /health -> {"ok": true}

The detector is 6-class (FROZEN obj_id order: anker_kurz=0, anker_lang=1, ...,
zahnrad=5; box_src/train_detector_armvis.py:38). This service scope is D1 = the
2 Anker classes only; any class id >= 2 (incl. zahnrad) is FILTERED OUT.

Env:
  YOLO_OBB_WEIGHTS  path to the .pt weights (default /weights/detector.pt)
  YOLO_OBB_CONF     confidence threshold (default 0.40 — matches e2e_infer)
  YOLO_OBB_IMGSZ    inference image size (default 1280 — matches training)
"""
import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

WEIGHTS = os.environ.get("YOLO_OBB_WEIGHTS", "/weights/detector.pt")
CONF = float(os.environ.get("YOLO_OBB_CONF", "0.40"))
IMGSZ = int(os.environ.get("YOLO_OBB_IMGSZ", "1280"))

# Mesh-Klassen-Scope: die 2 Anker (D1) + zahnrad (T-178c, Max: Zahnraeder in
# der MoE-Pipeline mit einbeziehen — der GDRNPP-zahnrad-Checkpoint existiert,
# der :8078-Worker nutzt ihn seit T-115). Die 3 untrainierten Distraktoren
# (buerstenhalter_2polig, getriebegehaeuse_typ4, ringmagnet; ids 2-4) bleiben
# gedroppt. Detector ist 6-class FROZEN (train_detector_armvis:38); obj_id =
# cls_id + 1 — zahnrad: cls 5 -> obj_id 6.
MESH_CLASS_NAMES = {0: "anker_kurz", 1: "anker_lang", 5: "zahnrad"}

app = FastAPI(title="yolo-obb-svc")
MODEL = None


@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = YOLO(WEIGHTS)
    names = getattr(MODEL, "names", {}) or {}
    print(f"[yolo-obb-svc] loaded weights: {WEIGHTS} (conf={CONF}, imgsz={IMGSZ})",
          flush=True)
    print(f"[yolo-obb-svc] model class names: {names}", flush=True)
    # Sanity: the FROZEN order must place anker_kurz=0, anker_lang=1. If a model
    # ever ships a different order we still trust its own r.names per-detection,
    # but warn loudly because the gdrnpp coupling assumes obj_id = cls_id + 1.
    for cid, expect in MESH_CLASS_NAMES.items():
        got = str(names.get(cid, "")).lower()
        if got and got != expect:
            print(f"[yolo-obb-svc] WARN class-id {cid} is '{got}', expected "
                  f"'{expect}' — FROZEN obj_id mapping may be violated", flush=True)


class SegmentReq(BaseModel):
    rgb_b64: str
    # Optional, ignored here (frozen-contract fields for other seg sources).
    K: list | None = None
    depth_b64: str | None = None
    prompts: dict | None = None


def _b64_png_to_bgr(b64):
    raw = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _mask_to_b64(mask):
    ok, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _quad_to_full_mask(corners, h, w):
    """Rasterize an oriented-box quad (4 corners) to a full-image 0/255 mask.

    OBB has no pixel polygon, so the contract's mandatory `mask_b64` is the
    filled oriented rectangle. The precise box travels in `obb`; downstream
    GDRNPP reads `obb`, the mask only satisfies the drop-in contract + any
    mask-consuming sibling.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(corners, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return mask


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/segment")
def segment(req: SegmentReq):
    img = _b64_png_to_bgr(req.rgb_b64)
    h, w = img.shape[:2]
    result = MODEL.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

    detections = []
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return {"detections": detections}

    names = getattr(result, "names", None) or {}
    xywhr = obb.xywhr.cpu().numpy()          # (N,5): cx,cy,w,h,theta(rad)
    quads = obb.xyxyxyxy.cpu().numpy()       # (N,4,2): corner points
    classes = obb.cls.cpu().numpy().astype(int)
    confs = obb.conf.cpu().numpy()

    for i in range(len(xywhr)):
        cls_id = int(classes[i])
        # D1 scope: keep only the two Anker classes; drop the rest (incl. zahnrad).
        cls_name = MESH_CLASS_NAMES.get(cls_id)
        if cls_name is None:
            continue
        # Cross-check against the model's own label (defensive; FROZEN order).
        model_name = str(names.get(cls_id, "")).lower()
        if model_name and model_name != cls_name:
            cls_name = model_name if model_name in ("anker_kurz", "anker_lang") else cls_name

        cx, cy, bw, bh, theta = [float(v) for v in xywhr[i]]
        if bw < 4 or bh < 4:                 # degenerate box -> skip
            continue
        mask = _quad_to_full_mask(quads[i], h, w)
        detections.append({
            "id": len(detections),
            "class": cls_name,
            "conf": float(confs[i]),
            "mask_b64": _mask_to_b64(mask),
            "obb": [cx, cy, bw, bh, theta],
        })

    return {"detections": detections}
