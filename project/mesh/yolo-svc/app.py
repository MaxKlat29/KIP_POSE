"""
YOLO segmentation micro-service. Runs INSIDE the foundationpose:blackwell image
(torch cu128). Loads one ultralytics YOLO seg model at startup and rasterizes
each detection's polygon to a full-image binary mask.

POST /segment
  { "rgb_b64": <base64 PNG, uint8 RGB> }
->{ "detections": [ {"id":0, "class":"anker_kurz", "conf":0.91,
                     "mask_b64": <base64 PNG, 0/255 single channel>}, ... ] }

GET /health -> {"ok": true}

Env:
  YOLO_WEIGHTS  path to the .pt weights (default /weights/best.pt)
  YOLO_CONF     confidence threshold (default 0.25)
"""
import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

WEIGHTS = os.environ.get("YOLO_WEIGHTS", "/weights/best.pt")
CONF = float(os.environ.get("YOLO_CONF", "0.25"))
CLASS_NAMES = {0: "anker_kurz", 1: "anker_lang"}

app = FastAPI(title="yolo-svc")
MODEL = None


@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = YOLO(WEIGHTS)
    print(f"[yolo-svc] loaded weights: {WEIGHTS} (conf={CONF})", flush=True)


class SegmentReq(BaseModel):
    rgb_b64: str


def _b64_png_to_bgr(b64):
    raw = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _mask_to_b64(mask):
    ok, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/segment")
def segment(req: SegmentReq):
    img = _b64_png_to_bgr(req.rgb_b64)
    h, w = img.shape[:2]
    result = MODEL.predict(img, conf=CONF, verbose=False)[0]

    detections = []
    if result.masks is None or result.boxes is None:
        return {"detections": detections}

    polys = result.masks.xy
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for i, poly in enumerate(polys):
        poly = np.asarray(poly)
        if poly.shape[0] < 3:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.round().astype(np.int32)], 255)
        cls_id = int(classes[i])
        detections.append({
            "id": len(detections),
            "class": CLASS_NAMES.get(cls_id, str(cls_id)),
            "conf": float(confs[i]),
            "mask_b64": _mask_to_b64(mask),
        })

    return {"detections": detections}
