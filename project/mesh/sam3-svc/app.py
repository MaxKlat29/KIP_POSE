"""
SAM3 promptable-concept segmentation micro-service. Runs INSIDE the
foundationpose:blackwell-sam3 image (torch cu128 + transformers>=5) and serves
the same /segment contract as yolo-svc, so the gateway can treat it as a
drop-in alternative segmentation source.

SAM3 performs Promptable Concept Segmentation: a short noun phrase returns
masks for EVERY instance matching the concept. The pose stage downstream needs
a known class per instance (class -> CAD mesh), so this service runs one
concept prompt per configured class and labels the resulting detections with
that class. Instances matched by more than one prompt are deduplicated with a
cross-class mask-IoU NMS, keeping the higher-scoring class.

KNOWN LIMITATION (measured on the bundled sample scene): the text prompts
find armature instances reliably, but do NOT separate anker_kurz from
anker_lang - both prompts match every armature and one consistently outscores
the other, so all detections come back as one class. 2D mask geometry cannot
fix this either (random 3D poses make pixel lengths overlap completely:
kurz 38-161 px vs lang 18-181 px on the sample). Distinguishing the two
classes needs a trained classifier - that is YOLO's job. Use SAM3 here for
training-free, class-agnostic instance masks; treat its class labels as
approximate.

UPGRADE PATH (validated offline in the 2026-06-05 SAM3 spike, not yet
implemented here): two-pass self-prompting - (1) generic text prompt
("small metal object"), (2) back-project each candidate mask through metric
depth, keep 3D PCA lengths in [85, 180] mm (CAD-size filter), (3) re-prompt
with the best survivor's bounding box as an exemplar. Measured 90.5% recall
/ 92.7% precision / 0.858 mIoU on poc500, beating finetuned YOLO26n-seg.
Class identity: banded classifier on depth-PCA FULL extent along PC1
(<128 mm -> kurz, >133 mm -> lang; ambiguous band -> run FoundationPose with
both meshes, keep the better fit). Requires depth_b64 + K in /segment, which
the gateway already has at predict time. Use full extent, NOT
percentile-trimmed (trimming amplifies occlusion truncation).

POST /segment
  { "rgb_b64": <base64 PNG, uint8 RGB>,
    "prompts": {class: concept prompt, ...} }   # optional per-request override,
                                                # merged over the env defaults
->{ "detections": [ {"id":0, "class":"anker_kurz", "conf":0.31,
                     "mask_b64": <base64 PNG, 0/255 single channel>}, ... ] }

GET /health -> {"ok": true}

Env:
  SAM3_MODEL      HF model id (default facebook/sam3; gated - the weights must
                  already be in the mounted HF cache, see docker-compose.yml)
  SAM3_PROMPTS    JSON {class: concept prompt} (default both KIP classes)
  SAM3_CONF       presence threshold (default 0.2 - SAM3 scores run low on
                  out-of-domain images; tune per scene, NOT the 0.5 docs value)
  SAM3_MASK_THR   mask binarization threshold (default 0.5)
  SAM3_NMS_IOU    cross-class dedup IoU (default 0.5)
"""
import base64
import json
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.environ.get("SAM3_MODEL", "facebook/sam3")
# or-fallback (not .get default): compose passes SAM3_PROMPTS as "" when unset.
PROMPTS = json.loads(os.environ.get("SAM3_PROMPTS") or
                     '{"anker_kurz": "short metal motor armature part",'
                     ' "anker_lang": "long metal motor armature part"}')
CONF = float(os.environ.get("SAM3_CONF", "0.2"))
MASK_THR = float(os.environ.get("SAM3_MASK_THR", "0.5"))
NMS_IOU = float(os.environ.get("SAM3_NMS_IOU", "0.5"))

app = FastAPI(title="sam3-svc")
MODEL = None
PROCESSOR = None


@app.on_event("startup")
def load_model():
    global MODEL, PROCESSOR
    from transformers import Sam3Model, Sam3Processor
    MODEL = Sam3Model.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    MODEL.eval()
    PROCESSOR = Sam3Processor.from_pretrained(MODEL_ID)
    print(f"[sam3-svc] loaded {MODEL_ID} (conf={CONF}, prompts={PROMPTS})",
          flush=True)


class SegmentReq(BaseModel):
    rgb_b64: str
    # Optional per-request {class: concept prompt} override. Merged over the
    # SAM3_PROMPTS env defaults per class; blank values fall back to defaults.
    prompts: dict[str, str] | None = None


def _b64_png_to_pil(b64):
    raw = base64.b64decode(b64)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    # PIL only: Sam3Processor misreads HWC ndarrays (shape[-2:] is not W,H).
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _mask_to_b64(mask):
    ok, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def _cross_class_nms(dets):
    """Drop lower-scoring detections whose mask overlaps a kept one (any class).
    SAM3 runs one pass per class prompt, so the same physical object can match
    both prompts; downstream must see it exactly once."""
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in dets:
        if all(_mask_iou(d["bool_mask"], k["bool_mask"]) < NMS_IOU for k in kept):
            kept.append(d)
    return kept


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/segment")
def segment(req: SegmentReq):
    img = _b64_png_to_pil(req.rgb_b64)
    w, h = img.size

    overrides = {k: v.strip() for k, v in (req.prompts or {}).items() if v.strip()}
    prompts = {**PROMPTS, **overrides}
    classes = list(prompts.keys())
    # One batch entry per class prompt: SAM3 segments every instance of one
    # concept per forward pass; batching the prompts shares the image encoder.
    inputs = PROCESSOR(
        images=[img] * len(classes),
        text=[prompts[c] for c in classes],
        return_tensors="pt",
    ).to(MODEL.device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = MODEL(**inputs)

    results = PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=CONF,
        mask_threshold=MASK_THR,
        target_sizes=inputs.get("original_sizes").tolist(),
    )

    raw = []
    for cls, res in zip(classes, results):
        masks = res["masks"]
        scores = res["scores"]
        for i in range(len(scores)):
            m = masks[i]
            bool_mask = (m.cpu().numpy() if torch.is_tensor(m) else
                         np.asarray(m)).astype(bool).reshape(h, w)
            if not bool_mask.any():
                continue
            raw.append({
                "class": cls,
                "conf": float(scores[i]),
                "bool_mask": bool_mask,
            })

    detections = []
    for d in _cross_class_nms(raw):
        detections.append({
            "id": len(detections),
            "class": d["class"],
            "conf": d["conf"],
            "mask_b64": _mask_to_b64(d["bool_mask"].astype(np.uint8) * 255),
        })
    return {"detections": detections}
