"""
FoundationPose micro-service. Runs INSIDE the foundationpose:blackwell image
(torch cu128 + pytorch3d + nvdiffrast + FoundationPose, sm_120). Loads one
estimator per CAD mesh at startup, then poses each supplied instance mask.

POST /pose
  {
    "rgb_b64":   <base64 PNG, uint8 RGB>,
    "depth_b64": <base64 PNG, uint16>,
    "depth_scale": 1.0,   # uint16 -> millimetres factor; metres = png*depth_scale/1000
    "K":         [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "iterations": 5,
    "instances": [ {"id": 0, "class": "anker_kurz", "mask_b64": <base64 PNG, 0/255>}, ... ]
  }
->{ "poses": [ {"id":0, "class":"anker_kurz", "T_cam_obj":[[..4x4..]]}, ... ] }   # OpenCV cam frame, metres

Env:
  FP_REPO     path to the FoundationPose checkout (default /workspace/FoundationPose)
  FP_MESH_DIR dir with <class>.obj meshes (metres)
"""
import base64
import io
import os

import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

FP_REPO = os.environ.get("FP_REPO", "/workspace/FoundationPose")
MESH_DIR = os.environ.get("FP_MESH_DIR", "/assets/meshes")
# zahnrad seit T-178e: FoundationPose ist mesh-basiert zero-shot — das Zahnrad-
# Mesh (aus BOP obj_000006, mm->m konvertiert) reicht, kein Training noetig.
CLASS_TO_MESH = {"anker_kurz": "anker_kurz.obj", "anker_lang": "anker_lang.obj",
                 "zahnrad": "zahnrad.obj"}

import sys
sys.path.insert(0, FP_REPO)
import trimesh
import nvdiffrast.torch as dr
import torch
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

app = FastAPI(title="foundationpose-svc")
ESTIMATORS, MESHES = {}, {}


def _png_b64_to_array(b64, flags):
    raw = base64.b64decode(b64)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    return arr


@app.on_event("startup")
def load_models():
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    for cls, fn in CLASS_TO_MESH.items():
        path = os.path.join(MESH_DIR, fn)
        m = trimesh.load(path)
        MESHES[cls] = m
        ESTIMATORS[cls] = FoundationPose(
            model_pts=m.vertices.astype(np.float32),
            model_normals=m.vertex_normals.astype(np.float32),
            mesh=m, scorer=scorer, refiner=refiner, glctx=glctx,
            debug=0, debug_dir="/tmp/fp_debug")
    print(f"[fp-svc] loaded estimators: {list(ESTIMATORS)}", flush=True)


class PoseReq(BaseModel):
    rgb_b64: str
    depth_b64: str
    # uint16-depth -> millimetres factor. Real mm-depth sensors (Zivid/Realsense)
    # use 1.0 (= the historical default). BOP/SDG depth PNGs are written with a
    # depth_scale (e.g. 0.1: png*0.1 = mm) and MUST forward it or the back-projected
    # cloud is off by exactly that factor (T-156: 0.1 dropped -> depth 10x -> ~2.4m X).
    depth_scale: float = 1.0
    K: list[float]
    iterations: int = 5
    instances: list[dict]


@app.get("/health")
def health():
    return {"ok": True, "classes": list(ESTIMATORS), "cuda": torch.cuda.is_available()}


@app.post("/pose")
def pose(req: PoseReq):
    rgb = cv2.cvtColor(_png_b64_to_array(req.rgb_b64, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # png(uint16) * depth_scale = millimetres; /1000 -> metres. depth_scale default
    # 1.0 keeps the mm-sensor path byte-identical; BOP/SDG frames pass 0.1 (T-156).
    depth = _png_b64_to_array(req.depth_b64, cv2.IMREAD_UNCHANGED).astype(np.float32) * req.depth_scale / 1000.0
    K = np.array(req.K, dtype=np.float64).reshape(3, 3)
    out = []
    for inst in req.instances:
        cls = inst["class"]
        if cls not in ESTIMATORS:
            continue
        mask = _png_b64_to_array(inst["mask_b64"], cv2.IMREAD_GRAYSCALE) > 0
        T = ESTIMATORS[cls].register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                     iteration=req.iterations)
        out.append({"id": inst.get("id"), "class": cls,
                    "T_cam_obj": np.asarray(T).reshape(4, 4).tolist()})
        torch.cuda.empty_cache()
    return {"poses": out}
