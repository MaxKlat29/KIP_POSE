"""
GigaPose pose-estimation microservice.

Wraps the real GigaPose coarse matcher + MegaPose ("GenFlow") multi-hypothesis
refiner + an optional ICP/Kabsch depth-alignment step, via the in-repo adapter
`gigapose_infer.GigaPoseInfer` (lives in the mounted GigaPose repo). One warmed
model serves both pipelines; the request `pipeline` field ('rgbd' | 'rgb') gates
the depth/Kabsch tail.

Env:
  GP_REPO=/workspace/GigaPose         GigaPose checkout (on PYTHONPATH)
  GP_DATASET=kip2                     onboarded template dataset
  GP_ENABLE_REFINER=1                 1 = coarse+GenFlow(+Kabsch); 0 = coarse(+Kabsch) only
  (also: TORCH_HOME, XFORMERS_DISABLED=1, CONDA_PREFIX, CUDA_VISIBLE_DEVICES — set in the image)
"""
import base64
import io
import os
import sys
import time
import traceback
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

GP_REPO = os.environ.get("GP_REPO", "/workspace/GigaPose")
GP_DATASET = os.environ.get("GP_DATASET", "kip2")
sys.path.insert(0, GP_REPO)

app = FastAPI(title="GigaPose Service", version="2.0")

_infer = None          # GigaPoseInfer singleton (warmed model+templates+refiner)
_load_error = None     # str if model construction failed


def _build_infer():
    global _infer, _load_error
    if _infer is not None or _load_error is not None:
        return
    try:
        from gigapose_infer import GigaPoseInfer  # in the mounted GigaPose repo
        enable_refiner = os.environ.get("GP_ENABLE_REFINER", "1") == "1"
        t0 = time.time()
        _infer = GigaPoseInfer(dataset_name=GP_DATASET, enable_refiner=enable_refiner)
        print(
            f"[gigapose-svc] model ready in {time.time()-t0:.1f}s "
            f"(refiner={'on' if _infer.pose_estimator is not None else 'off'})",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        _load_error = f"{e}\n{traceback.format_exc()}"
        print(f"[gigapose-svc] model load FAILED: {_load_error}", flush=True)


@app.on_event("startup")
def _startup():
    _build_infer()


# ---------------------------------------------------------------- decode utils
def _decode_rgb(b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)


def _decode_depth(b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.array(img).astype(np.float32) / 1000.0  # uint16 mm -> metres


def _decode_mask(b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


# ----------------------------------------------------------------- I/O schemas
class PoseReq(BaseModel):
    rgb_b64: str
    depth_b64: Optional[str] = None   # required for pipeline=='rgbd', null for 'rgb'
    K: list[float]                    # flat 9, row-major
    iterations: int = 5              # refiner n_iterations
    hypotheses: int = 5              # coarse top-K carried into refinement
    pipeline: str = "rgbd"           # 'rgbd' | 'rgb' — single depth/Kabsch gate
    instances: list[dict]            # each: {id, class|class_name, mask_b64}


class PoseResp(BaseModel):
    poses: list[dict]


# ----------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {
        "status": "ok" if _infer is not None else ("error" if _load_error else "loading"),
        "dataset": GP_DATASET,
        "refiner": (_infer.pose_estimator is not None) if _infer is not None else None,
        "refiner_error": _infer.refiner_error if _infer is not None else None,
        "load_error": _load_error.splitlines()[-1] if _load_error else None,
    }


@app.post("/pose", response_model=PoseResp)
def pose(req: PoseReq):
    if _infer is None:
        # try a (re)build in case startup raced; otherwise report
        _build_infer()
        if _infer is None:
            raise HTTPException(status_code=503, detail=f"model not loaded: {(_load_error or '').splitlines()[-1:] }")

    pipeline = (req.pipeline or "rgbd").lower()
    if pipeline not in ("rgbd", "rgb"):
        raise HTTPException(status_code=400, detail=f"unknown pipeline: {req.pipeline!r}")

    try:
        rgb = _decode_rgb(req.rgb_b64)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad rgb: {e}")

    depth = None
    if pipeline == "rgbd":
        if not req.depth_b64:
            raise HTTPException(status_code=400, detail="depth_b64 required for pipeline='rgbd'")
        try:
            depth = _decode_depth(req.depth_b64)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad depth: {e}")

    kabsch = pipeline == "rgbd" and depth is not None
    K = np.array(req.K, dtype=np.float64).reshape(3, 3)

    poses = []
    for inst in req.instances:
        cls = inst.get("class") or inst.get("class_name")
        inst_id = inst.get("id")
        mask_b64 = inst.get("mask_b64")
        if cls is None or mask_b64 is None:
            continue
        try:
            mask = _decode_mask(mask_b64)
        except Exception:
            continue
        try:
            T, score, stage = _infer.estimate(
                class_name=cls, K=K, rgb=rgb, mask=mask,
                depth=depth if kabsch else None,
                iterations=req.iterations, hypotheses=req.hypotheses, kabsch=kabsch,
            )
        except ValueError:
            # unknown class / empty mask -> skip (like fp-svc)
            continue
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"pose failed for {cls}: {e}")
        poses.append({
            "id": inst_id,
            "class": cls,
            "T_cam_obj": np.asarray(T, dtype=np.float64).reshape(4, 4).tolist(),
            "score": float(score),
            "stage": stage,
        })
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {"poses": poses}
