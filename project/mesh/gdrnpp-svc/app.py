"""gdrnpp-svc — the GDRNPP pose micro-service (Pipeline A = combo 1, yolo-obb -> GDRNPP).

GDRNPP is our accuracy main line. This service wraps the per-object GDRNPP
checkpoints (anker_kurz=obj_id 1, anker_lang=obj_id 2; D1 = 2 classes, no zahnrad)
behind the FROZEN mesh /pose contract (CONTRACT.md §3 + §4). Unlike the other pose
services it reads the ORIENTED box `obb` from each instance, not `mask_b64`
(CONTRACT §4: GDRNPP needs the OBB, not a mask).

POST /pose
  {
    "rgb_b64": <base64 PNG uint8 RGB>,
    "K":       [fx,0,cx, 0,fy,cy, 0,0,1],     # flat 9, row-major
    "iterations": 5,                          # accepted, GDRNPP is single-shot (no refiner iters)
    "instances": [
      {"id": 0, "class": "anker_kurz",
       "obb": [cx, cy, w, h, theta],          # PREFERRED: yolo-obb oriented box (theta rad)
       "mask_b64": <base64 PNG 0/255>}        # OPTIONAL fallback: AABB derived from mask bbox
    ]
  }
  -> { "poses": [ {"id":0, "class":"anker_kurz", "T_cam_obj":[[..4x4..]], "score":1.0}, ... ] }
     T_cam_obj = OpenCV cam frame, METRES, mesh->cam (CONTRACT §1).

WHY native venv + stdlib HTTP, NOT Docker (image decision, see report):
  GDRNPP needs detectron2 + the gdrnpp checkout + the trained per-object
  checkpoints, all already wired in the box-native /mnt/data/bop/gdrnpp-venv (the
  same env the live :8078 worker uses). The gdrnpp-venv has no fastapi, so — exactly
  like kip_infer_worker.py — this is a stdlib http.server. A fresh detectron2/GDRNPP
  Docker image would be ~20GB and reproduce a notoriously painful build for zero gain.

DET-DRIVEN, NOT GT-DEPENDENT (CONTRACT/T-115):
  We reuse the live worker's `run_upload(updir, det_json)` path. That path is the
  T-115-safe one: it builds an ISOLATED det-consistent dataset root (symlinked rgb +
  dummy GT) so every DETECTED object gets posed, even ones with no scene_gt backing.
  We never touch the live :8078 worker or any real scene_gt.

  POSE = R_m2c, t_m2c(mm) from GDRNPP. BOP/GDRNPP already emits the OpenCV cam frame
  (model->cam), so T_cam_obj = [[R_m2c, t_m2c/1000],[0,0,0,1]] — direct, no flip, no
  world transform (bop_adapter.py:10-11 confirm R_m2c/t_m2c(mm) convention).

Env:
  GDRN_ROOT     gdrnpp checkout (default /mnt/data/bop/repos/gdrnpp)
  GDRNPP_PORT   listen port (default 8012)
  GDRNPP_MEMFRAC  per-process VRAM cap fraction (default 0.30 — 2 models, separate
                  instance, must coexist with the live :8078 worker's 0.40 cap)
"""
import argparse
import copy
import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

GDRN = os.environ.get("GDRN_ROOT", "/mnt/data/bop/repos/gdrnpp")
PORT = int(os.environ.get("GDRNPP_PORT", "8012"))
MEMFRAC = float(os.environ.get("GDRNPP_MEMFRAC", "0.30"))

sys.path.insert(0, GDRN)
sys.path.insert(0, os.path.join(GDRN, "core/gdrn_modeling"))

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(MEMFRAC, 0)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# D1 scope: 2 Anker classes only. Mesh class name (lowercase) -> BOP obj_id.
# Matches CONTRACT §6 (anker_kurz=1, anker_lang=2). Zahnrad explicitly out-of-scope.
CLASS_TO_OBJ_ID = {"anker_kurz": 1, "anker_lang": 2}
OBJ_ID_TO_CLASS = {v: k for k, v in CLASS_TO_OBJ_ID.items()}
# (slug, obj_id) load order — only the 2 Anker checkpoints (no zahnrad in the mesh).
SVC_OBJS = [("anker_kurz", 1), ("anker_lang", 2)]

STATE = {"ready": False, "error": None, "models": {}}  # obj_id -> dict(cfg,model,slug)


# ── GDRNPP model loading (mirrors kip_infer_worker._load_one, but model-only:
#    we do NOT pre-cache any val-loader inputs — this service is purely det-driven
#    from the request, never from a fixed val split) ──────────────────────────
def _load_one(slug, obj_id):
    from core.gdrn_modeling.main_gdrn import setup
    from core.utils.my_checkpoint import MyCheckpointer

    CFG = f"{GDRN}/configs/gdrn/poseIsaacPbrSO/{slug}.py"
    W = f"{GDRN}/output/gdrn/poseIsaacPbrSO/{slug}/model_best.pth"
    if not os.path.exists(W):
        raise FileNotFoundError(f"model_best.pth missing for {slug}: {W}")
    args = argparse.Namespace(
        config_file=CFG, eval_only=True, num_gpus=1, num_machines=1,
        resume=False, launcher="none", local_rank=0,
        opts={"MODEL.WEIGHTS": W, "MODEL.POSE_NET.INPUT_RES": 320,
              "MODEL.POSE_NET.OUTPUT_RES": 80})
    cfg = setup(args)
    cfg.DATALOADER.NUM_WORKERS = 0
    builder = importlib.import_module(f"core.gdrn_modeling.models.{cfg.MODEL.POSE_NET.NAME}")
    model, _ = builder.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR,
                   prefix_to_remove="_module.").resume_or_load(W, resume=False)
    model.cuda().eval()
    return {"slug": slug, "obj_id": obj_id, "cfg": cfg, "model": model}


def warm():
    try:
        for slug, oid in SVC_OBJS:
            t0 = time.time()
            STATE["models"][oid] = _load_one(slug, oid)
            print(f"[gdrnpp-svc] WARM {slug} (obj_id {oid}) in {time.time()-t0:.1f}s", flush=True)
        STATE["ready"] = True
        print(f"[gdrnpp-svc] READY: {sorted(STATE['models'])}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        STATE["error"] = str(e)


def _forward(m, batch_inputs):
    from core.gdrn_modeling.engine.engine_utils import batch_data
    b = batch_data(m["cfg"], batch_inputs, phase="test")
    out = m["model"](b["roi_img"], roi_classes=b["roi_cls"], roi_cams=b["roi_cam"],
                     roi_whs=b["roi_wh"], roi_centers=b["roi_center"],
                     resize_ratios=b["resize_ratio"], roi_coord_2d=b.get("roi_coord_2d"),
                     roi_coord_2d_rel=b.get("roi_coord_2d_rel"), roi_extents=b.get("roi_extent"))
    return out["rot"].detach().cpu().numpy(), out["trans"].detach().cpu().numpy()


# ── obb / mask -> BOP AABB [x,y,w,h] ────────────────────────────────────────
def _obb_to_aabb(obb):
    """[cx,cy,w,h,theta(rad)] -> enclosing axis-aligned [x,y,w,h] (top-left + wh).
    Mirrors box_src/obb_to_aabb_dets.obb_corners_to_aabb on the 4 rotated corners."""
    cx, cy, w, h, th = [float(v) for v in obb[:5]]
    c, s = np.cos(th), np.sin(th)
    dx, dy = w / 2.0, h / 2.0
    # 4 corners of the rotated rectangle (local -> rotated -> translated)
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    corners = local @ R.T + np.array([cx, cy])
    x0, y0 = corners[:, 0].min(), corners[:, 1].min()
    x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


# Visible-mask bboxes are ~16% tighter than the amodal detector boxes
# (sqrt-area ratio 0.842 median, T-177 probe D) — but compensating with a
# 1.19x expansion MEASURED WORSE: yolo_seg__gdrnpp AR 0.856 (x1.19) vs 0.885
# (x1.0) on the same 30 val frames. GDRNPP's own ROI pipeline (DZI-style
# zoom-out padding) already absorbs the tight-box bias; an extra expansion
# double-pads and shifts the crop statistics away from training. Default
# stays 1.0; the knob remains for future per-model calibration experiments.
MASK_AABB_EXPAND = float(os.environ.get("GDRNPP_MASK_AABB_EXPAND", "1.0"))


def _mask_to_aabb(mask_b64):
    """Fallback when no obb: AABB of the mask's nonzero bbox, expanded by
    MASK_AABB_EXPAND around the center (amodal-ROI calibration, T-177)."""
    import cv2
    raw = base64.b64decode(mask_b64)
    m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    w, h = float(x1 - x0 + 1), float(y1 - y0 + 1)
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    w, h = w * MASK_AABB_EXPAND, h * MASK_AABB_EXPAND
    return [cx - w / 2.0, cy - h / 2.0, w, h]


# ── temp BOP scene + det.json, then GDRNPP det-driven inference ─────────────
def _build_scene(updir, rgb_b64, K, instances_oids):
    """Write a minimal BOP scene (scene 0 / image 0): rgb + scene_camera with the
    request K. We supply NO real scene_gt — the worker's det-consistent-root builder
    (run_upload) injects dummy GT keyed off the detections, so this is purely
    det-driven (T-115-safe). The camera extrinsics are identity: T_cam_obj already
    lives in the cam frame, so we never need world->cam here."""
    import cv2
    sdir = os.path.join(updir, "000000")
    os.makedirs(os.path.join(sdir, "rgb"), exist_ok=True)
    raw = base64.b64decode(rgb_b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    cv2.imwrite(os.path.join(sdir, "rgb", "000000.png"), img)
    # scene_camera: cam_K + identity world<->cam (we stay in the cam frame).
    cam = {"cam_K": [float(x) for x in K],
           "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
           "cam_t_w2c": [0, 0, 0],
           "depth_scale": 1.0}
    json.dump({"0": cam}, open(os.path.join(sdir, "scene_camera.json"), "w"))
    return (h, w)


def _gdrnpp_infer(updir, det_by_key):
    """Run GDRNPP det-driven over the temp scene. det_by_key:
       {"0/0": [{obj_id, bbox_est:[x,y,w,h], score, time}]}.
    Returns ordered preds [{obj_id, R(9), t_mm(3), score}] — order matches the
    per-obj_id, per-detection iteration so the caller can re-attach instance ids.
    Reuses the worker's det-consistent-root + register-dataset machinery exactly."""
    from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
    from core.gdrn_modeling.datasets import pose_isaac_pbr as PIP
    from core.gdrn_modeling.engine.gdrn_evaluator import GDRN_Evaluator
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import kip_infer_worker as KIW  # for _build_det_consistent_root (T-115)

    ds_root = updir
    try:
        r = KIW._build_det_consistent_root(updir, det_by_key)
        if r:
            ds_root = r
    except Exception as e:
        print(f"[gdrnpp-svc] WARN det-root build: {e}", flush=True)

    preds = []
    with torch.no_grad():
        for oid, m in STATE["models"].items():
            filtered = {k: [d for d in ds if d.get("obj_id") == oid]
                        for k, ds in det_by_key.items()}
            filtered = {k: ds for k, ds in filtered.items() if ds}
            if not filtered:
                continue
            slug = m["slug"]
            det_oid = os.path.join(updir, f"det_{slug}.json")
            json.dump(filtered, open(det_oid, "w"))
            ds_name = f"kip_gdrnppsvc_{slug}"
            cfg_up = dict(PIP.SPLITS_POSE_ISAAC[f"pose_isaac_{slug}_val"])
            cfg_up.update(dataset_root=ds_root, with_masks=False, with_depth=False,
                          filter_invalid=False, use_cache=False)
            if ds_name in DatasetCatalog.list():
                DatasetCatalog.remove(ds_name)
                MetadataCatalog.remove(ds_name)
            PIP.register_with_name_cfg(ds_name, cfg_up)
            c2 = copy.deepcopy(m["cfg"])
            c2.DATASETS.TEST = (ds_name,)
            c2.DATASETS.DET_FILES_TEST = (det_oid,)
            c2.MODEL.LOAD_DETS_TEST = True
            c2.TEST.TEST_BBOX_TYPE = "est"
            c2.DATASETS.DET_TOPK_PER_OBJ = 30
            c2.DATALOADER.NUM_WORKERS = 0
            ev = GDRN_Evaluator(c2, ds_name, distributed=False,
                                output_dir=f"/tmp/gdrnppsvc_out_{slug}", train_objs=None)
            try:
                loader = build_gdrn_test_loader(c2, ds_name, train_objs=ev.train_objs)
            except Exception as e:
                print(f"[gdrnpp-svc] WARN loader {slug} (obj {oid}): {e}", flush=True)
                continue
            for inp in loader:
                rots, trans = _forward(m, inp if isinstance(inp, list) else [inp])
                for i in range(len(rots)):
                    preds.append({"obj_id": oid, "R": rots[i].reshape(9).tolist(),
                                  "t_mm": [float(x) * 1000.0 for x in trans[i]],
                                  "score": 1.0})
            torch.cuda.empty_cache()
    return preds


def _T_cam_obj(R9, t_mm):
    """GDRNPP R_m2c(9) + t_m2c(mm) -> 4x4 T_cam_obj (OpenCV cam, METRES, mesh->cam)."""
    R = np.asarray(R9, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_mm, dtype=np.float64) / 1000.0  # mm -> m
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.tolist()


def run_pose(req):
    """Contract /pose. Reads obb (preferred) or mask AABB (fallback) per instance,
    builds a det.json, runs GDRNPP det-driven, maps R/t -> T_cam_obj.

    Skip behaviour (CONTRACT §3): unknown class or no usable box -> instance silently
    skipped, NOT an error; the pose list may be shorter than instances[]."""
    rgb_b64 = req["rgb_b64"]
    K = req["K"]
    instances = req.get("instances", [])

    # Build det.json keyed "0/0". We track (obj_id ordinal) so we can re-attach the
    # original instance id to each pred in iteration order. GDRNPP iterates obj_id
    # outer, detection inner — so we emit dets in the same grouping.
    det_by_key = {"0/0": []}
    # remember (obj_id, original instance id, class) in the order GDRNPP will emit:
    # grouped by obj_id (model load order), then detection order.
    id_track = {oid: [] for _, oid in SVC_OBJS}
    for inst in instances:
        cls = inst.get("class")
        oid = CLASS_TO_OBJ_ID.get(cls)
        if oid is None:
            continue  # unknown / out-of-scope class -> skip
        bbox = None
        if inst.get("obb") is not None:
            bbox = _obb_to_aabb(inst["obb"])
        elif inst.get("mask_b64"):
            bbox = _mask_to_aabb(inst["mask_b64"])
        if bbox is None:
            continue
        det_by_key["0/0"].append({"obj_id": oid, "bbox_est": bbox,
                                  "score": float(inst.get("conf", 1.0)), "time": 0.0})
        id_track[oid].append({"id": inst.get("id"), "class": cls})

    if not det_by_key["0/0"]:
        return {"poses": []}

    with tempfile.TemporaryDirectory(prefix="gdrnppsvc_") as updir:
        _build_scene(updir, rgb_b64, K, det_by_key)
        preds = _gdrnpp_infer(updir, det_by_key)

    # Re-attach instance ids. preds come grouped by obj_id (model load order),
    # detection order within. Walk id_track in the same obj_id load order.
    poses = []
    cursor = {oid: 0 for _, oid in SVC_OBJS}
    for p in preds:
        oid = p["obj_id"]
        track = id_track.get(oid, [])
        idx = cursor[oid]
        meta = track[idx] if idx < len(track) else {"id": None, "class": OBJ_ID_TO_CLASS.get(oid)}
        cursor[oid] = idx + 1
        poses.append({"id": meta["id"], "class": meta["class"],
                      "T_cam_obj": _T_cam_obj(p["R"], p["t_mm"]),
                      "score": p.get("score", 1.0)})
    return {"poses": poses}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode()) if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            # CONTRACT §3 health: report load state for the VRAM-lifecycle manager.
            state = "ready" if STATE["ready"] else ("error" if STATE["error"] else "loading")
            self._send({"ok": STATE["ready"], "state": state, "error": STATE["error"],
                        "classes": sorted(CLASS_TO_OBJ_ID),
                        "loaded_obj_ids": sorted(STATE["models"]),
                        "cuda": torch.cuda.is_available()})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/pose":
            if not STATE["ready"]:
                self._send({"error": STATE["error"] or "not ready", "poses": None}, 503)
                return
            try:
                req = self._read_json()
                self._send(run_pose(req))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send({"error": str(e), "poses": None}, 500)
        else:
            self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    threading.Thread(target=warm, daemon=True).start()
    print(f"[gdrnpp-svc] listening on 0.0.0.0:{PORT} (memfrac={MEMFRAC})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
