#!/usr/bin/env python3
"""kip_server.py — FastAPI-Backend für den KIP-Pose-Web-Viewer.

Läuft ON-DEMAND auf der GPU-Workstation (100.85.216.95). Serviert das
2-Screen-Frontend (frontend/) und stellt die On-Demand-Pipeline als API bereit:

  Screen 1 "Real"  — Foto-Upload  -> Detektor + GDRNPP -> pose_result -> 3D-Render
  Screen 2 "Sim"   — live Isaac-SDG -> label -> infer -> GT(blau) vs Pred(rot)

Path-aware: hinter Caddy auf max-utils.com/KIP gemountet (KEIN path-strip dort),
deshalb root_path=$KIP_BASE_PATH (default "/KIP"). Lokal/direkt: leer lassen.

Architektur (siehe Brain projects/max-utils + patterns/gpu-on-demand-inference):
  Browser -> CF-Tunnel -> cloudflared(ai-desk) -> Caddy @kip
          -> reverse_proxy 100.85.216.95:$KIP_PORT  (Tailscale)
          -> DIESER Server (root_path=/KIP)

Start (auf der Workstation):
  KIP_BASE_PATH=/KIP KIP_PORT=8077 \
    /mnt/data/<venv>/bin/python -m uvicorn kip_server:app --host 0.0.0.0 --port 8077

GPU-Hinweis: Real-Infer (Detektor+GDRNPP) ist leicht und teilt sich die GPU
gefahrlos mit einem laufenden Training. Live-SDG (Isaac Sim) ist GPU-schwer —
der /api/sim/generate-Endpoint prüft per GUARD ob ein Training läuft und
verweigert dann (override via force=true), um das Training nicht zu crashen.
"""
from __future__ import annotations
import os, json, time, pathlib, subprocess, shutil, uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Pfade / Config ──────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent          # project/
FRONTEND   = HERE / "frontend"
TEMP       = HERE / "temp"
UPLOADS    = TEMP / "uploads"
RENDERS    = TEMP / "kip_renders"
BASE_PATH  = os.environ.get("KIP_BASE_PATH", "/KIP").rstrip("/")   # "" = root
PORT       = int(os.environ.get("KIP_PORT", "8077"))

# Workstation-spezifische Pfade (nur dort vorhanden; lokal -> None-tolerant).
GDRNPP_OUT = pathlib.Path("/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO")
VAL_ROOT   = pathlib.Path("/mnt/data/kip_pose/project/bop/pose_isaac/val")
SELECT_BEST = {  # obj-slug -> select_best.json (Eval-Kennzahlen pro Modell)
    "anker_kurz": GDRNPP_OUT / "anker_kurz" / "select_best.json",
    "anker_lang": GDRNPP_OUT / "anker_lang" / "select_best.json",
    "zahnrad":    GDRNPP_OUT / "zahnrad"    / "select_best.json",
}
# IC-BIN AR (official BOP19/IC-BIN localisation protocol, multi-instance
# bin-picking matching via bop_toolkit pose_matching+score; see box_src/eval_bop.py
# --icbin and T-109). This is the MAINTAINED, displayed metric. select_best.json's
# best_full_ar (translation-match AR) is kept as fallback only.
IC_BIN = {
    "anker_kurz": GDRNPP_OUT / "anker_kurz" / "ic_bin.json",
    "anker_lang": GDRNPP_OUT / "anker_lang" / "ic_bin.json",
    "zahnrad":    GDRNPP_OUT / "zahnrad"    / "ic_bin.json",
}
TRAINED_OBJS = {"anker_kurz"}   # wird aus vorhandenen preds_best.csv abgeleitet (s.u.)

for d in (TEMP, UPLOADS, RENDERS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KIP Pose Viewer", root_path=BASE_PATH)

# Job-Tracking fuer async-Inferenz mit Phasen-Fortschritt (Real + Sim).
# Frontend pollt /api/{real,sim}/job/<id> ~3x/s und zeigt phase + pct.
import threading
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()

def _job_set(job, **kv):
    with _JOBS_LOCK:
        _JOBS.setdefault(job, {}).update(kv)

def _job_get(job):
    with _JOBS_LOCK:
        return dict(_JOBS.get(job, {}))


# ── Helpers ─────────────────────────────────────────────────────
def _gpu_busy_with_training() -> bool:
    """True wenn echt ein GDRNPP-Trainings-Prozess läuft.

    Sucht gezielt nach `phase2_chain.sh` als Skript-Aufruf, NICHT als String in
    irgendeiner Befehlszeile. Sonst matched sich ein einfacher Monitor-Loop
    selbst (`bash -c '... pgrep phase2_chain ... sleep 600'`) und das Badge
    bleibt für immer "Training laeuft" haengen.
    """
    try:
        out = subprocess.run(["pgrep", "-af", r"\.sh phase2_chain|phase2_chain\.sh"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            # echte Aufrufe: `bash phase2_chain.sh ...` ODER `/.../phase2_chain.sh ...`
            # NICHT: `bash -c '... phase2_chain ...'` (Monitor-Loops mit pgrep darin).
            if "pgrep" in line or "while true" in line or "sleep " in line:
                continue
            if "phase2_chain.sh" in line:
                return True
        return False
    except Exception:
        return False


def _discover_trained() -> set[str]:
    """welche Objekte haben ein fertiges Modell (preds_best.csv vorhanden)?"""
    found = set()
    for slug in SELECT_BEST:
        if (GDRNPP_OUT / slug / "preds_best.csv").exists():
            found.add(slug)
    return found or set(TRAINED_OBJS)


# ── Pipeline-Vergleich-Seam (additiv; bricht den Boot NIE) ──────
# Erlaubt den Vergleich komplett anderer Pose-Pipelines (siehe pipelines/ +
# docs/PIPELINE_INTEGRATION.md). Aktuell nur die gdrnpp-Referenz live; fremde
# Pipelines werden als Adapter unter pipelines/<id>/ angebunden.
try:
    from pipelines import registry as _pipe_registry
except Exception:  # noqa: BLE001 — Seam ist optional, darf den Server nie crashen
    _pipe_registry = None


def _resolve_pipeline(pipeline: str) -> str:
    """Validiert den optionalen pipeline-Param. Default 'gdrnpp' = UNVERAENDERTER
    Live-Pfad. Fremde, noch nicht angebundene Pipelines -> 501 (kein stiller Fallback)."""
    pid = (pipeline or "gdrnpp").strip()
    if pid == "gdrnpp":
        return pid
    if _pipe_registry is None:
        raise HTTPException(501, "Pipeline-Vergleich nicht verfuegbar (pipelines-Paket fehlt).")
    try:
        ad = _pipe_registry.get(pid)
    except KeyError:
        raise HTTPException(404, f"Unbekannte Pipeline '{pid}'.")
    if not ad.available:
        raise HTTPException(501, f"Pipeline '{pid}' ist noch nicht angebunden (Scaffold).")
    # Routing in einen fremden Adapter wird verdrahtet, sobald welche geliefert sind.
    raise HTTPException(501, f"Routing fuer Pipeline '{pid}' folgt — bisher nur gdrnpp live.")


# ── Static frontend (gemountet zuletzt, damit /api Vorrang hat) ──
# /api/* zuerst definieren, dann StaticFiles auf "/" als Catch-all.

@app.get("/api/health")
def health():
    trained = sorted(_discover_trained())
    return {
        "status": "ok",
        "service": "kip-pose-viewer",
        "base_path": BASE_PATH or "/",
        "gpu_training_active": _gpu_busy_with_training(),
        "trained_objects": trained,
        "frontend_present": FRONTEND.exists(),
        "ts": time.strftime("%FT%TZ", time.gmtime()),
    }


@app.get("/api/metrics")
def metrics():
    """Eval-Kennzahlen pro Objekt.

    Maßgebliche + angezeigte Metrik: ``ic_bin_ar`` = AR nach dem offiziellen
    BOP19/IC-BIN-Localisation-Protokoll (Multi-Instanz-/Bin-Picking-Matching via
    bop_toolkit pose_matching+score, siehe box_src/eval_bop.py --icbin, T-109).
    ``best_full_ar`` (Translation-Match-AR aus select_best.json) bleibt als
    Fallback erhalten, falls ic_bin.json (noch) fehlt — die Werte sind auf dem
    val-Split ohnehin praktisch identisch (Δ < 0.001)."""
    out = {}
    for slug, p in SELECT_BEST.items():
        entry = {"status": "training_or_pending"}
        if p.exists():
            try:
                sel = json.load(open(p))["selection"][0]
                entry = {
                    "best_ckpt": sel.get("best_ckpt"),
                    "best_full_ar": sel.get("best_full_ar"),
                    "best_is_final": sel.get("best_is_final"),
                    "n_checkpoints": sel.get("n_checkpoints"),
                    "status": "trained",
                }
            except Exception as e:
                entry = {"status": "error", "detail": str(e)}
        # IC-BIN AR überlagern (bevorzugt). Fällt auf best_full_ar zurück.
        icp = IC_BIN.get(slug)
        ic_ar = None
        if icp and icp.exists():
            try:
                icj = json.load(open(icp))
                ic_ar = icj.get("ic_bin_ar")
                entry["ic_bin_ar"] = ic_ar
                entry["ar_protocol"] = icj.get("protocol", "ic_bin_bop19_localisation")
            except Exception:
                pass
        if ic_ar is None:
            # noch kein IC-BIN-Eval -> Translation-Match-AR als IC-BIN-Anzeigewert
            ic_ar = entry.get("best_full_ar")
            if ic_ar is not None:
                entry["ic_bin_ar"] = ic_ar
                entry["ar_protocol"] = "fallback_translation_match"
        # ``ar`` ist der EINE maßgebliche Anzeigewert (IC-BIN, sonst Fallback).
        entry["ar"] = ic_ar
        out[slug] = entry
    return {"objects": out, "trained": sorted(_discover_trained()),
            "metric": "ar_ic_bin"}


@app.get("/api/pipelines")
def pipelines():
    """Registrierte Pose-Pipelines fuer den Vergleich (Dropdown + Harness).

    Fallback wenn das Seam-Paket fehlt: nur die gdrnpp-Referenz. Der Web-Viewer
    befuellt das Modell-Dropdown hieraus (verfuegbar=enabled, sonst disabled)."""
    if _pipe_registry is None:
        return {"seam": "unavailable",
                "pipelines": [{"id": "gdrnpp", "name": "GDRNPP (RGB)",
                               "description": "Referenz/Baseline", "available": True}]}
    return {"seam": "ok", "pipelines": _pipe_registry.all_pipelines()}


@app.get("/api/compare")
def compare(scene: int = 0, im: int = -1, pipelines: str = ""):
    """STUB: Multi-Pipeline-Side-by-Side. Aktiv sobald >=2 Pipelines verfuegbar sind
    (siehe docs/PIPELINE_INTEGRATION.md + compare_pipelines.py)."""
    avail = _pipe_registry.available_ids() if _pipe_registry else ["gdrnpp"]
    if len(avail) < 2:
        raise HTTPException(501, f"Vergleich braucht >=2 verfuegbare Pipelines; aktuell verfuegbar: "
                                 f"{avail}. Fremde Pipelines werden via pipelines/<id>/ angebunden.")
    raise HTTPException(501, "Side-by-Side-Verdrahtung folgt (siehe docs/PIPELINE_INTEGRATION.md).")


# ── Live-Tab: on-demand Proxy zum Jetson-Zellen-Controller ──────────────────
# Die Workstation erreicht den Jetson (172.22.192.166, KIT-wbk) on-demand ueber
# SCOPED KIT-VPN — NUR Route zum Jetson, KEIN Full-Tunnel (box_src/kit_vpn_scoped_connect.sh).
# Auf dem Jetson laeuft on-demand (von Max gestartet, NICHT von uns deployt)
# jetson_live/live_server.py. Solange das nicht laeuft -> saubere 503 (kein Crash).
LIVE_JETSON = os.environ.get("LIVE_JETSON_URL", "http://172.22.192.166:8090").rstrip("/")


def _live_fetch(path, method="GET", timeout=8):
    import urllib.request
    req = urllib.request.Request(f"{LIVE_JETSON}{path}", method=method)
    r = urllib.request.urlopen(req, timeout=timeout)
    return r.status, r.headers.get_content_type(), r.read()


@app.get("/api/live/status")
def live_status():
    """Erreichbarkeit des Jetson-live_server (on-demand-Check, kein Dauerzustand)."""
    try:
        _st, _ct, body = _live_fetch("/health", timeout=5)
        d = json.loads(body.decode())
        return {"reachable": True, "jetson": LIVE_JETSON,
                "camera_connected": d.get("camera_connected"), "detail": d}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={
            "reachable": False, "jetson": LIVE_JETSON,
            "hint": "Jetson-live_server nicht erreichbar. Workstation-VPN aktiv "
                    "(box_src/kit_vpn_scoped_connect.sh)? live_server auf dem Jetson gestartet? Kamera dran?",
            "detail": str(e)})


@app.get("/api/live/preview")
def live_preview():
    """Proxyt EIN aktuelles Vorschau-Frame (JPEG) vom Jetson. FE pollt on-demand."""
    import io as _io
    try:
        _st, ct, body = _live_fetch("/preview", timeout=10)
        return StreamingResponse(_io.BytesIO(body), media_type=ct or "image/jpeg")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Live-Vorschau nicht verfuegbar: {e}")


@app.post("/api/live/capture_infer")
def live_capture_infer():
    """Triggert on-demand Capture+Inferenz auf dem Jetson, reicht das Ergebnis durch."""
    try:
        _st, _ct, body = _live_fetch("/capture_infer", method="POST", timeout=120)
        return JSONResponse(json.loads(body.decode()))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Capture+Inferenz fehlgeschlagen (Jetson erreichbar? Kamera dran?): {e}")


@app.get("/api/live/frame/{name}")
def live_frame(name: str):
    """Proxyt ein vom Jetson nach /capture_infer geliefertes Ergebnis-Bild (image_url)."""
    import io as _io, re as _re
    if not _re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise HTTPException(400, "ungueltiger Name")
    try:
        _st, ct, body = _live_fetch(f"/frame/{name}", timeout=10)
        return StreamingResponse(_io.BytesIO(body), media_type=ct or "image/png")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Frame nicht verfuegbar: {e}")


@app.get("/api/sim/scenes")
def sim_scenes():
    """Liste der verfügbaren (gelabelten) Val-Szenen für den Sim-Screen-Pool."""
    scenes = []
    if VAL_ROOT.exists():
        for d in sorted(VAL_ROOT.glob("[0-9]*")):
            if (d / "scene_gt.json").exists():
                imgs = sorted((d / "rgb").glob("*.png")) if (d / "rgb").exists() else []
                scenes.append({"scene": d.name, "n_images": len(imgs)})
    return {"scenes": scenes, "val_root_present": VAL_ROOT.exists()}


_TRAIN_VENV = "/mnt/data/train-venv/bin/python"
_DETECTOR = "/mnt/data/kip_pose/data/detector_armvis/detector.pt"
_BOX_SRC = "/mnt/data/kip_pose/box_src"

def _zivid_cam():
    """Feste Zivid-Kamera (K + world-Extrinsics) — identisch fuer alle Bilder
    (fix montiert). Aus val/000000 als Template."""
    return json.load(open(VAL_ROOT / "000000" / "scene_camera.json"))["0"]


def _run_detector(img_path, out_json):
    """YOLOv8-OBB-Detektor (train-venv subprocess) -> det-json im load_detections-
    Format: dict {scene_im_id: [{obj_id, bbox_est(xywh), score, time}]}.
    Filtert auf trainierte Objekte (anker_kurz=1, anker_lang=2, zahnrad=6).

    conf=0.1 (T-094-Sweep, val 4043 GT): senkt die Detektor-Confidence von 0.3
    -> 0.1 fuer hoeheren Recall unter Occlusion (rec_occ 0.971 -> 0.979) bei
    nur -0.5pp Precision (0.920 -> 0.915, kein Einbruch). imgsz/NMS-iou bleiben
    (1280 / ult-default 0.7) — die waren bereits optimal. Detektor ist NICHT der
    Pipeline-Cap (GDRNPP-Pose-AR ~0.87 ist es); dies ist eine kleine, risikoarme
    Recall-Verbesserung, kein Retrain."""
    trained_csv = ",".join(str(_OID[s]) for s in _discover_trained() if s in _OID) or "1"
    code = (
        "import sys,json; sys.path.insert(0,'%s')\n"
        "from obb_to_aabb_dets import run_detector_to_bop\n"
        "trained={%s}\n"
        "d=run_detector_to_bop('%s',['%s'],[(0,0)],conf=0.1,imgsz=1280,device=0)\n"
        "d=[x for x in d if x.get('category_id') in trained]\n"
        "out={}\n"
        "for x in d:\n"
        "    k=str(x['scene_id'])+'/'+str(x['image_id'])\n"
        "    out.setdefault(k,[]).append({'obj_id':x['category_id'],'bbox_est':x['bbox'],'score':x['score'],'time':x.get('time',0.0)})\n"
        "json.dump(out,open('%s','w')); print(len(d))"
    ) % (_BOX_SRC, trained_csv, _DETECTOR, img_path, out_json)
    r = subprocess.run([_TRAIN_VENV, "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"Detektor fehlgeschlagen: {r.stderr[-300:]}")
    return int(r.stdout.strip().splitlines()[-1])


@app.post("/api/real/infer")
async def real_infer(image: UploadFile = File(...),
                     tta: bool = Form(False),
                     refine_rc: bool = Form(False)):
    """Foto-Upload (Zivid-Format) -> Detektor + warmer GDRNPP-Worker -> pose_result.

    Baut ein temp-Dataset (maskenfrei), laesst den Detektor laufen, ruft den
    warmen Worker /infer_upload, transformiert via bop_pose_to_world + Snap.
    """
    import numpy as np
    job = uuid.uuid4().hex[:8]
    updir = UPLOADS / f"real_{job}"
    (updir / "000000" / "rgb").mkdir(parents=True, exist_ok=True)
    src = updir / "000000" / "rgb" / "000000.png"
    with open(src, "wb") as f:
        shutil.copyfileobj(image.file, f)
    # temp-Dataset: scene_camera (feste Zivid) + dummy gt/info (maskenfrei)
    cam = _zivid_cam()
    json.dump({"0": cam}, open(updir / "000000" / "scene_camera.json", "w"))
    # Dummy-GT pro trainiertes Objekt (+ pro Instanz eine Dummy-Maske). Jedes
    # single-obj-Dataset (kip_upload_<slug>) liest nur seine obj_id, die anderen
    # werden ignoriert. load_detections ersetzt die Instanzen mit den echten
    # Detektor-Boxen. Ohne >=1 GT pro Dataset -> "Dataset is empty".
    trained_oids = sorted(_trained_oids()) or [1]
    dummy_gt = [{"obj_id": oid, "cam_R_m2c": [1,0,0,0,1,0,0,0,1], "cam_t_m2c": [0,0,1000]}
                for oid in trained_oids]
    dummy_gi = [{"bbox_visib": [0,0,40,40], "bbox_obj": [0,0,40,40],
                 "visib_fract": 1.0, "px_count_visib": 1600} for _ in trained_oids]
    json.dump({"0": dummy_gt}, open(updir / "000000" / "scene_gt.json", "w"))
    json.dump({"0": dummy_gi}, open(updir / "000000" / "scene_gt_info.json", "w"))
    from PIL import Image as _Img, ImageDraw as _Draw
    for sub in ("mask", "mask_visib"):
        (updir / "000000" / sub).mkdir(exist_ok=True)
        for i in range(len(trained_oids)):
            m = _Img.new("L", (1280, 720), 0)
            _Draw.Draw(m).rectangle([0, 0, 40, 40], fill=255)
            m.save(updir / "000000" / sub / f"000000_{i:06d}.png")
    det_json = updir / "det.json"
    try:
        n_det = _run_detector(str(src), str(det_json))
        if n_det == 0:
            wp = []                       # keine Anker_Kurz im Bild -> 0 Posen (kein Worker-Call)
        else:
            wp = _worker_infer_upload(str(updir), str(det_json))
            if wp is None:
                raise RuntimeError("Worker nicht erreichbar (laeuft kip_infer_worker auf 8078?)")
    except Exception as e:
        raise HTTPException(500, f"Inferenz fehlgeschlagen: {e}")
    # preds -> world poses (+ snap), pose_result-Doc
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
    t_w2c = np.array(cam["cam_t_w2c"], float)
    table_origin = _table_origin()
    results, kept_proj, n_drop = [], [], 0
    iid = 0
    for p in wp:
        iid += 1
        if not _add_pred_filtered(results, kept_proj,
                                  np.array(p["R"], float).reshape(3, 3), np.array(p["t"], float),
                                  R_w2c, t_w2c, table_origin, p["obj_id"], iid, p.get("score", 1.0), K=K):
            n_drop += 1
    doc = {"meta": {"source_image": f"real_{job}", "table_origin": table_origin, "units": "m",
                    "n_det": n_det, "n_offtable_dropped": n_drop, "kept_proj": kept_proj,
                    "camera": _camera_pose(R_w2c, t_w2c, K, results, table_origin)},
           "results": results}
    json.dump(doc, open(RENDERS / f"real_{job}.json", "w"))
    return {"job": job, "n_parts": len(results), "n_det": n_det,
            "pose_result_url": f"api/real/result/{job}"}


def _worker_infer_upload(updir, det_json):
    try:
        import urllib.request, urllib.parse
        qs = urllib.parse.urlencode({"dir": updir, "det": det_json})
        with urllib.request.urlopen(f"http://127.0.0.1:8078/infer_upload?{qs}", timeout=60) as r:
            return json.loads(r.read().decode()).get("preds")
    except Exception:
        return None


@app.get("/api/real/result/{job}")
def real_result(job: str):
    p = RENDERS / f"real_{job}.json"
    if not p.exists():
        raise HTTPException(404, "Ergebnis nicht gefunden")
    return JSONResponse(json.load(open(p)))


@app.get("/api/real/rgb/{job}")
def real_rgb(job: str):
    p = UPLOADS / f"real_{job}" / "000000" / "rgb" / "000000.png"
    if not p.exists():
        raise HTTPException(404, "Bild nicht gefunden")
    return FileResponse(str(p), media_type="image/png")


def _real_infer_job(job, img_bytes, fname):
    """Hintergrund-Thread fuer Real-Upload mit Phasen-Fortschritt (5 Phasen)."""
    import numpy as np
    try:
        updir = UPLOADS / f"real_{job}"
        (updir / "000000" / "rgb").mkdir(parents=True, exist_ok=True)
        src = updir / "000000" / "rgb" / "000000.png"
        with open(src, "wb") as f: f.write(img_bytes)
        cam = _zivid_cam()
        json.dump({"0": cam}, open(updir / "000000" / "scene_camera.json", "w"))
        trained_oids = sorted(_trained_oids()) or [1]
        dummy_gt = [{"obj_id": oid, "cam_R_m2c": [1,0,0,0,1,0,0,0,1], "cam_t_m2c": [0,0,1000]} for oid in trained_oids]
        dummy_gi = [{"bbox_visib": [0,0,40,40], "bbox_obj": [0,0,40,40], "visib_fract": 1.0, "px_count_visib": 1600} for _ in trained_oids]
        json.dump({"0": dummy_gt}, open(updir / "000000" / "scene_gt.json", "w"))
        json.dump({"0": dummy_gi}, open(updir / "000000" / "scene_gt_info.json", "w"))
        from PIL import Image as _Img, ImageDraw as _Draw
        for sub in ("mask", "mask_visib"):
            (updir / "000000" / sub).mkdir(exist_ok=True)
            for i in range(len(trained_oids)):
                m = _Img.new("L", (1280, 720), 0)
                _Draw.Draw(m).rectangle([0, 0, 40, 40], fill=255)
                m.save(updir / "000000" / sub / f"000000_{i:06d}.png")
        det_json = updir / "det.json"
        _job_set(job, phase="Detektor (YOLOv8-OBB)", pct=20)
        n_det = _run_detector(str(src), str(det_json))
        _job_set(job, phase=f"Detektor: {n_det} Box(en) gefunden", pct=45)
        if n_det == 0:
            wp = []
        else:
            _job_set(job, phase="GDRNPP-Inferenz (Multi-Objekt-Worker)", pct=55)
            wp = _worker_infer_upload(str(updir), str(det_json))
            if wp is None:
                raise RuntimeError("Worker nicht erreichbar (Port 8078?)")
        _job_set(job, phase="BOP -> Welt + Boden-Snap", pct=85)
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        table_origin = _table_origin()
        results, kept_proj, n_drop = [], [], 0
        iid = 0
        for p in wp:
            iid += 1
            if not _add_pred_filtered(results, kept_proj,
                                      np.array(p["R"], float).reshape(3, 3), np.array(p["t"], float),
                                      R_w2c, t_w2c, table_origin, p["obj_id"], iid, p.get("score", 1.0), K=K):
                n_drop += 1
        doc = {"meta": {"source_image": f"real_{job}", "table_origin": table_origin, "units": "m",
                        "n_det": n_det, "n_offtable_dropped": n_drop, "kept_proj": kept_proj,
                        "camera": _camera_pose(R_w2c, t_w2c, K, results, table_origin)},
               "results": results}
        # Atomar schreiben (Frontend pollt -> kein Race auf halb-geschriebenes JSON)
        with open(RENDERS / f"real_{job}.json", "w") as _f:
            json.dump(doc, _f); _f.flush(); os.fsync(_f.fileno())
        counts = {}
        for r in results: counts[r["part"]] = counts.get(r["part"], 0) + 1
        _job_set(job, phase="Fertig", pct=100, result_url=f"real/result/{job}",
                 rgb_url=f"real/rgb/{job}", boxes_url=f"real/boxes/{job}",
                 n_det=n_det, n_parts=len(results), counts=counts)
    except Exception as e:
        import traceback; traceback.print_exc()
        _job_set(job, phase=f"Fehler: {e}", pct=-1, error=str(e))


@app.post("/api/real/infer_async")
async def real_infer_async(image: UploadFile = File(...),
                           pipeline: str = Form("gdrnpp")):
    """Startet Real-Upload-Inferenz im Hintergrund. Frontend pollt /api/real/job/<id>.

    pipeline (optional, default 'gdrnpp'): waehlt die Pose-Pipeline. Default = der
    unveraenderte Live-Pfad; fremde Pipelines -> 501 bis angebunden (Seam-Scaffold)."""
    _resolve_pipeline(pipeline)
    job = uuid.uuid4().hex[:8]
    _job_set(job, phase="Upload empfangen", pct=5)
    img_bytes = await image.read()
    fname = image.filename or "upload.png"
    threading.Thread(target=_real_infer_job, args=(job, img_bytes, fname), daemon=True).start()
    return {"job": job}


@app.get("/api/real/job/{job}")
def real_job_status(job: str):
    st = _job_get(job)
    return st or {"error": "unknown job"}


def _load_world_extrinsics(scene_dir: pathlib.Path, im: int):
    import numpy as np
    cam = json.load(open(scene_dir / "scene_camera.json"))[str(im)]
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
    t_w2c = np.array(cam["cam_t_w2c"], float)            # mm
    return K, R_w2c, t_w2c


_MODELS_DIR = pathlib.Path("/mnt/data/kip_pose/project/bop/pose_isaac/models")
_MESH_CACHE: dict = {}

def _mesh_verts_m(obj_id):
    """CAD-Body-Vertices in Metern (fuer planar_z_snap), gecached. trimesh-basiert.

    ⚠️ ZENTRIERUNG: der Viewer rendert das Mesh um seinen geometrischen Center
    (siehe partMeshes.js `_recenter`). Damit der Snap die GLEICHE Referenz hat,
    muessen die Vertices hier ebenfalls um den AABB-Mittelpunkt zentriert werden.
    Sonst schnappt der Backend-Algorithmus den PLY-Origin-Tiefpunkt auf den Tisch,
    der Viewer aber den Mesh-Center — Drift = (PLY-origin − AABB-center) Z-Anteil,
    typisch 1-2 cm. Effekt: Teile schweben oder versinken im Tisch.
    """
    if obj_id in _MESH_CACHE:
        return _MESH_CACHE[obj_id]
    verts = None
    try:
        import trimesh, numpy as np
        p = _MODELS_DIR / f"obj_{int(obj_id):06d}.ply"
        if p.exists():
            m = trimesh.load(str(p), process=False)
            verts = np.asarray(m.vertices, dtype=float) / 1000.0   # mm -> m
            # AABB-Center (gleiche Definition wie THREE.Box3.getCenter im Viewer).
            bb_center = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
            verts = verts - bb_center
    except Exception:
        verts = None
    _MESH_CACHE[obj_id] = verts
    return verts


def _bop_pose_to_result(R_m2c, t_m2c, R_w2c, t_w2c, table_origin, obj_id,
                        inst_id, color, conf, snap=False):
    """BOP cam-frame (R_m2c,t_m2c mm) -> world pose_result-Eintrag (blau/rot).

    snap=True: planar-Z-Snap auf die Tischebene (z=0 im pose-frame) — zieht das
    Teil auf den Boden statt schweben zu lassen ("Physik"). Guard 0.10m gegen
    katastrophale Snaps (Held-/Luft-Teile). Wird für inferred (rot) angewandt;
    GT (blau) bleibt roh (= Wahrheit). Spiegelt PLANAR_REFINE der Gesamt-Pipeline.
    """
    import numpy as np
    from bop_adapter import bop_pose_to_world, part_for_obj_id
    Rw, tw = bop_pose_to_world(R_m2c, t_m2c, R_w2c, t_w2c, table_origin)
    if snap:
        verts = _mesh_verts_m(obj_id)
        if verts is not None:
            from bop_adapter import planar_z_snap
            # Tischebene liegt im pose-frame bei z = -table_origin.z (der Viewer
            # addiert +table_origin → Tisch bei viewer-z 0). GT ruht dort; der
            # Snap zieht den tiefsten pred-Mesh-Punkt auf dieselbe Ebene.
            table_z = -float(table_origin[2])
            tw, _dz = planar_z_snap(Rw, tw, verts, table_z=table_z, max_snap_m=0.10)
    # upright-Heuristik: Objekt-Y-Achse (Längsachse Anker) ~ Welt-Z => steht.
    up = bool(abs(float(Rw[2, 1])) > 0.6)
    return {
        "instance_id": int(inst_id),
        "part": part_for_obj_id(int(obj_id)),
        "face": "—",
        "confidence": float(conf),
        "t_world": [float(tw[0]), float(tw[1]), float(tw[2])],
        "R_world": [float(x) for x in np.asarray(Rw).reshape(9)],
        "upright": up,
        "color": color,                 # "gt" (blau) | "pred" (rot)
    }


def _preds_for(scene: int, im: int):
    """preds_best.csv aller trainierten Objekte für scene/im -> [(obj_id,R,t,score)]."""
    import csv, numpy as np
    out = []
    for slug in _discover_trained():
        oid = {"anker_kurz": 1, "anker_lang": 2, "zahnrad": 6}.get(slug)
        csvp = GDRNPP_OUT / slug / "preds_best.csv"
        if oid is None or not csvp.exists():
            continue
        with open(csvp) as f:
            for row in csv.DictReader(f):
                if int(row["scene_id"]) == scene and int(row["im_id"]) == im and int(row["obj_id"]) == oid:
                    R = np.array([float(x) for x in row["R"].split()], float).reshape(3, 3)
                    t = np.array([float(x) for x in row["t"].split()], float)
                    out.append((oid, R, t, float(row["score"])))
    return out


def _table_origin():
    try:
        from e2e_infer import TABLE_ORIGIN_SCENE as TO
        return list(TO)
    except Exception:
        return [0.0, 0.0, 0.08]


# ── T-113: Off-Table-Pred-Filter ────────────────────────────────────────────
# Der OBB-Detektor halluziniert gelegentlich ein "Teil" auf Zell-Wand-Features
# (z.B. das wbk-Zahnkranz-Logo, T-112-Befund). Solche FPs erzeugen entweder eine
# Pred, deren back-projizierte Welt-Position WEIT neben der Arbeitsfläche liegt
# (Wand statt Tisch), oder eine 2D-Box ohne überhaupt eine valide 3D-Pose. Beides
# soll raus — aus den 3D-results UND der 2D-PiP (konsistent), damit kein Phantom-
# Teil/Phantom-Box mehr erscheint. NUR Pred wird gefiltert; GT (echte gespawnte
# Teile aus scene_gt) ist immer on-table und bleibt unberührt.
#
# Kriterium = Welt-XY der Pred muss in der ARBEITSFLÄCHE liegen. Z ist durch
# planar_z_snap auf den Tisch gezogen → XY ist das diskriminierende Maß.
#
# WORKSPACE_BOX (WORLD-Frame, Meter) — empirisch aus 4043 GT-Posen über 10 val-
# Szenen kalibriert: deckt ALLE echten Teile ab (0/4043 fälschlich gedroppt) mit
# Sicherheitsmarge. WICHTIG: das spawn-window aus gen_sdg_arm_visible.py
# (x[0.057,0.783] y[0.042,0.537]) ist NUR der DROP-Bereich — die Teile rollen
# danach physikbasiert über die ganze Tischfläche, u.a. in die beidseitigen Trays
# (Poltopf_Tray x[-0.204,-0.004], Anker_Tray x[0.806,1.021]). Würde man nur das
# spawn-window (+ kleine Toleranz) als Kriterium nehmen, fielen 574/4043 echte
# GT-Teile raus — viel zu aggressiv. Die Workbox umschließt die volle gemessene
# GT-Ausdehnung (x[-0.152,0.965] y[-0.056,0.653]) plus ~10cm Rand. KONSERVATIV:
# im Zweifel behalten (lieber ein FP durch als ein echtes Tisch-Teil gefiltert).
# Override via env (kommagetrennt "xmin,xmax,ymin,ymax").
def _workspace_box():
    raw = os.environ.get("KIP_WORKSPACE_BOX")
    if raw:
        try:
            xl, xh, yl, yh = (float(v) for v in raw.split(","))
            return (xl, xh), (yl, yh)
        except Exception:
            pass
    return (-0.25, 1.07), (-0.12, 0.72)


def _pred_world_xy(t_world, table_origin):
    """Welt-XY einer Pred (t_world ist pose-frame = world - table_origin).

    world_xy = t_world_xy + table_origin_xy. Bei der aktuellen Tisch-Definition
    ist table_origin.xy == 0 (nur z=0.08), wir addieren es trotzdem explizit →
    frame-robust falls der Tisch-Nullpunkt künftig in XY verschoben wird."""
    return (float(t_world[0]) + float(table_origin[0]),
            float(t_world[1]) + float(table_origin[1]))


def _pred_on_table(t_world, table_origin):
    """True wenn die Pred-Welt-XY in der Arbeitsfläche liegt (sonst Off-Table-FP)."""
    wx, wy = _pred_world_xy(t_world, table_origin)
    (xl, xh), (yl, yh) = _workspace_box()
    return (xl <= wx <= xh) and (yl <= wy <= yh)


def _project_world_to_image(t_world, table_origin, R_w2c, t_w2c, K):
    """Pred-Welt-Position -> Bild-Pixel (u,v). Für 2D↔3D-Box-Korrelation.

    Verifiziert (T-113): die Rückprojektion einer behaltenen Pred landet exakt im
    Zentrum ihrer Detektor-Box → eine det-Box wird nur gezeichnet, wenn eine
    behaltene Pred dorthin projiziert (immun gegen Loader-Reorder + löst Box-ohne-
    Pred wie das Wand-Zahnrad). Gibt None bei pc.z<=0 (hinter der Kamera)."""
    import numpy as np
    wmm = np.array([float(t_world[0]) + float(table_origin[0]),
                    float(t_world[1]) + float(table_origin[1]),
                    float(t_world[2]) + float(table_origin[2])], float) * 1000.0
    pc = np.asarray(R_w2c, float) @ wmm + np.asarray(t_w2c, float)
    if pc[2] <= 1e-6:
        return None
    Kk = np.asarray(K, float)
    u = Kk[0, 0] * pc[0] / pc[2] + Kk[0, 2]
    v = Kk[1, 1] * pc[1] / pc[2] + Kk[1, 2]
    return (float(u), float(v))


def _box_has_kept_pred(bbox_xywh, oid, kept_proj):
    """True wenn eine behaltene Pred dieses obj_id in diese Box projiziert.

    kept_proj: Liste [{obj_id,u,v}] der behaltenen (on-table) Preds. Eine Box wird
    nur gezeichnet, wenn mind. eine behaltene Pred gleicher obj_id mit ihrer Bild-
    Projektion innerhalb der Box (kleiner Rand) liegt. Off-table-Preds sind nicht
    in kept_proj → ihre Box verschwindet; Detektionen ohne valide Pose (Wand-
    Zahnrad) haben keine Projektion → ihre Box verschwindet. 2D == 3D."""
    x, y, w, h = bbox_xywh
    pad = 0.25 * max(w, h)            # toleranter Rand (Projektion ~ Box-Zentrum)
    for kp in kept_proj:
        if int(kp.get("obj_id", -1)) != int(oid):
            continue
        u, v = kp.get("u"), kp.get("v")
        if u is None or v is None:
            continue
        if (x - pad) <= u <= (x + w + pad) and (y - pad) <= v <= (y + h + pad):
            return True
    return False


def _add_pred_filtered(results, kept_proj, R, t, R_w2c, t_w2c, table_origin,
                       obj_id, inst_id, conf, K=None):
    """Pred -> world-Result, NUR anhängen wenn on-table (T-113-Off-Table-Filter).

    Hängt bei on-table das pred-Result an `results` an und (wenn K gegeben) die
    Bild-Projektion an `kept_proj` (für 2D-Box-Konsistenz). Off-table → verworfen
    (nicht in results, nicht in kept_proj → Box wird auch nicht gezeichnet).
    Gibt True zurück wenn behalten, False wenn als Off-Table-FP gedroppt."""
    res = _bop_pose_to_result(R, t, R_w2c, t_w2c, table_origin, obj_id,
                              inst_id, "pred", conf, snap=True)
    if not _pred_on_table(res["t_world"], table_origin):
        return False
    results.append(res)
    if K is not None and kept_proj is not None:
        proj = _project_world_to_image(res["t_world"], table_origin, R_w2c, t_w2c, K)
        if proj is not None:
            kept_proj.append({"obj_id": int(obj_id), "u": proj[0], "v": proj[1]})
    return True


_OID = {"anker_kurz": 1, "anker_lang": 2, "zahnrad": 6}

def _trained_oids():
    return {_OID[s] for s in _discover_trained() if s in _OID}


def _first_trained_im(gt_all, trained):
    for k in sorted(gt_all, key=int):
        if any(o["obj_id"] in trained for o in gt_all[k]):
            return int(k)
    return int(sorted(gt_all, key=int)[0])


def _camera_pose(R_w2c, t_w2c, K, results, table_origin):
    """Aufnahme-Kamera im Viewer-Frame (world): pos/look_at/up/fov_y für Foto-View."""
    import numpy as np
    Rwc = np.asarray(R_w2c, float); twc = np.asarray(t_w2c, float)
    cam_pos = (-Rwc.T @ twc) / 1000.0                 # world meters
    up = Rwc.T @ np.array([0.0, -1.0, 0.0])           # OpenCV y-down -> up = -y
    if results:
        cen = np.mean([[r["t_world"][i] + table_origin[i] for i in range(3)] for r in results], axis=0)
    else:
        cen = np.asarray(table_origin, float)
    fov_y = float(2 * np.degrees(np.arctan(0.5 * 720.0 / float(K[1, 1]))))
    return {"cam_pos": [float(x) for x in cam_pos], "look_at": [float(x) for x in cen],
            "up": [float(x) for x in up], "fov_y": fov_y}


def _worker_infer(scene, im):
    """Persistenten Infer-Worker (Port 8078) fragen. None wenn nicht erreichbar."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:8078/infer?scene={scene}&im={im}"
        with urllib.request.urlopen(url, timeout=40) as r:
            d = json.loads(r.read().decode())
        return d.get("preds")     # [{obj_id, R[9], t[3], score}]
    except Exception:
        return None


def _build_sim_doc(scene, im=-1, use_worker=False):
    """GT(blau)+Pred(rot)-Doc einer Val-Szene. use_worker=True: on-demand Live-
    Inferenz via warmem Worker; Fallback auf preds_best.csv wenn Worker offline."""
    import numpy as np
    table_origin = _table_origin()
    scene_dir = VAL_ROOT / f"{scene:06d}"
    if not scene_dir.exists():
        raise HTTPException(404, f"Szene {scene} nicht gefunden")
    gt_all = json.load(open(scene_dir / "scene_gt.json"))
    trained = _trained_oids()
    if im < 0:
        im = _first_trained_im(gt_all, trained)
    K, R_w2c, t_w2c = _load_world_extrinsics(scene_dir, im)
    results, iid = [], 0
    for o in gt_all.get(str(im), []):
        if o["obj_id"] in trained:
            iid += 1
            results.append(_bop_pose_to_result(
                np.array(o["cam_R_m2c"]).reshape(3, 3), np.array(o["cam_t_m2c"]),
                R_w2c, t_w2c, table_origin, o["obj_id"], iid, "gt", 1.0))
    source = "preds_best"
    preds = None
    if use_worker:
        wp = _worker_infer(scene, im)
        if wp is not None:
            preds = [(int(p["obj_id"]), np.array(p["R"], float).reshape(3, 3),
                      np.array(p["t"], float), float(p.get("score", 1.0))) for p in wp]
            source = "worker"
    if preds is None:
        preds = _preds_for(scene, im)
    kept_proj, n_drop = [], 0
    for oid, R, t, sc in preds:
        iid += 1
        if not _add_pred_filtered(results, kept_proj, R, t, R_w2c, t_w2c,
                                  table_origin, oid, iid, sc, K=K):
            n_drop += 1
    return {
        "meta": {"source_image": f"{scene:06d}/rgb/{im:06d}.png", "table_origin": table_origin,
                 "units": "m", "scene": scene, "im": im, "source": source,
                 "n_offtable_dropped": n_drop, "kept_proj": kept_proj,
                 "camera": _camera_pose(R_w2c, t_w2c, K, results, table_origin),
                 "n_gt": sum(1 for r in results if r["color"] == "gt"),
                 "n_pred": sum(1 for r in results if r["color"] == "pred")},
        "results": results,
    }


@app.get("/api/sim/scene")
def sim_scene(scene: int = 0, im: int = -1):
    """GT vs Pred aus preds_best.csv (schnell, vorberechnet)."""
    return _build_sim_doc(scene, im, use_worker=False)


@app.get("/api/sim/infer")
def sim_infer(scene: int = 0, im: int = -1):
    """On-demand: warmer Worker rechnet die Szene frisch (Fallback preds_best)."""
    return _build_sim_doc(scene, im, use_worker=True)


# Klassen-Farben fuer 2D-Boxes (pro obj_id, gut unterscheidbar auf Werkstatt-RGB).
_BBOX_COLOR = {
    1: (255, 140, 66),     # Anker_Kurz — Orange
    2: (232, 63, 142),     # Anker_Lang — Magenta
    6: (0, 194, 209),      # Zahnrad    — Cyan
}
_BBOX_LABEL = {1: "Anker_Kurz", 2: "Anker_Lang", 6: "Zahnrad"}


def _draw_label(draw, x, y, text, fill):
    """Label mit halbtransparentem Hintergrund-Rechteck. Schrift gross genug
    dass Max das auf einem PiP-Thumb lesen kann."""
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        try: font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 18)
        except Exception: font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


@app.get("/api/sim/boxes")
def sim_boxes(scene: int = 0, im: int = -1):
    """RGB der Szene mit 2D-Boxen (bbox_visib der trainierten Objekte) fuers PiP.

    Pro Klasse eine eigene Farbe + lesbares Label mit Hintergrund-Rechteck."""
    from PIL import Image, ImageDraw
    import io
    scene_dir = VAL_ROOT / f"{scene:06d}"
    if not scene_dir.exists():
        raise HTTPException(404, f"Szene {scene} nicht gefunden")
    gt_all = json.load(open(scene_dir / "scene_gt.json"))
    gi_all = json.load(open(scene_dir / "scene_gt_info.json"))
    trained = _trained_oids()
    if im < 0:
        im = _first_trained_im(gt_all, trained)
    rgb = scene_dir / "rgb" / f"{im:06d}.png"
    if not rgb.exists():
        raise HTTPException(404, "Bild nicht gefunden")
    img = Image.open(rgb).convert("RGB"); draw = ImageDraw.Draw(img)
    for idx, o in enumerate(gt_all.get(str(im), [])):
        if o["obj_id"] in trained:
            bb = gi_all[str(im)][idx].get("bbox_visib")
            if bb and bb[2] > 0 and bb[3] > 0:
                x, y, w, h = bb
                col = _BBOX_COLOR.get(o["obj_id"], (52, 211, 153))
                draw.rectangle([x, y, x + w, y + h], outline=col, width=4)
                _draw_label(draw, x + 4, max(0, y - 24),
                            _BBOX_LABEL.get(o["obj_id"], str(o["obj_id"])), col)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/real/boxes/{job}")
def real_boxes(job: str):
    """Uploaded RGB mit Detektor-Boxen overlayed (pro Klasse eingefaerbt + Label).

    Liest die bbox_est aus dem temp/det.json des Real-Inferenz-Jobs."""
    from PIL import Image, ImageDraw
    import io
    rgb = UPLOADS / f"real_{job}" / "000000" / "rgb" / "000000.png"
    detp = UPLOADS / f"real_{job}" / "det.json"
    if not rgb.exists() or not detp.exists():
        raise HTTPException(404, "Bild oder Detektionen nicht gefunden")
    img = Image.open(rgb).convert("RGB"); draw = ImageDraw.Draw(img)
    det = json.load(open(detp))
    # T-113: nur Boxen zeichnen, zu denen eine behaltene (on-table) Pred projiziert
    # — Off-Table-FPs (Wand-Zahnrad) erscheinen weder in 3D noch hier (2D == 3D).
    # Doc fehlt -> konservativ alle Boxen zeichnen (kein Filter).
    docp = RENDERS / f"real_{job}.json"
    _meta = json.load(open(docp)).get("meta", {}) if docp.exists() else None
    _filter = _meta is not None
    kept_proj = (_meta or {}).get("kept_proj", [])
    # det.json ist dict {scene/im: [{obj_id, bbox_est=[x,y,w,h], score, ...}]}
    boxes = []
    for k, lst in det.items():
        for b in lst:
            boxes.append((int(b["obj_id"]), b["bbox_est"], float(b.get("score", 0.0))))
    for oid, (x, y, w, h), sc in boxes:
        if _filter and not _box_has_kept_pred((x, y, w, h), oid, kept_proj):
            continue
        col = _BBOX_COLOR.get(oid, (52, 211, 153))
        draw.rectangle([x, y, x + w, y + h], outline=col, width=4)
        label = f"{_BBOX_LABEL.get(oid, str(oid))}  {sc*100:.0f}%"
        _draw_label(draw, x + 4, max(0, y - 24), label, col)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


_SIM_DOCS: dict = {}                 # job_id -> fertiges sim-doc (in-memory cache)

def _sim_infer_job(job, scene, im):
    """Hintergrund-Thread fuer Sim-Inferenz mit Phasen-Fortschritt (3 Phasen)."""
    try:
        _job_set(job, phase=f"GDRNPP-Worker rechnet Szene {scene}", pct=30)
        doc = _build_sim_doc(scene, im, use_worker=True)
        _job_set(job, phase="BOP -> Welt + Boden-Snap", pct=85)
        # Snap ist Teil von _build_sim_doc; hier nur Phase markieren.
        _SIM_DOCS[job] = doc
        s_im = doc["meta"]["im"]; n_gt = doc["meta"]["n_gt"]; n_pred = doc["meta"]["n_pred"]
        _job_set(job, phase="Fertig", pct=100, result_url=f"sim/job_result/{job}",
                 rgb_url=f"sim/rgb?scene={scene}&im={s_im}",
                 boxes_url=f"sim/boxes?scene={scene}&im={s_im}",
                 scene=scene, im=s_im, n_gt=n_gt, n_pred=n_pred,
                 source=doc["meta"].get("source", "preds_best"))
    except HTTPException as he:
        _job_set(job, phase=f"Fehler: {he.detail}", pct=-1, error=str(he.detail))
    except Exception as e:
        import traceback; traceback.print_exc()
        _job_set(job, phase=f"Fehler: {e}", pct=-1, error=str(e))


@app.get("/api/sim/infer_async")
def sim_infer_async(scene: int = 0, im: int = -1):
    """Startet Sim-Inferenz im Hintergrund. Frontend pollt /api/sim/job/<id>."""
    job = uuid.uuid4().hex[:8]
    _job_set(job, phase="Inferenz startet", pct=10)
    threading.Thread(target=_sim_infer_job, args=(job, scene, im), daemon=True).start()
    return {"job": job}


# ── Live-Isaac-Generation Configuration ─────────────────────────────────────
_ISAAC_VENV   = "/mnt/data/isaacsim-venv/bin/python"
_BOP_VENV     = "/mnt/data/bop/gdrnpp-venv/bin/python"
_GEN_SCRIPT   = "/mnt/data/kip_pose/box_src/gen_sdg_arm_visible.py"
_CONV_SCRIPT  = "/mnt/data/kip_pose/box_src/isaac_to_bop.py"
_SCENE_USD    = "/mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd"
_USD_DIR      = "/mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files"
_LIVE_ROOT    = pathlib.Path("/mnt/data/kip_pose/project/temp/kip_live")
_LIVE_ROOT.mkdir(parents=True, exist_ok=True)


def _sim_generate_job(job):
    """Live-Isaac-Pipeline: Isaac rendert frisches Frame → BOP-Konvertierung →
    Detektor → warmer GDRNPP-Worker → fertige Sim-Doc.

    Phasen (gesamt ~60-80s):
      10%  Isaac startet (booting SimulationApp, ~25s)
      40%  Isaac rendert (spawn + physics + render, ~25s)
      70%  BOP-Konvertierung (~2s)
      80%  Detektor (YOLOv8-OBB, ~3s)
      90%  GDRNPP-Worker (~4s)
     100%  Fertig (Welt-Transform + Boden-Snap)
    """
    import numpy as np
    try:
        rawdir = _LIVE_ROOT / job
        rawdir.mkdir(parents=True, exist_ok=True)
        _job_set(job, phase="Isaac Sim startet (booting)", pct=10)

        import random as _r
        seed = _r.randint(1, 2**31 - 1)
        # 3-5 Teile pro Szene; force-counts garantiert dass semantic labels
        # stabil registriert werden (single-scene-Bug bei num-scenes=1 ohne).
        # focus-frac=1.0 ⇒ nur trainierte Klassen (Anker_Kurz/Lang/Zahnrad).
        # Sonst kann die Zufalls-Auswahl reine Distractor liefern (Demo-Killer).
        n_obj = _r.randint(3, 5)
        gen = subprocess.Popen(
            [_ISAAC_VENV, _GEN_SCRIPT,
             "--scene", _SCENE_USD,
             "--usd-dir", _USD_DIR,
             "--output", str(rawdir),
             "--num-scenes", "1",
             "--force-counts", str(n_obj),
             "--focus-frac", "1.0",
             "--seed", str(seed)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # parse stdout for progress markers
        last_phase_pct = 10
        for line in gen.stdout:
            if "scene loaded" in line:
                _job_set(job, phase="Isaac Sim rendert Frame", pct=40)
                last_phase_pct = 40
            elif "DONE 1 scenes" in line:
                _job_set(job, phase="Isaac fertig — BOP-Konvertierung", pct=70)
                last_phase_pct = 70
        gen.wait(timeout=180)
        if gen.returncode != 0:
            raise RuntimeError(f"Isaac-Render fehlgeschlagen (exit {gen.returncode})")
        if not (rawdir / "rgb_0000.png").exists():
            raise RuntimeError("Isaac produzierte kein rgb_0000.png")

        # BOP-Konvertierung in temp scene-id 0 (Worker erwartet 000000 als Scene-Dir).
        bopdir = rawdir / "bop"; bopdir.mkdir(exist_ok=True)
        cv = subprocess.run(
            [_BOP_VENV, _CONV_SCRIPT,
             "--raw-dir", str(rawdir),
             "--bop-root", str(bopdir),
             "--split", "test",
             "--scene-id", "0",
             "--start-im", "0",
             "--limit", "1"],
            capture_output=True, text=True, timeout=60)
        if cv.returncode != 0:
            raise RuntimeError(f"BOP-Konvertierung fehlgeschlagen: {cv.stderr[-300:]}")
        scene_dir = bopdir / "test" / "000000"
        if not (scene_dir / "scene_gt.json").exists():
            raise RuntimeError("BOP-Konverter produzierte keine scene_gt.json")
        # Worker erwartet updir mit unmittelbar darin liegendem 000000-scene-dir.
        # bopdir/test/ erfuellt das Schema (000000 = unsere Live-Szene). Wir
        # kopieren bopdir/test/ NICHT um — wir verwenden es direkt als updir.
        updir = bopdir / "test"

        # Sim-Doc bauen — wir mounten das live scene_dir temporaer als VAL_ROOT
        # NICHT global um. Stattdessen replizieren wir _build_sim_doc inline.
        _job_set(job, phase="Detektor (YOLOv8-OBB)", pct=80)
        gt_all = json.load(open(scene_dir / "scene_gt.json"))
        cam_all = json.load(open(scene_dir / "scene_camera.json"))
        trained = _trained_oids()
        # einziger Frame: "0"
        im = 0
        cam = cam_all[str(im)]
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        table_origin = _table_origin()

        results, iid = [], 0
        # GT-Teile (blau)
        for o in gt_all.get(str(im), []):
            if o["obj_id"] in trained:
                iid += 1
                results.append(_bop_pose_to_result(
                    np.array(o["cam_R_m2c"]).reshape(3, 3), np.array(o["cam_t_m2c"]),
                    R_w2c, t_w2c, table_origin, o["obj_id"], iid, "gt", 1.0))

        # Detektor auf dem Live-Bild
        rgb_src = scene_dir / "rgb" / "000000.png"
        det_json = rawdir / "det.json"
        n_det = _run_detector(str(rgb_src), str(det_json))
        _job_set(job, phase=f"Detektor: {n_det} Box(en) gefunden", pct=85)

        # GDRNPP-Worker auf dem Live-Frame
        if n_det > 0:
            _job_set(job, phase="GDRNPP-Inferenz (Multi-Objekt-Worker)", pct=90)
            wp = _worker_infer_upload(str(updir), str(det_json))
            if wp is None:
                raise RuntimeError("Worker nicht erreichbar")
        else:
            wp = []

        kept_proj, n_drop = [], 0
        for p in wp:
            iid += 1
            if not _add_pred_filtered(results, kept_proj,
                                      np.array(p["R"], float).reshape(3, 3), np.array(p["t"], float),
                                      R_w2c, t_w2c, table_origin, p["obj_id"], iid, p.get("score", 1.0), K=K):
                n_drop += 1

        _job_set(job, phase="BOP -> Welt + Boden-Snap", pct=95)
        n_gt = sum(1 for r in results if r["color"] == "gt")
        n_pred = sum(1 for r in results if r["color"] == "pred")
        doc = {
            "meta": {"source_image": f"live/{job}", "table_origin": table_origin,
                     "units": "m", "scene": 99, "im": 0, "source": "isaac-live",
                     "n_offtable_dropped": n_drop, "kept_proj": kept_proj,
                     "camera": _camera_pose(R_w2c, t_w2c, K, results, table_origin),
                     "n_gt": n_gt, "n_pred": n_pred, "seed": seed, "n_obj": n_obj},
            "results": results,
        }
        _SIM_DOCS[job] = doc
        _job_set(job, phase="Fertig", pct=100, result_url=f"sim/job_result/{job}",
                 rgb_url=f"sim/live_rgb/{job}", boxes_url=f"sim/live_boxes/{job}",
                 scene=99, im=0, n_gt=n_gt, n_pred=n_pred, source="isaac-live",
                 seed=seed, n_obj=n_obj)
    except subprocess.TimeoutExpired:
        _job_set(job, phase="Isaac-Timeout (>3 min)", pct=-1, error="timeout")
    except Exception as e:
        import traceback; traceback.print_exc()
        _job_set(job, phase=f"Fehler: {e}", pct=-1, error=str(e))


@app.get("/api/sim/generate_async")
def sim_generate_async(pipeline: str = "gdrnpp"):
    """Live-Isaac-Generation: NEUES Isaac-Bild rendern (~60s) → Detektor → GDRNPP.

    pipeline (optional, default 'gdrnpp'): Default = unveraenderter Live-Pfad;
    fremde Pipelines -> 501 bis angebunden (Seam-Scaffold)."""
    _resolve_pipeline(pipeline)
    job = uuid.uuid4().hex[:8]
    _job_set(job, phase="Isaac Sim startet", pct=5)
    threading.Thread(target=_sim_generate_job, args=(job,), daemon=True).start()
    return {"job": job}


@app.get("/api/sim/live_rgb/{job}")
def sim_live_rgb(job: str):
    """RGB des live-generierten Isaac-Frames."""
    p = _LIVE_ROOT / job / "bop" / "test" / "000000" / "rgb" / "000000.png"
    if not p.exists():
        raise HTTPException(404, "Live-Bild nicht gefunden")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/sim/live_boxes/{job}")
def sim_live_boxes(job: str):
    """RGB + farbige Detektor-Boxen ueberlagert (Live-Frame)."""
    from PIL import Image, ImageDraw
    import io
    rgb = _LIVE_ROOT / job / "bop" / "test" / "000000" / "rgb" / "000000.png"
    detp = _LIVE_ROOT / job / "det.json"
    if not rgb.exists():
        raise HTTPException(404, "Live-Bild nicht gefunden")
    img = Image.open(rgb).convert("RGB"); draw = ImageDraw.Draw(img)
    if detp.exists():
        det = json.load(open(detp))
        # T-113: nur Boxen mit behaltener (on-table) Pred — Off-Table-FPs raus (2D == 3D).
        # Doc fehlt (z.B. Server-Neustart nach Generation) -> konservativ: alle
        # Boxen zeichnen (lieber ein FP durch als alle Boxen verstecken).
        _doc = _SIM_DOCS.get(job)
        _filter = _doc is not None
        kept_proj = (_doc or {}).get("meta", {}).get("kept_proj", [])
        for k, lst in det.items():
            for b in lst:
                oid = int(b["obj_id"]); x, y, w, h = b["bbox_est"]
                if _filter and not _box_has_kept_pred((x, y, w, h), oid, kept_proj):
                    continue
                col = _BBOX_COLOR.get(oid, (52, 211, 153))
                draw.rectangle([x, y, x + w, y + h], outline=col, width=4)
                label = f"{_BBOX_LABEL.get(oid, str(oid))}  {float(b.get('score',0))*100:.0f}%"
                _draw_label(draw, x + 4, max(0, y - 24), label, col)
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/sim/random_async", deprecated=True)
def sim_random_async():
    """DEPRECATED — Pool-Fallback. Entfernt 2026-05-30: Sim laeuft ausschliesslich
    ueber /api/sim/generate_async (echte Live-Isaac-Generation pro Klick). Diese
    Route bleibt nur als 410-Stub falls FE-Caches noch alte URLs treffen."""
    raise HTTPException(410, "Pool-Endpoint entfernt — nutze /api/sim/generate_async (live).")
    # Toter Code unten nur fuer Referenz; nicht erreichbar.
    import random
    scenes = []
    if VAL_ROOT.exists():
        for d in sorted(VAL_ROOT.glob("[0-9]*")):
            if (d / "scene_gt.json").exists():
                gt = json.load(open(d / "scene_gt.json"))
                trained = _trained_oids()
                # nur Frames die mind. 1 trainiertes Objekt zeigen
                hot = [int(k) for k, ents in gt.items()
                       if any(o["obj_id"] in trained for o in ents)]
                if hot:
                    scenes.append((int(d.name), hot))
    if not scenes:
        raise HTTPException(503, "Keine gelabelten Val-Szenen verfuegbar")
    scene, frames = random.choice(scenes)
    im = random.choice(frames)
    job = uuid.uuid4().hex[:8]
    _job_set(job, phase=f"Neue Szene gewaehlt (Szene {scene} / Bild {im})", pct=10)
    threading.Thread(target=_sim_infer_job, args=(job, scene, im), daemon=True).start()
    return {"job": job, "scene": scene, "im": im}


@app.get("/api/sim/job/{job}")
def sim_job_status(job: str):
    st = _job_get(job)
    return st or {"error": "unknown job"}


@app.get("/api/sim/job_result/{job}")
def sim_job_result(job: str):
    doc = _SIM_DOCS.get(job)
    if not doc:
        raise HTTPException(404, "Ergebnis nicht gefunden (Job laeuft noch oder ist abgelaufen)")
    return doc


@app.get("/api/sim/rgb")
def sim_rgb(scene: int = 0, im: int = 0):
    """Roh-RGB einer Val-Szene fürs 2D-PiP."""
    p = VAL_ROOT / f"{scene:06d}" / "rgb" / f"{im:06d}.png"
    if not p.exists():
        raise HTTPException(404, "Bild nicht gefunden")
    return FileResponse(str(p), media_type="image/png")


@app.post("/api/sim/generate")
def sim_generate(n_scenes: int = Form(1), force: bool = Form(False)):
    """Live Isaac-SDG -> label -> infer -> GT/Pred. (Sim-Screen)

    Parallel-Betrieb ausdrücklich erlaubt (Max: parallel inferieren). Kein
    Training-Guard mehr — Inferenz/SDG teilen sich die GPU mit einem laufenden
    Training. Vollständige Verdrahtung folgt in S-KIP-4.
    """
    # S-KIP-4: gen_sdg_arm_visible.py (1 Szene) -> BOP-convert -> e2e infer
    #          -> render_gt_vs_pred -> blau/rot pose_result für den Viewer.
    raise HTTPException(501, "Live-SDG-Pipeline wird in S-KIP-4 verdrahtet.")


# ── Frontend entry: "/" -> kip.html (2-Screen-Shell). Der alte Single-Viewer
#    index.html bleibt unter /index.html erreichbar. ─────────────
@app.get("/")
def kip_index():
    f = FRONTEND / "kip.html"
    if not f.exists():
        raise HTTPException(404, "kip.html fehlt")
    return FileResponse(str(f))


# ── Catch-all static (MUSS nach allen /api-Routen + "/" kommen) ──
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kip_server:app", host="0.0.0.0", port=PORT, reload=False)
