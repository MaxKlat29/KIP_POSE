"""kip_infer_worker.py v2 — Multi-Objekt warmer GDRNPP-Inferenz-Daemon (Port 8078).

Laedt alle trainierten Modelle (anker_kurz, anker_lang, zahnrad) ins VRAM (cap 0.40
~9.6GB), cached die Test-Loader-Inputs pro Objekt/Szene, inferiert per
  /infer?scene=N&im=M       -> alle obj_ids parallel inferiert
  /infer_upload?dir=PATH    -> pro obj_id temp-Dataset (kip_upload_<slug>) + Inferenz
stdlib-HTTP (gdrnpp-venv hat kein fastapi). kip_server (8077) ruft diesen Worker.
"""
import os, sys, time, argparse, importlib, threading, json, copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
GDRN = "/mnt/data/bop/repos/gdrnpp"
sys.path.insert(0, GDRN); sys.path.insert(0, os.path.join(GDRN, "core/gdrn_modeling"))
import torch
torch.cuda.set_per_process_memory_fraction(0.40, 0)   # 3 Modelle ~3-6GB, cap ~9.6GB
try: torch.multiprocessing.set_sharing_strategy('file_system')
except Exception: pass
import numpy as np

# (slug, obj_id) — alle trainierten Objekte (Reihenfolge = Lade-Reihenfolge)
WORKER_OBJS = [("anker_kurz", 1), ("anker_lang", 2), ("zahnrad", 6)]

STATE = {"ready": False, "error": None, "models": {}}   # obj_id -> dict(cfg,model,by_scene,slug,n_inputs)


def _sii(it):
    s = it["scene_im_id"]
    if isinstance(s, (list, tuple)): s = s[0]
    return str(s)


def _load_one(slug, obj_id):
    """Lade EIN Modell (cfg + checkpoint) + cache val-loader-inputs nach scene."""
    from core.gdrn_modeling.main_gdrn import setup
    from core.utils.my_checkpoint import MyCheckpointer
    from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
    from core.gdrn_modeling.engine.gdrn_evaluator import GDRN_Evaluator
    CFG = f"{GDRN}/configs/gdrn/poseIsaacPbrSO/{slug}.py"
    W = f"{GDRN}/output/gdrn/poseIsaacPbrSO/{slug}/model_best.pth"
    if not os.path.exists(W):
        raise FileNotFoundError(f"model_best.pth fehlt fuer {slug}: {W}")
    args = argparse.Namespace(config_file=CFG, eval_only=True, num_gpus=1, num_machines=1,
        resume=False, launcher="none", local_rank=0,
        opts={"MODEL.WEIGHTS": W, "MODEL.POSE_NET.INPUT_RES": 320, "MODEL.POSE_NET.OUTPUT_RES": 80})
    cfg = setup(args); cfg.DATALOADER.NUM_WORKERS = 0
    builder = importlib.import_module(f"core.gdrn_modeling.models.{cfg.MODEL.POSE_NET.NAME}")
    model, _ = builder.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(W, resume=False)
    model.cuda().eval()
    dn = cfg.DATASETS.TEST[0]
    ev = GDRN_Evaluator(cfg, dn, distributed=False, output_dir=f"/tmp/kipworker_{slug}", train_objs=None)
    loader = build_gdrn_test_loader(cfg, dn, train_objs=ev.train_objs)
    by_scene = {}; n = 0
    for inp in loader:
        for it in (inp if isinstance(inp, list) else [inp]):
            sid = int(_sii(it).split("/")[0])
            by_scene.setdefault(sid, []).append(it); n += 1
    return {"slug": slug, "obj_id": obj_id, "cfg": cfg, "model": model, "by_scene": by_scene, "n_inputs": n}


def warm():
    """Lade alle Modelle sequenziell. Verfuegbar via STATE['models'][obj_id]."""
    try:
        for slug, oid in WORKER_OBJS:
            t0 = time.time()
            STATE["models"][oid] = _load_one(slug, oid)
            print(f"[worker] WARM {slug} (obj_id {oid}) in {time.time()-t0:.1f}s, "
                  f"{STATE['models'][oid]['n_inputs']} inputs", flush=True)
        STATE["ready"] = True
        print(f"[worker] ALL READY: {sorted(STATE['models'].keys())}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); STATE["error"] = str(e)


def _forward(m, batch_inputs):
    """Liste input-dicts durch ein Modell -> (rots [N,3,3], trans [N,3])."""
    from core.gdrn_modeling.engine.engine_utils import batch_data
    b = batch_data(m["cfg"], batch_inputs, phase="test")
    out = m["model"](b["roi_img"], roi_classes=b["roi_cls"], roi_cams=b["roi_cam"],
                    roi_whs=b["roi_wh"], roi_centers=b["roi_center"],
                    resize_ratios=b["resize_ratio"], roi_coord_2d=b.get("roi_coord_2d"),
                    roi_coord_2d_rel=b.get("roi_coord_2d_rel"), roi_extents=b.get("roi_extent"))
    return out["rot"].detach().cpu().numpy(), out["trans"].detach().cpu().numpy()


def run_infer(scene, im):
    """Alle obj_ids inferieren auf scene/im. Sammelt preds (mit korrektem obj_id)."""
    preds = []; t0 = time.time()
    with torch.no_grad():
        for oid, m in STATE["models"].items():
            inputs = [it for it in m["by_scene"].get(scene, [])
                      if im < 0 or int(_sii(it).split("/")[1]) == im]
            for it in inputs:
                rots, trans = _forward(m, [it])
                for i in range(len(rots)):
                    preds.append({"obj_id": oid, "R": rots[i].reshape(9).tolist(),
                                  "t": [float(x)*1000.0 for x in trans[i]], "score": 1.0})
    return {"preds": preds, "n": len(preds), "ms": round((time.time()-t0)*1000)}


def _build_det_consistent_root(updir, full_det):
    """T-115-FIX: baut ein ISOLIERTES det-getriebenes Dataset-Root neben updir.

    PROBLEM: Der base-Loader (pose_isaac_pbr) baut Records NUR aus scene_gt-Instanzen
    mit obj_id in cat_ids und droppt jedes Bild ohne passende GT ('if len(insts)==0:
    continue'). Findet der Detektor ein obj (z.B. Zahnrad/obj 6), das NICHT in der
    scene_gt steht (Live-Sim-GT ohne dieses Spawn, oder FP), wird sein single-obj-
    Dataset leer -> detectron2 'Dataset is empty!' -> das obj wird (frueher still)
    NIE posiert.

    FIX: ein neues Root '<updir>/_t115_det_root' anlegen, das rgb/depth/scene_camera
    je Szene per SYMLINK referenziert und eine det-vollstaendige Dummy-scene_gt(+info,
    +masken) traegt — eine Instanz je im det vorkommender obj_id. So hat JEDES
    detektierte obj ein nicht-leeres base-Dataset. Bei LOAD_DETS_TEST=True ist die GT
    semantisch irrelevant (load_detections ersetzt die Annotationen durch die echten
    Det-Boxen), also unschaedlich fuer bereits funktionierende Objekte (Anker).

    WICHTIG: das ORIGINAL-updir bleibt voellig UNANGETASTET — dort liegt im Live-Sim-
    Pfad die ECHTE scene_gt, die kip_server (:8077) fuer die blauen GT-Overlays liest.
    Wir ueberschreiben sie NICHT, sondern bauen ein separates Root. Gibt das neue Root
    zurueck (als dataset_root fuer alle obj-Loader), None wenn kein Bild vorhanden."""
    import shutil as _sh
    from PIL import Image as _Img, ImageDraw as _Draw
    # det-keys "scene/im" -> {scene_id: {int_im_id: [obj_id, ...]}}
    per_scene = {}
    for sii, dets in full_det.items():
        if not dets:
            continue
        parts = str(sii).split("/")
        sid = int(parts[0]); iid = int(parts[1]) if len(parts) > 1 else 0
        per_scene.setdefault(sid, {}).setdefault(iid, [])
        for d in dets:
            oid = d.get("obj_id")
            if oid is not None and oid not in per_scene[sid][iid]:
                per_scene[sid][iid].append(oid)
    root = os.path.join(updir, "_t115_det_root")
    if os.path.exists(root):
        _sh.rmtree(root, ignore_errors=True)   # frisch je Request (Dummy-GT, kein Verlust)
    built_any = False
    for sid, ims in per_scene.items():
        src = os.path.join(updir, f"{sid:06d}")
        if not os.path.isdir(src):
            continue   # kein Bild-Verzeichnis -> Loader saehe es eh nicht
        dst = os.path.join(root, f"{sid:06d}")
        os.makedirs(os.path.join(dst, "mask"), exist_ok=True)
        os.makedirs(os.path.join(dst, "mask_visib"), exist_ok=True)
        # rgb/depth per Symlink referenzieren (Original-Bilder unberuehrt)
        for sub in ("rgb", "depth"):
            s_sub = os.path.join(src, sub)
            if not os.path.isdir(s_sub):
                continue
            d_sub = os.path.join(dst, sub); os.makedirs(d_sub, exist_ok=True)
            for fn in os.listdir(s_sub):
                link = os.path.join(d_sub, fn)
                if not os.path.lexists(link):
                    os.symlink(os.path.abspath(os.path.join(s_sub, fn)), link)
        # scene_camera.json kopieren (klein, Loader liest sie pro Bild)
        s_cam = os.path.join(src, "scene_camera.json")
        if os.path.exists(s_cam):
            _sh.copyfile(s_cam, os.path.join(dst, "scene_camera.json"))
        gt = {}; gi = {}
        for iid, oids in ims.items():
            gt[str(iid)] = [{"obj_id": oid, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                             "cam_t_m2c": [0, 0, 1000]} for oid in oids]
            gi[str(iid)] = [{"bbox_visib": [0, 0, 40, 40], "bbox_obj": [0, 0, 40, 40],
                             "visib_fract": 1.0, "px_count_visib": 1600} for _ in oids]
            # Dummy-Maske pro Instanz (Loader liest mask/mask_visib unbedingt, >=30 px)
            for ai in range(len(oids)):
                msk = _Img.new("L", (1280, 720), 0)
                _Draw.Draw(msk).rectangle([0, 0, 40, 40], fill=255)
                msk.save(os.path.join(dst, "mask", f"{iid:06d}_{ai:06d}.png"))
                msk.save(os.path.join(dst, "mask_visib", f"{iid:06d}_{ai:06d}.png"))
        json.dump(gt, open(os.path.join(dst, "scene_gt.json"), "w"))
        json.dump(gi, open(os.path.join(dst, "scene_gt_info.json"), "w"))
        built_any = True
    return root if built_any else None


def run_upload(updir, det_json):
    """Upload-Inferenz: pro obj_id ein temp-Dataset (kip_upload_<slug>) + Inferenz.
    Det-json hat alle obj_ids (1/2/6), pro Modell wird der det auf diese obj_id gefiltert."""
    from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
    from core.gdrn_modeling.datasets import pose_isaac_pbr as PIP
    from core.gdrn_modeling.engine.gdrn_evaluator import GDRN_Evaluator
    from detectron2.data import DatasetCatalog, MetadataCatalog
    full = json.load(open(det_json))
    # T-115-FIX: isoliertes det-getriebenes Dataset-Root (Symlinks + Dummy-GT), damit
    # jedes detektierte obj (z.B. Zahnrad ohne scene_gt-Backing) ein nicht-leeres
    # base-Dataset bekommt. Original-updir-scene_gt bleibt UNANGETASTET (kip_server
    # liest sie fuer die blauen GT-Overlays). Fallback: updir, falls Bauen scheitert.
    ds_root = updir
    try:
        r = _build_det_consistent_root(updir, full)
        if r:
            ds_root = r
    except Exception as e:
        print(f"[worker] WARN _build_det_consistent_root: {e}", flush=True)
    preds = []; t0 = time.time()
    with torch.no_grad():
        for oid, m in STATE["models"].items():
            # filtere det auf diese obj_id
            filtered = {sii: [d for d in dets if d.get("obj_id") == oid] for sii, dets in full.items()}
            filtered = {sii: ds for sii, ds in filtered.items() if ds}
            if not filtered:
                continue
            slug = m["slug"]
            det_oid = os.path.join(updir, f"det_{slug}.json")
            json.dump(filtered, open(det_oid, "w"))
            ds_name = f"kip_upload_{slug}"
            cfg_up = dict(PIP.SPLITS_POSE_ISAAC[f"pose_isaac_{slug}_val"])
            cfg_up.update(dataset_root=ds_root, with_masks=False, with_depth=False,
                          filter_invalid=False, use_cache=False)
            if ds_name in DatasetCatalog.list():
                DatasetCatalog.remove(ds_name); MetadataCatalog.remove(ds_name)
            PIP.register_with_name_cfg(ds_name, cfg_up)
            c2 = copy.deepcopy(m["cfg"])
            c2.DATASETS.TEST = (ds_name,); c2.DATASETS.DET_FILES_TEST = (det_oid,)
            c2.MODEL.LOAD_DETS_TEST = True
            c2.TEST.TEST_BBOX_TYPE = "est"
            c2.DATASETS.DET_TOPK_PER_OBJ = 30
            c2.DATALOADER.NUM_WORKERS = 0
            ev = GDRN_Evaluator(c2, ds_name, distributed=False, output_dir=f"/tmp/kipup_out_{slug}", train_objs=None)
            try:
                loader = build_gdrn_test_loader(c2, ds_name, train_objs=ev.train_objs)
            except Exception as e:
                # T-115: kein stiller Drop mehr — den echten Fehler loggen, dann skip.
                print(f"[worker] WARN loader fuer {slug} (obj {oid}) leer/fehlgeschlagen: {e}", flush=True)
                continue
            for inp in loader:
                rots, trans = _forward(m, inp if isinstance(inp, list) else [inp])
                for i in range(len(rots)):
                    preds.append({"obj_id": oid, "R": rots[i].reshape(9).tolist(),
                                  "t": [float(x)*1000.0 for x in trans[i]], "score": 1.0})
    return preds


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/health":
            loaded = sorted(STATE.get("models", {}).keys())
            scenes = sorted(next(iter(STATE["models"].values()))["by_scene"].keys()) if loaded else []
            self._send({"ready": STATE["ready"], "error": STATE["error"],
                        "loaded_obj_ids": loaded, "scenes": scenes})
        elif u.path == "/infer":
            if not STATE["ready"]:
                self._send({"error": STATE["error"] or "not ready", "preds": None}, 503); return
            try:
                scene = int(q.get("scene", ["0"])[0]); im = int(q.get("im", ["-1"])[0])
                self._send(run_infer(scene, im))
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e), "preds": None}, 500)
        elif u.path == "/infer_upload":
            if not STATE["ready"]:
                self._send({"error": STATE["error"] or "not ready", "preds": None}, 503); return
            try:
                updir = q.get("dir", ["/tmp/kipup"])[0]; det = q.get("det", [updir + "/det.json"])[0]
                t0 = time.time(); pr = run_upload(updir, det)
                self._send({"preds": pr, "n": len(pr), "ms": round((time.time()-t0)*1000)})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e), "preds": None}, 500)
        else:
            self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    threading.Thread(target=warm, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8078), H).serve_forever()
