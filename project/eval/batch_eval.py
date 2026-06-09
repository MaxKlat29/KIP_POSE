#!/usr/bin/env python3
"""batch_eval.py — Batch-Eval-Runner: 7 Kombis × N SDG-Seeds → AR IC-BIN + Runtime.

S-012 / T-138. Faehrt die **7 ComposedPipeline-Kombis** (CONTRACT.md §5, inkl. Pipeline A
als Referenz) ueber **N SDG-Seed-Szenen MIT GT** und produziert Max' Vergleichstabelle:
pro Config AR IC-BIN (bop_toolkit, sym-aware, NUR die 2 Anker-Klassen D1) + seg_ms/pose_ms
+ Coverage + Crash-Rate, aggregiert (mean/median/std) ueber die Seeds.

Pro (config, szene):
  1. Gateway-`/predict` (rgb,depth,K,seg_source,pose_source) → instances[{T_cam_obj}] + timings
  2. T_cam_obj (OpenCV-cam, Meter, mesh→cam) → pose_result-Welt-Frame
     (`bop_pose_to_world`; round-trip gegen `world_to_bop_cam` getestet, S-003)
  3. pose_result → BOP-results-CSV-Zeilen (`compare_pipelines.doc_to_bop_rows`)
  4. eval_bop.py --icbin → AR IC-BIN (offizielles BOP19/IC-BIN-Localisation-Protokoll)

**Code jetzt, echter GPU-Lauf spaeter** (Training laeuft, Services nicht permanent
deployed). Das Modul ist FASTAPI-FREI + komplett MOCK-INJIZIERBAR:
  * `predict_fn(config, scene) -> {"instances":[...], "timings":{seg_ms,pose_ms}}`
    Default = HTTP gegen das Mesh-Gateway (`http_predict`); Tests injizieren einen Mock.
  * `eval_fn(csv_path, scene_dir, out_dir) -> report_json`
    Default = subprocess gegen `box_src/eval_bop.py --icbin` (nur auf der GPU-Box);
    Tests injizieren einen Fixture-Scorer.

Persistenz: `project/temp/batch_eval/<run-id>/` → `results.json` (7×N-Matrix) + `EVAL.md`.
Idempotent: re-run mit derselben run-id patcht die Matrix (kein Doppel-Append).

CLI (der reale Post-Training-Lauf — vom Repo-Root, das Script self-pfadet project/):
  python3 project/eval/batch_eval.py \
      --scenes-dir /mnt/data/kip_pose/project/bop/pose_isaac/val --seeds 20 \
      --gateway http://localhost:8090 \
      --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
      --out project/temp/batch_eval
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# Reuse the round-trip-tested world<->cam conversion + BOP-row builder.
from compare_pipelines import doc_to_bop_rows  # noqa: E402


# ── Die Configs — ABGELEITET aus combos.FEASIBLE_COMBOS (T-138-PIVOT: ALLE feasible
# Kombis ~12, registry-Kreuzprodukt, NICHT fix 7). combos liefert n/id/seg/pose/seg_id/
# pose_id/needs_depth/is_pipeline_a/recommended/degraded/class_ambiguity; wir reichern
# NUR die Gateway-seg_source/pose_source (mesh/gateway/app.py) + die FE-Verfahren-Note
# an. So driftet EVAL_CONFIGS nie von der feasibility-Wahrheit (Anti-Drift); ein neues
# Seg-/Pose-Modul in combos.SEG_SOURCES/POSE_SOURCES taucht automatisch auf. ─────────

# combos-seg-id (Mesh-/FE-Name) → Gateway-seg_source (INFER_SOURCES-id, app.py).
#   * yolo-seg == "yolo" (mask-emittierender YOLO26m-seg-Service).
#   * sam3 == "sam3".
#   * yolo-obb == "yolo-obb": die echte OBB-Quelle. Fuer die 3 yolo-obb-MESH-Kombis
#     (FP/GigaPose, die die Maske konsumieren) routet der Runner ueber den mask-Pfad
#     "yolo" (siehe _build_eval_configs); NUR Pipeline A (yolo-obb→gdrnpp) braucht die
#     echten orientierten Boxen "yolo-obb" → gdrnpp-svc liest `obb` (S-004) nativ.
_SEG_SOURCE = {"yolo-obb": "yolo", "yolo-seg": "yolo", "sam3": "sam3"}
# FE-Verfahren-Note (Lena batch.js NOTE_BY-Degrade-Fallback) pro pose_id.
_NOTE = {"gdrnpp": "Pipeline A", "foundationpose": "RGB-D 6DoF",
         "gigapose_rgbd": "coarse+ICP", "gigapose_rgb": "coarse"}


def _build_eval_configs():
    from pipelines.combos import FEASIBLE_COMBOS
    out = []
    for w in FEASIBLE_COMBOS:
        pose_source = "gdrnpp" if w["is_pipeline_a"] else w["pose_id"]
        # T-157: Pipeline A (yolo-obb→gdrnpp) faehrt im EVAL-Pfad jetzt SEHR WOHL uebers
        # Gateway — mit der echten OBB-Quelle "yolo-obb" (NICHT dem mask-Pfad "yolo"),
        # damit gdrnpp-svc die orientierten Boxen `obb` nativ liest (S-004) und die
        # echte Pipeline-A-Pose liefert (= der Genauigkeits-Koenig). So bekommt Pipeline
        # A eine nicht-leere AR-Zeile in der Vergleichstabelle, statt einer leeren
        # Referenzzelle. Der LIVE-Pfad (kip_server:8077 pipeline=gdrnpp) bleibt davon
        # voellig unberuehrt — er nutzt weiterhin den Live-Monolith (e2e_infer).
        seg_source = "yolo-obb" if w["is_pipeline_a"] else _SEG_SOURCE[w["seg_id"]]
        # Verfahren-Note: "Pipeline A" NUR fuer die echte Kombi 1; die degradierten
        # GDRNPP-Kombis (yolo-seg/sam3 → gdrnpp) sind NICHT Pipeline A.
        note = _NOTE[w["pose_id"]]
        if w["pose_id"] == "gdrnpp" and not w["is_pipeline_a"]:
            note = "GDRNPP (degr.)"
        out.append({
            "n": w["n"], "id": w["id"], "seg": w["seg"], "pose": w["pose"],
            "seg_source": seg_source, "pose_source": pose_source,
            "needs_depth": w["needs_depth"], "is_pipeline_a": w["is_pipeline_a"],
            "note": note,
            # PIVOT-Flags für die Tabelle (FE markiert recommended/degraded/ambig).
            "recommended": w["recommended"],
            "degraded": w["degraded"], "degraded_reason": w["degraded_reason"],
            "class_ambiguity": w["class_ambiguity"],
        })
    return out


EVAL_CONFIGS = _build_eval_configs()

# §6: lowercase Mesh-Klasse → BOP obj_id (= numerisch konsistent ueber das Mesh).
# Identisch zu composed.CLASS_TO_OBJ_ID (Single-Source des 2-Klassen-Scope D1).
CLASS_TO_OBJ_ID = {"anker_kurz": 1, "anker_lang": 2}

# sam3-Default-Prompts (combos._SAM3_PROMPTS) — sam3 ist promptable.
SAM3_PROMPTS = {"anker_kurz": "short anchor metal part",
                "anker_lang": "long anchor metal part"}

TABLE_ORIGIN_M = (0.0, 0.0, 0.08)   # e2e_infer.TABLE_ORIGIN_SCENE (Welt-Nullpunkt, m)


def config_key(cfg: dict) -> str:
    """Stabiler, EINDEUTIGER run_config_id (FE: `run_config_id`, BEST-Highlight/Sort).

    Nutzt die combo-`id` (combos.FEASIBLE_COMBOS, = "<seg_id>__<pose_id>" bzw. "gdrnpp")
    — die ist ueber alle ~12 feasiblen Kombis eindeutig. Die Gateway-source-Labels
    (seg_source/pose_source) sind es NICHT (yolo-obb + yolo-seg teilen seg_source 'yolo')."""
    cid = cfg.get("id")
    if cid:
        return cid
    return f"{cfg['seg_source']}__{cfg['pose_source']}"


# ── Stage A: Gateway-/predict instances → pose_result-Welt-Doc ───────────────────
def instances_to_doc(instances: list, camera: dict, source_image: str,
                     table_origin=TABLE_ORIGIN_M, warn=None, snap: bool = False) -> dict:
    """Gateway-`/predict`-instances [{class,T_cam_obj}] → pose_result-Welt-Doc.

    T_cam_obj→Welt + §6-Klassenmapping, OHNE die Seg/Pose-Stages erneut zu fahren
    (die Posen kommen schon vom Gateway). Nur Klassen mit obj_id-Mapping (2-Klassen-
    Scope §6); unbekannte still gefiltert.

    **Boden-Snap (T-163): im EVAL-/Vergleichspfad AUS (snap=False default).**
    `planar_z_snap` ist ein LIVE-VIEWER-Cosmetic (Teile auf die gerenderte Tischebene
    ziehen) und hardcodet `table_z=0.0`. Im pose_isaac-Eval ruhen die GT-Teile aber bei
    Welt-z ≈ -0.075 m (= -table_origin.z), NICHT 0 — der Snap hob darum JEDE Pose um
    +75–79 mm an. Effekt: das 2D reprojiziert noch (MSPD ok), aber der 3D-Punktabstand
    sprengt jede MSSD-Schwelle → **AR_MSSD = 0** über alle Kombis (gemessen: snap=True
    AR_MSSD 0.000 vs snap=False 0.796, median MSSD 78 mm → 5 mm, 10 Szenen). Die
    etablierte 0.87-Baseline (T-109, preds_best.csv) wurde auf RAW GDRNPP-Posen OHNE
    Snap gemessen — der Eval muss apples-to-apples ungesnappt scoren. Der Live-Viewer
    (kip_server, mit korrekt berechnetem table_z + Guard) und das Gateway-`/predict`
    bleiben unberührt. snap=True bleibt für Aufrufer wählbar.

    **Single-Source-Delegation:** Existiert S-013's `gateway_proxy.
    gateway_predict_to_pose_result` (= dieselbe §1/§6-Mathematik via
    `composed.tcamobj_to_world_entry`, geteilt mit ComposedPipeline + dem
    /api/predict-Proxy), delegieren wir dorthin — KEINE zweite Mapping-Implementierung
    (S-013-Docstring). Vor dem S-013-Merge faellt es auf den self-contained Pfad
    `_instances_to_doc_local` (byte-gleiche Mathematik, round-trip-getestet) zurueck,
    damit dieser Branch eigenstaendig testbar bleibt.

    camera: ein BOP scene_camera-Eintrag {cam_K[9], cam_R_w2c[9], cam_t_w2c[3] mm}.
    Liefert ein assemble_doc-Doc (results[] mit part/R_world/t_world/confidence) —
    genau die Form, die `doc_to_bop_rows` zurueck in den BOP-cam-Frame rechnet.
    """
    try:
        from pipelines.gateway_proxy import gateway_predict_to_pose_result
    except Exception:  # noqa: BLE001 — S-013 noch nicht gemerged → lokaler Pfad
        gateway_predict_to_pose_result = None
    if gateway_predict_to_pose_result is not None:
        return gateway_predict_to_pose_result(
            {"instances": instances}, camera=camera,
            table_origin=list(table_origin), source_image=source_image,
            warn=warn, snap=snap)
    return _instances_to_doc_local(instances, camera, source_image,
                                   table_origin=table_origin, warn=warn, snap=snap)


def _instances_to_doc_local(instances: list, camera: dict, source_image: str,
                            table_origin=TABLE_ORIGIN_M, warn=None, snap: bool = False) -> dict:
    """Self-contained T_cam_obj→Welt-Mapping (Fallback, byte-gleich zu S-013/composed).

    Spiegelt `composed.ComposedPipeline.infer`: §1 (T in Meter → mm,
    `bop_pose_to_world`), Boden-Snap (T-163: im Eval AUS, snap=False default — siehe
    `instances_to_doc`), §6 (obj_id→CamelCase-Part → `canonicalize_rotation` ueber
    PART_SYMMETRY). Round-trip gegen `world_to_bop_cam` getestet
    (test_batch_eval.test_coordframe_roundtrip_translation_preserved)."""
    import numpy as np
    from pipelines import contract
    from bop_adapter import (
        bop_pose_to_world, planar_z_snap, canonicalize_rotation, part_for_obj_id,
    )

    warn = warn or (lambda *a, **k: None)
    cam = camera or {}
    R_w2c = cam.get("cam_R_w2c")
    t_w2c = cam.get("cam_t_w2c")
    if R_w2c is None or t_w2c is None:
        raise ValueError(
            "instances_to_doc braucht camera={cam_R_w2c,cam_t_w2c} (BOP scene_camera). "
            "Ohne Extrinsics kein Welt-Frame → eval_bop wuerde still 0 matchen.")
    to = list(table_origin)

    entries = []
    for inst in instances:
        cls = str(inst.get("class", "")).lower()
        if cls not in CLASS_TO_OBJ_ID:
            continue                                  # 2-Klassen-Scope (§6)
        T = inst.get("T_cam_obj")
        if T is None:
            continue
        obj_id = CLASS_TO_OBJ_ID[cls]
        part = part_for_obj_id(obj_id)                # "anker_kurz" → "Anker_Kurz" (§6)

        T = np.asarray(T, float).reshape(4, 4)
        R_m2c = T[:3, :3]
        t_m2c_mm = T[:3, 3] * 1000.0                  # §1: T ist in METER → mm

        R_world, t_world = bop_pose_to_world(R_m2c, t_m2c_mm, R_w2c, t_w2c, to)

        # Boden-Snap (T-163: im Eval AUS, snap default False — der Snap mit
        # hardcodetem table_z=0.0 hebt jede Pose +75–79 mm an und killt AR_MSSD;
        # best-effort, fehlt das CAD lokal → ohnehin kein Snap).
        if snap:
            verts = _mesh_verts_for(obj_id, warn)
            if verts is not None:
                t_world, _dz = planar_z_snap(R_world, t_world, verts, table_z=0.0)
        R_world = canonicalize_rotation(R_world, part)  # §6: continuous-sym Anker

        bbox = inst.get("bbox_2d")
        entries.append({
            "instance_id": int(inst.get("id", len(entries))),
            "part": part,
            "R_world": [float(x) for x in np.asarray(R_world, float).reshape(9)],
            "t_world": [float(x) for x in np.asarray(t_world, float).reshape(3)],
            "confidence": float(inst.get("conf") if inst.get("conf") is not None else 1.0),
            "bbox_2d": [int(x) for x in bbox] if bbox else [0, 0, 0, 0],
        })
    return contract.assemble_doc(source_image, entries, table_origin=to)


def _mesh_verts_for(obj_id: int, warn):
    """CAD-Body-Vertices (Meter) fuer den Boden-Snap. None wenn kein CAD lokal."""
    try:
        from e2e_infer import _load_mesh_verts_m
        return _load_mesh_verts_m(int(obj_id), warn=warn)
    except Exception:  # noqa: BLE001 — CAD-Mesh ist best-effort
        return None


# ── Default predict_fn: HTTP gegen das Mesh-Gateway-/predict ─────────────────────
def http_predict(gateway_url: str, iterations: int = 5, top_n=None, timeout: float = 900.0):
    """Fabrik fuer eine predict_fn(config, scene) -> {instances, timings}.

    Spricht das Mesh-Gateway-`/predict` (multipart) — die Quelle der Wahrheit fuer
    seg+pose + die seg_ms/pose_ms-Telemetrie (gateway/app.py timings).

    **Pipeline A** (`is_pipeline_a`, Kombi 1 yolo-obb→gdrnpp) laeuft im EVAL-Pfad jetzt
    EBENFALLS ueber das Gateway (T-157): seg_source='yolo-obb' + pose_source='gdrnpp'
    OHNE `degraded` → das Gateway segmentiert mit yolo-obb-svc (echte orientierte Boxen)
    und reicht das `obb`-Feld an gdrnpp-svc durch (CONTRACT §4, app.py:566), das es nativ
    liest (S-004). Ergebnis = die echte Pipeline-A-Pose (Genauigkeits-Koenig) als nicht-
    leere AR-Zeile in der Tabelle. **Der LIVE-Pfad (kip_server:8077 pipeline=gdrnpp)
    bleibt der byte-identische Live-Monolith — er beruehrt das Gateway NIE.** Hier geht
    es ausschliesslich um die Eval-/Vergleichstabelle.

    **GDRNPP-degraded** (yolo-seg→gdrnpp, sam3→gdrnpp, pose_source='gdrnpp' aber
    NICHT is_pipeline_a) laeuft mit `degraded=true` (T-153 Gateway-Opt-in), damit der
    gewaehlte mask-emittierende seg_source erhalten bleibt und gdrnpp-svc die AABB aus
    der Maske ableitet (dokumentierter Fallback). Pipeline A setzt `degraded` NICHT —
    sie nutzt die echten obb-Boxen, NICHT den AABB-aus-Maske-Fallback. So bekommen sowohl
    Pipeline A (echte obb) als auch die degraded-Kombis (AABB) echte AR-Zeilen.

    Lazy `import httpx` — kein Pin in project/requirements.txt (Box-Stack).
    """
    base = gateway_url.rstrip("/")

    def _predict(cfg: dict, scene: dict) -> dict:
        # T-157: Pipeline A geht im Eval JETZT auch uebers Gateway (seg_source=yolo-obb,
        # pose_source=gdrnpp, degraded=false → echte obb → gdrnpp-svc nativ). Nur die
        # degraded gdrnpp-Kombis (yolo-seg/sam3 → gdrnpp) setzen degraded=true.
        import httpx
        # Bytes vorab lesen (kleine PNGs) — keine offenen File-Handles ueber den
        # Netz-Call (die leakten, wenn client.post wirft).
        files = {"rgb": ("rgb.png", pathlib.Path(scene["rgb"]).read_bytes(), "image/png")}
        if cfg["needs_depth"] and scene.get("depth"):
            files["depth"] = ("depth.png", pathlib.Path(scene["depth"]).read_bytes(), "image/png")
        K = scene["K"]                                # {fx,fy,cx,cy}
        data = {
            "fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"],
            "iterations": iterations,
            "seg_source": cfg["seg_source"], "pose_source": cfg["pose_source"],
        }
        # BOP/SDG depth PNGs are written with a depth_scale (png*depth_scale = mm).
        # Forward it so the gateway+refiner decode metres = png*depth_scale/1000.
        # Dropping it decodes depth 10x too far (depth_scale=0.1) -> the ~2.4m
        # lateral X error that zeroed every RGB-D combo's AR (T-156).
        if cfg["needs_depth"] and scene.get("depth_scale") is not None:
            data["depth_scale"] = scene["depth_scale"]
        # degraded gdrnpp-Kombis: opt-in, damit das Gateway den mask-seg_source nicht
        # auf yolo-obb zurueckzwingt (mask→gdrnpp AABB-Fallback).
        if cfg["pose_source"] == "gdrnpp" and not cfg.get("is_pipeline_a"):
            data["degraded"] = "true"
        if top_n is not None:
            data["top_n"] = top_n
        if cfg["seg_source"] == "sam3":
            data["seg_prompts"] = json.dumps(SAM3_PROMPTS)
        with httpx.Client(timeout=timeout) as client:
            r = client.post(base + "/predict", data=data, files=files)
            r.raise_for_status()
            body = r.json()
        return {"instances": body.get("instances", []),
                "timings": body.get("timings", {})}

    return _predict


class PipelineANotOnGateway(Exception):
    """Eine predict_fn signalisiert: diese Kombi laeuft NICHT uebers Gateway.

    **T-157:** Der Default-`http_predict` wirft das NICHT MEHR — Pipeline A faehrt im
    Eval jetzt ueber das Gateway (seg_source=yolo-obb→gdrnpp nativ, echte obb). Die
    Exception bleibt als testbare Naht erhalten: `run_one` faengt sie weiterhin (ok=False,
    error='pipeline_a_no_gateway', zaehlt NICHT als echter Versuch), damit eine custom
    predict_fn eine Kombi explizit vom Gateway-Scoring ausnehmen kann (z.B. wenn
    yolo-obb-svc nicht deployt ist und man Pipeline A bewusst skippen will). Im realen
    12×N-Lauf wird sie fuer Pipeline A nicht mehr ausgeloest → echte AR-Zeile."""


# ── Default eval_fn: subprocess gegen box_src/eval_bop.py --icbin ────────────────
_EVAL_BOP = pathlib.Path("/mnt/data/kip_pose/box_src/eval_bop.py")
_BOP_VENV = pathlib.Path("/mnt/data/bop/bop-venv/bin/python")


def subprocess_eval(dataset_dir: str, split: str = "val",
                    eval_bop=_EVAL_BOP, bop_venv=_BOP_VENV, n_points: int = 2000):
    """Fabrik fuer eine eval_fn(csv_path, scene_dir, out_dir) -> report_json.

    Ruft `eval_bop.py --icbin` (offizielles BOP19/IC-BIN-Localisation-Protokoll,
    sym-aware, multi-instance) gegen das GT-Dataset. NUR auf der GPU-Box (eval_bop +
    bop-venv + bop_toolkit_lib).

    **Denominator-Scoping (T-152, ehrliche AR):** IC-BIN ist ein RECALL-Protokoll —
    der Nenner ist die Summe der `inst_count` ueber die im targets-File gelisteten
    (scene,im)-Paare. Der Batch-Lauf praediziert NUR die discover_scenes-Teilmenge
    (im=0 pro Szene, ~10 Bilder), nicht den ganzen Split (1000 Bilder). Ohne Scoping
    rechnet eval_bop den Recall gegen ALLE 1000 Bilder → AR ~0 (strukturell, nicht
    Modell-Versagen). Wir bauen darum pro Lauf ein **scoped targets-File** aus genau
    den (scene_id, im_id)-Paaren der CSV (= die praedizierten Bilder), gefiltert aus
    dem vollen Dataset-targets-File, und reichen es via `--targets`. eval_bop wertet
    dann GENAU diese Bilder als Nenner (official-BOP-Semantik: das targets-File IST
    die Eval-Bild-Liste). Faellt das volle targets-File, faellt es auf den ungescopten
    Default zurueck (eval_bop loest ihn selbst auf).
    """
    import subprocess

    full_targets = _full_targets_path(dataset_dir, split)

    def _eval(csv_path: str, scene_dir: str, out_dir: str) -> dict:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(bop_venv), str(eval_bop),
               "--dataset-dir", str(dataset_dir), "--split", split,
               "--icbin", "--preds", str(csv_path),
               "--n-points", str(n_points), "--out", str(out)]
        scoped = _scoped_targets_for_csv(csv_path, full_targets, out)
        if scoped is not None:
            cmd += ["--targets", str(scoped)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"eval_bop --icbin FAIL: {r.stderr[-400:]}")
        return json.load(open(out / "report.json"))

    return _eval


def _full_targets_path(dataset_dir: str, split: str):
    """Default BOP targets-File <dataset>/test_targets_<dataset>_<split>.json o. None."""
    ds = pathlib.Path(dataset_dir)
    cand = ds / f"test_targets_{ds.name}_{split}.json"
    return cand if cand.is_file() else None


def _scoped_targets_for_csv(csv_path, full_targets, out_dir):
    """Scoped targets-File: die targets-Zeilen der (scene_id,im_id)-Paare der CSV.

    Liefert den Pfad zum geschriebenen scoped-File, oder None wenn kein volles
    targets-File da ist (→ ungescopt, eval_bop nutzt seinen Default). Robust gegen
    eine leere/header-only CSV (Pipeline-A-Referenz) — dann gibt es keine Paare und
    wir scopen nicht (der Aufrufer ruft eval_bop ohnehin nur bei n_ok>0)."""
    import csv as _csv
    if not full_targets:
        return None
    pairs = set()
    try:
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                try:
                    pairs.add((int(row["scene_id"]), int(row["im_id"])))
                except (KeyError, ValueError, TypeError):
                    continue
    except FileNotFoundError:
        return None
    if not pairs:
        return None
    rows = json.loads(pathlib.Path(full_targets).read_text())
    scoped = [t for t in rows if (int(t["scene_id"]), int(t["im_id"])) in pairs]
    if not scoped:
        return None
    sp = pathlib.Path(out_dir) / "targets_scoped.json"
    sp.write_text(json.dumps(scoped))
    return sp


def active_class_parts() -> list:
    """Die D1-aktiven Klassen-Parts (= die Hauptmetrik-Klassen), generisch.

    Single-Source: die Parts der `CLASS_TO_OBJ_ID`-Werte (= der aktive 2-Klassen-
    Scope D1: Anker_Kurz, Anker_Lang). **Nicht hartcodiert 2** — kommt spaeter eine
    dritte trainierte Klasse in CLASS_TO_OBJ_ID dazu (z.B. Zahnrad → obj_id 6), mittelt
    die Hauptmetrik automatisch ueber 3. Die 4 untrainierten BOP-Klassen
    (Buerstenhalter/Getriebe/Ringmagnet/Zahnrad) stehen NICHT in CLASS_TO_OBJ_ID und
    bleiben damit aus der primaeren AR raus (sie haben AR=0 und wuerden die Zahl 3x
    verwaessern — T-171)."""
    from bop_adapter import part_for_obj_id
    return [part_for_obj_id(oid) for oid in CLASS_TO_OBJ_ID.values()]


def _mean_active(per_class: dict) -> "float | None":
    """Mittel der AR ueber die D1-aktiven Klassen-Parts (= die primaere/Haupt-AR).

    Mittelt GENAU ueber `active_class_parts()` — case-insensitiv gematcht gegen die
    `per_class`-Keys (eval_bop schreibt CamelCase 'Anker_Kurz', aeltere Reports/
    Fixtures evtl. lowercase). Eine aktive Klasse, die im Report fehlt, zaehlt als 0.0
    (kein Recall = 0, nicht weglassen — sonst wuerde sam3 mit nur Anker_Kurz faelschlich
    hoch erscheinen statt (0.8187+0.0)/2). None wenn per_class leer ist oder keine
    aktive Klasse aufgeloest werden kann (kein stiller 0-Wert)."""
    if not per_class:
        return None
    lut = {str(k).lower(): v for k, v in per_class.items()}
    vals = [float(lut.get(p.lower(), 0.0)) for p in active_class_parts()]
    if not vals:
        return None
    return round(statistics.fmean(vals), 4)


def ar_from_report(report: dict) -> "tuple[float | None, dict, float | None]":
    """Primaere AR IC-BIN (D1-aktive Klassen) + per-Klasse + 6-obj-AR aus report.json.

    report.json (eval_bop --icbin): {"results":{"per_object":{obj_id:{AR,name,...}},
    "overall":{"AR":...}}}.

    **T-171 — primaere AR = Mittel der aktiven D1-Klassen, NICHT der 6-obj-overall.AR:**
    eval_bop mittelt `overall.AR` ueber ALLE Objekte im targets-File — inkl. der 4
    untrainierten Klassen (Buerstenhalter/Getriebe/Ringmagnet/Zahnrad, AR=0). Das zog
    die angezeigte AR ~3x runter (gdrnpp 0.295 statt 0.886). Die Hauptmetrik ist jetzt
    `_mean_active(per_class)` = Mittel ueber die `CLASS_TO_OBJ_ID`-aktiven Klassen (= der
    2-Klassen-Scope D1, generisch). Die alte 6-obj-Zahl bleibt als `ar_6obj` erhalten.

    Liefert (primary_ar, {part: AR}, ar_6obj):
      * primary_ar = Mittel der aktiven D1-Klassen (die Haupt-AR; None wenn kein
        per_class oder keine aktive Klasse aufloesbar).
      * per_class  = {part: AR} ALLER Objekte im Report (FE-Spalte + Transparenz).
      * ar_6obj    = overall.AR (eval_bop ueber alle Objekte; None wenn nicht gescort)."""
    res = (report or {}).get("results", report or {})
    overall = res.get("overall", {})
    ar6_raw = overall.get("AR")
    ar_6obj = float(ar6_raw) if isinstance(ar6_raw, (int, float)) else None
    per_class = {}
    for oid, po in res.get("per_object", {}).items():
        name = po.get("name") or str(oid)
        if po.get("AR") is not None:
            per_class[name] = round(float(po["AR"]), 4)
    primary_ar = _mean_active(per_class)
    return primary_ar, per_class, ar_6obj


# ── Stage B: ein (config, szene) — predict → doc → CSV → eval ────────────────────
def run_one(cfg: dict, scene: dict, predict_fn, warn=None) -> dict:
    """Eine (config, szene): Gateway-/predict → Welt-Doc → BOP-CSV-Zeilen + timings.

    Gibt {ok, rows, seg_ms, pose_ms, n_instances, error}. Ein Crash (predict-Fehler,
    Service-503) wird GEFANGEN (ok=False, error) — die Crash-Rate ist eine Achse,
    kein Abbruch. Pipeline-A-ohne-Gateway (PipelineANotOnGateway) ist KEIN Crash:
    ok=False mit error="pipeline_a_no_gateway" (der Runner zaehlt das separat).
    """
    warn = warn or (lambda *a, **k: None)
    try:
        out = predict_fn(cfg, scene)
    except PipelineANotOnGateway:
        return {"ok": False, "rows": [], "seg_ms": None, "pose_ms": None,
                "n_instances": 0, "error": "pipeline_a_no_gateway"}
    except Exception as e:  # noqa: BLE001 — Crash ist eine Mess-Achse
        warn(f"[batch_eval] {config_key(cfg)} :: {scene.get('scene_id')} CRASH: {e}")
        return {"ok": False, "rows": [], "seg_ms": None, "pose_ms": None,
                "n_instances": 0, "error": str(e)[:200]}

    instances = out.get("instances", [])
    timings = out.get("timings", {})
    doc = instances_to_doc(instances, scene["camera"], scene["rgb"],
                           warn=warn)
    rows = doc_to_bop_rows(doc, scene["camera"],
                           scene_id=scene["scene_id"], im_id=scene.get("im_id", 0))
    return {"ok": True, "rows": rows,
            "seg_ms": timings.get("seg_ms"), "pose_ms": timings.get("pose_ms"),
            "n_instances": len(instances), "error": None}


# ── Stage C: aggregate ueber die Seeds pro Config ────────────────────────────────
def _stats(xs):
    """(mean, median, std) ueber nicht-None-Werte; alle None → (None,None,None)."""
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None, None, None
    mean = statistics.fmean(vals)
    med = statistics.median(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return round(mean, 4), round(med, 4), round(std, 4)


def aggregate_config(cfg: dict, per_scene: list, ar_mean=None, ar_std=None,
                     per_class=None, ar_6obj=None) -> dict:
    """Aggregiert die N Seed-Ergebnisse einer Config zur FE-Config-Zeile.

    per_scene: Liste von run_one-Outputs (ein Eintrag pro Seed). AR kommt aus EINEM
    eval_bop-Lauf ueber die gepoolte CSV (multi-instance IC-BIN braucht das ganze
    Dataset) — daher ar_mean/ar_std hier reingereicht (nicht aus per_scene).

    **T-171:** `ar_mean` ist die **primaere** AR = Mittel der D1-aktiven Klassen
    (anker_kurz/lang), NICHT der eval_bop-6-obj-overall. `ar_6obj` traegt die alte
    6-Objekt-Zahl als sekundaeres Feld weiter (Transparenz). `per_class` zeigt alle 6.

    Liefert exakt Lenas batch.js-Felder:
      {seg,pose, modality, ar_mean,ar_std, ar_6obj, seg_ms,pose_ms, coverage,crash_rate,
       note, is_pipeline_a, run_config_id, per_class?}
    coverage/crash_rate ∈ 0..1. modality (T-159) = "RGB"|"RGBD" (FE Input-Spalte).
    """
    n = len(per_scene)
    # "echte" Versuche (Pipeline-A-ohne-Gateway zaehlt NICHT als Versuch — apples).
    real = [s for s in per_scene if s.get("error") != "pipeline_a_no_gateway"]
    n_real = len(real)
    oks = [s for s in real if s["ok"]]
    n_crash = sum(1 for s in real if not s["ok"])
    n_with_inst = sum(1 for s in oks if s["n_instances"] > 0)

    seg_mean, _, _ = _stats([s["seg_ms"] for s in oks])
    pose_mean, _, _ = _stats([s["pose_ms"] for s in oks])

    crash_rate = round(n_crash / n_real, 4) if n_real else None
    # Coverage = Anteil der (erfolgreichen) Szenen mit >=1 detektierten Instanz.
    coverage = round(n_with_inst / n_real, 4) if n_real else None

    row = {
        "seg": cfg["seg"], "pose": cfg["pose"],
        "modality": _modality(cfg),       # T-159: RGB | RGBD (FE Input-Spalte)
        "ar_mean": ar_mean, "ar_std": ar_std,   # T-171: ar_mean = D1-aktive Klassen
        "ar_6obj": ar_6obj,               # T-171: sekundaer = eval_bop 6-obj-overall
        "seg_ms": seg_mean, "pose_ms": pose_mean,
        "coverage": coverage, "crash_rate": crash_rate,
        "note": cfg.get("note"),
        "is_pipeline_a": bool(cfg.get("is_pipeline_a")),
        "run_config_id": config_key(cfg),
        # PIVOT-Flags (T-138): FE markiert recommended (kuratierte 7) + degraded
        # (gdrnpp AABB-aus-Maske) + class_ambiguity (sam3 kurz/lang schwach, S006).
        "recommended": bool(cfg.get("recommended")),
        "degraded": bool(cfg.get("degraded")),
        "degraded_reason": cfg.get("degraded_reason"),
        "class_ambiguity": bool(cfg.get("class_ambiguity")),
        "n_scenes": n, "n_real": n_real, "n_ok": len(oks), "n_crash": n_crash,
    }
    if per_class:
        row["per_class"] = per_class
    return row


# ── Stage C2: Live-Akkumulator pro Config (T-153 — inkrementelle Standings) ───────
class _ConfigAcc:
    """Laufender Zustand einer Config waehrend des seed-major Round-Robin-Laufs.

    Sammelt pro (config, szene) die BOP-CSV-Zeilen + timings + ok/crash-Flags, haelt
    eine **running mean** fuer seg_ms/pose_ms (T-153 verlangt keinen Voll-Rescan) und
    den zuletzt auf der **akkumulierten** CSV neu berechneten AR IC-BIN. Aus diesem
    Zustand baut `standings_entry` die Live-Scoreboard-Zeile (Contract /api/eval/job).
    """

    __slots__ = ("cfg", "key", "rows", "n_scenes", "n_real", "n_ok", "n_crash",
                 "n_with_inst", "_seg_sum", "_seg_cnt", "_pose_sum", "_pose_cnt",
                 "ar", "per_class", "ar_6obj")

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.key = config_key(cfg)
        self.rows: list = []          # akkumulierte BOP-CSV-Zeilen ueber alle Szenen
        self.n_scenes = 0             # (config, szene)-Versuche, gesamt
        self.n_real = 0               # echte Versuche (Pipeline-A-ohne-Gateway zaehlt NICHT)
        self.n_ok = 0
        self.n_crash = 0
        self.n_with_inst = 0
        self._seg_sum = 0.0; self._seg_cnt = 0    # running mean seg_ms (nur oks)
        self._pose_sum = 0.0; self._pose_cnt = 0  # running mean pose_ms (nur oks)
        self.ar = None                # primaere AR IC-BIN (D1-aktive Klassen, T-171)
        self.per_class = None
        self.ar_6obj = None           # sekundaer: eval_bop overall.AR ueber alle 6 Objekte

    def add_scene(self, res: dict) -> None:
        """Eine (config, szene): run_one-Output in die Akkumulatoren falten."""
        self.n_scenes += 1
        if res.get("error") == "pipeline_a_no_gateway":
            return                                # kein echter Versuch (apples-to-apples)
        self.n_real += 1
        if res["ok"]:
            self.n_ok += 1
            if res["n_instances"] > 0:
                self.n_with_inst += 1
            if res.get("seg_ms") is not None:
                self._seg_sum += float(res["seg_ms"]); self._seg_cnt += 1
            if res.get("pose_ms") is not None:
                self._pose_sum += float(res["pose_ms"]); self._pose_cnt += 1
            self.rows.extend(res["rows"])
        else:
            self.n_crash += 1

    @property
    def seg_ms(self):
        return round(self._seg_sum / self._seg_cnt, 4) if self._seg_cnt else None

    @property
    def pose_ms(self):
        return round(self._pose_sum / self._pose_cnt, 4) if self._pose_cnt else None

    @property
    def coverage(self):
        return round(self.n_with_inst / self.n_real, 4) if self.n_real else None

    @property
    def crash_rate(self):
        return round(self.n_crash / self.n_real, 4) if self.n_real else None

    def standings_entry(self) -> dict:
        """Eine Scoreboard-Zeile (OHNE rank — den setzt `build_standings` global).

        Contract-Felder (T-153, /api/eval/job.standings[]): config_key, seg, pose,
        modality, ar, ar_std, n_scenes, seg_ms, pose_ms, coverage, crash_rate,
        recommended, degraded, degraded_reason, class_ambiguity, is_pipeline_a.

        `ar` (T-171) = die **primaere** AR = Mittel ueber die D1-aktiven Klassen (NICHT
        die 6-obj-overall.AR — die zog die Zahl ~3x runter). `ar_6obj` traegt die alte
        6-Objekt-Zahl als sekundaeres Feld weiter (Transparenz, nicht primaer).

        `modality` (T-159) = "RGB" | "RGBD" (FE Input-Spalte, aus needs_depth).

        `seg`/`pose` sind die kompakten Source-Ids des Contracts (seg = combo-seg-id
        'yolo-seg', pose = Gateway-`pose_source` 'foundationpose') — NICHT die FE-
        Display-Labels (cfg['pose'] = 'FoundationPose'). seg_ms/pose_ms gerundet auf
        ganze ms (FE-Tabelle). ar_std = 0.0 wenn gescort (IC-BIN ist ein Set-Score,
        kein Per-Szene-Mittel), sonst null."""
        cfg = self.cfg
        ar = self.ar
        return {
            "config_key": self.key,
            "seg": cfg["seg"], "pose": cfg["pose_source"],
            "modality": _modality(cfg),       # T-159: RGB | RGBD (FE Input-Spalte)
            "ar": ar,                         # T-171: primaer = D1-aktive Klassen
            "ar_6obj": self.ar_6obj,          # T-171: sekundaer = eval_bop 6-obj-overall
            "ar_std": 0.0 if ar is not None else None,
            "n_scenes": self.n_ok,            # gescorte Szenen (= was in der CSV steht)
            "seg_ms": _round_ms(self.seg_ms), "pose_ms": _round_ms(self.pose_ms),
            "coverage": self.coverage, "crash_rate": self.crash_rate,
            "recommended": bool(cfg.get("recommended")),
            "degraded": bool(cfg.get("degraded")),
            "degraded_reason": cfg.get("degraded_reason"),
            "class_ambiguity": bool(cfg.get("class_ambiguity")),
            "is_pipeline_a": bool(cfg.get("is_pipeline_a")),
        }


def _round_ms(v):
    return None if v is None else int(round(v))


def _modality(cfg: dict) -> str:
    """Input-Modalitaet einer Kombi (T-159, FE Input-Spalte): "RGBD" wenn die Pose
    Depth braucht (FoundationPose + GigaPose-3D), sonst "RGB" (GDRNPP + GigaPose-2D).

    Abgeleitet aus `needs_depth` (combos.FEASIBLE_COMBOS, CONTRACT.md §5) — die Seg-
    Stufe ist immer RGB, die Pose-Stufe entscheidet. Single-Source, kein Netz."""
    return "RGBD" if cfg.get("needs_depth") else "RGB"


def build_standings(accs) -> list:
    """Sortierte + geranke Live-Standings aus den Config-Akkumulatoren (T-153).

    Vertrag (/api/eval/job.standings[]): sortiert nach `ar` DESC, `rank` ab 1 gesetzt,
    **alle** Configs vertreten — auch noch-nicht-gestartete (n_scenes=0 → ar=null,
    rank ans Ende). Configs ohne AR sortieren stabil ans Ende (nach config_key, damit
    die Reihenfolge deterministisch ist und das FE nicht flackert)."""
    entries = [a.standings_entry() for a in accs]
    # ar=None nach hinten; innerhalb gleicher ar-Gruppe stabil nach config_key.
    entries.sort(key=lambda e: (e["ar"] is None, -(e["ar"] or 0.0), e["config_key"]))
    for rank, e in enumerate(entries, 1):
        e["rank"] = rank
    return entries


# ── Stage D: der ganze Lauf (12 × N, seed-major Round-Robin) ─────────────────────
def run_batch(configs, scenes, predict_fn, eval_fn, out_dir,
              run_id=None, progress=None, warn=None, standings_cb=None) -> dict:
    """Faehrt alle Configs × alle Szenen **seed-major** (Round-Robin), scored
    inkrementell, streamt Live-Standings, aggregiert, persistiert.

    **T-153 Reihenfolge-Pivot:** statt Config-fuer-Config (erst alle N Szenen von
    Kombi 1, dann Kombi 2 ...) laeuft fuer **jeden Seed 1..N: ALLE Configs**. Damit
    hat nach Seed 1 jede Kombi schon 1 Szene → das Scoreboard ist von Anfang an voll
    besetzt und verfeinert sich, statt erst am Ende zu erscheinen.

    **Inkrementelle AR:** nach jeder (config, szene) wird die Zeile an die config-CSV
    angehaengt und der AR IC-BIN dieser Config auf der **akkumulierten** CSV neu
    berechnet (eval_bop --icbin ist schnell). Danach `standings_cb(standings)` mit der
    aktuellen, nach ar sortierten + geranken Tabelle aller Configs.

    Args:
      configs     : Liste von Config-dicts (Default EVAL_CONFIGS = die 12 feasiblen).
      scenes      : Liste von Szenen-dicts {scene_id, im_id, rgb, depth?, camera, K, dir}.
      predict_fn  : (cfg, scene) -> {instances, timings}  (Mock-Naht).
      eval_fn     : (csv_path, scene_dir, out_dir) -> report_json  (Mock-Naht).
      out_dir     : project/temp/batch_eval — der Run landet unter <out_dir>/<run_id>/.
      run_id      : default = "run-<utc-stamp>". Idempotent: gleiche id patcht.
      progress    : optional callback(pct:int, phase:str) fuer die Job-Bar.
      warn        : optional log-callback.
      standings_cb: optional callback(standings:list, n_done:int, n_total:int) —
                    nach jeder gescorten (config, szene) mit den Live-Standings.

    Liefert das results-dict (= /api/eval/result-Body, {run_id,date,configs,standings})
    und schreibt <out_dir>/<run_id>/results.json + EVAL.md.
    """
    warn = warn or (lambda *a, **k: None)
    progress = progress or (lambda *a, **k: None)
    standings_cb = standings_cb or (lambda *a, **k: None)
    run_id = run_id or ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = pathlib.Path(out_dir) / run_id
    csv_dir = run_dir / "csv"
    eval_dir = run_dir / "eval"
    for d in (run_dir, csv_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    accs = [_ConfigAcc(cfg) for cfg in configs]
    n_total = max(1, len(configs) * len(scenes))
    n_done = 0

    # Seed-major Round-Robin: aeussere Schleife = Szene/Seed, innere = alle Configs.
    for si, scene in enumerate(scenes):
        for acc in accs:
            cfg = acc.cfg
            res = run_one(cfg, scene, predict_fn, warn=warn)
            acc.add_scene(res)
            n_done += 1

            # Inkrementelle AR IC-BIN: config-CSV (akkumuliert ueber die bisherigen
            # Seeds) neu schreiben + eval_bop auf dem akkumulierten Set neu rechnen.
            csv_path = csv_dir / f"{acc.key}.csv"
            _write_bop_csv(csv_path, acc.rows)
            if acc.n_ok and acc.rows:
                try:
                    report = eval_fn(str(csv_path), str(scene.get("dir", "")),
                                     str(eval_dir / acc.key))
                    acc.ar, acc.per_class, acc.ar_6obj = ar_from_report(report)
                except Exception as e:  # noqa: BLE001 — Eval-Fehler = AR unbekannt, kein Abbruch
                    warn(f"[batch_eval] eval_bop fuer {acc.key} fehlgeschlagen: {e}")

            pct = int(5 + 90 * n_done / n_total)
            progress(pct, f"Seed {si + 1}/{len(scenes)} · {acc.key} "
                          f"(AR={acc.ar if acc.ar is not None else '—'})")
            standings_cb(build_standings(accs), n_done, n_total)

    # ── Final-Aggregation: results.configs bleibt kompatibel zu /api/eval/result ──
    # (aggregate_config-Form), results.standings = finale Live-Tabelle.
    config_rows = []
    for acc in accs:
        per_scene = _acc_to_per_scene(acc)
        row = aggregate_config(acc.cfg, per_scene, ar_mean=acc.ar,
                               ar_std=(0.0 if acc.ar is not None else None),
                               per_class=acc.per_class, ar_6obj=acc.ar_6obj)
        config_rows.append(row)
        warn(f"[batch_eval] {acc.key}: AR={acc.ar} (6obj={acc.ar_6obj}) cov={row['coverage']} "
             f"crash={row['crash_rate']} seg_ms={row['seg_ms']} pose_ms={row['pose_ms']}")

    duration_s = round(time.time() - t0, 1)
    standings = build_standings(accs)
    results = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": duration_s,
        "n_configs": len(config_rows),
        "n_scenes": len(scenes),
        "configs": config_rows,
        "standings": standings,
    }
    _persist(run_dir, results)
    progress(100, "Lauf fertig")
    standings_cb(standings, n_total, n_total)
    return results


def _acc_to_per_scene(acc: "_ConfigAcc") -> list:
    """Rekonstruiert eine per_scene-Liste fuer `aggregate_config` aus den Acc-Zaehlern.

    `aggregate_config` braucht nur die ok/error/n_instances/seg_ms/pose_ms-Felder zum
    Zaehlen — nicht die echten Per-Szene-Werte (AR kommt separat aus dem Set-Score).
    Wir geben deshalb synthetische Eintraege mit den korrekten Aggregat-Zaehlern: die
    oks tragen die running-mean seg_ms/pose_ms (gleiches Aggregat-Ergebnis), die
    Pipeline-A-ohne-Gateway-Versuche bleiben als solche markiert."""
    out = []
    seg = acc.seg_ms; pose = acc.pose_ms
    for _ in range(acc.n_with_inst):
        out.append({"ok": True, "seg_ms": seg, "pose_ms": pose,
                    "n_instances": 1, "error": None, "rows": []})
    for _ in range(acc.n_ok - acc.n_with_inst):
        out.append({"ok": True, "seg_ms": seg, "pose_ms": pose,
                    "n_instances": 0, "error": None, "rows": []})
    for _ in range(acc.n_crash):
        out.append({"ok": False, "seg_ms": None, "pose_ms": None,
                    "n_instances": 0, "error": "crash", "rows": []})
    for _ in range(acc.n_scenes - acc.n_real):
        out.append({"ok": False, "seg_ms": None, "pose_ms": None,
                    "n_instances": 0, "error": "pipeline_a_no_gateway", "rows": []})
    return out


def _write_bop_csv(path: pathlib.Path, rows: list) -> None:
    """BOP-results-CSV (scene_id,im_id,obj_id,score,R,t,time) — was eval_bop liest."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "im_id", "obj_id",
                                          "score", "R", "t", "time"])
        w.writeheader()
        w.writerows(rows)


def _persist(run_dir: pathlib.Path, results: dict) -> None:
    """Schreibt results.json (maschinell) + EVAL.md (Tabelle). Idempotent (overwrite)."""
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "EVAL.md").write_text(render_markdown(results))


def render_markdown(results: dict) -> str:
    """EVAL.md — eine Zeile pro feasibler Config. Pipeline A = Referenz, kuratierte
    7 = RECOMMENDED (★), degraded (gdrnpp AABB-aus-Maske) + class-ambig (sam3) markiert.

    **T-171:** Die Haupt-Spalte `AR IC-BIN` ist die **2-Klassen-AR** (D1-aktive Klassen,
    Mittel anker_kurz/lang = `ar_mean`). Die alte 6-Objekt-Zahl steht zur Transparenz in
    der Neben-Spalte `AR 6-obj` (`ar_6obj`) — sie war primaer und zog die Zahl ~3x runter
    (gdrnpp 0.295), weil 4 untrainierte BOP-Klassen (AR=0) mitgemittelt wurden."""
    active = ", ".join(active_class_parts())
    n_rec = sum(1 for c in results["configs"] if c.get("recommended"))
    lines = [
        f"# Batch-Eval {results['run_id']}", "",
        f"- Datum: {results['date']}",
        f"- Configs: {results['n_configs']} feasible ({n_rec} recommended) "
        f"· Szenen/Seeds: {results['n_scenes']} · Dauer: {results['duration_s']} s",
        f"- Haupt-AR = Mittel ueber die {len(active_class_parts())} aktiven D1-Klassen "
        f"({active}); `AR 6-obj` = eval_bop ueber alle 6 BOP-Objekte (4 untrainiert, "
        "nur Transparenz).", "",
        "| # | Seg | Pose | Input | AR IC-BIN | ±std | AR 6-obj | seg ms | pose ms "
        "| Coverage | Crash | Verfahren | Flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _f(v, nd=3):
        return "—" if v is None else f"{v:.{nd}f}"

    def _pct(v):
        return "—" if v is None else f"{round(v * 100)}%"

    def _ms(v):
        return "—" if v is None else f"{round(v)}"

    for i, c in enumerate(results["configs"], 1):
        ref = " (Ref)" if c.get("is_pipeline_a") else ""
        flags = []
        if c.get("recommended"):
            flags.append("★empf.")
        if c.get("degraded"):
            flags.append(f"degr.({c.get('degraded_reason') or 'ja'})")
        if c.get("class_ambiguity"):
            flags.append("klassen-ambig")
        lines.append(
            f"| {i}{ref} | {c['seg']} | {c['pose']} | {c.get('modality') or '—'} "
            f"| {_f(c['ar_mean'])} "
            f"| {_f(c['ar_std'])} | {_f(c.get('ar_6obj'))} "
            f"| {_ms(c['seg_ms'])} | {_ms(c['pose_ms'])} "
            f"| {_pct(c['coverage'])} | {_pct(c['crash_rate'])} | {c.get('note') or '—'} "
            f"| {', '.join(flags) or '—'} |"
        )
    lines += ["", "_AR IC-BIN: offizielles BOP19/IC-BIN-Localisation-Protokoll "
              "(sym-aware). **Haupt-AR = Mittel ueber die aktiven D1-Klassen** "
              f"({', '.join(active_class_parts())}); `AR 6-obj` = eval_bop ueber alle 6 "
              "BOP-Objekte (4 untrainiert → AR 0, nur Transparenz, T-171). "
              "Coverage/Crash ∈ 0..1. ★empf. = kuratierte Recommended-7; degr. = GDRNPP "
              "AABB-aus-Maske; klassen-ambig = sam3 trennt kurz/lang schwach (S006)._", ""]
    return "\n".join(lines)


# ── Run-Discovery (fuer /api/eval/runs + /api/eval/result) ───────────────────────
def list_runs(out_dir) -> list:
    """Alle persistierten Laeufe unter <out_dir>, neuester zuerst.

    Liefert exakt Lenas batch.js-runs-Form: [{run_id, date, duration_s, n_configs}].
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        return []
    runs = []
    for rj in root.glob("*/results.json"):
        try:
            r = json.loads(rj.read_text())
        except Exception:  # noqa: BLE001 — kaputten Run ueberspringen
            continue
        runs.append({"run_id": r.get("run_id", rj.parent.name),
                     "date": r.get("date"), "duration_s": r.get("duration_s"),
                     "n_configs": r.get("n_configs")})
    runs.sort(key=lambda x: (x.get("date") or ""), reverse=True)
    return runs


def load_run(out_dir, run_id) -> "dict | None":
    """Das results.json eines Laufs (= /api/eval/result-Body) oder None."""
    rj = pathlib.Path(out_dir) / run_id / "results.json"
    if not rj.is_file():
        return None
    try:
        return json.loads(rj.read_text())
    except Exception:  # noqa: BLE001
        return None


# ── Szenen-Discovery (vorgerenderte SDG-Seeds mit GT) ────────────────────────────
def discover_scenes(scenes_dir, seeds=None, split_layout=True, frames=None) -> list:
    """Findet vorgerenderte SDG-Seed-Szenen MIT GT unter `scenes_dir`.

    Erwartet BOP-Layout: <scenes_dir>/<scene_id>/ mit rgb/<im>.png, depth/<im>.png,
    scene_gt.json, scene_camera.json. `seeds` cappt auf die ersten N SZENEN. Liefert
    {scene_id, im_id, rgb, depth, camera, K, dir}.

    `frames` (T-170, D-5 PIVOT): wenn None (Default) → EIN Eintrag pro Szene (im_ids[0]),
    byte-identisch zum Vor-T-170-Verhalten. Wenn eine Liste von im_ids (z.B.
    [0,10,20,..,90]) → PRO Szene ein Eintrag JE angefragtem im_id, der im scene_camera
    existiert (deterministisches Sub-Sampling fuer robustere AR ueber ~100 statt 10 Frames).
    Die eval-Naht (subprocess_eval) scoped das targets-File ohnehin per (scene_id,im_id),
    also wird jeder Frame korrekt gegen sein scene_gt[im_id] gewertet.

    camera = {cam_K[9], cam_R_w2c[9], cam_t_w2c[3] mm} aus scene_camera.json[im].
    Fehlt cam_R_w2c/cam_t_w2c im scene_camera (BOP speichert Extrinsics oft separat),
    faellt scene_gt_world.json / camera.json zurueck (Box-Konvention).
    """
    root = pathlib.Path(scenes_dir)
    out = []
    if not root.is_dir():
        return out
    scene_dirs = sorted([d for d in root.iterdir() if d.is_dir()
                         and (d / "scene_camera.json").is_file()])
    if seeds is not None:
        scene_dirs = scene_dirs[:seeds]
    for sd in scene_dirs:
        cam_all = json.loads((sd / "scene_camera.json").read_text())
        im_ids_all = sorted(int(k) for k in cam_all.keys())
        if not im_ids_all:
            continue
        if frames is None:
            sel_ims = [im_ids_all[0]]                 # Default: 1 Frame/Szene (byte-identisch)
        else:
            avail = set(im_ids_all)
            sel_ims = [im for im in frames if im in avail]  # nur existierende, in Wunsch-Reihenfolge
        for im_id in sel_ims:
            cam = cam_all[str(im_id)]
            K9 = cam.get("cam_K")
            rgb = sd / "rgb" / f"{im_id:06d}.png"
            depth = sd / "depth" / f"{im_id:06d}.png"
            camera = {"cam_K": K9,
                      "cam_R_w2c": cam.get("cam_R_w2c"),
                      "cam_t_w2c": cam.get("cam_t_w2c")}
            K = (_K_dict(K9) if K9 else None)
            out.append({
                "scene_id": int(sd.name), "im_id": im_id,
                "rgb": str(rgb), "depth": str(depth) if depth.exists() else None,
                # BOP depth PNGs encode metres = png*depth_scale/1000 (Isaac/SDG writes
                # 0.1). Carry it so the gateway+refiner decode the right scale (T-156);
                # default 1.0 (mm) when scene_camera omits it.
                "depth_scale": float(cam.get("depth_scale", 1.0)),
                "camera": camera, "K": K, "dir": str(sd),
            })
    return out


def _K_dict(K9):
    """[fx,0,cx,0,fy,cy,0,0,1] → {fx,fy,cx,cy} fuer die Gateway-Form-Felder."""
    return {"fx": float(K9[0]), "fy": float(K9[4]),
            "cx": float(K9[2]), "cy": float(K9[5])}


# ── CLI (der reale Post-Training-Lauf) ───────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch-Eval-Runner: 7 Kombis × N SDG-Seeds → AR IC-BIN + Runtime.")
    ap.add_argument("--scenes-dir", required=True,
                    help="BOP-Val-Root mit <scene_id>/{rgb,depth,scene_gt,scene_camera}.")
    ap.add_argument("--seeds", type=int, default=20, help="Anzahl Seed-Szenen (cap).")
    ap.add_argument("--gateway", default="http://localhost:8090",
                    help="Mesh-Gateway-Basis-URL (/predict).")
    ap.add_argument("--dataset-dir", required=True,
                    help="BOP-GT-Dataset fuer eval_bop --icbin (Box).")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="project/temp/batch_eval")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args(argv)

    scenes = discover_scenes(args.scenes_dir, seeds=args.seeds)
    if not scenes:
        print(f"[batch_eval] keine Szenen mit GT unter {args.scenes_dir}", file=sys.stderr)
        return 2
    predict_fn = http_predict(args.gateway, iterations=args.iterations, top_n=args.top_n)
    eval_fn = subprocess_eval(args.dataset_dir, split=args.split)

    results = run_batch(EVAL_CONFIGS, scenes, predict_fn, eval_fn, args.out,
                        run_id=args.run_id, warn=lambda m: print(m, file=sys.stderr))
    print(f"[batch_eval] {results['run_id']}: {results['n_configs']} Configs, "
          f"{results['n_scenes']} Szenen, {results['duration_s']}s")
    print(f"[batch_eval] -> {args.out}/{results['run_id']}/results.json + EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
