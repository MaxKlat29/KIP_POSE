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
# yolo-obb hat kein Gateway-seg (Pipeline-A-Live-Monolith bzw. obb-Maske) → "yolo".
_SEG_SOURCE = {"yolo-obb": "yolo", "yolo-seg": "yolo", "sam3": "sam3"}
# FE-Verfahren-Note (Lena batch.js NOTE_BY-Degrade-Fallback) pro pose_id.
_NOTE = {"gdrnpp": "Pipeline A", "foundationpose": "RGB-D 6DoF",
         "gigapose_rgbd": "coarse+ICP", "gigapose_rgb": "coarse"}


def _build_eval_configs():
    from pipelines.combos import FEASIBLE_COMBOS
    out = []
    for w in FEASIBLE_COMBOS:
        pose_source = "gdrnpp" if w["is_pipeline_a"] else w["pose_id"]
        # Pipeline A (yolo-obb→gdrnpp) hat KEIN Gateway-/predict → seg_source=gdrnpp
        # signalisiert den Live-Monolith-Pfad (PipelineANotOnGateway im Runner).
        seg_source = "gdrnpp" if w["is_pipeline_a"] else _SEG_SOURCE[w["seg_id"]]
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
                     table_origin=TABLE_ORIGIN_M, warn=None) -> dict:
    """Gateway-`/predict`-instances [{class,T_cam_obj}] → pose_result-Welt-Doc.

    T_cam_obj→Welt + §6-Klassenmapping, OHNE die Seg/Pose-Stages erneut zu fahren
    (die Posen kommen schon vom Gateway). Nur Klassen mit obj_id-Mapping (2-Klassen-
    Scope §6); unbekannte still gefiltert.

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
            table_origin=list(table_origin), source_image=source_image, warn=warn)
    return _instances_to_doc_local(instances, camera, source_image,
                                   table_origin=table_origin, warn=warn)


def _instances_to_doc_local(instances: list, camera: dict, source_image: str,
                            table_origin=TABLE_ORIGIN_M, warn=None) -> dict:
    """Self-contained T_cam_obj→Welt-Mapping (Fallback, byte-gleich zu S-013/composed).

    Spiegelt `composed.ComposedPipeline.infer`: §1 (T in Meter → mm,
    `bop_pose_to_world`), Boden-Snap (best-effort), §6 (obj_id→CamelCase-Part →
    `canonicalize_rotation` ueber PART_SYMMETRY). Round-trip gegen `world_to_bop_cam`
    getestet (test_batch_eval.test_coordframe_roundtrip_translation_preserved)."""
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

        # Boden-Snap (best-effort, fehlt das CAD lokal → kein Snap).
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
    seg+pose + die seg_ms/pose_ms-Telemetrie (gateway/app.py timings). Pipeline A
    (gdrnpp) hat KEIN Gateway-/predict (das ist der Live-Monolith); dafuer wirft die
    predict_fn `PipelineANotOnGateway` → der Runner faellt auf die lokale
    e2e_infer-Referenz zurueck (apples-to-apples ueber dieselben Szenen).

    Lazy `import httpx` — kein Pin in project/requirements.txt (Box-Stack).
    """
    base = gateway_url.rstrip("/")

    def _predict(cfg: dict, scene: dict) -> dict:
        if cfg.get("pose_source") == "gdrnpp":
            raise PipelineANotOnGateway(cfg)
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
    """Pipeline A (gdrnpp) hat kein Gateway-/predict — Live-Monolith-Referenz."""


# ── Default eval_fn: subprocess gegen box_src/eval_bop.py --icbin ────────────────
_EVAL_BOP = pathlib.Path("/mnt/data/kip_pose/box_src/eval_bop.py")
_BOP_VENV = pathlib.Path("/mnt/data/bop/bop-venv/bin/python")


def subprocess_eval(dataset_dir: str, split: str = "val",
                    eval_bop=_EVAL_BOP, bop_venv=_BOP_VENV, n_points: int = 2000):
    """Fabrik fuer eine eval_fn(csv_path, scene_dir, out_dir) -> report_json.

    Ruft `eval_bop.py --icbin` (offizielles BOP19/IC-BIN-Localisation-Protokoll,
    sym-aware, multi-instance) gegen das GT-Dataset. NUR auf der GPU-Box (eval_bop +
    bop-venv + bop_toolkit_lib). `scene_dir` ist hier ungenutzt — eval_bop matched
    ueber das ganze Dataset (--dataset-dir/--split); wir reichen es fuer eine
    moegliche Per-Szene-Variante durch.
    """
    import subprocess

    def _eval(csv_path: str, scene_dir: str, out_dir: str) -> dict:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(bop_venv), str(eval_bop),
               "--dataset-dir", str(dataset_dir), "--split", split,
               "--icbin", "--preds", str(csv_path),
               "--n-points", str(n_points), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"eval_bop --icbin FAIL: {r.stderr[-400:]}")
        return json.load(open(out / "report.json"))

    return _eval


def ar_from_report(report: dict) -> "tuple[float | None, dict]":
    """AR IC-BIN (overall) + per-Klasse aus einem eval_bop-report.json.

    report.json (eval_bop --icbin): {"results":{"per_object":{obj_id:{AR,name,...}},
    "overall":{"AR":...}}}. Liefert (overall_AR, {part: AR}); per_class fuer die
    optionale FE-`per_class`-Spalte. None wenn AR fehlt (kein stiller 0-Wert)."""
    res = (report or {}).get("results", report or {})
    overall = res.get("overall", {})
    ar = overall.get("AR")
    per_class = {}
    for oid, po in res.get("per_object", {}).items():
        name = po.get("name") or str(oid)
        if po.get("AR") is not None:
            per_class[name] = round(float(po["AR"]), 4)
    return (float(ar) if isinstance(ar, (int, float)) else None), per_class


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
                     per_class=None) -> dict:
    """Aggregiert die N Seed-Ergebnisse einer Config zur FE-Config-Zeile.

    per_scene: Liste von run_one-Outputs (ein Eintrag pro Seed). AR kommt aus EINEM
    eval_bop-Lauf ueber die gepoolte CSV (multi-instance IC-BIN braucht das ganze
    Dataset) — daher ar_mean/ar_std hier reingereicht (nicht aus per_scene).

    Liefert exakt Lenas batch.js-Felder:
      {seg,pose, ar_mean,ar_std, seg_ms,pose_ms, coverage,crash_rate,
       note, is_pipeline_a, run_config_id, per_class?}
    coverage/crash_rate ∈ 0..1.
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
        "ar_mean": ar_mean, "ar_std": ar_std,
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


# ── Stage D: der ganze Lauf (7 × N) ──────────────────────────────────────────────
def run_batch(configs, scenes, predict_fn, eval_fn, out_dir,
              run_id=None, progress=None, warn=None) -> dict:
    """Faehrt alle Configs × alle Szenen, scored, aggregiert, persistiert.

    Args:
      configs   : Liste von Config-dicts (Default EVAL_CONFIGS = die 7).
      scenes    : Liste von Szenen-dicts {scene_id, im_id, rgb, depth?, camera, K, dir}.
      predict_fn: (cfg, scene) -> {instances, timings}  (Mock-Naht).
      eval_fn   : (csv_path, scene_dir, out_dir) -> report_json  (Mock-Naht).
      out_dir   : project/temp/batch_eval — der Run landet unter <out_dir>/<run_id>/.
      run_id    : default = "run-<utc-stamp>". Idempotent: gleiche id patcht.
      progress  : optional callback(pct:int, phase:str) fuer die Job-Bar.
      warn      : optional log-callback.

    Liefert das results-dict (= /api/eval/result-Body, {run_id,date,configs,...})
    und schreibt <out_dir>/<run_id>/results.json + EVAL.md.
    """
    warn = warn or (lambda *a, **k: None)
    progress = progress or (lambda *a, **k: None)
    run_id = run_id or ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = pathlib.Path(out_dir) / run_id
    csv_dir = run_dir / "csv"
    eval_dir = run_dir / "eval"
    for d in (run_dir, csv_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    config_rows = []
    n_total = max(1, len(configs))

    for ci, cfg in enumerate(configs):
        key = config_key(cfg)
        progress(int(5 + 90 * ci / n_total), f"Config {cfg['n']}/{len(configs)}: {key}")
        per_scene = []
        all_rows = []
        for scene in scenes:
            res = run_one(cfg, scene, predict_fn, warn=warn)
            per_scene.append(res)
            all_rows.extend(res["rows"])

        # AR IC-BIN: EIN eval_bop-Lauf ueber die ueber alle Seeds gepoolte CSV
        # (multi-instance IC-BIN-Matching braucht das ganze Dataset-Set). ar_std
        # ueber die Seeds ist hier 0/None — IC-BIN ist ein Set-Score, kein
        # Per-Szene-Mittel; wir reporten den Set-AR als ar_mean.
        ar_mean, ar_std, per_class = None, None, None
        csv_path = csv_dir / f"{key}.csv"
        _write_bop_csv(csv_path, all_rows)
        scored_any = any(s["ok"] for s in per_scene
                         if s.get("error") != "pipeline_a_no_gateway")
        if scored_any and all_rows:
            try:
                report = eval_fn(str(csv_path),
                                 str(scenes[0].get("dir", "")), str(eval_dir / key))
                ar_mean, per_class = ar_from_report(report)
            except Exception as e:  # noqa: BLE001 — Eval-Fehler = AR unbekannt, kein Abbruch
                warn(f"[batch_eval] eval_bop fuer {key} fehlgeschlagen: {e}")

        row = aggregate_config(cfg, per_scene, ar_mean=ar_mean, ar_std=ar_std,
                               per_class=per_class)
        config_rows.append(row)
        warn(f"[batch_eval] {key}: AR={ar_mean} cov={row['coverage']} "
             f"crash={row['crash_rate']} seg_ms={row['seg_ms']} pose_ms={row['pose_ms']}")

    duration_s = round(time.time() - t0, 1)
    results = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": duration_s,
        "n_configs": len(config_rows),
        "n_scenes": len(scenes),
        "configs": config_rows,
    }
    _persist(run_dir, results)
    progress(100, "Lauf fertig")
    return results


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
    7 = RECOMMENDED (★), degraded (gdrnpp AABB-aus-Maske) + class-ambig (sam3) markiert."""
    n_rec = sum(1 for c in results["configs"] if c.get("recommended"))
    lines = [
        f"# Batch-Eval {results['run_id']}", "",
        f"- Datum: {results['date']}",
        f"- Configs: {results['n_configs']} feasible ({n_rec} recommended) "
        f"· Szenen/Seeds: {results['n_scenes']} · Dauer: {results['duration_s']} s", "",
        "| # | Seg | Pose | AR IC-BIN | ±std | seg ms | pose ms | Coverage | Crash "
        "| Verfahren | Flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
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
            f"| {i}{ref} | {c['seg']} | {c['pose']} | {_f(c['ar_mean'])} "
            f"| {_f(c['ar_std'])} | {_ms(c['seg_ms'])} | {_ms(c['pose_ms'])} "
            f"| {_pct(c['coverage'])} | {_pct(c['crash_rate'])} | {c.get('note') or '—'} "
            f"| {', '.join(flags) or '—'} |"
        )
    lines += ["", "_AR IC-BIN: offizielles BOP19/IC-BIN-Localisation-Protokoll "
              "(sym-aware, 2 Anker-Klassen D1). Coverage/Crash ∈ 0..1. ★empf. = "
              "kuratierte Recommended-7; degr. = GDRNPP AABB-aus-Maske; klassen-ambig "
              "= sam3 trennt kurz/lang schwach (S006)._", ""]
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
def discover_scenes(scenes_dir, seeds=None, split_layout=True) -> list:
    """Findet vorgerenderte SDG-Seed-Szenen MIT GT unter `scenes_dir`.

    Erwartet BOP-Layout: <scenes_dir>/<scene_id>/ mit rgb/<im>.png, depth/<im>.png,
    scene_gt.json, scene_camera.json. Eine Szene pro scene_id (im=0). `seeds` cappt
    auf die ersten N. Liefert {scene_id, im_id, rgb, depth, camera, K, dir}.

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
    for sd in scene_dirs:
        cam_all = json.loads((sd / "scene_camera.json").read_text())
        im_ids = sorted(int(k) for k in cam_all.keys())
        if not im_ids:
            continue
        im_id = im_ids[0]
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
            "camera": camera, "K": K, "dir": str(sd),
        })
    if seeds is not None:
        out = out[:seeds]
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
