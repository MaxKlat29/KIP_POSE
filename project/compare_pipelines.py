#!/usr/bin/env python3
"""compare_pipelines.py — Vergleichs-Harness fuer komplett andere Pose-Pipelines.

Misst registrierte Pipelines (pipelines.registry) ueber IDENTISCHE Szenen auf VIER
Achsen (Max-Entscheidung 2026-06-01):

  1. Genauigkeit vs GT  — BOP-Metriken (AR/ADD/MSSD/MSPD, sym-aware) via box_src/eval_bop.py
                          ueber dieselben Szenen. Braucht GT-Dataset + GPU-Box (bop_toolkit).
  2. Visueller Side-by-Side — pro Szene ein Multi-Pipeline-Payload (alle pose_result-Docs
                          nebeneinander) den der Viewer overlay/umschaltbar rendert.
  3. Latenz/Laufzeit    — Wall-Clock pro Pipeline pro Bild (PipelineAdapter.run_timed).
  4. Robustheit/Coverage — n_results, Crash-/Fehlerrate, leere Outputs.

Output:
  <out>/compare_result.json   — maschinenlesbar, alle 4 Achsen
  <out>/COMPARE.md            — Vergleichstabelle (Stil wie box_src/e2e_ab.py)
  <out>/sidebyside/<scene>.json — Multi-Pipeline-Payload pro Szene (Achse 2)

Status 2026-06-01: SCAFFOLD. Die Orchestrierung + Telemetrie + Tabelle + die
World->Cam-Rueckrechnung fuer die Eval sind FERTIG und ``--dry-run`` laeuft offline.
Die echten fremden Pipelines sind noch leer (registry kennt nur die gdrnpp-Referenz);
nicht-verfuegbare Pipelines werden mit EXPLIZITEM Log uebersprungen (kein stilles
Weglassen). Der eval_bop-Aufruf ist hinter --dataset-dir + Box-Praesenz gegated.

Usage:
  # Offline-Trockenlauf (kein Bild, keine Box) — exerziert Tabelle + Aggregation:
  python3 project/compare_pipelines.py --dry-run --out /tmp/cmp

  # Echt (auf der Box): alle verfuegbaren Pipelines ueber Val-Szenen, mit GT-Eval:
  python3 project/compare_pipelines.py \
      --pipelines gdrnpp,pipeline_x \
      --images project/input/scene_0001.png,... \
      --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac --split val \
      --out project/temp/compare_run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipelines import registry, contract  # noqa: E402

# Box-seitiger Eval-Harness (nur dort vorhanden).
_EVAL_BOP = pathlib.Path("/mnt/data/kip_pose/box_src/eval_bop.py")
_BOP_VENV = pathlib.Path("/mnt/data/bop/bop-venv/bin/python")


# ── Achse 1: World -> BOP-cam-Frame Rueckrechnung (fuer eval_bop) ─────────────
def world_to_bop_cam(R_world, t_world_m, R_w2c, t_w2c_mm, table_origin_m):
    """Inverse von bop_adapter.bop_pose_to_world (R_model_to_body = I).

    eval_bop.py arbeitet im BOP-cam-Frame (R_m2c, t_m2c mm); pose_result liegt im
    Welt-Frame. Diese Funktion rechnet zurueck:

        R_m2w   = R_world                      (world = R @ body, R_mb = I)
        R_m2c   = R_w2c @ R_m2w
        t_m2w   = (t_world + table_origin) * 1000      [mm]
        t_m2c   = R_w2c @ t_m2w + t_w2c                [mm]

    Round-Trip gegen bop_pose_to_world ist in tests/test_pipelines_scaffold.py
    auf Identitaet geprueft.
    """
    import numpy as np
    R_world = np.asarray(R_world, float).reshape(3, 3)
    t_world = np.asarray(t_world_m, float).reshape(3)
    R_w2c = np.asarray(R_w2c, float).reshape(3, 3)
    t_w2c = np.asarray(t_w2c_mm, float).reshape(3)
    to = np.asarray(table_origin_m, float).reshape(3)

    R_m2c = R_w2c @ R_world
    t_m2w_mm = (t_world + to) * 1000.0
    t_m2c_mm = R_w2c @ t_m2w_mm + t_w2c
    return R_m2c, t_m2c_mm


def doc_to_bop_rows(doc, camera, scene_id, im_id):
    """pose_result-Doc -> BOP-results-CSV-Zeilen (das Format das eval_bop.preds_from_csv
    liest: scene_id, im_id, obj_id, score, R(9 space-sep), t(3 space-sep), time)."""
    import numpy as np
    from bop_adapter import PART_TO_OBJ_ID
    R_w2c = np.asarray(camera["cam_R_w2c"], float).reshape(3, 3)
    t_w2c = np.asarray(camera["cam_t_w2c"], float).reshape(3)
    table_origin = doc.get("meta", {}).get("table_origin", [0.0, 0.0, 0.08])
    rows = []
    for r in doc.get("results", []):
        obj_id = PART_TO_OBJ_ID.get(r["part"])
        if obj_id is None:
            continue
        R_m2c, t_m2c = world_to_bop_cam(r["R_world"], r["t_world"], R_w2c, t_w2c, table_origin)
        rows.append({
            "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id,
            "score": float(r.get("confidence", 1.0)),
            "R": " ".join(f"{x:.8f}" for x in np.asarray(R_m2c).reshape(9)),
            "t": " ".join(f"{x:.6f}" for x in np.asarray(t_m2c).reshape(3)),
            "time": -1.0,
        })
    return rows


# ── Orchestrierung ────────────────────────────────────────────────────────────
def select_pipelines(ids):
    """ids=None -> alle registrierten; sonst die genannten. Gibt eine flache Liste
    von PipelineAdapter-Instanzen (Verfuegbarkeit wird in run_matrix geprueft)."""
    if ids:
        chosen = []
        for pid in ids:
            try:
                chosen.append(registry.get(pid))
            except KeyError as e:
                print(f"[compare] WARN: {e}")
        return chosen
    return [registry.get(p["id"]) for p in registry.all_pipelines()]


def run_matrix(adapters, images, cameras, table_origin, log=print):
    """Jede Pipeline ueber jedes Bild. Ueberspringt nicht-verfuegbare Pipelines MIT
    explizitem Log. Gibt {pipeline_id: [PipelineResult, ...]} (parallel zu images)."""
    out = {}
    for a in adapters:
        if not a.available:
            log(f"[compare] SKIP {a.id}: nicht verfuegbar (Stub/Deps fehlen/noch nicht angebunden)")
            out[a.id] = []
            continue
        res = []
        for img, cam in zip(images, cameras):
            pr = a.run_timed(img, cam, table_origin)
            status = "ok" if pr.ok else f"FAIL({pr.error[:60]})"
            log(f"[compare] {a.id} :: {pathlib.Path(img).name} -> {pr.n_results} Teile, "
                f"{pr.latency_ms:.0f}ms [{status}]")
            res.append(pr)
        out[a.id] = res
    return out


def aggregate_telemetry(results_by_pid):
    """Latenz (Achse 3) + Robustheit/Coverage (Achse 4) pro Pipeline."""
    agg = {}
    for pid, results in results_by_pid.items():
        n = len(results)
        oks = [r for r in results if r.ok]
        lats = [r.latency_ms for r in oks]
        agg[pid] = {
            "n_images": n,
            "n_ok": len(oks),
            "n_failed": n - len(oks),
            "crash_rate": round((n - len(oks)) / n, 3) if n else None,
            "latency_ms_mean": round(sum(lats) / len(lats), 1) if lats else None,
            "latency_ms_max": round(max(lats), 1) if lats else None,
            "results_total": sum(r.n_results for r in oks),
            "results_per_image": round(sum(r.n_results for r in oks) / len(oks), 2) if oks else None,
            "errors": sorted({r.error for r in results if r.error}),
        }
    return agg


def score_accuracy(results_by_pid, cameras, dataset_dir, split, out_dir, log=print):
    """Achse 1: pro Pipeline BOP-results-CSV bauen + via eval_bop.py scoren.

    Gated: nur wenn dataset_dir gesetzt UND eval_bop + bop-venv auf der Box da sind.
    Sonst 'pending' (mit Log). Erzeugt <out>/eval/<pid>/preds.csv + report.json.
    """
    import csv
    import subprocess
    acc = {}
    if not dataset_dir or not _EVAL_BOP.exists() or not _BOP_VENV.exists():
        log("[compare] Genauigkeit (Achse 1): uebersprungen — kein --dataset-dir "
            "oder eval_bop/bop-venv nicht vorhanden (laeuft nur auf der GPU-Box).")
        return {pid: {"status": "pending", "reason": "no GT/box"} for pid in results_by_pid}

    # Pro-Bild Kamera/Scene-ID-Verdrahtung ist noch offen (Scaffold, siehe main()).
    # eval_bop matched STRIKT auf (scene_id, im_id) + braucht echte Extrinsics. Mit
    # Platzhalter-Kameras / synthetischen ids waere das ein STILLER Null-Match
    # (AR=0, n_matched=0) statt eines Fehlers — also hier ehrlich abbrechen, statt
    # eine falsche 0 zu berichten.
    if any(not (isinstance(c, dict) and c.get("cam_R_w2c") and c.get("cam_t_w2c")) for c in cameras):
        log("[compare] Genauigkeit (Achse 1): uebersprungen — pro-Bild Kamera-Extrinsics "
            "+ passende scene/im-ids noch nicht verdrahtet (Scaffold). Sonst stiller "
            "Null-Match. Siehe docs/PIPELINE_INTEGRATION.md §4.")
        return {pid: {"status": "pending", "reason": "per-image camera/scene-id wiring TODO"}
                for pid in results_by_pid}

    for pid, results in results_by_pid.items():
        if not results:
            acc[pid] = {"status": "skipped", "reason": "pipeline nicht verfuegbar"}
            continue
        evd = pathlib.Path(out_dir) / "eval" / pid
        evd.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (pr, cam) in enumerate(zip(results, cameras)):
            if pr.ok and pr.doc:
                rows += doc_to_bop_rows(pr.doc, cam, scene_id=i, im_id=0)
        csvp = evd / "preds.csv"
        with open(csvp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
            w.writeheader()
            w.writerows(rows)
        rep = evd / "report.json"
        r = subprocess.run(
            [str(_BOP_VENV), str(_EVAL_BOP), "--dataset-dir", str(dataset_dir),
             "--split", split, "--preds", str(csvp), "--out", str(evd)],
            capture_output=True, text=True)
        if r.returncode != 0:
            acc[pid] = {"status": "error", "detail": r.stderr[-300:]}
            log(f"[compare] eval_bop FAIL fuer {pid}: {r.stderr[-120:]}")
            continue
        try:
            report = json.load(open(rep))
            acc[pid] = {"status": "scored", "report": report}
        except Exception as e:  # noqa: BLE001
            acc[pid] = {"status": "error", "detail": f"report unlesbar: {e}"}
    return acc


def render_markdown(meta, telemetry, accuracy):
    """COMPARE.md — eine Zeile pro Pipeline ueber alle 4 Achsen."""
    lines = [
        "# Pipeline-Vergleich", "",
        f"- Pipelines: {', '.join(meta['pipelines']) or '(keine verfuegbar)'}",
        f"- Bilder: {meta['n_images']}",
        f"- Genauigkeits-Eval: {meta['accuracy_mode']}",
        "",
        "| Pipeline | verfuegbar | Ø Latenz (ms) | Crash-Rate | Teile/Bild | mean-AR | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for pid in meta["all_ids"]:
        t = telemetry.get(pid, {})
        a = accuracy.get(pid, {})
        avail = "✓" if pid in meta["pipelines"] else "—"
        ar = "—"
        if a.get("status") == "scored":
            # eval_bop schreibt report.json als {"results": {"overall": {"AR": <float>}}}.
            ov = a.get("report", {}).get("results", {}).get("overall", {})
            ar_val = ov.get("AR")
            ar = f"{ar_val:.4f}" if isinstance(ar_val, (int, float)) else "?"
        status = a.get("status", "?")
        lines.append(
            f"| {pid} | {avail} | {t.get('latency_ms_mean','—')} | {t.get('crash_rate','—')} "
            f"| {t.get('results_per_image','—')} | {ar} | {status} |"
        )
    lines += ["", "_Scaffold — fremde Pipelines noch leer. Genauigkeit nur auf der GPU-Box mit GT._", ""]
    return "\n".join(lines)


def _synth_dry_run(adapters, out_dir, log=print):
    """Offline: synthetisiert je Pipeline einen Datenpunkt, damit Aggregation +
    Tabelle ohne Bild/Box exerziert werden. Deterministisch (kein Zufall)."""
    from pipelines.base import PipelineResult
    results_by_pid, cams = {}, [{}]
    for i, a in enumerate(adapters):
        if a.available:
            # 1 synthetischer ok-Datenpunkt (latenz deterministisch aus index)
            results_by_pid[a.id] = [PipelineResult(a.id, {"results": []}, 100.0 + 10 * i, 0, True, "")]
        else:
            results_by_pid[a.id] = []  # Stub -> wird als 'nicht verfuegbar' aggregiert
    log("[compare] DRY-RUN: synthetische Datenpunkte, keine echte Inferenz.")
    return results_by_pid, cams


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vergleich komplett anderer Pose-Pipelines (4 Achsen).")
    ap.add_argument("--pipelines", default="", help="Komma-Liste von ids; leer = alle registrierten.")
    ap.add_argument("--images", default="", help="Komma-Liste von Bildpfaden (echter Lauf).")
    ap.add_argument("--dataset-dir", default="", help="BOP-GT-Dataset fuer Genauigkeit (Achse 1, Box).")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="project/temp/compare_run")
    ap.add_argument("--dry-run", action="store_true", help="Offline: keine echte Inferenz, nur Skelett.")
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in args.pipelines.split(",") if s.strip()] or None
    adapters = select_pipelines(ids)
    all_ids = [a.id for a in adapters]
    table_origin = list(contract.TABLE_ORIGIN_SCENE)

    if args.dry_run:
        results_by_pid, cameras = _synth_dry_run(adapters, out_dir)
        accuracy = {pid: {"status": "pending", "reason": "dry-run"} for pid in all_ids}
        n_images = 1
    else:
        images = [s.strip() for s in args.images.split(",") if s.strip()]
        if not images:
            print("[compare] Kein --images und kein --dry-run -> nichts zu tun. "
                  "Nutze --dry-run fuer einen Offline-Skelettlauf.")
            return 2
        n_images = len(images)
        # Kamera pro Bild: hier Platzhalter (echte Verdrahtung beim ersten echten
        # Lauf — pro Bild scene_camera.json laden). Fuer den Scaffold: leere Kamera.
        # score_accuracy bricht bei Platzhalter-Kameras ehrlich ab (kein stiller 0-Match).
        cameras = [{} for _ in images]
        results_by_pid = run_matrix(adapters, images, cameras, table_origin)
        accuracy = score_accuracy(results_by_pid, cameras, args.dataset_dir, args.split, out_dir)

    telemetry = aggregate_telemetry(results_by_pid)
    available_ids = [pid for pid, r in results_by_pid.items() if r]
    meta = {
        "all_ids": all_ids,
        "pipelines": available_ids,
        "n_images": n_images,
        "accuracy_mode": "dry-run" if args.dry_run else ("eval_bop" if args.dataset_dir else "skipped"),
    }
    result = {"meta": meta, "telemetry": telemetry, "accuracy": accuracy}
    (out_dir / "compare_result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "COMPARE.md").write_text(render_markdown(meta, telemetry, accuracy))
    print(f"[compare] -> {out_dir/'compare_result.json'}")
    print(f"[compare] -> {out_dir/'COMPARE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
