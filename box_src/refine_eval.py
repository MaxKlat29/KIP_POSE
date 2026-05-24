#!/usr/bin/env python3
"""refine_eval.py — training-freie Rotations-Refinements auf BESTEHENDE GDRNPP-val-
Predictions anwenden + neue BOP-results-CSVs schreiben (T-041 / S-047).

Liest die rohen per-Objekt GDRNPP-val-Prediction-CSVs (`*-iter0_pose_isaac-val.csv`,
BOP-results-Format, R_m2c/t_m2c im KAMERA-Frame), transformiert sie via scene_camera
in den WELT-Frame (bop_adapter.bop_pose_to_world), wendet die gewünschten Maßnahmen
an (Z-Snap + M1 Stable-Pose-Snap + M4 Contour-Yaw-Lock), transformiert zurück in den
Kamera-Frame und schreibt EINE kombinierte BOP-results-CSV pro Konfiguration. Diese
CSV wird dann mit `eval_bop.py` (symmetrie-bewusst, gefiltert) bewertet — so misst man
jede Maßnahme einzeln UND kombiniert, ehrlich, an denselben Predictions.

KEIN Training, KEINE GPU. Reines numpy + trimesh (CPU) + PIL (Masken).

USAGE (auf der Box, bop-venv ODER lokal mit Daten-Kopie):
  python box_src/refine_eval.py \
      --bop-root  /mnt/data/kip_pose/project/bop/pose_isaac \
      --preds anker_kurz=<csv> anker_lang=<csv> zahnrad=<csv> \
      --out-dir /tmp/refine_out \
      --config zsnap            # baseline (nur Z-Snap)
      # weitere: m1, m4, m1_zsnap, m1_m4_zsnap, raw

Der Welt-Frame-Round-Trip ist EXAKT invertierbar (siehe test_bop_adapter
world_to_bop): das Schreiben einer NICHT-refinten Config (`raw`) reproduziert die
Eingabe-CSV bit-nah (Sanity).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))
import bop_adapter as A  # noqa: E402

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"FATAL: trimesh nötig (M1/M4): {exc!r}\n")
    raise

# obj_id der per-Objekt-CSVs (Detektor-Klassenname -> obj_id).
NAME_TO_OBJ = {"anker_kurz": 1, "anker_lang": 2, "zahnrad": 6}
# C_N pro obj_id (aus models_info: Anker continuous-Y -> kein discrete N; Zahnrad C_7).
OBJ_NFOLD = {1: None, 2: None, 6: 7}
SYM_AXIS = (0.0, 1.0, 0.0)

# Tisch-Ebene im VAL-Welt-Frame (Meter). Aus den GT-Auflagepunkten gemessen
# (resting-contact-z Median ≈ 0.018 m; der CAD-Ursprung sitzt nicht im Boden).
DEFAULT_TABLE_Z = 0.018
# Im pose_result-Contract ist t_world relativ zum Tisch-Nullpunkt; hier arbeiten wir
# direkt im VAL-Welt-Frame (table_origin=0), Z-Snap auf table_z. Eval ist rein im
# KAMERA-Frame -> der absolute Welt-Z-Offset hebt sich im Round-Trip exakt weg; nur
# der Z-Snap nutzt table_z (für die Rotation M1/M4 ist table_z irrelevant).
TABLE_ORIGIN = np.zeros(3)


def load_cams(bop_root):
    cams = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        cams[sid] = json.load(open(os.path.join(sc, "scene_camera.json")))
    return cams


def load_meshes(bop_root):
    md = os.path.join(bop_root, "models_eval")
    if not os.path.isdir(md):
        md = os.path.join(bop_root, "models")
    meshes, downs = {}, {}
    for oid in (1, 2, 6):
        m = trimesh.load(os.path.join(md, f"obj_{oid:06d}.ply"), process=False)
        meshes[oid] = m
        d, p = A.stable_pose_body_downs(mesh=m, prob_min=0.02, cache_key=oid)
        downs[oid] = (d, p)
    return meshes, downs


def load_gt_index(bop_root):
    """(scene,im,obj_id) -> Liste von (inst_idx, t_world_mm-Hilfe, R_m2c, t_m2c).

    Zum Matchen Prediction<->GT-Instanz (für die M4-Maske) + Inst-Index für die
    mask_visib-Datei. Match per Translations-Nähe im Kamera-Frame (wie eval).
    """
    idx = {}
    for sc in sorted(glob.glob(os.path.join(bop_root, "val", "*"))):
        if not os.path.isdir(sc):
            continue
        sid = int(os.path.basename(sc))
        gt = json.load(open(os.path.join(sc, "scene_gt.json")))
        for im, insts in gt.items():
            for i, inst in enumerate(insts):
                key = (sid, int(im), int(inst["obj_id"]))
                t = np.array(inst["cam_t_m2c"], float)
                idx.setdefault(key, []).append((i, t))
    return idx


def load_mask(bop_root, sid, im, inst_idx):
    from PIL import Image
    p = os.path.join(bop_root, "val", f"{sid:06d}", "mask_visib",
                     f"{im:06d}_{inst_idx:06d}.png")
    if not os.path.isfile(p):
        return None
    return np.array(Image.open(p)) > 0


def read_preds(path, obj_id):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "scene_id": int(r["scene_id"]), "im_id": int(r["im_id"]),
                "obj_id": obj_id, "score": float(r["score"]),
                "R": np.array([float(x) for x in r["R"].split()]).reshape(3, 3),
                "t": np.array([float(x) for x in r["t"].split()]),
                "time": float(r.get("time", -1)),
            })
    return rows


def match_gt_inst(gt_index, sid, im, obj_id, t_m2c, used):
    """Nächste noch-freie GT-Instanz (Kamera-Translation) -> inst_idx oder None."""
    cands = gt_index.get((sid, im, obj_id), [])
    best, bd = None, None
    for i, tg in cands:
        if (sid, im, obj_id, i) in used:
            continue
        d = float(np.linalg.norm(t_m2c - tg))
        if bd is None or d < bd:
            bd, best = d, i
    if best is not None:
        used.add((sid, im, obj_id, best))
    return best


def world_to_cam(R_world, t_world, R_w2c, t_w2c, table_origin):
    """Inverse der Adapter-Kette (R_model_to_body=I): Welt-Pose -> R_m2c/t_m2c (mm)."""
    t_m2w_mm = (t_world + table_origin) * 1000.0
    R_m2c = R_w2c @ R_world
    t_m2c = R_w2c @ t_m2w_mm + t_w2c
    return R_m2c, t_m2c


def refine_rows(rows, cams, meshes, downs, gt_index, bop_root, config,
                table_z=DEFAULT_TABLE_Z):
    """Wende config auf alle rows an. config: set aus {zsnap,m1,m4}."""
    do_z = "zsnap" in config
    do_m1 = "m1" in config
    do_m4 = "m4" in config
    used = set()
    out = []
    stats = {"m1_snapped": 0, "m1_skipped": 0, "m4_applied": 0, "m4_changed": 0,
             "z_snapped": 0, "n": 0}
    for r in rows:
        sid, im, oid = r["scene_id"], r["im_id"], r["obj_id"]
        cam = cams[sid][str(im)]
        R_w2c = np.array(cam["cam_R_w2c"], float).reshape(3, 3)
        t_w2c = np.array(cam["cam_t_w2c"], float)
        K = np.array(cam["cam_K"], float).reshape(3, 3)
        # cam -> world
        R_world, t_world = A.bop_pose_to_world(r["R"], r["t"], R_w2c, t_w2c,
                                               TABLE_ORIGIN)
        d, p = downs[oid]
        # M1 (+ optional Z-Snap) via planar_refine.
        if do_m1 or do_z:
            R_world, t_world, info = A.planar_refine(
                R_world, t_world, np.asarray(meshes[oid].vertices, float) / 1000.0,
                table_z=table_z, z_snap=do_z, tilt_correct=False,
                max_snap_m=A.DEFAULT_MAX_SNAP_M,
                stable_pose_snap=do_m1, stable_downs=d, stable_probs=p,
                max_tilt_snap_deg=A.DEFAULT_MAX_TILT_SNAP_DEG)
            if info.get("m1_snapped"):
                stats["m1_snapped"] += 1
            if info.get("m1_skipped"):
                stats["m1_skipped"] += 1
            if info.get("z_snap"):
                stats["z_snapped"] += 1
        # M4 contour yaw-lock (nur C_N-Teile mit Maske).
        if do_m4 and OBJ_NFOLD.get(oid):
            inst_idx = match_gt_inst(gt_index, sid, im, oid, r["t"], used)
            mask = load_mask(bop_root, sid, im, inst_idx) if inst_idx is not None else None
            if mask is not None:
                R_new, m4 = A.contour_yaw_lock(
                    R_world, mask, K, t_world, TABLE_ORIGIN, R_w2c, t_w2c,
                    mesh=meshes[oid], n_fold=OBJ_NFOLD[oid], sym_axis=SYM_AXIS)
                if m4["applied"]:
                    stats["m4_applied"] += 1
                    if m4["k"] != 0:
                        stats["m4_changed"] += 1
                    R_world = R_new
        # world -> cam
        R_m2c, t_m2c = world_to_cam(R_world, t_world, R_w2c, t_w2c, TABLE_ORIGIN)
        out.append({**r, "R": R_m2c, "t": t_m2c})
        stats["n"] += 1
    return out, stats


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for r in rows:
            R = " ".join(f"{v:.9g}" for v in r["R"].reshape(-1))
            t = " ".join(f"{v:.9g}" for v in r["t"].reshape(-1))
            f.write(f"{r['scene_id']},{r['im_id']},{r['obj_id']},{r['score']:.9g},"
                    f"{R},{t},{r['time']:.9g}\n")


CONFIGS = {
    "raw": set(),
    "zsnap": {"zsnap"},
    "m1": {"m1"},
    "m1_zsnap": {"m1", "zsnap"},
    "m4": {"m4"},
    "m4_zsnap": {"m4", "zsnap"},
    "m1_m4_zsnap": {"m1", "m4", "zsnap"},
}


def main():
    ap = argparse.ArgumentParser(description="T-041 training-free rotation refine -> CSV")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="name=csv (name in anker_kurz/anker_lang/zahnrad)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", nargs="+", default=list(CONFIGS.keys()),
                    help=f"eine/mehrere aus {list(CONFIGS)}")
    ap.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cams = load_cams(args.bop_root)
    meshes, downs = load_meshes(args.bop_root)
    gt_index = load_gt_index(args.bop_root)

    all_rows = []
    for spec in args.preds:
        name, path = spec.split("=", 1)
        oid = NAME_TO_OBJ[name]
        all_rows.append((oid, read_preds(path, oid)))

    for cfg in args.config:
        cset = CONFIGS[cfg]
        combined, allstats = [], {}
        for oid, rows in all_rows:
            ref, st = refine_rows(rows, cams, meshes, downs, gt_index,
                                  args.bop_root, cset, table_z=args.table_z)
            combined.extend(ref)
            allstats[oid] = st
        out = os.path.join(args.out_dir, f"preds_{cfg}.csv")
        write_csv(combined, out)
        sys.stderr.write(f"[refine] {cfg:14s} -> {out}  ({len(combined)} preds)\n")
        for oid, st in allstats.items():
            sys.stderr.write(
                f"           obj{oid}: n={st['n']} m1_snapped={st['m1_snapped']} "
                f"m1_skipped={st['m1_skipped']} z={st['z_snapped']} "
                f"m4_applied={st['m4_applied']} m4_changed_yaw={st['m4_changed']}\n")


if __name__ == "__main__":
    main()
