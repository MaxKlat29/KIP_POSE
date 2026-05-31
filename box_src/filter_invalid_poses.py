#!/usr/bin/env python3
"""filter_invalid_poses.py — post-SDG-Filter: Teile die NICHT auf der Tisch-
Arbeitsfläche ruhen, sondern auf dem Roboterarm liegen oder durch den Tisch
gefallen sind, werden aus gt_raw_*.json entfernt.

WARUM:
  Isaac-Physik produziert manchmal:
   • Teile ruhen auf dem LARA5-Arm (Z weit über Tisch) — in real-life impossible
   • Teile durch Tisch-Mesh gefallen (Z < 0) — Physik-Bug, sim only
  Beide Fälle: das Modell soll diese NICHT lernen, sonst predicts es Posen die
  unmöglich sind.

  Threshold (in Z über Tisch-Plane, table_z=-0.007m):
   • z_above >  +0.05 m → wahrscheinlich auf Arm           → REMOVE
   • z_above <  -0.02 m → durch Tisch gefallen             → REMOVE
   • -0.02 ≤ z_above ≤ +0.05 → legitim (Tisch / Stack)    → KEEP

  Stack-Toleranz +50mm reicht für: Tray-Rim (13mm Höhe), 2-3 gestackte Teile
  (Anker 24mm hoch, Zahnrad ~15mm hoch).

OUTPUT:
  Schreibt gt_raw_*.json zurück mit gefiltertem `instances`-Array
  (default: --apply). Mit --dry-run nur Statistik, keine writes.
  Frames die ALLE Instanzen verlieren werden mit "skip": true markiert →
  isaac_to_bop.py kann sie überspringen.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

TABLE_Z = -0.007
Z_UPPER = 0.05     # max 5cm über Tisch erlaubt
Z_LOWER = -0.02    # max 2cm unter Tisch erlaubt (kleine penetration tolerieren)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdg-dir", required=True, help="dir mit gt_raw_*.json")
    ap.add_argument("--apply", action="store_true",
                    help="schreibt gefilterte gt_raw_*.json zurück (default: dry-run)")
    ap.add_argument("--upper", type=float, default=Z_UPPER,
                    help=f"max z_above_table (m), default {Z_UPPER}")
    ap.add_argument("--lower", type=float, default=Z_LOWER,
                    help=f"min z_above_table (m), default {Z_LOWER}")
    a = ap.parse_args()

    files = sorted(Path(a.sdg_dir).glob("gt_raw_*.json"))
    if not files:
        sys.exit(f"keine gt_raw_*.json in {a.sdg_dir}")

    n_inst_total = n_removed = 0
    frames_clean = frames_partial = frames_empty = 0
    per_class_removed = {}
    for f in files:
        d = json.load(open(f))
        insts = d.get("instances", [])
        n_before = len(insts)
        kept = []
        for inst in insts:
            T = inst.get("T_obj2world")
            if T is None:
                kept.append(inst); continue
            try:
                z = float(T[2][3])
            except Exception:
                kept.append(inst); continue
            z_above = z - TABLE_Z
            if a.lower <= z_above <= a.upper:
                kept.append(inst)
            else:
                cls = inst.get("label", "?")
                per_class_removed[cls] = per_class_removed.get(cls, 0) + 1
                n_removed += 1
        n_inst_total += n_before
        n_after = len(kept)
        if n_after == n_before:
            frames_clean += 1
        elif n_after == 0:
            frames_empty += 1
        else:
            frames_partial += 1
        if a.apply:
            d["instances"] = kept
            if n_after == 0:
                d["skip"] = True
                d["skip_reason"] = "all parts filtered (arm/underfloor)"
            json.dump(d, open(f, "w"), indent=2)

    print(f"frames scanned : {len(files)}")
    print(f"frames clean   : {frames_clean}")
    print(f"frames partial : {frames_partial}  (some parts filtered, others kept)")
    print(f"frames empty   : {frames_empty}    (ALL parts filtered → marked skip)")
    print(f"instances total: {n_inst_total}")
    print(f"instances removed: {n_removed}  ({100*n_removed/max(1,n_inst_total):.1f}%)")
    print(f"  per class : {per_class_removed}")
    print(f"mode: {'APPLIED writes' if a.apply else 'DRY-RUN (no writes)'}")


if __name__ == "__main__":
    main()
