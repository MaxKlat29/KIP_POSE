#!/usr/bin/env python3
"""e2e_report.py - turn an eval_bop report.json into the Phase-2 results docs.

Given a freshly-produced BOP eval report (report.json from eval_bop.py, --preds
mode over the >20%-visibility val split), this:

  1. Reads the per-object AR / trans-median / rot-median numbers.
  2. Compares them against the frozen §0 in-house baseline
        Anker_Kurz 0.59 · Anker_Lang 0.61 · Zahnrad 0.36   (AR).
  3. Writes  project/docs/RESULTS_PHASE2.md  (a self-contained Baseline -> Phase-2
     comparison with a verdict per object).
  4. Patches the "### 4.2 GDRNPP pose accuracy" table inside
        project/docs/PROJECT_REPORT.md  with the new numbers (idempotent: it
     replaces the rows between the table header and the next blank/heading).

This is laptop-side, stdlib-only (no bop_toolkit needed). It is the same code
path for --dry-run (reads an existing report.json) and the full finish (reads the
report.json the GPU eval just produced) -- only the *input file* differs.

Usage:
  python3 e2e_report.py \
      --report   results/eval/report.json \
      --report-md project/docs/PROJECT_REPORT.md \
      --out-md    project/docs/RESULTS_PHASE2.md \
      [--mode dry-run|full] [--preds-note "preds_all.csv (val, gt-bbox)"]
"""
import argparse
import datetime
import json
import os
import re
import sys

# Frozen §0 in-house baseline (AR per trainable object). These are the numbers the
# whole Phase-2 effort is measured against -- do NOT recompute them here.
BASELINE_AR = {
    "Anker_Kurz": 0.59,
    "Anker_Lang": 0.61,
    "Zahnrad": 0.36,
}
# the in-house baseline's single headline rotation error (sym-resolved), §0.
BASELINE_ROT_MED_DEG = 91.0
TRAINABLE_OBJ_IDS = (1, 2, 6)  # Anker_Kurz, Anker_Lang, Zahnrad


def _f(x, nd=3):
    return None if x is None else round(float(x), nd)


def load_report(path):
    """Return (per_object dict obj_id->row, overall dict). Tolerant to the two
    shapes eval_bop.py can emit (per_object as dict or list)."""
    doc = json.load(open(path))
    res = doc.get("results", doc)
    po = res.get("per_object", {})
    overall = res.get("overall", {})
    rows = {}
    items = po.items() if isinstance(po, dict) else enumerate(po)
    for k, r in items:
        oid = int(r.get("obj_id", k))
        rows[oid] = r
    return rows, overall, doc.get("mode", "eval")


def verdict(name, ar_now):
    base = BASELINE_AR.get(name)
    if base is None or ar_now is None:
        return "—"
    if ar_now >= base + 0.02:
        return f"BEATS baseline (+{(ar_now - base):.2f})"
    if ar_now <= base - 0.02:
        return f"below baseline ({(ar_now - base):+.2f})"
    return f"on par ({(ar_now - base):+.2f})"


def build_results_table_rows(rows):
    """The markdown body rows for the §4.2 table, trainable objects only."""
    out = []
    for oid in TRAINABLE_OBJ_IDS:
        r = rows.get(oid)
        if not r:
            out.append(f"| {oid} | (no prediction) | — | — | — | — | — | — |")
            continue
        name = r.get("name", "?")
        sym = r.get("sym") or r.get("sym_kind") or "—"
        ar = _f(r.get("AR"), 3)
        add = _f(r.get("ADD/ADI") or r.get("add_adi_mean_mm"), 1)
        tmed = _f(r.get("trans_err_median_mm"), 1)
        tmean = _f(r.get("trans_err_mean_mm"), 1)
        rmed = _f(r.get("rot_err_median_deg"), 1)
        rmean = _f(r.get("rot_err_mean_deg"), 1)
        nm = r.get("n_matched")
        ng = r.get("n_gt") or r.get("n")
        nstr = f"{nm} / {ng}" if (nm is not None or ng is not None) else "—"
        out.append(
            f"| {oid} | {name} | {sym} | **{ar if ar is not None else '—'}** | "
            f"{add if add is not None else '—'} | "
            f"{tmed if tmed is not None else '—'} / {tmean if tmean is not None else '—'} | "
            f"{rmed if rmed is not None else '—'} / {rmean if rmean is not None else '—'} | "
            f"{nstr} |"
        )
    return out


def patch_project_report(report_md, rows, overall, stamp):
    """Replace the data rows of the §4.2 table in PROJECT_REPORT.md in place.

    We locate the table by its header line (the '| obj | part | sym | AR ...'
    row), keep the header + separator, and swap the data rows up to the first
    non-table line. Idempotent."""
    if not os.path.isfile(report_md):
        sys.stderr.write(f"[e2e_report] WARN PROJECT_REPORT.md not found: {report_md}\n")
        return False
    src = open(report_md).read().splitlines()
    # find the §4.2 table header row
    hdr_idx = None
    for i, ln in enumerate(src):
        if ln.strip().startswith("| obj | part | sym | AR"):
            hdr_idx = i
            break
    if hdr_idx is None:
        sys.stderr.write("[e2e_report] WARN could not find §4.2 table header — skipping in-place patch\n")
        return False
    # header is hdr_idx, separator is hdr_idx+1; data rows follow until a
    # non-'|' line.
    sep_idx = hdr_idx + 1
    j = sep_idx + 1
    while j < len(src) and src[j].lstrip().startswith("|"):
        j += 1
    new_rows = build_results_table_rows(rows)
    note = (f"\n_Numbers regenerated by `box_src/e2e_finish.sh` ({stamp}); "
            f"overall AR = {_f(overall.get('AR'), 3)}._")
    patched = src[:sep_idx + 1] + new_rows + src[j:]
    # drop a stale regeneration note if one already sits right after the table
    out_lines = []
    k = 0
    while k < len(patched):
        ln = patched[k]
        if ln.startswith("_Numbers regenerated by `box_src/e2e_finish.sh`"):
            k += 1
            continue
        out_lines.append(ln)
        k += 1
    # insert the fresh note right after the (new) last table row
    last_row = sep_idx + len(new_rows)
    out_lines = out_lines[:last_row + 1] + [note] + out_lines[last_row + 1:]
    open(report_md, "w").write("\n".join(out_lines) + "\n")
    sys.stderr.write(f"[e2e_report] patched §4.2 table in {report_md}\n")
    return True


def write_results_phase2(out_md, rows, overall, mode, preds_note, stamp,
                         planar_refine):
    base_tbl = []
    cmp_tbl = []
    for oid in TRAINABLE_OBJ_IDS:
        r = rows.get(oid) or {}
        name = r.get("name", {1: "Anker_Kurz", 2: "Anker_Lang", 6: "Zahnrad"}[oid])
        ar = _f(r.get("AR"), 3)
        b = BASELINE_AR.get(name)
        tmed = _f(r.get("trans_err_median_mm"), 1)
        rmed = _f(r.get("rot_err_median_deg"), 1)
        delta = None if (ar is None or b is None) else round(ar - b, 3)
        base_tbl.append(
            f"| {name} | {b if b is not None else '—'} | "
            f"{ar if ar is not None else '—'} | "
            f"{('+' if (delta or 0) >= 0 else '') + str(delta) if delta is not None else '—'} | "
            f"{verdict(name, ar)} |"
        )
        cmp_tbl.append(
            f"| {name} | {ar if ar is not None else '—'} | "
            f"{tmed if tmed is not None else '—'} | {rmed if rmed is not None else '—'} |"
        )
    ov_ar = _f(overall.get("AR"), 3)
    pr = "ON (planar refine applied)" if planar_refine else "OFF (raw GDRNPP poses)"
    body = f"""# RESULTS — Phase 2 (GDRNPP, >20%-visibility val split)

_Generated by `box_src/e2e_finish.sh` ({mode} mode) on {stamp}._
_Predictions: {preds_note}_
_Planar refine: {pr}_

This is the autonomous finish report: real GDRNPP weights evaluated with the
symmetry-aware `box_src/eval_bop.py` harness over the **>20%-visibility**
filtered val split, scored against the frozen §0 in-house baseline.

## 1. Baseline → Phase-2  (AR per trainable object)

The §0 in-house baseline AR is the bar each part has to clear.

| object | §0 baseline AR | Phase-2 AR | Δ AR | verdict |
|---|---|---|---|---|
{os.linesep.join(base_tbl)}

**Overall Phase-2 AR (all scored objects): {ov_ar if ov_ar is not None else '—'}**

## 2. Phase-2 detail (the median that describes the typical case)

`AR` = mean(AR_MSSD, AR_MSPD); `rot` is **symmetry-resolved** (the metric that
makes the Anker's continuous-Y and the Zahnrad's C_7 symmetry not punish a
pose that is correct up to its own symmetry).

| object | AR | trans median (mm) | rot median (°) |
|---|---|---|---|
{os.linesep.join(cmp_tbl)}

_The §0 baseline's single headline rotation error was **{BASELINE_ROT_MED_DEG:.0f}°**
(sym-resolved); compare the per-object rot-median column above._

## 3. How to reproduce

```bash
# dry-run (re-score existing predictions, rebuild docs + split-screen + viewer):
box_src/e2e_finish.sh --dry-run

# full finish (GPU inference on val scenes, then everything above):
box_src/e2e_finish.sh
```

## Related
- [[PROJECT_REPORT]] — §4 Results (this table is mirrored into §4.2)
- `box_src/eval_bop.py` — the symmetry-aware metric harness
- `box_src/e2e_finish.sh` — the one-command finish that produced this file
"""
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w").write(body)
    sys.stderr.write(f"[e2e_report] wrote {out_md}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="eval_bop report.json")
    ap.add_argument("--report-md", required=True, help="PROJECT_REPORT.md to patch")
    ap.add_argument("--out-md", required=True, help="RESULTS_PHASE2.md to write")
    ap.add_argument("--mode", default="full", choices=["dry-run", "full"])
    ap.add_argument("--preds-note", default="preds_all.csv (val, gt-bbox)")
    ap.add_argument("--planar-refine", action="store_true")
    a = ap.parse_args()

    rows, overall, _ = load_report(a.report)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    write_results_phase2(a.out_md, rows, overall, a.mode, a.preds_note, stamp,
                         a.planar_refine)
    patch_project_report(a.report_md, rows, overall, stamp)

    # echo a compact baseline-comparison summary to stdout for the harness log
    print("== Phase-2 vs §0 baseline (AR) ==")
    for oid in TRAINABLE_OBJ_IDS:
        r = rows.get(oid) or {}
        name = r.get("name", {1: "Anker_Kurz", 2: "Anker_Lang", 6: "Zahnrad"}[oid])
        ar = _f(r.get("AR"), 3)
        print(f"  {name:12s} baseline={BASELINE_AR.get(name)}  phase2={ar}  "
              f"-> {verdict(name, ar)}")
    print(f"  OVERALL phase2 AR = {_f(overall.get('AR'), 3)}")


if __name__ == "__main__":
    main()
