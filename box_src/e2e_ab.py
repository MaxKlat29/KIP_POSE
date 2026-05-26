#!/usr/bin/env python3
"""e2e_ab.py — A/B-over-the-levers + auto-best + decision flags  (T-048 / S-050).

The brain of the hardened finish. The lever A/B is scored at the *prediction*
level: each config (planar-only / +M2 / +TTA / +M2+TTA) produces one refined
BOP-results CSV, each CSV is scored by box_src/eval_bop.py into ONE report.json.
This script ingests those per-config reports, computes a mean-AR over the
trainable objects, picks the best config, and emits:

  1. A comparison table (markdown + stdout) of every config vs.
       §0 in-house baseline      (Anker_Kurz 0.59 · Anker_Lang 0.61 · Zahnrad 0.36)
       §Phase-1 planar baseline  (Anker_Kurz 0.645 · Anker_Lang 0.650 · Zahnrad 0.36)
  2. The auto-best config (highest mean-AR over trainable objects).
  3. Decision flags that document remaining-AR gaps (purely advisory; both the
     SO(3)-head plan and the MegaPose-M2 path were measured to ceiling and
     scoped out post-Phase-2 — see project/docs/RESULTS_PHASE2.md).
       - Zahnrad AR (best config) < 0.70  ->  ZAHNRAD_GAP (informational)
       - any Anker  AR (best config) < 0.80 ->  ANKER_CEILING (informational)
  4. A machine-readable ab_result.json the harness reads to know which
     config to materialise into the real pose_result.json.

stdlib-only. Same code path for --dry-run and full — only the *input reports*
differ (dry-run synthesises the configs it has no real CSV for, see --synth).

Usage:
  # full: one --report per config that actually got scored on the box
  python3 e2e_ab.py \
      --report planar=results/eval/planar/report.json \
      --report m2=results/eval/m2/report.json \
      --report tta=results/eval/tta/report.json \
      --report m2_tta=results/eval/m2_tta/report.json \
      --out-json results/eval/ab_result.json \
      --out-md   project/docs/AB_LEVERS.md

  # dry-run: a single real report stands in for the planar baseline, the other
  # three configs are SYNTHESISED (deterministic small deltas) so the
  # orchestration + auto-best + decision logic is exercised end-to-end offline.
  python3 e2e_ab.py --report planar=results/eval/report.json \
      --synth --out-json ... --out-md ...
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

# ── frozen baselines (the bars every config is measured against) ──────────────
BASELINE_AR = {"Anker_Kurz": 0.59, "Anker_Lang": 0.61, "Zahnrad": 0.36}     # §0
PHASE1_AR = {"Anker_Kurz": 0.645, "Anker_Lang": 0.650, "Zahnrad": 0.36}     # §Phase-1 planar
TRAINABLE_OBJ_IDS = (1, 2, 6)
OBJ_NAME = {1: "Anker_Kurz", 2: "Anker_Lang", 6: "Zahnrad"}

# ── decision thresholds (the levers that decide the *next* step) ──────────────
ZAHNRAD_SO3_THR = 0.70      # Zahnrad AR below this -> train the SO(3) head (Phase-3)
ANKER_MEGAPOSE_THR = 0.80   # Anker AR below this  -> re-check M2-MegaPose settings

# canonical config order (also the human-readable labels)
CONFIG_ORDER = ["planar", "m2", "tta", "m2_tta"]
CONFIG_LABEL = {
    "planar": "Planar-only (baseline)",
    "m2":     "+M2 MegaPose-refiner",
    "tta":    "+TTA",
    "m2_tta": "+M2+TTA",
}


def _f(x, nd=3):
    return None if x is None else round(float(x), nd)


def load_report(path):
    """Return obj_id -> per-object row dict for the trainable objects. Tolerant
    to per_object being a dict (obj_id-keyed) or a list."""
    doc = json.load(open(path))
    res = doc.get("results", doc)
    po = res.get("per_object", {})
    rows = {}
    items = po.items() if isinstance(po, dict) else enumerate(po)
    for k, r in items:
        oid = int(r.get("obj_id", k))
        rows[oid] = r
    return rows


def mean_ar(rows):
    """Mean AR over the trainable objects present (None-safe). The selection
    metric — only the three trained objects count, false-zero rows for
    untrained objects (Buerstenhalter/Ring) are excluded."""
    vals = [rows[o]["AR"] for o in TRAINABLE_OBJ_IDS
            if o in rows and rows[o].get("AR") is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def synth_configs(planar_rows):
    """DRY-RUN ONLY. From the real planar report, derive plausible deltas for the
    other three configs so the auto-best + decision logic runs offline. The
    deltas are DETERMINISTIC and clearly synthetic (so a real run never collides):
      - m2     : Anker +0.06 each (MegaPose fixes the 180° flip), Zahnrad +0.05
      - tta    : Anker +0.02 each, Zahnrad +0.03 (rot-vote helps the C_7 a bit)
      - m2_tta : the better of the two per-object deltas, additively capped
    Returns {config_name: rows-like-dict}."""
    def bump(rows, deltas):
        out = {}
        for oid, r in rows.items():
            nr = dict(r)
            d = deltas.get(oid, 0.0)
            if nr.get("AR") is not None:
                nr["AR"] = min(1.0, round(float(nr["AR"]) + d, 4))
            out[oid] = nr
        return out

    m2 = bump(planar_rows, {1: 0.06, 2: 0.06, 6: 0.05})
    tta = bump(planar_rows, {1: 0.02, 2: 0.02, 6: 0.03})
    m2_tta = bump(planar_rows, {1: 0.07, 2: 0.07, 6: 0.07})
    return {"m2": m2, "tta": tta, "m2_tta": m2_tta}


def build_decision(best_name, best_rows):
    """The engineering-decision flags off the WINNING config."""
    zr = best_rows.get(6, {}).get("AR")
    ak = best_rows.get(1, {}).get("AR")
    al = best_rows.get(2, {}).get("AR")
    flags = []
    train_so3 = zr is not None and zr < ZAHNRAD_SO3_THR
    check_megapose = (ak is not None and ak < ANKER_MEGAPOSE_THR) or \
                     (al is not None and al < ANKER_MEGAPOSE_THR)
    if train_so3:
        flags.append({
            "flag": "ZAHNRAD_GAP",
            "reason": f"Zahnrad AR={_f(zr)} < {ZAHNRAD_SO3_THR} on the best config "
                      f"({CONFIG_LABEL.get(best_name, best_name)}). Phase-2 retrain "
                      f"with C_7-symmetry-aware PM-loss + best-by-val selection "
                      f"is the operational target; the SO(3)-classification-head "
                      f"plan was measured to ceiling and scoped out.",
            "action": "informational — see project/docs/RESULTS_PHASE2.md",
        })
    if check_megapose:
        low = []
        if ak is not None and ak < ANKER_MEGAPOSE_THR:
            low.append(f"Anker_Kurz={_f(ak)}")
        if al is not None and al < ANKER_MEGAPOSE_THR:
            low.append(f"Anker_Lang={_f(al)}")
        flags.append({
            "flag": "ANKER_CEILING",
            "reason": f"{', '.join(low)} < {ANKER_MEGAPOSE_THR} on the best config. "
                      f"MegaPose-M2 (cpu_edge + megapose-scorer), n_iter sweep, "
                      f"mask-IoU ensemble, end-region scoring, custom flip-classifier "
                      f"all stuck at or below planar-Z-Snap — C_2 head-tail-flip "
                      f"ambiguity is not resolvable from a single top-down RGB.",
            "action": "informational — physical ceiling, needs depth or 2nd view",
        })
    return {
        "zahnrad_gap": train_so3,
        "anker_ceiling": check_megapose,
        "flags": flags,
    }


def fmt_ar(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def fmt_delta(now, base):
    if not isinstance(now, (int, float)) or not isinstance(base, (int, float)):
        return "—"
    d = now - base
    return f"{'+' if d >= 0 else ''}{d:.3f}"


def write_markdown(out_md, configs, best_name, decision, mode, stamp):
    """The A/B levers comparison doc: every config's per-object AR + mean, vs §0
    and §Phase-1, the winner marked, and the decision flags."""
    lines = []
    A = lines.append
    A("# A/B over the levers — autonomous finish\n")
    A(f"_Generated by `box_src/e2e_finish.sh` -> `e2e_ab.py` ({mode} mode) on {stamp}._\n")
    A("The improved pipeline's levers, scored against each other on the "
      "**>20%-visibility** filtered val split with the symmetry-aware "
      "`box_src/eval_bop.py` harness. AR per **trainable** object; `mean` is the "
      "selection metric.\n")

    # main comparison table
    A("## 1. Per-config AR (trainable objects)\n")
    A("| config | Anker_Kurz | Anker_Lang | Zahnrad | **mean** | winner |")
    A("|---|---|---|---|---|---|")
    for name in CONFIG_ORDER:
        if name not in configs:
            continue
        rows = configs[name]
        mar = mean_ar(rows)
        marker = " ✅" if name == best_name else ""
        A(f"| {CONFIG_LABEL.get(name, name)} | "
          f"{fmt_ar(rows.get(1, {}).get('AR'))} | "
          f"{fmt_ar(rows.get(2, {}).get('AR'))} | "
          f"{fmt_ar(rows.get(6, {}).get('AR'))} | "
          f"**{mar:.3f}** |{marker} |")

    # vs baselines, best config only
    best = configs[best_name]
    A("\n## 2. Best config vs. baselines\n")
    A(f"Auto-selected: **{CONFIG_LABEL.get(best_name, best_name)}** "
      f"(highest mean AR = {mean_ar(best):.3f}).\n")
    A("| object | §0 baseline | §Phase-1 planar | best config | Δ vs §0 | Δ vs Phase-1 |")
    A("|---|---|---|---|---|---|")
    for oid in TRAINABLE_OBJ_IDS:
        name = OBJ_NAME[oid]
        now = best.get(oid, {}).get("AR")
        b0 = BASELINE_AR[name]
        b1 = PHASE1_AR[name]
        A(f"| {name} | {b0:.3f} | {b1:.3f} | {fmt_ar(now)} | "
          f"{fmt_delta(now, b0)} | {fmt_delta(now, b1)} |")
    mean0 = sum(BASELINE_AR.values()) / 3
    mean1 = sum(PHASE1_AR.values()) / 3
    A(f"| **mean** | {mean0:.3f} | {mean1:.3f} | **{mean_ar(best):.3f}** | "
      f"{fmt_delta(mean_ar(best), mean0)} | {fmt_delta(mean_ar(best), mean1)} |")

    # decision flags (informational; both directions measured to ceiling, see RESULTS_PHASE2.md)
    A("\n## 3. Outcome flags (informational)\n")
    if not decision["flags"]:
        A("- ✅ No gap raised — every trainable object clears its threshold "
          f"(Zahnrad ≥ {ZAHNRAD_SO3_THR}, Anker ≥ {ANKER_MEGAPOSE_THR}). "
          "The pipeline is finish-ready as-is.")
    for fl in decision["flags"]:
        A(f"- ⚠️ **{fl['flag']}** — {fl['reason']}")
        A(f"  - {fl['action']}")

    A("\n## Related")
    A("- [[RESULTS_PHASE2]] — the headline finish results")
    A("- `box_src/e2e_finish.sh` — the one-command finish")
    A("- `box_src/refine_eval.py` (planar) · `box_src/rc_refine_eval.py` (M2) · "
      "`project/tta_pose.py` (TTA) — the levers")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w").write("\n".join(lines) + "\n")
    sys.stderr.write(f"[e2e_ab] wrote {out_md}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", default=[],
                    help="config=path/to/report.json (repeatable). config in "
                         f"{CONFIG_ORDER}")
    ap.add_argument("--synth", action="store_true",
                    help="dry-run: synthesise the configs not supplied (from the "
                         "planar report) so auto-best+decision run offline")
    ap.add_argument("--out-json", required=True, help="ab_result.json (harness reads this)")
    ap.add_argument("--out-md", required=True, help="AB_LEVERS.md comparison doc")
    ap.add_argument("--mode", default="full", choices=["dry-run", "full"])
    a = ap.parse_args()

    # parse the supplied reports
    configs = {}
    for spec in a.report:
        if "=" not in spec:
            ap.error(f"--report must be config=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if name not in CONFIG_ORDER:
            ap.error(f"unknown config {name!r}; expected one of {CONFIG_ORDER}")
        if not os.path.isfile(path):
            ap.error(f"report not found for config {name!r}: {path}")
        configs[name] = load_report(path)

    if not configs:
        ap.error("no --report given")

    # dry-run: synthesise the missing configs off the planar report
    if a.synth and "planar" in configs:
        for name, rows in synth_configs(configs["planar"]).items():
            configs.setdefault(name, rows)

    # auto-best: highest mean AR over the trainable objects
    scored = {name: mean_ar(rows) for name, rows in configs.items()}
    best_name = max(scored, key=lambda n: scored[n])
    best_rows = configs[best_name]
    decision = build_decision(best_name, best_rows)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    write_markdown(a.out_md, configs, best_name, decision, a.mode, stamp)

    # machine-readable result the harness reads
    out = {
        "mode": a.mode,
        "generated": stamp,
        "configs": {
            name: {
                "mean_ar": round(scored[name], 4),
                "per_object_ar": {
                    OBJ_NAME[o]: _f(configs[name].get(o, {}).get("AR"))
                    for o in TRAINABLE_OBJ_IDS
                },
            }
            for name in configs
        },
        "best_config": best_name,
        "best_label": CONFIG_LABEL.get(best_name, best_name),
        "best_mean_ar": round(scored[best_name], 4),
        "baselines": {"section0": BASELINE_AR, "phase1_planar": PHASE1_AR},
        "decision": decision,
    }
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=2)
    sys.stderr.write(f"[e2e_ab] wrote {a.out_json}\n")

    # compact stdout summary for the harness log
    print("== A/B over the levers ==")
    for name in CONFIG_ORDER:
        if name not in scored:
            continue
        star = "  <-- BEST" if name == best_name else ""
        print(f"  {CONFIG_LABEL.get(name, name):26s} mean_AR={scored[name]:.3f}{star}")
    print(f"== AUTO-BEST: {CONFIG_LABEL.get(best_name, best_name)} "
          f"(mean_AR={scored[best_name]:.3f}) ==")
    print("== decision flags ==")
    if not decision["flags"]:
        print("  (none — pipeline finish-ready)")
    for fl in decision["flags"]:
        print(f"  {fl['flag']}: {fl['reason']}")


if __name__ == "__main__":
    main()
