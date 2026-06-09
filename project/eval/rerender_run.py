#!/usr/bin/env python3
"""rerender_run.py — T-171: einen bestehenden Batch-Eval-Run re-aggregieren + neu rendern.

**KEINE Re-Inferenz, KEIN Re-Scoring.** Liest das schon-gescorte `results.json` (das pro
Config ein `per_class`-Dict mit allen 6 BOP-Klassen-AR traegt), berechnet daraus die
**korrigierte primaere AR** = Mittel ueber die D1-aktiven Klassen (anker_kurz/lang, via
`batch_eval.ar_from_report`-Logik) und schreibt `results.json` + `EVAL.md` neu.

Vorher (Bug): primaere AR = eval_bop-overall.AR ueber ALLE 6 Objekte (4 untrainiert,
AR=0) → gdrnpp 0.295. Nachher: 2-Klassen-AR → gdrnpp 0.886. Die alte 6-obj-Zahl bleibt
als `ar_6obj`.

Idempotent: re-run aendert nichts mehr (primary_ar deckt sich dann schon mit ar_mean).
Backup: schreibt results.json.bak6obj einmalig (ueberschreibt es nicht beim Re-Run).

CLI:
  python3 project/eval/rerender_run.py <run_dir>
  # run_dir = .../batch_eval/run-20260608T201857Z  (enthaelt results.json + EVAL.md)
"""
from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from eval import batch_eval as be  # noqa: E402


def _report_from_per_class(per_class: dict, ar_6obj) -> dict:
    """Rekonstruiert einen eval_bop-report-Shape aus dem schon gescorten per_class.

    Das genuegt `be.ar_from_report`, um die korrigierte primaere AR (D1-aktive Klassen)
    + per_class + ar_6obj zu liefern — OHNE die CSV neu zu scoren. `overall.AR` =
    ar_6obj (die alte 6-obj-Zahl, falls bekannt). per_object.name traegt den Part-Namen
    direkt (per_class-Keys sind die CamelCase-Parts)."""
    per_object = {}
    for i, (name, ar) in enumerate(per_class.items(), 1):
        per_object[str(i)] = {"name": name, "AR": ar}
    overall = {"AR": ar_6obj} if ar_6obj is not None else {}
    return {"results": {"overall": overall, "per_object": per_object}}


def rerender(run_dir: str) -> dict:
    run = pathlib.Path(run_dir)
    rj = run / "results.json"
    if not rj.is_file():
        raise FileNotFoundError(f"kein results.json unter {run}")
    results = json.loads(rj.read_text())

    # Einmaliges Backup der 6-obj-Variante (nicht ueberschreiben beim Re-Run).
    bak = run / "results.json.bak6obj"
    if not bak.is_file():
        bak.write_text(json.dumps(results, indent=2))

    # ── configs: primaere ar_mean = D1-aktive Klassen, alte Zahl → ar_6obj ──
    ar_by_key = {}
    for c in results.get("configs", []):
        per_class = c.get("per_class") or {}
        # ar_6obj = die ALTE primaere Zahl (war eval_bop-overall.AR), falls nicht schon
        # auf den Re-Run-Zustand gesetzt. Idempotent: existiert ar_6obj bereits, nutze es.
        old_overall = c.get("ar_6obj")
        if old_overall is None:
            old_overall = c.get("ar_mean")     # erstes Re-Render: ar_mean IST noch 6-obj
        report = _report_from_per_class(per_class, old_overall)
        primary, _pc, ar_6obj = be.ar_from_report(report)
        c["ar_mean"] = primary
        c["ar_6obj"] = ar_6obj
        if c.get("ar_std") is None and primary is not None:
            c["ar_std"] = 0.0
        ar_by_key[c.get("run_config_id")] = (primary, ar_6obj)

    # ── standings: ar = primaere D1-AR, ar_6obj sekundaer, neu sortieren + ranken ──
    for s in results.get("standings", []):
        primary, ar_6obj = ar_by_key.get(s.get("config_key"), (s.get("ar"), s.get("ar_6obj")))
        s["ar"] = primary
        s["ar_6obj"] = ar_6obj
        if s.get("ar_std") is None and primary is not None:
            s["ar_std"] = 0.0
    # Re-Sort + Re-Rank (gleiche Semantik wie build_standings: ar DESC, None ans Ende).
    st = results.get("standings", [])
    st.sort(key=lambda e: (e.get("ar") is None, -(e.get("ar") or 0.0), e.get("config_key", "")))
    for rank, e in enumerate(st, 1):
        e["rank"] = rank

    # ── persist: results.json + EVAL.md neu (be.render_markdown nutzt die neuen Felder) ──
    rj.write_text(json.dumps(results, indent=2))
    (run / "EVAL.md").write_text(be.render_markdown(results))
    return results


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: rerender_run.py <run_dir>", file=sys.stderr)
        return 2
    results = rerender(argv[0])
    print(f"[rerender] {results.get('run_id')}: re-aggregiert + neu gerendert "
          f"({results.get('n_configs')} Configs)")
    for s in sorted(results.get("standings", []), key=lambda e: e.get("rank", 99)):
        print(f"  #{s['rank']:>2} {s['config_key']:<26} "
              f"AR={s['ar']:.3f}  (6obj={s.get('ar_6obj'):.3f})"
              if s.get("ar") is not None else
              f"  #{s['rank']:>2} {s['config_key']:<26} AR=—")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
