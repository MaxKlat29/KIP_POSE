"""merge_runs.py — mehrere Batch-Eval-Teilläufe zu EINEM Run zusammenführen (T-177).

Warum: der 12-Kombi-Lauf passt mit dem fp-svc-Peak (~13.4 GB, T-133) nicht mehr
all-resident in die 24 GB — der Final-Lauf fährt deshalb als VRAM-gestaffelte
Teilläufe (Nicht-sam3 / sam3 ohne FP / sam3+FP), die hier zu einem Run-Ordner
mit Standard-run-id gemerged werden. Der Viewer (list_runs) sieht danach EINEN
kuratierten Lauf, byte-kompatibel zu run_batch-Output (results.json + EVAL.md +
csv/ + eval/).

Konflikt-Schutz: eine config_key darf nur in EINEM Teillauf vorkommen.

Aufruf (Box, cwd /mnt/data/kip_pose):
  /mnt/data/isaacsim-venv/bin/python project/eval/merge_runs.py \
      --runs t177-final-b1,t177-final-b2a,t177-final-b2b \
      [--run-id run-<utc>] [--date-from-newest]
"""
import argparse
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, "project")
from eval import batch_eval as be  # noqa: E402

OUT = pathlib.Path("project/temp/batch_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Komma-Liste von Quell-run-ids")
    ap.add_argument("--run-id", default=None,
                    help="Ziel-run-id (Default run-<utc> = Viewer-Konvention)")
    args = ap.parse_args()

    src_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    srcs = []
    for rid in src_ids:
        rj = OUT / rid / "results.json"
        if not rj.is_file():
            print(f"[merge] FEHLT: {rj}", file=sys.stderr)
            return 2
        srcs.append((rid, json.loads(rj.read_text())))

    # Konflikt-Check: jede config_key genau einmal.
    seen = {}
    for rid, res in srcs:
        for s in res["standings"]:
            k = s["config_key"]
            if k in seen:
                print(f"[merge] KONFLIKT: {k} in {seen[k]} UND {rid}", file=sys.stderr)
                return 2
            seen[k] = rid

    run_id = args.run_id or ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = OUT / run_id
    if run_dir.exists():
        print(f"[merge] Ziel existiert schon: {run_dir}", file=sys.stderr)
        return 2
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "eval").mkdir(parents=True)

    standings, configs = [], []
    duration = 0.0
    n_scenes = 0
    newest_date = ""
    for rid, res in srcs:
        standings.extend(res["standings"])
        configs.extend(res.get("configs", []))
        duration += float(res.get("duration_s") or 0.0)
        n_scenes = max(n_scenes, int(res.get("n_scenes") or 0))
        newest_date = max(newest_date, res.get("date") or "")
        sdir = OUT / rid
        for f in (sdir / "csv").glob("*.csv"):
            shutil.copy2(f, run_dir / "csv" / f.name)
        for d in (sdir / "eval").iterdir():
            if d.is_dir():
                shutil.copytree(d, run_dir / "eval" / d.name)

    # Re-Rank wie build_standings: ar DESC, None ans Ende, stabil nach config_key.
    standings.sort(key=lambda e: (e["ar"] is None, -(e["ar"] or 0.0), e["config_key"]))
    for rank, e in enumerate(standings, 1):
        e["rank"] = rank

    results = {
        "run_id": run_id,
        "date": newest_date,
        "duration_s": round(duration, 1),
        "n_configs": len(standings),
        "n_scenes": n_scenes,
        "configs": configs,
        "standings": standings,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "EVAL.md").write_text(be.render_markdown(results))
    print(f"[merge] {run_id}: {len(standings)} Kombis aus {len(srcs)} Teilläufen "
          f"-> {run_dir}")
    for e in standings:
        ar = f"{e['ar']:.3f}" if e.get("ar") is not None else "None"
        print(f"  #{e['rank']:2d} {e['config_key']:28s} AR={ar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
