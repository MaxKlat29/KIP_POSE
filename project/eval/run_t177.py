"""run_t177.py — T-177 Perf-Pass: gezielte Kombi-Subset-Evals auf der batch_eval-Naht.

Wie run_final_t170.py (EXAKT dieselbe Naht: discover_scenes(frames=...) + run_batch +
http_predict + subprocess_eval, KEINE zweite Eval-Implementierung), aber parametrisiert:

  --configs     Komma-Liste von config_keys (Pflicht) — nur diese Kombis laufen
  --frames      Komma-Liste im_ids (Default 0,10,..,90 = identisch zum Final-Run,
                AR direkt vergleichbar mit run-20260608T201857Z)
  --iterations  Pose-Iterationen (Default 5; FP/GigaPose-Sweep T-177)
  --run-id      expliziter Run-Name (Default run-<utc>); t177-Prefix empfohlen,
                damit Experimente vom kuratierten Viewer-Run unterscheidbar bleiben

Aufruf (Box, cwd /mnt/data/kip_pose):
  /mnt/data/isaacsim-venv/bin/python project/eval/run_t177.py \
      --configs yolo_obb__foundationpose,yolo_obb__gigapose_rgbd \
      --run-id t177-depthfix-sanity --frames 0,10,20
"""
import argparse
import sys
import time

sys.path.insert(0, "project")
from eval import batch_eval as be  # noqa: E402

VAL_ROOT = "project/bop/pose_isaac/val"
DATASET_DIR = "/mnt/data/kip_pose/project/bop/pose_isaac"
SPLIT = "val"
GATEWAY = "http://localhost:8090"
OUT = "project/temp/batch_eval"
N_SCENES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--frames", default=",".join(str(i) for i in range(0, 100, 10)))
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gateway", default=GATEWAY)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="Inkrementelles Scoring nur jede N-te Szene (+ letzte). "
                         "End-AR identisch; spart eval_bop-Wall-Clock (T-177).")
    args = ap.parse_args()

    want = {c.strip() for c in args.configs.split(",") if c.strip()}
    known = {be.config_key(c) for c in be.EVAL_CONFIGS}
    unknown = want - known
    if unknown:
        print(f"[t177] unbekannte --configs: {sorted(unknown)}\n"
              f"       bekannt: {sorted(known)}", file=sys.stderr)
        return 2
    configs = [c for c in be.EVAL_CONFIGS if be.config_key(c) in want]

    frames = [int(f) for f in args.frames.split(",") if f.strip() != ""]
    scenes = be.discover_scenes(VAL_ROOT, seeds=N_SCENES, frames=frames)
    if not scenes:
        print(f"[t177] KEINE Szenen unter {VAL_ROOT}", file=sys.stderr)
        return 2

    print(f"[t177] {len(configs)} Kombis × {len(scenes)} Frames, "
          f"iterations={args.iterations}, run_id={args.run_id or '(auto)'}",
          file=sys.stderr)

    predict_fn = be.http_predict(args.gateway, iterations=args.iterations)
    eval_fn = be.subprocess_eval(DATASET_DIR, split=SPLIT)

    t0 = time.time()
    results = be.run_batch(configs, scenes, predict_fn, eval_fn, OUT,
                           run_id=args.run_id,
                           warn=lambda m: print(m, file=sys.stderr),
                           eval_every=args.eval_every)
    print(f"[t177] {results['run_id']}: {results['n_configs']} Kombis, "
          f"{results['n_scenes']} Frames, {time.time() - t0:.0f}s")
    for s in results.get("standings", []):
        ar = f"{s['ar']:.3f}" if s.get("ar") is not None else "None"
        print(f"[t177]   {s['config_key']:32s} AR={ar} "
              f"seg={s['seg_ms']}ms pose={s['pose_ms']}ms cov={s['coverage']:.0%} "
              f"crash={s.get('crash_rate', 0):.0%}")
    print(f"[t177] -> {OUT}/{results['run_id']}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
