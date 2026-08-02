"""run_final_t170.py — T-170 FINALER 12×100-Eval-Lauf (D-5 PIVOT, Max).

Faehrt die echte 12-Kombi-Eval ueber ~100 DETERMINISTISCH gesampelte Frames statt 10,
fuer robustere AR: im=0,10,20,..,90 PRO Szene × 10 Szenen (000000-000009) = 100 Frames,
DIESELBEN 100 fuer ALLE 12 Kombis (run_batch ist seed-major Round-Robin). Nutzt EXAKT
die batch_eval-Naht (discover_scenes(frames=...) + run_batch + http_predict +
subprocess_eval) — KEINE zweite Eval-Implementierung, kein Drift von der API-Tabelle.

Warum als CLI-Skript statt :8077/api/eval/run: kip-server.service ist systemd OHNE
--reload; der Multi-Frame-Code (discover_scenes frames=) braeuchte sonst einen
kip-server-Restart (Mission: :8077 NICHT anfassen). Dieser Runner ist ein eigener
Prozess, schreibt den Run mit der STANDARD-Timestamp-run-id (run-<utc>) direkt in
EVAL_OUT (project/temp/batch_eval/<run-id>/) -> der Viewer liest ihn via
list_runs IDENTISCH (Run = Ordner + results.json + EVAL.md). Output byte-gleich zum
API-Job, nur 100 statt 10 Frames.

depth-init (GP_PRE_REFINE_SNAP) wurde in T-170 VERWORFEN (pre-refine snap -0.5..-0.8%,
nie besser) -> dieser finale Run laeuft auf der BASELINE (Container SNAP=0, gigapose
ICP=0.025 aus T-167).

Aufruf (Box, isaacsim-venv):
  /mnt/data/isaacsim-venv/bin/python project/eval/run_final_t170.py
"""
import sys
import time

sys.path.insert(0, "project")
from eval import batch_eval as be  # noqa: E402

VAL_ROOT = "project/bop/pose_isaac/val"
DATASET_DIR = "/mnt/data/kip_pose/project/bop/pose_isaac"
SPLIT = "val"
GATEWAY = "http://localhost:8090"
OUT = "project/temp/batch_eval"
ITERATIONS = 5
FRAMES = list(range(0, 100, 10))   # [0,10,20,..,90] — 10 Frames/Szene
N_SCENES = 10                       # 000000-000009


def main():
    scenes = be.discover_scenes(VAL_ROOT, seeds=N_SCENES, frames=FRAMES)
    if not scenes:
        print(f"[final-t170] KEINE Szenen unter {VAL_ROOT}", file=sys.stderr)
        return 2
    from collections import Counter
    per_scene = Counter(s["scene_id"] for s in scenes)
    print(f"[final-t170] {len(be.EVAL_CONFIGS)} Kombis × {len(scenes)} Frames "
          f"({N_SCENES} Szenen × {len(FRAMES)} im_ids)", file=sys.stderr)
    print(f"[final-t170] Frames/Szene: {dict(per_scene)}", file=sys.stderr)
    print(f"[final-t170] im_ids: {FRAMES}", file=sys.stderr)

    predict_fn = be.http_predict(GATEWAY, iterations=ITERATIONS)
    eval_fn = be.subprocess_eval(DATASET_DIR, split=SPLIT)

    t0 = time.time()
    last = [0]

    def _prog(pct, phase):
        # grober Heartbeat, damit man im run_in_background Fortschritt sieht
        if pct >= last[0] + 5:
            last[0] = pct
            print(f"[final-t170] {pct}% · {phase}", file=sys.stderr, flush=True)

    results = be.run_batch(
        be.EVAL_CONFIGS, scenes, predict_fn, eval_fn, OUT,
        progress=_prog, warn=lambda m: print(m, file=sys.stderr, flush=True))

    dur = round(time.time() - t0, 1)
    print(f"\n[final-t170] RUN-ID = {results['run_id']}  ({dur}s, "
          f"{results['n_configs']} Kombis × {results['n_scenes']} Frames)")
    print("[final-t170] === STANDINGS (AR DESC) ===")
    for s in results["standings"]:
        ar = s.get("ar")
        ar_s = f"{ar:.4f}" if isinstance(ar, (int, float)) else "None"
        print(f"  {s.get('rank'):>2}. {s['config_key']:30} AR={ar_s} "
              f"mod={s.get('modality'):4} seg_ms={s.get('seg_ms')} "
              f"pose_ms={s.get('pose_ms')} cov={s.get('coverage')} "
              f"crash={s.get('crash_rate')}")
    print(f"[final-t170] -> {OUT}/{results['run_id']}/results.json + EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
