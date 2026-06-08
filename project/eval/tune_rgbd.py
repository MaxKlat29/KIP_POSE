"""tune_rgbd.py — T-167: schnelles RGB-D-Refiner-Tuning-Harness.

Faehrt NUR eine gewaehlte Untermenge der EVAL_CONFIGS (per --combos) ueber die
fixen 10 SDG-val-Szenen, mit variabler --iterations, gegen das laufende Mesh-
Gateway. Nutzt EXAKT die batch_eval-Naht (run_batch + http_predict +
subprocess_eval) — keine zweite Eval-/Mapping-Implementierung, kein Drift von
der 12x10-Tabelle. Pro Lauf eine results.json + EVAL.md unter --out.

Zweck (T-167): die Translations-Hebel der RGB-D-Refiner messen — FoundationPose
`est_refine_iter` (= iterations) und GigaPose-3D GenFlow-iter/ICP. Vergleicht AR
+ pose_ms gegen die iter=5-Baseline (run-20260608T113628Z).

Aufruf (Box, isaacsim-venv):
  python project/eval/tune_rgbd.py \
    --scenes-dir project/bop/pose_isaac/val \
    --dataset-dir project/bop/pose_isaac --split val \
    --gateway http://localhost:8090 \
    --combos yolo_seg__foundationpose,yolo_seg__gigapose_rgbd \
    --iterations 10 --out project/temp/batch_eval --run-id tune-fp10
"""
import argparse
import sys

from batch_eval import (
    EVAL_CONFIGS, discover_scenes, http_predict, subprocess_eval, run_batch,
)


def _select(combo_ids):
    """EVAL_CONFIGS gefiltert auf die gewuenschten combo-ids (FEASIBLE_COMBOS-id).
    Reihenfolge = Eingabe-Reihenfolge. Unbekannte id -> Fehler (kein stilles Skip)."""
    by_id = {c["id"]: c for c in EVAL_CONFIGS}
    out = []
    for cid in combo_ids:
        if cid not in by_id:
            raise SystemExit(
                f"[tune_rgbd] unbekannte combo-id {cid!r}. Bekannt: {sorted(by_id)}")
        out.append(by_id[cid])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-167 RGB-D-Refiner-Tuning-Harness.")
    ap.add_argument("--scenes-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gateway", default="http://localhost:8090")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--combos", required=True,
                    help="Komma-Liste der combo-ids (FEASIBLE_COMBOS-id).")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--out", default="project/temp/batch_eval")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    configs = _select([c.strip() for c in args.combos.split(",") if c.strip()])
    scenes = discover_scenes(args.scenes_dir, seeds=args.seeds)
    if not scenes:
        print(f"[tune_rgbd] keine Szenen mit GT unter {args.scenes_dir}", file=sys.stderr)
        return 2

    print(f"[tune_rgbd] {len(configs)} Kombis x {len(scenes)} Szenen @ iterations={args.iterations}",
          file=sys.stderr)
    for c in configs:
        print(f"  - {c['id']:32} seg_source={c['seg_source']:8} pose_source={c['pose_source']:16} "
              f"needs_depth={c['needs_depth']}", file=sys.stderr)

    predict_fn = http_predict(args.gateway, iterations=args.iterations, top_n=args.top_n)
    eval_fn = subprocess_eval(args.dataset_dir, split=args.split)
    results = run_batch(configs, scenes, predict_fn, eval_fn, args.out,
                        run_id=args.run_id, warn=lambda m: print(m, file=sys.stderr))
    print(f"[tune_rgbd] {results['run_id']}: {results['n_configs']} Configs, "
          f"{results['n_scenes']} Szenen, {results['duration_s']}s", file=sys.stderr)

    # Kurz-Standings auf stdout (AR + pose_ms je Kombi, nach AR sortiert).
    # AR kann None sein (alle Szenen gecrasht/keine Coverage) -> defensiv formatieren.
    for s in results["standings"]:
        ar = s.get("ar")
        ar_s = f"{ar:.4f}" if isinstance(ar, (int, float)) else "None"
        print(f"{s['config_key']:34} ar={ar_s} pose_ms={s.get('pose_ms')} "
              f"cov={s.get('coverage')} crash={s.get('crash_rate')} modality={s.get('modality')}")
    print(f"[tune_rgbd] -> {args.out}/{results['run_id']}/results.json + EVAL.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
