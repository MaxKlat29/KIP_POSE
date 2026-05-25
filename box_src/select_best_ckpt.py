#!/usr/bin/env python3
"""select_best_ckpt.py — best-by-val checkpoint selection for GDRNPP (T-068).

OVERFITTING GUARD (Max). GDRNPP has NO in-training validation, NO early-stopping,
and the chain blindly deploys ``model_final.pth`` = the LAST epoch (160). On a
160-epoch RGB-only run over a small synthetic set that last epoch is a prime
overfitting candidate. This module picks the checkpoint that actually generalises
best on the held-out ``val`` split, by SYMMETRY-AWARE AR (our ``eval_bop.py``,
NOT GDRNPP's internal eval, which crashes with ``KeyError: pose_isaac`` in
``vsd_deltas`` — see the W4 note in projects/pose.md).

------------------------------------------------------------------------------
HOW IT WORKS (per object)

  1. ENUMERATE the kept checkpoints under
     ``output/gdrn/poseIsaacPbrSO/<obj>/model_*.pth`` (+ ``model_final.pth``).
     The T-068 retention (SOLVER.CHECKPOINT_PERIOD=10, MAX_TO_KEEP=20) keeps a
     wide even SPREAD (epochs 10,20,...,160 = 16 ckpts) so there is something
     EARLY to compare the final against.

  2. RANK each checkpoint cheaply: run GDRNPP inference ONCE per checkpoint via
        main_gdrn.py --eval-only --opts MODEL.WEIGHTS=<ckpt> VAL.SAVE_BOP_CSV_ONLY=True
     This is the CRASH-FREE path: ``save_and_eval_results`` writes the BOP-results
     CSV FIRST, then — because ``VAL.SAVE_BOP_CSV_ONLY=True`` — SKIPS the buggy
     bop_toolkit eval subprocess entirely (the one that raises KeyError:pose_isaac).
     The CSV is scored by our ``eval_bop.py`` (sym-aware AR). For speed the rank is
     computed on a val SUBSET (``--rank-max-images``); the final pick is re-scored
     on the FULL val.

  3. PICK the checkpoint with the highest AR (mean over the trained obj(s)). It may
     well be an EARLIER epoch than model_final — that is exactly the overfitting we
     are guarding against.

  4. MARK the winner as the deployed checkpoint: a symlink (or copy) ``model_best.pth``
     next to the per-object checkpoints. e2e_finish + inference then use
     ``model_best.pth`` instead of ``model_final.pth``.

The eval / ranking LOGIC (CSV parsing, AR aggregation, the pick) is pure and
unit-tested in project/tests/test_select_best_ckpt.py. The GPU inference is shelled
out to main_gdrn.py and is only exercised on the box.

------------------------------------------------------------------------------
USAGE (on the GPU box)

  # full select for one object (kept ckpts -> rank on 120-image subset -> best):
  /mnt/data/bop/gdrnpp-venv/bin/python box_src/select_best_ckpt.py \
      --obj zahnrad \
      --gdrn-root /mnt/data/bop/repos/gdrnpp \
      --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
      --bop-venv /mnt/data/bop/bop-venv/bin/python \
      --rank-max-images 120 \
      --apply                       # write model_best.pth symlink

  # DRY-RUN proof against the existing Phase-1 256/64 zahnrad checkpoints
  # (override the crop res so the older ckpts load; do NOT write the symlink):
  ... --obj zahnrad --input-res 256 --output-res 80 --no-apply --rank-max-images 100

NOTE the older Phase-1 checkpoints were trained at INPUT_RES=256/OUTPUT_RES=64;
the new Phase-2 ones at 320/80. ``--input-res/--output-res`` override the crop res
so a given checkpoint loads with matching shapes. ``--auto-res`` (default) sniffs the
res from the checkpoint's PnP head weight shape and overrides automatically.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Object id <-> name (FROZEN, see ADR §1: obj_id = detector class + 1)
# ---------------------------------------------------------------------------
OBJ_NAME_TO_ID = {
    "anker_kurz": 1,
    "anker_lang": 2,
    "buerstenhalter_2polig": 3,
    "getriebegehaeuse_typ4": 4,
    "ringmagnet": 5,
    "zahnrad": 6,
}


# ===========================================================================
# PURE LOGIC (unit-tested off-GPU)
# ===========================================================================
def epoch_from_ckpt(path):
    """Sort key: model_final sorts LAST (highest), model_<iter> by its iter.

    GDRNPP names per-epoch checkpoints ``model_<7-digit-iter>.pth`` and the last
    one ``model_final.pth``. We sort by iteration so the spread is in training
    order; model_final is the terminal one.
    """
    base = os.path.basename(path)
    if base == "model_final.pth":
        return float("inf")
    m = re.search(r"model_(\d+)\.pth$", base)
    return int(m.group(1)) if m else -1


def list_checkpoints(obj_dir):
    """All kept checkpoints for one object, in training order (final last).

    Returns a list of absolute paths. Excludes any ``model_best.pth`` symlink we
    may have written previously (never rank our own selection).
    """
    paths = []
    for p in glob.glob(os.path.join(obj_dir, "model_*.pth")):
        if os.path.basename(p) == "model_best.pth":
            continue
        paths.append(os.path.abspath(p))
    paths.sort(key=epoch_from_ckpt)
    return paths


def find_pred_csv(infer_dir):
    """The single BOP-results CSV GDRNPP wrote under an inference dir.

    GDRNPP writes it one level DOWN, under the per-dataset subdir:
      ``inference_<model>/<dataset>/<EXP_ID-iter0>_<dataset>-<split>.csv``
    and in --eval-only mode EXP_ID gets a ``_test`` suffix (e.g.
    ``zahnrad-test-iter0_pose_isaac-val.csv``). We glob the stable tail at BOTH the
    top level and one subdir deep (the real location) so the lookup is robust to the
    dataset-subdir nesting and the eval-only suffix.
    """
    patterns = [
        os.path.join(infer_dir, "*-iter0_pose_isaac-val.csv"),
        os.path.join(infer_dir, "*", "*-iter0_pose_isaac-val.csv"),
        os.path.join(infer_dir, "*.csv"),
        os.path.join(infer_dir, "*", "*.csv"),
    ]
    for pat in patterns:
        cands = sorted(glob.glob(pat))
        if cands:
            return cands[0]
    return None


def ar_for_object(report, obj_id):
    """Pull AR for one obj_id out of an eval_bop report dict (None if absent)."""
    r = report.get("results", report)
    po = r.get("per_object", {})
    if isinstance(po, dict):
        row = po.get(str(obj_id)) or po.get(obj_id)
    else:  # list shape
        row = next((x for x in po if x.get("obj_id") == obj_id), None)
    if not row:
        return None
    return row.get("AR")


def mean_ar(report, obj_ids):
    """Mean AR over the requested obj_ids, ignoring None / missing (None-safe)."""
    vals = [ar_for_object(report, o) for o in obj_ids]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def pick_best(scored):
    """Pick the (ckpt, ar) with the highest AR.

    ``scored`` is a list of dicts {ckpt, ar, ...}. Ties broken toward the EARLIER
    epoch (lower iteration) — a smaller-epoch model with equal val-AR is the safer,
    less-overfit choice. Returns the winning dict, or None if nothing scored.
    """
    usable = [s for s in scored if s.get("ar") is not None]
    if not usable:
        return None
    return max(usable, key=lambda s: (s["ar"], -epoch_from_ckpt(s["ckpt"])))


# ===========================================================================
# GPU SIDE (box-only): run inference for one checkpoint, then score it
# ===========================================================================
def infer_ckpt_csv(ckpt, obj, args):
    """Run GDRNPP --eval-only on ONE checkpoint -> a BOP-results CSV path.

    Uses VAL.SAVE_BOP_CSV_ONLY=True so the buggy internal eval (KeyError:pose_isaac)
    is skipped — the CSV is written before that block and we never reach it. Returns
    the CSV path, or None on failure. The non-zero exit of main_gdrn is tolerated as
    long as the CSV materialises (belt-and-suspenders against the known crash).
    """
    cfg_file = os.path.join(args.gdrn_root, "configs/gdrn/poseIsaacPbrSO", f"{obj}.py")
    out_dir = os.path.join(args.gdrn_root, "output/gdrn/poseIsaacPbrSO", obj)
    model_name = os.path.splitext(os.path.basename(ckpt))[0]
    # eval-only EXP_ID = "<obj>_test" -> inference_<obj>_test ... but do_test uses
    # the WEIGHTS basename for the inference dir; we glob both to be safe.
    infer_dir = os.path.join(out_dir, f"inference_{model_name}")

    in_res, out_res = resolve_res(ckpt, args)

    opts = [
        f"MODEL.WEIGHTS={ckpt}",
        "VAL.SAVE_BOP_CSV_ONLY=True",        # <- crash-free: skip buggy bop eval
        "VAL.USE_BOP=True",
        f"SOLVER.IMS_PER_BATCH={args.batch}",
        f"MODEL.POSE_NET.INPUT_RES={in_res}",
        f"MODEL.POSE_NET.OUTPUT_RES={out_res}",
        "TEST.SAVE_RESULTS_ONLY=False",      # we WANT the do_test CSV path
    ]
    cmd = [
        args.gdrn_venv,
        os.path.join(args.gdrn_root, "core/gdrn_modeling/main_gdrn.py"),
        "--config-file", cfg_file,
        "--eval-only", "--num-gpus", "1",
        "--opts", *opts,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = args.gdrn_root
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stderr.write(f"[select] infer {obj} ckpt={os.path.basename(ckpt)} "
                     f"res={in_res}/{out_res}\n")
    rc = subprocess.call(cmd, cwd=args.gdrn_root, env=env)
    csv = find_pred_csv(infer_dir)
    if csv is None:
        # fall back: scan any inference_* dir touched for this object (the eval-only
        # EXP_ID suffix can change the dir name)
        for d in sorted(glob.glob(os.path.join(out_dir, "inference_*")), key=os.path.getmtime, reverse=True):
            csv = find_pred_csv(d)
            if csv:
                break
    if csv is None:
        sys.stderr.write(f"[select] WARN no CSV for {os.path.basename(ckpt)} (rc={rc})\n")
    return csv


def resolve_res(ckpt, args):
    """Decide INPUT_RES/OUTPUT_RES for loading this checkpoint.

    Explicit --input-res/--output-res win. Otherwise (default --auto-res) try to
    sniff OUTPUT_RES from the checkpoint's geo-head / PnP weight shapes; fall back
    to the base_so defaults (320/80).
    """
    if args.input_res and args.output_res:
        return args.input_res, args.output_res
    if args.auto_res:
        sniffed = sniff_output_res(ckpt)
        if sniffed:
            in_res = sniffed * 4   # GDRNPP base: INPUT_RES = 4 * OUTPUT_RES
            return in_res, sniffed
    return args.input_res or 320, args.output_res or 80


def sniff_output_res(ckpt):
    """Best-effort OUTPUT_RES from a checkpoint's tensor shapes (torch needed).

    The XYZ/region head emits an OUTPUT_RES×OUTPUT_RES grid; its last conv/deconv
    weight or the PnP input dim encodes it. We look at a couple of known keys and
    map the spatial size back to OUTPUT_RES. Returns None if it can't tell.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd)
    except Exception:
        return None
    # ConvPnPNet first linear (pnp_net.fc1.weight): in_features = 128 * fss^2 where
    # final_spatial_size fss = OUTPUT_RES // 8. Verified on a real checkpoint:
    #   OUTPUT_RES=64 -> fss=8  -> 128*64  = 8192  (Phase-1 256/64 ckpts)
    #   OUTPUT_RES=80 -> fss=10 -> 128*100 = 12800 (Phase-2 320/80 ckpts)
    # Match ONLY the exact pnp_net.fc1.weight key — a loose `fc1.weight` match would
    # wrongly hit the backbone MLP fc1 layers.
    for k, v in sd.items():
        if k.endswith("pnp_net.fc1.weight"):
            in_feat = int(v.shape[1])
            for out_res in (64, 72, 80, 96, 128):
                fss = out_res // 8
                if in_feat == 128 * fss * fss:
                    return out_res
    return None


def score_csv(csv, args, max_images):
    """Score a BOP-results CSV with eval_bop.py -> the report dict.

    Runs eval_bop.py with the bop-venv (it imports bop_toolkit_lib). max_images=0
    means full val. Returns the parsed report dict or None.
    """
    out_dir = os.path.join(args.work_dir, "score_" + re.sub(r"[^a-zA-Z0-9_]", "_",
                                                            os.path.basename(csv)))
    cmd = [
        args.bop_venv,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_bop.py"),
        "--dataset-dir", args.dataset_dir,
        "--split", "val",
        "--preds", csv,
        "--out", out_dir,
    ]
    if max_images:
        cmd += ["--max-images", str(max_images)]
    rc = subprocess.call(cmd)
    rep_path = os.path.join(out_dir, "report.json")
    if rc != 0 or not os.path.isfile(rep_path):
        sys.stderr.write(f"[select] WARN eval_bop failed for {csv} (rc={rc})\n")
        return None
    with open(rep_path) as f:
        return json.load(f)


def apply_best(obj_dir, best_ckpt, copy=False):
    """Mark best_ckpt as model_best.pth (symlink by default, copy if asked)."""
    dst = os.path.join(obj_dir, "model_best.pth")
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if copy:
        import shutil
        shutil.copy2(best_ckpt, dst)
    else:
        # relative symlink so the output dir stays relocatable
        os.symlink(os.path.basename(best_ckpt), dst)
    return dst


# ===========================================================================
# DRIVER
# ===========================================================================
def select_for_object(obj, args):
    """Full select for one object. Returns a result dict."""
    obj_id = OBJ_NAME_TO_ID[obj]
    obj_dir = os.path.join(args.gdrn_root, "output/gdrn/poseIsaacPbrSO", obj)
    ckpts = list_checkpoints(obj_dir)
    if not ckpts:
        return {"obj": obj, "error": f"no checkpoints in {obj_dir}"}
    sys.stderr.write(f"[select] {obj}: {len(ckpts)} checkpoints "
                     f"(epochs {[epoch_from_ckpt(c) for c in ckpts]})\n")

    scored = []
    for ckpt in ckpts:
        csv = infer_ckpt_csv(ckpt, obj, args)
        if not csv:
            scored.append({"ckpt": ckpt, "ar": None, "csv": None})
            continue
        rep = score_csv(csv, args, max_images=args.rank_max_images)
        ar = mean_ar(rep, [obj_id]) if rep else None
        scored.append({"ckpt": ckpt, "ar": ar, "csv": csv})
        sys.stderr.write(f"[select]   {os.path.basename(ckpt)}: "
                         f"rank-AR={ar} (subset {args.rank_max_images or 'full'})\n")

    best = pick_best(scored)
    if best is None:
        return {"obj": obj, "error": "no checkpoint produced a scorable CSV",
                "scored": scored}

    # final FULL-val re-score on the winner (honest headline number)
    final_rep = score_csv(best["csv"], args, max_images=0)
    final_ar = mean_ar(final_rep, [obj_id]) if final_rep else best["ar"]

    final_ckpt = next((s for s in scored if os.path.basename(s["ckpt"]) == "model_final.pth"), None)
    result = {
        "obj": obj,
        "obj_id": obj_id,
        "n_checkpoints": len(ckpts),
        "best_ckpt": os.path.basename(best["ckpt"]),
        "best_rank_ar": best["ar"],
        "best_full_ar": final_ar,
        "model_final_rank_ar": final_ckpt["ar"] if final_ckpt else None,
        "best_is_final": os.path.basename(best["ckpt"]) == "model_final.pth",
        "ranking": [{"ckpt": os.path.basename(s["ckpt"]),
                     "epoch": epoch_from_ckpt(s["ckpt"]),
                     "rank_ar": s["ar"]} for s in scored],
    }
    if args.apply:
        dst = apply_best(obj_dir, best["ckpt"], copy=args.copy)
        result["deployed"] = dst
        # also stage the winner's BOP-results CSV under a STABLE name so e2e_finish
        # can consume the best-checkpoint predictions directly (not model_final's).
        if best.get("csv") and os.path.isfile(best["csv"]):
            import shutil
            best_csv = os.path.join(obj_dir, "preds_best.csv")
            shutil.copy2(best["csv"], best_csv)
            result["best_csv"] = best_csv
        sys.stderr.write(f"[select] {obj}: model_best.pth -> {os.path.basename(best['ckpt'])} "
                         f"(full-val AR={final_ar})\n")
    else:
        sys.stderr.write(f"[select] {obj}: would deploy {os.path.basename(best['ckpt'])} "
                         f"(--no-apply, nothing written)\n")
    return result


def main():
    ap = argparse.ArgumentParser(description="best-by-val GDRNPP checkpoint selector (T-068)")
    ap.add_argument("--obj", action="append", required=True,
                    help="object name (repeatable): anker_kurz | anker_lang | zahnrad | ...")
    ap.add_argument("--gdrn-root", default="/mnt/data/bop/repos/gdrnpp")
    ap.add_argument("--dataset-dir", default="/mnt/data/kip_pose/project/bop/pose_isaac")
    ap.add_argument("--gdrn-venv", default="/mnt/data/bop/gdrnpp-venv/bin/python",
                    help="python that runs main_gdrn.py (gdrnpp env)")
    ap.add_argument("--bop-venv", default="/mnt/data/bop/bop-venv/bin/python",
                    help="python that runs eval_bop.py (bop_toolkit env)")
    ap.add_argument("--rank-max-images", type=int, default=120,
                    help="images/scene cap for the cheap ranking pass (0 = full val)")
    ap.add_argument("--batch", type=int, default=16, help="IMS_PER_BATCH for inference")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--input-res", type=int, default=0, help="override crop INPUT_RES (0=auto/base)")
    ap.add_argument("--output-res", type=int, default=0, help="override OUTPUT_RES (0=auto/base)")
    ap.add_argument("--auto-res", dest="auto_res", action="store_true", default=True,
                    help="sniff OUTPUT_RES from the checkpoint shapes (default on)")
    ap.add_argument("--no-auto-res", dest="auto_res", action="store_false")
    ap.add_argument("--apply", dest="apply", action="store_true", default=True,
                    help="write model_best.pth (default). Use --no-apply for a dry run.")
    ap.add_argument("--no-apply", dest="apply", action="store_false")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink for model_best.pth")
    ap.add_argument("--work-dir", default="/mnt/data/bop/results/ckpt_select",
                    help="scratch dir for per-ckpt eval reports")
    ap.add_argument("--out-json", help="write the full selection result here")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    for o in args.obj:
        if o not in OBJ_NAME_TO_ID:
            ap.error(f"unknown object '{o}' (known: {sorted(OBJ_NAME_TO_ID)})")

    results = [select_for_object(o, args) for o in args.obj]

    print(json.dumps({"selection": results}, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"selection": results}, f, indent=2)
        sys.stderr.write(f"[select] wrote {args.out_json}\n")

    # non-zero exit if any object failed to select (so the chain can react)
    if any("error" in r for r in results):
        sys.exit(3)


if __name__ == "__main__":
    main()
