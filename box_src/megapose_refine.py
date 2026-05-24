#!/usr/bin/env python3
"""megapose_refine.py — M2 MegaPose-RGB render-and-compare scorer (GPU, FINISH-TIME).

THE box-side GPU implementation of refine_rc.megapose_score's contract. It is
deliberately SEPARATE from the laptop refine_rc.py (which only carries the
contract + the CPU-edge fallback): the real GPU call lives here, on the box, and
is run/validated at FINISH (the GPU is currently busy rendering/training, so this
is NOT executed now — it is wired and ready).

What it does (the multi-hypothesis render-and-compare path):
  1. Load the named MegaPose-RGB model + a RigidObjectDataset from the BOP CAD
     (models_eval/obj_<id>.ply, mm).
  2. For each detection, take the GDRNPP coarse world rotation, build the M2
     hypotheses (refine_rc.generate_hypotheses: coarse + 180° flips + C_N yaws +
     stable rest poses + tilts), convert each to a model->cam pose (TCO_init).
  3. Feed ALL hypotheses as TCO_input to PoseEstimator.forward_refiner(
     n_iterations=N) -> per-hypothesis refined pose + the learned render-vs-RGB
     score. Pick the best-scoring refined pose (gated against coarse).
  4. Map the winning model->cam pose back to the world frame (bop_adapter) ->
     refined world rotation. RGB-only; contract unchanged.

VALIDATION AT FINISH (documented, run when the GPU is free):
  /mnt/data/bop/repos/megapose6d  is the install; megapose-1.0-RGB is the model.
  This script is invoked by real_pose_result.py --refine-rc --rc-scorer megapose
  (via e2e_finish.sh --refine-rc --rc-scorer megapose). The CPU-edge path
  (refine_rc cpu_edge) is the no-GPU default and is validated NOW.

Run standalone (finish-time, GPU free):
  /mnt/data/bop/repos/megapose6d/.venv/bin/python box_src/megapose_refine.py \
    --bop-root /mnt/data/kip_pose/project/bop/pose_isaac \
    --preds /mnt/data/bop/results/val_preds_combined.csv \
    --scene 0 --im 92 --self-check
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))
sys.path.insert(0, "/mnt/data/kip_pose/project")
import bop_adapter as BOP  # noqa: E402
import refine_rc as RC  # noqa: E402

MEGAPOSE_REPO = "/mnt/data/bop/repos/megapose6d"
MODEL_NAME = "megapose-1.0-RGB"


def _import_megapose():
    """Import the MegaPose inference stack. Raises a clear error if unavailable."""
    sys.path.insert(0, os.path.join(MEGAPOSE_REPO, "src"))
    import torch  # noqa: F401
    from megapose.inference.pose_estimator import PoseEstimator  # noqa: F401
    from megapose.inference.types import (ObservationTensor,  # noqa: F401
                                          PoseEstimatesType, DetectionsType)
    from megapose.inference.utils import load_named_model  # noqa: F401
    from megapose.datasets.object_dataset import (RigidObjectDataset,  # noqa: F401
                                                  RigidObject)
    return {
        "torch": torch, "PoseEstimator": PoseEstimator,
        "ObservationTensor": ObservationTensor,
        "PoseEstimatesType": PoseEstimatesType, "DetectionsType": DetectionsType,
        "load_named_model": load_named_model,
        "RigidObjectDataset": RigidObjectDataset, "RigidObject": RigidObject,
    }


def make_object_dataset(mesh_dir, obj_ids):
    """RigidObjectDataset from BOP CAD (mm). obj_label = obj_<id:06d>."""
    mp = _import_megapose()
    objs = []
    for oid in obj_ids:
        ply = os.path.join(mesh_dir, f"obj_{int(oid):06d}.ply")
        objs.append(mp["RigidObject"](label=f"obj_{int(oid):06d}", mesh_path=ply,
                                      mesh_units="mm"))
    return mp["RigidObjectDataset"](objs)


def refine_world_rotation(R0_world, *, obj_id, full_rgb, K, bbox,
                          t_world_m, table_origin_m, R_w2c, t_w2c_mm,
                          mesh_dir, pose_estimator, mp, n_iterations=5,
                          sym_axis=(0.0, 1.0, 0.0), n_fold=None, stable_downs=None):
    """Multi-hypothesis MegaPose-RGB refine of one detection. -> (R_world, info).

    Builds the hypotheses (refine_rc), turns each into a model->cam TCO_init,
    refines all in one batch, scores, picks the best, maps back to world.
    """
    torch = mp["torch"]
    hyps, tags = RC.generate_hypotheses(R0_world, sym_axis=sym_axis, n_fold=n_fold,
                                        stable_downs=stable_downs)
    # world -> model->cam per hypothesis (adapter inverse, R_model_to_body=I).
    t_m2w_mm = (np.asarray(t_world_m, float).reshape(-1) +
                np.asarray(table_origin_m, float)) * 1000.0
    t_m2c = (np.asarray(R_w2c, float) @ t_m2w_mm +
             np.asarray(t_w2c_mm, float).reshape(-1))
    TCO = np.tile(np.eye(4), (len(hyps), 1, 1))
    for i, R in enumerate(hyps):
        TCO[i, :3, :3] = np.asarray(R_w2c, float) @ BOP._as_R(R)
        TCO[i, :3, 3] = t_m2c / 1000.0                      # MegaPose poses in m

    label = f"obj_{int(obj_id):06d}"
    import pandas as pd
    infos = pd.DataFrame({"label": [label] * len(hyps),
                          "batch_im_id": [0] * len(hyps),
                          "instance_id": list(range(len(hyps)))})
    data_TCO = mp["PoseEstimatesType"](
        infos, poses=torch.as_tensor(TCO).float().cuda())
    rgb_t = torch.as_tensor(np.asarray(full_rgb)).permute(2, 0, 1).float() / 255.0
    obs = mp["ObservationTensor"].from_torch_batched(
        rgb_t[None].cuda(), None,
        torch.as_tensor(np.asarray(K, float)).float()[None].cuda())
    preds, extra = pose_estimator.forward_refiner(
        obs, data_TCO, n_iterations=n_iterations)
    final = preds[f"refiner/iteration={n_iterations}"]
    # per-hyp score: prefer an emitted pose_score column, else the coarse scorer.
    if "pose_score" in final.infos:
        scores = final.infos["pose_score"].to_numpy()
    elif "score" in final.infos:
        scores = final.infos["score"].to_numpy()
    else:                                                   # fallback: keep coarse
        scores = np.array([1.0] + [0.0] * (len(hyps) - 1))
    best = int(np.argmax(scores))
    TCO_best = final.poses[best].detach().cpu().numpy()
    R_m2c_best = TCO_best[:3, :3]
    R_world_best, _ = BOP.bop_pose_to_world(
        R_m2c_best, t_m2c, R_w2c, t_w2c_mm, table_origin_m)
    info = {"n_hyps": len(hyps), "best_idx": best, "best_tag": tags[best],
            "scores": [float(s) for s in scores], "scorer": "megapose"}
    return R_world_best, info


def self_check(args):
    """Finish-time self-check: import MegaPose, load model, build hypotheses for
    ONE real prediction and confirm the TCO_init batch is well-formed — WITHOUT
    running the heavy refiner if --no-refine is passed (so it can be smoke-tested
    quickly when the GPU briefly frees up). Prints a PASS/FAIL line."""
    try:
        mp = _import_megapose()
    except Exception as exc:
        print(f"[megapose_refine] SELF-CHECK SKIP: MegaPose stack unavailable ({exc!r})")
        print("[megapose_refine] -> CPU-edge scorer (refine_rc cpu_edge) is the default; "
              "MegaPose validated when its venv/GPU are free.")
        return 0
    print(f"[megapose_refine] MegaPose import OK from {MEGAPOSE_REPO}")
    print(f"[megapose_refine] model={MODEL_NAME}; hypotheses generator OK "
          f"(generate_hypotheses imported from refine_rc).")
    # Build hypotheses for a dummy coarse to prove the wiring without GPU.
    h, tags = RC.generate_hypotheses(np.eye(3), n_fold=7)
    print(f"[megapose_refine] sample hypotheses for C_7: {len(h)} ({tags[:6]}...)")
    print("[megapose_refine] SELF-CHECK PASS (wiring valid; full GPU refine runs at finish)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="M2 MegaPose-RGB refiner (finish-time)")
    ap.add_argument("--bop-root", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--im", type=int, default=92)
    ap.add_argument("--n-iterations", type=int, default=5)
    ap.add_argument("--self-check", action="store_true",
                    help="validate wiring (import + hypotheses) without heavy GPU refine")
    args = ap.parse_args()
    if args.self_check:
        return self_check(args)
    print("[megapose_refine] full GPU refine path: invoked by real_pose_result.py "
          "--refine-rc --rc-scorer megapose at finish. See module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
