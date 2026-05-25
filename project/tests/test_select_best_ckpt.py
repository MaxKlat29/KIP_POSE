#!/usr/bin/env python3
"""Tests for box_src/select_best_ckpt.py — the best-by-val checkpoint selector
(T-068, overfitting guard).

Covers the PURE logic that runs OFF the GPU: checkpoint enumeration + sort order
(model_final last), CSV discovery (eval-only `_test` suffix), AR extraction from an
eval_bop report (dict + list shapes, None-safe), the best-pick (highest AR,
earlier-epoch tie-break), and the model_best.pth materialisation (symlink/copy).

The GPU inference (main_gdrn --eval-only) and eval_bop scoring are shelled out and
are exercised only on the box; they are NOT unit-tested here.
"""
import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEL = os.path.normpath(os.path.join(_HERE, "..", "..", "box_src", "select_best_ckpt.py"))


def _load():
    spec = importlib.util.spec_from_file_location("select_best_ckpt", _SEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEL = _load()


# ── epoch_from_ckpt + list_checkpoints sort order ─────────────────────────────

def test_epoch_from_ckpt_orders_final_last():
    assert SEL.epoch_from_ckpt("/x/model_0000704.pth") == 704
    assert SEL.epoch_from_ckpt("/x/model_0011679.pth") == 11679
    assert SEL.epoch_from_ckpt("/x/model_final.pth") == float("inf")


def test_list_checkpoints_sorted_final_last_and_excludes_best(tmp_path):
    d = str(tmp_path)
    for n in ("model_0011679.pth", "model_0000704.pth", "model_final.pth",
              "model_0005000.pth"):
        open(os.path.join(d, n), "w").close()
    # a previously-written selection that must be ignored
    open(os.path.join(d, "model_best.pth"), "w").close()
    got = [os.path.basename(p) for p in SEL.list_checkpoints(d)]
    assert got == ["model_0000704.pth", "model_0005000.pth",
                   "model_0011679.pth", "model_final.pth"]
    assert "model_best.pth" not in got


def test_list_checkpoints_empty(tmp_path):
    assert SEL.list_checkpoints(str(tmp_path)) == []


# ── find_pred_csv: eval-only `_test` suffix ───────────────────────────────────

def test_find_pred_csv_matches_eval_only_suffix(tmp_path):
    d = str(tmp_path)
    name = "zahnrad-test-iter0_pose_isaac-val.csv"
    open(os.path.join(d, name), "w").close()
    assert os.path.basename(SEL.find_pred_csv(d)) == name


def test_find_pred_csv_matches_train_naming(tmp_path):
    d = str(tmp_path)
    name = "zahnrad-iter0_pose_isaac-val.csv"
    open(os.path.join(d, name), "w").close()
    assert os.path.basename(SEL.find_pred_csv(d)) == name


def test_find_pred_csv_none_when_absent(tmp_path):
    assert SEL.find_pred_csv(str(tmp_path)) is None


# ── ar_for_object / mean_ar (dict + list shapes, None-safe) ───────────────────

def _report(per_object):
    return {"mode": "eval", "results": {"per_object": per_object, "overall": {}}}


def test_ar_for_object_dict_shape():
    rep = _report({"6": {"name": "Zahnrad", "AR": 0.36}})
    assert SEL.ar_for_object(rep, 6) == 0.36


def test_ar_for_object_list_shape():
    rep = _report([{"obj_id": 6, "name": "Zahnrad", "AR": 0.42}])
    assert SEL.ar_for_object(rep, 6) == 0.42


def test_ar_for_object_missing_is_none():
    rep = _report({"1": {"AR": 0.5}})
    assert SEL.ar_for_object(rep, 6) is None


def test_mean_ar_none_safe():
    rep = _report({"1": {"AR": 0.6}, "2": {"AR": None}})
    assert SEL.mean_ar(rep, [1, 2]) == pytest.approx(0.6)


def test_mean_ar_all_none_returns_none():
    rep = _report({"1": {"AR": None}})
    assert SEL.mean_ar(rep, [1]) is None


# ── pick_best: highest AR, earlier-epoch tie-break, the OVERFITTING case ───────

def test_pick_best_highest_ar():
    scored = [
        {"ckpt": "/x/model_0001000.pth", "ar": 0.30},
        {"ckpt": "/x/model_0005000.pth", "ar": 0.58},   # winner
        {"ckpt": "/x/model_final.pth", "ar": 0.41},
    ]
    best = SEL.pick_best(scored)
    assert os.path.basename(best["ckpt"]) == "model_0005000.pth"
    assert best["ar"] == 0.58


def test_pick_best_prefers_earlier_epoch_on_tie():
    # equal val-AR -> the earlier (less-overfit) checkpoint wins
    scored = [
        {"ckpt": "/x/model_0003000.pth", "ar": 0.50},   # earlier -> winner
        {"ckpt": "/x/model_final.pth", "ar": 0.50},
    ]
    best = SEL.pick_best(scored)
    assert os.path.basename(best["ckpt"]) == "model_0003000.pth"


def test_pick_best_final_can_lose_to_earlier_epoch():
    # the whole point: model_final is NOT automatically best
    scored = [
        {"ckpt": "/x/model_0009000.pth", "ar": 0.61},   # earlier epoch, higher AR
        {"ckpt": "/x/model_final.pth", "ar": 0.45},     # last epoch, overfit
    ]
    best = SEL.pick_best(scored)
    assert os.path.basename(best["ckpt"]) != "model_final.pth"


def test_pick_best_skips_none_ar():
    scored = [
        {"ckpt": "/x/model_0001000.pth", "ar": None},
        {"ckpt": "/x/model_final.pth", "ar": 0.4},
    ]
    best = SEL.pick_best(scored)
    assert os.path.basename(best["ckpt"]) == "model_final.pth"


def test_pick_best_all_none_returns_none():
    assert SEL.pick_best([{"ckpt": "/x/model_final.pth", "ar": None}]) is None


# ── apply_best: model_best.pth materialisation ────────────────────────────────

def test_apply_best_symlink(tmp_path):
    d = str(tmp_path)
    src = os.path.join(d, "model_0005000.pth")
    open(src, "w").close()
    dst = SEL.apply_best(d, src, copy=False)
    assert os.path.basename(dst) == "model_best.pth"
    assert os.path.islink(dst)
    # relative target so the output dir stays relocatable
    assert os.readlink(dst) == "model_0005000.pth"


def test_apply_best_copy(tmp_path):
    d = str(tmp_path)
    src = os.path.join(d, "model_0005000.pth")
    with open(src, "w") as f:
        f.write("weights")
    dst = SEL.apply_best(d, src, copy=True)
    assert not os.path.islink(dst)
    assert open(dst).read() == "weights"


def test_apply_best_overwrites_existing(tmp_path):
    d = str(tmp_path)
    a = os.path.join(d, "model_0001000.pth"); open(a, "w").close()
    b = os.path.join(d, "model_0009000.pth"); open(b, "w").close()
    SEL.apply_best(d, a, copy=False)
    dst = SEL.apply_best(d, b, copy=False)   # re-point
    assert os.readlink(dst) == "model_0009000.pth"


# ── sniff_output_res: shape -> OUTPUT_RES mapping (pure arithmetic) ────────────

def test_sniff_output_res_no_torch_or_bad_path_returns_none(tmp_path):
    # a non-checkpoint file -> torch.load fails -> None (no crash)
    p = os.path.join(str(tmp_path), "model_0001.pth")
    open(p, "w").close()
    assert SEL.sniff_output_res(p) is None


def test_sniff_output_res_arithmetic_from_fake_ckpt(tmp_path):
    """The pnp_net.fc1.weight in_features -> OUTPUT_RES map, on real torch shapes.

    Verified against a real checkpoint: OUTPUT_RES=64 -> 128*8*8 = 8192 ;
    OUTPUT_RES=80 -> 128*10*10 = 12800. We build a minimal state dict with ONLY
    the pnp_net.fc1.weight (and a decoy backbone fc1) and save it with torch so the
    real torch.load path is exercised. Skips cleanly if torch is unavailable.
    """
    torch = pytest.importorskip("torch")
    # OUTPUT_RES=64 case (Phase-1 256/64 checkpoint shape)
    sd64 = {
        "backbone.stages_0.blocks.0.mlp.fc1.weight": torch.zeros(512, 128),  # decoy
        "module.pose_net.pnp_net.fc1.weight": torch.zeros(1024, 8192),
    }
    p64 = os.path.join(str(tmp_path), "model_64.pth")
    torch.save(sd64, p64)
    assert SEL.sniff_output_res(p64) == 64

    # OUTPUT_RES=80 case (Phase-2 320/80 checkpoint shape), nested under "model"
    sd80 = {"model": {"pose_net.pnp_net.fc1.weight": torch.zeros(1024, 12800)}}
    p80 = os.path.join(str(tmp_path), "model_80.pth")
    torch.save(sd80, p80)
    assert SEL.sniff_output_res(p80) == 80
