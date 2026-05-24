#!/usr/bin/env python3
"""Tests for box_src/e2e_ab.py — the A/B-over-the-levers auto-best + decision-flag
engine of the hardened finish (T-048 / S-050).

Covers the orchestration logic that runs OFF the GPU: report loading, the mean-AR
selection metric (None-safe, untrained-object exclusion), the deterministic dry-run
synthesis, auto-best selection, both decision thresholds (Zahnrad<0.70 -> SO(3);
Anker<0.80 -> MegaPose), and the JSON/markdown emission.
"""
import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AB = os.path.normpath(os.path.join(_HERE, "..", "..", "box_src", "e2e_ab.py"))


def _load_ab():
    spec = importlib.util.spec_from_file_location("e2e_ab", _AB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _load_ab()


def _report(ar1, ar2, ar6, extra_zero=True):
    """A minimal eval_bop-shaped report.json dict with the three trainable
    objects (+ an untrained false-zero row that must NOT count)."""
    po = {
        "1": {"name": "Anker_Kurz", "AR": ar1},
        "2": {"name": "Anker_Lang", "AR": ar2},
        "6": {"name": "Zahnrad", "AR": ar6},
    }
    if extra_zero:
        po["3"] = {"name": "Buerstenhalter_2polig", "AR": 0.0, "n_matched": 0}
    return {"mode": "eval", "results": {"per_object": po, "overall": {"AR": 0.3}}}


def _write(tmp, name, doc):
    p = os.path.join(tmp, name)
    json.dump(doc, open(p, "w"))
    return p


# ── load_report + mean_ar ─────────────────────────────────────────────────────

def test_load_report_dict_shape(tmp_path):
    p = _write(str(tmp_path), "r.json", _report(0.6, 0.65, 0.4))
    rows = AB.load_report(p)
    assert set(rows) >= {1, 2, 6}
    assert rows[1]["AR"] == 0.6


def test_load_report_list_shape(tmp_path):
    doc = {"results": {"per_object": [
        {"obj_id": 1, "name": "Anker_Kurz", "AR": 0.6},
        {"obj_id": 6, "name": "Zahnrad", "AR": 0.4},
    ]}}
    p = _write(str(tmp_path), "r.json", doc)
    rows = AB.load_report(p)
    assert rows[1]["AR"] == 0.6 and rows[6]["AR"] == 0.4


def test_mean_ar_excludes_untrained_false_zeros(tmp_path):
    # the Buerstenhalter AR=0.0 row must NOT drag the mean down
    rows = AB.load_report(_write(str(tmp_path), "r.json", _report(0.6, 0.6, 0.6)))
    assert AB.mean_ar(rows) == pytest.approx(0.6)


def test_mean_ar_none_safe(tmp_path):
    rows = AB.load_report(_write(str(tmp_path), "r.json", _report(0.6, None, 0.4)))
    # only the two non-None trainable objects count
    assert AB.mean_ar(rows) == pytest.approx((0.6 + 0.4) / 2)


# ── synth (dry-run) ───────────────────────────────────────────────────────────

def test_synth_is_deterministic_and_monotone(tmp_path):
    planar = AB.load_report(_write(str(tmp_path), "r.json", _report(0.589, 0.606, 0.356)))
    s1 = AB.synth_configs(planar)
    s2 = AB.synth_configs(planar)
    assert set(s1) == {"m2", "tta", "m2_tta"}
    # deterministic
    assert s1["m2"][1]["AR"] == s2["m2"][1]["AR"]
    # every synth config beats planar on the mean
    base = AB.mean_ar(planar)
    for name in ("m2", "tta", "m2_tta"):
        assert AB.mean_ar(s1[name]) > base
    # m2_tta is the strongest synth config
    assert AB.mean_ar(s1["m2_tta"]) >= AB.mean_ar(s1["m2"])
    assert AB.mean_ar(s1["m2_tta"]) >= AB.mean_ar(s1["tta"])


def test_synth_caps_at_one(tmp_path):
    planar = AB.load_report(_write(str(tmp_path), "r.json", _report(0.99, 0.99, 0.99)))
    s = AB.synth_configs(planar)
    for name in s:
        for oid in (1, 2, 6):
            assert s[name][oid]["AR"] <= 1.0


# ── auto-best ─────────────────────────────────────────────────────────────────

def test_auto_best_picks_highest_mean():
    configs = {
        "planar": {1: {"AR": 0.5}, 2: {"AR": 0.5}, 6: {"AR": 0.3}},
        "m2": {1: {"AR": 0.7}, 2: {"AR": 0.7}, 6: {"AR": 0.5}},
        "tta": {1: {"AR": 0.55}, 2: {"AR": 0.55}, 6: {"AR": 0.35}},
    }
    scored = {n: AB.mean_ar(r) for n, r in configs.items()}
    best = max(scored, key=lambda n: scored[n])
    assert best == "m2"


# ── decision flags ────────────────────────────────────────────────────────────

def test_decision_zahnrad_below_070_triggers_so3():
    rows = {1: {"AR": 0.85}, 2: {"AR": 0.85}, 6: {"AR": 0.42}}
    dec = AB.build_decision("m2_tta", rows)
    assert dec["train_so3_phase3"] is True
    assert any(f["flag"] == "TRAIN_SO3_PHASE3" for f in dec["flags"])
    so3 = next(f for f in dec["flags"] if f["flag"] == "TRAIN_SO3_PHASE3")
    assert "train_so3cls_phase3.sh" in so3["action"]


def test_decision_zahnrad_at_or_above_070_no_so3():
    rows = {1: {"AR": 0.85}, 2: {"AR": 0.85}, 6: {"AR": 0.71}}
    dec = AB.build_decision("m2_tta", rows)
    assert dec["train_so3_phase3"] is False
    assert not any(f["flag"] == "TRAIN_SO3_PHASE3" for f in dec["flags"])


def test_decision_anker_below_080_triggers_megapose():
    rows = {1: {"AR": 0.72}, 2: {"AR": 0.90}, 6: {"AR": 0.80}}
    dec = AB.build_decision("m2", rows)
    assert dec["check_megapose_settings"] is True
    mp = next(f for f in dec["flags"] if f["flag"] == "CHECK_MEGAPOSE_SETTINGS")
    assert "Anker_Kurz" in mp["reason"] and "Anker_Lang" not in mp["reason"]


def test_decision_all_clear_no_flags():
    rows = {1: {"AR": 0.85}, 2: {"AR": 0.85}, 6: {"AR": 0.75}}
    dec = AB.build_decision("m2_tta", rows)
    assert dec["flags"] == []
    assert dec["train_so3_phase3"] is False
    assert dec["check_megapose_settings"] is False


def test_thresholds_are_the_spec_values():
    assert AB.ZAHNRAD_SO3_THR == 0.70
    assert AB.ANKER_MEGAPOSE_THR == 0.80
    assert AB.BASELINE_AR == {"Anker_Kurz": 0.59, "Anker_Lang": 0.61, "Zahnrad": 0.36}
    assert AB.PHASE1_AR == {"Anker_Kurz": 0.645, "Anker_Lang": 0.650, "Zahnrad": 0.36}


# ── end-to-end main (synth) ───────────────────────────────────────────────────

def test_main_synth_writes_json_and_md(tmp_path, monkeypatch):
    planar = _write(str(tmp_path), "planar.json", _report(0.589, 0.606, 0.356))
    out_json = os.path.join(str(tmp_path), "ab_result.json")
    out_md = os.path.join(str(tmp_path), "AB_LEVERS.md")
    argv = ["e2e_ab.py", "--report", f"planar={planar}", "--synth",
            "--mode", "dry-run", "--out-json", out_json, "--out-md", out_md]
    monkeypatch.setattr("sys.argv", argv)
    AB.main()
    assert os.path.isfile(out_json) and os.path.isfile(out_md)
    res = json.load(open(out_json))
    # all four configs present, best is the highest-mean synth config
    assert set(res["configs"]) == {"planar", "m2", "tta", "m2_tta"}
    assert res["best_config"] == "m2_tta"
    # both flags fire on the (still-low) synthesised numbers
    flags = {f["flag"] for f in res["decision"]["flags"]}
    assert "TRAIN_SO3_PHASE3" in flags
    assert "CHECK_MEGAPOSE_SETTINGS" in flags
    md = open(out_md).read()
    assert "+M2+TTA" in md and "✅" in md and "Decision flags" in md


def test_main_full_picks_best_of_real_reports(tmp_path, monkeypatch):
    # four real reports; m2 is the winner -> no --synth
    rp = {}
    rp["planar"] = _write(str(tmp_path), "p.json", _report(0.65, 0.65, 0.36))
    rp["m2"] = _write(str(tmp_path), "m2.json", _report(0.85, 0.86, 0.55))
    rp["tta"] = _write(str(tmp_path), "tta.json", _report(0.70, 0.70, 0.40))
    rp["m2_tta"] = _write(str(tmp_path), "mt.json", _report(0.84, 0.84, 0.52))
    out_json = os.path.join(str(tmp_path), "ab.json")
    out_md = os.path.join(str(tmp_path), "ab.md")
    argv = ["e2e_ab.py", "--mode", "full", "--out-json", out_json, "--out-md", out_md]
    for n, p in rp.items():
        argv += ["--report", f"{n}={p}"]
    monkeypatch.setattr("sys.argv", argv)
    AB.main()
    res = json.load(open(out_json))
    assert res["best_config"] == "m2"           # highest mean of the four real reports
    # Anker now > 0.80 -> no MegaPose flag; Zahnrad 0.55 < 0.70 -> SO(3) flag
    flags = {f["flag"] for f in res["decision"]["flags"]}
    assert "TRAIN_SO3_PHASE3" in flags
    assert "CHECK_MEGAPOSE_SETTINGS" not in flags
