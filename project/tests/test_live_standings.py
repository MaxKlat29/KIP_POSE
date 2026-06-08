#!/usr/bin/env python3
"""Tests fuer das Live-Scoreboard (T-153) — seed-major Round-Robin + inkrementelle
Standings + der /api/eval/job-Contract. Gegen Fixtures + Mocks (kein GPU/Gateway).

Prueft die T-153-Erweiterung des Batch-Eval-Runners:

  * Reihenfolge = seed-major Round-Robin: fuer JEDEN Seed werden ALLE Configs
    durchlaufen (nicht Config-fuer-Config). Nach Seed 1 hat jede Kombi genau 1 Szene.
  * Inkrementelle AR pro Config: AR IC-BIN wird nach jeder (config, szene) auf der
    AKKUMULIERTEN config-CSV neu berechnet; n_scenes waechst Seed fuer Seed.
  * build_standings: nach ar DESC sortiert, rank ab 1 gesetzt, ALLE ~12 Configs
    vertreten (auch noch-nicht-gestartete → ar=null, rank am Ende).
  * Standings-Eintrag-Schema EXAKT wie der Contract (rank,config_key,seg,pose,ar,
    ar_std,n_scenes,seg_ms,pose_ms,coverage,crash_rate,recommended,degraded,
    degraded_reason,class_ambiguity,is_pipeline_a).
  * /api/eval/job/<job> liefert {status,pct,phase,n_done,n_total,run_id,standings}
    mit der exakten standings[]-Form, live nach jeder Szene aktualisiert.
  * GDRNPP-degraded-Kombis (yolo-seg/sam3 → gdrnpp, NICHT is_pipeline_a) fahren ueber
    das Gateway (degraded=true) und werden gescort; nur die echte Pipeline A skippt.

Lauf:  .venv/bin/python -m pytest project/tests/test_live_standings.py -q
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from eval import batch_eval as be  # noqa: E402
from tests.test_batch_eval import _scene, _gateway_reply, _mock_eval_fn  # noqa: E402


# ── Contract: das exakte standings[]-Schema (T-153) ─────────────────────────────
_STANDINGS_KEYS = {
    "rank", "config_key", "seg", "pose", "modality", "ar", "ar_std", "n_scenes",
    "seg_ms", "pose_ms", "coverage", "crash_rate", "recommended", "degraded",
    "degraded_reason", "class_ambiguity", "is_pipeline_a",
}


def _gateway_or_skip_a(cfg, scene):
    """Mock-predict: nur die echte Pipeline A (is_pipeline_a) skippt das Gateway;
    die degraded gdrnpp-Kombis fahren mit (wie der echte Runner mit degraded=true)."""
    if cfg.get("is_pipeline_a"):
        raise be.PipelineANotOnGateway(cfg)
    return _gateway_reply(n_anker=2)


# ── 1) Reihenfolge: seed-major Round-Robin ──────────────────────────────────────
def test_run_batch_seed_major_round_robin(tmp_path):
    """Fuer jeden Seed werden ALLE Configs durchlaufen — Reihenfolge (scene, config)."""
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(3)]
    out = tmp_path / "batch_eval"
    calls = []

    def predict(cfg, scene):
        calls.append((scene["scene_id"], be.config_key(cfg)))
        return _gateway_or_skip_a(cfg, scene)

    be.run_batch(be.EVAL_CONFIGS, scenes, predict, _mock_eval_fn, out, run_id="rr")
    n_cfg = len(be.EVAL_CONFIGS)
    assert len(calls) == n_cfg * len(scenes)
    # Block i = Seed i: alle n_cfg Configs in EVAL_CONFIGS-Reihenfolge, derselbe seed.
    for si in range(len(scenes)):
        block = calls[si * n_cfg:(si + 1) * n_cfg]
        assert all(sid == scenes[si]["scene_id"] for sid, _ in block)
        assert [k for _, k in block] == [be.config_key(c) for c in be.EVAL_CONFIGS]
    # Anti-Config-major: die ersten n_cfg Calls sind NICHT alle dieselbe Config.
    assert len({k for _, k in calls[:n_cfg]}) == n_cfg


# ── 2) Inkrementelle Standings: nach N Szenen sortiert, gerankt, n_scenes waechst ─
def test_standings_streamed_incremental(tmp_path):
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(2)]
    out = tmp_path / "batch_eval"
    snapshots = []

    def cb(standings, n_done, n_total):
        # tiefe Kopie der relevanten Felder (der Runner mutiert die Liste nicht, aber
        # wir wollen einen stabilen Snapshot pro Callback).
        snapshots.append((n_done, n_total,
                          [(e["config_key"], e["rank"], e["ar"], e["n_scenes"])
                           for e in standings]))

    be.run_batch(be.EVAL_CONFIGS, scenes, _gateway_or_skip_a, _mock_eval_fn, out,
                 run_id="inc", standings_cb=cb)

    n_cfg = len(be.EVAL_CONFIGS)
    # Ein Callback pro gescorter (config, szene) + ein finaler.
    assert len(snapshots) == n_cfg * len(scenes) + 1
    # n_done waechst monoton; n_total konstant.
    n_dones = [s[0] for s in snapshots]
    assert n_dones == sorted(n_dones)
    assert all(s[1] == n_cfg * len(scenes) for s in snapshots)
    # Jeder Snapshot: ALLE Configs vertreten, rank lueckenlos 1..n_cfg.
    for _, _, entries in snapshots:
        assert len(entries) == n_cfg
        assert sorted(r for _, r, _, _ in entries) == list(range(1, n_cfg + 1))
    # ar DESC: nicht-None-ARs absteigend, None-ARs am Ende.
    for _, _, entries in snapshots:
        ars = [ar for _, _, ar, _ in entries]
        non_none = [a for a in ars if a is not None]
        assert non_none == sorted(non_none, reverse=True)
        # sobald ein None kam, kommt kein nicht-None mehr.
        seen_none = False
        for a in ars:
            if a is None:
                seen_none = True
            else:
                assert not seen_none
    # n_scenes einer gescorten Gateway-Config waechst von Seed 1 (1) auf Seed 2 (2).
    gw_key = "yolo_seg__foundationpose"

    def _nsc(snap_entries, key):
        return next(n for k, _, _, n in snap_entries if k == key)

    after_seed1 = snapshots[n_cfg - 1][2]      # letzter Callback in Seed-Block 1
    final = snapshots[-1][2]
    assert _nsc(after_seed1, gw_key) == 1
    assert _nsc(final, gw_key) == 2


# ── 3) build_standings: Sortierung, Rank, alle Configs (auch n_scenes=0) ─────────
def test_build_standings_all_configs_and_ranks():
    accs = [be._ConfigAcc(c) for c in be.EVAL_CONFIGS]
    # Nur zwei Configs "scoren" (verschiedene ARs); der Rest bleibt ar=null.
    by_key = {a.key: a for a in accs}
    by_key["yolo_seg__foundationpose"].ar = 0.80
    by_key["yolo_seg__foundationpose"].n_ok = 5
    by_key["sam3__gigapose_rgb"].ar = 0.65
    by_key["sam3__gigapose_rgb"].n_ok = 5

    st = be.build_standings(accs)
    assert len(st) == len(be.EVAL_CONFIGS)          # ALLE Configs
    assert [e["rank"] for e in st] == list(range(1, len(st) + 1))
    # Top-2 = die gescorten, ar DESC.
    assert st[0]["config_key"] == "yolo_seg__foundationpose" and st[0]["ar"] == 0.80
    assert st[1]["config_key"] == "sam3__gigapose_rgb" and st[1]["ar"] == 0.65
    # Der Rest: ar=null, rank am Ende, deterministisch nach config_key sortiert.
    rest = st[2:]
    assert all(e["ar"] is None for e in rest)
    assert [e["config_key"] for e in rest] == sorted(e["config_key"] for e in rest)


def test_standings_entry_schema_exact():
    acc = be._ConfigAcc(next(c for c in be.EVAL_CONFIGS
                             if be.config_key(c) == "yolo_seg__foundationpose"))
    acc.ar = 0.87
    acc.n_ok = 14
    acc.n_real = 14
    acc.n_with_inst = 14
    acc._seg_sum, acc._seg_cnt = 45.0 * 14, 14
    acc._pose_sum, acc._pose_cnt = 320.0 * 14, 14
    e = be.build_standings([acc])[0]
    # Exakt die Contract-Keys (nicht mehr, nicht weniger).
    assert set(e.keys()) == _STANDINGS_KEYS
    assert e["rank"] == 1
    assert e["config_key"] == "yolo_seg__foundationpose"
    assert e["seg"] == "yolo-seg" and e["pose"] == "foundationpose"  # Source-Ids!
    assert e["ar"] == 0.87 and e["ar_std"] == 0.0
    assert e["n_scenes"] == 14
    assert e["seg_ms"] == 45 and e["pose_ms"] == 320           # ganze ms (int)
    assert e["coverage"] == 1.0 and e["crash_rate"] == 0.0
    assert e["recommended"] is True
    assert e["degraded"] is False and e["degraded_reason"] is None
    assert e["class_ambiguity"] is False and e["is_pipeline_a"] is False


def test_standings_entry_pipeline_a_and_degraded_flags():
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}
    # Pipeline A (gdrnpp): is_pipeline_a, kein Gateway → ar/seg/pose null.
    a = be._ConfigAcc(by_key["gdrnpp"])
    ea = a.standings_entry()
    assert ea["is_pipeline_a"] is True and ea["ar"] is None
    assert ea["pose"] == "gdrnpp"
    # Degraded (yolo-seg→gdrnpp): degraded flag + reason, NICHT is_pipeline_a.
    d = be._ConfigAcc(by_key["yolo_seg__gdrnpp"])
    ed = d.standings_entry()
    assert ed["degraded"] is True and ed["degraded_reason"] == "aabb_from_mask"
    assert ed["is_pipeline_a"] is False and ed["pose"] == "gdrnpp"
    # sam3-Kombi: class_ambiguity.
    s = be._ConfigAcc(by_key["sam3__foundationpose"])
    assert s.standings_entry()["class_ambiguity"] is True


def test_modality_rgb_vs_rgbd_per_combo():
    """T-159: jede Kombi traegt modality = 'RGB'|'RGBD', abgeleitet aus needs_depth.
    Depth-Posen (FoundationPose, GigaPose-3D) = RGBD; GDRNPP + GigaPose-2D = RGB.
    Geprueft in BEIDEN Vertraegen — Live-Standings (/api/eval/job) UND der finalen
    Config-Zeile (/api/eval/result, aggregate_config)."""
    by_key = {be.config_key(c): c for c in be.EVAL_CONFIGS}

    def _entry_mod(key):
        return be._ConfigAcc(by_key[key]).standings_entry()["modality"]

    def _config_mod(key):
        return be.aggregate_config(by_key[key], per_scene=[])["modality"]

    # RGBD: alle Kombis mit Depth-Pose (foundationpose, gigapose_rgbd).
    rgbd_keys = [k for k, c in by_key.items() if c["needs_depth"]]
    # RGB: alle ohne Depth (gdrnpp inkl. Pipeline A, gigapose_rgb).
    rgb_keys = [k for k, c in by_key.items() if not c["needs_depth"]]
    assert rgbd_keys and rgb_keys                       # beide Klassen vertreten

    for k in rgbd_keys:
        assert _entry_mod(k) == "RGBD", k
        assert _config_mod(k) == "RGBD", k
        assert ("foundationpose" in k) or ("gigapose_rgbd" in k), k
    for k in rgb_keys:
        assert _entry_mod(k) == "RGB", k
        assert _config_mod(k) == "RGB", k
        assert ("gdrnpp" in k) or ("gigapose_rgb" in k), k

    # Konkrete Anker (T-159 AK): FoundationPose/GigaPose-3D=RGBD, GDRNPP/GigaPose-2D=RGB.
    assert _entry_mod("yolo_seg__foundationpose") == "RGBD"
    assert _entry_mod("yolo_seg__gigapose_rgbd") == "RGBD"
    assert _entry_mod("gdrnpp") == "RGB"                # Pipeline A
    assert _entry_mod("yolo_seg__gigapose_rgb") == "RGB"


def test_eval_md_has_input_column():
    """T-159: render_markdown traegt die Input-Spalte (RGB/RGBD) — Header + Werte."""
    results = {
        "run_id": "md-mod", "date": "2026-06-08T00:00:00Z", "duration_s": 1.0,
        "n_configs": 2, "n_scenes": 1,
        "configs": [
            {"seg": "yolo-seg", "pose": "GDRNPP", "modality": "RGB",
             "ar_mean": 0.1, "ar_std": 0.0, "seg_ms": 40, "pose_ms": 100,
             "coverage": 1.0, "crash_rate": 0.0, "note": "x", "is_pipeline_a": False},
            {"seg": "yolo-seg", "pose": "FoundationPose", "modality": "RGBD",
             "ar_mean": 0.2, "ar_std": 0.0, "seg_ms": 40, "pose_ms": 300,
             "coverage": 1.0, "crash_rate": 0.0, "note": "y", "is_pipeline_a": False},
        ],
    }
    md = be.render_markdown(results)
    assert "| Input |" in md
    assert "| RGB |" in md and "| RGBD |" in md


# ── 4) Inkrementelle AR-Recompute auf akkumulierter CSV ─────────────────────────
def test_ar_recomputed_on_accumulated_csv(tmp_path):
    """eval_fn wird mit der WACHSENDEN config-CSV gerufen — der AR-Recompute sieht
    nach Seed 2 mehr Zeilen als nach Seed 1 (akkumuliert, nicht per-Szene)."""
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(3)]
    out = tmp_path / "batch_eval"
    seen_rowcounts = {}          # config_key -> Liste der CSV-Zeilenzahlen pro eval-Call

    def counting_eval(csv_path, scene_dir, out_dir):
        key = pathlib.Path(csv_path).stem
        n = max(0, len(pathlib.Path(csv_path).read_text().strip().splitlines()) - 1)
        seen_rowcounts.setdefault(key, []).append(n)
        return _mock_eval_fn(csv_path, scene_dir, out_dir)

    be.run_batch(be.EVAL_CONFIGS, scenes, _gateway_or_skip_a, counting_eval, out,
                 run_id="acc")
    # Eine gescorte Gateway-Config: 2 Anker/Szene → CSV waechst 2,4,6 ueber 3 Seeds.
    rc = seen_rowcounts["yolo_seg__foundationpose"]
    assert rc == [2, 4, 6]            # akkumuliert, monoton wachsend
    # AR-Recompute pro Szene aufgerufen (3×), nicht nur einmal am Ende.
    assert len(rc) == 3


# ── 5) results.standings = finale Tabelle, kompatibel zu /api/eval/result ────────
def test_results_has_final_standings_and_configs(tmp_path):
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(2)]
    out = tmp_path / "batch_eval"
    results = be.run_batch(be.EVAL_CONFIGS, scenes, _gateway_or_skip_a, _mock_eval_fn,
                           out, run_id="fin")
    # configs bleibt die /api/eval/result-Form (aggregate_config).
    assert results["n_configs"] == len(be.EVAL_CONFIGS) and results["n_scenes"] == 2
    assert "configs" in results and len(results["configs"]) == len(be.EVAL_CONFIGS)
    # standings = finale Live-Tabelle, alle Configs, ranked.
    st = results["standings"]
    assert len(st) == len(be.EVAL_CONFIGS)
    assert [e["rank"] for e in st] == list(range(1, len(st) + 1))
    # Persistiert mit Standings.
    loaded = be.load_run(out, "fin")
    assert "standings" in loaded and len(loaded["standings"]) == len(be.EVAL_CONFIGS)
    # Mit dem degraded-Gateway-Routing werden 11 Configs gescort (alle ausser der
    # echten Pipeline A); nur die ist is_pipeline_a → ar null.
    a_rows = [e for e in st if e["is_pipeline_a"]]
    assert len(a_rows) == 1 and a_rows[0]["ar"] is None
    scored = [e for e in st if e["ar"] is not None]
    assert len(scored) == len(be.EVAL_CONFIGS) - 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
