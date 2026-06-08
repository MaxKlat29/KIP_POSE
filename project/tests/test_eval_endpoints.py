#!/usr/bin/env python3
"""Integrationstests fuer die kip_server-/api/eval/*-Endpoints (S-012 / T-138).

Gegen einen GEMOCKTEN Eval-Lauf (kein Gateway/GPU/best.pt). Prueft die HTTP-Naht
(FastAPI TestClient) gegen Lenas batch.js-Contract (Mia §14):

  * GET  /api/eval/runs        → {runs:[{run_id,date,duration_s,n_configs}]}
  * GET  /api/eval/result/<id> → {configs:[{seg,pose,ar_mean,ar_std,seg_ms,pose_ms,
                                  coverage,crash_rate,note,is_pipeline_a,...}]}, 404 sonst
  * POST /api/eval/run         → {job}; Training-Guard → 503 (override force)
  * GET  /api/eval/job/<job>   → {pct,phase,...} (gleiches Schema wie /api/sim/job)
  * Felder coverage/crash_rate ∈ 0..1; is_pipeline_a markiert die Referenzzeile.

Lauf:  .venv/bin/python -m pytest project/tests/test_eval_endpoints.py -q
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

# kip_server liest /mnt-Pfade beim Import → in Tests auf tmp umlenken (Box hat /mnt,
# CI nicht). Gleicher Guard wie S-013's test_kip_server_predict.py.
os.environ.setdefault("KIP_LIVE_ROOT", tempfile.mkdtemp(prefix="kip_live_eval_test_"))
os.environ.setdefault("KIP_BASE_PATH", "")

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

import kip_server as K  # noqa: E402
from eval import batch_eval as be  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Eval-Output in eine frische tmp-Senke umlenken (kein project/temp-Leak).
    monkeypatch.setattr(K, "EVAL_OUT", tmp_path / "batch_eval")
    K.EVAL_OUT.mkdir(parents=True, exist_ok=True)
    return TestClient(K.app)


def _persist_fixture_run(out_dir: pathlib.Path, run_id="run-fixture"):
    """Schreibt ein results.json mit der vollen 7-Config-Form (wie run_batch)."""
    configs = []
    for i, cfg in enumerate(be.EVAL_CONFIGS):
        configs.append({
            "seg": cfg["seg"], "pose": cfg["pose"],
            "ar_mean": None if cfg["is_pipeline_a"] else round(0.30 + 0.02 * i, 3),
            "ar_std": None if cfg["is_pipeline_a"] else 0.01,
            "seg_ms": 12.0, "pose_ms": 300.0 + 10 * i,
            "coverage": None if cfg["is_pipeline_a"] else 0.95,
            "crash_rate": None if cfg["is_pipeline_a"] else 0.05,
            "note": cfg["note"], "is_pipeline_a": cfg["is_pipeline_a"],
            "run_config_id": be.config_key(cfg),
        })
    results = {"run_id": run_id, "date": "2026-06-07T18:00:00Z",
               "duration_s": 1234.5, "n_configs": len(configs), "n_scenes": 20,
               "configs": configs}
    rd = out_dir / run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results.json").write_text(json.dumps(results))
    return results


# ── runs ────────────────────────────────────────────────────────────────────────
def test_runs_empty(client):
    r = client.get("/api/eval/runs")
    assert r.status_code == 200
    assert r.json() == {"runs": []}


def test_runs_lists_persisted(client):
    _persist_fixture_run(K.EVAL_OUT, "run-a")
    time.sleep(0.01)
    _persist_fixture_run(K.EVAL_OUT, "run-b")
    body = client.get("/api/eval/runs").json()
    assert "runs" in body
    ids = {x["run_id"] for x in body["runs"]}
    assert ids == {"run-a", "run-b"}
    for x in body["runs"]:
        # Genau Lenas runs-Form.
        assert set(x.keys()) >= {"run_id", "date", "duration_s", "n_configs"}
        assert x["n_configs"] == 12


# ── result ──────────────────────────────────────────────────────────────────────
def test_result_404(client):
    assert client.get("/api/eval/result/missing").status_code == 404


def test_result_schema(client):
    _persist_fixture_run(K.EVAL_OUT, "run-fixture")
    body = client.get("/api/eval/result/run-fixture").json()
    assert body["run_id"] == "run-fixture"
    cfgs = body["configs"]
    assert len(cfgs) == 12
    # Pflicht-Felder (Lena batch.js): seg,pose,ar_mean,ar_std,seg_ms,pose_ms,
    # coverage,crash_rate,note,is_pipeline_a.
    for c in cfgs:
        assert {"seg", "pose", "ar_mean", "ar_std", "seg_ms", "pose_ms",
                "coverage", "crash_rate", "note", "is_pipeline_a"} <= set(c.keys())
        if c["coverage"] is not None:
            assert 0.0 <= c["coverage"] <= 1.0       # FE formatiert auf %
        if c["crash_rate"] is not None:
            assert 0.0 <= c["crash_rate"] <= 1.0
    # Genau eine Pipeline-A-Referenzzeile.
    assert sum(1 for c in cfgs if c["is_pipeline_a"]) == 1


# ── run (async job) + job-status ────────────────────────────────────────────────
def test_run_training_guard_503(client, monkeypatch):
    monkeypatch.setattr(K, "_gpu_busy_with_training", lambda: True)
    r = client.post("/api/eval/run", data={"seeds": 2})
    assert r.status_code == 503
    # force=true override.
    r2 = client.post("/api/eval/run", data={"seeds": 2, "force": "true"})
    assert r2.status_code == 200 and "job" in r2.json()


def test_run_returns_job_and_job_status_schema(client, monkeypatch):
    monkeypatch.setattr(K, "_gpu_busy_with_training", lambda: False)
    r = client.post("/api/eval/run", data={"seeds": 1})
    assert r.status_code == 200
    job = r.json()["job"]
    # Job-Status = sim-job-Schema {pct, phase, ...}. (Lauf findet lokal keine Szenen
    # → endet bei pct=-1/no_scenes; das Schema ist trotzdem korrekt.)
    deadline = time.time() + 3.0
    st = {}
    while time.time() < deadline:
        st = client.get(f"/api/eval/job/{job}").json()
        if st.get("pct") in (100, -1):
            break
        time.sleep(0.05)
    assert "pct" in st and "phase" in st             # gleiches Schema wie sim-job


def test_job_unknown(client):
    assert client.get("/api/eval/job/nope").json() == {"error": "unknown job"}


# ── voller async-Lauf gegen Mock-Gateway + Mock-eval (end-to-end durch die HTTP-Naht) ─
def test_full_async_run_through_endpoints(client, tmp_path, monkeypatch):
    from tests.test_batch_eval import _scene, _gateway_reply, _mock_eval_fn

    monkeypatch.setattr(K, "_gpu_busy_with_training", lambda: False)
    # Szenen-Discovery auf eine Mini-Szene umlenken.
    scenes = [_scene(tmp_path / "scenes", scene_id=i) for i in range(2)]
    monkeypatch.setattr(be, "discover_scenes", lambda *a, **k: scenes)

    def _predict_factory(*a, **k):
        def _p(cfg, scene):
            if cfg["pose_source"] == "gdrnpp":
                raise be.PipelineANotOnGateway(cfg)
            return _gateway_reply(n_anker=2)
        return _p

    monkeypatch.setattr(be, "http_predict", _predict_factory)
    monkeypatch.setattr(be, "subprocess_eval", lambda *a, **k: _mock_eval_fn)

    job = client.post("/api/eval/run", data={"seeds": 2}).json()["job"]
    deadline = time.time() + 8.0
    st = {}
    while time.time() < deadline:
        st = client.get(f"/api/eval/job/{job}").json()
        if st.get("pct") in (100, -1):
            break
        time.sleep(0.05)
    assert st.get("pct") == 100, st
    run_id = st["run_id"]
    # Ergebnis ist sofort ueber /api/eval/result abrufbar (Lena's loadRuns-Flow).
    res = client.get(f"/api/eval/result/{run_id}").json()
    assert res["n_configs"] == 12 and res["n_scenes"] == 2
    # Gateway-Kombis (pose != gdrnpp) sind gescort; die 3 gdrnpp-Kombis (A + 2 degraded)
    # gehen ueber PipelineANotOnGateway → ar_mean None.
    gw = [c for c in res["configs"] if not c["run_config_id"].endswith("gdrnpp")
          and c["run_config_id"] != "gdrnpp"]
    assert len(gw) == 9 and all(c["ar_mean"] is not None for c in gw)
    # Und im runs-Dropdown.
    runs = client.get("/api/eval/runs").json()["runs"]
    assert any(x["run_id"] == run_id for x in runs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
