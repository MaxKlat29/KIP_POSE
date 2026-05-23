#!/usr/bin/env python3
"""E2E-Test: ganze Inferenz-Kette mit MOCK-GDRNPP grün + schema-valide.

Erzeugt eine selbst-enthaltene Fixture (synthetisches RGB + bbox_2d-JSON im
SDG-Annotator-Format), läuft die volle Kette
  detections_for -> estimate_poses(MOCK) -> BOP-Adapter -> build_pose_result
und prüft: pose_result ist schema-valide (gegen pose_result.schema.json wenn
jsonschema da, sonst stdlib-Gate), eine Pose pro Detektion, korrekte obj_id->part-
Zuordnung, t_world auf/nahe der Tischebene.

Lauf:  python3 -m pytest project/tests/test_e2e_mock.py -q
   oder python3 project/tests/test_e2e_mock.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import e2e_infer as E  # noqa: E402


# ── Fixture: synthetisches RGB + SDG-bbox-JSON neben dem Bild ─────────────────
def _make_fixture(tmp: pathlib.Path):
    W, H = 1280, 720
    rng = np.random.default_rng(0)
    img = (rng.integers(40, 90, size=(H, W, 3))).astype(np.uint8)   # grauer BG
    img_path = tmp / "scene_0000.png"
    Image.fromarray(img).save(img_path)

    # 3 Detektionen: anker_kurz (cls0), anker_lang (cls1), zahnrad (cls5).
    boxes = [
        [0, 300, 200, 360, 480, 0.10],     # anker_kurz
        [1, 700, 150, 760, 470, 0.05],     # anker_lang
        [5, 540, 320, 640, 420, 0.00],     # zahnrad
    ]
    bbox_doc = {
        "data": boxes,
        "info": {"idToLabels": {"0": {"class": "anker_kurz"},
                                "1": {"class": "anker_lang"},
                                "5": {"class": "zahnrad"}}},
    }
    json.dump(bbox_doc, open(tmp / "bbox_2d_scene.json", "w"))
    return img_path


def test_e2e_mock_runs_green_and_schema_valid():
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        img_path = _make_fixture(tmp)
        out = tmp / "pose_result.json"
        cfg = E.GdrnppConfig(mock=True)
        doc = E.run(str(img_path), str(out), cfg=cfg, warn=lambda *_: None)

    # stdlib + jsonschema Gate (run() würde sonst raisen — hier nochmal explizit).
    assert E.check_pose_result(doc) == []
    assert E._check_with_jsonschema(doc) == []          # leer wenn jsonschema fehlt

    res = doc["results"]
    assert len(res) == 3, f"erwartet 3 Posen, bekam {len(res)}"
    parts = sorted(r["part"] for r in res)
    assert parts == ["Anker_Kurz", "Anker_Lang", "Zahnrad"]

    for r in res:
        assert len(r["R_world"]) == 9
        assert len(r["t_world"]) == 3
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["bbox_2d"][0] <= r["bbox_2d"][2]
        assert r["bbox_2d"][1] <= r["bbox_2d"][3]
        assert isinstance(r["upright"], bool)
        assert r["face"]
        # R_world muss eine gültige Rotation sein.
        R = np.array(r["R_world"]).reshape(3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        # Mock setzt die Teile auf z ~ 0 (Tischebene rel. table_origin).
        assert abs(r["t_world"][2]) < 0.05, f"z nicht auf Tisch: {r['t_world'][2]}"

    # meta-Felder gemäß Contract.
    assert doc["meta"]["units"] == "m"
    assert len(doc["meta"]["table_origin"]) == 3
    assert doc["meta"]["schema_version"].count(".") == 2


def test_call_gdrnpp_returns_bop_pose():
    """call_gdrnpp (MOCK) liefert (R_m2c 3x3, t_m2c 3,) in mm — Adapter-Vertrag."""
    K = E._default_K(1280, 720)
    crop = np.zeros((50, 50, 3), np.uint8)
    R_m2c, t_m2c = E.call_gdrnpp(crop, K, [300, 200, 360, 480], obj_id=1,
                                 cfg=E.GdrnppConfig(mock=True))
    assert np.asarray(R_m2c).shape == (3, 3)
    assert np.asarray(t_m2c).shape == (3,)
    # gültige Rotation.
    R = np.asarray(R_m2c)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_real_call_falls_back_to_mock_without_checkpoint():
    """Ohne Checkpoint fällt call_gdrnpp graceful auf MOCK zurück (keine Exception),
    auch wenn cfg.mock explizit False ist."""
    cfg = E.GdrnppConfig(mock=False)                    # erzwingt _gdrnpp_real-Pfad
    assert cfg.checkpoint is None
    K = E._default_K(1280, 720)
    crop = np.zeros((50, 50, 3), np.uint8)
    R_m2c, t_m2c = E.call_gdrnpp(crop, K, [0, 0, 100, 100], obj_id=6, cfg=cfg)
    assert np.asarray(R_m2c).shape == (3, 3)


def test_config_auto_mock_when_no_checkpoint():
    """GdrnppConfig: kein/fehlender Checkpoint -> mock=True automatisch."""
    assert E.GdrnppConfig().mock is True
    assert E.GdrnppConfig(checkpoint="/nope/missing.pth").mock is True
    # mit Mock-Override.
    assert E.GdrnppConfig(checkpoint="/nope/missing.pth", mock=True).mock is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} e2e tests green")
