#!/usr/bin/env python3
"""T-188: Upload-Normalisierung aufs Pipeline-Format 1280x720.

Zivid-Vollaufloesung (2448x2048) crashte den GDRNPP-Loader (SizeMismatchError),
gemeldet als "Worker nicht erreichbar (Port 8078?)". _to_pipeline_format bringt
jeden Upload per 16:9-Center-Crop + Downscale auf 1280x720; 1280x720 bleibt
byte-identisch.

Lauf:  .venv/bin/python -m pytest project/tests/test_upload_normalize.py -q
"""
from __future__ import annotations

import io
import os
import pathlib
import sys
import tempfile

from PIL import Image

os.environ.setdefault("KIP_LIVE_ROOT", tempfile.mkdtemp(prefix="kip_live_test_"))
os.environ.setdefault("KIP_BASE_PATH", "")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kip_server import _to_pipeline_format  # noqa: E402


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_1280x720_bleibt_byte_identisch():
    data = _png(Image.new("RGB", (1280, 720), (10, 20, 30)))
    assert _to_pipeline_format(data) == data


def test_zivid_fullres_wird_1280x720_center_crop():
    # oben rot / Mitte gruen / unten blau: der 16:9-Crop einer 6:5-Aufnahme
    # muss oben+unten wegschneiden, die Mitte bleibt gruen.
    im = Image.new("RGB", (2448, 2048), (0, 255, 0))
    im.paste((255, 0, 0), (0, 0, 2448, 300))
    im.paste((0, 0, 255), (0, 1748, 2448, 2048))
    out = Image.open(io.BytesIO(_to_pipeline_format(_png(im))))
    assert out.size == (1280, 720)
    assert out.getpixel((640, 0)) != (255, 0, 0)      # roter Rand weggecroppt
    assert out.getpixel((640, 360))[1] > 200          # Mitte weiter gruen


def test_depth_16bit_nearest_wertetreu():
    im = Image.new("I;16", (2448, 2048), 1234)        # konstante mm-Tiefe
    out = Image.open(io.BytesIO(_to_pipeline_format(_png(im), is_depth=True)))
    assert out.size == (1280, 720)
    assert out.getpixel((640, 360)) == 1234           # NEAREST: Wert unveraendert
