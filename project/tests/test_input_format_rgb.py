"""
Input-Format-Contract — RGB-Pipelines (KIP-POSE Multi-Pipeline-Plattform)
========================================================================

Standalone-Test, der das Input-Format festnagelt, das unsere Plattform für die
**RGB-Kombis** erwartet (GDRNPP / GigaPose-2D — alles ohne Tiefe). Wenn euer
Datensample `validate_rgb_frame(...)` ohne Fehler besteht, trefft ihr das Format,
das unser `/segment`- und `/predict`-Endpoint konsumiert.

Ein RGB-Frame = 1 Farbbild + Kamera-Intrinsik K.

ERWARTETES FORMAT (eingefroren, CONTRACT.md §2/§3/§5):
  rgb : np.ndarray, shape (H, W, 3), dtype uint8, Werte 0..255,
        **RGB-Reihenfolge** (NICHT BGR — Achtung bei cv2.imread!). H, W > 0.
  K   : Kamera-Intrinsik, 3x3 row-major ODER flach [fx,0,cx, 0,fy,cy, 0,0,1].
        fx, fy > 0 ; 0 < cx < W ; 0 < cy < H ; K[8] == 1.

WIRE-FORMAT (HTTP-Body):
  rgb_b64 = base64( PNG-Encode des uint8-RGB-Arrays )
  K       = flache 9-Liste [fx,0,cx, 0,fy,cy, 0,0,1]

Lauf:  pip install pytest numpy pillow  &&  pytest test_input_format_rgb.py
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Validator — ruft das andere Team auf EUREN Daten auf (wirft ValueError mit
# klarer Meldung, wenn das Format nicht stimmt).
# ---------------------------------------------------------------------------
def validate_rgb_frame(rgb, K) -> None:
    # --- RGB-Bild ---
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"rgb muss ein np.ndarray sein, ist {type(rgb).__name__}")
    if rgb.dtype != np.uint8:
        raise ValueError(
            f"rgb-dtype muss uint8 sein (Werte 0..255), ist {rgb.dtype} "
            f"— kein float/normalisiertes Bild übergeben"
        )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"rgb muss (H, W, 3) sein = 3-Kanal RGB, ist {rgb.shape} "
            f"— kein Graustufen (H,W) und kein RGBA/4-Kanal"
        )
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"rgb-Auflösung ungültig: {w}x{h}")

    # --- Kamera-Intrinsik K ---
    K = np.asarray(K, dtype=float).reshape(-1)
    if K.size != 9:
        raise ValueError(
            f"K muss 9 Werte haben (3x3 row-major, flach [fx,0,cx, 0,fy,cy, 0,0,1]), "
            f"hat {K.size}"
        )
    fx, _, cx, _, fy, cy, _, _, one = K
    if fx <= 0 or fy <= 0:
        raise ValueError(
            f"K: fx, fy müssen > 0 sein (fx={fx}, fy={fy}) — "
            f"Reihenfolge ist [fx,0,cx, 0,fy,cy, 0,0,1], nicht vertauscht"
        )
    if not (0 < cx < w):
        raise ValueError(f"K: cx ({cx}) muss in (0, W={w}) liegen — Prinzipalpunkt im Bild")
    if not (0 < cy < h):
        raise ValueError(f"K: cy ({cy}) muss in (0, H={h}) liegen — Prinzipalpunkt im Bild")
    if abs(one - 1.0) > 1e-6:
        raise ValueError(f"K[8] (unten rechts) muss 1.0 sein (homogen), ist {one}")


# ---------------------------------------------------------------------------
# Wire-Encode/Decode — so geht euer Frame exakt in die HTTP-API.
# ---------------------------------------------------------------------------
def encode_rgb_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_rgb_b64(rgb_b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(rgb_b64))).convert("RGB")
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Beispiel-Frame (genau so soll euer Sample aussehen)
# ---------------------------------------------------------------------------
def make_valid_sample(h=720, w=1280):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    K = [900.0, 0.0, w / 2.0, 0.0, 900.0, h / 2.0, 0.0, 0.0, 1.0]
    return rgb, K


# ---------------------------------------------------------------------------
# Tests: korrektes Sample besteht, typische Fehler fallen mit klarer Meldung.
# ---------------------------------------------------------------------------
def test_valid_rgb_frame_passes():
    rgb, K = make_valid_sample()
    validate_rgb_frame(rgb, K)              # darf nicht werfen
    validate_rgb_frame(rgb, np.asarray(K).reshape(3, 3))  # 3x3 auch ok


def test_wire_roundtrip_is_lossless():
    rgb, _ = make_valid_sample(64, 96)
    rgb[10:20, 10:20] = [255, 0, 0]
    back = decode_rgb_b64(encode_rgb_b64(rgb))
    assert back.shape == rgb.shape and back.dtype == np.uint8
    assert np.array_equal(back, rgb)        # PNG ist verlustfrei


def test_float_image_rejected():
    rgb, K = make_valid_sample()
    with pytest.raises(ValueError, match="uint8"):
        validate_rgb_frame(rgb.astype(np.float32) / 255.0, K)


def test_grayscale_rejected():
    rgb, K = make_valid_sample()
    with pytest.raises(ValueError, match=r"3-Kanal"):
        validate_rgb_frame(rgb[:, :, 0], K)


def test_rgba_4channel_rejected():
    rgb, K = make_valid_sample()
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    with pytest.raises(ValueError, match=r"3-Kanal"):
        validate_rgb_frame(rgba, K)


def test_wrong_K_length_rejected():
    rgb, _ = make_valid_sample()
    with pytest.raises(ValueError, match="9 Werte"):
        validate_rgb_frame(rgb, [900, 0, 640, 0, 900, 360])  # nur 6


def test_principal_point_out_of_image_rejected():
    rgb, _ = make_valid_sample(720, 1280)
    with pytest.raises(ValueError, match="cx"):
        validate_rgb_frame(rgb, [900, 0, 5000, 0, 900, 360, 0, 0, 1])  # cx > W


def test_negative_focal_rejected():
    rgb, _ = make_valid_sample()
    with pytest.raises(ValueError, match=r"fx, fy"):
        validate_rgb_frame(rgb, [-900, 0, 640, 0, 900, 360, 0, 0, 1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
