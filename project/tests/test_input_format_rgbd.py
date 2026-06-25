"""
Input-Format-Contract — RGBD-Pipelines (KIP-POSE Multi-Pipeline-Plattform)
=========================================================================

Standalone-Test, der das Input-Format festnagelt, das unsere Plattform für die
**RGBD-Kombis** erwartet (FoundationPose / GigaPose-3D — Tiefe ist PFLICHT).
Wenn euer Datensample `validate_rgbd_frame(...)` ohne Fehler besteht, trefft ihr
das Format, das unser `/pose`- und `/predict`-Endpoint konsumiert.

Ein RGBD-Frame = 1 Farbbild + 1 Tiefenbild + Kamera-Intrinsik K.

ERWARTETES FORMAT (eingefroren, CONTRACT.md §3/§5):
  rgb   : np.ndarray (H, W, 3), dtype uint8, 0..255, **RGB-Reihenfolge** (NICHT BGR).
  depth : np.ndarray (H, W),    dtype **uint16**, Einheit **MILLIMETER**,
          EXAKT dieselbe Auflösung wie rgb. 0 = ungültig/kein-Wert.
          (Der Service teilt intern /1000 -> Meter.)
  K     : 3x3 row-major ODER flach [fx,0,cx, 0,fy,cy, 0,0,1]; fx,fy>0; cx<W; cy<H.

WIRE-FORMAT (HTTP-Body):
  rgb_b64   = base64( PNG-Encode des uint8-RGB-Arrays )
  depth_b64 = base64( PNG-Encode des uint16-Tiefen-Arrays, Millimeter )
  K         = flache 9-Liste [fx,0,cx, 0,fy,cy, 0,0,1]

WICHTIG zu Tiefe:
  * Einheit am API-Rand ist MILLIMETER als uint16 (z.B. 1000 == 1.0 m).
  * Werte in Metern (z.B. 0.3..2.0) sind FALSCH -> *1000 rechnen.
  * Kommt die Tiefe aus einem BOP-Datensatz mit `depth_scale` (z.B. 0.1),
    erst `mm = png * depth_scale` anwenden, dann als uint16-mm liefern.

Lauf:  pip install pytest numpy pillow  &&  pytest test_input_format_rgbd.py
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Validator — ruft das andere Team auf EUREN Daten auf.
# ---------------------------------------------------------------------------
def validate_rgbd_frame(rgb, depth, K) -> None:
    # --- RGB (wie RGB-only-Contract) ---
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("rgb muss np.ndarray uint8 (Werte 0..255) sein")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb muss (H,W,3) = 3-Kanal RGB sein, ist {getattr(rgb,'shape',None)}")
    h, w = rgb.shape[:2]

    # --- Tiefe ---
    if not isinstance(depth, np.ndarray):
        raise ValueError(f"depth muss ein np.ndarray sein, ist {type(depth).__name__}")
    if depth.dtype != np.uint16:
        raise ValueError(
            f"depth-dtype muss uint16 sein (Millimeter), ist {depth.dtype} "
            f"— kein float, kein uint8"
        )
    if depth.ndim != 2:
        raise ValueError(
            f"depth muss (H, W) single-channel sein, ist {depth.shape} "
            f"— kein 3-Kanal-Tiefenbild"
        )
    if depth.shape != (h, w):
        raise ValueError(
            f"depth-Auflösung {depth.shape[::-1]} != rgb-Auflösung {(w, h)} "
            f"— Tiefe muss EXAKT pixelgleich zum RGB sein (gleiche Kamera/Resolution)"
        )
    valid = depth[depth > 0]
    if valid.size:
        if valid.max() < 100:
            raise ValueError(
                f"depth sieht nach METERN aus (max={valid.max()}) — Einheit muss "
                f"MILLIMETER sein (uint16), z.B. 1000 = 1 m. *1000 rechnen."
            )
        if valid.max() > 60000:
            raise ValueError(
                f"depth-Maximum {valid.max()} mm (>60 m) unplausibel — falsche Einheit/Skala?"
            )

    # --- Kamera-Intrinsik K ---
    K = np.asarray(K, dtype=float).reshape(-1)
    if K.size != 9:
        raise ValueError(f"K muss 9 Werte haben ([fx,0,cx, 0,fy,cy, 0,0,1]), hat {K.size}")
    fx, _, cx, _, fy, cy, _, _, one = K
    if fx <= 0 or fy <= 0:
        raise ValueError(f"K: fx, fy müssen > 0 sein (fx={fx}, fy={fy})")
    if not (0 < cx < w) or not (0 < cy < h):
        raise ValueError(f"K: Prinzipalpunkt (cx={cx}, cy={cy}) muss im Bild ({w}x{h}) liegen")
    if abs(one - 1.0) > 1e-6:
        raise ValueError(f"K[8] muss 1.0 sein, ist {one}")


# ---------------------------------------------------------------------------
# Wire-Encode/Decode der Tiefe (uint16-PNG, Millimeter).
# ---------------------------------------------------------------------------
def encode_depth_b64(depth: np.ndarray) -> str:
    if depth.dtype != np.uint16:
        raise ValueError("depth muss uint16 (Millimeter) sein vor dem Encode")
    buf = io.BytesIO()
    Image.fromarray(depth).save(buf, format="PNG")  # uint16 -> 16-bit PNG (I;16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_depth_b64(depth_b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(depth_b64)))
    return np.asarray(img, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Beispiel-Frame (genau so soll euer Sample aussehen)
# ---------------------------------------------------------------------------
def make_valid_sample(h=720, w=1280):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 1200, dtype=np.uint16)  # 1.2 m in Millimeter
    K = [900.0, 0.0, w / 2.0, 0.0, 900.0, h / 2.0, 0.0, 0.0, 1.0]
    return rgb, depth, K


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_valid_rgbd_frame_passes():
    rgb, depth, K = make_valid_sample()
    validate_rgbd_frame(rgb, depth, K)


def test_depth_wire_roundtrip_lossless():
    _, depth, _ = make_valid_sample(64, 96)
    depth[10:20, 10:20] = 800
    back = decode_depth_b64(encode_depth_b64(depth))
    assert back.dtype == np.uint16 and back.shape == depth.shape
    assert np.array_equal(back, depth)


def test_depth_float_rejected():
    rgb, depth, K = make_valid_sample()
    with pytest.raises(ValueError, match="uint16"):
        validate_rgbd_frame(rgb, depth.astype(np.float32), K)


def test_depth_uint8_rejected():
    rgb, depth, K = make_valid_sample()
    with pytest.raises(ValueError, match="uint16"):
        validate_rgbd_frame(rgb, depth.astype(np.uint8), K)


def test_depth_in_meters_rejected():
    rgb, depth, K = make_valid_sample()
    meters = np.full(depth.shape, 1, dtype=np.uint16)  # 1..2 statt 1000..2000
    with pytest.raises(ValueError, match="METERN"):
        validate_rgbd_frame(rgb, meters, K)


def test_depth_resolution_mismatch_rejected():
    rgb, _, K = make_valid_sample(720, 1280)
    wrong = np.full((360, 640), 1200, dtype=np.uint16)  # halbe Auflösung
    with pytest.raises(ValueError, match="EXAKT pixelgleich"):
        validate_rgbd_frame(rgb, wrong, K)


def test_depth_3channel_rejected():
    rgb, depth, K = make_valid_sample()
    d3 = np.dstack([depth, depth, depth])
    with pytest.raises(ValueError, match="single-channel"):
        validate_rgbd_frame(rgb, d3, K)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
