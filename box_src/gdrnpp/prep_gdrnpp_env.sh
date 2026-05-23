#!/usr/bin/env bash
# =============================================================================
# prep_gdrnpp_env.sh — one-time GDRNPP venv + source prep on the box (T-068)
# -----------------------------------------------------------------------------
# Sam's gdrnpp install built detectron2 but left the GDRNPP training stack
# unrunnable on this box's Python 3.11 + torch 2.5.1. This script applies the
# exact dependency pins + source patches + the C++ EGL renderer build that make
# `from core.gdrn_modeling import main_gdrn` import cleanly. Idempotent.
#
#   bash box_src/gdrnpp/prep_gdrnpp_env.sh
#
# Verified working set (2026-05-23):
#   mmcv-full 1.7.2 (source build, `from mmcv import Config` = 1.x API)
#   pytorch-lightning 1.9.0 (LightningLite kept; wrappers renamed -> patched)
#   numpy 1.23.5 + scipy 1.10.1 (np.float / maximum_sctype aliases restored)
#   fairscale, pyassimp 4.1.4, setuptools<81 (pkg_resources for mmcv setup.py)
#   CppEGLRenderer built (vendored pybind11 2.3 -> 2.13.6, +<iostream> for gcc13)
# =============================================================================
set -uo pipefail

GDRN=/mnt/data/bop/repos/gdrnpp
PIP=/mnt/data/bop/gdrnpp-venv/bin/pip
PY=/mnt/data/bop/gdrnpp-venv/bin/python
SRC=/mnt/data/kip_pose/box_src/gdrnpp

echo "[prep] 1) setuptools<81 (restore pkg_resources for legacy setup.py builds)"
$PIP install "setuptools<81" wheel "pip<24.1" 2>&1 | tail -1

echo "[prep] 2) mmcv-full 1.7.2 (source build if not present)"
if ! $PY -c "from mmcv import Config" 2>/dev/null; then
    MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 \
        $PIP install mmcv-full==1.7.2 --no-build-isolation 2>&1 | tail -2
else
    echo "  mmcv already importable"
fi

echo "[prep] 3) pin PL 1.9.0 + numpy 1.23.5 + scipy 1.10.1 + fairscale + pyassimp"
$PIP install "pytorch-lightning==1.9.0" "numpy==1.23.5" "scipy==1.10.1" \
             fairscale "pyassimp==4.1.4" 2>&1 | tail -2

echo "[prep] 4) GDRNPP source patches (collections.abc, is_binary_file, egl iostream)"
$PY "$SRC/patch_gdrnpp_py311.py"
echo "[prep] 4b) PL 1.9 _LiteModule -> _FabricModule"
$PY "$SRC/patch_pl19_compat.py"

echo "[prep] 5) build CppEGLRenderer if missing"
if [ -z "$(find "$GDRN/lib/egl_renderer" -maxdepth 1 -name 'CppEGLRenderer*.so' 2>/dev/null)" ]; then
    # ensure a modern pybind11 is vendored (3.11-compatible)
    PB="$GDRN/lib/egl_renderer/pybind11"
    if ! grep -q "PYBIND11_VERSION_MINOR 1[0-9]" "$PB/include/pybind11/detail/common.h" 2>/dev/null; then
        echo "  swapping vendored pybind11 -> 2.13.6"
        mv "$PB" "${PB}.old" 2>/dev/null || true
        ( cd /tmp && rm -rf pybind11-2.13.6 pb.tgz \
          && curl -sL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz -o pb.tgz \
          && tar xzf pb.tgz && mv pybind11-2.13.6 "$PB" )
    fi
    sudo -n apt-get install -y libglm-dev 2>&1 | tail -1 || true
    ( cd "$GDRN/lib/egl_renderer" && rm -rf build \
      && PYTHONPATH="$GDRN" $PY setup.py build_ext --inplace ) \
      && echo "  CppEGLRenderer built" || echo "  CppEGLRenderer BUILD FAILED"
else
    echo "  CppEGLRenderer already built"
fi

echo "[prep] 6) verify full import chain"
cd "$GDRN" && PYTHONPATH="$GDRN" $PY -c \
  "from core.gdrn_modeling import main_gdrn; print('MAIN_GDRN_IMPORT_OK')" \
  2>&1 | grep -E "MAIN_GDRN_IMPORT_OK|Error|Traceback" | tail -3

echo "PREP_GDRNPP_ENV_DONE"
