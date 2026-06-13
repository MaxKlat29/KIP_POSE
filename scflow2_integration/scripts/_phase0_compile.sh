#!/usr/bin/env bash
set -uo pipefail
cd /home/age/Desktop/Gruppe3/KIP_POSE/scflow2
PIP=.venv/bin/pip
PY=.venv/bin/python
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
export MAX_JOBS=8
mkdir -p _deps

echo "############ A) pointnet2_ops (--no-build-isolation) ############"
echo "--- arch line (should already be patched to incl 8.9) ---"
grep -n "TORCH_CUDA_ARCH_LIST" _deps/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py
$PIP install --no-build-isolation ./_deps/Pointnet2_PyTorch/pointnet2_ops_lib 2>&1 | tail -15
$PY -c "import pointnet2_ops._ext as e; print('pointnet2_ops _ext OK')" 2>&1 | tail -3

echo "############ B) lietorch (--no-build-isolation) ############"
if [ ! -d _deps/lietorch ]; then
  git clone --recursive https://github.com/princeton-vl/lietorch.git _deps/lietorch 2>&1 | tail -3
fi
$PIP install --no-build-isolation ./_deps/lietorch 2>&1 | tail -30
$PY -c "import lietorch, lietorch_extras; from lietorch import SE3; print('lietorch OK')" 2>&1 | tail -5

echo "############ C) pytorch3d (prebuilt wheel) ############"
$PIP install fvcore iopath 2>&1 | tail -2
$PIP install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html 2>&1 | tail -12
$PY -c "import pytorch3d; from pytorch3d.ops import knn_points; print('pytorch3d', pytorch3d.__version__, 'OK')" 2>&1 | tail -5

echo "############ COMPILE-PHASE-DONE ############"
