#!/usr/bin/env bash
set -uo pipefail
cd /home/age/Desktop/Gruppe3/KIP_POSE/scflow2
PY=.venv/bin/python
PIP=".venv/bin/pip"
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# Ada Lovelace = sm_89; include common archs for safety
export TORCH_CUDA_ARCH_LIST="8.9"

echo "######## nvcc check ########"
nvcc --version 2>&1 | tail -2 || echo "NO NVCC"

echo "######## [1/3] pin numpy<2 + curated non-compiled deps ########"
$PIP install "numpy==1.26.4" 2>&1 | tail -2
$PIP install \
  addict yapf==0.40.1 packaging "opencv-python==4.9.0.80" "scipy==1.12.0" \
  "scikit-image==0.22.0" "imageio==2.34.0" "einops==0.7.0" "kornia==0.6.1" \
  "timm==0.9.16" transforms3d terminaltables "trimesh==4.2.2" "pypng==0.20220715.0" \
  "omegaconf==2.3.0" tensorboardX matplotlib "pyyaml==6.0.1" "ruamel.yaml==0.18.6" \
  regex ftfy "transformations==2024.6.1" "roma==1.4.4" 2>&1 | tail -4

echo "######## [2/3] openmim ########"
$PIP install -U openmim 2>&1 | tail -3

echo "######## [3/3] mmcv-full==1.7.2 via mim (may build from source) ########"
.venv/bin/mim install mmcv-full==1.7.2 2>&1 | tail -40

echo "######## import check: mmcv ########"
$PY -c "import mmcv, mmcv.ops; print('mmcv', mmcv.__version__, '| ops OK')" 2>&1 | tail -5
echo "######## BUILD-PHASE-DONE ########"
