#!/usr/bin/env bash
set -uo pipefail
cd /home/age/Desktop/Gruppe3/KIP_POSE/scflow2
PIP=.venv/bin/pip
PY=.venv/bin/python
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
mkdir -p _deps

echo "############ gdown ############"
$PIP install gdown 2>&1 | tail -2

echo "############ bop_toolkit ############"
if [ ! -d _deps/bop_toolkit ]; then
  git clone --depth 1 https://github.com/thodan/bop_toolkit _deps/bop_toolkit 2>&1 | tail -2
fi
$PIP install --no-build-isolation -e ./_deps/bop_toolkit 2>&1 | tail -15
$PY -c "from bop_toolkit_lib.inout import load_depth; print('bop_toolkit_lib OK')" 2>&1 | tail -5

echo "############ FULL IMPORT TEST (datasets + models) ############"
$PY -c "
import warnings; warnings.filterwarnings('ignore')
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
from datasets import build_dataset
print('datasets.build_dataset import OK')
from models import build_refiner, build_estimator
print('models.build_refiner import OK')
import mmcv
from mmcv import Config
cfg = Config.fromfile('configs/flow_refine/scflow2.py')
print('config loaded; model type =', cfg.model.type)
print('ALL-IMPORTS-OK')
" 2>&1 | tail -40
echo "############ BOP-IMPORTS-DONE ############"
