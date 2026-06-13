#!/usr/bin/env bash
set -uo pipefail
cd /home/age/Desktop/Gruppe3/KIP_POSE/scflow2
PIP=.venv/bin/pip
PY=.venv/bin/python
SP=.venv/lib/python3.10/site-packages

echo "############ fix numpy + opencv ############"
$PIP uninstall -y opencv-python-headless opencv-python 2>&1 | tail -3
$PIP install --no-deps --force-reinstall "numpy==1.26.4" "opencv-python==4.9.0.80" 2>&1 | tail -4

echo "############ sksparse stub (solve_type=reg => cholmod never called) ############"
mkdir -p "$SP/sksparse"
printf '%s\n' '# stub package: real scikit-sparse needs system SuiteSparse (not installed).' > "$SP/sksparse/__init__.py"
cat > "$SP/sksparse/cholmod.py" <<'PYEOF'
# Stub for scikit-sparse CHOLMOD. SCFlow2 imports this at module load in
# models/utils/raft_3d_basic_blocks.py, but with solve_type='reg' the CHOLMOD
# least-squares solver is never invoked. If it ever is, fail loudly.
def _unavailable(*args, **kwargs):
    raise NotImplementedError(
        "sksparse/CHOLMOD stub: SuiteSparse is not installed. "
        "This path is only used by solve_type in {'pnp'/non-reg cholmod solve}. "
        "Install libsuitesparse-dev + scikit-sparse to enable it.")

def analyze_AAt(*args, **kwargs):
    return _unavailable()

def analyze(*args, **kwargs):
    return _unavailable()

def cholesky(*args, **kwargs):
    return _unavailable()

def cholesky_AAt(*args, **kwargs):
    return _unavailable()

class Factor:
    pass

class CholmodError(Exception):
    pass
PYEOF
echo "stub written to $SP/sksparse/"

echo "############ clean import sanity ############"
$PY -c "
import warnings; warnings.filterwarnings('ignore')
import numpy as np; print('numpy', np.__version__)
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import cv2; print('cv2', cv2.__version__)
import mmcv; print('mmcv', mmcv.__version__)
import albumentations; print('albumentations', albumentations.__version__)
from sksparse import cholmod; print('sksparse stub import OK')
from bop_toolkit_lib.inout import load_depth; print('bop_toolkit_lib OK')
import pointnet2_ops._ext, lietorch, lietorch_extras, pytorch3d
print('compiled deps OK')
from datasets import build_dataset; print('datasets.build_dataset OK')
from models import build_refiner, build_estimator; print('models.build_refiner OK')
from tools.eval import single_gpu_test; print('tools.eval OK')
from mmcv import Config
cfg = Config.fromfile('configs/flow_refine/scflow2.py')
print('config OK; model.type =', cfg.model.type)
print('ALL-IMPORTS-OK')
" 2>&1 | tail -40
echo "############ CLEANUP-DONE ############"
