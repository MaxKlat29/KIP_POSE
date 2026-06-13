"""Forward-pass de-risk: run SCFlow2 refiner on our Anker_Kurz frames (FP init poses),
no GT. Dumps refined BOP poses and prints init->refined deltas (mm / deg)."""
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, torch
from functools import partial
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, collate
from torch.utils.data import DataLoader, SequentialSampler
from datasets import build_dataset
from models import build_refiner
from tools.eval import single_gpu_test

# mmcv 1.7.2 x torch 2.1: MMDataParallel.scatter passes int device to torch's
# _get_stream which now requires a torch.device. Coerce it.
import mmcv.parallel._functions as _mpf
_orig_get_stream = _mpf._get_stream
_mpf._get_stream = lambda d: _orig_get_stream(torch.device('cuda', d) if isinstance(d, int) else d)

CKPT = 'checkpoints/scflow2_files/scflow2_pretrained.pth'
SAVE = 'results/kip_refined'

cfg = Config.fromfile('configs/flow_refine/scflow2_kip.py')
dataset = build_dataset(cfg.data.test)
print(f'dataset: {len(dataset)} images')
loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                    num_workers=0, collate_fn=partial(collate, samples_per_gpu=1), shuffle=False)
model = build_refiner(cfg.model)
load_checkpoint(model, CKPT, map_location='cpu')
model = MMDataParallel(model.cuda().eval(), device_ids=[0])

outputs = single_gpu_test(model, loader, validate=False)
print(f'\ngot {len(outputs)} refined results')
dataset.format_results(outputs, save_dir=SAVE)

def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

init = json.load(open('data/kip/init/000000/scene_gt.json'))
ref = json.load(open(f'{SAVE}/000000/scene_gt.json'))
print('\nframe |  init t (mm)            | refined t (mm)         | dt(mm) | dR(deg)')
print('-'*84)
for k in sorted(init, key=lambda x: int(x)):
    Ri = np.array(init[k][0]['cam_R_m2c']).reshape(3, 3); ti = np.array(init[k][0]['cam_t_m2c'])
    Rr = np.array(ref[k][0]['cam_R_m2c']).reshape(3, 3); tr = np.array(ref[k][0]['cam_t_m2c'])
    dt = float(np.linalg.norm(tr - ti)); dR = geodesic_deg(Ri, Rr)
    print(f'  {k:>3} | [{ti[0]:7.1f} {ti[1]:7.1f} {ti[2]:7.1f}] | [{tr[0]:7.1f} {tr[1]:7.1f} {tr[2]:7.1f}] | {dt:6.1f} | {dR:6.2f}')
print('\nPHASE1-FORWARD-OK')
