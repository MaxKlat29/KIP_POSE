# Reproducing POSE from scratch

This document describes how to reproduce the POSE pipeline end-to-end: the
**local** inference + tests (laptop / workstation) and the **GPU-box** stacks
(synthetic-data generation, detector training, BOP 6D-pose estimation/eval).

There are two distinct environments:

| Environment | What runs there | How it's set up |
|---|---|---|
| **Local** | `e2e_infer.py`, `bop_adapter.py`, the notebooks, the test suite, the 3D viewer | one `pip install` (this doc, §1) |
| **GPU box** | Isaac-Sim SDG, YOLOv8-OBB detector training, BOP toolkit (convert/validate/eval), GDRNPP training/inference | per-task venvs (this doc, §2) + `box_src/BOP_SETUP.md` |

Nothing in `project/` requires a GPU. The box is only needed to (re)generate
training data, train the detector/pose models, and run BOP evaluation.

---

## 0. Secrets / configuration first

The pipeline talks to a remote GPU box. Host/IP/MAC/paths are **not** committed.
Copy the template and fill in your own values:

```bash
cp project/.env.example project/.env
# then edit project/.env — GPU_HOST, WOL_HOST, WOL_MAC, BOX_REPO, BOX_*_PY
```

`.env` is git-ignored. The notebooks read it via `ENV.get(KEY, <default>)`, and
`box_src/gpu_run.sh` takes the same values as `BOX_USER` / `BOX_HOST` /
`BOX_MAC` / `WOL_RELAY` / `REMOTE_REPO` env overrides (all have placeholder-safe
defaults documented in the script header).

> **History note:** if you cloned this repo before the `.env` was removed from
> tracking, the old values may still live in the git **history**. Scrub them
> (`git filter-repo` / BFG) or rotate the credentials before publishing.

---

## 1. Local setup (inference + tests)

Requirements: Python 3.11+ (verified on 3.11 and 3.12).

```bash
pip install -r project/requirements.txt
```

Core deps are tiny: `numpy`, `pillow`, `pytest`, `jsonschema`. The real
YOLOv8-OBB detector (`ultralytics` + `torch`) is **optional** and commented out
in `requirements.txt` — without it `e2e_infer.py` falls back to SDG annotator
boxes, then a whole-image dummy box, so the pipeline never hard-fails.

### Run the tests

```bash
cd project && python3 -m pytest tests/ -q     # 21 tests, all green
```

### Run inference on one image

```bash
python3 project/e2e_infer.py --image project/input/<scene>.png
# writes project/temp/pose_result.json (schema-validated against
# project/pose_result.schema.json)

# ...and open the 3D viewer on the result:
python3 project/e2e_infer.py --image project/input/<scene>.png --serve
```

Without a trained `models/detector.pt` and a GDRNPP checkpoint, the pose stage
runs in **MOCK** mode (deterministic plausible poses on the table plane) so the
whole chain produces a schema-valid `pose_result.json` immediately. Drop a real
`models/detector.pt` (YOLOv8-OBB) and pass `--checkpoint <gdrnpp.pth>` to switch
to the real backends.

---

## 2. GPU-box setup (SDG · training · BOP eval)

The box runs four isolated venvs (pinned independently because their stacks
collide). The canonical, detailed install + smoke-test guide is
**`box_src/BOP_SETUP.md`** — this section is the high-level map.

| venv | Python | Purpose | Key pins (snapshot 2026-05-23) |
|---|---|---|---|
| `isaacsim-venv` | 3.11 | Isaac-Sim synthetic-data generation (SDG) | Isaac Sim 5.1 (bundled wheels; `pip freeze` is empty — packages ship inside the Isaac runtime, not as PyPI dists) |
| `train-venv` | 3.12 | YOLOv8-OBB detector training | `ultralytics==8.4.53`, `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `numpy==2.4.4`, `opencv-python==4.13.0.92` |
| `bop-venv` | 3.11 | BOP toolkit: Isaac→BOP convert, validate, eval; CNOS/GigaPose/MegaPose | `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `numpy==1.26.4`, `opencv-python==4.11.0.86`, `scipy==1.17.1`, `pyrender==0.1.45`, `trimesh==4.12.2`, `kornia==0.8.3`, `pytorch-lightning==2.6.4`, `hydra-core==1.3.2`, `omegaconf==2.3.0` |
| `gdrnpp-venv` | 3.11 | GDRNPP 6D-pose training/inference (isolated — old pins) | `torch==2.5.1+cu121`, `numpy==1.23.5`, `detectron2` (git e0ec4e1), `mmcv-full==1.7.2`, `pytorch-lightning==1.9.0`, `timm==0.6.7`, `scipy==1.10.1`, `opencv-python==4.11.0.86` |

> The version snapshot above was read off the live box on 2026-05-23 via
> `ssh max@<GPU_HOST> '<venv>/bin/pip freeze'`. It is a reference, not a lock
> file — recreate each venv following `box_src/BOP_SETUP.md`, which also covers
> CUDA/driver requirements (RTX 3090, CUDA 12.1 wheels), the source-built
> `mmcv-full`, and the GDRNPP weight-download caveats (password-gated).

### Box directory layout (see `box_src/BOP_SETUP.md` for the full tree)

```
<BOX_REPO>/                 # your repo checkout on the box (BOX_REPO)
<bop-root>/                 # venvs, repos, weights, logs, results
  ├─ bop-venv/  gdrnpp-venv/  isaacsim-venv/  train-venv/
  ├─ repos/{bop_toolkit,cnos,gigapose,megapose6d,gdrnpp}
  ├─ weights/...   logs/...   results/...
```

---

## 3. The remote-run workflow (WoL → SSH → run → rsync)

All box jobs go through one composable harness, `box_src/gpu_run.sh`. It:

1. **Wakes** the box (Wake-on-LAN magic packet via the relay host) if it's asleep,
2. **waits** for SSH to come up,
3. **runs** one command on the box (in `BOX_REPO` by default),
4. optionally **rsyncs** result artifacts back to the local machine.

```bash
# one-shot: check the GPU
box_src/gpu_run.sh -- 'nvidia-smi'

# smoke import inside a venv
box_src/gpu_run.sh -- '<bop-venv>/bin/python -c "import torch; print(torch.cuda.is_available())"'

# long job: launch under nohup on the box (returns immediately), poll later
box_src/gpu_run.sh -- 'cd <bop-root> && nohup ./job.sh > logs/job.log 2>&1 & echo PID=$!'
box_src/gpu_run.sh -- 'tail -n 30 <bop-root>/logs/job.log; pgrep -f job.sh || echo DONE'

# run + pull a results dir back to the local worktree
box_src/gpu_run.sh -p '<bop-root>/results/run42:./results/run42' \
  -- 'cd <bop-root> && <bop-venv>/bin/python eval.py --out results/run42'
```

Host/MAC/relay/repo come from env (`BOX_HOST`, `BOX_MAC`, `WOL_RELAY`,
`REMOTE_REPO`, `BOX_USER`) or fall back to the script's documented defaults —
set them from your `.env` to point at your own box. The harness deliberately
does **not** daemonize; wrap long jobs in `nohup ... &` *inside* the remote
command (as above) so SSH returns immediately.

---

## 4. Full reproduction order (zero → results)

1. **Local:** `pip install -r project/requirements.txt`; `pytest` green (§1).
2. **Box venvs:** create the four venvs per `box_src/BOP_SETUP.md` (§2).
3. **SDG:** generate arm-visible top-down data with the Isaac venv, then
   convert Isaac → BOP — see `box_src/README_BOP_DATA.md`.
4. **Detector:** train the YOLOv8-OBB detector in `train-venv`
   (`box_src/train_detector_armvis.py`); copy `detector.pt` into
   `project/models/`.
5. **Pose:** train/run GDRNPP in `gdrnpp-venv`; point `e2e_infer.py
   --checkpoint` at the resulting `.pth`.
6. **Eval:** score predictions with the BOP toolkit — `box_src/eval_bop.sh` /
   `box_src/EVAL_BOP.md`.
7. **Inference + viewer (local):** `python3 project/e2e_infer.py --image ...
   --serve` to produce a schema-valid `pose_result.json` and open the 3D viewer.

For the box-side specifics (commands, weights, troubleshooting), always defer to
`box_src/BOP_SETUP.md`, `box_src/README_BOP_DATA.md`, and `box_src/EVAL_BOP.md`.
