# KIP_POSE

6D pose estimation for robotic pick-and-place: a synthetic-data-generation
pipeline plus the surrounding ML/eval/visualization tooling for detecting and
posing parts (`Anker_Kurz`, `Anker_Lang`, …) on a tray so a robot arm can grasp
them.

## Repository layout

| Path | What |
|------|------|
| `sim_code/` | Synthetic-data-generation pipeline (NVIDIA **Isaac Sim** + Replicator) |
| `scripts/` | Workstation setup + remote-run harness (Wake-on-LAN → Tailscale → headless run) |
| `data/` | USD assets (`data/usd/`) and rendered datasets (`data/output/`) — git-ignored, delivered/generated separately |
| `konzept/` | Project concept document |
| `.research/` | State-of-the-art research brief |
| `.claude-project/` | Roadmap, phases and concept managed by the `claude-project` pipeline |
| `.claude-kanban/` | Live work board |

## Synthetic data generation (`sim_code/`)

Generates annotated training images from a USD scene: RGB, 2D bounding boxes,
semantic + instance segmentation and depth maps. Two spawn modes — `tray`
(parts placed in a grid of tray slots) and `random` (parts dropped with physics
settling).

- `datagenerationscript.py` — the original pipeline, made for the Isaac Sim
  **Script Editor** (interactive, async editor loop).
- `run_sdg.py` — **headless standalone runner** for the workstation; boots a
  `SimulationApp`, opens the scene and re-uses the same generation helpers.

Asset paths and output location are configurable via env vars
(`SDG_USD_DIR`, `SDG_OUTPUT_DIR`, `SDG_CAMERA_PATH`) or CLI flags on `run_sdg.py`.

## Running it

Isaac Sim 5.1 runs headless on the GPU workstation (Ubuntu 24.04, RTX 3090).

**One-time setup** (installs Isaac Sim into a Python 3.11 venv on `/mnt/data`):

```bash
bash scripts/setup_isaacsim_workstation.sh
# smoke test:
/mnt/data/isaacsim-venv/bin/python scripts/verify_isaacsim.py
```

**Minimal demo — simulation → top-down image with labels** (proven end-to-end):

```bash
bash scripts/demo_minimal.sh
```

This wakes the workstation, runs the minimal SDG (`sim_code/run_minimal.py`:
real Anker parts on a ground plane, top-down camera), pulls the result back, and
overlays the labels (`sim_code/visualize_labels.py`) into
`data/output/minimal/annotated_*.png`.

**Full dataset run** (uses a scene with the Zivid camera, once configured):

```bash
bash scripts/wake_and_run.sh           # see the script header for env overrides
```

## Requirements

- NVIDIA Isaac Sim 5.1 (RTX GPU with RT cores, ≥16 GB VRAM)
- USD assets for the parts + a scene containing the tray and the Zivid camera

## Credits

Original Isaac Sim data-generation script: [@Marc8350](https://github.com/Marc8350).
