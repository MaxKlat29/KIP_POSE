# KIP_POSE — CLAUDE.md

6D pose estimation pipeline for robotic pick-and-place.
Generates synthetic training data (RGB + annotations) via NVIDIA Isaac Sim + Replicator.

## Environment (this machine)

| Thing | Path |
|-------|------|
| Isaac Sim 5.1 | `/home/age/isaacsim` |
| Isaac Sim Python | `/home/age/isaacsim/python.sh` |
| USD assets (original) | `/home/age/Downloads/pivarom-SDG-SDG-IsaacSim-USD-Files/SDG/IsaacSim/USD-Files/` |
| USD assets (copy) | `/home/age/Desktop/Gruppe3/USD-Files/` |
| Output (git-ignored) | `data/output/` |
| Symlinked USD dir | `data/usd/` → original Downloads path (texture paths resolve correctly from there) |

GPU: NVIDIA RTX 5000 Ada Generation (32 GB VRAM)

## Key parts / USD files

| File | Description |
|------|-------------|
| `Anker_Kurz.usd` | Short armature (primary training object) |
| `Anker_Lang.usd` | Long armature |
| `NEURA_LARA5_Pose_Zivid_Detection.usd` | Full assembled scene: LARA5 robot + table + tray + Zivid camera (`/World/Zivid`) |
| `GST_Scene.usd` | Alternate full scene |
| `GST_Foto_Scene.usd` | Photo variant of scene |
| `Tray_Anker_leer.usd` | Empty tray |
| `lara5.usd` | LARA5 robot arm only |

## Running locally

All scripts use `/home/age/isaacsim/python.sh` and default USD dir to
`../USD-Files` (relative to the repo root).

### Quickest start — minimal demo (no scene file needed)
```bash
bash scripts/run_local_minimal.sh
```
Builds a temporary scene from scratch, places Anker parts, renders 3 frames.

### Scene-based render (real assembled scene + oriented bounding boxes)
```bash
bash scripts/run_local_scene.sh
# or
SCENE=GST_Scene.usd SPAWN=random NUM_RENDERS=20 bash scripts/run_local_scene.sh
```

### Full SDG (requires scene with Zivid camera + tray)
```bash
bash scripts/run_local_sdg.sh
# or
MODE=random NUM_RENDERS=50 bash scripts/run_local_sdg.sh
```

### Visualise output labels
```bash
python3 sim_code/visualize_labels.py --dir data/output/minimal
```

### Common env var overrides (any script)
```bash
ISAACSIM_DIR=/home/age/isaacsim   # change if Isaac Sim moves
USD_DIR=/path/to/other/usds        # point at different asset folder
NUM_RENDERS=50                     # how many frames to generate
HEADLESS=0                         # open the GUI (useful for debugging)
```

## Repository layout

```
KIP_POSE/
├── sim_code/
│   ├── datagenerationscript.py   # core SDG logic (script-editor + helpers)
│   ├── run_minimal.py            # build-your-own-scene runner
│   ├── run_scene.py              # open existing scene + oriented bbox
│   ├── run_sdg.py                # headless full-pipeline runner
│   └── visualize_labels.py       # overlay label PNGs
├── scripts/
│   ├── run_local_minimal.sh      # LOCAL: minimal demo
│   ├── run_local_scene.sh        # LOCAL: scene-based render
│   ├── run_local_sdg.sh          # LOCAL: full SDG
│   ├── demo_minimal.sh           # REMOTE: wake workstation + run + pull
│   └── wake_and_run.sh           # REMOTE: full remote run
├── data/
│   ├── usd/                      # symlink → ../USD-Files (git-ignored payload)
│   └── output/                   # generated datasets (git-ignored)
├── konzept/                      # project concept document
└── .claude-project/              # roadmap + phases
```

## Config entry points

- **Asset paths / labels**: `datagenerationscript.py` → `ASSETS` list
  (overridden by `SDG_USD_DIR` env var at import time)
- **Camera path**: `ZIVID_CAMERA_PATH` constant or `SDG_CAMERA_PATH` env var
- **Output dir**: `OUTPUT_DIR` constant or `SDG_OUTPUT_DIR` env var
- **Spawn mode**: `SPAWN_MODE` = `"tray"` or `"random"`
- **Physics settling**: `PHYSICS_SETTLE_STEPS` (default 120 frames = 2 s)

## Adding new part assets

1. Drop the `.usd` file into `USD-Files/`
2. Add an entry in `datagenerationscript.py`:
   ```python
   ASSETS = [
       {"path": os.path.join(_USD_DIR, "Anker_Kurz.usd"),  "label": "Anker_Kurz"},
       {"path": os.path.join(_USD_DIR, "NewPart.usd"),      "label": "NewPart"},
   ]
   ```
3. Add the label to the `part_classes` set in `run_scene.py:compute_oriented_boxes`
   if you want oriented bounding boxes for it.

## Isaac Sim notes

- Always run via `python.sh` — it sets `CARB_APP_PATH`, `ISAAC_PATH`, and sources
  `setup_python_env.sh` before calling `kit/python/bin/python3`.
- `OMNI_KIT_ACCEPT_EULA=YES` is set automatically by all run scripts.
- Set `/app/asyncRendering false` and `/omni/replicator/asyncRendering false` for
  headless captures (already done in `run_minimal.py` and `run_scene.py`).
- First run after install is slow (~5 min shader compile); subsequent runs are fast.
