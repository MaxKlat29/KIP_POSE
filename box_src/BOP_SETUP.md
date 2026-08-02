# BOP Pose-Estimation Stack — GPU-Box Setup

> Ticket **T-063 / Story S-204** — Infra-only (kein Training).
> Installiert alle BOP-Benchmark-Repos auf der RTX-3090-Box, mit Run-Harness
> und Smoke-Verifikation. Stand: 2026-05-22.

## TL;DR

| Was | Wert |
|---|---|
| Box | `$GPU_HOST` (Tailscale) / `<LAN-IP>` (LAN), Hostname `<gpu-hostname>` |
| GPU | NVIDIA RTX 3090, 24576 MiB, Driver 590.48.01 (CUDA 13.1-capable) |
| System-CUDA (nvcc) | 12.0 — Treiber abwärtskompatibel, **cu121-Wheels** laufen |
| Python | `python3.11` (3.11.15) — bewusst NICHT 3.12 (BOP-Repos brechen dort) |
| Haupt-venv | `/mnt/data/bop/bop-venv` (torch 2.5.1+cu121, numpy 1.26.4) |
| GDRNPP-venv | `/mnt/data/bop/gdrnpp-venv` (isoliert — eigene Pins, s.u.) |
| Repos | `/mnt/data/bop/repos/{bop_toolkit,cnos,gigapose,megapose6d,gdrnpp}` |
| Weights | `/mnt/data/bop/weights/...` |
| Logs (nohup) | `/mnt/data/bop/logs/...` |
| Run-Harness | `box_src/gpu_run.sh` (lokal im Worktree) |

## Verzeichnis-Layout auf der Box

```
/mnt/data/bop/
├── bop-venv/                 # py3.11 + torch2.5.1+cu121 — bop_toolkit, cnos, gigapose, megapose6d
├── gdrnpp-venv/              # py3.11 + torch2.5.1+cu121 — gdrnpp (isoliert, kollidierende Pins)
├── repos/
│   ├── bop_toolkit/          # thodan/bop_toolkit  (MIT)        — Format + Eval-Metriken
│   ├── cnos/                 # nv-nguyen/cnos      (MIT)        — Detektion/Segmentierung (SAM, NICHT FastSAM)
│   ├── gigapose/             # nv-nguyen/gigapose  (MIT)        — Pose-Estimation (bundlet megapose-Fork)
│   ├── megapose6d/           # megapose6d/megapose6d (MIT)      — canonical MegaPose
│   └── gdrnpp/               # shanice-l/gdrnpp_bop2022 (Apache)— GDRN++ (custom CUDA-ops)
├── weights/
│   ├── segment-anything/sam_vit_h_4b8939.pth        # 2.4 GB (Apache SAM)
│   ├── gigapose/gigaPose_v1.ckpt                    # 3.6 GB
│   ├── megapose-models/coarse-rgb-906902141/        # checkpoint.pth.tar + config.yaml (83 MB)
│   ├── megapose-models/refiner-rgb-653307694/       # checkpoint.pth.tar + config.yaml (83 MB)
│   ├── megapose-data/                               # MEGAPOSE_DATA_DIR (Beispiele/Modelle on-demand)
│   └── pretrained/                                  # Symlink-Layout für hydra root_dir-Configs:
│       ├── segment-anything -> ../segment-anything  #   CNOS sucht ${root_dir}/pretrained/segment-anything/
│       ├── megapose-models  -> ../megapose-models
│       └── gigaPose_v1.ckpt -> ../gigapose/gigaPose_v1.ckpt
└── logs/                     # nohup-Logs der Heavy-Downloads/-Builds
```

## Lizenz-Gate (adr.md §5-R2) — ERFÜLLT

Alle Repos sind MIT/Apache. **Einziger echter Stolperstein war CNOS' Segmentor:**

- CNOS liefert zwei Segmentor-Configs: `configs/model/segmentor_model/sam.yaml` (SAM, **Apache**) und `fast_sam.yaml` (FastSAM, **AGPL — VERBOTEN**).
- **Gute Nachricht:** CNOS' Default in `configs/model/cnos.yaml` ist bereits `segmentor_model: sam`. FastSAM ist opt-in.
- **Regel:** Nur `python -m src.scripts.download_sam` benutzen, **NIE** `download_fastsam`. CNOS bundlet sein eigenes Apache-`segment_anything/` (das wird genutzt, kein AGPL-Paket installiert).

## GPU-Run-Harness (`box_src/gpu_run.sh`)

Wiederverwendbar, parametrierbar. Macht: WoL (falls Box schläft) → warte-auf-SSH → Befehl-auf-Box → optional rsync zurück.

```bash
# Quick check
box_src/gpu_run.sh -- 'nvidia-smi'

# Run inside bop-venv from a repo dir
box_src/gpu_run.sh -d /mnt/data/bop/repos/cnos -- \
  '/mnt/data/bop/bop-venv/bin/python -c "import src.model.detector; print(\"ok\")"'

# Heavy job als nohup (non-blocking) + später pollen
box_src/gpu_run.sh -- 'cd /mnt/data/bop && nohup ./fetch_weights.sh > logs/x.log 2>&1 & echo PID=$!'
box_src/gpu_run.sh -- 'tail -n 30 /mnt/data/bop/logs/x.log'

# Eval + Ergebnisse zurückholen
box_src/gpu_run.sh -p '/mnt/data/bop/results/run42:./results/run42' \
  -- 'cd /mnt/data/bop && bop-venv/bin/python eval.py --out results/run42'
```

Wichtige Konvention: das Skript **daemonisiert nicht selbst**. Für lange Jobs das
`nohup ... &` INNERHALB des Remote-Commands setzen, dann mit zweitem Aufruf pollen.
ENV-Overrides (BOX_HOST, REMOTE_REPO, WAKE_WAIT_SECS, ...) siehe Skript-Header.

## Wie man jedes Repo invokiert

### 1. bop_toolkit (MIT) — Format + Eval-Metriken
- Installiert editable als `bop_toolkit_lib` in **bop-venv**.
```bash
/mnt/data/bop/bop-venv/bin/python -c "from bop_toolkit_lib import inout, pose_error; print('ok')"
# Eval-Scripts liegen in repos/bop_toolkit/scripts/ (z.B. eval_bop19_pose.py)
```

### 2. CNOS (MIT) — Detection/Segmentation mit SAM
- Läuft aus dem **Repo-Root** (`src.`-Imports), bop-venv.
- DINOv2 wird beim ersten Lauf via `torch.hub.load('facebookresearch/dinov2')` geladen.
- SAM-Weights: `/mnt/data/bop/weights/segment-anything/sam_vit_h_4b8939.pth`.
```bash
cd /mnt/data/bop/repos/cnos
/mnt/data/bop/bop-venv/bin/python -m src.scripts.inference_custom \
  segmentor_model=sam machine.root_dir=/mnt/data/bop/weights  # SAM, NICHT fast_sam
```
> `machine.root_dir=/mnt/data/bop/weights` funktioniert direkt — das `pretrained/`-Symlink-Layout
> (s. Verzeichnis-Layout) zeigt bereits auf SAM/MegaPose/GigaPose-Weights.

### 3. GigaPose (MIT) — Pose-Estimation
- Installiert editable `--no-deps` in bop-venv (Paketname intern `megapose`, s. Kollisions-Hinweis).
- Läuft aus dem **Repo-Root** (`src.`-Imports). Checkpoint: `weights/gigapose/gigaPose_v1.ckpt`.
```bash
cd /mnt/data/bop/repos/gigapose
/mnt/data/bop/bop-venv/bin/python test.py  # configs via hydra; root_dir auf weights setzen
```

### 4. MegaPose6d (MIT, canonical) — Pose-Estimation
- Installiert editable `--no-deps` in bop-venv. **Importiert als `megapose`** (gewinnt gegen Gigapose-Bundle, weil es das einzige pip-installierte `megapose` ist; Gigapose nutzt sein lokales `src.megapose` nur aus eigenem Repo-Root).
- **PFLICHT-ENV (sonst KeyError CONDA_PREFIX):**
```bash
export CONDA_PREFIX=/mnt/data/bop/bop-venv          # megapose/config.py erwartet conda
export MEGAPOSE_DATA_DIR=/mnt/data/bop/weights/megapose-data
cd /mnt/data/bop/repos/megapose6d
/mnt/data/bop/bop-venv/bin/python -c "from megapose.inference.pose_estimator import PoseEstimator; print('ok')"
# Modelle: python -m megapose.scripts.download --megapose_models  (oder schon unter weights/megapose-models/)
```

### 5. GDRNPP (Apache) — GDRN++
- **Eigenes** venv: `/mnt/data/bop/gdrnpp-venv` (kollidierende Pins, s. Stolpersteine).
- Custom-Ops müssen kompiliert werden: `sh scripts/compile_all.sh` (fps/flow/ransac_voting/uncertainty_pnp/torch_nndistance/egl_renderer).
- **Weights password-gated** (Onedrive/BaiDu/ModelScope) — manueller Download nötig, NICHT automatisierbar.
```bash
cd /mnt/data/bop/repos/gdrnpp
PYTHONPATH=. /mnt/data/bop/gdrnpp-venv/bin/python -c "import core.gdrn_modeling as m; print('ok')"
```

## Pretrained-Weights — Status

| Modell | Pfad | Größe | Quelle | Status |
|---|---|---|---|---|
| SAM vit_h | `weights/segment-anything/sam_vit_h_4b8939.pth` | 2.4 GB | fbaipublicfiles (Apache) | ✓ geladen + CPU-load-verifiziert (641M params) |
| GigaPose v1 | `weights/gigapose/gigaPose_v1.ckpt` | 3.6 GB | HF nv-nguyen/gigaPose | ✓ geladen + CPU-load-verifiziert (PL-ckpt) |
| MegaPose coarse-rgb | `weights/megapose-models/coarse-rgb-906902141/` | 83 MB | INRIA archive | ✓ geladen |
| MegaPose refiner-rgb | `weights/megapose-models/refiner-rgb-653307694/` | 83 MB | INRIA archive | ✓ geladen |
| DINOv2 (CNOS descriptor) | torch.hub cache | ~1.1 GB | facebookresearch/dinov2 | on-demand beim 1. Lauf |
| GDRNPP models | `repos/gdrnpp/output/` (manuell) | — | Onedrive/BaiDu (password) | ⚠ NICHT automatisierbar (password-gated) |

## Bekannte Stolpersteine (gelöst / dokumentiert)

1. **Python-Version:** System hat py3.12 — aber BOP-Repos pinnen py3.9/3.10. Wir nutzen **py3.11**
   (verfügbar als `python3.11`). 3.12 bricht detectron2/dataclasses/diverse Pins.
2. **CUDA-Mismatch (scheinbar):** Driver 590 meldet CUDA 13.1, nvcc ist 12.0. Kein Problem —
   Treiber ist abwärtskompatibel, **cu121-Wheels** laufen (`torch.cuda.is_available()==True`).
3. **xformers 0.0.18 Pin (CNOS/GigaPose):** baut NICHT gegen torch2.5/py3.11. Übersprungen
   (`--no-deps`); DINOv2 fällt automatisch auf Non-xformers-Attention zurück.
4. **MegaPose `CONDA_PREFIX` KeyError:** `megapose/config.py:45` macht `os.environ["CONDA_PREFIX"]`.
   Im venv (kein conda) unset → KeyError. **Fix:** `export CONDA_PREFIX=/mnt/data/bop/bop-venv`.
5. **MegaPose py3.11 dataclass:** `training_config.py:145` hatte `hardware: HardwareConfig = HardwareConfig()`
   (mutable default — unter py3.10+ verboten). **Patch:** `field(default_factory=HardwareConfig)`
   (verhaltensgleich). Lokale Repo-Änderung auf der Box.
6. **megapose-Paketname-Kollision:** GigaPose UND megapose6d deklarieren beide `name = megapose` und
   `src/megapose`. **Lösung:** nur megapose6d wird als pip-Paket `megapose` installiert (canonical);
   GigaPose `--no-deps` installiert, läuft über sein lokales `src.megapose` aus eigenem Repo-Root.
7. **numpy 2 vs <2:** `pin`/`cmeel-boost` (MegaPose-Robotics-Extra) ziehen numpy>=2, brechen aber
   bop_toolkit + Pose-Nets (numpy<2). **Entscheidung:** numpy auf **1.26.4** gepinnt; `pin` ist nur
   für optionale Visualisierung, nicht für Inference nötig. cmeel-boost-Warnung ist kosmetisch.
8. **detectron2 build (GDRNPP):** `setup.py` importiert `torch` zur Build-Zeit; PEP-517-Isolation
   hat kein torch → `ModuleNotFoundError: torch`. **Fix:** `pip install --no-build-isolation ...`.
9. **GDRNPP Pin-Hölle:** `onnx==1.8.1`/`onnxruntime==1.8.0`/`mmcv-full`/`deepspeed`/`pillow-simd`
   sind unbaubar/inkompatibel auf py3.11/torch2.5. **Lösung:** isoliertes `gdrnpp-venv` + sanierte
   `gdrnpp_reqs_sane.txt` (alte onnx-Pins/optionale Compiler-Pakete entfernt). GDRNPP-Custom-Ops
   brauchen apt-libs (eigen3/glog/suitesparse/egl) — per `sudo apt` installiert (NOPASSWD aktiv).
10. **GDRNPP-Weights password-gated:** Onedrive/BaiDu/ModelScope mit Passwort — kein Auto-Download
    möglich. Muss manuell nach `repos/gdrnpp/output/` bzw. `pretrained_models/yolox/` gelegt werden.
11. **Geteilte GPU:** Auf der Box laufen u.U. parallele Trainings (z.B. `train_refiner.py` in
    train-venv). Vor GPU-lastigen Smokes `nvidia-smi` checken — Stages sequenziell laden, nie
    fremde PIDs killen. Import-Smokes brauchen keine GPU.

## Smoke-Verify-Ergebnisse (Import-Level, kein Training)

| Repo | Smoke | Ergebnis |
|---|---|---|
| bop_toolkit | `import bop_toolkit_lib; from bop_toolkit_lib import inout, pose_error` | ✓ |
| cnos | `import src.model.{sam,dinov2,detector}` + bundled `segment_anything` | ✓ |
| gigapose | `import src.models.gigaPose` (+ bundled `src.megapose`) | ✓ |
| megapose6d | `from megapose.inference.pose_estimator import PoseEstimator` | ✓ |
| gdrnpp | `import core.gdrn_modeling` (nach detectron2-build + compile_all) | siehe Log-Status |

Versionen: torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 1.26.4, py3.11.15, CUDA-runtime 12.1.
`torch.cuda.is_available() == True`, Device: NVIDIA GeForce RTX 3090 (24576 MiB).

## Laufende Hintergrund-Jobs (nohup) — Stand Session-Ende

GDRNPP ist der Long-Pole (detectron2-CUDA-Build + custom-op-Compile). Self-completing:

- `logs/gdrnpp_pip2.log` — detectron2-Build + GDRNPP-sane-reqs (PID 182542 zur Session-Zeit).
- `finish_gdrnpp.sh` → `logs/finish_gdrnpp.log` (PID 183973): wartet auf den pip-Job,
  ruft dann `scripts/compile_all.sh` (mit gdrnpp-venv im PATH) und macht die
  `import core.gdrn_modeling`-Smoke. **Poll-Befehl:**
  ```bash
  box_src/gpu_run.sh -- 'tail -n 40 /mnt/data/bop/logs/finish_gdrnpp.log; \
    pgrep -f finish_gdrnpp || echo FINISH_GDRNPP_EXITED'
  ```
  Bei `FINISH_GDRNPP_DONE rc=0` + `GDRNPP core.gdrn_modeling OK` ist GDRNPP grün.
  GDRNPP-Inference braucht zusätzlich die **password-gated Weights** (manuell, s.o.).
