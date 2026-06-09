# T-167 — gigapose-svc ICP env-Knöpfe (Box-Deploy-Notiz)

Die GigaPose-3D-ICP-Konfiguration (`gigapose_infer._icp`) war hardcoded
`max_corr=0.01, iters=50`. T-167 macht sie env-steuerbar, um den RGB-D-Translations-
Fehler zu tunen (siehe `project/docs/EVAL_T167_rgbd_refiner_tuning.md`).

## Was auf der Box geändert wurde (NICHT in diesem git-Repo — Box-Mesh liegt unter
`/home/max/kip_mesh` + Bind-Mount `/home/max/kip_build/GigaPose`):

1. **`gigapose_infer.py`** (Bind-Mount, persistent über Recreates) gepatcht via
   `patch_icp_t167.py` (idempotent, py_compile-verifiziert). Neue env:
   - `GP_ICP_MAX_CORR` (m, Default 0.01) — open3d-ICP-Korrespondenz-Radius
   - `GP_ICP_ITERS` (Default 50) — max ICP-Iterationen
   - `GP_ICP_ESTIMATION` (`point`|`plane`, Default point) — Punkt-zu-Punkt vs -Ebene
   Backup: `gigapose_infer.py.bak-T167`.

2. **`/home/max/kip_mesh/docker-compose.yml`** — `environment:`-Block auf `gigapose-svc`
   (host-overridebar): `GP_ICP_MAX_CORR: "${GP_ICP_MAX_CORR:-0.025}"` u.a.
   Backup: `docker-compose.yml.bak-T167`.

## Reproduktion eines Sweep-Werts

```bash
ssh max@100.85.216.95
cd /home/max/kip_mesh
GP_ICP_MAX_CORR=0.025 GP_ICP_ESTIMATION=point \
  docker compose up -d --force-recreate --no-deps gigapose-svc
# ~60s Modell-Reload, dann:
cd /mnt/data/kip_pose
/mnt/data/isaacsim-venv/bin/python project/eval/tune_rgbd.py \
  --scenes-dir project/bop/pose_isaac/val \
  --dataset-dir project/bop/pose_isaac --split val \
  --gateway http://localhost:8090 \
  --combos yolo_seg__gigapose_rgbd --iterations 5 \
  --out project/temp/batch_eval --run-id tune-gp3d-mc025
```

## Endzustand (T-167)

Bester schadensfreier Wert = `max_corr=0.025` / `point` / `iters=50` (AR 0.2195→0.2219,
+1.1 %). Compose-Default darauf gesetzt. **Fazit: marginal, schließt die ~33-mm-Translations-
Lücke NICHT** (Detail im EVAL-Doc). FP-iter VRAM-blockiert (Peak schon bei iter=5 randvoll).
