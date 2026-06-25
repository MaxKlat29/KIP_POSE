# S168 — Stack-Up nach Box-Reboot (T-168)

**Session:** `2026-06-07-multi-pipeline-pose` · **Agent:** Sam (DevOps) · **Box:** `100.85.216.95` (RTX 3090, sm_86)
**Datum:** 2026-06-08 · **Trigger:** Box per WoL geweckt, frischer Reboot (uptime 2 min), GPU leer.
**Ziel:** Kompletten POSE-Stack hochfahren + verifizieren — startklar für Kais Final-Eval. **Kein Rebuild, nur Start.**

---

## TL;DR

**Stack live, startklar für Kais Final-Eval.** Alles grün, kein Service neu gebaut, kein `pkill`, Civion nicht angefasst.

---

## IST-Zustand bei Ankunft (Hardware schlägt Bus — alles gemessen)

| Komponente | Status bei Ankunft | Aktion |
|---|---|---|
| **kip-server :8077** | `active` (systemd auto-start) | — lief schon |
| **kip-worker :8078** | `active` (systemd auto-start), Modelle noch am Laden | — lief schon, abgewartet bis `ready` |
| **Mesh 6 Container** | alle `Exited (0/137)` (Reboot, keine restart-policy) | `docker compose up -d --no-build` |
| **gdrnpp-svc :8012** | down (nicht im compose, nativ) | `run_box.sh`-Mechanik, nohup-Start |
| **Weights/best.pt/project** | alle da (`/mnt/data` Reboot-persistent) | nur verifiziert, nichts angefasst |
| **:8000** | `uvicorn` = civion-api (unrelated) | NICHT angefasst — Gateway läuft auf 8090 |

---

## Was gestartet wurde

### 1. Mesh-Stack (6 Container) — `docker compose up -d --no-build` in `~/kip_mesh`
- gateway → `127.0.0.1:8090->8000` (Loopback; :8000 = civion-api, Port-Kollision aus T-151)
- yolo-svc :8001, fp-svc :8002, gigapose-svc :8003, sam3-svc :8004, yolo-obb-svc :8011 (intern, via Gateway)
- **gigapose ICP-Env (Kai T-167) verifiziert im laufenden Container:** `GP_ICP_MAX_CORR=0.025`, `GP_ICP_ITERS=50`, `GP_ICP_ESTIMATION=point`. Compose-Default (`${GP_ICP_MAX_CORR:-0.025}`), kein `.env`-Override → Patch überlebt Reboot persistent.

### 2. gdrnpp-svc nativ :8012 — `run_box.sh`-Mechanik
- `/mnt/data/bop/gdrnpp-venv/bin/python ~/kip_mesh/gdrnpp-svc/app.py`, `GDRNPP_MEMFRAC=0.30`, via `nohup` (PID 3967).
- **PYTHONPATH-Override** auf real `box_src=/mnt/data/kip_pose/box_src` (run_box-Default würde `/home/max/box_src` resolven — Box-Layout-Quirk, T-151).
- Log: `~/kip_mesh/gdrnpp-svc/svc_box.log`. `state: loading → ready` in ~60s.

### 3. Native Services (liefen schon)
- kip-server :8077 + kip-worker :8078 — systemd auto-start nach Reboot. kip-server-Drop-in (`KIP_GATEWAY_URL=http://127.0.0.1:8090`, T-152) überlebte Reboot.

---

## Verify — alle Health grün

### `/api/pipelines` → **12/12 available**
```
gdrnpp                    avail=True
yolo_obb__foundationpose  avail=True   yolo_obb__gigapose_rgbd  avail=True   yolo_obb__gigapose_rgb  avail=True
yolo_seg__gdrnpp          avail=True   yolo_seg__foundationpose avail=True   yolo_seg__gigapose_rgbd avail=True   yolo_seg__gigapose_rgb avail=True
sam3__gdrnpp              avail=True   sam3__foundationpose     avail=True   sam3__gigapose_rgbd     avail=True   sam3__gigapose_rgb     avail=True
```

### Gateway `127.0.0.1:8090/health` → **6/6 Knoten ok**
`ok:true` · yolo ✓ · fp ✓ (cuda:true, [anker_kurz, anker_lang]) · gigapose ✓ (refiner:true, dataset kip2) · sam3 ✓ · yolo_obb ✓ · gdrnpp ✓ (state:ready, cuda:true, obj [1,2])

### Native Health
- **:8077** `/api/pipelines` → 200, 12 Kombis (Endpoint heißt `/api/pipelines`, nicht `/health`).
- **:8078** `/health` → `ready:true`, `loaded_obj_ids:[1,2,6]` (alle 3 Modelle warm: anker_lang, anker_kurz, zahnrad).
- **:8012** `/health` → `ok:true`, `state:ready`.

### Smoke — Sim-Infer Pipeline A (gdrnpp), byte-identisch zum Live-Worker-Pfad
`GET /api/sim/infer_async?scene=0` → job `762220da` → **`phase:Fertig, pct:100, n_gt:2, n_pred:2`**.
`job_result` → `results` mit echten 6DoF-Posen (`t_world` + `R_world` + part/face/confidence pro Instanz). Pose geliefert ✓.

### VRAM
**10288 / 24576 MiB** (~42 %, viel Headroom). 5 compute apps resident: Live-Worker (1189) + fp/gigapose/sam3-Container + gdrnpp-svc:8012 (3967).

---

## Listening Ports (final)
`0.0.0.0:8077` (kip-server) · `127.0.0.1:8078` (worker) · `127.0.0.1:8090` (gateway loopback) · `0.0.0.0:8012` (gdrnpp-svc, Tailscale-intern)

---

## Hard Rules eingehalten
- **Kein Rebuild** — nur `up -d --no-build` + nativer Start. Reboot = starten, nicht bauen.
- **Kein `pkill`** — kip-server/worker via systemd auto-gestartet, nichts terminiert.
- **Civion nicht angefasst** — :8000 (civion-api) + civion-redis laufen unberührt weiter.
- **gigapose ICP=0.025** (Kai T-167) live im Container bestätigt — persistent über Reboot.

**→ Stack live, startklar für @kai-ml Final-Eval.**
