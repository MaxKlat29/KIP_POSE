# S174 — Peak-Config robust ausrollen + Eval sauber online + Cleanup

**Ticket:** T-174 · **Agent:** Sam (DevOps) · **Session:** 2026-06-07-multi-pipeline-pose
**Box:** `max@100.85.216.95` (Tailscale, RTX 3090, `maxgpuserverobk`) · **Datum:** 2026-06-09

> **Ergebnis: Plattform ist reboot-fest am Peak + Eval sauber online.**
> Ein Reboot bringt die volle 12-Kombi-Plattform automatisch zurück (6 Mesh-Container
> `restart: unless-stopped` + native `gdrnpp-svc:8012` als enabled systemd-Unit), und
> `/KIP` zeigt genau die eine ehrliche Tabelle (1 Run, AR 0.886 2-Klassen).

---

## 1. Peak-Performance-Config robust + reboot-fest

### 1a. Mesh-Container — `restart: unless-stopped` (das „Ausrollen")

**Problem (vorher):** alle 6 Mesh-Container hatten `RestartPolicy=no` → nach Box-Reboot
blieben sie `Exited`, mussten manuell hochgefahren werden.

**Fix:** `~/kip_mesh/docker-compose.yml` — `restart: unless-stopped` nach jedem
`build:`-Block für alle 6 Services eingefügt (yolo-svc, yolo-obb-svc, fp-svc,
gigapose-svc, sam3-svc, gateway). `docker compose up -d` → 6 recreated.

| Beleg | Wert |
|---|---|
| Compose-Backup (Pflicht vor Edit) | `~/kip_mesh/docker-compose.yml.bak-T174-20260609_051458` (md5 == Original vor Edit) |
| `docker compose config --quiet` | `COMPOSE_VALID` |
| RestartPolicy nach Apply | **6/6 = `unless-stopped`** (`docker inspect`) |
| Gateway-Aggregate nach Apply | `127.0.0.1:8090/health` → **6/6 UP** (yolo/fp/gigapose/sam3/yolo_obb/gdrnpp) |

**Peak-Config persistent verifiziert** (`~/kip_mesh/.env` + compose-Defaults):
- gigapose-svc `GP_ICP_MAX_CORR` = compose-Default **`0.025`** (T-167) ✓
- yolo-svc `YOLO_WEIGHTS_PT` = `/mnt/data/kip_pose/data/anker_seg/best.pt` (**trainiert, NICHT nano**) ✓
- yolo-obb-svc `YOLO_OBB_WEIGHTS_PT` = `/mnt/data/kip_pose/data/detector_armvis/detector.pt` ✓
- fp/gigapose Weights-Mounts (`/mnt/data/kip_pose_weights` RO) + sam3 `HF_CACHE_DIR` ✓

### 1b. `gdrnpp-svc:8012` nativ — reboot-fest via systemd

**Problem (vorher):** `:8012` lief als nackter `python app.py` (pid 3967, **PPID 1** =
manuell via `nohup &` nach letztem Reboot gestartet, kein systemd) → kein Auto-Start
nach Reboot. `run_box.sh` ist zudem relativ-pfad-fehlerhaft, wenn aus `~/kip_mesh/gdrnpp-svc`
gestartet (`PROJECT_ROOT=HERE/../../..` resolved zu `/home` → `box_src` nicht gefunden).

**Fix:** neue systemd-Unit `/etc/systemd/system/kip-gdrnpp-svc.service` (analog
kip-server/kip-worker), mit **explizitem PYTHONPATH** statt run_box.sh:

```ini
[Unit]
Description=KIP gdrnpp-svc (Mesh Pipeline A, Port 8012, native gdrnpp-venv, isoliert von :8078)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=max
WorkingDirectory=/home/max/kip_mesh/gdrnpp-svc
Environment=GDRN_ROOT=/mnt/data/bop/repos/gdrnpp
Environment=GDRNPP_PORT=8012
Environment=GDRNPP_MEMFRAC=0.30
Environment=PYTHONPATH=/mnt/data/kip_pose/box_src:/mnt/data/bop/repos/gdrnpp:/mnt/data/bop/repos/gdrnpp/core/gdrn_modeling
ExecStart=/mnt/data/bop/gdrnpp-venv/bin/python /home/max/kip_mesh/gdrnpp-svc/app.py
Restart=on-failure
RestartSec=15
StandardOutput=append:/mnt/data/kip_pose/kip_gdrnpp_svc.log
StandardError=append:/mnt/data/kip_pose/kip_gdrnpp_svc.log
[Install]
WantedBy=multi-user.target
```

| Beleg | Wert |
|---|---|
| `systemd-analyze verify` | `VERIFY_CLEAN` |
| `systemctl enable` | symlink `multi-user.target.wants/` angelegt → `is-enabled = enabled` (**reboot-persist**) |
| **Live-Test** (swap orphan→systemd) | `kill -TERM 3967` (graceful, **kein pkill**, single pid) → `:8012` freigegeben → `systemctl start` |
| Status nach Start | `active (running)`, Main PID 63850 (systemd-CGroup, nicht mehr orphan) |
| `:8012/health` nach ~55s Reload | `{ok:true, state:ready, classes:[anker_kurz,anker_lang], obj:[1,2], cuda:true}` |
| `:8078`-Worker (heilig) | **pid 1189 unberührt** (separater Prozess, separate VRAM-Fraktion) |
| Gateway nach Swap | **6/6 UP** (host-gateway-Alias re-bindet auf neue pid) |

→ **Reboot-fest belegt:** Container-Policy + enabled systemd-Unit. Ein realer Reboot war
nicht nötig — Config gesetzt + Einzel-Restart-Test (Container recreate + systemd start/health)
ist der Beleg.

---

## 2. Eval-Cleanup — nur der ehrliche finale Run bleibt

**Verzeichnis:** `/mnt/data/kip_pose/project/temp/batch_eval/`

| Aktion | Eintrag | Grund |
|---|---|---|
| **gelöscht** | `t173-gp-centroidA` | T-173-experimentell, `n_configs=1` |
| **gelöscht** | `t173-gp-centroidA50` | T-173-experimentell, `n_configs=1` |
| **gelöscht** | `t173-gp-centroidB` | T-173-experimentell, `n_configs=1` |
| **gelöscht** | `t173-gp-centroidB50` | T-173-experimentell, `n_configs=1` |
| **gelöscht** | `run-20260608T201857Z/results.json.bak6obj` | alter 6-obj-AR vor 2-Klassen-Korrektur |
| **BEHALTEN** | `run-20260608T201857Z` | finaler Run: `n_configs=12`, `n_scenes=100`, date `2026-06-09T00:02:42Z`, dur 13424s |

Deletion-Guard: exakte Dir-Namen (kein Glob), `REFUSING to touch keeper`-Check. Keeper
intakt: `results.json` + `EVAL.md` + `csv/` + `eval/`.

**Verify (`:8077`):**
- `/api/eval/runs` → **run_count = 1** (nur `run-20260608T201857Z`)
- `/api/eval/result/run-20260608T201857Z` → **HTTP 200**, Pipeline A (`seg=yolo-obb, pose=gdrnpp`)
  **`ar=0.8861` (2-Klassen)**, `modality=RGB`, `rank=1`, `recommended=true`,
  `ar_6obj=0.2954` (Legacy-Sekundärfeld, nicht primär)

---

## 3. Verify — Eval sauber online (max-utils.com/KIP)

### Public-API (über Caddy `/KIP`)
| Check | Ergebnis |
|---|---|
| `/api/pipelines` | **12/12 available** (gdrnpp recommended + 11 Kombis, alle `avail=true`) |
| `/api/eval/runs` | **genau 1 Run** (`run-20260608T201857Z`, n_cfg=12, date 00:02) |
| `/api/eval/result/run-20260608T201857Z` | **HTTP 200**, top = yolo-obb+gdrnpp, AR **0.8861** (2-Klassen), RGB |

### Frontend (headless Chromium, `.venv` playwright)
| Check | Ergebnis |
|---|---|
| Batch-Eval-Tab | klickbar (`#tab-batch`) |
| Run-Dropdown | **genau 1 Option = „2026-06-09 00:02"** |
| Tabelle | AR **0.886** im Text, RGB + RGBD beide vorhanden (Input-Spalte) |
| Spalten-Header | `Rang · Konfiguration · Input · AR IC-BIN · Szenen · Laufzeit · Abdeckung · Absturz` |
| Methode/Verfahren/Badge-Spalte | **abwesend** (keine „Methode"/„Verfahren") |
| Console-Errors | **0** |
| Screenshots | `recon/s174_kip_batcheval.png` (Voll-Tabelle, 88.6% Top-Row), `recon/s174_kip_table_crop.png` (Header) |

### Plattform-Health
| Check | Ergebnis |
|---|---|
| `:8077/api/health` | `status=ok`, trained `[anker_kurz, anker_lang, zahnrad]`, frontend_present=true |
| FE-Regressionsguard | `tab-live=0` (Live-Tab raus), `cell-loader` present, 3 Tab-Marker (real/sim/batch) |
| Gateway `127.0.0.1:8090/health` | **6/6 UP**, ok=true |
| Sim-Tab non-A-Routing | `yolo_seg__gigapose_rgbd` → HTTP 200, `used_combo/used_seg/used_pose/modality=RGBD` gesetzt, **kein 501 „Routing folgt"** |
| `:8078` Worker + `:8012` gdrnpp-svc | beide listening, heilig/unberührt |

**Sim-Routing-Hinweis:** Der non-A-Job (`e1b1e189`) endete mit `exit -15` — das ist **mein
SIGTERM** auf den Isaac-Render-Subprozess (gen_sdg pid 64856, GPU-Pressure-Schutz, S165-Lesson),
**kein Routing-Bug**. Der Routing-Beleg ist die sofortige HTTP-200-Antwort mit gefülltem
`used_combo`. Render danach beendet → GPU baseline 10.4 GB, Ports responsiv.

---

## Constraints eingehalten
- **NIE `pkill -f`** — nur gezieltes `kill -TERM <single-pid>` (orphan gdrnpp-svc 3967, gen_sdg 64856)
- `:8077`/`:8078`/gdrnpp-worker (pid 1189) **durchgehend heilig**, nie berührt
- Nur Mesh-Container + native `gdrnpp-svc:8012` gemanagt
- **Compose-Backup vor Edit** (`.bak-T174-20260609_051458`)
- Destruktiv-Guard befolgt: Bus-`blocker` + Checkpoint + Inbox-Check vor jeder rm/kill/recreate

## Artefakte auf der Box
- `~/kip_mesh/docker-compose.yml` (6× `restart: unless-stopped`) + `.bak-T174-20260609_051458`
- `/etc/systemd/system/kip-gdrnpp-svc.service` (enabled), Log `/mnt/data/kip_pose/kip_gdrnpp_svc.log`
- `/mnt/data/kip_pose/project/temp/batch_eval/run-20260608T201857Z` (einziger Run)
