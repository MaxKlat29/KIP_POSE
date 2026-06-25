# S154 — FE + Gateway LIVE-Deploy auf /KIP (T-154)

**Agent:** Sam (DevOps) · **Session:** `2026-06-07-multi-pipeline-pose` · **Datum:** 2026-06-08
**Ticket:** T-154 (P0, `#sprint-multipipe #deploy`)
**Verdikt:** ✅ **FE LIVE auf `max-utils.com/KIP` + startklar für Eval.** Pipeline A intakt, Gateway degraded-rebuilt, beide Live-Services stabil.
**NICHT nach main** — E2E-Gate (T-142/S-016) kommt zuerst. Alles bleibt auf `team/multipipe/integration`.

---

## TL;DR

- **Schritt 1 — Konsolidieren:** T-147 (Lena, @b57dda0) + T-153 (Jonas, @a955775) additiv in `team/multipipe/integration` (HEAD `d2aaf7f`). Konfliktfrei. Test-Suite **223 passed / 7 skipped / 1 known-flake** (T-088 TTA-medoid Float-Toleranz, kein Regress). `test_gateway_proxy.py` assert **7→12** angepasst (T-147 hat `/api/pipelines` bewusst auf 12 relaxed).
- **Schritt 2 — Gateway degraded-rebuild:** `gateway`-Container auf der Box mit Jonas' backward-compat `degraded`-Form-Feld neu gebaut. `/predict` akzeptiert `degraded=true` (default-off). Andere 5 Mesh-Services NICHT angefasst. `/health` 6/6 ok.
- **Schritt 3 — kip_server:8077 + FE LIVE deployed:** Backup `project/.bak-T154-20260608_093901` (501M, FE+kip_server.py+pipelines) zuerst. Neue FE-Files + `kip_server.py` + Backend-Module deployed. **Restart via systemd** (`systemctl restart kip-server.service`, KEIN pkill). `:8078`-Worker unberührt. Config-Gap (Gateway-URL) per systemd-drop-in gefixt.
- **Schritt 4 — Live-Verifikation:** `max-utils.com/KIP` lädt das neue FE (2 Dropdowns, Batch-Eval-Tab, Live-Scoreboard). `/api/pipelines` → **12** Kombis. `/api/eval/job/<dummy>` → sauberes Schema. Pipeline-A-Sim-Smoke grün (`n_gt=2, n_pred=2`). Beide Services stabil, NRestarts=0.

---

## Schritt 1 — Konsolidierung T-147 + T-153 → integration

| | |
|---|---|
| Ziel-Branch | `team/multipipe/integration` (HEAD nach T-151) |
| Merge-Quelle | `team/multipipe/T-147` @ `b57dda0` (Lena: FE-Relax-12 + Live-Scoreboard) + `team/multipipe/T-153` @ `a955775` (Jonas: Live-Standings-Runner) |
| Konflikte | **Keine.** Beide berühren `kip_server.py` (T-147=`/api/pipelines`→12 via `pipelines_status`, T-153=`/api/eval/job`), Overlap disjunkt. T-153 zusätzlich `gateway/app.py` (`degraded`-Flag), T-147 FE-Files (`kip.js`/`pipeline.js`/`batch.js`/`kip.css`/`index.html`). |
| Integration-HEAD | `d2aaf7f` (test-consolidation commit) |

### Test-Suite (`python -m pytest project/tests/ -q`)

```
1 failed, 223 passed, 7 skipped in 0.70s
```

- **assert 7→12 (erwartet, gefixt):** `test_gateway_proxy.py:189` `assert len(ps) == 12  # T-147: Gating-Relax 7→12`. Test umbenannt `returns_7`→`returns_12`. `test_gateway_proxy.py` isoliert: **22/22 grün**.
- **1 failed = T-088 TTA-Flake (bekannt, kein Block):** `test_tta_pose.py::test_tta_wrapper_recovers_true_rotation` — `agg=medoid: 0.0000° off`, `assert 1.2074e-06 < 1e-06`. Reine Float-Toleranz-Sache (geodesic_angle Rundung), kein funktionaler Regress. Im Auftrag explizit als bekannt markiert.

---

## Schritt 2 — Gateway degraded-Rebuild auf der Box

- **Was:** Jonas' `gateway/app.py` hat ein neues backward-kompatibles `degraded:bool=False` Form-Feld am `/predict`. Nur bei `degraded=true` wird der seg→yolo-obb-Force für `pose=gdrnpp` übersprungen (Maske bleibt → gdrnpp-svc AABB-Fallback). Default-Verhalten + Pipeline A + Whitelist UNVERÄNDERT (live FE setzt es nie).
- **Deploy:** `mesh/`-Tree additiv nach Box rsync (`--exclude assets/meshes,.env` — Lehre aus T-151, sonst löscht `--delete` box-lokale FP-Meshes), dann `docker compose up -d --build gateway` — **nur** der `gateway`-Container wurde recreated.
- **Andere 5 Services unberührt:** yolo, fp, gigapose, sam3, yolo_obb, gdrnpp — kein Rebuild.
- **Verify:** `GET 127.0.0.1:8090/health` → `{"ok":true, ...}`, alle 6 Services up (`yolo`,`fp`,`gigapose`,`sam3`,`yolo_obb`,`gdrnpp:ready`).

---

## Schritt 3 — kip_server:8077 + FE LIVE-Deploy (Live-/KIP-Replacement)

### Backup ZUERST (1-Befehl-Rollback)

```
/mnt/data/kip_pose/project/.bak-T154-20260608_093901/   (501M)
├── frontend/        # FE-static (pre-T154)
├── kip_server.py    # 68723 bytes (pre-T154)
└── pipelines/       # Backend-Module (pre-T154)
```

**Rollback-Kommando** (falls nötig):

```bash
ssh max@100.85.216.95 '
  cd /mnt/data/kip_pose/project &&
  cp -a .bak-T154-20260608_093901/kip_server.py kip_server.py &&
  rm -rf frontend pipelines &&
  cp -a .bak-T154-20260608_093901/frontend  frontend &&
  cp -a .bak-T154-20260608_093901/pipelines pipelines &&
  sudo systemctl restart kip-server.service
'
```

### Deployed (auf `/mnt/data/kip_pose/project`, Live-Pfad NICHT zerschossen)

- **5 FE-Files:** `frontend/index.html` (2 Dropdowns + Batch-Eval-Tab + Scoreboard-Markup), `frontend/src/kip.js`, `frontend/src/pipeline.js` (2-Dropdown-12-Kombi-Gating), `frontend/src/batch.js` (Live-Scoreboard, pollt `/api/eval/job`), `frontend/src/kip.css`.
- **`kip_server.py`** (proxy + `/api/predict` + `/api/pipelines`→12 + `/api/eval/*`).
- **Backend-Module** `pipelines/` (combos, composed, gateway_proxy, pose_base, seg_base) + `compare_pipelines`/`e2e_infer` + `eval/`.

### Restart-Choreo (heikelster Schritt)

- **systemd, KEIN pkill:** `sudo systemctl restart kip-server.service`. `pkill -f uvicorn` wäre Self-Kill **und** hätte `:8078` getroffen — strikt vermieden. Bus-Guard + Checkpoint vor dem Restart gepostet.
- **Config-Gap entdeckt + behoben:** kip_server-Default `KIP_GATEWAY_URL=gateway:8000` (Docker-DNS-Name, von systemd nicht auflösbar) → 11/12 Kombis `service_down`. **Fix = reiner Env-Override**, kein Code: systemd-drop-in `/etc/systemd/system/kip-server.service.d/override.conf`:

  ```ini
  [Service]
  Environment=KIP_GATEWAY_URL=http://127.0.0.1:8090
  Environment=MESH_GATEWAY_URL=http://127.0.0.1:8090
  ```

  → `daemon-reload` + 2. Restart. Danach `MainPID=193000`, `ActiveEnterTimestamp=07:44:01 UTC`, `NRestarts=0` (kein Flapping).
- **`:8078`-Worker unberührt:** separater Prozess (pid 84836), `ready:true`, `loaded_obj_ids=[1,2,6]` — durchgehend.
- **Pipeline A byte-identisch:** Guard `pipeline=gdrnpp` (oder leer=Default) → `is_pipeline_a` → unveränderter Live-Pfad (`kip_server.py:129-130, 387`).

---

## Schritt 4 — Live-Verifikation (Belege)

### FE auf `max-utils.com/KIP` (public Edge)

| Check | Ergebnis |
|---|---|
| `GET https://max-utils.com/KIP/` | HTTP 200, 12096 bytes, `<title>KIP Posenschätzung</title>` |
| Index-Marker | `Batch-Eval`, `Pipeline-A`, `Dropdown`, `Scoreboard` alle present |
| `kip.js` (23358 B) | importiert `pipeline.js` + `batch.js`, referenziert `api/pipelines` |
| `pipeline.js` (13034 B) | 2-Dropdown-12-Kombi-Gating |
| `batch.js` (21554 B) | Live-Scoreboard: `standings`(23×), `rank`, `poll`(5×), `n_total`, `Scoreboard`(4×), `api/eval/run`, `api/eval/job` |
| `kip.css` (16178 B) | HTTP 200 (Scoreboard-Styles) |

### API-Verify (Box :8077)

- **`GET /api/pipelines` → 12 Kombis.** Flag-Breakdown: **available 9, recommended 7, degraded 2, class_ambiguity 4**, pipeline_a=`gdrnpp`. ✅ matcht Auftrag (7 reco hervorgehoben, 2 degraded, 4 class_ambiguity).
- **`GET /api/eval/job/<dummy>` → HTTP 200 `{"error":"unknown job"}`** — sauberes Schema, kein 500.
- **`GET /api/health`** → `status:ok, gpu_training_active:false, frontend_present:true, trained_objects:[anker_kurz,anker_lang,zahnrad]`.

### Per-Kombi-Verfügbarkeit (live)

| Kombi | avail | reco | degr | ambig |
|---|---|---|---|---|
| `gdrnpp` (Pipeline A) | ✅ | ✅ | | |
| `yolo_obb__foundationpose` | ❌ service_down | | | |
| `yolo_obb__gigapose_rgbd` | ❌ service_down | | | |
| `yolo_obb__gigapose_rgb` | ❌ service_down | | | |
| `yolo_seg__gdrnpp` | ✅ | | ✅ | |
| `yolo_seg__foundationpose` | ✅ | ✅ | | |
| `yolo_seg__gigapose_rgbd` | ✅ | ✅ | | |
| `yolo_seg__gigapose_rgb` | ✅ | ✅ | | |
| `sam3__gdrnpp` | ✅ | | ✅ | ✅ |
| `sam3__foundationpose` | ✅ | ✅ | | ✅ |
| `sam3__gigapose_rgbd` | ✅ | ✅ | | ✅ |
| `sam3__gigapose_rgb` | ✅ | ✅ | | ✅ |

### Pipeline-A Live-Smoke (kein Regress)

`GET /api/sim/infer_async?scene=0` (Default `pipeline=gdrnpp`):

```json
{ "phase": "Fertig", "pct": 100, "scene": 0, "im": 0,
  "n_gt": 2, "n_pred": 2, "source": "worker" }
```

→ Live-Pipeline-A über :8078-Worker läuft byte-identisch. `:8078` ready, `:8077` ready, Gateway 6/6 ok **nach** Smoke + voller Verify. `NRestarts=0`.

> `/api/live/status` (Jetson-Cam) → `reachable:false` — **erwartet** (VPN-aus-Default, Jetson `172.22.192.166` ist TABU/separat). Kein FE-Blocker, der Sim-Pfad deckt den Live-Smoke ab.

---

## ⚠️ Ehrlicher Blocker (nicht-blockend für FE-live, blockt 3 Eval-Kombis)

**3 yolo-obb→Mesh-Kombis fälschlich `service_down`** (`yolo_obb__foundationpose`, `yolo_obb__gigapose_rgbd`, `yolo_obb__gigapose_rgb`), obwohl `yolo_obb-svc` im Gateway-`/health` `ok:true` liefert.

- **Root Cause:** `pipelines/gateway_proxy.py::_gateway_service_up()` mappt den `yolo_obb`-Health-Knoten **nicht** in die `svc_up`-Map (sie enthält nur `yolo, sam3, foundationpose, gigapose_rgbd, gigapose_rgb`). Die 3 yolo-obb-Mesh-Kombis sind zudem nicht in `COMBO_TO_GATEWAY` registriert → fallen im `else`-Zweig auf `svc_up.get("yolo_obb")` = `False` → `service_down`.
- **Lokal im `integration`-Branch identisch reproduziert** (`.venv` + live Gateway-Health) → **KEIN Deploy-Drift**, sondern ein Backend-Logik-Bug in Jonas' `gateway_proxy`. Das Deployment spiegelt den Branch korrekt.
- **Scope:** Nicht in DevOps-Hoheit — ich habe **nicht** still gefixt (würde Ravis Gate umgehen). Als Bus-`finding` @jonas-backend gepostet.
- **Impact:** FE-Gating zeigt die 3 als wählbar-aber-disabled ("Dienst nicht aktiv") — das FE ist live + korrekt. Aber der **12×20-Batch-Eval (T-152) kann diese 3 Kombis nicht real fahren**, bis Jonas `yolo_obb` in die `svc_up`-Map + `COMBO_TO_GATEWAY` aufnimmt.

---

## Status für die Queen

- ✅ **FE LIVE auf `max-utils.com/KIP`** — 2 Dropdowns, 12 Kombis, Batch-Eval-Tab, Live-Scoreboard. Startklar damit Max beim Eval live zuschaut.
- ✅ Gateway degraded-rebuilt (`/predict degraded=true`, default-off). Pipeline A byte-identisch. `:8078` unberührt. Beide Services stabil (NRestarts=0).
- ✅ Backup + 1-Befehl-Rollback dokumentiert.
- ⚠️ **1 ehrlicher Blocker für den Eval (nicht fürs FE):** 3 yolo-obb→Mesh-Kombis `service_down` durch `gateway_proxy`-Logik-Bug → @jonas-backend (`svc_up`-Map + `COMBO_TO_GATEWAY`). Blockt T-152 für diese 3.
- 🚫 **NICHT nach main** — bleibt auf `team/multipipe/integration` bis E2E-Gate (T-142/S-016) grün.

---

## 🔁 T-155-REDEPLOY (Routing-Fix, 2026-06-08) — alle 12 Kombis live

**Verdikt:** ✅ **Der T-154-Eval-Blocker ist behoben + live.** `/api/pipelines` → **available=12 / unavailable=0** (vorher 9 verfügbar, 3 yolo-obb-Mesh fälschlich `service_down`). Plattform startklar für den 12×20-Eval.

### Schritt 1 — Merge T-155 → integration
- Branch `team/multipipe/T-155` (`c6adb1c`) war **1 ahead / 0 behind** integration → FF-able, additiv, **0 Drift** (`git diff --name-status` = nur `kip_server.py` + `gateway_proxy.py` + 3 Test-Files).
- Baseline integration: **223 passed / 1 T-088-Flake** (`test_tta_pose::test_tta_wrapper_recovers_true_rotation`, fp-Toleranz `1.2074e-06 < 1e-06`, deterministisch bekannt, **nicht** T-155).
- `--no-ff` Merge → **`997d344`**, 0 Konflikte, 5 Files. Post-merge: **236 passed / 1 T-088-Flake / 7 skipped** (+13 Tests, exakt wie Jonas ankündigte). Interpreter: `.venv/bin/python` (3.12.13, pytest 9.0.3).

### Schritt 2 — kip_server-Code auf die Box + systemd-Restart
- **Was der Fix tut (Jonas, additiv):** (1) `_gateway_service_up()` mappt jetzt die Health-Knoten `yolo_obb` + `gdrnpp` → die 3 yolo-obb-Mesh-Kombis sind nicht mehr fälschlich service_down. (2) `COMBO_TO_GATEWAY` baut aus `combos.FEASIBLE_COMBOS` (alle 12) statt `COMBO_WHITELIST` (7) → die 5 fehlenden Kombis routen (3 yolo-obb-Mesh via mask-emittierenden `yolo`-Pfad + 2 gdrnpp-degraded mit `degraded=true`-Flag).
- **Gateway selbst unverändert** (degraded-Patch schon @T-154 deployt) → **nur kip_server-Restart**, wie Jonas sagte.
- **Kein Box-Drift:** Box-Disk-shasums beider Files == `integration@d2aaf7f` (T-154-Deploy-Quelle) → ganze-Datei-`scp` korrekt+sicher (keine chirurgische Assembly nötig). `pipelines/combos.py` Box == local (`21c41ac7`, byte-identisch, schon @T-154). Box-venv = `/mnt/data/isaacsim-venv/bin/python`.
- **Backup:** frisches gezieltes `.bak-T155-20260608_082127/` (kip_server.py + pipelines/gateway_proxy.py, `cp -a`) zusätzlich zum vollen `.bak-T154-20260608_093901`.
- **Transfer-Beweis:** Box-Disk nach `scp` == lokale post-merge-shasums (`979b52ed…` kip_server.py, `500c4a31…` gateway_proxy.py) → byte-identisch.
- **Pre-Restart-Smoke auf Box:** `py_compile` OK; import `gateway_proxy` → `COMBO_TO_GATEWAY` count = **11** (NICHT-A) + 1 Pipeline A = 12, `degraded`-Flag exakt auf `yolo_seg__gdrnpp` + `sam3__gdrnpp`.
- **`sudo systemctl restart kip-server.service`** (NIE pkill). Active, neue PID **195799** (war 193000), clean Stop→Start im Journal, `NRestarts=0`. **`:8078` kip-worker.service unberührt (active).**

### Schritt 3 — Live-Verifikation (alle grün)
| Check | Ergebnis |
|---|---|
| `GET /api/pipelines` | **total=12, available=12, unavailable=0** (alle `reason=None`); `degraded=True` nur bei `yolo_seg__gdrnpp` + `sam3__gdrnpp` |
| `GET /api/health` | `status:ok`, `frontend_present:true`, `gpu_training_active:false` |
| `GET /api/eval/job/<dummy>` | HTTP 200 `{"error":"unknown job"}` — graceful, kein 500 |
| Gateway `:8090/health` | `ok:true`; per-Service: yolo, fp, sam3, **yolo_obb=ok**, **gdrnpp=ok**, gigapose `status:ok` |
| `_gateway_service_up()` live | alle 7 source-ids `True` (inkl. der vorher fehlenden `yolo-obb` + `gdrnpp`) |
| `:8077` / `:8078` | beide HTTP 200 / active |

### Pipeline-A Live-Smoke (kein Regress)
`GET /api/sim/infer_async?scene=0` (Default `pipeline=gdrnpp`) → `{"phase":"Fertig","pct":100,"scene":0,"im":0,"n_gt":2,"n_pred":2,"source":"worker"}` — **byte-identisch zu T-154**. Live-Pipeline A über `:8078`-Worker unverändert.

### Optional-Smoke — vorher-kaputte Kombis liefern jetzt eine Pose (statt 400/service_down)
Routing gegen die **Live-Gateway-Health** aufgelöst (kein Test-PNG auf der Box → der aussagekräftige Beweis ist die Resolve-+-Availability-Naht, die 400/service_down vs. Pose entscheidet):

| Kombi (Status @T-154) | resolve → Gateway-Target | Verdikt |
|---|---|---|
| `yolo_obb__gigapose_rgb` (service_down) | `seg=yolo, pose=gigapose_rgb, degraded=False`, beide up | **AVAILABLE → routes to pose** |
| `yolo_obb__foundationpose` (service_down) | `seg=yolo, pose=foundationpose`, beide up | **AVAILABLE → routes to pose** |
| `yolo_seg__gdrnpp` (unrouted) | `seg=yolo, pose=gdrnpp, degraded=True`, beide up | **AVAILABLE → routes (Maske bleibt)** |
| `sam3__gdrnpp` (unrouted) | `seg=sam3, pose=gdrnpp, degraded=True`, beide up | **AVAILABLE → routes (Maske bleibt)** |

`is_pipeline_a(seg=yolo-obb, pose=gdrnpp)` UND `pose=GDRNPP` (case-insensitiv) = **True** → Pipeline A bleibt Live-Monolith, fällt NIE durch zum Gateway.

### Status für die Queen (Redeploy)
- ✅ **Alle 12 Kombis live + korrekt geroutet** auf `team/multipipe/integration` HEAD `997d344`. T-154-Eval-Blocker geschlossen.
- ✅ Pipeline A byte-identisch (`n_gt=2/n_pred=2`), `:8078` unberührt, `NRestarts=0`.
- ✅ **Startklar für den 12×20-Eval (T-152/@kai-ml).**
- 🚫 **NICHT nach main** — integration HEAD `997d344`, bleibt bis E2E-Gate (T-142/S-016) grün.

### Rollback (1 Befehl)
```bash
ssh max@100.85.216.95 '
  cd /mnt/data/kip_pose/project &&
  cp -a .bak-T155-20260608_082127/kip_server.py kip_server.py &&
  cp -a .bak-T155-20260608_082127/pipelines/gateway_proxy.py pipelines/gateway_proxy.py &&
  sudo systemctl restart kip-server.service'
```
