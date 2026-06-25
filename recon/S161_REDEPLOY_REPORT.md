# S161 — SAMMEL-REDEPLOY REPORT (T-161, Sam)

Session `2026-06-07-multi-pipeline-pose` · Mode B (Merge + Redeploy) · 2026-06-08
Box `max@100.85.216.95` (maxgpuserverobk, RTX 3090, 24 GB)

**TL;DR:** Alle 5 Fixes konsolidiert + live. T-156/157/158/159/160 sauber in `integration`
gemergt (0 textuelle Konflikte, alle 3 `batch_eval.py`-Änderungen koexistieren verifiziert).
fp-svc + gigapose-svc + gateway mit Tiefen-Fix neu gebaut, kip_server + batch_eval + FE
redeployed + `systemctl restart kip-server.service`. **RGB-D-Tiefen-Fix LIVE BELEGT:** Pose-X
trifft GT auf **17 mm** (vorher/Kontroll-Pfad: **2805 mm ≈ 2,4 m** daneben). FE clean (kein
Live-Tab, 3 Tabs), `/api/pipelines` = **12 Kombis**, Pipeline-A Sim-Smoke intakt (source=worker).
**integration-HEAD `6ec482e2dd7e483fdae125ef0cb7b41871ee3b8c` — NICHT nach main.**
**Startklar für Kais Re-Run** (gateway `http://127.0.0.1:8090`, `batch_eval.py` frisch auf Box).

---

## 1. Konsolidierungs-Merge (lokal, integration) — PASS

Basis `team/multipipe/integration` `b8e5fae` (Kais IC-BIN-Scoping T-152). Dependency-Order,
`--no-ff`. Nicht-Code-Artefakte (board.json, recon-Reports, glb) vor Merge gestasht, danach
zurückgepoppt (board.json mit T-161 wiederhergestellt).

| # | Branch | Inhalt | Konflikt | Auflösung |
|---|---|---|---|---|
| 1 | T-158 `6a06d69` | FE-Cleanup + Live-Tab raus + RGB/RGBD-Input-Spalte | — | clean (nur Frontend) |
| 2 | T-156 `9de2870` | depth_scale end-to-end | — | clean (verschiedene Funktionen als b8e5fae) |
| 3 | T-157 `5fdaf69` | Pipeline-A via Gateway + modality | — | `ort` auto-merge (batch_eval/kip_server) |

**Kein textueller Konflikt.** Die erwartete Drei-Wege-Überlappung in `batch_eval.py`
(T-156 depth_scale + T-157 pipeline-A + T-159 modality) hat `ort` semantisch zusammengeführt,
weil die drei Patches verschiedene Funktionen treffen. **Koexistenz manuell verifiziert:**

- **depth_scale (T-156):** `batch_eval.py:277-278` (`data["depth_scale"]` forward) + `:882-885`
  (`float(cam.get("depth_scale", 1.0))` read).
- **modality (T-159):** `_modality(cfg)` `:603`, in `standings_entry` `:484` + `config_rows`
  `:585` + Tabelle `:801`. Werte RGB | RGBD.
- **pipeline-A (T-157):** `:77-98` `is_pipeline_a` → `seg_source="yolo-obb"`, `pose_source="gdrnpp"`
  (echte OBB-Quelle, nicht mask-Pfad).

**Tests** (`.venv` py3.12, `pytest project/tests/ -q`): **246 passed, 7 skipped, 1 failed.**
Der 1 Fail = bekannter **T-088-TTA-Flake** (`test_tta_wrapper_recovers_true_rotation`,
`1.2074e-06 < 1e-06`, FP-Toleranz an 6. Nachkommastelle, agg=medoid „0.0000° off"). Pre-existing,
kein Merge-Regress.

**integration-HEAD: `6ec482e2dd7e483fdae125ef0cb7b41871ee3b8c`** (3 Merge-Commits). NICHT nach main.

---

## 2. Container-Rebuild (Tiefen-Fix) — PASS

Die `app.py` ist **ins Image gebakt** (`build: context: …/<svc>`; Volumes mounten nur
Weights/Meshes/Repos) → Rebuild nötig, nicht nur Restart.

- **app.py Box-Backup:** `~/kip_mesh/<svc>/app.py.bak-T161-20260608_112947` für fp-svc, gigapose-svc, gateway.
- **Diff Box→lokal verifiziert = AUSSCHLIESSLICH depth_scale** (default 1.0 = mm, Live unverändert;
  BOP-Frames 0.1). Kein Box-Drift, der überschrieben würde.
- **scp** der 3 neuen `app.py` → byte-identisch zu lokal-merged (sha-Match `75544213` / `192d1dba` / `0035b7fb`).
- **`docker compose up -d --build fp-svc gigapose-svc gateway`:** nur diese 3 **Recreated**;
  yolo-svc / yolo-obb-svc / sam3-svc blieben **Up 2 h** (unberührt). gdrnpp-svc nativ unberührt.
- **gateway-aggregate `127.0.0.1:8090/health` → `ok:true` nach 70 s**, alle 6 Knoten ok:
  yolo, fp (cuda, anker_kurz/lang), gigapose (refiner on), sam3, yolo_obb, gdrnpp (ready, obj 1,2).

---

## 3. kip_server + batch_eval + FE Redeploy — PASS

Box-Live-Root `/mnt/data/kip_pose/project` (WorkingDir kip-server.service).
**Box-Backup ZUERST:** `/mnt/data/kip_pose/project/.bak-T161-20260608_093258/` (kip_server.py,
eval/batch_eval.py, pipelines/__init__.py, frontend/kip.html + src/{batch,pipeline,kip.js,kip.css},
**inkl. live.js** für Rollback).

Deployed (alle byte-identisch zu lokal-merged, sha-Match verifiziert):

| File | Grund |
|---|---|
| `kip_server.py` | T-156 depth_scale + T-157/159 Doc-Strings |
| `eval/batch_eval.py` | alle 3 Fixes (Kais Re-Run nutzt es — sha `d7cb5eb9`) |
| `pipelines/__init__.py` | `_autoload_combos()` (alle 12 Kombis registriert; Box-Version hatte es noch nicht) |
| `frontend/kip.html` + `src/{batch.js,pipeline.js,kip.js,kip.css}` | T-158/160 FE-Cleanup |
| `frontend/src/live.js` | **ENTFERNT** (T-158, Backup im .bak-T161) |

`gateway_proxy.py` u. übrige pipelines waren bereits byte-identisch → nicht angefasst.

**Restart:** `sudo systemctl restart kip-server.service` (KEIN pkill). NRestarts 0→0 (kein
Crash-Loop), kip-server active. **kip-worker `:8078` durchgehend `active`** (separater systemd-Baum,
unberührt). Frischer Prozess = Kais „Modul-Cache"-Problem des ersten Laufs gelöst.

---

## 4. Verify (Belege)

### FE aufgeräumt — PASS
- `:8077/` Tab-Buttons: genau **`tab-real` (Reales Foto) · `tab-sim` (Simulation) · `tab-batch`
  (Batch-Eval)** — **KEIN Live-Kamera-Tab**.
- **`live.js`: 0 Referenzen** im ausgelieferten HTML.
- (Live-**Scoreboard** `eval-live` bleibt — das ist das Echtzeit-Ranking während des Batch-Eval-Laufs
  T-147, gewollt; nicht der Live-Kamera-Tab.)

### 12 Kombis + Gateway-Health — PASS
- **`/api/pipelines` → count = 12** (available 12, degraded 2).
- gateway-aggregate `127.0.0.1:8090/health` `ok:true`, alle 6 Knoten gesund (final re-verified).

### RGB-D-Tiefen-Fix-Smoke (der wichtige Beleg) — PASS
Szene `project/bop/pose_isaac/val/000000` frame 0, **BOP depth_scale = 0.1**,
GT anker_kurz (obj 1) X = **-0.2963 m**. gateway `/predict` (seg=yolo, **pose=gigapose_rgbd**):

| Pfad | depth_scale | PRED X | dX vs GT | dist3D |
|---|---|---|---|---|
| **FIX** (BOP korrekt) | **0.1** | -0.3134 m | **17 mm** | 64 mm |
| Kontroll (alter Bug) | 1.0 | -3.1008 m | **2805 mm (≈ 2,4 m)** | 10671 mm |

Der Kontroll-Pfad (`depth_scale=1.0`) reproduziert exakt den T-156-Bug: PRED-Werte sind **10×** zu
groß (-3.10 vs -0.31; Z 11.17 vs 1.12) → der ~2,4-m-X-Shift. Mit dem Fix trifft X auf **17 mm**.
**depth_scale wird end-to-end durchgereicht (Gateway → Service) und angewandt (`png*0.1/1000`).**

> **Hinweis fp-svc:** Der RGB-D-Smoke wurde über `gigapose_rgbd` gefahren, weil **fp-svc bei einem
> einzelnen `/predict` CUDA-OOM** wirft (hält ~11 GiB reserved nach FP-Inference-Spike; mit allen
> residenten Services + gdrnpp-nativ ist der 24-GiB-Pool randvoll). **Kein Code-Bug, kein Deploy-Regress**
> — reiner VRAM-Pressure. gigapose_rgbd (RGB-D, wendet depth_scale identisch an) belegt den Fix sauber.
> Für den Batch-Eval an @kai-ml weitergegeben (Bus-finding): ggf. `PYTORCH_CUDA_ALLOC_CONF=
> expandable_segments:True` oder fp-Kombis sequenziell, sonst OOMen die fp-Kombis im Re-Run.

### Pipeline-A Live-Smoke (Sim) — PASS
`:8077/api/sim/infer?scene=0`: `meta.source = "worker"` (= Pipeline A / gdrnpp), `kept_proj` 2 GT
(obj 1 + 6), `n_offtable_dropped = 0`, 4 results mit voller `t_world`/`R_world` (Anker_Kurz).
Live-Pfad byte-identisch unberührt von den Eval-Änderungen.

### Service-Health
- `:8077` active, health-ok (trained anker_kurz/anker_lang/zahnrad, FE present).
- `:8078` worker ready (`loaded_obj_ids [1,2,6]`, 10 scenes).
- GPU final **10369 / 24576 MiB used (13757 frei)** — fp-svc nach Smoke zurück auf idle.

---

## 5. Stolpersteine (ehrlich)

1. **board.json im Stash:** Der aktuelle Board-State (inkl. T-161) lag uncommittet im Working-Tree
   und wurde mit-gestasht → `kanban`-CLI fand T-161 kurz nicht. Nach `git stash pop` (board.json
   ist Nicht-Code, kein Merge-Konflikt) wieder da. *Lesson:* board.json vor Merge stashen ist ok,
   aber sofort nach den Code-Merges zurückpoppen, bevor man loggt.
2. **fp-svc CUDA-OOM bei Einzel-/predict** (siehe §4): fp-svc allokiert ~11 GiB beim FP-Scorer-Spike
   und gibt sie als PyTorch-reserved **nicht frei** → mit allen residenten Services kein Headroom.
   Container-Restart gibt es frei. **Relevant für Kais Re-Run** (fp-Kombis), als Bus-finding gemeldet.
   Kein Regress dieses Deploys.

---

## 6. Was ist LIVE / Startklar für Kais Re-Run

- **Eval-Target-Gateway:** `http://127.0.0.1:8090` (NICHT :8000 — civion-api). Alle 6 Mesh-Knoten ok,
  Tiefen-Fix (depth_scale) in fp-svc + gigapose-svc + gateway aktiv.
- **`batch_eval.py` frisch auf Box** (`/mnt/data/kip_pose/project/eval/batch_eval.py`, sha `d7cb5eb9`,
  byte-identisch zu integration `6ec482e`) — alle 3 Fixes drin (depth_scale-forward + pipeline-A-via-gateway
  + modality). kip_server frisch neugestartet (Modul-Cache-Problem gelöst).
- **FE live auf `max-utils.com/KIP`:** clean (3 Tabs, marken-frei, RGB/RGBD-Input-Spalte), 12 Kombis.
- **VRAM-Warnung an @kai-ml:** fp-svc-Kombis können OOMen — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  oder fp sequenziell, sonst nur gigapose/gdrnpp-Kombis sauber.

**Rollback:**
- Mesh-app.py: `~/kip_mesh/<svc>/app.py.bak-T161-20260608_112947` zurückkopieren + `docker compose up -d --build <svc>`.
- Live-App/FE: `/mnt/data/kip_pose/project/.bak-T161-20260608_093258/` (inkl. live.js) + `systemctl restart kip-server.service`.
- Mesh stoppen: `cd ~/kip_mesh && docker compose down`. gdrnpp-nativ per PID (NICHT pkill). Live :8077/:8078 davon unberührt.

**Git:** integration-HEAD `6ec482e2dd7e483fdae125ef0cb7b41871ee3b8c` (3 Merge-Commits, alle 5 Fixes).
**NICHT nach main** — main-Merge ist ein separater autorisierter Schritt (Ravi-Gate nach Kais Re-Run).

---
*Bus: status/finding/done gepostet, presence done. Kanban T-161 alle Schritte geloggt. D-1 (Kai restart-request) durch systemd-restart erledigt.*
