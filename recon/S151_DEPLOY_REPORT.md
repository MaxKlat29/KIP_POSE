# S151 — FINAL DEPLOY REPORT (T-151, Sam)

Session `2026-06-07-multi-pipeline-pose` · Mode B (Deploy-Master) · 2026-06-08
Box `max@100.85.216.95` (maxgpuserverobk, RTX 3090, 24 GB)

**TL;DR:** Backend ist live. Das volle Mesh läuft zum ersten Mal komplett mit echtem
trainiertem yolo-seg. 6 Services `/health` grün, gateway-aggregate `ok:true`, yolo-seg
liefert echte Anker-Masken (vs 0 mit nano), gateway `/predict` E2E grün. Live-Services
:8077/:8078 durchgehend unberührt. VRAM all-resident 8.7/24 GB. M1+M2 angewandt.
integration HEAD `7b909ce` — **NICHT** nach main (wartet E2E-Gate S-016).

---

## 1. best.pt gesichert + verifiziert — PASS

- Quelle: `/mnt/data/kip_pose/data/anker_seg/_runs/seg/weights/best.pt` (162 MB, mtime 06:23,
  bisheriger mAP50-0.99-Checkpoint; Training lief zu dem Zeitpunkt noch Epoch 97/120).
- Kopiert → `/mnt/data/kip_pose/data/anker_seg/best.pt`, **byte-identisch**
  (sha256 `fec0124f9c576104e9101d78c2f53f4d7b0efbb07d182f1bb485335422c9ff93` auf beiden).
- **names-Gate:** `YOLO(best.pt).names = {0:'anker_kurz', 1:'anker_lang'}` (genau 2 Klassen),
  `task=segment`. CPU-only geladen (`CUDA_VISIBLE_DEVICES=""`), keine GPU vom Training/Worker geklaut.

## 2. Training gestoppt (GPU freigemacht) — PASS, Live intakt

- Prozessbaum verifiziert VOR Kill: tmux `yolo_seg_train` pane = bash `172009` →
  python `172010` (`box_src/train_yolo_seg_anker.py ... --epochs 120 --batch 6`). Live-Worker
  `84836` (`zahnrad...`) ist ein SEPARATER Baum.
- 🚨 Guard-Bus-Post + Inbox-Read (kein Veto) → **`tmux kill-session -t yolo_seg_train`**
  (nur dieser Baum, kein `pkill`-Pattern).
- **Verifik:** GPU **13845 → 1693 MiB** (12 GB Training frei). PID 172010 weg, tmux-Server weg.
  kip-server :8077 `active`/health-ok (`gpu_training_active:false`), kip-worker :8078 loopback
  listening, Worker-PID 84836 (1684 MiB) unberührt. **Kein Kollateralschaden.**

## 3. Konsolidierungs-Merge (lokal, integration) — PASS

Basis `team/multipipe/integration` (`22220df`, hatte S-002/003/006/sam3-svc). Additiv gemergt
(`--no-ff`, dependency-order):

| Branch | Konflikt | Auflösung |
|---|---|---|
| S-004 (gdrnpp-svc + mesh smoke) | — | clean |
| S-005 (yolo-obb-svc) | docker-compose.yml (Kommentar-Wording) | trivial, HEAD behalten |
| S-008 (yolo-seg trainer) | — | clean |
| S-010-011 (FE gating + batch-tab) | — | clean |
| S-013 (kip_server combo-proxy) | — | clean |
| S-012 (eval runner + combos-pivot) | kip_server.py `_LIVE_ROOT` (identischer KIP_LIVE_ROOT-Guard) | trivial, HEAD behalten |

- Vor Merge: untracked `project/mesh/yolo-obb-svc/` (byte-id zu S-005, Leftover-Kopie) +
  `assets/meshes`-Frage → yolo-obb-svc-Leftover non-destruktiv nach
  `/tmp/pre-merge-backup-T151/` verschoben (hätte S-005-Merge geblockt).
- **Test-Suite (`.venv` py3.12):** `214 passed, 7 skipped, 1 failed`. Der 1 Fail = bekannter
  **T-088 TTA-Flake** (`test_tta_wrapper_recovers_true_rotation`, `1.2074e-6 < 1e-6`,
  Floating-Point-Toleranz an 6. Nachkommastelle, agg=medoid „0.0000° off"). Ravi byte-id
  pre-existing → **kein Regress.** kip_server.py syntax OK.
- **Latenter Merge-Bug gefangen** (siehe §6): gateway-env `YOLO_OBB_URL` doppelt definiert
  (S-004 hardcoded + S-005 env-overridable) → `docker compose config` reject (strict YAML
  dup-key). De-duped auf env-Form. Per-Branch-pytest fängt das nicht (kein compose-parse).
- integration HEAD nach Merge + Fixes: **`7b909ce`**. **NICHT nach main.**

## 4. Security-Hardening M1 + M2 — applied (commit `f398569`)

- **M1 (compose):** Die 5 Seg/Pose-Services (yolo-svc:8001, yolo-obb-svc:8011, fp-svc:8002,
  gigapose-svc:8003, sam3-svc:8004) — `ports:` → **`expose:`** (nur compose-net, kein Host-Port;
  Gateway erreicht sie per Service-Name). Gateway → **`127.0.0.1:${GATEWAY_PORT:-8000}:8000`**
  (loopback statt 0.0.0.0; Box hat kein forward_auth). `docker compose config` valid, 6 services.
- **M2 (kip_server.py:1690 `/api/eval/result/{run_id}`):** `run_id` gegen
  `[A-Za-z0-9_.-]+` validiert **VOR** `batch_eval.load_run` → path-traversal zu (HTTP 400 bei
  ungültig). Identisches Pattern wie `live_frame` (Z. 506). pytest danach unverändert
  214 passed → kein Regress.

## 5. Box-Deploy (all-resident) — PASS

- Mesh per `rsync -az --delete --exclude .env` von `project/mesh/` → `~/kip_mesh` (rsync-copy,
  kein git). gateway-Dir + yolo-obb-svc neu auf der Box gelandet.
- **Box `.env` (Backup `.env.bak-T151`):** `YOLO_WEIGHTS_PT=/mnt/data/kip_pose/data/anker_seg/best.pt`
  (das TRAINIERTE Modell, nicht mehr `yolo26n.pt`-nano!), `YOLO_OBB_WEIGHTS_PT=.../detector_armvis/detector.pt`,
  `HF_CACHE_DIR=/mnt/data/kip_pose_weights/hf_cache`, `SAM3_CONF=0.2`,
  `GATEWAY_BIND=127.0.0.1`, **`GATEWAY_PORT=8090`**.
- **Port-Kollision behandelt:** `:8000` ist auf der Box von **`civion-api`** (PID 131142,
  unrelated, + `civion-api-redis-1`) belegt → mesh-gateway auf **`127.0.0.1:8090`** gebunden
  (additive `GATEWAY_PORT`-Var, default bleibt 8000). Civion **nicht angefasst.**
- S-006-Prereqs verifiziert: FP `mycpp.*.so` gebaut, GigaPose templates kip2 gerendert,
  FP weights-Mountpoint vorhanden.
- `docker compose up -d --build` (gateway + yolo-obb-svc neu gebaut, 4 andere aus vorhandenen
  Images) → alle 6 Container Up. **gdrnpp-svc:8012 nativ** via box-eigenes gdrnpp-venv
  (`GDRNPP_MEMFRAC=0.30`, korrigierter PYTHONPATH `box_src=/mnt/data/kip_pose/box_src`, da
  `run_box.sh` PROJECT_ROOT-Resolution nicht zur `~/kip_mesh`-Box-Layout passt).

**Health (gateway-aggregate `127.0.0.1:8090/health` → `ok:true`):**

| Service | Port (intern) | Status |
|---|---|---|
| yolo-svc (seg, **best.pt**) | 8001 | ok |
| yolo-obb-svc | 8011 | ok |
| fp-svc (foundationpose) | 8002 | ok (anker_kurz/lang, cuda) |
| gigapose-svc | 8003 | ok (kip2, refiner on) |
| sam3-svc | 8004 | ok |
| gdrnpp-svc (nativ) | 8012 | ready (obj 1,2, cuda) |
| **gateway** | **127.0.0.1:8090** | **ok:true (aggregate)** |

**Smokes:**
- **yolo-seg `/segment` (best.pt) auf SDG-Anker-RGB:** **3 echte Anker-Masken** —
  2× `anker_lang` (conf 0.976 / 0.961) + 1× `anker_kurz` (conf 0.940), je ~3 KB Mask-PNG.
  → Der Kern-Win: vorher (stock nano, S-006) **0 Masken**, jetzt echte Klassen-getrennte Anker.
- **gateway `/predict` E2E** (BOP-val-Scene 000000, seg=yolo, pose=gigapose_rgbd):
  1 `anker_kurz` conf 0.95 + `T_cam_obj` + pointcloud + timings. **Voller Mesh-Wire grün.**
  (t_cam-Magnitude ist Pose-Qualitäts-Frage für den Eval, kein Deploy-Block — Wire funktioniert.)

**VRAM all-resident:** **8705 / 24576 MiB** (große Reserve). Compute-Procs: Live-Worker 84836
(1684) + gdrnpp 185444 (1098) + 4 docker-svc-Procs. < 24 GB locker erfüllt.

**Live durchgehend intakt:** kip-server :8077 `active` (`status:ok`,
trained_objects `[anker_kurz,anker_lang,zahnrad]`, `frontend_present:true`), kip-worker :8078
listening. Kein `pkill`, kein Civion-Touch.

---

## 6. Ehrliche Blocker / Stolpersteine

1. **[gefangen+gefixt] Merge-Collision `YOLO_OBB_URL` doppelt** (S-004 + S-005) → `docker compose
   config` strict-YAML-reject. De-duped auf env-Form, commit `7b909ce`. *Lesson:* per-Branch-pytest
   parst kein compose → `docker compose config` gehört in die Merge-Verifik wenn mehrere Branches
   denselben Service-env touchen.
2. **[gefangen+gefixt] rsync `--delete` löschte FP-Meshes.** `~/kip_mesh/assets/meshes/anker_*.obj`
   sind box-lokale S-006-Setup-Kopien (aus `GigaPose/datasets/kip2/models/obj00000{1,2}.obj`),
   **nicht im Repo** → `--delete` entfernte sie → fp-svc crashte (`ValueError: string is not a file:
   /assets/meshes/anker_kurz.obj`). Fix: `chown max` + re-cp aus GigaPose-Source → fp-svc up.
   *Lesson:* `assets/meshes` ist **box-state**, künftig `rsync --exclude assets/meshes` ODER die
   obj-Kopie als Deploy-Step skripten.
3. **Port-Kollision `:8000` = civion-api** (unrelated Projekt, läuft + Redis-Container). Gelöst via
   `GATEWAY_PORT=8090`. *Wichtig für Eval:* gateway ist auf **`http://127.0.0.1:8090`**, nicht :8000.
4. **Kein Block:** T-088 TTA-Flake (1 Test, FP-Toleranz, pre-existing). Pose-Translation-Magnitude
   im `/predict`-Smoke groß — Pose-Genauigkeit ist Eval-Sache, nicht Deploy.

---

## 7. Was ist LIVE (Input für 12×20-Eval)

**Eval-Target-Gateway:** `http://127.0.0.1:8090` (NICHT :8000 — civion).
`batch_eval.http_predict(gateway_url=...)` / CLI `--gateway http://127.0.0.1:8090`.

Alle 4 Modell-Typen resident + ansprechbar:
- **seg-Quellen:** yolo-seg (**best.pt**, 2 Klassen, echte Masken), yolo-obb (OBB→combo1/Pipeline A),
  sam3 (klassen-approx, YOLO bleibt Klassen-Quelle), gt.
- **pose-Quellen:** foundationpose (rgbd), gigapose_rgbd, gigapose_rgb, gdrnpp (nativ:8012, Pipeline A).
- → Die ~12 feasible Kombis (S-012-Pivot, FEASIBLE=12 / recommended=7) sind real ausführbar.

**Rollback:** `cd ~/kip_mesh && docker compose down` stoppt die 6 Container; gdrnpp-svc nativ killen
per PID (185444, NICHT pkill). Live :8077/:8078 davon unberührt (eigene systemd-Units). `.env.bak-T151`
für env-Rollback.

**Git:** integration HEAD `7b909ce` (5 Merge-Commits + M1/M2 + 2 compose-Fixes). **NICHT nach main**
— wartet das E2E-Gate S-016. main-Merge ist ein separater autorisierter Schritt.

---
*Bus: `done` gepostet, presence done. Kanban T-151 alle Schritte geloggt (5× success, 2× warn).*
