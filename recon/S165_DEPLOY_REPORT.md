# S165 — SAMMEL-DEPLOY 2 REPORT (T-165, Sam)

Session `2026-06-07-multi-pipeline-pose` · Mode B (Merge + Deploy, KEIN Rebuild) · 2026-06-08
Box `max@100.85.216.95` (maxgpuserverobk, RTX 3090, 24 GB)

**TL;DR:** T-163 (AR-Fix) + T-140 (Sim/Upload-Combo-Routing) + T-164 (FE „Inferiert mit")
sauber in `team/multipipe/integration` gemergt (0 Konflikte, keine File-Überlappung). 7 Files
auf die Box deployed (kip_server.py, pipelines/gateway_proxy.py, eval/batch_eval.py + 4 FE),
**KEIN Container-Rebuild** (keine Mesh-`app.py` berührt), `sudo systemctl restart kip-server.service`.
**Verify live belegt:** Non-A-Sim-Routing wirft **kein „Routing folgt"-501** mehr (job startet,
`used_combo/used_seg/used_pose/modality` im Result, error=None, Phasen laufen); **Pipeline A intakt**
(Worker-Pfad, `meta.source=worker`); **FE „Inferiert mit"-Zeile live** (pipeline.js served sha
`2f73fb18`, 2× „Inferiert mit", 20× `used_*`). `:8077`/`:8078`/`:8090` health ok, 12/12 Kombis.
**integration-HEAD `3a941334e410de31459de89bd7e96671a6a8733f` — NICHT nach main.**
**Startklar für Kais Re-Run** (Gateway `http://127.0.0.1:8090`, `batch_eval.py` frisch mit AR-Fix).

---

## 1. Merge → integration — PASS

Basis `team/multipipe/integration` `6ec482e` (Stand nach S161). Dependency-/reihenfolge-neutral
(keine File-Überlappung), `--no-ff`. board.json + untracked recon-Reports + glb stören nicht
(Nicht-Code, kein Merge-Konflikt).

| # | Branch | Commit | Inhalt | Files | Konflikt |
|---|---|---|---|---|---|
| 1 | T-163 | `11540c2` | AR-Fix: Eval-Pfad kein `planar_z_snap` | `eval/batch_eval.py` + test | — clean |
| 2 | T-140 | `4262804` | Sim/Upload routen alle Kombis (NICHT-A übers Gateway) | `kip_server.py`, `pipelines/gateway_proxy.py` + test | — clean |
| 3 | T-164 | `05a86cb` | FE „Inferiert mit <seg>+<pose> (<modality>)" | 4× FE (`kip.html`, `src/{kip.css,kip.js,pipeline.js}`) + smoke | — clean |

**Kein textueller Konflikt** — die drei Branches treffen disjunkte Files (T-163 Eval, T-140
Backend, T-164 Frontend). Erwartete Überlappung T-163↔T-140 (`batch_eval.py` vs `kip_server.py`)
existiert nicht — verschiedene Module. **Alle Änderungen behalten.**

**Tests** (`.venv` py3.13, `pytest project/tests/ -q`): **263 passed, 7 skipped, 1 failed.**
Der 1 Fail = bekannter **T-088-TTA-Flake** (`test_tta_wrapper_recovers_true_rotation`,
agg=medoid `1.2074e-06 < 1e-06`, FP-Toleranz an 6. Nachkommastelle, „0.0000° off"). Pre-existing,
kein Merge-Regress. Neue Tests grün: `test_sim_real_routing.py` (T-140), `test_batch_eval.py` (T-163).

**integration-HEAD: `3a941334e410de31459de89bd7e96671a6a8733f`** (3 Merge-Commits). NICHT nach main.

---

## 2. Deploy (kein Rebuild) — PASS

Box-Live-Root `/mnt/data/kip_pose/project` (WorkingDir kip-server.service). **Kein** Mesh-Service
(`fp/gigapose/gateway/yolo/sam3`) `app.py` geändert → **kein `docker compose build`**, Container
unberührt (`Up 2–4 h`).

**Box-Backup ZUERST:** `/mnt/data/kip_pose/project/.bak-T165-20260608_132450/` (alle 7 Files, für Rollback).

**Drift-Check:** Box-sha aller 7 Files == sha bei `6ec482e` (vorherige integration-Basis) →
**kein Box-Drift, sauberes Overwrite von bekannter Basis.**

Deployed (alle byte-identisch zu integration-HEAD `3a94133`, post-scp sha-Match auf der Box verifiziert):

| File | sha (neu) | sha (alt = 6ec482e) | Quelle |
|---|---|---|---|
| `kip_server.py` | `63f70b4f` | `b3973f00` | T-140 |
| `pipelines/gateway_proxy.py` | `eff70c95` | `500c4a31` | T-140 |
| `eval/batch_eval.py` | `5120df0b` | `d7cb5eb9` | T-163 (Kai braucht's frisch) |
| `frontend/kip.html` | `44cbcef8` | `ee50aa55` | T-164 |
| `frontend/src/kip.css` | `7cd78b49` | `e58464cd` | T-164 |
| `frontend/src/kip.js` | `aab73b4e` | `8e9c832f` | T-164 |
| `frontend/src/pipeline.js` | `2f73fb18` | `0ddda9c0` | T-164 |

**Restart:** `sudo systemctl restart kip-server.service` (KEIN pkill). NRestarts **0→0** (kein
Crash-Loop), kip-server `active`. **Worker `:8078` durchgehend `active`, pid 84836 unverändert**
(separater systemd-Baum, komplett unberührt). Mesh-Stack unberührt.

---

## 3. Verify (Belege)

### Beleg 1 — Non-A-Sim-Routing: KEIN „Routing folgt"-501 mehr — PASS
`/api/sim/generate_async` mit einer Nicht-A-Kombi, beide Param-Wege:

| Aufruf | Antwort |
|---|---|
| `?seg=sam3&pose=foundationpose` | `{"job":"2ff84b1e","used_combo":"sam3__foundationpose","used_seg":"sam3","used_pose":"FoundationPose","modality":"RGBD"}` |
| `?pipeline=sam3__foundationpose` | `{"job":"a730958a", ... identisch ... }` |

Job-Poll (`/api/sim/job/2ff84b1e`): `phase="Isaac Sim startet (booting)"`, `pct=10`, **`error=None`**
über mehrere Polls — der Job **läuft** (Isaac-Render gespawnt → Pose-Stage übers Gateway), **fällt
NICHT in den 501-Stub**. Result trägt `used_combo/used_seg/used_pose/modality`.

- **Gültige Kombi** `seg=sam3&pose=gdrnpp` → HTTP 200 (feasible).
- **Image-only-Guard** `seg=gt&pose=foundationpose` → **HTTP 400** (GT-Masken im interaktiven Sim
  nicht erlaubt, nur Batch-Eval).

> *Hinweis:* Volle Szene-Gen dauert ~60–80 s (Isaac-Boot+Render). Der Routing-Beleg braucht das
> nicht — `job`+Meta+`error=None`+Phasenfortschritt zeigen, dass die gewählte Kombi wirklich
> geroutet wird (gleiche Naht wie `/api/predict`, S-013). Die Test-Render-Subprozesse wurden nach
> dem Beleg sauber gestoppt (siehe §5).

### Beleg 2 — Pipeline A intakt (Worker-Pfad) — PASS
- `/api/sim/infer?scene=0`: `meta.source = "worker"`, 4 results (cached-scene Infer, Pipeline A
  hardcoded = heilig, unverändert).
- `/api/sim/generate_async?pipeline=gdrnpp` **und** leere Params → resolven beide zu
  `used_combo=gdrnpp, used_seg=yolo-obb, used_pose=GDRNPP, modality=RGB` (Worker-Pfad, **nicht**
  Gateway). Pipeline A byte-identisch im neuen Routing-Code.

### Beleg 3 — FE „Inferiert mit"-Zeile live — PASS
Served by `:8077`:
- `kip.html` sha `44cbcef8` (= deployed), enthält `inferred`-Anchor + „Inferiert"-Text.
- `pipeline.js` served sha `2f73fb18` (= deployed): **2× „Inferiert mit"**, **20× `used_seg/used_pose/modality/used_combo`**.
- `kip.js` served sha `aab73b4e` (= deployed).

→ Sim- **und** Real-Card zeigen „Inferiert mit: <seg> + <pose> (<modality>)" aus `result.meta` mit Fallback.

### Service-Health (final, nach Cleanup)
- `:8077` `active`, HTTP 200, **12/12 Kombis available, seam=ok**, NRestarts 0.
- `:8078` worker `active`, HTTP 200, pid 84836 (unverändert über den ganzen Deploy).
- `:8090` gateway `ok:true`, alle 6 Knoten (yolo, fp, gigapose, sam3, yolo_obb, gdrnpp).
- Mesh-Stack: alle 6 Container `Up 2–4 h` (nie angefasst — kein Rebuild).
- GPU final **13276 / 24576 MiB**.

---

## 4. Was ist LIVE / Startklar für Kais Re-Run

- **Eval-Target-Gateway:** `http://127.0.0.1:8090` (NICHT :8000 = civion-api). Alle 6 Knoten ok.
- **`batch_eval.py` frisch auf Box** (`/mnt/data/kip_pose/project/eval/batch_eval.py`, sha
  `5120df0b`, byte-identisch zu integration `3a94133`) — **AR-Fix drin** (Eval-Pfad `snap=False`,
  `planar_z_snap` nur noch Live-Viewer-Cosmetic). Erwartung Pipeline A 2-Klasse ~0.9 (T-109-Baseline).
- **kip_server frisch neugestartet** → Modul-Cache-frei, lädt `batch_eval` frisch beim Re-Run.
- **FE live auf `/KIP`:** Sim-Tab fährt jede Kombi (kein 501-Stub), „Inferiert mit"-Zeile auf
  Sim+Real-Card, 12 Kombis.
- **VRAM-Hinweis an @kai-ml (aus S161, weiter gültig):** fp-svc-Kombis können bei vollem Pool OOMen
  — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` oder fp-Kombis sequenziell.

**Rollback:** `/mnt/data/kip_pose/project/.bak-T165-20260608_132450/` (alle 7 Files) zurückkopieren
+ `sudo systemctl restart kip-server.service`. Mesh + Worker davon unberührt.

---

## 5. Stolpersteine (ehrlich)

1. **Test-Renders saturieren die GPU:** Meine 5 `generate_async`-Verify-Calls haben je einen
   `gen_sdg_arm_visible.py` (Isaac SimulationApp ~25 s Boot + Render, ~3 GB) gespawnt — parallel
   → GPU kurz auf ~19.6 GB, `:8077`/`:8090`-curls liefen 1× in den 8-s-Timeout. **Kein
   Deploy-Regress** — reiner Last-Spike durch meine Tests. Render-Subprozesse nach dem Routing-Beleg
   per `kill -TERM` (NUR die `gen_sdg`-PIDs, nicht kip-server/worker/mesh) gestoppt → GPU zurück auf
   13.3 GB, alle Ports wieder responsiv, 12/12 Kombis. *Lesson:* für den 501-Beleg reicht EIN
   non-A-`generate_async` + Job-Poll; nicht 5 parallel feuern, das Isaac-Boot ist GPU-teuer.
2. **`/api/pipelines` `available` spiegelt Gateway-Health:** Während der GPU-Saturation pollte
   kip_server `_gateway_health()` in den Timeout → `available` fiel kurz auf 1/12. Nach
   Render-Stop + Gateway-Retry (HTTP 200) sofort wieder 12/12. Kein Code-Problem — transiente
   Last, das Verhalten (Gateway down → Kombis degraded) ist korrekt/gewollt.

---

## 6. Git

**integration-HEAD `3a941334e410de31459de89bd7e96671a6a8733f`** (3 Merge-Commits: f880f3a T-163,
69a9b0b T-140, 3a94133 T-164). **NICHT nach main** — main-Merge ist ein separater autorisierter
Schritt (nach Kais Re-Run + Ravi-Gate).

---
*Bus: status (vor Restart + Render-Cleanup) + done gepostet, presence done. Kanban T-165 alle
Schritte geloggt. Theo D-2 (nur batch_eval redeployen) befolgt + per Mehr-Datei-Deploy erfüllt.*
