# S172 — ABSCHLUSS / FINAL GO-LIVE REPORT (T-172, Sam)

Session `2026-06-07-multi-pipeline-pose` · Mode B (Merge + Deploy + main-Merge) · 2026-06-09
Box `max@100.85.216.95` (maxgpuserverobk, RTX 3090, 24 GB) · Repo `github.com/MaxKlat29/KIP_POSE`

**TL;DR:** Restliche in_review-Branches (T-166/167/169/171) sauber in `team/multipipe/integration`
gemergt (0 echte Konflikte, `batch_eval.py` auto-merged — T-167 frames + T-171 AR-2-Klassen
koexistieren). Gate `pytest 266 passed / 7 skipped / 1 known-flake (T-088)`. 3 geänderte Runtime-
Files auf die Box deployed (Backup `.bak-T172`), `kip-server` restart (NRestarts=0, **Worker :8078
PID 1189 unberührt**), KEIN Container-Rebuild. Verify live: **gdrnpp AR = 0.8861 (2-Klassen, NICHT
0.295)**, Run als **Datum/Uhrzeit `2026-06-09 00:02`**, FE-Tabelle ohne Verfahren-Spalte/Badges,
kein Live-Tab, Sim-Routing intakt, Pipeline A heilig. **`integration` → `main` gemergt (`--no-ff`),
gepusht nach origin/main.**

**main-Merge-SHA: `2c25079080abfb736f1ff339147ef22d4abffa20`**
**Push-Status: PUSHED (`2a77580..2c25079 main -> main`, fast-forward, KEIN force)**

---

## 1. Konsolidierung → integration — PASS

Basis `team/multipipe/integration` `3a94133` (Stand nach S165). Merges in Dependency-Order, `--no-ff`:

| # | Branch | Merge-Commit | Inhalt | batch_eval? | Konflikt |
|---|---|---|---|---|---|
| 1 | T-166 | `88d0934` | Symmetrie-Untersuchung (nur recon-Docs) | — | clean |
| 2 | T-167 | `1b0edf8` | gigapose ICP max_corr=0.025 + RGBD-Refiner-Tuning + `discover_scenes(frames=)` (T-170 D-5) | ja (untere Hälfte) | clean |
| 3 | T-171 | `f1a2a0c` | AR=2-Klassen (`active_class_parts`/`_mean_active`/`ar_6obj`) + `rerender_run.py` | ja (obere/mittlere Hälfte) | **auto-merged** |
| 4 | T-169 | `032b8e1` | FE Datum/Uhrzeit-Run-Dropdown (`runLabel`) | — (nur batch.js) | clean |

**`batch_eval.py`-Auflösung:** T-167 und T-171 gehen beide von base `516f3e5` aus, treffen aber
**disjunkte Funktionen** — T-167 = `discover_scenes` (Z. 934), T-171 = `ar_from_report`/
`aggregate_config`/`_ConfigAcc`/`render_markdown` (Z. 411–890). Git `ort` 3-way mergte automatisch
(`Auto-merging project/eval/batch_eval.py`, **0 Konflikt-Marker**). **ALLE Änderungen behalten** —
verifiziert: `frames=None` (T-167) UND `active_class_parts`/`ar_6obj` (T-171) koexistieren, AST OK.
Die ganze Kette (ICP-config + AR-2-Klassen + modality + depth_scale + pipeline-A-routing) intakt.

**integration-HEAD nach Merges: `032b8e1`.**

### T-147 — bewusst NICHT gemergt
`git branch --no-merged` zeigte zusätzlich `team/multipipe/T-147`. **Nicht gemergt** (begründet):
- Sein Commit `b57dda0` **IST der merge-base** von integration → Inhalt (FE-Gating-Relax 12-Matrix)
  ist längst über spätere Tickets eingeflossen. Die Branch-Spitze `ba5d893` ist nur eine veraltete
  story-note.
- Ein naiver Merge würde **30 Commits** (T-153…T-171) rückabwickeln: u.a. den AR-2-Klassen-Fix, den
  `kip_server.py`-Refactor (−366 Z.) und die Entfernung des separaten `live.js`-Scoreboards.
- `live.js` ist in HEAD korrekt **abwesend** (kein separater Live-Tab — Briefing-Anforderung).
→ Worktree + Branch T-147 stehen gelassen (nicht-destruktiv).

## 2. Gate (pytest) — PASS

`.venv` (pytest 9.0.3) · `python -m pytest project/tests/ -q`:
**266 passed · 7 skipped · 1 failed.**

Der 1 Fail = bekannter **T-088-TTA-Flake** `test_tta_wrapper_recovers_true_rotation`
(agg=medoid `1.2074e-06 < 1e-06`, „0.0000° off" — float64-Toleranz-Schranke 12× zu eng).
Deterministisch, pre-existing, **kein Merge-Regress** (`test_tta_pose.py` von keinem gemergten
Branch berührt). Backlog-Ticket T-088 (P3 tech-debt). **Gate grün.**
Auf `main`-HEAD nach Merge erneut bestätigt: **266 passed / 7 skipped / 1 known-flake** (Parität).

## 3. Deploy (kip-server-Restart + FE, KEIN Rebuild) — PASS

Box-Live-Root `/mnt/data/kip_pose/project`. **Drift-Check:** 5 Runtime-Files (`kip_server.py`,
`gateway_proxy.py`, `pipeline.js`, `kip.js`, `kip.html`) waren Box-sha == integration-sha →
**kein Drift, nicht angefasst**. Nur die geänderten Files deployed:

**Backup ZUERST:** `/mnt/data/kip_pose/project/.bak-T172-20260609_002411/` (alte batch_eval.py
`31d8c9d6`, batch.js `a950fe0e`, run_final_t170.py, tune_rgbd.py).

| File | sha (deployed) | sha (alt Box) | Quelle |
|---|---|---|---|
| `eval/batch_eval.py` | `7075c228` | `31d8c9d6` | T-167+T-171 |
| `frontend/src/batch.js` | `51379ce1` | `a950fe0e` | T-169 |
| `eval/rerender_run.py` | `04d4f217` | (fehlte) | T-171, additiv |

`run_final_t170.py`/`tune_rgbd.py` waren bereits byte-identisch (`a195dbf1`/`647dacc2`) → nicht
re-deployed. Post-scp Box-shas == local integration-HEAD (verifiziert). Das re-renderte
`results.json`/`EVAL.md` (T-171, run `run-20260608T201857Z`) lag schon auf der Box
(`temp/batch_eval/`).

**Restart:** `sudo systemctl restart kip-server.service` (**NIE pkill**). Vorher AST-Check der
deployten batch_eval.py mit Service-Python (`/mnt/data/isaacsim-venv/bin/python`) — OK.
kip-server `active`, **NRestarts=0** (kein Crash-Loop), MainPID 45211 (neu).
**Worker `:8078` MainPID 1189 vorher == nachher** (separater systemd-Baum, komplett unberührt).
Mesh-Stack nie angefasst.

## 4. Verify (Belege) — PASS

| Beleg | Endpoint / Quelle | Ergebnis |
|---|---|---|
| **AR 2-Klassen** | `/api/eval/result/run-20260608T201857Z` | gdrnpp(RGB) `ar_mean=0.8861` **(NICHT 0.295)**; `ar_6obj=0.2953` sekundär (Transparenz). 12 Configs, sam3 0.21–0.41. |
| **Datum/Uhrzeit-Run** | `/api/eval/runs` | `run-20260608T201857Z @ date 2026-06-09T00:02:42Z` → FE-Label „2026-06-09 00:02". 1 Run (= Default/neuester). |
| **/api/pipelines = 12** | `/api/pipelines` | `count: 12`, `seam` vorhanden. |
| **FE-Tabelle clean** | served `batch.js` sha `51379ce1` | `COLS = Konfiguration · Input(RGB/RGBD) · AR IC-BIN · Laufzeit · Abdeckung · Absturz`. **Keine** Verfahren-Spalte, **keine** Badges (T-158). `runLabel` (T-169) aktiv. |
| **Kein Live-Tab** | `kip.html` served | 0 separate Live-Tab-Nav-Buttons; `eval-live-tbl` = In-Batch-Standings (gewollt, kein eigener Tab). `live.js` absent. |
| **Sim-Routing non-A** | `/api/sim/generate_async?seg=sam3&pose=gdrnpp` | `job=16071262, used_combo=sam3__gdrnpp, used_seg=sam3, used_pose=GDRNPP, modality=RGB, error=None` — **kein „Routing folgt"-501**. |
| **Pipeline A intakt** | `/api/sim/generate_async?pipeline=gdrnpp` | `used_combo=gdrnpp, used_seg=yolo-obb, used_pose=GDRNPP, modality=RGB` (Worker-Pfad, heilig). |
| **Service-Health** | `:8077` `/api/pipelines` HTTP 200 · `:8078` worker `active` PID 1189 · `:8090` gateway `{ok:true}` (yolo/fp(cuda,anker_kurz/lang)/gigapose(refiner)/sam3) · `/KIP` Edge HTTP 200. |

## 5. → main — PASS

`main` (`2a77580`) war **direkter Vorfahr** von integration (keine Divergenz, integration 52 Commits
voraus). `git checkout main && git merge team/multipipe/integration --no-ff` → sauberer Merge-Commit.

- **main-Merge-SHA: `2c25079080abfb736f1ff339147ef22d4abffa20`**
- main-HEAD enthält AR-2-Klassen (7 Marker) + T-169 batch.js (verifiziert), pytest-Parität 266 pass.
- `origin/main` war noch `2a77580` (= erwarteter Vorfahr) → **fast-forward Push, KEIN force**:
  `2a77580..2c25079  main -> main`. origin/main == local main == `2c25079`.

**Cleanup:** 18 vollständig-gemergte Worktrees `git worktree remove` + lokale Branches `git branch -d`
(safe-guard, nur voll-gemergte). **Behalten:** `team/multipipe/T-147` (stale, nicht gemergt — Worktree
+ Branch) + `team/multipipe/integration` (Konsolidierungs-Branch).

## 6. Blocker / Ehrlichkeit

- **Kein harter Blocker.** Go-Live vollständig.
- **T-088-TTA-Flake** bleibt offen (P3, bekannt, kein Block) — Assert-Schranke `1e-6` ist für die
  medoid-Float-Akkumulation zu eng; mathematisch 0.0000° korrekt. Fix = Schranke auf `5e-6` o.ä.
  lockern (separates Ticket).
- **T-147** absichtlich nicht gemergt (begründet oben) — Worktree/Branch stehen gelassen statt
  verworfen (nicht-destruktiv). Falls erwünscht, kann Max ihn löschen.
- `ar_6obj` (alte 6-obj-Zahl) bleibt bewusst als Sekundärfeld in `results.json` — Transparenz, nicht
  primär gerendert.

**Live & final: integration → main `@2c25079`, gepusht, AR 0.886, FE clean, Services gesund.**
