# Deploy Report — pose-tune + groundclamp (2026-06-02)

**Agent:** Sam (DevOps) · **Mode:** B (Merge Coordinator + Deploy)
**Gate:** Ravi PASS x2 (T-094 + T-095) · Queen pre-authorisiert (bus 13:28)

## Merge

| Ticket | Branch | Source SHA | Merge Commit | Files |
|---|---|---|---|---|
| T-094 | team/2026-06-02-pose-tune | 0d3aeeb | **ae206e8** (`--no-ff`) | project/kip_server.py (conf 0.3→0.1) |
| T-095 | team/2026-06-02-groundclamp | b07662e | **475b82a** (`--no-ff`) | project/frontend/src/scene.js (groundClamp) |

- Topologie: `groundclamp` baut linear auf `pose-tune` (0d3aeeb gemeinsam, kip_server.py identisch). Zweiter Merge brachte nur b07662e (scene.js) neu → **kein Konflikt**, 2 saubere no-ff Merge-Commits.
- Pre-merge chore: b4cc1f5 (live board state).
- **Push:** `ccdbe55..475b82a` → origin/main (kein force).
- Tests: `pytest test_refine_rc.py` 33/33 grün. `node --check scene.js` OK. (T-088 tta-Flake: andere Datei, nicht berührt.)

## Deploy — Box max@100.85.216.95 (/mnt/data/kip_pose/project), VPN aus, Box AN

### T-094 kip_server.py — ganze Datei (scp)
- Hash-Check: Box `kip_server.py` == ccdbe55 (`18e183b9…`) → **kein Box-eigener Hotfix**.
- → scp main HEAD (`ffd9b70…`) als ganze Datei. Box-Hash == main HEAD verifiziert.
- Effektiv: `conf=0.1` in `_run_detector` (war 0.3). Backup `.bak-tune`.

### T-095 scene.js — CHIRURGISCH (Drift-Falle gemeistert)
- Box-scene.js (`cf22dc24…`, 400 Z.) DIVERGENT von main-base UND main-HEAD.
- **Drift bestätigt:** Box hatte (a) cell-loader-Hotfix im `loadCell`-Bereich (Box Z.80-141, +31 Z. vs repo, 6 Treffer) NICHT im Repo, UND (b) einen eigenen groundClamp-Vorgänger (bidirektionaler snap, `tableZ===-Infinity→0`, `Math.abs(dz)>0.0005`, 3x3-MAX-Korridor) — ebenfalls nicht im Repo.
- **Analyse:** Lenas neue groundClamp subsumiert den Box-groundClamp (gleiche Konzept-Linie, weiterentwickelt: Center-Anker statt Korridor, MEDIAN statt MAX, `_MAX_CLAMP=0.6` Guard, bidirektional bleibt). Kein einzigartiges Box-groundClamp-Feature geht verloren. Ravi 8/8 gated.
- **Operation:** Box-Block `function groundClamp(){…}` (Z.213-277) ersetzt durch Lenas Block (HEAD Z.206-283: `_MAX_CLAMP` + `_median` + `_surfacesAt` + neue `groundClamp`). `_raycaster`/`_down` (Box Z.211-212) UNBERÜHRT (Lena reused outer). cell-loader + alles andere der Box erhalten.
- Verify merged file: node --check OK, cell-loader 6/6, je 1× Deklaration (keine Dupes), Seams sauber. Box-Hash == lokal-merged (`984d4ca…`). Backup `.bak-tune`. Static-FE → kein Restart.

## Restart + Live-Health (nach `systemctl restart kip-server.service`)

| Check | Ergebnis |
|---|---|
| is-active | **active** |
| /api/health | **ok** (frontend_present:true, trained_objects: anker_kurz/anker_lang/zahnrad) |
| /api/live/status | **HTTP 503 graceful** (Jetson unreachable, VPN aus — erwartet; `-m 5` gab vorher 000 weil Jetson-Poll ~5.007s, mit 10s sauber 503) |
| FE index tab-live + cell-loader | **3** (≥2 ✓) |
| served /src/scene.js groundClamp / loadCell / _MAX_CLAMP | **2 / 3 / 2** (neue groundClamp + cell-loader-Wiring live ✓) |

**Keine Regression. Kein cell-loader-Verlust. Kein Rollback nötig.**

## Rollback (falls je nötig)
```
ssh max@100.85.216.95 'cd /mnt/data/kip_pose/project && \
  cp kip_server.py.bak-tune kip_server.py && \
  cp frontend/src/scene.js.bak-tune frontend/src/scene.js && \
  sudo -n systemctl restart kip-server.service'
```

## Constraints eingehalten
- Jetson TABU: nur read-probe via /api/live/status (graceful fail, VPN aus). Kein Eingriff.
- Kein force/reset/--no-verify. Push nur main (Queen OK).
- Box AN gelassen (Live-Site, Max arbeitet weiter). Kein Shutdown.
