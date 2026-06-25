# QA-Gate — Wave 2b (Ravi/QS) — Session 2026-06-07-multi-pipeline-pose

> Gate = **Tests + Correctness**. Security ist Brunos Revier (NICHT hier geprüft).
> Tickets: T-129 (S-003), T-131 (S-005), T-132 (S-006). **Verdikt: 3/3 PASS, 0 Blocker.**

---

## S-003 / T-129 — pipelines/ 2-stufig (Seg→Pose-Seam) — **APPROVED**

**Commit:** `1135043` (team/multipipe/integration) · **Diff:** +1061 LOC, rein additiv (6 neue Files).

### Review-Findings (Correctness)
- **Falle 1 (Coord-Frame §1) korrekt:** `composed.py` zerlegt `T_cam_obj` (Meter) und ruft
  `bop_pose_to_world(R_m2c, t_m2c_mm, ...)` mit `t_m2c_mm = T[:3,3]*1000.0` (METER→mm). Round-Trip
  `world_to_bop_cam(bop_pose_to_world(T)) == T` über **50 Zufalls-SO3, atol 1e-9** gegen die
  **ECHTEN Live-Funktionen** (`compare_pipelines` + `bop_adapter` als reale Imports). mm-Unit-Test
  separat (Identität 0.30 m, t_w2c 500 mm → t_world.z = -0.2-0.08 m). Kanonisierung NACH Welt-Mapping.
- **Falle 2 (Klassen-Mapping §6) korrekt:** `part_for_obj_id(CLASS_TO_OBJ_ID[d.cls])` statt direktem
  lowercase-Lookup. Antitest `PART_SYMMETRY.get("anker_kurz") is None` (die Falle) + CamelCase-Part hat
  `continuous`-Y-Symmetrie. Vorfilter `dets = [d for d in dets if d.cls in CLASS_TO_OBJ_ID]` dropt
  unbekannte Klassen still (§3-Skip).
- **Whitelist = 7 (nicht 12):** 6 NICHT-A als ComposedPipeline + Kombi 1 (gdrnpp = Pipeline-A-Monolith,
  NICHT als Composed gebaut). `needs_depth`-Regel pro Kombi gegen §5 verifiziert.
- **Autoload defensiv + idempotent:** `_autoload_combos()` beim `import pipelines` bricht nie; nach
  Re-Register bleibt Registry bei 7. `registry.get("gdrnpp")` ist weiterhin `GdrnppReferenceAdapter`
  (NICHT durch Composed verdrängt) — empirisch verifiziert.

### Tests
- `test_composed_pipeline.py`: **14/14 grün** (Round-Trip, mm-Unit, Klassen-Mapping+Antitest,
  Kanonisierung, needs_depth-Gating, pipeline-Flag, Whitelist=7, Skip-Verhalten, Helfer).
- Vollsuite `project/tests/`: **150 passed / 7 skip / 1 fail**. Der eine Fail
  (`test_tta_pose.py::test_tta_wrapper_recovers_true_rotation`, `1.2074e-6 < 1e-6`) ist der
  **pre-existing TTA-Float-Flake T-088** — byte-identisch zu `main`, NICHT im S-003-Diff. Kein Block.

### Regression (Live-Pfad)
8/8 Live-Files **byte-identisch zu main** (`kip_server`, `gdrnpp_adapter`, `pipelines/base`,
`contract`, `registry`, `compare_pipelines`, `bop_adapter`, `e2e_infer`). Diff = NUR additive
pipelines/-Files + Test. `pipeline=gdrnpp` unverändert.

### Security → Bruno
Rein additiv, lokal, stdlib-only Transport (kein `requests`-Pin), kein User-Input-Pfad geändert. Keine
neue Security-Oberfläche. Bruno-Review nicht erforderlich für diesen Diff.

**Gate: APPROVED.**

---

## S-005 / T-131 — yolo-obb-svc (3. Seg-Quelle, OBB→combo 1) — **APPROVED**

**Commit:** `abed283` (team/multipipe/S-005) · 7 Files (app/Dockerfile/req/smoke/README + compose/.env).

### Review-Findings (Correctness + Contract §2/§4/§6)
- **Contract-konform §2:** `/segment` → `detections[]` mit `id` (int, dense), `class` (lowercase ∈
  {anker_kurz, anker_lang}), `conf` float, `mask_b64` PFLICHT (rasterisiertes OBB-Quad, volle Bildgröße,
  0/255 via `fillPoly`), `obb` [cx,cy,w,h,θ rad] (`obb.xywhr`). `/health` → `{"ok":true}`.
- **2-Klassen-Filter (§6):** `cls_id >= 2` (incl. zahnrad=5) → `None` → `continue`. **Kein Zahnrad-Leak.**
- **Label-Mapping `obj_id=cls+1`:** das `id`-Feld ist korrekt der Detection-Counter (§2, NICHT obj_id);
  obj_id lebt in S-003 `CLASS_TO_OBJ_ID`. class-Strings == CLASS_TO_OBJ_ID-Keys → Naht passt.
- **Label-Drift-Defense:** Service folgt der per-Detection `r.names`-Map des Modells (statt blind
  positional) + fällt bei Müll-Label positional zurück + WARN-Log bei Order-Mismatch. Bewusste,
  dokumentierte Robustheit — **kein Bug.**
- **Empirisch bewiesen** (reine Filter/Label/Counter-Logik ohne ultralytics/GPU, 4 Szenarien):
  kein Zahnrad-Leak, dichter id-Counter (auch bei degenerate-box/filtered-skip), obj_id-Alignment,
  Drift-Defense + Junk-Fallback.

### Tests
- `smoke_box.py` testet den **echten `segment()`-Pfad** gegen Contract §2 (id==i, class∈2, conf 0..1,
  mask volle Größe + 0/255 binär, obb 5-float). **Box-Smoke war PASS: 5 dets** (2× kurz, 3× lang,
  classes 2-5 gefiltert). Code-Replik der Filter/Label-Invarianten: alle PASS.

### Build/Wiring
Dockerfile leichtgewichtig (`pytorch:2.5.1-cuda12.4-runtime` + ultralytics, sm_86-Wheel — kein
ampere-base nötig), libGL/glib für headless-cv2, YOLO_CONFIG_DIR auf /tmp. compose/.env sauber
(`yolo-obb-svc:8011`, RO-Weights-Mount, CONF 0.40 / IMGSZ 1280 matchen e2e_infer). YAML valid.

### Nits (kein Block)
- `@app.on_event("startup")` deprecated in neuerem FastAPI (funktional auf Box-Version).
- `MODEL=None` pre-startup — `/segment` vor Startup würde NoneType.predict werfen (in ASGI
  unkritisch: startup läuft vor Requests; smoke ruft load_model() explizit).

### Security → Bruno
Neuer HTTP-Service nimmt `rgb_b64` (+ ignorierte Felder) entgegen. Input ist base64-Bild
(`cv2.imdecode`), kein eval/shell/Path-Traversal. **Da neuer Service-Endpoint + Input-Boundary:
⚠️ falls noch nicht durch Bruno: Security-Review der /segment-Input-Boundary (b64-decode, model-load,
weight-path-env) empfohlen vor Merge.** Aus QS-Sicht: kein offensichtlicher Injection-Vektor.

**Gate: APPROVED** (Bruno-Hinweis vermerkt).

---

## S-006 / T-132 — yannics 3 Services (ampere + Box-Weights) — **APPROVED**

**Commit:** `db12aad` (team/multipipe/integration) · NUR 4 Files (3 Dockerfiles + compose), reines Build/Wiring.

### Review-Findings (Correctness)
- **Dockerfiles:** `:blackwell` → `:ampere` (sm_86 RTX 3090) — minimal, korrekt für fp/gigapose/yolo.
- **Mount-Logik korrekt:** fp-svc 3 RO-mounts (FP-repo + weights-overlay + meshes); gigapose 3
  (checkout + weights-store + bop-weights **am selben absoluten Pfad** → dangling-Symlinks resolven).
  mycpp-/Template-/Weights-Overlay-/COCO-Fallen ausführlich im compose dokumentiert.
- **depth_scale (BOP=0.1) korrekt platziert:** BEIDE Services teilen depth `/1000.0` (mm→m, Contract §3).
  `depth_scale=0.1` lebt korrekt im **Test-Harness/Daten-Layer** (BOP-PNG ist nicht mm), NICHT im
  Service. Kein Service-Bug — Report dokumentiert das ehrlich (Stolperstein 3).
- **gigapose pipeline-Gating:** `rgbd` → depth Pflicht + kabsch=True; `rgb` → kein depth/kabsch (§5). Korrekt.
- **CUDA-Context-Ordering:** lebt in `gigapose_infer.py` (gemounteter GigaPose-Checkout, §7) — **out of
  S-006-Diff** (db12aad ändert keine Service-Logik). Live verifiziert von Sam (`refiner=on`, ordering ok).
- **git-hunk-Trennung sauber:** S-006-compose committed nur Sams Hunks; Kais yolo-obb-Hunks additiv im
  working tree → working-tree-compose YAML valid mit 6 services, kein Konflikt.

### Tests (auf der Box, Sam)
- **fp-svc PASS:** trans_err 28–81 mm vs BOP-GT. **gigapose-svc PASS** beide Pipelines: rgbd 32–94 mm,
  rgb 95–439 mm vs GT (rot_err = continuous-Y-Symmetrie-Artefakt der Anker, kein Fehler; rgbd>rgb beweist
  Kabsch greift). VRAM sequenziell gemessen.
- **yolo-seg = INFRA-PASS / DATEN-FAIL** (build/health/request grün, aber interim-Weight = stock COCO →
  0 Anker-Masken). **ERWARTET, KEIN BLOCKER** — echtes Anker-seg-Modell ist **S-008 / T-134**. Als bekannte
  Abhängigkeit im compose-Kommentar + Report dokumentiert.

### Bekannte Folge-Abhängigkeiten (kein Block)
- yolo-seg braucht S-008/T-134 (YOLO26m-seg) für Kombis 2/4/5.
- VRAM-Lifecycle (fp+gigapose nicht gleichzeitig sicher resident) = S-007/T-133.
- sam3-svc = T-144 (gated HF-Modell).

### Security → Bruno
Build/Config-Diff. RO-Weights-Mounts (keine schreibbaren Secret-Pfade). Keine neue App-Logik/Input-
Boundary. Bruno-Review für diesen Diff nicht erforderlich.

**Gate: APPROVED.**

---

## Zusammenfassung

| Story | Ticket | Verdikt | Tests | Regression | Bruno-Hinweis |
|---|---|---|---|---|---|
| S-003 | T-129 | **APPROVED** | 14/14 + Vollsuite 150p (1 pre-existing TTA-Flake) | Live-Pfad byte-id zu main | nein |
| S-005 | T-131 | **APPROVED** | smoke §2-PASS (5 dets) + Logik empirisch | n/a (neuer Svc) | ⚠️ /segment-Input-Boundary, falls noch offen |
| S-006 | T-132 | **APPROVED** | fp+gigapose Box-PASS vs GT | n/a (Build/Wiring) | nein |

**0 Blocker. yolo-seg DATEN-FAIL = S-008, kein Block. Pre-existing TTA-Flake = T-088, kein Block.**
**Ein Bruno-Hinweis** für S-005 (neuer /segment-Service-Endpoint).

---
---

# QA-Gate — **Wave 2 / Batch 2** (Ravi/QS) — Session 2026-06-07-multi-pipeline-pose

> Gate = **Tests + Correctness**. Security ist Brunos Revier (T-145, NICHT hier geprüft).
> Tickets: **T-130 (S-004), T-136 (S-010), T-137 (S-011), T-139 (S-013), T-138 (S-012), T-146 (sam3-svc)**.
> **Verdikt: 6/6 PASS, 0 Blocker.** Methodik: alle load-bearing Claims empirisch reproduziert
> (numpy/node), NICHT nur Code gelesen. Test-Suiten lokal re-gefahren wo möglich.

---

## S-004 / T-130 — gdrnpp-svc (5. Pose-Service, Kombi 1) — **APPROVED**

**Commits:** `c5c1da0` + `0bc6474` (team/multipipe/S-004) · gdrnpp-svc/app.py (neu, native venv:8012,
stdlib-HTTP) + gateway/app.py (+66, §4-Kopplung) + compose (+11) + 2 Box-Tests.

### Review-Findings (Correctness) — alle 3 Kern-Claims empirisch bewiesen (numpy, ohne GPU)
- **`_obb_to_aabb` == box_src-Referenz exakt:** Service baut die 4 Ecken aus `[cx,cy,w,h,θ]`
  (`local @ R.T + center`) identisch zur ultralytics-`xyxyxyxy`-Geometrie, dann min/max. **6 Fälle
  MATCH** (axis-aligned, 90°-w/h-swap → 20×40, 45°-Vergrößerung → 42.426, arbitrary 30°, neg −60°, 180°)
  gegen `box_src/obb_to_aabb_dets.obb_corners_to_aabb`, atol 1e-9.
- **`_T_cam_obj` mm→m = CONTRACT §1 bit-genau:** Identität @300mm → `T[2,3]=0.30`, full-Matrix == §1-Beispiel;
  `t_mm=[125,-64,1080]→t_m=[0.125,-0.064,1.08]`. R direkt, kein Flip/World-Transform.
- **id-Reattach order-robust (subtilster Bug-Kandidat):** preds kommen GRUPPIERT nach obj_id (model load
  order), `cursor` ist **per-obj_id** → korrekt auch bei interleaved (kurz/lang/kurz/lang/kurz),
  nicht-sequentiellen ids [7,3,9,1,5], und wenn das Modell obj_ids in [2,1]-Reihenfolge emittiert.
  Unbekannte Klasse (zahnrad) gedroppt → restliche 3 bleiben aligned.

### Regression (Live-Pfad heilig — KEINE :8078-Doppelbelastung)
- **5/5 Live-Pfad-Files byte-identisch** (0 diff-Zeilen): `kip_infer_worker`, `kip_server`, `e2e_infer`,
  `bop_adapter`, `compare_pipelines`.
- Service importiert `kip_infer_worker._build_det_consistent_root` **read-only** (kein Monkeypatch).
  Diese Funktion (worker:101) baut einen **isolierten** symlink-Root + Dummy-scene_gt (T-115-safe) —
  „Original-updir-scene_gt bleibt UNANGETASTET". gdrnpp-svc übergibt seinen EIGENEN frischen temp-`updir`
  (eigenes scene_camera, kein echtes scene_gt) → kann Live-Daten nie berühren.
- Separater Port (8012), eigene VRAM-Fraktion (0.30). Box-Roundtrip vs Live-:8078: **Δt=0mm, ΔR=0°** (Kai).

### Gateway-Gating (§4) korrekt
`pose=gdrnpp & seg≠yolo-obb` → force yolo-obb; `seg=yolo-obb & pose≠gdrnpp` → **400** (off-whitelist-reject);
`obb` ins /pose-instance geforwarded. health + compose-Wiring sauber.

### Security → Bruno
Neuer /pose-Service nimmt `rgb_b64`+`obb`+`K` (cv2.imdecode, kein eval/shell). ⚠️ Neuer Service-Endpoint +
Input-Boundary (wie S-005): **falls noch nicht durch Bruno → /pose-Input-Boundary-Review empfohlen.**

**Gate: APPROVED** (Bruno-Hinweis vermerkt).

---

## S-013 / T-139 — kip_server-Proxy (6 NICHT-A-Kombis → Gateway) — **APPROVED**

**Commit:** `80c3c94` (team/multipipe/S-013) · kip_server.py (+231/-12), gateway_proxy.py (neu, pure),
composed.py (tcamobj_to_world_entry extrahiert), 2 Test-Files.

### Review-Findings (Correctness) — empirisch (node + pytest)
- **`is_pipeline_a`-Guard EXAKT** (regression-kritisch): alle 4 A-Formen (pipeline=gdrnpp / leer / None /
  seg=yolo-obb&pose=gdrnpp) → True; alle **7 adversarialen non-A** (inkl. off-whitelist-mix
  yolo-obb+foundationpose, seg-only, pose-only, garbage) → **False**. → **Kein non-A erreicht je den
  Live-Worker.**
- **`resolve_combo_id`:** 6 non-A by-id UND by-axis-Label resolved; **6 invalide** (yolo-obb+fp,
  yolo-obb+gigapose, sam3+gdrnpp, yolo-seg+gdrnpp, bogus, nonexist) → `InvalidCombo` (→ **400**).
  „Niemals 12 Kombis"-Invariante hält.
- **`unavailable_reason`** Präzedenz korrekt: training > service_down; available → None.
- **`pipelines_status`:** Pipeline A IMMER available wenn Live up (Gateway voll down → nur `gdrnpp`,
  Rest `service_down`); training_segs → `training`; healthy → 7/7; total 7 Kombis.
- **`gateway_predict_to_pose_result`** reused die EXAKT geteilte `composed.tcamobj_to_world_entry`
  (`*1000` m→mm + `bop_pose_to_world` + `canonicalize_rotation`) — **keine 2. Mapping-Implementierung**
  (byte-gleiche Docs zum Composed-Pfad, Eval-Konsistenz). Per `git show` als Extraktion verifiziert
  (− inline / + shared fn).

### Regression (Live-Pfad `pipeline=gdrnpp` byte-identisch)
- **NULL Live-Pfad-Funktion berührt:** alle `-`-Diff-Zeilen liegen NUR in `/api/pipelines` (Stub→voll),
  `/api/compare` (Stub-Text), `_LIVE_ROOT` (env-override). KEINE Zeile in `_real_infer_job`/`real_infer`/
  `_bop_pose_to_result`/`sim_*`/`_zivid_cam` entfernt oder geändert.
- Pipeline-A-Branch delegiert byte-identisch an `_real_infer_job` (Thread, identisch zu /api/real/infer_async).
- `_LIVE_ROOT` env-overridebar (`KIP_LIVE_ROOT`), **Box-Default unverändert**.

### Tests
- 50 grün für die 3 S-013-Files (22 proxy + 14 predict + 14 composed) lokal re-gefahren.
- **Vollsuite: 186 passed / 7 skip / 1 fail.** Der eine Fail = **pre-existing TTA-Flake T-088**
  (`1.2074182697257333e-06 < 1e-6`, byte-identisch, test_tta_pose.py 0 diff-Zeilen). Kein Block.

### Security → Bruno
Proxy nimmt FE-Multipart (rgb/depth/K) + reicht an Gateway weiter (httpx). Neuer /api/predict-Pfad +
Gateway-Forward. ⚠️ Falls Auth/Origin-Boundary für die neue Origin relevant → Brunos Scope.

**Gate: APPROVED.**

---

## S-012 / T-138 — Batch-Eval-Runner + D-1-Pivot (combos.py) — **APPROVED**

**Commits:** `78b2b61` (Runner) + `7b23eda` (PIVOT) · eval/batch_eval.py (neu), combos.py (+99 Pivot),
kip_server.py (+106 eval-Endpoints), 2 Test-Files.

### Review-Findings (Correctness) — empirisch (python)
- **combos.py-Pivot ADDITIV:** `COMBO_WHITELIST` **unverändert 7** (exakte ids, needs_depth = CONTRACT §5,
  genau 1 is_pipeline_a). `FEASIBLE_COMBOS = 12` (3 seg × 4 pose), `recommended`-Flag = **exakt die 7**.
  Die 5 neuen feasiblen: 3× yolo-obb→non-gdrnpp (obb liefert auch Maske → nicht degraded) + 2× degraded
  gdrnpp-via-Maske (`aabb_from_mask`; sam3 zusätzlich class_ambiguity).
- **feasibility-Predicate korrekt:** gdrnpp+yolo-obb = native (nicht degraded); gdrnpp+yolo-seg/sam3 →
  degraded `aabb_from_mask`; sam3 → class_ambiguity; needs_depth pro pose erhalten.
- **`_combo_id`:** yolo-obb+gdrnpp → `"gdrnpp"` (Monolith-id, kein Doppel-Register).
- **Registry-Side-Effect SAFE (Batch-1-Lesson):** nach `import pipelines` ist `registry.get("gdrnpp")`
  weiterhin `GdrnppReferenceAdapter` (NICHT Composed verdrängt), 7 ids registriert,
  `register_combos()` idempotent (has-Guard, 3× aufgerufen → stabil). build_combos() = 6 non-A (S-003/S-013 intakt).
- **Coord-Frame Round-Trip** grün (geteiltes `*1000`+`bop_pose_to_world`+`canonicalize` via
  `_instances_to_doc_local`, byte-gleich zu composed/S-013).
- **Aggregation:** `coverage`/`crash_rate` IMMER in [0,1] (oder None@n_real=0, kein div0) über 6 adversariale
  Szenarien bewiesen; `pipeline_a_no_gateway`-Szenen aus n_real raus (apples-to-apples); Pivot-Flags passthrough.
- **Endpoint-Schemata = Lenas batch.js:** `/api/eval/{runs,result,run,job}`, sim-job-Schema, run_config_id.

### Tests
- 28 grün für die 2 eval-Files (inkl. roundtrip/coordframe/aggregate/coverage). **Vollsuite 178 passed /
  7 skip / 1 fail** = derselbe pre-existing TTA-Flake T-088 (byte-identisch). Kein Block.

### MERGE-NOTE (für Sam, kein Block)
S-012 UND S-013 ändern beide die `_LIVE_ROOT`-Zeile mit **identischem** env-override (`KIP_LIVE_ROOT`,
gleicher Box-Default) → semantisch identisch, Merge-Auflösung trivial (egal welche Seite).

### Security → Bruno
Eval-Runner + Endpoints additiv. Kein neuer User-Input-Pfad mit Injection-Risiko (Form-Felder seeds/iterations
numerisch). Bruno-Review für diesen Diff nicht zwingend.

**Gate: APPROVED.**

---

## S-010 / T-136 + S-011 / T-137 — FE 2-Dropdown-Gating + Batch-Eval-Reiter — **APPROVED**

**Commit:** `d049d99` (team/multipipe/S-010-011) · pipeline.js (neu, Gating-Engine), batch.js (neu, Eval-Tab),
kip.js/kip.html/kip.css (Verdrahtung). Vanilla-JS, keine Deps.

### Review-Findings (Correctness) — empirisch via node-Import der ECHTEN pipeline.js gegen backend-JSON
- **FE WHITELIST == backend COMBO_WHITELIST (0 enum-drift):** 7 rows × 6 fields byte-identisch
  (n/seg/pose/id/needs_depth/is_pipeline_a). Spiegel ist dokumentiert + jetzt verifiziert konsistent.
- **Exakt 7 von 3×4=12 valid;** die 5 invaliden = exakt die GDRNPP-Kopplungs-Ausschlüsse (yolo-obb+non-gdrnpp,
  non-yolo-obb+gdrnpp).
- **KEINE 12-Sackgasse:** jedes Seg auto-springt auf eine valide Kombi (yolo-seg→FP, sam3→FP); nie ein Landing
  auf einer invaliden Auswahl.
- **GDRNPP-Gating mit Grund (nicht weggeblendet):** „nur mit YOLO-OBB" / „braucht Maske (nicht YOLO-OBB)".
- **unavailable_reason surfaced:** training → „Modell trainiert noch", service_down → „Dienst nicht aktiv".
- **Empty-State / 404:** availById leer → nur Pipeline A available (`anyAvailable=true`, kein Crash); batch.js
  `setEmpty()` in beiden catch (loadRuns/loadResult) → klare Meldung statt Crash.
- **A11y/UX:** `prefers-reduced-motion: reduce` (kip.css:255), `aria-sort` auf aktiver Spalte, BEST-Zeile
  3-fach (row--best + BEST-Pille + Akzent-Border/aria-label, bestId = höchster AR unabhängig von Sortierspalte),
  `fmtPct` 0..1→% robust gg 0..100 + null→„—".
- **Keine Emojis im NEU-Code:** 4 Glyph-Hits (`⛶`/`⤫`/`⚠`) sind ALLE pre-existing (Fullscreen-Toggle +
  Live-Kamera-Warnung), NICHT im d049d99-Diff. pipeline.js + batch.js emoji-frei.

### Tests
Lenas 56 Playwright + 21 node-Gating grün (ephemer, nicht committet). Gating-Logik unabhängig per node gegen
backend-JSON re-bewiesen (no-drift, no-dead-end, reasons, empty-state). Live-Playwright-Re-Run nicht zwingend:
die Gating-Engine ist pure JS, end-to-end via node verifiziert; ein Browser-Lauf bräuchte ein laufendes
kip_server-Backend (nicht permanent up) und würde nur Rendering testen, nicht die schon abgedeckte Logik.

### Security → Bruno
Rein client-seitig (Vanilla-JS, kein neuer Input-Sink, kein eval). Bruno-Review nicht erforderlich.

**Gate: APPROVED** (beide Tickets).

---

## sam3-svc / T-146 — sam3-svc:8004 hochfahren (offline-Mount) — **APPROVED**

**Commit:** `22220df` (team/multipipe/integration) · Dockerfile + docker-compose + requirements + yolo-obb-svc-Wiring.
Build/Config-Story.

### Review-Findings (Correctness)
- **Dockerfile base blackwell→ampere** (sm_86, `foundationpose:ampere`), `transformers==5.10.2`
  (Sam3Model/Sam3Processor, v5) + `accelerate` (für `device_map="cuda"`; blackwell-base hatte es, ampere nicht).
- **Offline DOPPELT gehärtet:** Image-ENV `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` UND compose
  `HF_HUB_OFFLINE=1` → Container erreicht das Netz nie.
- **HF-Mount-Fix korrekt:** `HF_HOME=/hf_cache` == container mount target (vorher `/hf-cache` vs `/hf_cache`
  Mismatch). Host-Default = `/mnt/data/kip_pose_weights/hf_cache` (Brunos S-009-Snapshot,
  `hub/models--facebook--sam3`, safetensors-only/kein pickle), **READ-ONLY** (`:ro`). Per regex-Parse präzise
  verifiziert (Mount-Target = /hf_cache, mode = ro).
- **SAM3_CONF=0.2** in compose (`${SAM3_CONF:-0.2}`) UND app.py:70 — Kommentar dokumentiert die ML_RECON/
  yannic-Falle (0.2 statt docs-0.5, SAM3-Scores laufen niedrig auf out-of-domain).
- **YAML valid** (6 services). Kombis 3/6/7 schon in combos.py registriert → kein whitelist-edit nötig.

### Tests
Smoke **INFRA-PASS** (Sam): /health `{ok:true}` + /segment (4 lang + 3 kurz Masken), offline-load 45s,
VRAM idle 1.88GB / peak 3.42GB. **kurz/lang-Trennungs-Limit dokumentiert** (per-Klasse-prompt-Test negativ →
YOLO bleibt Klassen-Quelle) = bekannte Limitation, **KEIN Bug** (sam3 = class-agnostic masks by design).

### Security → Bruno
Gated HF-Modell (facebook/sam3), RO-Mount, safetensors-only (kein pickle-RCE — Bruno S-009 hat das geprüft).
Kein neuer App-Input-Sink. Bruno-Review für diesen Diff nicht erforderlich (S-009 deckt das Weight-Sourcing ab).

**Gate: APPROVED.**

---

## Zusammenfassung — Batch 2

| Story | Ticket | Verdikt | Tests | Regression / Correctness-Beweis | Bruno-Hinweis |
|---|---|---|---|---|---|
| S-004 | T-130 | **APPROVED** | Box-Roundtrip Δt=0/ΔR=0° + 3 Claims empirisch (numpy) | 5/5 Live byte-id, T-115-safe, keine :8078-Doppel | ⚠️ /pose-Input-Boundary (wie S-005) |
| S-013 | T-139 | **APPROVED** | 186 passed (1 pre-existing T-088) | Guard exakt, NULL Live-Fkt berührt, mapping geteilt | ⚠️ nur falls Origin/Auth relevant |
| S-012 | T-138 | **APPROVED** | 178 passed (1 pre-existing T-088) | combos-Pivot additiv, Registry nicht verdrängt, [0,1] bewiesen | nein |
| S-010 | T-136 | **APPROVED** | 56 PW + 21 node + node-Re-Beweis | FE-WHITELIST 0 drift, kein Sackgasse, emoji-frei | nein |
| S-011 | T-137 | **APPROVED** | (s. S-010) | Empty-State 404 ok, a11y+reduced-motion, BEST 3-fach | nein |
| sam3-svc | T-146 | **APPROVED** | Smoke INFRA-PASS (4lang+3kurz) | Mount-Fix korrekt, CONF=0.2, offline 2-fach gehärtet, RO | nein (S-009 deckt) |

**6/6 PASS, 0 Blocker.** Pre-existing TTA-Flake T-088 (byte-id, test 0 diff) = kein Block — 4. Session bestätigt.
sam3 kurz/lang-Limit + gdrnpp-AABB-Maske-Fallback (`degraded`) = dokumentierte Limitationen, keine Bugs.
**MERGE-NOTE (Sam):** S-012+S-013 `_LIVE_ROOT`-Zeile identischer env-override → konfliktfrei.
**Bruno-Hinweise:** S-004 (neuer /pose-Service-Endpoint, Input-Boundary) — analog zum S-005-Hinweis aus Batch 1.
