# S-006 — Yannics 4 Services build+health auf der Box (T-132)

**Owner:** Sam (DevOps) · **Session:** 2026-06-07-multi-pipeline-pose · **Branch:** `team/multipipe/integration` · **Commit:** `db12aad`
**Box:** `max@100.85.216.95` (RTX 3090, 24 GB) · **Datum:** 2026-06-07

**Scope (D1, 2-Klassen):** nur `anker_kurz` / `anker_lang`. Kein Zahnrad.
**sam3-svc:8004 DEFERRED** → T-144 (kein HF-Token für gated `facebook/sam3`).

---

## TL;DR — pro Service

| Service | Port | Build | /health | Smoke (Pose/Maske) | VRAM (gemessen) | Verdikt |
|---|---|---|---|---|---|---|
| **yolo-seg-svc** | 8001 | ✅ | ✅ `{ok:true}` | ⚠️ 0 Anker-Masken (interim-weight = stock COCO) | **~0.42 GB** | **INFRA-PASS / DATEN-FAIL** |
| **fp-svc** | 8002 | ✅ | ✅ `{ok,classes:[kurz,lang],cuda}` | ✅ T_cam_obj, trans_err 28–81 mm vs GT | **~2.95 GB** peak | **PASS** |
| **gigapose-svc** | 8003 | ✅ | ✅ `{status:ok,refiner:true}` | ✅ rgbd 32–94 mm / rgb 95–439 mm vs GT | **~3.37 GB** peak / ~2.45 GB steady | **PASS** |
| sam3-svc | 8004 | — | — | — | — | DEFERRED (T-144) |

**Ehrlicher Negativ:** yolo-seg-svc läuft infrastrukturell sauber, aber das einzige interim-Gewicht auf der Box (`/mnt/data/kip_pose/yolo26n.pt`) ist **stock COCO** (`task=detect`, 80 COCO-Klassen, kein Anker, kein seg) → liefert **0 Masken** für Anker. Das echte Anker-seg-Modell ist **S-008 / T-134** (YOLO26m-seg, 2 Klassen) und existiert noch nicht. fp+gigapose wurden daher mit **BOP-GT-Masken** gesmoked (entkoppelt Pose-Validierung vom fehlenden seg-Modell und erlaubt zusätzlich Validierung gegen GT-Posen).

---

## VRAM-Tabelle — Input für S-007 (VRAM-Lifecycle-Manager)

Sequenziell gemessen (nie fp+gigapose gleichzeitig resident). Baseline = nur Live-Worker (`kip_server`/`gdrnpp`, PID 84836) = **1684 MiB**. Service-Werte = per-Prozess (`nvidia-smi --query-compute-apps`), additiv zur Baseline.

| Service | VRAM resident (idle) | VRAM peak (inference/load) | Load-Zeit | Inferenz-Zeit |
|---|---|---|---|---|
| yolo-seg-svc:8001 | ~0.42 GB | ~0.42 GB | <5 s | <0.5 s/Bild |
| fp-svc:8002 | ~0.9 GB (Modelle geladen) | **~2.95 GB** (register) | ~24 s | ~2.5 s/Pose |
| gigapose-svc:8003 | **~3.37 GB** (warm, Templates onboarded) | ~3.37 GB transient, **~2.45 GB** steady | ~77 s | rgbd ~3.0 s · rgb ~0.4 s |
| yolo-obb-svc:8011 (Kai/S-005) | ~0.46 GB | ~0.46 GB | <3 s | sub-s |

**Konsequenz für S-007:**
- **fp + gigapose passen NICHT gleichzeitig in 24 GB** neben dem Live-Worker, wenn beide ihren Peak ziehen (1.68 + 2.95 + 3.37 ≈ 8 GB rein für Pose — passt zahlenmäßig, aber FoundationPose `register()` allokiert kurzzeitig viel mehr über den nvdiffrast/warp-Kontext; konservativ sequenziell halten bis S-007 das per LRU-Eviction managt).
- **Seg-Quellen (yolo-seg, yolo-obb, sam3) sind billig** (~0.4–0.5 GB) → S-007-Empfehlung: **Seg-Service persistent halten**, Pose-Service per LRU laden/entladen.
- gigapose ist **EIN warmes Modell für beide Pipelines** (rgb+rgbd) — kein doppeltes Residieren nötig.

---

## Start-Befehle (reproduzierbar)

Repo auf Box: `~/kip_mesh` (rsync von `project/mesh/`), `.env` gesetzt (siehe unten). Images aus S-001: `foundationpose:ampere`, `gigapose:ampere`.

```bash
# .env (Box) — out-of-repo Pfade
FOUNDATIONPOSE_DIR=/home/max/kip_build/FoundationPose
GIGAPOSE_DIR=/home/max/kip_build/GigaPose
FP_WEIGHTS_DIR=/mnt/data/kip_pose_weights/foundationpose/weights
YOLO_WEIGHTS_PT=/mnt/data/kip_pose/yolo26n.pt   # interim (stock COCO! -> S-008)

cd ~/kip_mesh

# --- yolo-seg-svc ---
docker compose build yolo-svc && docker compose up -d yolo-svc
curl -s localhost:8001/health   # {"ok":true}

# --- fp-svc (sequenziell, vorher yolo stoppen) ---
docker compose stop yolo-svc
docker compose build fp-svc && docker compose up -d fp-svc
curl -s localhost:8002/health   # {"ok":true,"classes":["anker_kurz","anker_lang"],"cuda":true}

# --- gigapose-svc (sequenziell, vorher fp stoppen) ---
docker compose stop fp-svc
docker compose build gigapose-svc && docker compose up -d gigapose-svc
curl -s localhost:8003/health   # {"status":"ok","dataset":"kip2","refiner":true,...}
```

### Einmalige Vorbereitung auf der Box (KRITISCH — sonst Startup-Fails)

```bash
# 1) FP mycpp C++-Extension kompilieren (foundationpose:ampere ist BASE-Image, kein FP-repo gebaked).
docker run --rm --gpus all -v /home/max/kip_build/FoundationPose:/workspace/FoundationPose \
  --entrypoint bash foundationpose:ampere \
  -c "cd /workspace/FoundationPose && export PATH=/opt/venv/bin:\$PATH && bash build_mycpp.sh"
# -> erzeugt mycpp/build/mycpp.cpython-310-*.so

# 2) FP weights-Mountpoint anlegen (Overlay-Subdir eines RO-Mounts braucht existierendes Ziel)
mkdir -p /home/max/kip_build/FoundationPose/weights

# 3) FP-Meshes (metres) — GigaPose kip2-Modelle = Anker (obj1=kurz, obj2=lang)
mkdir -p ~/kip_mesh/assets/meshes
cp /home/max/kip_build/GigaPose/datasets/kip2/models/obj000001.obj ~/kip_mesh/assets/meshes/anker_kurz.obj
cp /home/max/kip_build/GigaPose/datasets/kip2/models/obj000002.obj ~/kip_mesh/assets/meshes/anker_lang.obj

# 4) GigaPose pretrained verkabeln (Symlinks; werden im Container über den /mnt/data-RO-Mount aufgelöst)
mkdir -p /home/max/kip_build/GigaPose/pretrained
ln -sfn /mnt/data/kip_pose_weights/gigapose/pretrained/gigaPose_v1.ckpt /home/max/kip_build/GigaPose/pretrained/gigaPose_v1.ckpt
ln -sfn /mnt/data/kip_pose_weights/gigapose/pretrained/megapose-models  /home/max/kip_build/GigaPose/pretrained/megapose-models

# 5) GigaPose-Templates rendern (waren NICHT auf der Box; panda3d/EGL, ~13 s für 2 obj x 162 views)
docker run --rm --gpus all -e PYOPENGL_PLATFORM=egl \
  -v /home/max/kip_build/GigaPose:/workspace/GigaPose --entrypoint bash gigapose:ampere \
  -c "cd /workspace/GigaPose && export PATH=/opt/venv/bin:\$PATH && python -m src.scripts.render_custom_templates custom_dataset_name=kip2 machine.num_workers=2"
# -> datasets/templates/kip2/{000001,000002}/<view>.png+_depth.png (324 je obj) + object_poses/*.npy
```

---

## Smoke-Belege (BOP `_smoke_bop_t038`, GT-Maske + GT-Pose)

Testbed: `/mnt/data/kip_pose/data/_smoke_bop_t038/train_pbr/000000` — RGB-D + GT-Masken + GT-Posen + `scene_camera` (K, **depth_scale=0.1**). obj1=anker_kurz, obj2=anker_lang. Frame3=kurz, Frame1=lang.

**Contract-Falle:** BOP-Depth-PNG ist NICHT in mm — erst `* depth_scale (0.1)` ergibt mm. Der Contract (`depth_b64 = uint16 PNG mm`) erwartet echte mm; Smoke-Harness re-encodiert.

### fp-svc:8002 (RGB-D, Pflicht-Depth)
```
frame 000003 (anker_kurz):  GT t_mm=[-332.7,183.5,921.8]  pred t_mm=[-359.1,198.3,996.5]  trans_err= 80.6mm  rot_err=117.4°  (2.5s)
frame 000001 (anker_lang):  GT t_mm=[-190.6,-143.0,1036.8] pred t_mm=[-194.6,-146.1,1063.9] trans_err= 27.5mm  rot_err=139.7°  (2.6s)
```

### gigapose-svc:8003 — `pipeline=rgbd` (coarse + GenFlow + Kabsch/ICP, stage=`refined+kabsch`)
```
frame 000003 (anker_kurz):  trans_err= 94.1mm  rot_err=172.5°  (3.4s)
frame 000001 (anker_lang):  trans_err= 32.3mm  rot_err= 62.9°  (2.9s)
```

### gigapose-svc:8003 — `pipeline=rgb` (coarse + GenFlow, KEIN Depth, stage=`refined`)
```
frame 000003 (anker_kurz):  trans_err=438.8mm  rot_err=150.8°  (0.3s)
frame 000001 (anker_lang):  trans_err= 94.8mm  rot_err= 61.1°  (0.4s)
```

**Interpretation:**
- **Translation ist valide** (cm-Skala auf ~1 m entfernten Objekten). fp und gigapose-rgbd liefern ehrlich gute Posen.
- **rgbd > rgb** (Kabsch greift): Kabsch-Depth-Alignment zieht trans_err von 95–439 mm auf 32–94 mm. Beweis, dass der Depth/Kabsch-Tail tatsächlich aktiv ist (nicht nur Import).
- **rot_err 61–173° ist KEIN Fehler:** Anker haben laut `models_info.json` **continuous symmetry um die Y-Achse** (rotationssymmetrische Stifte). Eine Drehung um die Symmetrieachse ist eine gültige Pose → naive geodätische Rotationsdistanz ist für diese Objekte bedeutungslos. Die Translationsgenauigkeit belegt korrekte Registrierung.
- gigapose-rgb ist ~10× schneller (0.3–0.4 s vs 3 s) weil der Depth-Refine entfällt.

---

## Stolpersteine (gebrannt)

1. **`mycpp.cluster_poses` fehlt** (fp-svc Startup-Crash). Root-Cause: `foundationpose:ampere` ist ein **BASE-Image ohne gebaktes FP-Repo**; FP-Code (inkl. C++-Ext `mycpp`) kommt aus dem RO-Mount, war aber im Host-Checkout **nie kompiliert**. Fix: `build_mycpp.sh` einmal via Image-Toolchain → `.so` landet via Mount im Container.
2. **Weights-Overlay auf RO-Mount** (`mkdirat ... read-only file system`). FP-Weights-Mount auf `/workspace/FoundationPose/weights` schlägt fehl, weil das Eltern-`/workspace/FoundationPose` RO ist und Docker den Mountpoint nicht anlegen kann. Fix: `weights/`-Dir vorab im Host-Checkout anlegen → existiert dann im RO-Mount als Mountpoint.
3. **BOP `depth_scale=0.1`** — Depth 10× zu groß führte zu 10× zu weit platziertem Objekt (trans_err 9915 mm → 80 mm nach Korrektur). Test-Harness-/Datenkonventions-Sache, kein Service-Bug.
4. **GigaPose-Templates fehlten komplett** auf der Box (Kai-Recon-Annahme „committed im Repo" war falsch — 0 PNGs). Selbst gerendert via `render_custom_templates custom_dataset_name=kip2` (panda3d glxGraphicsPipe headless, 12.8 s). Output: `datasets/templates/kip2/{000001,000002}` (324 PNG je obj) + `object_poses/*.npy`.
5. **GigaPose-Symlinks dangling im Container** (`pretrained/gigaPose_v1.ckpt` FileNotFound). Symlinks zeigen auf absolute `/mnt/data/...`-Pfade, die nicht in den Container gemountet waren. Fix: `/mnt/data/kip_pose_weights` + `/mnt/data/bop/weights` (Symlink-Ziel) RO am **selben absoluten Pfad** mitmounten → Symlinks resolven.
6. **CUDA-Context-Ordering (ML_RECON §B) — bereits in `gigapose_infer.py` gelöst:** Modell auf CPU bauen → Refiner forken → dann GPU; `Panda3dSingleProcessRenderer` in-process (kein Worker-Fork). `GP_REFINER_RENDERER=single`, `XFORMERS_DISABLED=1` (xFormers not available Warning ist harmlos), `HF_HUB_OFFLINE` n/a hier (DINOv2 über torch_cache). Load: `ckpt loaded (missing=0 unexpected=0)`, `MegaPose refiner ready`, `refiner=on`. Funktionierte beim ersten Versuch.

---

## compose-/Service-Edits (committed `db12aad`)

- `yolo-svc/`, `fp-svc/`, `gigapose-svc/` Dockerfiles: `FROM …:blackwell` → `…:ampere`.
- `docker-compose.yml`:
  - fp-svc: FP-Weights RO-Mount (`${FP_WEIGHTS_DIR}:/workspace/FoundationPose/weights:ro`), Meshes via `${FP_MESH_DIR}`.
  - gigapose-svc: Pretrained-Store + BOP-Weights RO-Mounts am selben absoluten Pfad (Symlink-Resolution).
  - Kommentar-Dokumentation der mycpp-/Template-/Weights-Fallen.
- `yolo-obb-svc` (S-005, Kai) NICHT von mir committed — kommt via S-005-Merge (rein additiv im working tree gelassen, 0 reverts).

---

## Live-Services (NICHT angefasst, durchgehend intakt)

`kip_server:8077` (PID 84836, `/mnt/data/isaacsim-venv`) + `gdrnpp-Worker:8078` (HTTP 200) liefen die ganze Session ungestört. Kein `pkill -f uvicorn`. Alle 3 Mesh-Services nach Validierung gestoppt → GPU final clean bei 1693 MiB.

---

## Offene Folge-Items

| Item | Ticket | Warum |
|---|---|---|
| Anker-seg-Modell (yolo-seg liefert sonst 0 Masken für Kombi 2/4/5) | **S-008 / T-134** | interim yolo26n.pt = stock COCO |
| VRAM-Lifecycle-Manager (Seg persistent + Pose LRU) | **S-007 / T-133** | fp+gigapose nicht gleichzeitig sicher resident |
| ~~sam3-svc (Kombi 3/6/7)~~ | **T-146** | ✅ ERLEDIGT — siehe sam3-svc-Sektion unten (Bruno cachte Weights in T-144). |
| Mesh-Maße verifizieren (GigaPose-obj als FP-Mesh wiederverwendet) | — | obj-Extent 124/139 mm vs BOP-diameter 112/115 mm — konsistent genug, aber bei Genauigkeitsanspruch prüfen |

---

## sam3-svc:8004 — Nachzügler aus S-006 (T-146, 2026-06-07) — **PASS (infra) + Eval-Limit dokumentiert**

War in S-006 deferred (gated `facebook/sam3`, kein Token). Bruno hat den Snapshot in T-144 offline
gecacht (`/mnt/data/kip_pose_weights/hf_cache`, safetensors-only). Jetzt hochgefahren.

### Verdikt-Zeile (ergänzt die TL;DR-Tabelle)

| Service | Port | Build | /health | Smoke (Masken) | VRAM (gemessen) | Verdikt |
|---|---|---|---|---|---|---|
| **sam3-svc** | 8004 | ✅ (`sam3:ampere`, neu) | ✅ `{ok:true}` | ✅ 4–7 Masken/Frame (default-prompts) | **idle ~1.88 GB / peak ~3.42 GB** | **INFRA-PASS** (Klassen-Trennung schwach = Eval-Befund, kein Bug) |

### Start-Befehl (reproduzierbar)

```bash
# Image: sam3:ampere = foundationpose:ampere (torch cu128, sm_86) + transformers==5.10.2 + accelerate.
# Der originale Yannic-Dockerfile-FROM (foundationpose:blackwell-sam3) existiert auf dieser Box NICHT.
cd ~/kip_mesh
docker compose build sam3-svc && docker compose up -d sam3-svc
curl -s localhost:8004/health     # {"ok":true}
# Modell laedt OFFLINE aus /hf_cache (HF_HUB_OFFLINE=1, kein Netz/Token) — ~45 s, conf=0.2.
```

Mount (kritisch — S-009/Bruno): `HF_HOME=/hf_cache` MUSS = Container-Mountpunkt sein:
`-v /mnt/data/kip_pose_weights/hf_cache:/hf_cache:ro` + `HF_HUB_OFFLINE=1` + `SAM3_CONF=0.2` (**nicht 0.5** — ML_RECON/yannic-Falle, SAM3-Scores laufen auf diesen OOD-Bildern niedrig).

### Stolpersteine (gebrannt)

1. **`foundationpose:blackwell-sam3` existiert auf der Box NICHT.** Yannic baute das auf seiner Blackwell-Maschine.
   `foundationpose:ampere` (das echte lokale Base, sm_86) hat **kein transformers**. Fix: `sam3-svc/Dockerfile`
   `FROM foundationpose:ampere` + `transformers==5.10.2` (erste Version, die `Sam3Model`/`Sam3Processor` sauber
   gegen torch 2.11.0+cu128 importiert — vor dem Bake auf der Box verifiziert). torch bleibt das Base-cu128-venv,
   wird NIE neu installiert.
2. **`device_map="cuda"` braucht `accelerate`** (Startup-Crash: `ValueError ... requires accelerate`). `app.py`
   nutzt `Sam3Model.from_pretrained(..., device_map="cuda")`; der transformers-device_map-Pfad delegiert die
   Platzierung an accelerate. Das Blackwell-Base bundlete es, ampere nicht → `accelerate` in `requirements.txt`
   nachgezogen. (App-Code unverändert — Yannics `app.py` bleibt byte-identisch.)
3. **compose HF-Mount-Mismatch** (war vor T-146 falsch): `HF_HOME: /hf-cache` aber Brunos Cache + Report sagen
   `/hf_cache`; Default-Pfad war `~/.cache/huggingface` statt `/mnt/data/kip_pose_weights/hf_cache`. Beides
   korrigiert (`HF_HOME=/hf_cache`, Mount + Default auf den Box-Pfad), sonst hätte der Offline-Resolver das
   `hub/`-Layout nicht gefunden.

### /segment-Smoke (BOP `_smoke_bop_t038`, frame1 + frame3)

Default-Prompts (`app.py`: `short metal motor armature part` / `long metal motor armature part`), `SAM3_CONF=0.2`:

```
frame 000001:  4 Masken (alle non-empty)   classes={anker_lang:4}            conf 0.24–0.59   (1.5s)
frame 000003:  7 Masken (alle non-empty)   classes={anker_lang:4, anker_kurz:3}  conf 0.24–0.86   (0.4s)
```

→ **Masken kommen sauber zurück, /segment-Contract erfüllt** (id/class/conf/mask_b64). Latenz sub-2s.

### Bekanntes Limit — Klassen-Trennung kurz/lang (Eval-Befund, KEIN Bug)

Bestätigt yannics Messung (S-009/Bruno): **SAM3 trennt `anker_kurz` vs `anker_lang` NICHT zuverlässig.**
- frame1 kam **komplett `anker_lang`** zurück, obwohl die Szene gemischt ist — beide per-Klasse-Prompts
  matchen dieselben Armature-Instanzen, einer überscored konsistent den anderen → 2D-Maskengeometrie kann das
  nicht fixen (zufällige 3D-Posen lassen Pixel-Längen überlappen).
- **AK3 — geprüft, ob spezifischere per-Klasse-Prompts helfen:** explizite Prompts
  `{anker_kurz: "short anchor bolt", anker_lang: "long anchor bolt"}` → **0 Masken** auf beiden Frames.
  Zu out-of-domain; der präzisere Wortlaut killt den Recall komplett. **→ Hilft NICHT, schadet.** Die
  generischen Armature-Prompts sind der Recall-Sweet-Spot; die Klassen-Labels bleiben approximativ.

**Konsequenz für die Batch-Eval:** Die sam3-Kombis (3/6/7) werden bei der **Klassifikation** schwächer abschneiden
als die yolo-seg-Kombis — erwartetes Verhalten, kein Service-Defekt. **YOLO bleibt die klassifizierende Quelle;**
sam3 liefert training-free klassenagnostische Instanz-Masken (Maske/Detection, nicht die Klasse).
Der validierte Upgrade-Pfad (Depth-PCA-Bänder kurz/lang, `app.py`-Docstring) ist hier NICHT implementiert.

### VRAM (Input für S-007)

| Phase | sam3-Prozess (additiv zur 1684-MiB-Baseline) |
|---|---|
| idle (Modell geladen, offline) | **~1.88 GB** (1928 MiB) |
| /segment-Inferenz peak | **~3.42 GB** (3416 MiB) |

Live-Worker (PID 84836) durchgehend **unangetastet** (1684 MiB). Service nach Smoke gestoppt → GPU clean
**1693 MiB**. Für S-007: sam3 ist eine **Seg-Quelle** → Empfehlung wie yolo-seg/yolo-obb: persistent halten;
der idle-Footprint (1.88 GB) ist mittel, der Inferenz-Peak (3.42 GB) höher als yolo (0.42 GB), aber unkritisch
neben einem einzelnen residenten Pose-Modell.

### GPU-Koordination (T-146 ‖ Kais S-008-Training)

Smoke bewusst zügig gehalten (laden → /segment → stop). Während des Smokes lief Kais yolo26m-Training (S-008)
noch nicht; nach dem Smoke Service gestoppt + GPU für Kai freigegeben (1693 MiB), im Bus gemeldet.

### compose-/Service-Edits (committed auf `team/multipipe/integration`)

- `sam3-svc/Dockerfile`: `FROM foundationpose:blackwell-sam3` → `foundationpose:ampere` + transformers-Layer +
  default-offline-ENV.
- `sam3-svc/requirements.txt`: `transformers==5.10.2` + `accelerate` (torch NICHT — kommt vom Base).
- `docker-compose.yml` sam3-svc: `HF_HOME=/hf_cache` (= Mountpunkt), Mount + Default-Pfad auf
  `/mnt/data/kip_pose_weights/hf_cache`, `HF_HUB_DISABLE_TELEMETRY`, Kommentar-Doku.
- `.env.example`: `HF_CACHE_DIR` Default auf den Box-Pfad.

### Wiring der Kombis 3/6/7 (AK5) — bereits registriert, KEIN Whitelist-Edit nötig

Die 7-Kombi-Whitelist ist **Code, nicht Config**: `project/pipelines/combos.py` registriert die 6 NICHT-A-Kombis
(inkl. **alle drei sam3-Kombis** #3 `sam3→FoundationPose`, #6 `sam3→GigaPose-3D`, #7 `sam3→GigaPose-2D`)
**unbedingt** beim `import pipelines` (`_autoload_combos`), unabhängig davon, ob der Service erreichbar ist
(„lokal nicht erreichbar zu sein ist OK"). Das Gateway-Seg-Registry `INFER_SOURCES` enthält `sam3` bereits
(`gateway/app.py:53`) und `SAM3_URL` ist verkabelt (`docker-compose.yml`). Die sam3-Kombis waren also **nie
durch eine fehlende Whitelist-Eintragung blockiert** — nur durch den nicht laufenden Service. Verifiziert:
`pytest tests/test_composed_pipeline.py` = **14/14 grün** (asserted: Whitelist=genau 7, alle 3 sam3-Kombis
present mit korrektem `needs_depth`, GDRNPP=Monolith=Kombi 1). Mit dem jetzt laufenden sam3-svc sind 3/6/7
**am Netz** = real ausführbar. **Alle 4 Modell-Typen** (yolo-seg, yolo-obb, sam3, fp/gigapose) sind damit live.
