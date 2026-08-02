# Mesh-HTTP-Contract — FROZEN v1 (Multi-Pipeline-POSE)

> **Status: FROZEN.** Single-Source-of-Truth für den HTTP-Contract des Pose-Service-Mesh.
> Eingefroren 2026-06-07 (T-128).
> Abgeleitet aus dem **echten Code** der vendored Services (`gateway/app.py`,
> `yolo-svc/app.py`, `sam3-svc/app.py`, `fp-svc/app.py`, `gigapose-svc/app.py`) — NICHT
> aus Doku-Hörensagen. Verwandt: ADR-021 (Multipipe-Service-Mesh),
> ADR-017 (frozen `pose_result`), `project/pipelines/contract.py`.
>
> **Diese Datei ist die Wahrheit.** Jede Änderung am Wire-Contract braucht ein Contract-Bump
> (v1 → v2) + ADR-Update. Service-Interna sind frei; die hier definierte Grenze ist eingefroren.

---

## 0. Versionierung

| Feld | Wert |
|---|---|
| Contract-Version | **v1** |
| Frozen am | 2026-06-07 |
| Quelle | vendored `project/mesh/{gateway,yolo-svc,sam3-svc,fp-svc,gigapose-svc}/app.py` |
| Klassen-Scope | **2 Klassen** (`anker_kurz`, `anker_lang`) — D1, Zahnrad out-of-scope |
| GPU-Ziel | RTX 3090, Ampere **sm_86** (D2, Images rebuild) |
| Maschinen-Schema | `contract_schema.json` (neben dieser Datei) |

---

## 1. `T_cam_obj` — die Koordinaten-Konvention (DIE Falle, exakt definiert)

`T_cam_obj` ist das **einzige** Pose-Format, das durch das ganze Mesh fließt. Jeder Pose-Service
gibt es aus, das Gateway reicht es durch, das FE rendert es. Die Definition ist HART:

| Eigenschaft | Wert | Beleg im Code |
|---|---|---|
| **Form** | 4×4 Matrix, **row-major** (Liste von 4 Listen à 4 floats) | `fp-svc/app.py:93`, `gigapose-svc/app.py:160` (`reshape(4,4).tolist()`) |
| **Richtung** | **mesh → cam** (Objekt-Frame → Kamera-Frame). Synonyme: object→camera, model-to-camera. Ein Punkt `p_obj` im Mesh-Frame wird zu `p_cam = T_cam_obj @ [p_obj, 1]`. | `gateway/app.py:60-62`, `fp-svc` docstring `:14`, `gigapose_infer.estimate` return |
| **Kamera-Frame** | **OpenCV**: x rechts, y runter, **+z vorwärts in die Szene** | `gateway/app.py:22-23`, README `:325` |
| **Einheit** | **Meter** | `gateway/app.py:23`, `fp-svc/app.py:14`, GigaPose intern mm→m in `_coarse` |
| **Rotation** | obere-linke 3×3 = R_mesh→cam (orthonormal, det=+1) | — |
| **Translation** | rechte Spalte `[0:3,3]` = t in **Meter** im OpenCV-Cam-Frame | — |

**Beispiel (Identität an Position 30 cm vor Kamera):**
```json
"T_cam_obj": [[1,0,0,0.0],[0,1,0,0.0],[0,0,1,0.30],[0,0,0,1]]
```

**Abgrenzung gegen `pose_result` (ADR-017):** Unser eingefrorenes `pose_result`-Doc ist ein
**Welt-Frame** (Z-up, Meter rel. Tisch). `T_cam_obj` ist NICHT `pose_result`. Die Umrechnung
`T_cam_obj` (OpenCV-cam, m) → Welt-Frame ist `bop_adapter.bop_pose_to_world` + Snap; die
Rückrechnung ist `compare_pipelines.world_to_bop_cam` (round-trip-getestet). **Jeder neue
Pose-Service muss diese Umrechnung verifizieren** — Vorzeichen-/Transpose-Fallen
(`D_FLIP = diag(1,-1,-1,1)` ist USD→OpenCV, NICHT cam↔world). Siehe Pattern
„Cross-Frame-Eval-Rückrechnung".

**FE-Flip (NICHT Teil des Wire-Contracts, nur Info):** OpenCV → three.js via
`F = diag(1,-1,-1,1)`, `three_matrix = F · T_cam_obj`. Kein Mirror.

---

## 2. Seg-Service-Contract — `POST /segment`

Jeder Maskenlieferant (yolo-seg, yolo-obb, sam3) erfüllt EXAKT diesen Contract — drop-in austauschbar.

### Request
```jsonc
POST /segment   (application/json)
{
  "rgb_b64": "<base64 PNG, uint8 RGB>",         // PFLICHT
  "prompts": {"anker_kurz": "concept text"},    // optional, NUR sam3 (yolo ignoriert es)
  "K": [fx,0,cx, 0,fy,cy, 0,0,1],               // optional, NUR Quellen die K brauchen (depth-PCA-Upgrade)
  "depth_b64": "<base64 PNG uint16 mm>"         // optional, NUR Quellen die Depth brauchen (sam3 2-pass-upgrade)
}
```
> **Frozen-Befund:** Yannics aktueller `yolo-svc` + `sam3-svc` lesen **nur `rgb_b64`** (+ `prompts`
> bei sam3). `K`/`depth_b64` im Seg-Request sind ein **dokumentierter, noch nicht implementierter**
> Erweiterungspfad (sam3 depth-PCA-CAD-Size-Filter, `sam3-svc/app.py:24-34`). v1 friert sie als
> **optionale** Felder ein, damit der spätere sam3-Upgrade KEIN Contract-Bump braucht.

### Response
```jsonc
{
  "detections": [
    {
      "id": 0,                    // int, stabil über den Request (Gateway mergt Pose per id)
      "class": "anker_kurz",      // ∈ {anker_kurz, anker_lang}  (lowercase! s. §6)
      "conf": 0.91,               // float 0..1
      "mask_b64": "<base64 PNG, single-channel, 0/255>",   // PFLICHT: volle-Bild-Maske, NICHT crop
      "obb": [cx, cy, w, h, theta]   // optional, NUR yolo-obb (Oriented Box; θ in rad)
    }
  ]
}
```
> **`mask_b64` ist Pflicht** und die einzige Init für die Pose-Stage (KEINE BBox-Init; alle 3
> Pose-Estimatoren konsumieren eine Maske pro Instanz). yolo-svc rasterisiert sein Polygon zu
> einer Voll-Bild-Maske (`yolo-svc/app.py:78-79`); sam3 liefert binäre Maske `*255`
> (`sam3-svc/app.py:182`). **`obb` ist NEU in v1** (Yannics yolo-seg hat es nicht) — siehe §4
> GDRNPP-Kopplung.

### `GET /health` → `{"ok": true}`  (yolo, sam3). Gateway aggregiert (`gateway/app.py:292-311`).

---

## 3. Pose-Service-Contract — `POST /pose`

Jeder 6-DoF-Estimator (gdrnpp, foundationpose, gigapose) erfüllt EXAKT diesen Contract.

### Request
```jsonc
POST /pose   (application/json)
{
  "rgb_b64":   "<base64 PNG uint8 RGB>",        // PFLICHT
  "depth_b64": "<base64 PNG uint16 mm>",        // PFLICHT bei needs_depth=true, sonst weglassen (null)
  "K":         [fx,0,cx, 0,fy,cy, 0,0,1],       // PFLICHT, flat 9, row-major
  "iterations": 5,                              // optional (Refiner-Iterationen), default 5
  "hypotheses": 5,                              // optional, NUR gigapose (coarse top-K in Refine), default 5
  "pipeline":  "rgbd",                          // optional, NUR gigapose: "rgbd"|"rgb" (Depth/Kabsch-Gate)
  "instances": [                                // PFLICHT, mindestens 1
    {
      "id": 0,                                  // muss der Seg-Detection-id entsprechen
      "class": "anker_kurz",                    // PFLICHT (gigapose akzeptiert auch "class_name")
      "mask_b64": "<base64 PNG 0/255>",         // PFLICHT (Init-Maske)
      "obb": [cx,cy,w,h,theta]                  // optional, NUR gdrnpp braucht es (s. §4)
    }
  ]
}
```
> **Depth-Einheit:** uint16-PNG in **Millimeter**, jeder Service teilt intern `/1000` → Meter
> (`fp-svc/app.py:82`, `gigapose-svc/app.py:71`). Konsistent über alle Pose-Services.
> **`depth_b64`-Gating:** Das Gateway forwardet `depth_b64` NUR wenn `needs_depth=true`
> für die gewählte Pose-Source (`gateway/app.py:526-527`). RGB-only-Pfad sendet kein Depth.

### Response
```jsonc
{
  "poses": [
    {
      "id": 0,                    // = instances[].id, durchgereicht
      "class": "anker_kurz",
      "T_cam_obj": [[...4x4...]], // §1: OpenCV-cam, Meter, mesh→cam, row-major
      "score": 0.83,             // optional (fp lässt es weg; gigapose liefert es)
      "stage": "refined+kabsch"  // optional, NUR gigapose ∈ {coarse, refined, refined+kabsch}
    }
  ]
}
```
> **Skip-Verhalten (frozen):** Unbekannte Klasse oder leere Maske → Instanz wird **still
> übersprungen** (`fp-svc/app.py:87-88`, `gigapose-svc/app.py:152-154`), NICHT als Fehler.
> Die Pose-Liste kann also kürzer sein als `instances`. Das Gateway mergt `conf`+`mask` per `id`
> zurück (`gateway/app.py:548-558`).

### `GET /health`
- fp-svc: `{"ok": true, "classes": [...], "cuda": bool}` (`fp-svc/app.py:74-76`)
- gigapose-svc: `{"status": "ok|loading|error", "dataset", "refiner", ...}` (`gigapose-svc/app.py:95-103`)
- **Für VRAM-Lifecycle (ADR-021):** der Health-State trägt das Lade-Stadium (`loading|ready|evicted`)
  — der Eviction-Manager pollt ihn. gigapose nutzt schon `status: loading|ok|error`.

---

## 4. GDRNPP-Kopplungsregel (NICHT frei kombinierbar)

GDRNPP (= unsere Pipeline-A-Hauptlinie, der `gdrnpp-svc` ist in S-001 noch zu bauen) ist die
**einzige** nicht-frei-kombinierbare Pose-Source. Gründe:

1. **GDRNPP braucht OBB-Boxen**, keinen generischen Masken-Input. → Es konsumiert das `obb`-Feld
   der Detection/Instance, das **nur `yolo-obb` liefert**. sam3 + yolo-seg liefern kein `obb`.
2. **GDRNPP nutzt per-Objekt-trainierte Checkpoints** (kein Mesh-Input zur Laufzeit). Ein Objekt
   ohne Checkpoint kann GDRNPP nicht posieren.
3. Darum: **GDRNPP ist HART an `yolo-obb` gekoppelt** = nur Kombi 1 (= Pipeline A). Das FE + das
   Gateway gaten das: wählt der User GDRNPP, wird der Seg-Dropdown auf `yolo-obb` gezwungen
   (und umgekehrt: yolo-obb erlaubt nur GDRNPP als sinnvollen Post — die anderen Posen wollen
   Maske, nicht obb). → **keine 12 Kombis, nur 7.**

> Implementierungs-Konsequenz für S-001 (gdrnpp-svc): der `/pose`-Service muss `obb` aus
> `instances[].obb` lesen statt `mask_b64`. v1 friert `obb` als optionales Instance-Feld ein.
> Der `mask_b64` darf bei GDRNPP-Instanzen weggelassen werden (oder wird ignoriert).

---

## 5. Die 7-Kombi-Whitelist (Dropdown-Gating + `needs_depth`)

Seg-Source × Pose-Source. Nur diese 7 sind valide (`gateway` `INFER_SOURCES` × `POSE_SOURCES`
+ GDRNPP-Kopplung). `needs_depth` steuert, ob das Gateway `depth_b64` forwardet
(`gateway/app.py:71-97`, Felder `needs_depth`/`rgb_only`).

| # | Seg | Pose | `pose_source`-id (Gateway) | `pipeline` | **needs_depth** | rgb_only | Anmerkung |
|---|---|---|---|---|---|---|---|
| 1 | **yolo-obb** | **GDRNPP** | `gdrnpp` *(S-001)* | — | **nein** (RGB; opt. depth-refine) | — | **= Pipeline A** (Hauptlinie, Genauigkeits-King). GDRNPP nur hier (§4). |
| 2 | yolo-seg | FoundationPose | `foundationpose` | — | **JA (Pflicht)** | nein | FP ist RGB-D-only (`fp-svc` verlangt `depth_b64`). |
| 3 | sam3 | FoundationPose | `foundationpose` | — | **JA (Pflicht)** | nein | sam3-Masken → FP. sam3 trennt kurz/lang nicht zuverlässig (§6). |
| 4 | yolo-seg | GigaPose-3D | `gigapose_rgbd` | `rgbd` | **JA (Pflicht)** | nein | coarse + GenFlow + **Kabsch/ICP auf Depth-Pointcloud**. |
| 5 | yolo-seg | GigaPose-2D | `gigapose_rgb` | `rgb` | **nein** | **JA** | coarse + GenFlow, **kein Depth, kein Kabsch**. |
| 6 | sam3 | GigaPose-3D | `gigapose_rgbd` | `rgbd` | **JA (Pflicht)** | nein | |
| 7 | sam3 | GigaPose-2D | `gigapose_rgb` | `rgb` | **nein** | **JA** | |

**needs_depth-Regel kompakt:**
- **Depth Pflicht:** FoundationPose (immer), GigaPose-3D (`pipeline=rgbd`, für Kabsch).
- **Kein Depth:** GDRNPP (Pipeline A, RGB; Depth nur optionaler Refine), GigaPose-2D (`pipeline=rgb`).
- GigaPose-2D + GigaPose-3D teilen **EINEN** Modell-Load — der Unterschied ist allein das
  `pipeline`-Feld im Request (kein Service-Swap, `gigapose-svc/app.py:132` `kabsch = pipeline=='rgbd'`).

**Depth-Verfügbarkeit pro Modus (UX-Konsequenz):**
- **Live (Zivid) + Sim (SDG):** Depth liegt immer vor → alle 7 Kombis wählbar.
- **Upload:** User muss Depth liefern → Kombis 2/3/4/6 brauchen Depth-Upload-Pflichtfeld;
  ohne Depth nur Kombis 1/5/7 wählbar. FE blendet die depth-pflichtigen Kombis ohne Depth aus.

**Gateway-Predict-Wrapper (FE-facing, multipart):** `POST /predict`
mit `rgb, depth?, fx,fy,cx,cy, seg_source, pose_source, iterations, top_n?, want_pointcloud?,
gt_masks?, seg_prompts?` → `{width,height,K,instances:[{id,class,conf,T_cam_obj,mask_b64}],
timings:{seg_ms,pose_ms,num_posed}}` (`gateway/app.py:460-583`). Das FE kennt nur dieses
eine Endpoint, NICHT die 7 Services — Komposition ist serverseitig.

---

## 6. 2-Klassen-Alignment (verifiziert, NICHT erweitert)

**Scope (D1):** genau 2 Klassen `anker_kurz`, `anker_lang`. Zahnrad explizit out-of-scope.
**obj_id-Mapping ist über alle Services KONSISTENT:**

| Quelle | Mapping | Beleg |
|---|---|---|
| yolo-svc | `{0:"anker_kurz", 1:"anker_lang"}` (class-id → name) | `yolo-svc/app.py:28` |
| gigapose_infer | `CLASS_TO_OBJ_ID = {"anker_kurz":1, "anker_lang":2}` | `gigapose/gigapose_infer.py:42` |
| fp-svc | `CLASS_TO_MESH = {"anker_kurz":"anker_kurz.obj", "anker_lang":"anker_lang.obj"}` | `fp-svc/app.py:31` |
| GigaPose models | `obj000001.obj`=anker_kurz, `obj000002.obj`=anker_lang | `datasets/kip2/models/` |
| sam3-svc | prompts `{anker_kurz:"short...", anker_lang:"long..."}` | `sam3-svc/app.py:67-69` |
| **Unser** `bop_adapter` | `OBJ_ID_TO_PART = {1:"Anker_Kurz", 2:"Anker_Lang", ...}` | `project/bop_adapter.py:52-54` |

→ **Numerisch alignt alles: anker_kurz=obj_id 1, anker_lang=obj_id 2.** ✓

### ⚠️ GEFUNDENE INKONSISTENZEN (Input für S-003 / S-006)

1. **Capitalization-Mismatch (die Mapping-Falle):**
   - Das ganze **Mesh** (yolo/sam3/fp/gigapose) nutzt **lowercase** `anker_kurz` / `anker_lang`.
   - **Unser `bop_adapter`** nutzt **CamelCase** `Anker_Kurz` / `Anker_Lang` als Keys in
     `OBJ_ID_TO_PART`, `PART_SYMMETRY`, `PART_LONG_AXIS` (`bop_adapter.py:52-104`).
   - **Folge:** Die `ComposedPipeline` (S-003) MUSS beim Mappen `T_cam_obj`→`pose_result` die
     Mesh-Klasse (`"anker_kurz"`) auf den Registry-Part (`"Anker_Kurz"`) umsetzen, sonst greifen
     `PART_SYMMETRY.get("anker_kurz")` → `None` und die Symmetrie-Kanonisierung wird still
     übersprungen → Eval-Müll bei symmetrischen Ankern.
   - **Fix-Ort (S-003):** eine `MESH_CLASS_TO_PART = {"anker_kurz":"Anker_Kurz", "anker_lang":"Anker_Lang"}`-
     Map im `ComposedPipeline`-Adapter (oder `bop_adapter.part_for_obj_id(CLASS_TO_OBJ_ID[cls])`,
     das gibt schon den CamelCase-Part). **Letzteres ist sauberer** — geht über obj_id, vermeidet
     eine zweite Mapping-Tabelle.

2. **Symmetrie ist NUR auf unserer Seite definiert:**
   - `PART_SYMMETRY["Anker_Kurz"/"Anker_Lang"] = {type:"continuous", axis:[0,1,0]}` (rotationssym.
     um Body-Y, `bop_adapter.py:86-87`). Anker sind **continuous-Y / sym-aware**.
   - Das **Mesh kennt keine Symmetrie** — die Pose-Services geben rohe `T_cam_obj` ohne
     Kanonisierung. Das ist OK: die AR-Eval (`eval_bop.py --icbin`) ist via `bop_toolkit`
     **symmetrie-bewusst** und rechnet das selbst. Aber: für **deterministisches Overlay-Rendering**
     (Viewer) muss S-003 die Kanonisierung (`canonicalize_rotation`) nach dem `T_cam_obj`→Welt-Mapping
     anwenden — sonst flackern continuous-symmetrische Anker im Yaw.

3. **GigaPose `models_info.json` ist in Millimeter, die `.obj` in Meter:**
   - `models_info.json`: `diameter 128.58`, `size_y 124.0/139.0` → das sind **mm** (BOP-Konvention).
   - Aber `RigidObject(..., mesh_units="m")` (`gigapose_infer.py:118`) lädt die `.obj` als **Meter**.
   - → Konsistent (Templates wurden BOP-mm gerendert, Mesh ist m), aber **eine Falle**: wer
     `models_info` für Skalierung nutzt, muss mm annehmen; wer das `.obj` nutzt, Meter. **Nicht
     mischen.** Für S-001/S-006 relevant wenn neue models_info erzeugt werden.

4. **3-vs-2-Klassen-Asymmetrie (bewusst, kein Bug):**
   - Unser `bop_adapter` kennt 6 obj_ids (inkl. Zahnrad=6). Das Mesh kennt nur 2.
   - **Im Batch-Eval (S-007) zählen nur die 2 Anker-Klassen** (apples-to-apples gegen Pipeline A,
     D1). Pipeline A bleibt funktional 6-klassig, aber der Vergleich filtert auf {1,2}.
   - **KEINE Erweiterung des Mesh auf Zahnrad in dieser Session** (D1). Zahnrad = Follow-up-Backlog.

---

## 7. CAD-/Asset-Mounts (NICHT vendored — Referenz)

Die großen Binär-Assets sind **bewusst nicht** nach `project/mesh/` kopiert (Vendoring nur des
Code-Layers, 124K). Sie bleiben Mount-Referenzen (docker-compose `volumes`):

| Asset | Quelle | Mount-Ziel | Einheit |
|---|---|---|---|
| FP-Meshes `anker_{kurz,lang}.obj` | `_integration/kip-pose-detection/assets/meshes/` | `/assets/meshes:ro` | **Meter** |
| GigaPose-Meshes `obj00000{1,2}.obj` | `_integration/gigapose/datasets/kip2/models/` | `/workspace/GigaPose/...` | **Meter** (`.obj`) |
| GigaPose-Templates (162/Objekt) | `_integration/gigapose/datasets/templates/kip2/{000001,000002}/` | mit GigaPose-Checkout | mm-gerendert |
| FP-Weights (refiner+scorer) | upstream Google-Drive (nicht im Repo) | `/workspace/FoundationPose/weights:ro` | — |
| GigaPose-Weights `gigaPose_v1.ckpt`, megapose | download-scripts / Box `/mnt/data/bop/weights` | `pretrained/` | — |
| sam3 HF-Snapshot `facebook/sam3` | gated HF (manueller Download auf Box) | `/hf-cache:ro` (offline) | — |
| yolo-Weights `best.pt` | `assets/weights/best.pt` (bundled) oder Training-Run | `/weights/best.pt:ro` | — |

> **Begründung Vendoring-Strategie:** **Copy/Vendor des Code-Layers, NICHT Git-Submodule.**
> Wir müssen die Dockerfiles für **sm_86** rebuilden (D2) und das Mesh um `gdrnpp-svc` + `yolo-obb-svc`
> erweitern → die Forks würden divergieren, ein Submodule-Pin gegen Yannics Upstream bremst nur.
> Der Code-Layer (124K) ist klein genug zum Vendoren; die schweren Assets bleiben Host-Mounts
> (kein Git-LFS-Ballast). Die FoundationPose/gigapose-**Repos** selbst bleiben separate Checkouts
> (mounted), weil die Service-Apps in-repo-Adapterklassen importieren (`gigapose_infer`,
> `estimater`) — die sind zu groß und ändern sich unabhängig.

---

## 8. Was für sm_86 + unsere Erweiterung noch fehlt (Pointer, nicht Teil von v1)

Dieser Contract ist die Grenze; die folgenden Build-Tasks füllen die Services dahinter (eigene Stories):

- **S-001 (läuft):** Base-Images `foundationpose:ampere` + `gigapose:ampere` (`TORCH_CUDA_ARCH_LIST="8.6"`),
  pytorch3d/nvdiffrast/xformers neu kompiliert. **Berührt diesen Contract NICHT** (nur Build-Layer).
- **gdrnpp-svc:** neuer Pose-Service, liest `obb` statt `mask_b64` (§4), per-Objekt-Checkpoints,
  `gdrnpp`-id im `POSE_SOURCES`-Registry des Gateways ergänzen.
- **yolo-obb-svc:** neuer Seg-Service, unser OBB-Detektor, liefert `obb`-Feld (§2), `yolo-obb`-id
  im `INFER_SOURCES`-Registry ergänzen.
- **Gateway-Erweiterung:** `POSE_SOURCES["gdrnpp"]` + `INFER_SOURCES["yolo-obb"]` + die GDRNPP-Kopplungs-
  Gating-Regel (§4) im `/predict`-Pfad + `/sources`/`/pose_sources`.
- **VRAM-Lifecycle (ADR-021):** persistente Seg + LRU=1 Pose-Swap, kooperativ pro Service über den
  `/health`-State.

---

*Frozen: T-128, 2026-06-07. Quelle = vendored Code, nicht Doku. Änderungen am
Wire-Contract = v2-Bump + ADR. Maschinen-Schema: `contract_schema.json`.*
