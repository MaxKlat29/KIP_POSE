# project/mesh/ — vendored Pose-Service-Mesh

Vendored Code-Layer des Multi-Pipeline-POSE-Service-Mesh. Ursprung: Yannics
`kip-pose-detection` (ein ~80% fertiges, scaffolded-aber-nie-end-to-end-gefahrenes
Docker-Compose-Mesh). Hierher gebracht in T-128 (S-002, 2026-06-07).

## Was hier liegt

| Pfad | Was | Rolle |
|---|---|---|
| `CONTRACT.md` | **FROZEN v1 HTTP-Contract** (Wahrheit) | `/segment` + `/pose` + `T_cam_obj`, 7-Kombi-Whitelist, GDRNPP-Kopplung, 2-Klassen-Alignment |
| `contract_schema.json` | maschinenlesbares Schema | gegen das S-003 + Tests validieren |
| `gateway/` | CPU-Orchestrator (Fan-out seg→pose, Komposition, Gating) | Port 8000 |
| `yolo-svc/` | YOLO26n-seg `/segment` | Port 8001 |
| `fp-svc/` | FoundationPose `/pose` (RGB-D) | Port 8002 |
| `gigapose-svc/` | GigaPose `/pose` (2D+3D via `pipeline`-Flag) | Port 8003 |
| `sam3-svc/` | facebook/sam3 promptable `/segment` | Port 8004 |
| `gdrnpp-svc/` | **GDRNPP `/pose` (Pipeline A, liest `obb`)** — S-004, native venv (kein Docker) | Port 8012 |
| `tools/` | `sdg_to_gt_overlay.py`, `usd_scene_to_glb.py` (uv-CLIs) | — |
| `docker-compose.yml`, `.env.example` | Orchestrierung (Mounts via .env) | — |

## Vendoring-Strategie: Copy/Vendor (NICHT Submodule)

**Entscheid (T-128):** der Code-Layer wird **kopiert/vendored**, nicht als Git-Submodule gepinnt.
- Wir müssen die Dockerfiles für **sm_86** (RTX 3090, D2) rebuilden — Yannics Images sind hart auf
  Blackwell sm_120 kompiliert.
- Wir erweitern das Mesh um `gdrnpp-svc` + `yolo-obb-svc` (Pipeline A) → die Forks divergieren.
- Ein Submodule-Pin gegen Yannics Upstream würde nur bremsen.
- Der Code-Layer ist klein (124K) → Vendoren billig; die schweren Assets (Meshes, Templates,
  Weights) bleiben **Host-Mounts** (`CONTRACT.md §7`), kein Git-LFS-Ballast.

**NICHT hier vendored** (bewusst, bleiben Mount-Referenzen):
- FP-/GigaPose-Meshes + GigaPose-Templates → `_integration/{kip-pose-detection,gigapose}`
- FP-/GigaPose-/sam3-Weights → upstream-Download / Box `/mnt/data/bop/weights` / gated HF
- Die FoundationPose- + GigaPose-**Repos** selbst → separate Checkouts (mounted), weil die
  Service-Apps in-repo-Adapter importieren (`estimater`, `gigapose_infer`).

## ⚠️ Noch zu bauen (eigene Stories, NICHT Teil von S-002)

- **S-001 (läuft):** `foundationpose:ampere` + `gigapose:ampere` Base-Images (sm_86 rebuild).
- ~~`gdrnpp-svc` (liest `obb`, per-Objekt-Checkpoints)~~ **✓ S-004** — `gdrnpp-svc/` (Port 8012, native
  venv, det-getrieben/T-115-safe, Round-Trip-exakt vs Live-:8078, Kombi-1-E2E grün).
- `yolo-obb-svc` (liefert `obb`) — **✓ S-005** (Port 8011).
- ~~Gateway: `POSE_SOURCES["gdrnpp"]` + `INFER_SOURCES["yolo-obb"]` + GDRNPP-Kopplungs-Gating~~
  **✓ S-004** (gateway/app.py: gdrnpp+yolo-obb registriert, §4-Kopplung erzwungen + off-whitelist
  abgewiesen, `obb`-Forward).
- VRAM-Lifecycle (ADR-021): persistente Seg + LRU=1 Pose-Swap.

> **Dieses Verzeichnis startet KEINE Services.** Das Mesh läuft auf der Box (S-006). Hier liegt
> nur der eingefrorene Code + Contract als Plattform-Basis.

Referenzen: ADR-021 (Multipipe-Service-Mesh), `CONTRACT.md` (eingefrorener HTTP-Contract).
