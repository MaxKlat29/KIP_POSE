# POSE — 3D Pose Viewer

Interactive Three.js viewer for `pose_result.json` (the frozen contract in
`project/pose_result.schema.json`). Renders the real cell CAD (`assets/cell.glb`:
table/cart + robot arm + trays) plus the **real per-part CAD meshes**
(`assets/parts/<part>.glb`) at each detected part's 6D pose, with orbit/zoom/pan,
a null-point marker, and a click-to-inspect info panel.

No build step — plain static web (ES modules + Three.js **vendored locally** in
`vendor/three/`, so it runs offline, no CDN).

## Run it

Serve **`project/`** (so the viewer at `/frontend/` can reach the inference
output in `/temp/`) and open the viewer:

```bash
# from project/
python3 -m http.server 8000 --bind 127.0.0.1
# then open:
#   http://127.0.0.1:8000/frontend/
```

> Use `127.0.0.1`, not `localhost` (Safari IPv6 quirk).

By default the viewer loads `../temp/pose_result.json` (the latest inference
output). To load a specific file — e.g. the frozen example fixture:

```
http://127.0.0.1:8000/frontend/?file=./test/pose_result.example.json
```

The `?file=` query overrides the default. Paths are resolved relative to
`frontend/`.

`e2e_infer.py --serve` is the one-command path: it runs the pipeline, writes a
schema-valid `temp/pose_result.json`, then serves `project/` so the viewer shows
the fresh result.

### Why a server and not a double-click?

Opening `frontend/index.html` directly via `file://` will fail to `fetch()` the
JSON (browsers block `file://` fetches as cross-origin) and the import map needs
an http(s) origin. `python3 -m http.server` is the robust, dependency-free way.
Any static server works (`npx serve`, etc.).

## Controls

| Action | Input |
|---|---|
| Orbit | drag (left mouse) |
| Zoom | scroll wheel |
| Pan | drag (right mouse) |
| Inspect a part | click it → info panel |
| Close panel | click empty space, or the × |
| Move null-point | **Shift**+click on the table |

## What you see

- **Cell CAD** (`assets/cell.glb`): the real table/cart, robot arm and trays,
  in the same Z-up metre world frame as the pipeline.
- **Real part meshes** (`assets/parts/<part>.glb`): each detected part's actual
  CAD geometry, placed at `R_world @ body + t_world`. Each template is re-centred
  on its own geometric bounding box at load time, so the visible mesh sits exactly
  on the predicted `t_world` even if a GLB was exported off-origin. A part with no
  exported mesh (or an empty/degenerate GLB) falls back to a coarse box sized per
  `src/partRegistry.js` (never a crash).
- **Cyan-blue** = lying flat, **amber** = `upright: true` (stands on its head).
- **Green triad + ring** = the null-point.
- Each part carries a small body-frame axes triad so its orientation reads.
- Parts render in `depth_order` so stacking matches the scene.

## Rotation convention

`R_world` is 9 floats, **row-major**, **column convention** `world = R @ body`
(same as the `faces_<part>.json` registry). It maps straight into the top-left
3×3 of a `THREE.Matrix4` via `Matrix4.set(...)` (which takes row-major args) —
**no transpose**. The single place this lives is `rotationToMatrix4()` in
`src/loadPose.js`; if a producer ever ships row convention, add `.transpose()`
there.

## Files

| File | Role |
|---|---|
| `index.html` | markup + import map (vendored three) + HUD/panel scaffolding |
| `src/style.css` | overlay UI styling |
| `src/main.js` | entry point: boot scene, load pose, place meshes, wire panel |
| `src/scene.js` | renderer, Z-up camera, lights, table, cell + part placement |
| `src/partMeshes.js` | loads + caches the real per-part CAD meshes (`assets/parts/`) |
| `src/loadPose.js` | fetch + validate + **R→Matrix4 conversion** |
| `src/partRegistry.js` | box-fallback sizes per part (used when a mesh is missing) |
| `src/origin.js` | null-point marker (S-011) |
| `src/infoPanel.js` | raycast click → info panel (S-011) |
| `assets/cell.glb` | the cell CAD (table + arm + trays) |
| `assets/parts/*.glb` | the per-part CAD meshes |
| `assets/part_meta.json` | per-part centroid offset + extent (export metadata) |
| `test/pose_result.example.json` | frozen schema-valid fixture (3 parts) |
| `test/fixtures/{empty,sparse,full,defensive}.json` | robustness fixtures (0 / 3 / 9 parts + malformed entries) |
| `test/rotation.test.mjs` | node smoke test pinning the R-convention |
| `test/viewer_scenarios.py` | Playwright: EMPTY / SPARSE / FULL end-to-end |

## Robustness

The viewer is built to survive any input the pipeline throws at it:

- **0 parts** (or a missing default `temp/pose_result.json`) → the cell + null-point
  render and a *"Keine Teile erkannt"* hint shows; no error, no crash.
- **Many parts** (10+) → all placed in `depth_order`; raycast picks the front part.
- **Unknown part name** → coarse box fallback (still pickable), no missing-GLB crash.
- **Missing / malformed fields** → coerced to safe defaults, broken entries dropped
  (logged to the console), the rest still render.
- An explicit `?file=` that 404s or isn't JSON → a clear error toast (the user asked
  for that file). A missing *default* file is treated as an empty scene.

## Test

**Rotation convention** — dependency-free node test pinning the R math against the
frozen example (verifies it isn't silently transposed):

```bash
node frontend/test/rotation.test.mjs
```

**End-to-end scenarios** — headless Playwright across EMPTY / SPARSE / FULL,
asserting real CAD meshes (not boxes), zero console/page/request errors, orbit, and
click→info-panel. Serve `project/` first, then point the test at the port:

```bash
# from project/
python3 -m http.server 8000 --bind 127.0.0.1 &
python3 frontend/test/viewer_scenarios.py 8000
```

Screenshot a scene for review (worktree-agnostic — SHOT / PORT / FILE are args):

```bash
python3 frontend/test/viewer_shot.py temp/viewer.png 8000 ./test/fixtures/full.json
```
