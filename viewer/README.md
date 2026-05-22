# POSE — 3D Pose Viewer

Interactive Three.js viewer for `pose_result.json` (the frozen contract in
`docs/pose_result.schema.json`). Renders a table plane + a placeholder box per
detected part at its 6D pose, with orbit/zoom, a null-point marker, and a
click-to-inspect info panel.

No build step — plain static web (ES modules + Three.js via CDN import map).

## Run it

Serve the **repo root** (so the viewer can reach `data/examples/`) and open the
viewer:

```bash
# from the repo root
python -m http.server 8000
# then open:
#   http://127.0.0.1:8000/viewer/
```

> Use `127.0.0.1`, not `localhost` (Safari IPv6 quirk).

Loading a different result file:

```
http://127.0.0.1:8000/viewer/?file=../data/examples/pose_result.example.json
```

The `?file=` query overrides the default (the frozen example). Paths are
resolved relative to `viewer/`.

### Why a server and not a double-click?

Opening `viewer/index.html` directly via `file://` will fail to `fetch()` the
JSON (browsers block `file://` fetches as cross-origin) and the CDN import map
needs an http(s) origin. `python -m http.server` is the robust, dependency-free
way. Any static server works (`npx serve`, etc.).

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

- **Table plane + grid** in the XY plane (the world is Z-up, origin = table
  plane null-point).
- **Boxes** = placeholder CAD. Real USD CADs aren't web-loadable, so each part
  is a coarse box sized per `src/partRegistry.js`. Unknown parts get a generic,
  slightly translucent fallback box.
- **Blue** = lying flat, **amber** = `upright: true` (stands on its head).
- **Green triad + ring** = the null-point.
- Each part carries a small body-frame axes triad so its orientation reads.

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
| `index.html` | markup + import map + HUD/panel scaffolding |
| `src/style.css` | overlay UI styling |
| `src/main.js` | entry point: boot scene, load pose, wire panel |
| `src/scene.js` | renderer, Z-up camera, lights, table, OrbitControls, parts |
| `src/loadPose.js` | fetch + validate + **R→Matrix4 conversion** |
| `src/partRegistry.js` | placeholder box sizes per part |
| `src/origin.js` | null-point marker (S-011) |
| `src/infoPanel.js` | raycast click → info panel (S-011) |
| `test/rotation.test.mjs` | node smoke test pinning the R-convention |

## Test

A dependency-free node test pins the rotation math against the frozen example
(verifies the convention isn't silently transposed):

```bash
node viewer/test/rotation.test.mjs
```
