# Frontend — KIP Pose Viewer

Three.js-basierter 2-Screen-Web-Viewer für die KIP-Pose-Pipeline.
Production-URL: <https://max-utils.com/KIP/>.

## Struktur

| Pfad | Inhalt |
|------|--------|
| `kip.html` | Entry-Point — Tabs (Reales Foto / Simulation), Top-Bar, Canvas, PiP, Legende |
| `src/kip.js` | Tab-Wiring, API-Calls, Ladebalken-Polling, Reset-View, PiP-Toggle/Fullscreen |
| `src/scene.js` | Three.js-Setup, Cell-/Teile-Loading, View-Persistenz, Raycast-Ground-Clamp |
| `src/loadPose.js` | Lädt + validiert `pose_result.json` gegen den Contract |
| `src/partMeshes.js` | Lädt CAD-Meshes (PLY bevorzugt, GLB-Fallback), Recenter auf AABB |
| `src/partRegistry.js` | Teile-Liste + Fallback-Boxgrössen |
| `src/origin.js` | Tisch-Nullpunkt-Marker |
| `src/kip.css` | KIP-Viewer-spezifische Styles (Top-Bar, Tabs, Modell-Dropdown, Bar, Card) |
| `src/style.css` | Grund-Styles (Body, PiP, Legend, Status) |
| `assets/cell_web.glb` | Web-leichte Maschinen-Zelle (~27 MB, Default) |
| `assets/cell.glb` | Fallback-Variante (~8 MB) |
| `assets/parts/ply/obj_*.ply` | Original-BOP-CAD-Meshes (alle 6 Teile, hi-res) |
| `assets/parts/*.glb` | GLB-Fallback (decimated) für jedes Teil |
| `vendor/three/` | Three.js + Loader-Add-ons lokal eingebunden |

## Verhalten

- **Reales Foto** — Upload, async-Job mit Phase-Bar, Render im 3D-Modell
- **Simulation** — Klick „Neue Szene live generieren" startet Isaac-Sim auf der
  GPU (~80 s), keine gecachten Daten
- **View+Zoom-Persist** — localStorage `kip.viewer.view.v1` (debounced 350 ms)
- **Reset-View** — Button in der Top-Bar
- **Ground-Clamp** — per-Teil Raycast gegen `cellGroup` → echte lokale Tisch-Höhe
- **PiP-Fullscreen** — ⛶-Button vergrössert auf ~4-fache Fläche, Esc bricht ab

## Lokal entwickeln

Die Files sind statisch — jeder simple HTTP-Server tut es:

```bash
cd project/frontend
python -m http.server 8000
# http://localhost:8000/kip.html?…
```

Für API-Calls muss der Workstation-Webservice erreichbar sein (Tailscale oder
public via `max-utils.com/KIP`). `src/kip.js` nutzt relative API-URLs (`./api/`),
also läuft das Frontend identisch unter `/KIP/` (public) und `/` (lokal).
