# Bug: RGB-D-Kombis AR=0.000 — systematischer ~2.4 m X-Translations-Fehler (T-156)

## Symptom
Kais 12×10-Eval: ALLE RGB-D-Kombis (FoundationPose + GigaPose-3D-rgbd) AR=0.000.
Szene 000000, obj 1 (anker_kurz): GT t = (-296, -205, 1059) mm.
- RGB-GigaPose (kein depth): (-252, -189, 978) mm — korrekt.
- RGB-D FoundationPose: (-2775, -104, 853) mm — X um ~-2.45 m daneben.
- RGB-D GigaPose-3D: (-2742, -156, 898) mm — fast bit-gleich falsch wie FP.

Schlüssel-Indiz: zwei unabhängige Refiner geben bit-gleich falsches X → der Bug
sitzt nicht im Modell, sondern upstream im gemeinsamen RGB-D-Pfad (depth+K), den
beide bekommen. Der RGB-Pfad (ohne depth) ist korrekt → der Bug sitzt nur im
depth-Handling.

## Repro (deterministisch, auf der Box)
`/mnt/data/kip_pose/project/bop/pose_isaac/val/000000` durch die Backprojection
`X=(u-cx)·Z/fx` jagen, mit der GT-Pixel-Tiefe und der echten K. Pro Szene/Instanz
exakt reproduzierbar, kein RNG.

Gemessen (scene 000000): cam_K cx=640 cy=360 fx=fy=1322.67; depth_scale=0.1;
depth-PNG uint16 median 11506, am GT-Pixel (269.9,104.3) raw=11131.

## Hypothesen
1. Crop-Origin fehlt (X-Shift durch fehlenden Offset) — VERWORFEN (kein Crop im
   gemeinsamen Pfad; Gateway reicht volle depth+K durch).
2. cx/cy falsch/vertauscht — VERWORFEN (cx=640=W/2, plausibel zentriert; der
   RGB-Pfad nutzt dieselbe K und ist korrekt → K ist nicht der Fehler).
3. **depth_scale=0.1 (BOP) wird ignoriert → Tiefe 10× zu groß** — BESTÄTIGT.

## Root Cause (bewiesen, numerisch)
Die uint16-Depth-PNGs sind im BOP-Format mit `depth_scale=0.1` geschrieben
(`box_src/isaac_to_bop.py:241` `dep_png = dep_mm / 0.1`, d.h. png-Wert = mm × 10).
Korrekte Decodierung: `metres = png × depth_scale / 1000 = png × 0.0001`.

Aber:
- **Gateway** (`mesh/gateway/app.py`) kennt/sammelt/leitet `depth_scale` nie.
- **Runner** (`eval/batch_eval.py discover_scenes/_K_dict`) zieht `depth_scale`
  nie aus `scene_camera.json`.
- **Beide Pose-Services** hardcoden `depth_m = png / 1000.0` (`fp-svc/app.py:82`,
  `gigapose-svc/app.py:71`) — Annahme: png ist direkt mm.

Folge: `png / 1000` statt `png × 0.0001` → Tiefe exakt 10× zu groß. Bei der
Backprojection `X=(u-cx)·Z/fx` skaliert X (und Y und Z) mit 10× → der laterale
X-Fehler (0.78–4.15 m je nach GT-X des Teils, ≈ Kais beobachtete ~2.4 m). Beide
Refiner sehen dieselbe verzerrte Punktwolke → bit-gleich falsch. Der RGB-Pfad
nutzt keine Tiefe → korrekt.

Numerischer Beleg (scene 000000, GT-Pixel, raw=11131):
| decode | X | Y | Z | GT (m) |
|---|---|---|---|---|
| korrekt `png×0.0001` | -0.311 | -0.215 | 1.113 | (-0.296,-0.205,1.059) ✓ |
| Bug `png/1000` | -3.115 | -2.152 | 11.131 | 10× zu weit ✗ |

## Fix (additiv, default 1.0 = mm — Live-Pfad byte-identisch)
`depth_scale` als optionalen Parameter end-to-end durchreichen; Decode überall auf
`metres = png × depth_scale / 1000`. Default 1.0 hält den echten mm-Sensor-Pfad
(Zivid/Realsense Live) unverändert; BOP/SDG-Frames liefern 0.1 aus ihrer
`scene_camera.json`.

- `mesh/gateway/app.py`: `depth_scale: float = Form(1.0)`; in `pose_payload`
  (wenn needs_depth) + in `_backproject_pointcloud`.
- `mesh/fp-svc/app.py`: `depth_scale: float = 1.0` im Schema; `depth = png *
  depth_scale / 1000`.
- `mesh/gigapose-svc/app.py`: dito, `_decode_depth(b64, depth_scale)`.
- `eval/batch_eval.py`: `discover_scenes` liest `cam["depth_scale"]` (default
  1.0); `http_predict` sendet `depth_scale` im Form-data wenn needs_depth.
- `kip_server.py`: `/api/predict` `depth_scale: float = Form(1.0)` durch
  `_gateway_predict_multipart` (Live-Default 1.0 = mm).

## Regression-Tests (`project/tests/test_batch_eval.py`)
- `test_discover_scenes` (erweitert) + `test_discover_scenes_depth_scale_defaults_to_one`
  — discover_scenes surface't depth_scale (0.1 aus scene_camera, 1.0 default).
- `test_http_predict_forwards_depth_scale_for_rgbd` — Runner sendet depth_scale
  fürs RGB-D-Combo, NICHT fürs RGB-only-Combo.
- `test_depth_decode_backprojection_honours_depth_scale` — numerischer Guard auf
  die Decode-Formel: korrekt → GT ±cm, Bug → exakt 10× / >2 m X.

Red→Green bewiesen: `git stash` der batch_eval-Code-Änderung (Tests behalten) →
3 Tests rot (`KeyError: 'depth_scale'`, `'depth_scale' not in data`); mit Fix grün.
Volle batch_eval-Suite 24/24 grün.

## Fix-Wirkung (Box, 14 Instanzen / 5 Szenen)
Backprojection-X-Fehler MIT Fix: 1–29 mm (median ~7 mm), Z 1.08–1.21 m (plausibel).
OHNE Fix (alt): X-Fehler 0.78–4.15 m, Z 10.8–12.2 m (10× zu weit). Der Input, den
FP + GigaPose-3D fürs Tiefen-Refinement/Kabsch bekommen, ist jetzt korrekt → beide
liefern eine plausible Pose (X nahe GT).

## Verdikt
**Prod-Bug gefixt.** Einheits-/Konventions-Drift (BOP depth_scale=0.1 nie
durchgereicht) im gemeinsamen RGB-D-Depth-Pfad. Minimal, additiv, default 1.0
hält den Live-mm-Sensor-Pfad unverändert.

## Deploy-Hinweis an Sam
Neu bauen/deployen müssen die 3 Mesh-Container, die den Depth-Pfad anfassen, +
das Gateway + kip_server:
- **fp-svc** (depth-decode), **gigapose-svc** (depth-decode), **gateway**
  (depth_scale-Form-Feld + Forward + Pointcloud) → Container-Rebuild.
- **kip_server** (`/api/predict` depth_scale-Form-Feld) → systemd-Restart wie
  T-154/T-155 (:8078 unberührt).
- **batch_eval.py** (Runner, reines Python) → auf der Box rsyncen wie sonst, dann
  den 12×10-Job re-triggern. Erst NACH dem Rebuild der Services geben die
  RGB-D-Kombis echte AR.
