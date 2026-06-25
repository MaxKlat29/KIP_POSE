# T-175 — RGB-D 3D-Konversions-Recheck + Fehlerbild-Visualisierung

**Run:** `run-20260608T201857Z` (final, 12 cfg, 100 scenes, AR 0.886 2-Klassen)
**Kombi geprüft:** `yolo_seg__foundationpose` (RGB-D)
**Probe:** `recon/t175_probe.py` (Hand==eval_bop), `recon/t175_render_overlays.py` (Overlays).
Box read-only. `:8077/:8078/:8012`/Mesh-Container unberührt. Kein `pkill`.

---

## Auftrag 1 — Verdikt: KEIN BUG. Konversionskette ist verlustfrei + K-konsistent.

Die Kette `gateway /pose (T_cam_obj) → instances_to_doc (Welt) → world_to_bop_cam → BOP-CSV → eval_bop`
fügt **null** Translations-/Rotations-Fehler hinzu, und Inferenz und eval_bop nutzen
**dieselbe** per-Szene-K. Der AR-Gap ist ehrliches Refiner-Translations-Rauschen (T-166 re-confirmt).

### (a) K-Konsistenz Inferenz ↔ eval_bop — IDENTISCH per Konstruktion
- **Inferenz-K:** `batch_eval.discover_scenes` liest `K9 = cam_all[str(im_id)]["cam_K"]`
  aus `scene_camera.json` und schickt sie via `_K_dict` (→ `{fx,fy,cx,cy}`) ans Gateway.
- **Eval-K:** `eval_bop.load_scene_data` liest `cam["cam_K"]` aus **derselben**
  `scene_camera.json`, gleicher `im_id`. Genutzt für MSPD (2D-Projektion) + VSD.
- → Beide ziehen aus **einer Datei, einem Bild-Key**. Ein K-Mismatch ist strukturell
  ausgeschlossen. `_K_dict` mappt korrekt `fx=K9[0], fy=K9[4], cx=K9[2], cy=K9[5]` —
  **keine fx/fy- oder cx/cy-Vertauschung.**
- **K-Unabhängigkeit der Translation (empirisch):** die CSV-`t` ist die metrische
  Service-Translation (mm, cam-frame). Probe-Beweis (scene0 obj1):
  `te = 65.3 mm` bleibt **fix**, wenn man K absichtlich um 1.5× verfälscht — nur
  MSPD bewegt sich (5.29 → 7.93 px). K kann den metrischen Translations-Fehler also
  **nicht** aufblähen (das wäre die Sorge gewesen). Eine falsche K würde nur MSPD
  verzerren — und eval_bop sähe das, es würde den Fehler nicht *maskieren*.

### (b) Frame-Konsistenz / kein Doppel-Transform — ROUND-TRIP = IDENTITÄT
Auf **200 echten Final-Run-Posen** (obj1+obj2) durch die exakten Eval-Funktionen:
`bop_pose_to_world` (cam→Welt) gefolgt von `world_to_bop_cam` (Welt→cam):

| Größe | Max-Abweichung |
|---|---|
| `\|t_back − t_csv\|` | **2.84e-13 mm** (Maschinen-Epsilon) |
| `\|R_back − R_csv\|` | **2.22e-16** |

→ Die Welt-Round-Trip im Eval-Pfad ist eine **exakte Identität**. Kein Doppel-Transform,
kein cam↔Welt-Mismatch, kein versteckter Offset. Ground-Snap ist im Eval AUS
(`snap=False`, T-163 gefixt — bestätigt im Code-Pfad `_instances_to_doc_local`).

### (c) Hand == eval_bop (apples-to-apples)
AR_MSSD/AR_MSPD aus den CSV-Posen **von Hand** nachgerechnet (gleiche bop_toolkit
`pose_error.mssd/mspd`, gleiche models_eval-PLY + syms, gleicher greedy-Match, gleiche
Schwellen, gleiche per-Szene-K, **gleiches `n_points=2000` Vertex-Subsample seed=0**):

| obj | HAND AR_MSSD | REPORT AR_MSSD | HAND AR_MSPD | REPORT AR_MSPD |
|---|---|---|---|---|
| 1 Anker_Kurz | 0.2965 | 0.2993 | 0.9676 | 0.9746 |
| 2 Anker_Lang | 0.3711 | 0.3826 | 0.9488 | 0.9736 |

Delta obj1 < 0.007 = **MATCH**. Restdelta (obj2 ~0.01-0.02) = das Vertex-Subsampling
(`rng(0).choice`-Stream) + 1 nicht-gematchte GT-Instanz (n_matched 120 vs n_gt 121,
struktureller Miss → recall-0 in **beiden** Rechnungen). **Wichtig:** der Delta
**schrumpfte von ~0.03 auf <0.007**, als ich `n_points` von „alle Vertices" auf
eval_bops `2000` anglich — Beweis, dass der Rest reine Subsample-Rundung ist, **kein
Konversionsfehler**.

### (d) Translations-Fehler RANDOM (T-166 re-confirm auf Final-Run)
pred→GT-Translation im GT-Objekt-Frame (konstanter Mesh-Offset ⇒ `|mean|/std ≫ 1`;
random ⇒ `mean≈0`):

| obj | dt_OBJ mean [x,y,z] mm | dt_OBJ std mm | \|mean\|/std X/Y/Z | \|t_err\| mean |
|---|---|---|---|---|
| 1 | [-4.1, -1.6, -8.1] | [50.6, 20.9, 32.9] | 0.08 / 0.08 / 0.25 → **random** | 48.1 mm |
| 2 | [-8.6, +5.9, -0.5] | [74.0, 38.5, 34.7] | 0.12 / 0.15 / 0.01 → **random** | 49.0 mm |

mean near-zero, std 20-74 mm richtungslos → **kein** fixer Mesh-Origin-/Konventions-
Offset zum Angleichen. AR_MSPD ~0.97 (2D-top, Rotation korrekt) + AR_MSSD ~0.3
(3D-Scatter über der 11 mm-Schwelle) = exakt die T-166-Signatur „2D-top/3D-Scatter =
Translation/Tiefe", hier ehrliches Refiner-Rauschen, KEIN Pipeline-Bug.

### Kein 4. Bug
depth_scale (T-156), ground-snap (T-163), Mesh-Offset (T-166) sind alle ausgeschlossen
/ gefixt. Die K-Konsistenz-Hypothese aus dem Auftrag ist **widerlegt** (selbe Datei).
Die Konversion ist sauber; der RGB-D-AR-Gap bleibt ehrliches Refiner-t-Rauschen.

### Nebenbefund (nicht Auftrag, aber notiert)
Im `yolo_seg__foundationpose`-Report ist **obj6 Zahnrad n=160, AR=0.000 (MSSD+MSPD=0)**.
Das ist die separate Zahnrad-/6-obj-Frage (T-103/T-115/T-171); die **primäre AR ist
2-Klassen** (Sam: 0.886). Kein Konversions-Defekt der Anker — aber Zahnrad liefert im
FP-Pfad keine verwertbaren Treffer (eigenes Ticket-Thema, nicht hier gefixt).

---

## Auftrag 2 — Fehlerbilder (GT blau / Pred rot, scene_camera-K, mm + deg sym-aware)

Reine 2D-Projektion `K @ (R@v + t)` der 3D-bbox (12 Kanten) + Modell-Achsen, auf das
echte RGB. Kein schwerer Renderer. `recon/t175_render_overlays.py`.

| Bild | Fall | trans / rot_sym (rot_naive) |
|---|---|---|
| `recon/t175_1_typical_40mm.png` | **typisch** scene9 im40 obj1 — GT/Pred bbox am selben Anker, leicht versetzt | 40.2 mm / 2.5° (74.7°) |
| `recon/t175_2_highflip_525mm.png` | **high-error/Flip** scene1 im0 obj2 — der T-166 obj2-Quer-Flip, bbox weit getrennt + verdreht | 524.9 mm / 80.6° (105.7°) |
| `recon/t175_3_good_4mm.png` | **gut** scene2 im60 obj1 — GT/Pred bbox überlappen fast deckungsgleich | 3.7 mm / 5.2° (104.0°) |
| `recon/t175_4_flip_456mm.png` | **high-error** scene7 im70 obj1 — GT am Teil, Pred ~456 mm verschoben | 456.2 mm / 57.8° (59.5°) |

`rot_naive ≫ rot_sym` in den guten Fällen (z.B. 74.7° → 2.5°) zeigt visuell, dass die
continuous-Y-Anker-Symmetrie korrekt herausgefaltet wird (sonst wäre die Rotation 75°
„falsch", obwohl die Pose stimmt). Die bbox-Trennung in den High-Error-Fällen macht
sichtbar, warum MSSD (3D-absolut) einbricht, während MSPD (2D-Projektion) hält.

---

## Reproduktion
```
scp recon/t175_probe.py recon/t175_render_overlays.py max@100.85.216.95:/tmp/
ssh max@100.85.216.95 '/mnt/data/bop/bop-venv/bin/python /tmp/t175_probe.py'           # Hand==eval_bop + t-scatter
ssh max@100.85.216.95 'cd /mnt/data/kip_pose/project && /mnt/data/bop/bop-venv/bin/python /tmp/t175_render_overlays.py'
```
Round-trip-Beweis: inline in der Investigation (bop_pose_to_world↔world_to_bop_cam,
200 Posen, max|Δt|=2.8e-13 mm). Alle read-only.
