# T-178 — MoE-Pipeline: IR-Schatten-Raytracing + RGB/RGB-D-Hybrid-Routing

**Ticket:** T-178 · **Agent:** Claude (solo) · **Datum:** 2026-06-12
**Auftrag (Max):** Die Depth-Kamera wirft durch den Arm einen IR-Schatten auf den
Tisch — dort keine Tiefe, Teile mit RGBD nicht erkennbar. Raytracing-Sim des
Schattens bauen, Bereich auf dem Tisch umranden (eigener Reiter), Routing:
Teile im Schatten → RGB, sonst → RGB-D. „Extra Screen diese MoE-Pipeline."

## 1. Rig-Geometrie (von Max bestätigt)

- RGB-Kamera: Welt (450, 88, 1100) mm (aus scene_camera, fix)
- **IR-/Depth-Kamera: 15 cm links der RGB-Cam → (300, 88, 1100) mm**, gleiche
  Höhe/Montagelinie, geneigt (Neigung irrelevant für Punktquellen-Schatten)
- Arm: LARA5 in der Zivid-Detection-Pose (aus `cell.glb`, identisch Isaac)
- Tisch: x 57–783, y 42–537 mm, Platte z=0 (Welt)

## 2. Raytracing-Sim (`box_src/moe_shadow_sim.py`)

Eigener vektorisierter Möller–Trumbore-Raytracer (kein embree auf der Box,
isaacsim-venv unangetastet; Quelle fix pro Lauf → tvec/qvec/t-Zähler als
Dreieck-Konstanten vorberechnet). Occluder = `cell.glb` komplett (Arm +
Basiswagen + Trays, 279k Tris > z=0.02). 3 Quellen: IR, RGB-Cam (View-Schatten),
Nadir (Arm-Footprint).

**Structured-Light-Logik:** Depth fehlt, wo der PROJEKTOR blockiert ist ODER
die Kamera nicht hinsieht. Die nutzbare **MoE-RGB-Zone = IR-Schatten ∧ ¬View-
Schatten** (dort sieht RGB noch hin). Im View-Schatten sieht niemand etwas.

| Zone | Fläche | Anteil |
|---|---|---|
| IR-Schatten gesamt | 1569 cm² | 42.9 % |
| View-Schatten (RGB-Cam blockiert) | 1435 cm² | 39.3 % |
| **MoE-RGB-Zone (Routing-Bereich)** | **485 cm²** | **13.3 %** |

Dominanter Streifen am rechten Tischrand (Quelle links → Arm-Schatten fällt
nach rechts) + Patch im Arm-Zwischenraum. 2 Polygone.
Bilder: `recon/t178_shadow_topdown.png` (Operator-Sicht) +
`t178_shadow_overlay.png` (auf echtem Kamera-Frame).
**SSOT: `project/config/moe_shadow.json`** (Polygone Welt-mm + Rig-Parameter;
mtime-gecacht, neue Sim-Läufe greifen ohne Server-Restart).

## 3. Routing (Gateway `pose_source=moe`)

- **Kuratierte Spezial-Pipeline AUSSERHALB der Seg×Pose-Matrix** (Matrix/Eval/
  Dropdowns bleiben bei 12; Invarianten-Tests mit dokumentierter Ausnahme).
  Adressierung nur per `pipeline=moe`.
- seg=yolo-obb nativ (obb für den GDRNPP-RGB-Zweig, rasterisierte Maske für
  den RGB-D-Zweig).
- **Routing ohne Tiefe** (der Witz): Anker-Pixel (obb-Zentrum) → Sehstrahl →
  Schnitt mit Tischebene z_w=0 (via `moe_w2c`-Rig-Extrinsics) → Punkt-im-
  Polygon (`moe_zone`) → RGB- bzw. RGB-D-Zweig.
- Zweige: RGB = **GDRNPP** (0.886) · RGB-D = **GigaPose-3D** (0.838,
  lizenz-clean; `moe_rgbd_source=foundationpose` als Eval-Option — FP ist
  non-commercial). Response: `moe: {rgb_n, rgbd_n, ...}` + pro Instanz
  `route: "rgb"|"rgbd"`.

## 4. FE — neuer Reiter „MoE-Pipeline"

- `src/moe.js`: Three.js-Overlay — Zone rot gefüllt + umrandet auf dem Tisch,
  **IR-Scheinwerfer** (rötliches SpotLight + transluzenter Kegel von der
  IR-Quelle) + **Grenz-Rays** (additiv) von der Quelle zur Zonen-Kontur,
  View-Schatten dezent grau. Toggles „Schattenbereich/IR-Kegel anzeigen".
- Button „Neue Szene live generieren (MoE)" → `?pipeline=moe`; Status zeigt
  Routing-Zähler („X im IR-Schatten → RGB · Y → RGB-D").
- Daten: `GET /api/moe/shadow`.

## 5. Verifikation (live auf der Box)

| Check | Ergebnis |
|---|---|
| Tests lokal | **281 passed** (Invarianten mit MoE-Ausnahme dokumentiert) |
| `/api/moe/shadow` (auch via max-utils.com/KIP) | 200, ir (300,88,1100), 2 Polygone |
| `/api/pipelines` | weiterhin **12** (Matrix unberührt) |
| E2E Sim-Pfad `?pipeline=moe` (job e364a577) | Fertig, `moe: {rgb_n:0, rgbd_n:2}`, modality RGB+RGBD |
| **Beide Zweige** (val-Frame 000000/3 mit Zonen-Teil, Gateway direkt) | `rgb_n:1, rgbd_n:1` — anker_kurz→**rgb**, anker_lang→**rgbd** ✓ |
| FE-Edge | Tab+Screen-Marker im HTML, moe.js 200 |
| Pipeline A / Worker :8078 | unberührt (heilig) |

## 6. Offen / Hinweise

- Visuelle Abnahme des Kegels/der Umrandung im 3D = Max im Browser (Reiter
  „MoE-Pipeline" auf max-utils.com/KIP).
- Zone ist Rig-Kalibrierung: ändert sich Kamera-/Arm-Position → Sim neu
  laufen lassen (`moe_shadow_sim.py`), JSON nach `project/config/` — Server
  greift es ohne Restart.
- Real-Zivid-Validierung der Zone (echter IR-Schatten vs. Sim) steht aus —
  gleiche Messung wie die offene Real-Depth-Validierung aus T-177.
