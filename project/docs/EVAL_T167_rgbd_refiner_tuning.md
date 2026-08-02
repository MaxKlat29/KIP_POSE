# T-167 — RGB-D-Refiner-Tuning (FoundationPose + GigaPose-3D)

**Session** `2026-06-07-multi-pipeline-pose` · **Ticket** T-167 · **Datum** 2026-06-08
**Vorbefund** T-166: RGB-D-Rotation top (~5°, sym-gefaltet, MSPD=1.000), aber
**Translation rauscht ~33-36 mm random** — knapp über der MSSD-Schwelle (11 mm = 0.1·Ø).
GDRNPP-RGB trifft die Translation (~4 mm) → führt mit AR 0.305 (6-obj). RGB-D bei ~0.22.
Kein Eval-/Symmetrie-/Mesh-Bug — der Hebel ist die Refiner-/ICP-Konfiguration.

Alle Messungen über die **fixen 10 SDG-val-Szenen** (000000–000009), AR IC-BIN sym-aware,
6-Objekt-Mean (= mit 4 untrainierten Klassen AR=0 → ×3-Dämpfung; relative Vergleiche valide).
Harness: `project/eval/tune_rgbd.py` (nutzt die `batch_eval`-Naht, kein Drift). Baseline-Lauf
zum Vergleich: `run-20260608T113628Z` (iter=5).

## Die gefundenen Knöpfe

| Estimator | Knopf | Wo | Steuerbar |
|---|---|---|---|
| FoundationPose | `iterations` (= `est_refine_iter` → `register(iteration=N)` → render-and-compare-Refine-Loop) | `fp-svc/app.py` ← Gateway `/predict` ← Runner `--iterations` | Runner-Flag (kein Build) |
| GigaPose-3D | GenFlow `iterations` (MegaPose `_refine` n_iterations, RGB) | `gigapose_infer._refine` ← gleicher Pfad | Runner-Flag |
| GigaPose-3D | **ICP `max_corr` / `iters` / estimation** (open3d-Kabsch auf Depth-Pointcloud) | `gigapose_infer._icp` — war **hardcoded** `max_corr=0.01, iters=50` | **NEU env-bar** (T-167-Patch) |
| GigaPose-3D | `GP_DEPTH_SNAP` (Strahl-Skalierung auf Median-Tiefe, projektions-erhaltend) | `gigapose_infer._icp` | env (vorbestehend, an) |

**T-167-Patch (`box_src/mesh_patches/patch_icp_t167.py`, idempotent, py_compile OK):** macht
`_icp` env-steuerbar — `GP_ICP_MAX_CORR` (Korrespondenz-Radius m), `GP_ICP_ITERS`,
`GP_ICP_ESTIMATION` (`point`|`plane`). 0 zusätzliches VRAM (open3d = CPU). `gigapose_infer.py`
liegt auf einem **Bind-Mount** (`$GIGAPOSE_DIR`) → Patch überlebt Recreates.
Compose-`environment:`-Block (host-overridebar) für die Sweeps; Default jetzt `0.025`.

## Befund 1 — FoundationPose: iter NICHT hochdrehbar (VRAM-Decke)

FP iter=5 solo reproduziert die Baseline sauber (**AR=0.2204**, 0% crash, pose_ms 7873).
**Aber VRAM-Peak = 24060/24576 MiB — schon bei iter=5 randvoll.** Strukturelle Enge: die
residenten Services (kip_server-Worker :8078 ~10 GB + yolo/sam3) fressen die Hälfte, FP-Peak
~13 GB. iter=10 sprengt die 24 GB → **502/OOM** (Service lebt, aber Request crasht). FP-iter
hochdrehen würde den :8078-Live-Worker antasten müssen (Mission-Verbot). → **FP-iter ist auf
diesem Box-Scope nicht der nutzbare Hebel.**

## Befund 2 — GigaPose-3D: ICP/GenFlow-Sweep saturiert sofort (+1.1 %)

GigaPose-svc ist leicht (~2.4 GB, ICP auf CPU) → frei tunebar. Vollständiger Sweep
(yolo-seg→GigaPose-3D, 10 Szenen):

| iter | max_corr | estimation | AR (6-obj) | pose_ms | Verdikt |
|---|---|---|---|---|---|
| 5 | 0.01 (Baseline) | point | 0.2195 | 1255 | — |
| 5 | **0.025** | point | **0.2219** | 2100 | **bester (+0.0024, +1.1 %)** |
| 5 | 0.05 | point | 0.2219 | 1711 | Plateau (kein weiterer Gewinn) |
| 5 | 0.04 | **plane** | **0.0547** | 1619 | **KOLLAPS** (−75 %) |
| 10 | 0.025 | point | 0.2133 | 2391 | schlechter (GenFlow überschießt) |

- **ICP-`max_corr` größer hilft minimal und saturiert sofort** (+1.1 %, dann flach). Der
  ~33 mm große laterale t-Fehler ist **random, nicht systematisch** → ICP zieht etwas mehr Punkte
  rein, aber es gibt keine konsistente Korrespondenz, die den Drift wegzieht.
- **PointToPlane @ 4 cm zerstört die Pose** (0.2195→0.0547): zu großer Radius + Flächen-
  Alignment latcht auf falsche Korrespondenzen, reißt die gute coarse+GenFlow-Ausrichtung weg.
- **Mehr GenFlow-Iterationen schaden** (0.2219→0.2133): die coarse+GenFlow-Pose ist lateral
  schon nahe am Optimum; weitere Iterationen überschießen.

## Verdikt (ehrlich, T-122-Kultur)

**Die obvious Refiner-Knöpfe drücken den ~33-mm-Translations-Fehler NICHT real.**
Der einzige saubere Gewinn ist GigaPose-3D `max_corr 0.01→0.025` = **+1.1 % AR** (0.2195→0.2219),
zum Preis von ~+850 ms. Das ist **innerhalb des Rauschens und ändert das Ranking nicht** —
GDRNPP-RGB (0.305) bleibt klar vorn, RGB-D bleibt 2. Klasse. FP-iter ist VRAM-blockiert.

**Das ist die Decke der generischen Refiner-Präzision auf diesem 2-Anker-D1-Scope.** Der
33-mm-Fehler ist random Refiner-Rauschen (T-166 bewiesen: kein Mesh-Origin-Offset), und
random Rauschen ist per Definition nicht durch schärfere/mehr-iterierte Geometrie-Refinement
korrigierbar — es bräuchte ein **besseres Translations-Prior** (z.B. depth-initialisierte
Translation wie FP's `guess_translation`, oder ein auf die Teile feingetuntes Modell), nicht
mehr ICP-Iterationen. **Kein voller 12×10-Re-Run gefahren** — das Ranking ist unverändert,
ein Re-Run würde nur die bestehende Tabelle mit +1.1 % auf den GigaPose-3D-Zeilen reproduzieren.

## Persistierter Endzustand

- `gigapose-svc` läuft auf `GP_ICP_MAX_CORR=0.025` / `point` / `iters=50` (bester schadensfreier
  Wert); Compose-Default `0.025`. fp-svc einmal neu gestartet (Cache-Free nach iter=10-OOM),
  reproduziert Baseline. **:8077/:8078 unberührt, kein pkill, alle Services healthy.**
- Run-Artefakte: `project/temp/batch_eval/tune-gp3d-{base,mc025,mc05,mc04plane,iter10}` +
  `tune-fp-iter5` (je results.json + EVAL.md) auf der Box.
- Box-Backups: `gigapose_infer.py.bak-T167`, `docker-compose.yml.bak-T167`.
