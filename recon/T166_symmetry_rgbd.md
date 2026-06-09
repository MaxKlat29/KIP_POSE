# T-166 — Wird RGB-D unfair penalisiert? (Symmetrie-Faltung & Mesh-Integration)

**Theo (Debug, scientific method)** · 2026-06-08 · Branch `team/multipipe/T-166`
Daten-Quelle: Kais Re-Run #2 `temp/batch_eval/run-20260608T113628Z` (post-T-163-AR-Fix, post-T-156-depth_scale-Fix), 10 Szenen/Seeds, 2-Anker-Scope (obj 1 Anker_Kurz, obj 2 Anker_Lang).

---

## Symptom (Max' Frage)
RGB-D (FoundationPose AR 0.666, GigaPose-3D 0.658) deutlich schlechter als GDRNPP-RGB (0.915/0.893) auf dem 2-Anker-Scope. Hypothese: (a) die AR faltet die continuous-Y-Symmetrie nicht korrekt raus → penalisiert generische Modelle; und/oder (b) RGB-D ist falsch integriert (Mesh-Origin/Konvention ≠ GT → systematischer Offset).

## Reproduziert
Per-Objekt `report.txt` der Configs reproduziert Max' Zahlen exakt (2-Klassen-Mittel):
- FP: obj1 AR=**0.691**, obj2 AR=**0.641** → 0.666 ✓
- GDRNPP: obj1 AR=**0.953**, obj2 AR=**0.877** → 0.915 ✓

---

## Test 1 — Wird die Symmetrie wirklich gefaltet? → **JA, korrekt.**

**1a. Deklaration (models_eval/models_info.json).** Beide Anker tragen
`symmetries_continuous: [{axis:[0,1,0], offset:[...]}]` — Achse = **Y = Längsachse**,
korrekt. `models/` und `models_eval/` identisch. (Kontroll-Objekte: obj 6 Zahnrad
discrete C₆, obj 5 Ringmagnet cont-Y — alle plausibel.)

**1b. Wird sie im Eval angewandt?** `box_src/eval_bop.py:119` expandiert sie via
`misc.get_symmetry_transformations` (cont-Y → ~315 Transforms); `pose_error.mssd/mspd`
(canonical, unmodifiziertes `/mnt/data/bop/repos/bop_toolkit`) minimieren über die
syms — Standard-BOP-Faltung. Es gibt sogar einen eingebauten Self-Test
(`noise axis='sym'`), der genau das prüft.

**1c. Decisive empirical test (probe, n=27 matched).** Sym-resolved vs naive
Rotations-Fehler pro Modell:

| Modell | obj1 rot_naive | obj1 **rot_sym** | obj1 AR_MSPD |
|---|---:|---:|---:|
| FoundationPose | 76.7° | **5.0°** (med 4.4) | **1.000** |
| GigaPose-3D | 65.8° | 15.2° (med 5.7) | 0.991 |
| GDRNPP | 83.3° | 7.2° (med 3.2) | 0.969 |

Der naive Fehler (~77°) ist der continuous-Y-Twist, der gefaltet **auf ~5° kollabiert**.
Wäre die Symmetrie NICHT (korrekt) gefaltet, bliebe FP bei ~77° und sein MSPD wäre
zerstört. Stattdessen: **FP MSPD = 1.000, rot_sym = 5°** — FP's Rotation ist sogar
leicht **besser** als GDRNPP's (5.0° vs 7.2°).

→ **Hypothese (a) widerlegt.** Die continuous-Y-Symmetrie wird für die generischen
RGB-D-Modelle exakt so korrekt gefaltet wie für GDRNPP. RGB-D wird NICHT durch
fehlende Symmetrie-Auflösung penalisiert.

---

## Test 2 — Konstanter Offset (Mesh-Konvention, fixbar) oder echt schlechter? → **echt (random scatter).**

Pro matched Instanz (BOP-greedy-by-translation, identisch zu eval_bop) den pred→GT
Translations-Offset im **GT-Objekt-Frame** zerlegt. Konstanter Mesh-Origin-Offset ⇒
großer, konsistenter MEAN bei kleinem std (|mean|/std ≫ 1). Random ⇒ mean ≈ 0, großer std.

| Modell obj1 | dt_OBJ mean [x,y,z] mm | dt_OBJ std [x,y,z] mm | \|mean\|/std je Achse |
|---|---|---|---|
| **FoundationPose** | [-2, -7, -4] | [23, 14, 29] | 0.09 / 0.47 / 0.13 → **random** |
| **GigaPose-3D** | [-1, -7, -5] | [20, 16, 25] | 0.06 / 0.45 / 0.19 → **random** |
| GDRNPP | [-1, +2, -1] | [3, 5, 2] | klein, \|t\|=4.1mm med 2.3 |

Der Mittel-Offset ist **near-zero** (kein fixer Versatz), der **std ~25 mm** ist groß
und richtungslos. Per-Instanz-Dump (FP obj1) bestätigt: Fehler streuen in alle
Richtungen (`+33,-2,-9` / `-34,0,-9` / `+24,-47,-39` / `-51,8,-10` …), keine gemeinsame
Achse. **|t_err| FP obj1 mean = 36.5 mm (med 33.3)** gegen **GDRNPP mean = 4.1 mm** —
eine Größenordnung loser, aber **random**.

→ **Hypothese (b) widerlegt.** Kein konstanter Mesh-Origin-/Konventions-Offset zwischen
Onboarding-Mesh und Eval-Mesh. Es gibt nichts „anzugleichen". Die RGB-D-Refiner
(FP-render-compare, GigaPose-3D coarse+ICP) sind in der **Translation genuin ~30 mm
verrauscht** auf diesen Anker-Crops.

**Warum kollabiert das die AR?** MSSD-Schwelle = 0.1·diameter ≈ **11 mm**. Ein
random ~33 mm Translations-Scatter sitzt oberhalb → **AR_MSSD bricht ein**
(FP 0.381/0.291 vs GDRNPP 0.938/0.855), während **AR_MSPD** (2D-Projektion, blind gegen
laterales/Tiefen-Rauschen bei korrekter Rotation) bei ~1.0 bleibt. Exakt die
**T-163-Signatur „2D top, 3D null = Tiefen-/Translations-Offset"** — hier aber als
ehrliches Refiner-Rauschen, nicht als Pipeline-Bug.

**obj2-Ausreißer sind keine Integration:** die 2 großen obj2-Fehler (te 525 mm /
rot_sym 80°, te 159 mm / 32°) sind die dokumentierten **Anker-180°-Quer-Flips**
(T-083, Single-View-Ambiguität bei partieller Sicht) — genuin, kein Bug. Die übrigen
9 obj2-Instanzen: rot_sym ≤ 14.5°, dieselbe ~15-91 mm random t-Streuung.

---

## depth_scale-Fix (T-156) wirkt — als Negativ-Kontrolle geprüft
Der vorige RGB-D-Killer (depth_scale=0.1 nie durchgereicht → Tiefe 10× → ~2.4 m
X-Shift, AR=0) ist gefixt und deployt: `batch_eval.py:290-295,899-902` reichen
depth_scale aus `scene_camera.json` durch. **Empirischer Beweis:** FP dt_CAM-Z mean =
**+35 mm auf ~1060 mm Range (3 %)**, NICHT der +10.000 mm (10×) Pre-Fix-Artefakt. Die
Tiefe ist korrekt skaliert; der Rest ist Refiner-Rauschen, kein Units-Leftover.

---

## Verdikt: **kein Artefakt, kein Integrationsbug — RGB-D ist auf diesem Scope ehrlich weniger präzise.**

| Hypothese | Verdikt | Beleg |
|---|---|---|
| (a) Symmetrie falsch gefaltet | **widerlegt** | FP rot_naive 77°→rot_sym 5°, MSPD=1.000; canonical bop_toolkit mssd/mspd über cont-Y-syms |
| (b) konstanter Mesh-Origin-Offset | **widerlegt** | dt_OBJ \|mean\|/std ~0.1 (random), mean near-zero, std ~25 mm richtungslos |
| RGB-D-Penalty = Refiner-t-Rauschen | **bestätigt** | \|t_err\| FP 36 mm / GDRNPP 4 mm; AR_MSSD-Kollaps bei intaktem MSPD; T-163-Signatur |

**Kein Fix.** Es gibt keinen Symmetrie-Deklarations-Fehler und keine Mesh-Konvention
anzugleichen. Die AR-Lücke ist die echte Genauigkeits-Lücke: **GDRNPP ist per-Objekt
trainiert und liefert ~4 mm Translation; die novel-object-Refiner (FP, GigaPose-3D)
liefern ~30 mm random Translation** auf diesen near-prismatischen Anker-Crops — das
sprengt die 11 mm MSSD-Schwelle. RGB-D ist auf diesem 2-Anker-Scope **ehrlich, nicht
unfair, schlechter.** Kein Schönfärben in die andere Richtung nötig.

**Wenn man RGB-D näher an GDRNPP bringen wollte** (nicht beauftragt, nur Richtung): das
ist ein Genauigkeits-, kein Bug-Problem — bessere Crop/Depth-Maske fürs ICP, oder eine
ICP-Verfeinerungs-Iteration mehr in fp-svc/gigapose-svc gegen die GT-Tiefe; der Hebel
liegt in der Refiner-Konfiguration, nicht im Eval/Symmetrie/Mesh-Pfad.

---

## Repro / Artefakte
- Probe: `box:/tmp/t166_probe.py` (read-only; lädt CSVs + `scene_gt.json`, BOP-greedy-
  Match, sym-resolved rot via canonical `pose_error.re`, dt im cam- & obj-Frame).
  Lauf: `/mnt/data/bop/bop-venv/bin/python /tmp/t166_probe.py`.
- Eval-Run: `box:/mnt/data/kip_pose/project/temp/batch_eval/run-20260608T113628Z`
  (`eval/<cfg>/report.txt` = per-obj AR/MSSD/MSPD; `csv/<cfg>.csv` = R,t cam-frame mm).
- models_info: `box:/mnt/data/kip_pose/project/bop/pose_isaac/models_eval/models_info.json`.
- Verwandt: T-083 (Anker-Flip, Single-View), T-156 (depth_scale 10×), T-163 (2D-top/3D-null = Z-Offset).
