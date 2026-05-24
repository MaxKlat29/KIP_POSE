"""TTA — Test-Time-Augmentation für die GDRNPP-Pose-Inferenz (T-058, S-049).

Inferenzseitig, RGB-only, kein Retrain, kein GPU-Pflicht-Anteil im Wrapper selbst
(die GPU-Kosten sind nur die N zusätzlichen Forward-Passes des bereits geladenen
GDRNPP-Checkpoints). Hinter Flag (`--tta` / `tta=True`), Default AUS.

WAS ES TUT
----------
Es wickelt EINEN `call_gdrnpp(crop, K, bbox, obj_id)`-Aufruf in N augmentierte
Aufrufe:

  1. Augmentiere den Crop mit einer view-Transform g (in-plane 90°-Rotationen,
     optional H-Flip).
  2. Rufe `call_gdrnpp` auf dem augmentierten Crop auf -> R_m2c^aug.
  3. INVERTIERE die Augmentierung an der vorhergesagten Rotation: die Kamera
     sieht denselben physischen Teil, nur um die optische Achse gedreht/gespiegelt.
     R_m2c = g^{-1}_cam · R_m2c^aug  (Rotation um die Kamera-Z-Achse).
  4. Aggregiere die N rück-transformierten R_m2c zu einer Schätzung.

WARUM IN-PLANE-ROTATION DER RICHTIGE TTA-KANAL IST (für UNS)
------------------------------------------------------------
Top-Down-Setup: die dominante view-Nuisance ist die In-Plane-Rotation um die
optische Achse. Genau diese Achse ist beim Zahnrad die C_7-Symmetrie-Achse und
beim Anker (liegend) die mehrdeutige Yaw. 90°-Schritte sind EXAKTE Pixel-Ops
(np.rot90, keine Interpolation -> kein Resampling-Verlust) und decken den
In-Plane-Kreis grob ab. Das härtet eine Regression, die pro View nur EINE
Rotation ausgibt (GDRNPP) gegen ihre View-Empfindlichkeit ab.

AGGREGATION (zwei Modi, weil die Fehlerverteilung bimodal ist)
--------------------------------------------------------------
- "chordal": chordaler L2-Mittelwert auf SO(3) (Frobenius-nächste SO(3)-Matrix
  zum arithmetischen Mittel, via SVD). Korrekt für UNIMODALE Streuung (z.B. der
  gut-gelöste Anker-Median). Quelle: Hartley/Trumpf/Dai/Li, Rotation Averaging
  (IJCV); Better Aggregation in TTA, arXiv:2011.11156.
- "score": wähle die Hypothese mit dem höchsten Score-Callback (z.B. ein
  render-vs-RGB-Score oder GDRNPP-Konfidenz). Korrekt für BIMODALE Streuung
  (Anker 180°-Flip, Zahnrad falsches Becken) — ein Mittelwert zwischen zwei
  Becken ist physikalisch falsch. Default für non-cont-Y-Teile.
- "medoid": geodätischer Medoid (die Hypothese mit minimaler Summe der
  Winkelabstände zu allen anderen) — score-frei, robust gegen einzelne Ausreißer,
  kollabiert NICHT zwei Becken zu ihrer Mitte. Default ohne Score-Callback.

EHRLICHE EINSCHÄTZUNG
---------------------
TTA härtet eine *unimodale* View-Empfindlichkeit ab (typisch +1..+3 AR in der
Literatur für gut-gelöste Teile). Es LÖST NICHT das falsche Becken / den echten
3D-Flip — dafür braucht es einen gelernten Multi-Hypothesen-Score (M2 MegaPose)
oder einen SO(3)-Verteilungs-Kopf (Phase-3). Validierbar erst GPU-frei am echten
Checkpoint (jetzt MOCK + Unit-Tests gegen die Transform-Algebra).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# View-Transformen: (crop-op, cam-frame-Rotations-Korrektur)
# ----------------------------------------------------------------------------
# Konvention: das Bild wird im UHRZEIGERSINN um die optische Achse gedreht
# (np.rot90 mit k>0 dreht das Array GEGEN den Uhrzeigersinn in Array-Koordinaten,
# was bei y-nach-unten-Bildachsen einer Bild-Drehung IM Uhrzeigersinn entspricht).
# Die vorhergesagte R_m2c^aug ist dann um die Kamera-Z-Achse mitgedreht; die
# Korrektur ist die INVERSE Rotation um Kamera-Z, links-multipliziert.

def _Rz(theta: float) -> np.ndarray:
    """Rotation um die Kamera-Z-Achse (optische Achse), rad."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)


def rot90_views(n_rot: int = 4):
    """In-plane-90°-View-Transformen (exakte Pixel-Rotationen).

    n_rot in {1,2,4}: 1 = nur Identität (kein TTA), 2 = {0°,180°}, 4 = alle vier.
    Liefert eine Liste von (name, aug_fn, undo_R) wobei:
      aug_fn(crop) -> augmentierter Crop (np.rot90)
      undo_R       -> 3x3 cam-frame-Korrektur: R_m2c = undo_R @ R_m2c^aug
    """
    if n_rot not in (1, 2, 4):
        raise ValueError("n_rot muss 1, 2 oder 4 sein")
    ks = {1: [0], 2: [0, 2], 4: [0, 1, 2, 3]}[n_rot]
    views = []
    for k in ks:
        # np.rot90(img, k): Drehung um k·90° gegen den Uhrzeigersinn in Array-Koord.
        # Die zugehörige cam-Z-Bilddrehung ist +k·90° (theta), die Korrektur -theta.
        theta = k * (np.pi / 2.0)
        undo = _Rz(-theta)
        views.append((f"rot{k*90}", _make_rot90(k), undo))
    return views


def _make_rot90(k: int):
    def _aug(crop):
        return np.ascontiguousarray(np.rot90(crop, k=k))
    return _aug


# Spiegelung (H-Flip) — optional, OFF by default.
# Eine Spiegelung ist KEINE eigentliche Rotation (det = -1). Für die Pose ist die
# korrekte Behandlung: die vorhergesagte Rotation an der Spiegelebene reflektieren.
# Bei H-Flip (x -> -x im Bild, also Spiegelung an der y-z-Kamera-Ebene) ist die
# Korrektur  R_m2c = S · R_m2c^aug · S  mit S = diag(-1,1,1) — eine konjugierte
# Reflexion, die die Chiralität ZWEIMAL umkehrt und so eine gültige SO(3)-Matrix
# zurückgibt. Begründung: das gespiegelte Bild entspricht einer an S gespiegelten
# Szene; die Pose dieser Szene ist S·R·S^{-1}, und S^{-1}=S.
_S_HFLIP = np.diag([-1.0, 1.0, 1.0]).astype(float)


def hflip_view():
    """H-Flip-View-Transform (Spiegelung). Liefert (name, aug_fn, undo_fn).

    ACHTUNG: undo ist hier eine Funktion R -> S·R·S (Konjugation), KEINE reine
    Links-Multiplikation, weil eine Spiegelung konjugiert wirkt. Default AUS,
    weil eine Spiegelung die Chiralität ändert und für chirale Teile (Anker mit
    Kopf/Schaft) eine *falsche* Hypothese erzeugen kann.
    """
    def _aug(crop):
        return np.ascontiguousarray(crop[:, ::-1])

    def _undo(R):
        return _S_HFLIP @ np.asarray(R, float) @ _S_HFLIP

    return ("hflip", _aug, _undo)


# ----------------------------------------------------------------------------
# Aggregation auf SO(3)
# ----------------------------------------------------------------------------
def chordal_mean(Rs) -> np.ndarray:
    """Chordaler L2-Mittelwert: die SO(3)-Matrix, die ‖R - R_i‖_F^2 minimiert.

    = nächste Spezial-Orthogonale zum arithmetischen Mittel (SVD-Projektion).
    Quelle: Hartley et al., Rotation Averaging (IJCV); Frobenius-Mittel.
    """
    Rs = [np.asarray(R, float) for R in Rs]
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:                 # Reflexion vermeiden -> letzte SV flippen
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def geodesic_angle(A, B) -> float:
    """Geodätischer Winkel zwischen zwei Rotationen (rad), numerisch geklemmt."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    tr = np.trace(A.T @ B)
    cos = (tr - 1.0) / 2.0
    return float(np.arccos(max(-1.0, min(1.0, cos))))


def geodesic_medoid(Rs) -> int:
    """Index der Hypothese mit minimaler Summe geodätischer Abstände (Medoid).

    Kollabiert — anders als der Mittelwert — KEINE zwei Becken zu ihrer Mitte;
    wählt das dichteste Becken. Score-frei.
    """
    Rs = [np.asarray(R, float) for R in Rs]
    n = len(Rs)
    if n == 1:
        return 0
    best_i, best_cost = 0, np.inf
    for i in range(n):
        cost = sum(geodesic_angle(Rs[i], Rs[j]) for j in range(n) if j != i)
        if cost < best_cost:
            best_i, best_cost = i, cost
    return best_i


def aggregate(Rs, mode="medoid", scores=None) -> tuple[np.ndarray, dict]:
    """Aggregiere N rück-transformierte cam-frame-Rotationen zu einer Schätzung.

    mode:
      "chordal" -> chordaler Mittelwert (unimodal; gut-gelöste Teile)
      "medoid"  -> geodätischer Medoid (bimodal-robust, score-frei) [DEFAULT]
      "score"   -> argmax(scores) (braucht `scores`, sonst Fallback medoid)
    Returns: (R_agg, info)
    """
    Rs = [np.asarray(R, float) for R in Rs]
    info = {"n": len(Rs), "mode": mode}
    if len(Rs) == 1:
        info["mode"] = "single"
        return Rs[0], info
    if mode == "chordal":
        return chordal_mean(Rs), info
    if mode == "score":
        if scores is None:
            info["fallback"] = "medoid (no scores)"
            i = geodesic_medoid(Rs)
            info["picked"] = i
            return Rs[i], info
        i = int(np.argmax(np.asarray(scores, float)))
        info["picked"] = i
        return Rs[i], info
    # default: medoid
    i = geodesic_medoid(Rs)
    info["picked"] = i
    return Rs[i], info


# ----------------------------------------------------------------------------
# Der TTA-Wrapper um call_gdrnpp
# ----------------------------------------------------------------------------
def tta_call_gdrnpp(call_fn, crop, K, bbox, obj_id, *, cfg=None,
                    n_rot=4, use_hflip=False, agg="medoid",
                    score_fn=None, warn=None):
    """Wickelt `call_fn(crop, K, bbox, obj_id, cfg=cfg) -> (R_m2c, t_m2c_mm)` in
    eine TTA-Schleife.

    Die Translation wird vom Identitäts-View (rot0, kein Flip) genommen — eine
    In-Plane-Bilddrehung lässt die metrische Translation (Zentroid + Z) unberührt
    in unserer Pipeline (der Crop ist achsenparallel und t wird ohnehin nach dem
    Adapter via Zentroid+Z-Snap bestimmt). Nur die ROTATION wird augmentiert/
    aggregiert — der Hebel, den TTA für UNSERE (Rotations-)Fehler hat.

    score_fn(R_m2c, t_m2c_mm) -> float: optionaler render-vs-RGB- oder
    Konfidenz-Score pro Hypothese; nötig nur für agg="score".

    Returns: (R_m2c_agg, t_m2c_mm, info)
    """
    views = list(rot90_views(n_rot))
    undo_kind = "matmul"                       # rot-views: R <- undo @ R
    if use_hflip:
        views.append(hflip_view())             # (name, aug, undo_fn) — Konjugation

    Rs, names, scores = [], [], []
    t_ref = None
    for vi, view in enumerate(views):
        name, aug_fn, undo = view
        aug_crop = aug_fn(crop)
        R_aug, t_aug = call_fn(aug_crop, K, bbox, obj_id, cfg=cfg)
        R_aug = np.asarray(R_aug, float)
        # Undo: rot-views via Links-Multiplikation, hflip via Konjugation.
        if callable(undo):                     # hflip
            R_corr = undo(R_aug)
        else:                                  # rot90 -> 3x3 Korrektur-Matrix
            R_corr = undo @ R_aug
        Rs.append(R_corr)
        names.append(name)
        if name == "rot0":                     # Translation vom Identitäts-View
            t_ref = np.asarray(t_aug, float)
        if agg == "score" and score_fn is not None:
            scores.append(float(score_fn(R_corr, t_aug)))
    if t_ref is None:                          # falls rot0 nicht in der View-Liste
        t_ref = np.asarray(call_fn(crop, K, bbox, obj_id, cfg=cfg)[1], float)

    R_agg, info = aggregate(Rs, mode=agg, scores=(scores or None))
    info["views"] = names
    if warn is not None:
        picked = info.get("picked")
        pk = f", picked={names[picked]}" if picked is not None else ""
        warn(f"[tta] obj{obj_id}: {len(views)} views, agg={info['mode']}{pk}")
    return R_agg, t_ref, info
