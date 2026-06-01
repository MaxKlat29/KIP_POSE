#!/usr/bin/env python3
"""POSE — M2 Multi-Hypothesis Render-and-Compare Refiner (T-058 / S-048).

DER Render-and-Compare-Refiner aus dem PHASE3-Plan (M2). Training-freie Geometrie-
Priors (M1/M3/M4, T-041) sind ausgereizt — nur der Z-Snap hilft. Die zwei harten
Restfehler sind LOKAL nicht snapbar:
  • Anker: ECHTE 180°-3D-Flips (Stab Kopf-über-Schaft), kein view-Symmetrie-C_2.
  • Zahnrad: FALSCHES Rotations-Becken (~138° daneben), keine Tilt-/Yaw-Drift.

Beide sind GLOBAL-mehrdeutig: eine einzelne lokale Korrektur kommt nicht aus dem
falschen Becken. Ein MULTI-HYPOTHESEN-Render-and-Compare kann sie lösen: erzeuge
mehrere Kandidaten-Posen (Coarse + Achsen-Flips + Yaw-Varianten + Ruhelagen +
Tilts), RENDERE die CAD an jeder, VERGLEICHE mit dem RGB-Crop, nimm die beste.

Zwei Scorer (RGB-only, generalisierbar):
  (A) MegaPose-RGB-Refiner/Scorer  — render-and-compare gegen den RGB-Crop, der
      gelernte Score pro Hypothese (Box: /mnt/data/bop/repos/megapose6d). GPU,
      „finish-time" (verdrahtet, validiert beim Finish — GPU aktuell belegt).
  (B) CPU-Kanten/Silhouetten-Score  — leichtgewichtiger Fallback OHNE GPU. Metall
      hat starke, verlässliche Kanten (ContourPose-Idee): wir scoren jede
      Hypothesen-Silhouette/Kante gegen die Detektor-Maske UND gegen die Bild-
      Kanten des Crops (Chamfer). KEIN Training, kein GPU. Default-Scorer wenn
      MegaPose nicht verfügbar / Overhead zu groß.

WICHTIG (ehrlich, T-041-Lehre): die AR-Metrik ist symmetrie-bewusst. Für den
C_7-Zahnrad sind die 7 Zahn-Yaws metrisch ÄQUIVALENT und die Top-Down-Silhouette
ist C_7-invariant — ein reiner Silhouetten-Score kann den Yaw NICHT auflösen (und
muss es metrisch auch nicht). Der CPU-Scorer zielt deshalb primär auf den
**Anker-Flip** (Silhouette + Kanten UNTERSCHEIDEN Kopf vs. Schaft top-down, IoU
0.58/0.64 ≪ 1, M3-Befund) und auf die Tilt-/Becken-Wahl, NICHT auf den C_7-Yaw.
Der gelernte MegaPose-Score (A) ist der prinzipielle Hebel fürs Becken.

Selbst-enthalten: numpy + stdlib. trimesh optional (Ruhelagen + dichtes Sampling).
Re-nutzt die Silhouette-/IoU-/Dilations-Helfer aus bop_adapter (kein Code-Dup).
"""
from __future__ import annotations

import numpy as np

import bop_adapter as BOP

try:  # trimesh nur für Ruhelagen + dichtes Oberflächen-Sampling (optional).
    import trimesh as _trimesh
except Exception:  # pragma: no cover
    _trimesh = None


# ══════════════════════════════════════════════════════════════════════════════
# 1) HYPOTHESEN-GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
# Pro Detektion aus der GDRNPP-Coarse-Welt-Rotation R0: erzeuge eine Liste
# plausibler Welt-Rotations-Kandidaten, die GENAU die globalen Mehrdeutigkeiten
# abdecken, die ein lokaler Refiner nicht knackt:
#   • Coarse selbst (immer dabei, als Fallback-Gate)
#   • 180°-Flips um die Hauptachsen   → Anker End-über-End-Flip
#   • N Yaw-Varianten um die Sym-Achse (C_N-Teile)  → Zahnrad-Becken
#   • K stabile Ruhelagen (compute_stable_poses)    → richtige Auflagefläche
#   • ein paar Tilt-Varianten         → kleine Becken-Nachbarn
# Generalisierbar: Achsen/N/Ruhelagen kommen aus models_info + CAD, nicht hart.


def _rot_geodesic_deg(Ra, Rb):
    """Geodätischer SO(3)-Winkel zwischen Ra, Rb in Grad (zum Deduplizieren)."""
    Ra = BOP._as_R(Ra); Rb = BOP._as_R(Rb)
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _dedup_rotations(rots, tol_deg=3.0):
    """Dedupliziert eine Rotations-Liste (geodätischer Winkel < tol_deg)."""
    kept = []
    for R in rots:
        R = BOP._as_R(R)
        if all(_rot_geodesic_deg(R, K) >= tol_deg for K in kept):
            kept.append(R)
    return kept


def generate_hypotheses(R0_world, *,
                        sym_axis=(0.0, 1.0, 0.0), n_fold=None,
                        flip_axes=("x", "y", "z"),
                        stable_downs=None,
                        tilt_degs=(15.0,), tilt_axes=("x", "z"),
                        max_hypotheses=24, dedup_tol_deg=3.0):
    """Welt-Rotations-Kandidaten aus der Coarse-Rotation R0_world.

    Generalisierbar (Achsen/N/Ruhelagen aus models_info + CAD):
      • R0 selbst (immer index 0 → Fallback-Gate-Referenz).
      • 180°-Flips um die Body-Hauptachsen (flip_axes): R0 @ Rot(axis, π).
        Der Anker-End-über-End-Flip ist ein 180°-Flip um eine zur Längsachse
        senkrechte Achse → von X/Z abgedeckt.
      • C_N-Yaw-Varianten: R0 @ Rot(sym_axis, k·2π/N) für k=1..N-1 (Zahnrad N=7).
      • K stabile Ruhelagen: R_align(b_down → Welt-DOWN) · (Yaw von R0 erhalten),
        falls stable_downs gegeben → korrekte Auflagefläche.
      • Tilt-Varianten: kleine ±tilt um tilt_axes (Becken-Nachbarn).

    Args:
      R0_world      : (3,3) Coarse-Welt-Rotation (nach bop_adapter, vor Z-Snap).
      sym_axis      : Symmetrieachse im Body-Frame (Y bei den Fokus-Teilen).
      n_fold        : C_N (7 für Zahnrad); None/<2 → keine Yaw-Varianten.
      flip_axes     : Body-Achsen für die 180°-Flips ("x","y","z").
      stable_downs  : optional (K,3) Body-Down-Achsen der Ruhelagen
                      (bop_adapter.stable_pose_body_downs) → Ruhelagen-Kandidaten.
      tilt_degs     : Winkel der Tilt-Varianten.
      tilt_axes     : Welt-Achsen für die Tilt-Varianten.
      max_hypotheses: harte Obergrenze (Performance; R0 bleibt immer enthalten).
      dedup_tol_deg : Kandidaten näher als das werden zusammengelegt.

    Returns:
      (hyps (M,3,3), tags (M,) list[str]) — hyps[0] ist immer R0 (tag "coarse").
    """
    R0 = BOP._as_R(R0_world)
    axis_vec = {"x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0])}
    sax = np.asarray(sym_axis, float)

    cand = [R0]
    tags = ["coarse"]

    # 180°-Flips um die Body-Hauptachsen (Anker End-über-End).
    for a in flip_axes:
        cand.append(R0 @ BOP.axis_angle_matrix(axis_vec[a], np.pi))
        tags.append(f"flip180_{a}")

    # C_N-Yaw-Varianten um die Symmetrieachse (Zahnrad-Becken).
    if n_fold and n_fold >= 2:
        for k in range(1, int(n_fold)):
            cand.append(R0 @ BOP.axis_angle_matrix(sax, k * 2.0 * np.pi / n_fold))
            tags.append(f"yaw_{k}of{int(n_fold)}")

    # K stabile Ruhelagen: lege das Teil auf jede Auflagefläche, behalte R0-Yaw.
    if stable_downs is not None:
        downs = np.asarray(stable_downs, float)
        b_pred = R0.T @ np.array([0.0, 0.0, -1.0])
        b_pred = b_pred / (np.linalg.norm(b_pred) + 1e-12)
        for j, bd in enumerate(downs):
            bd = bd / (np.linalg.norm(bd) + 1e-12)
            R_align = BOP._min_rot_align(bd, b_pred)     # body: bd → b_pred
            cand.append(R0 @ R_align)
            tags.append(f"rest_{j}")

    # Tilt-Varianten (kleine Becken-Nachbarn um Welt-Achsen).
    for a in tilt_axes:
        for d in tilt_degs:
            cand.append(BOP.axis_angle_matrix(axis_vec[a], np.radians(d)) @ R0)
            tags.append(f"tilt_{a}+{int(d)}")
            cand.append(BOP.axis_angle_matrix(axis_vec[a], -np.radians(d)) @ R0)
            tags.append(f"tilt_{a}-{int(d)}")

    # Dedup (R0 zuerst → bleibt erhalten), Tag-erhaltend.
    kept_R, kept_tags = [], []
    for R, tg in zip(cand, tags):
        if all(_rot_geodesic_deg(R, K) >= dedup_tol_deg for K in kept_R):
            kept_R.append(BOP._as_R(R)); kept_tags.append(tg)
        if len(kept_R) >= max_hypotheses:
            break
    return np.stack(kept_R, axis=0), kept_tags


# ══════════════════════════════════════════════════════════════════════════════
# 2) CPU-KANTEN/SILHOUETTEN-SCORER  (kein GPU, Fallback-/Default-Scorer)
# ══════════════════════════════════════════════════════════════════════════════
# Für jede Hypothese: rendere die CAD-Silhouette in den Kamera-Frame (re-nutzt
# bop_adapter._camera_silhouette) und scoren gegen (a) die Detektor-Maske (IoU)
# und (b) die BILD-KANTEN des RGB-Crops (Chamfer Rendered-Edge↔Image-Edge). Metall
# hat starke Kanten → die Kanten-Übereinstimmung diskriminiert Posen, die die
# Silhouette allein nicht trennt (asymmetrische innere Kanten von Kopf vs. Schaft).
#
# EHRLICH: für den C_7-Zahnrad ist die Top-Down-Silhouette rotations-invariant —
# der CPU-Scorer kann den Zahn-Yaw NICHT auflösen (muss er metrisch auch nicht).
# Sein Ziel ist die Becken-/Flip-Wahl (Anker), nicht der C_7-Yaw.


def _sobel_edges(gray):
    """Billige Sobel-Kantenmagnitude (kein scipy). gray: (H,W) float 0..1."""
    g = np.asarray(gray, float)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = kx.T
    gx = _conv3(g, kx)
    gy = _conv3(g, ky)
    return np.hypot(gx, gy)


def _conv3(img, k):
    """3x3-Faltung via verschobene Summen (edge-replicate Padding, kein scipy)."""
    H, W = img.shape
    p = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img)
    for di in range(3):
        for dj in range(3):
            out += k[di, dj] * p[di:di + H, dj:dj + W]
    return out


def image_edges(crop_rgb, bbox=None, full_hw=None, thresh_pct=80.0):
    """Bool-Kantenmaske aus einem RGB-Crop, optional ins Vollbild eingebettet.

    Args:
      crop_rgb  : (h,w,3) uint8/float RGB des Crops (oder Vollbild).
      bbox      : [x0,y0,x1,y1] — wenn gegeben + full_hw, wird die Crop-Kantenmaske
                  in eine Vollbild-große (H,W)-Maske an die bbox-Position gelegt
                  (passt zur Silhouette, die im Vollbild-Frame gerendert wird).
      full_hw   : (H,W) des Vollbilds (nur mit bbox).
      thresh_pct: Perzentil der Kantenmagnitude als Schwelle.
    Returns: (H,W) bool (Vollbild wenn bbox+full_hw, sonst Crop-groß).
    """
    a = np.asarray(crop_rgb, float)
    if a.ndim == 3:
        gray = a[..., :3].mean(axis=2) / 255.0
    else:
        gray = a / 255.0
    mag = _sobel_edges(gray)
    if mag.max() <= 1e-9:
        edges = np.zeros_like(mag, bool)
    else:
        thr = np.percentile(mag, float(thresh_pct))
        edges = mag >= max(thr, 1e-6)
    if bbox is not None and full_hw is not None:
        H, W = full_hw
        out = np.zeros((H, W), bool)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        eh, ew = edges.shape
        out[y0:y0 + min(eh, y1 - y0), x0:x0 + min(ew, x1 - x0)] = \
            edges[:max(0, y1 - y0), :max(0, x1 - x0)]
        return out
    return edges


def _silhouette_edges(mask):
    """Rand-(Kontur-)Pixel einer gefüllten Silhouetten-Maske (XOR mit Erosion)."""
    m = np.asarray(mask, bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    return m & ~er


def _dilate_bool(mask, iters=1):
    """4-Nachbar-Dilation einer Bool-Maske um `iters` Pixel (kein scipy).

    Für die Visibility-Region: die sichtbare mask_visib wird um ein paar Pixel
    aufgeweitet, bevor sie die gerenderte Kontur / Bildkanten clippt — sonst
    fallen Konturpixel direkt AM sichtbaren Rand (Sub-Pixel-Offset Render vs.
    GT-Maske) fälschlich raus.
    """
    m = np.asarray(mask, bool)
    for _ in range(int(iters)):
        d = m.copy()
        d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        m = d
    return m


def _chamfer_score(rendered_edges, image_edges_mask, max_dist=20):
    """Mittlere Kanten-Übereinstimmung Rendered→Image (höher = besser, 0..1).

    Für jeden gerenderten Kantenpixel: 1 - (clamp(Nächst-Distanz zur Bildkante,
    0..max_dist) / max_dist). Distanz via beschränktem Dilations-Wachstum der
    Bildkanten (kein scipy/EDT nötig). Robuster Silhouetten-Kanten-Match für
    texturloses Metall (starke Außen-/Innenkanten).
    """
    re = np.asarray(rendered_edges, bool)
    ie = np.asarray(image_edges_mask, bool)
    if re.sum() == 0:
        return 0.0
    if ie.sum() == 0:
        return 0.0
    # Distanz-Transform via iteratives Dilatieren: dist[p] = Iteration, in der p
    # zum ersten Mal von einer Bildkante erreicht wird (gekappt bei max_dist).
    reached = ie.copy()
    dist = np.full(ie.shape, max_dist, dtype=np.int32)
    dist[ie] = 0
    frontier = ie.copy()
    for d in range(1, int(max_dist) + 1):
        gd = frontier.copy()
        gd[1:, :] |= frontier[:-1, :]; gd[:-1, :] |= frontier[1:, :]
        gd[:, 1:] |= frontier[:, :-1]; gd[:, :-1] |= frontier[:, 1:]
        newly = gd & ~reached
        if not newly.any():
            break
        dist[newly] = d
        reached |= newly
        frontier = newly
    d_at_re = dist[re].astype(float)
    return float(np.mean(1.0 - np.clip(d_at_re, 0, max_dist) / max_dist))


def render_silhouette(verts_mm, R_world, t_world_m, table_origin_m,
                      R_w2c, t_w2c_mm, K, hw):
    """Welt-Pose → CAD-Silhouette (bool (H,W)) im Kamera-Bild. CPU, kein GPU.

    Re-nutzt die exakte Adapter-Inverse (world→cam) + bop_adapter._camera_silhouette.
    """
    R_world = BOP._as_R(R_world)
    R_w2c = BOP._as_R(R_w2c)
    t_world = BOP._as_vec3(t_world_m)
    table_origin = BOP._as_vec3(table_origin_m)
    t_w2c = BOP._as_vec3(t_w2c_mm)
    t_m2w_mm = (t_world + table_origin) * 1000.0
    t_m2c = R_w2c @ t_m2w_mm + t_w2c
    R_m2c = R_w2c @ R_world
    return BOP._camera_silhouette(np.asarray(verts_mm, float), R_m2c, t_m2c,
                                  BOP._as_R(K), hw)


def cpu_edge_score(hyps, *, verts_mm, t_world_m, table_origin_m, R_w2c, t_w2c_mm,
                   K, hw, target_mask=None, image_edge_mask=None,
                   visib_mask=None, visib_dilate=2,
                   w_iou=0.5, w_chamfer=0.5):
    """CPU-Render-and-Compare-Score pro Hypothese (höher = besser).

    Für jede Hypothesen-Welt-Rotation: rendere die Silhouette, score gegen
      • die Detektor-Maske (IoU der gefüllten Silhouette), w_iou
      • die Bildkanten des Crops (Chamfer Silhouetten-Kontur ↔ Bildkanten), w_chamfer
    Score = w_iou·IoU + w_chamfer·Chamfer. Mindestens eines von (target_mask,
    image_edge_mask) muss gegeben sein.

    VISIBILITY-AWARE (ADR-020, S-003): wenn `visib_mask` (H,W bool, die im Bild
    tatsächlich sichtbare Teil-Region, z.B. BOP `mask_visib`) gegeben ist, wird
    der Score NUR über die sichtbare Region gerechnet:
      • IoU: gerenderte Silhouette UND target_mask auf visib_mask clippen
        (sil_v = sil & visib; tgt_v = target & visib). Die Phantom-Silhouette in
        der occludierten/abgeschnittenen Hälfte (Schaft) belegt dort keine
        sichtbare Evidenz → straft die falsche Flip-Hypothese nicht mehr fälschlich
        gleich gut ab. Das diskriminierende Kopf-Asymmetrie-Signal dominiert.
      • Chamfer: gerenderte Kontur NUR über Konturpixel innerhalb visib_mask
        (∪ kleine Dilation), gegen image_edge_mask & visib_dilated. Occluder-/
        Bildrand-Fremdkanten fließen nicht mehr in den Match.
    `visib_mask=None` → EXAKT das heutige Verhalten (rückwärtskompatibel).

    Args:
      visib_mask  : (H,W) bool sichtbare Region, oder None (= heutiges Verhalten).
      visib_dilate: Pixel-Dilation der Visibility-Region fürs Chamfer-Clipping
                    (Sub-Pixel-Offset Render↔GT-Maske abfedern). 0 = exaktes Clip.

    Returns: (scores (M,), detail list[dict] mit iou/chamfer pro Hypothese).
    """
    hyps = np.asarray(hyps, float)
    if hyps.ndim == 2:
        hyps = hyps[None]
    if target_mask is None and image_edge_mask is None:
        raise ValueError("cpu_edge_score braucht target_mask ODER image_edge_mask")
    tgt = None if target_mask is None else np.asarray(target_mask, bool)
    ie = None if image_edge_mask is None else np.asarray(image_edge_mask, bool)
    vis = None if visib_mask is None else np.asarray(visib_mask, bool)
    vis_dil = (None if vis is None
               else (_dilate_bool(vis, visib_dilate) if visib_dilate > 0 else vis))
    # Visibility-restringierte Score-Inputs einmalig vorbereiten (hyp-invariant).
    tgt_v = tgt if (tgt is None or vis is None) else (tgt & vis)
    ie_v = ie if (ie is None or vis_dil is None) else (ie & vis_dil)
    scores, detail = [], []
    for R in hyps:
        sil = render_silhouette(verts_mm, R, t_world_m, table_origin_m,
                                R_w2c, t_w2c_mm, K, hw)
        # IoU: Silhouette gegen target — beide auf die sichtbare Region clippen.
        if tgt is not None:
            sil_iou = sil if vis is None else (sil & vis)
            iou = BOP._mask_iou(sil_iou, tgt_v)
        else:
            iou = 0.0
        # Chamfer: Silhouetten-Kontur auf die (dilatierte) sichtbare Region clippen,
        # gegen die ebenso geclippten Bildkanten.
        if ie is not None:
            sil_edges = _silhouette_edges(sil)
            if vis_dil is not None:
                sil_edges = sil_edges & vis_dil
            cham = _chamfer_score(sil_edges, ie_v)
        else:
            cham = 0.0
        # Gewichte nur über vorhandene Terme normalisieren.
        wi = w_iou if tgt is not None else 0.0
        wc = w_chamfer if ie is not None else 0.0
        wsum = wi + wc if (wi + wc) > 0 else 1.0
        s = (wi * iou + wc * cham) / wsum
        scores.append(s)
        detail.append({"iou": float(iou), "chamfer": float(cham), "score": float(s)})
    return np.asarray(scores), detail


def select_best_hypothesis(hyps, scores, coarse_idx=0, min_margin=0.0,
                           visib_fract=None, min_visib_fract=0.0,
                           visible_px=None, min_visible_px=None):
    """Wähle die best-scorende Hypothese; gate gegen die Coarse (Index 0).

    GATE (MegaPose-Design): nur wechseln, wenn die beste Hypothese die Coarse um
    > min_margin schlägt — sonst Coarse behalten (ein schwacher Refiner soll eine
    gute Coarse nicht verschlechtern).

    VISIBILITY-KONDITIONIERTES GATE (ADR-020, S-003): zusätzlich zum Margin-Gate
    nur wechseln, wenn die sichtbare Region GROSS/aussagekräftig genug ist, dass
    der visibility-aware Score überhaupt Evidenz hat:
      • visib_fract >= min_visib_fract  (sichtbarer Flächenanteil der Instanz), UND
      • visible_px   >= min_visible_px  (absolute sichtbare Konturpixel), falls
        beide gesetzt.
    Ist eine der Bedingungen gesetzt aber nicht erfüllt → Coarse behalten (zu
    wenig Evidenz, dem Single-View-Restlimit folgend). Beide None/0 → reines
    Margin-Gate = exakt heutiges Verhalten. min_margin ist die harte Untergrenze
    (ADR-020: ≥ 0.15) und wird NIE global gelockert, nur visibility-konditioniert.

    Returns: (best_idx, info dict).
    """
    scores = np.asarray(scores, float)
    best = int(np.argmax(scores))
    coarse_score = float(scores[coarse_idx])
    margin = float(scores[best]) - coarse_score
    base = {"best_idx": best, "best_score": float(scores[best]),
            "coarse_score": coarse_score, "margin": float(margin)}

    # Visibility-Kondition: genug sichtbare Evidenz für einen Switch?
    visib_ok = True
    reasons = []
    if visib_fract is not None and min_visib_fract:
        if float(visib_fract) < float(min_visib_fract):
            visib_ok = False
            reasons.append(f"visib_fract {float(visib_fract):.3f} < {float(min_visib_fract):.3f}")
    if visible_px is not None and min_visible_px:
        if int(visible_px) < int(min_visible_px):
            visib_ok = False
            reasons.append(f"visible_px {int(visible_px)} < {int(min_visible_px)}")
    base["visib_gated_out"] = (not visib_ok)
    if reasons:
        base["visib_reason"] = "; ".join(reasons)

    if (not visib_ok) or (margin <= float(min_margin)):
        base["switched"] = False
        return coarse_idx, base
    base["switched"] = (best != coarse_idx)
    return best, base


# ══════════════════════════════════════════════════════════════════════════════
# 3) MEGAPOSE-RGB-SCORER  (GPU, „finish-time" — verdrahtet, validiert beim Finish)
# ══════════════════════════════════════════════════════════════════════════════
# MegaPose ist auf der Box installiert (/mnt/data/bop/repos/megapose6d, RGB-Modus).
# Der Refiner nimmt eine Pose-Initialisierung (TCO_input) pro Hypothese, rendert
# die CAD an der Hypothese, vergleicht render-vs-RGB und liefert eine verfeinerte
# Pose + einen Score. Wir füttern ALLE Hypothesen als TCO_init, refinen je n_iter
# Iterationen und nehmen die best-scorende — exakt der Multi-Hypothesen-Pfad.
#
# Diese Funktion war als „finish-time" GPU-Pfad geplant. Beim Phase-2-Finish
# wurde MegaPose-M2 (cpu_edge / megapose-scorer) durchgemessen und das Ergebnis
# blieb ≤ planar-Z-Snap — siehe project/docs/RESULTS_PHASE2.md und M2_REFINER.md.
# Der Call-Contract bleibt hier dokumentiert, der Pfad ist NotImplemented.


class MegaPoseUnavailable(RuntimeError):
    """MegaPose nicht importierbar/kein GPU — Caller fällt auf den CPU-Scorer."""


def megapose_score(hyps, *, t_world_m, table_origin_m, R_w2c, t_w2c_mm,
                   crop_rgb=None, K_full=None, bbox=None, obj_label=None,
                   mesh_dir=None, model_name="megapose-1.0-RGB",
                   n_refiner_iterations=5, device="cuda"):
    """MegaPose-RGB-Refiner/Scorer über alle Hypothesen (GPU, finish-time).

    Contract (so füttert man MegaPose den Multi-Hypothesen-Pfad):
      1. Lade das benannte MegaPose-RGB-Modell + den Object-Dataset (CAD aus
         mesh_dir, BOP-Layout obj_<id>.ply).
      2. Baue je Hypothese ein TCO_init (Welt-Rotation → R_m2c via Adapter-Inverse,
         t aus der Coarse-t) und eine Detections-Zeile (bbox, label).
      3. pose_estimator.forward_refiner(observation, data_TCO_input,
         n_iterations=n_refiner_iterations) → refinte Posen + per-Hyp-Score.
      4. Wähle die best-scorende refinte Pose, transformiere zurück in den Welt-
         Frame (bop_adapter.bop_pose_to_world) → verfeinerte Welt-Rotation.

    GPU-Pfad NotImplemented in diesem Repo (M2 wurde Phase-2 durchgemessen und
    ist gegenüber planar-Z-Snap nicht überlegen — Anker-C2-Flip ist aus einer
    Top-Down-RGB nicht stabil resolvierbar; siehe RESULTS_PHASE2.md).

    Returns: (best_idx, R_refined_world (3,3), per_hyp_score (M,)).
    """
    try:  # pragma: no cover - GPU-only, finish-time
        import torch
        from megapose.inference.pose_estimator import PoseEstimator  # noqa: F401
        from megapose.inference.types import (ObservationTensor,  # noqa: F401
                                              PoseEstimatesType)
        from megapose.inference.utils import load_named_model  # noqa: F401
        from megapose.datasets.object_dataset import RigidObjectDataset  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise MegaPoseUnavailable(
            f"MegaPose/Torch nicht verfügbar ({exc!r}); CPU-Scorer nutzen.")
    # pragma: no cover — M2 GPU-Pfad wurde Phase-2 abschließend durchgemessen
    # (Anker-Ceiling 0.61, M2 nicht > planar-Z-Snap). Pfad bleibt NotImplemented;
    # cpu_edge-Scorer ist der operative Default — siehe refine_rc.refine_pose().
    raise NotImplementedError(  # pragma: no cover
        "megapose_score: GPU-M2-Pfad wurde Phase-2 abschließend gemessen "
        "(siehe RESULTS_PHASE2.md). Operativ läuft cpu_edge.")


# ══════════════════════════════════════════════════════════════════════════════
# 4) ORCHESTRIERUNG  — eine Detektion → verfeinerte Welt-Rotation
# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT-GATE (T-058-MESSUNG, val): Der CPU-Kanten-Scorer wechselt mit offenem
# Gate (margin 0) ~460/519 Anker-Posen und VERSCHLECHTERT die AR (Anker 0.645/
# 0.650 → 0.533/0.577) — er führt Flips EIN, weil die Top-Down-Silhouette/Kante
# den 180°-Anker-Flip auf realem RGB nicht trennt (M3: Self-IoU 0.58/0.64). Mit
# striktem Gate (margin 0.15) wechseln nur 2/519 → AR neutral (0.330 ≈ raw). Der
# Default ist daher KONSERVATIV (0.15): kein Schaden, aber auch kein Gewinn — der
# echte Hebel ist der GELERNTE MegaPose-Score (scorer="megapose", finish-time).
DEFAULT_CPU_MIN_MARGIN = 0.15


# Visibility-aware Default-Gate (ADR-020, S-003): wenn eine visib_mask gegeben ist,
# darf nur unter einer Mindest-Sichtbarkeit geschaltet werden (sonst zu wenig
# Evidenz → Coarse behalten, Single-View-Restlimit). min_margin bleibt die harte
# Untergrenze ≥ DEFAULT_CPU_MIN_MARGIN und wird NIE global gelockert.
DEFAULT_MIN_VISIB_FRACT = 0.25


def refine_detection(R0_world, *, verts_mm, t_world_m, table_origin_m,
                     R_w2c, t_w2c_mm, K, hw,
                     target_mask=None, image_edge_mask=None,
                     visib_mask=None, visib_fract=None,
                     min_visib_fract=DEFAULT_MIN_VISIB_FRACT, min_visible_px=None,
                     visib_dilate=2,
                     sym_axis=(0.0, 1.0, 0.0), n_fold=None, stable_downs=None,
                     scorer="cpu_edge", min_margin=None,
                     megapose_kwargs=None, **gen_kwargs):
    """Eine Detektion render-and-compare-refinen (Multi-Hypothese).

    Ablauf: generate_hypotheses(R0) → scorer (cpu_edge | megapose) →
    select_best_hypothesis (gegen Coarse gegated) → verfeinerte Welt-Rotation.

    scorer:
      "cpu_edge"  : CPU-Kanten/Silhouetten-Scorer (kein GPU). DEFAULT-Gate 0.15
                    (konservativ — T-058-Messung: offenes Gate verschlechtert die
                    Anker-AR; siehe DEFAULT_CPU_MIN_MARGIN). Training-frei KEIN
                    Gewinn auf realem RGB ÜBER DEN VOLLEN CROP.
      "megapose"  : MegaPose-RGB (GPU, finish-time; bei Unavailable → cpu_edge).
                    DER eigentliche Hebel (gelernter render-vs-RGB-Score).

    VISIBILITY-AWARE (ADR-020, S-003): `visib_mask` (sichtbare Region, offline aus
    BOP `mask_visib`) restringiert den cpu_edge-Score auf die sichtbare Region und
    konditioniert das Gate (`visib_fract`/`min_visib_fract`, `min_visible_px`). So
    dominiert die Kopf-Asymmetrie statt im occludierten/abgeschnittenen Rauschen zu
    ertrinken. `visib_mask=None` → exakt heutiges Verhalten (rückwärtskompatibel).

    min_margin=None → DEFAULT_CPU_MIN_MARGIN (0.15). Explizit setzbar für Tests.
    Returns: (R_refined_world (3,3), info dict).
    """
    if min_margin is None:
        min_margin = DEFAULT_CPU_MIN_MARGIN
    hyps, tags = generate_hypotheses(
        R0_world, sym_axis=sym_axis, n_fold=n_fold, stable_downs=stable_downs,
        **gen_kwargs)
    info = {"n_hyps": len(hyps), "tags": tags, "scorer": scorer,
            "visib_aware": visib_mask is not None}

    if scorer == "megapose":
        try:
            mk = megapose_kwargs or {}
            best_idx, R_ref, scores = megapose_score(
                hyps, t_world_m=t_world_m, table_origin_m=table_origin_m,
                R_w2c=R_w2c, t_w2c_mm=t_w2c_mm, **mk)
            info.update(best_idx=best_idx, best_tag=tags[best_idx],
                        scores=[float(s) for s in scores], megapose=True)
            return BOP._as_R(R_ref), info
        except (MegaPoseUnavailable, NotImplementedError) as exc:
            info["megapose_fallback"] = repr(exc)
            scorer = "cpu_edge"
            info["scorer"] = "cpu_edge"

    scores, detail = cpu_edge_score(
        hyps, verts_mm=verts_mm, t_world_m=t_world_m,
        table_origin_m=table_origin_m, R_w2c=R_w2c, t_w2c_mm=t_w2c_mm, K=K, hw=hw,
        target_mask=target_mask, image_edge_mask=image_edge_mask,
        visib_mask=visib_mask, visib_dilate=visib_dilate)
    # sichtbare Konturpixel als absolutes Evidenz-Maß fürs Gate (falls gefordert).
    visible_px = None
    if visib_mask is not None and min_visible_px is not None:
        visible_px = int(np.asarray(visib_mask, bool).sum())
    best_idx, sel = select_best_hypothesis(
        hyps, scores, coarse_idx=0, min_margin=min_margin,
        visib_fract=visib_fract, min_visib_fract=min_visib_fract,
        visible_px=visible_px, min_visible_px=min_visible_px)
    info.update(sel)                                       # switched/best_score/...
    info.update(best_idx=best_idx, best_tag=tags[best_idx],
                scores=[float(s) for s in scores], detail=detail)
    return BOP._as_R(hyps[best_idx]), info
