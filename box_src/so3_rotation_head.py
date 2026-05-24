"""SO(3) Rotations-VERTEILUNGS-/Klassifikations-Kopf für GDRNPP (Phase-3 Scaffold).

> **Ticket:** T-058 · **Worktree:** `.worktrees/S-049` · **Status:** SCAFFOLD,
> NICHT TRAINIERT. Ready-to-train als Phase-3 NACH dem laufenden Retrain.
> **NICHT JETZT laufen lassen** (GPU trainiert/rendert). Siehe §"AKTIVIERUNG".

WARUM DIESER KOPF (der EINE stärkste Architektur-Hebel für UNSERE Fehler)
=========================================================================
Unser harter Restfehler ist das **Zahnrad im falschen Rotations-Becken**
(naive ~138°, AR 0.36): GDRNPP regrediert pro View EINE Rotation (allo_rot6d) und
kann sich beim C_7-fast-symmetrischen Zahnrad nicht auf EINEN Zahn committen ->
es landet zwischen Becken. Der Anker-180°-Flip (13–19%) ist dieselbe Krankheit:
eine bimodale Verteilung, in eine unimodale Regression gequetscht.

Die Forschung sagt klar: **klassifiziere die Rotation, regrediere sie nicht.**
- **SC6D** (3DV'22, arXiv:2208.02129): lernt eine SO(3)-Einbettung und ordnet die
  Rotation per **Cosinus-Ähnlichkeit** zu N gesampleten Rotations-Ankern zu —
  **symmetrie-agnostisch, ohne CAD-Symmetrie**, SOTA auf T-LESS (78.0 AR),
  ITODD **30.3 AR** (vs. CosyPose 13.1; unser nächster Analog = texturloses
  Metall). https://arxiv.org/pdf/2208.02129 · Code: https://github.com/dingdingcai/SC6D-pose
- **Implicit-PDF** (ICML'21, arXiv:2106.05965): nicht-parametrische Dichte auf
  SO(3), ausgewertet auf einem **HEALPix-äquivolumetrischen Grid** — modelliert
  Symmetrie/Mehrdeutigkeit als echte Multi-Modal-Verteilung.
  https://arxiv.org/pdf/2106.05965 · HEALPix-SO(3): https://implicit-pdf.github.io/

Dieser Kopf kombiniert beide pragmatisch: eine **gelernte SO(3)-Einbettung**
(SC6D-Stil) + **HEALPix-Anker** (Implicit-PDF-Stil) + Cosinus-Softmax. Inferenz
liefert eine **Top-K-Rotations-Verteilung**, aus der der planare Prior (Z-Snap /
stable-pose) den richtigen Modus wählen kann — genau das, was die unimodale
Regression nicht kann.

WAS DIESER FILE IST / NICHT IST
-------------------------------
IST: ein eigenständiges, UNIT-TESTBARES torch-Modul (Kopf + HEALPix-Anker +
Loss + Decode) — die Bausteine, faithful zur SC6D/IPDF-Mathematik. CPU-lauffähig
für Tests; KEIN GDRNPP-Import nötig zum Testen.
NICHT: eine fertig in GDRNPP eingehängte Trainings-Konfiguration. Die exakte
Einhänge-Anleitung steht in §INTEGRATION; der Phase-3-Config-Stub liegt unter
`box_src/configs_phase3/zahnrad_so3cls.py` (deployt nach configs/gdrn/...).

EINSCHRÄNKUNG (ehrlich)
-----------------------
Das ist ein **Retrain** (neuer Kopf + neuer Loss) -> Phase-3, NICHT jetzt. Es
ist KEIN training-freier Hebel und JETZT (GPU belegt) NICHT validierbar. Erwartung
geerdet an ITODD: SC6D 30.3 AR auf ITODD-Metall — auf UNSEREM per-Objekt-In-
Distribution-Setup mit planarem Prior plausibel **mehr fürs Zahnrad** als die
aktuelle Regression (0.36), aber das ist eine HYPOTHESE bis gemessen.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch fehlt lokal -> nur np-Anker testbar
    torch = None
    nn = object
    _HAS_TORCH = False


# ============================================================================
# 1) HEALPix-äquivolumetrisches SO(3)-Anker-Grid (Implicit-PDF-Stil)
# ============================================================================
# Implicit-PDF/SC6D nutzen ein gleichvolumetrisches SO(3)-Grid: HEALPix auf S^2
# für die Blickrichtung (viewpoints) x gleichmäßige In-Plane-Drehungen (tilts).
# Wir liefern hier einen pragmatischen, abhängigkeitsarmen Aufbau:
#   - Blickrichtungen via Fibonacci-Sphäre (≈ gleichflächig, deterministisch),
#   - In-Plane-Drehungen gleichmäßig in [0, 2π).
# Für die ECHTE HEALPix-Pixelierung beim Phase-3-Training: healpy ODER das in
# Implicit-PDF mitgelieferte `generate_healpix_grid` (siehe §INTEGRATION) —
# die API hier (gibt (N,3,3) Rotationsmatrizen) ist identisch austauschbar.

def fibonacci_sphere(n: int) -> np.ndarray:
    """n ≈ gleichflächig verteilte Einheitsvektoren auf S^2 (deterministisch)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))          # goldener Winkel
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


def _rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kürzeste Rotation, die Einheitsvektor a auf b dreht (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-8:                         # antiparallel -> 180° um ⊥-Achse
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis) + 1e-12
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
        return np.eye(3) + 2.0 * (K @ K)        # 180°
    s = np.linalg.norm(v)
    x, y, z = v
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s + 1e-12))


def healpix_so3_grid(n_view: int = 200, n_inplane: int = 24) -> np.ndarray:
    """SO(3)-Anker-Grid als (N,3,3); N = n_view * n_inplane.

    SC6D-Default ist riesig (4000 x 120 = 480k @ Inferenz, 5k @ Training). Für den
    SCAFFOLD/Test sind kleine Defaults gesetzt; das Phase-3-Training skaliert sie
    via Config hoch (siehe §INTEGRATION). Gibt deterministische Anker zurück.
    """
    base = np.array([0.0, 1.0, 0.0])            # Referenz-Blickachse
    views = fibonacci_sphere(n_view)
    inplanes = np.linspace(0.0, 2.0 * np.pi, n_inplane, endpoint=False)
    anchors = np.empty((n_view * n_inplane, 3, 3), np.float64)
    k = 0
    for v in views:
        R_view = _rot_from_to(base, v)
        for psi in inplanes:
            c, s = np.cos(psi), np.sin(psi)
            R_ip = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
            anchors[k] = R_view @ R_ip
            k += 1
    return anchors


def nearest_anchor_index(R: np.ndarray, anchors: np.ndarray) -> int:
    """Index des Ankers mit minimalem geodätischen Abstand zu R (für GT-Label)."""
    R = np.asarray(R, float)
    # geodät. Distanz monoton in trace(R^T A) -> argmax trace
    tr = np.einsum("ij,nij->n", R, anchors)     # = trace(R^T A_n) wegen Symmetrie
    return int(np.argmax(tr))


# ============================================================================
# 2) Der Kopf: SO(3)-Einbettung + Cosinus-Klassifikation (SC6D-Stil)
# ============================================================================
if _HAS_TORCH:

    class SO3RotationClassificationHead(nn.Module):
        """Bildet (a) ein Bild-Embedding aus den GDRNPP-Geo/PnP-Features und (b)
        ein Rotations-Embedding pro SO(3)-Anker; klassifiziert die Rotation per
        Cosinus-Softmax (SC6D). Liefert Logits über die Anker = eine Verteilung
        über SO(3). Decode = argmax/Top-K -> Rotationsmatrix(en).

        Args:
            in_dim:   Eingangs-Feature-Dim aus dem PnP/Geo-Stamm (GDRNPP: 1024er
                      Backbone -> geflachte PnP-Features; hier konfigurierbar).
            emb_dim:  Dim des gemeinsamen Einbettungsraums (SC6D: 64).
            anchors:  (N,3,3) np-Array der SO(3)-Anker (healpix_so3_grid).
            tau:      Temperatur der Cosinus-Softmax (SC6D: 0.1).
        """

        def __init__(self, in_dim: int, anchors: np.ndarray,
                     emb_dim: int = 64, tau: float = 0.1, hidden: int = 256):
            super().__init__()
            self.emb_dim = emb_dim
            self.tau = float(tau)
            A = torch.as_tensor(np.asarray(anchors, np.float32))
            self.register_buffer("anchors", A)              # (N,3,3)
            self.register_buffer("anchors_flat", A.reshape(A.shape[0], 9))  # (N,9)
            # Bild-Zweig: Geo/PnP-Feature -> Bild-Embedding
            self.img_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, emb_dim))
            # Rotations-Zweig: 9-D Anker (flach) -> Rotations-Embedding
            self.rot_mlp = nn.Sequential(
                nn.Linear(9, hidden), nn.GELU(),
                nn.Linear(hidden, emb_dim))

        def rot_embeddings(self) -> "torch.Tensor":
            """(N, emb_dim), L2-normalisiert — einmal pro forward genügt."""
            e = self.rot_mlp(self.anchors_flat)
            return F.normalize(e, dim=-1)

        def forward(self, feat: "torch.Tensor") -> "torch.Tensor":
            """feat: (B, in_dim) -> Logits (B, N) über die SO(3)-Anker."""
            img = F.normalize(self.img_mlp(feat), dim=-1)   # (B, emb_dim)
            rot = self.rot_embeddings()                     # (N, emb_dim)
            sim = img @ rot.t()                             # Cosinus (beide normiert)
            return sim / self.tau                           # Logits

        def decode(self, logits: "torch.Tensor", k: int = 1) -> "torch.Tensor":
            """Top-1 (k=1) oder Top-K Rotationsmatrizen aus den Logits.
            Returns: (B,3,3) bei k=1, sonst (B,k,3,3)."""
            if k == 1:
                idx = logits.argmax(dim=-1)                 # (B,)
                return self.anchors[idx]
            idx = logits.topk(k, dim=-1).indices            # (B,k)
            return self.anchors[idx]

        def loss(self, logits: "torch.Tensor", R_gt: "torch.Tensor",
                 sym_anchor_sets=None) -> "torch.Tensor":
            """Cross-Entropy gegen den GT-nächsten Anker (SC6D). Für SYMMETRISCHE
            Teile: `sym_anchor_sets[b]` = Liste gleichwertiger Anker-Indizes (alle
            C_N-Repräsentanten von R_gt) -> Multi-Positive-CE (one-to-many), das
            der Symmetrie erlaubt, mehrere Anker als korrekt zu zählen. Genau so
            wird der Kopf symmetrie-agnostisch fürs Zahnrad (vgl. SymCode one-to-
            many; SC6D braucht es nicht mal, weil das Embedding die Mehrdeutigkeit
            selbst lernt). Ohne sym_anchor_sets: Standard-Single-Label-CE.
            """
            if sym_anchor_sets is None:
                gt_idx = self._gt_indices(R_gt)             # (B,)
                return F.cross_entropy(logits, gt_idx)
            # Multi-Positive: -log( sum_pos softmax )
            logp = F.log_softmax(logits, dim=-1)            # (B,N)
            losses = []
            for b, pos in enumerate(sym_anchor_sets):
                pos = torch.as_tensor(pos, device=logits.device, dtype=torch.long)
                losses.append(-torch.logsumexp(logp[b, pos], dim=0))
            return torch.stack(losses).mean()

        @torch.no_grad()
        def _gt_indices(self, R_gt: "torch.Tensor") -> "torch.Tensor":
            """Nächster-Anker-Index pro GT-Rotation (Batch). trace(R^T A) argmax."""
            Rf = R_gt.reshape(R_gt.shape[0], 9).to(self.anchors_flat.dtype)
            tr = Rf @ self.anchors_flat.t()                 # (B,N) = trace(R^T A)
            return tr.argmax(dim=-1)


# ============================================================================
# 3) Self-Check (kein GDRNPP, kein GPU-Job) — `python box_src/so3_rotation_head.py`
# ============================================================================
def _self_check() -> int:
    print("[so3-head] HEALPix-SO(3)-Anker bauen (klein) ...")
    anchors = healpix_so3_grid(n_view=60, n_inplane=12)     # 720 Anker
    assert anchors.shape == (720, 3, 3)
    # Orthogonalität / det=1 der Anker
    for A in anchors[::97]:
        assert np.allclose(A.T @ A, np.eye(3), atol=1e-6), "Anker nicht orthogonal"
        assert abs(np.linalg.det(A) - 1.0) < 1e-6, "Anker det != 1"
    # nearest_anchor_index trifft den Anker selbst
    j = 333
    assert nearest_anchor_index(anchors[j], anchors) == j
    print(f"[so3-head] {len(anchors)} Anker ok (orthogonal, det=1, NN-self).")
    if not _HAS_TORCH:
        print("[so3-head] torch fehlt lokal -> Kopf-Forward übersprungen "
              "(am Box-bop-venv testbar). Anker-Mathematik validiert.")
        return 0
    head = SO3RotationClassificationHead(in_dim=128, anchors=anchors, emb_dim=64)
    feat = torch.randn(4, 128)
    logits = head(feat)
    assert logits.shape == (4, 720), logits.shape
    R = head.decode(logits, k=1)
    assert R.shape == (4, 3, 3)
    Rk = head.decode(logits, k=5)
    assert Rk.shape == (4, 5, 3, 3)
    # Loss läuft + ist endlich
    R_gt = torch.as_tensor(anchors[[1, 2, 3, 4]], dtype=torch.float32)
    l = head.loss(logits, R_gt)
    assert torch.isfinite(l), "Loss nicht endlich"
    print(f"[so3-head] Kopf-Forward ok: logits {tuple(logits.shape)}, "
          f"decode top1 {tuple(R.shape)}, top5 {tuple(Rk.shape)}, loss={l.item():.3f}")
    print("[so3-head] SELF-CHECK PASS — Scaffold bereit (NICHT trainiert).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
