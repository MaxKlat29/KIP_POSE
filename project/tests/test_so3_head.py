#!/usr/bin/env python3
"""Unit-Tests für den SO(3)-Rotations-Klassifikations-Kopf-Scaffold (T-058/S-049).

Testet die training-FREIEN, GDRNPP-unabhängigen Bausteine (Phase-3-Scaffold):
  1. HEALPix-SO(3)-Anker — orthogonal, det=1, NN-self, gleichflächig-plausibel.
  2. nearest_anchor_index — GT-Label-Zuordnung (geodätisch nächster Anker).
  3. Kopf-Forward/Decode/Loss — Form-Verträge, Top-K, Multi-Positive-CE für
     symmetrische Teile (one-to-many) ist endlich und ≤ Single-Label-CE.

torch-Teile werden übersprungen, wenn torch lokal fehlt (am Box-bop-venv laufen
sie). Die reine Anker-Mathematik läuft immer.

Lauf:  python3 -m pytest project/tests/test_so3_head.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SO3 = ROOT / "box_src" / "so3_rotation_head.py"
spec = importlib.util.spec_from_file_location("so3_rotation_head", SO3)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

torch = pytest.importorskip("torch") if H._HAS_TORCH else None


def is_SO3(R, tol=1e-6):
    R = np.asarray(R, float)
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < 1e-5)


# ── 1. Anker-Grid ─────────────────────────────────────────────────────────────
def test_fibonacci_sphere_unit_and_count():
    v = H.fibonacci_sphere(500)
    assert v.shape == (500, 3)
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_healpix_grid_is_valid_so3():
    A = H.healpix_so3_grid(n_view=50, n_inplane=8)
    assert A.shape == (400, 3, 3)
    for R in A[::37]:
        assert is_SO3(R), "Anker muss eine gültige Rotation sein"


def test_rot_from_to_maps_vectors():
    a = np.array([0.0, 1.0, 0.0])
    for seed in range(5):
        rng = np.random.default_rng(seed)
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        R = H._rot_from_to(a, b)
        assert is_SO3(R)
        assert np.allclose(R @ a, b, atol=1e-9)


def test_rot_from_to_antiparallel():
    a = np.array([0.0, 1.0, 0.0])
    R = H._rot_from_to(a, -a)
    assert is_SO3(R)
    assert np.allclose(R @ a, -a, atol=1e-9)


def test_nearest_anchor_index_self():
    A = H.healpix_so3_grid(n_view=40, n_inplane=10)
    for j in (0, 113, 399):
        assert H.nearest_anchor_index(A[j], A) == j


def test_nearest_anchor_index_perturbed():
    """Eine leicht gestörte Anker-Rotation mappt zurück auf denselben Anker."""
    A = H.healpix_so3_grid(n_view=80, n_inplane=16)
    j = 500
    # winzige Drehung um X (kleiner als der Anker-Abstand)
    d = 0.01
    c, s = np.cos(d), np.sin(d)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    assert H.nearest_anchor_index(Rx @ A[j], A) == j


# ── 2. Kopf-Forward / Decode / Loss (torch) ───────────────────────────────────
@pytest.mark.skipif(not H._HAS_TORCH, reason="torch nicht verfügbar")
def test_head_forward_shapes():
    A = H.healpix_so3_grid(n_view=60, n_inplane=12)     # 720
    head = H.SO3RotationClassificationHead(in_dim=128, anchors=A, emb_dim=64)
    feat = torch.randn(5, 128)
    logits = head(feat)
    assert logits.shape == (5, 720)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not H._HAS_TORCH, reason="torch nicht verfügbar")
def test_head_decode_topk():
    A = H.healpix_so3_grid(n_view=50, n_inplane=10)     # 500
    head = H.SO3RotationClassificationHead(in_dim=64, anchors=A)
    logits = head(torch.randn(3, 64))
    R1 = head.decode(logits, k=1)
    assert R1.shape == (3, 3, 3)
    Rk = head.decode(logits, k=4)
    assert Rk.shape == (3, 4, 3, 3)
    for b in range(3):
        assert is_SO3(R1[b].numpy())


@pytest.mark.skipif(not H._HAS_TORCH, reason="torch nicht verfügbar")
def test_head_single_label_loss_finite_and_minimizable():
    A = H.healpix_so3_grid(n_view=40, n_inplane=10)     # 400
    head = H.SO3RotationClassificationHead(in_dim=32, anchors=A, tau=0.1)
    feat = torch.randn(8, 32, requires_grad=True)
    R_gt = torch.as_tensor(A[[1, 50, 100, 150, 200, 250, 300, 350]],
                           dtype=torch.float32)
    logits = head(feat)
    loss0 = head.loss(logits, R_gt)
    assert torch.isfinite(loss0)
    # ein Gradientenschritt senkt den Loss (Lernbarkeit)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        l = head.loss(head(feat.detach()), R_gt)
        l.backward()
        opt.step()
    assert l.item() < loss0.item(), "Loss sollte über Schritte fallen"


@pytest.mark.skipif(not H._HAS_TORCH, reason="torch nicht verfügbar")
def test_multipositive_loss_for_symmetry():
    """Multi-Positive-CE (one-to-many) für C_N-Symmetrie ist endlich und KLEINER-
    gleich als die Single-Label-CE (mehr erlaubte Positive -> nie höher)."""
    A = H.healpix_so3_grid(n_view=60, n_inplane=12)     # 720
    head = H.SO3RotationClassificationHead(in_dim=48, anchors=A)
    feat = torch.randn(4, 48)
    logits = head(feat)
    R_gt = torch.as_tensor(A[[0, 10, 20, 30]], dtype=torch.float32)
    single = head.loss(logits, R_gt)
    # für jedes Sample mehrere "gleichwertige" Anker (simulierte C_N-Repräsentanten)
    sym_sets = [[0, 5, 11], [10, 15, 21], [20, 25, 31], [30, 35, 41]]
    multi = head.loss(logits, R_gt, sym_anchor_sets=sym_sets)
    assert torch.isfinite(multi)
    assert multi.item() <= single.item() + 1e-5, (
        "Multi-Positive-CE darf nicht über Single-Label-CE liegen")


def test_self_check_runs():
    """Der eingebaute Self-Check läuft fehlerfrei durch (Exit 0)."""
    assert H._self_check() == 0
