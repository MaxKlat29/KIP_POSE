#!/usr/bin/env python3
"""Small CNN: a 96×96×3 face snippet -> face-class logits.

torch is imported lazily inside ``build_model`` so this module (and everything
that imports it, e.g. ``infer``) loads fine on a box without torch — the
inference fallback never needs the network. Only ``train.py`` and a real
checkpoint-backed ``infer`` pull torch in.

Architecture: a compact 4-block conv net (~0.4 M params) — plenty for the few
face classes per part and the small per-part datasets, fast on CPU at inference.
"""
from __future__ import annotations

IMG_SIZE = 96


def build_model(n_classes, in_ch=3):
    """Construct the CNN. Raises a clear error if torch is unavailable."""
    try:
        import torch.nn as nn
    except ImportError as e:                       # pragma: no cover - env guard
        raise ImportError(
            "torch is required to build/train the model. The inference fallback "
            "(nearest-template) does NOT need torch — see infer.py."
        ) from e

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class FaceCNN(nn.Module):
        def __init__(self, n_classes, in_ch):
            super().__init__()
            self.features = nn.Sequential(
                block(in_ch, 16),    # 96 -> 48
                block(16, 32),       # 48 -> 24
                block(32, 64),       # 24 -> 12
                block(64, 64),       # 12 -> 6
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.2), nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return FaceCNN(n_classes, in_ch)
