#!/usr/bin/env python3
"""Train a per-part face classifier from a snippet manifest. (Max runs this.)

Reads ``data/snippets/<part>/manifest.json`` (built by build_manifest.py) and
trains the small CNN in ``model.py``, writing a checkpoint to the canonical path
``registry.checkpoint_path(part)`` so ``infer.py`` picks it up automatically.

Honours the ML hard-rules:
  * class-weighted cross-entropy (face priors are very imbalanced, e.g. Zahnrad
    92 % / 6 %), so the rare face is not ignored;
  * early stopping on val loss + best-checkpoint save (never the last epoch);
  * augmentation only as enabled in the manifest's ``augment`` block;
  * fixed seed for reproducibility.

torch is imported lazily — this module imports fine without torch (so the
package stays smoke-importable on the inference box). Running it needs torch.

    python faces/classifier/train.py --part Zahnrad
    python faces/classifier/train.py --part Anker_Lang --epochs 60 --lr 1e-3

NOTE: this is wired and verified to import, but is NOT auto-run in the pipeline —
training is a deliberate manual step. See README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import registry          # type: ignore  # noqa: E402
from model import build_model, IMG_SIZE  # type: ignore  # noqa: E402


def _dataset(part, split, augment, seed):
    """A torch Dataset over the manifest's snippets for one split."""
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    man = json.load(open(os.path.join(registry.part_dir(part), "manifest.json")))
    classes = man["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    items = [it for it in man["items"] if it["split"] == split]
    pdir = registry.part_dir(part)

    class Snips(Dataset):
        def __len__(self):
            return len(items)

        def _aug(self, img, rng):
            # apply only enabled hooks; train split only
            if split != "train":
                return img
            from scipy import ndimage
            if augment.get("rotate", {}).get("enabled"):
                deg = rng.uniform(-augment["rotate"]["max_deg"], augment["rotate"]["max_deg"])
                img = ndimage.rotate(img, deg, axes=(0, 1), reshape=False,
                                     order=1, mode="constant", cval=0.5)
            if augment.get("flip", {}).get("enabled") and rng.random() < 0.5:
                img = img[:, ::-1].copy()
            if augment.get("brightness", {}).get("enabled"):
                d = augment["brightness"]["delta"]
                img = np.clip(img * (1.0 + rng.uniform(-d, d)), 0, 1)
            return img

        def __getitem__(self, k):
            it = items[k]
            img = np.asarray(Image.open(os.path.join(pdir, it["path"])).convert("RGB"))
            img = img.astype(np.float32) / 255.0
            rng = np.random.default_rng(seed * 100003 + k)
            img = self._aug(img, rng)
            x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
            return x, cls_idx[it["face_id"]]

    return Snips(), classes, [it["face_id"] for it in items]


def train(part, epochs, lr, batch, patience, seed):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed); np.random.seed(seed)
    man = json.load(open(os.path.join(registry.part_dir(part), "manifest.json")))
    augment = man.get("augment", {})

    tr_ds, classes, tr_labels = _dataset(part, "train", augment, seed)
    va_ds, _, _ = _dataset(part, "val", augment, seed)
    if len(classes) < 2:
        print(f"[train] {part} has a single face class — a classifier is trivial; "
              f"infer() returns it directly. Skipping training.")
        return
    print(f"[train] {part}: {len(tr_ds)} train / {len(va_ds)} val, {len(classes)} classes {classes}")

    # class-weighted loss from the train-split prior
    counts = np.bincount([classes.index(l) for l in tr_labels], minlength=len(classes))
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)).astype("float32"))
    weights = weights / weights.mean()
    crit = nn.CrossEntropyLoss(weight=weights)
    print(f"[train] class counts {counts.tolist()} -> weights {weights.numpy().round(2).tolist()}")

    net = build_model(len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    tl = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(va_ds, batch_size=batch, num_workers=2)

    best, best_state, since = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        net.train()
        for x, y in tl:
            opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
        net.eval(); vloss = 0.0; correct = 0; n = 0
        with torch.no_grad():
            for x, y in vl:
                out = net(x); vloss += crit(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item(); n += len(y)
        vloss /= max(n, 1); acc = correct / max(n, 1)
        print(f"[train] ep{ep:3d}  val_loss={vloss:.4f}  val_acc={acc:.3f}"
              + ("  *best" if vloss < best else ""))
        if vloss < best:                            # best-checkpoint, never last
            best, since = vloss, 0
            best_state = {"model": {k: v.clone() for k, v in net.state_dict().items()},
                          "classes": classes, "val_loss": vloss, "val_acc": acc,
                          "part": part, "seed": seed, "img_size": IMG_SIZE}
        else:
            since += 1
            if since >= patience:
                print(f"[train] early stop at ep{ep} (no val improvement for {patience})")
                break

    ckpt = registry.checkpoint_path(part)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(best_state, ckpt)
    print(f"[train] best val_loss={best_state['val_loss']:.4f} "
          f"val_acc={best_state['val_acc']:.3f} -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    train(args.part, args.epochs, args.lr, args.batch, args.patience, args.seed)


if __name__ == "__main__":
    main()
