# Face-View Classifier

Per-part CNN that maps a segmented part snippet to its **face-view** (which
stable resting face the part is showing the camera). The face fixes the two
out-of-plane rotations; the OBB yaw + back-projected `(x, y)` complete the 6D
pose. This module is the classifier half — training, inference, and the stable
serving contract.

## The contract (for the serving layer)

```python
from faces.classifier.infer import infer

result = infer(crop, part)
# -> {"face": "face_2", "confidence": 0.87}
```

- **`crop`** — `HxWx3` (or `HxW`) uint8/array snippet of one segmented part.
  Ideally the deskewed depth-height-map snippet (same domain as training); a raw
  RGB crop also works for the fallback.
- **`part`** — part name string, e.g. `"Anker_Lang"`. Selects the model and the
  face class set.
- **returns** — `{"face": str, "confidence": float in [0,1]}`. **This signature
  is stable — do not change it.**

`infer()` **never crashes** for a missing model/checkpoint and **needs no torch**
for the fallback. `infer.backend(part)` reports `"checkpoint"` or `"fallback"`.

### Two backends, chosen automatically

1. **checkpoint** — if `faces/classifier/checkpoints/<model>.pt` exists *and*
   torch is importable, the trained CNN runs (softmax confidence).
2. **fallback** — otherwise, nearest-reference matching against a handful of
   training snippets per class (rotation + mirror NCC). Pure numpy/PIL, no torch,
   no checkpoint, no training. Measured fallback val-accuracy on the rendered
   facesets: **89 %** overall (single-class parts 100 %, Zahnrad 81 %,
   Buerstenhalter 67 %). Good enough to ship day-one; the trained model replaces
   it transparently.

The serving endpoint works immediately on the fallback and **silently upgrades**
to the trained model the moment a checkpoint is dropped in — same call, same
return shape.

## Part → model mapping

One model per **part family** (parts sharing geometry → same face class set).
`Anker_Lang` and `Anker_Kurz` are the prompted two-model split; the rest default
to one model per part.

| Part                     | Model      | Checkpoint                                    |
|--------------------------|------------|-----------------------------------------------|
| `Anker_Lang`             | `lang`     | `faces/classifier/checkpoints/lang.pt`        |
| `Anker_Kurz`             | `kurz`     | `faces/classifier/checkpoints/kurz.pt`        |
| `Zahnrad`                | `Zahnrad`  | `faces/classifier/checkpoints/Zahnrad.pt`     |
| `Poltopf_kurz_centered`  | (per-part) | `faces/classifier/checkpoints/Poltopf_kurz_centered.pt` |
| `Getriebegehaeuse_typ4`  | (per-part) | `faces/classifier/checkpoints/Getriebegehaeuse_typ4.pt` |
| `Buerstenhalter_2polig`  | (per-part) | `faces/classifier/checkpoints/Buerstenhalter_2polig.pt` |

The mapping lives in `registry.PART_TO_MODEL`. Point two parts at one model name
to share a network. `registry.checkpoint_path(part)` is the canonical path —
inference and training both use it, so a checkpoint always lands where `infer`
looks.

## How to train (Max runs this — NOT auto-run in the pipeline)

Training needs torch in the venv (`pip install torch torchvision`). The data
prep is already done (`extract_snippets.py` → snippets, `build_manifest.py` →
split). Then, per part:

```bash
# 1) (already done) extract labelled snippets + build the manifest
python faces/extract_snippets.py
python faces/build_manifest.py                    # seed=1234, val_frac=0.2

# 2) train one part -> writes faces/classifier/checkpoints/<model>.pt
python faces/classifier/train.py --part Zahnrad
python faces/classifier/train.py --part Anker_Lang     # writes lang.pt
python faces/classifier/train.py --part Anker_Kurz     # writes kurz.pt (needs faceset_Anker_Kurz)
```

Training honours the ML hard-rules:
- **class-weighted** cross-entropy (face priors are very imbalanced),
- **early stopping** on val loss + **best-checkpoint** save (never last epoch),
- **augmentation** only as enabled in the manifest's `augment` block,
- **fixed seed** for reproducibility.

Single-class parts (Anker_Lang, Poltopf, Getriebe currently) are trivial —
`infer` returns the only class directly; `train` no-ops with a note.

### Augmentation (off by default)

Hooks live in the manifest (`build_manifest.py`), all disabled. Enable without
re-extracting:

```bash
python faces/build_manifest.py --enable-augment rotate brightness
```

- **rotate** — random in-plane rotation. Safe: yaw is a free DOF of every
  resting face, so rotated snippets are valid samples.
- **flip** — horizontal mirror. Safe **only** for mirror-symmetric faces (e.g.
  Anker); leave off for chiral faces to avoid corrupting labels.
- **brightness** — multiplicative jitter (lighting on the table).

## How to plug a checkpoint in

There is nothing to wire — just place the `.pt` at the canonical path:

```
faces/classifier/checkpoints/<model>.pt      # e.g. lang.pt, Zahnrad.pt
```

`infer(crop, part)` detects it and switches from fallback to the trained model
on the next call (per-part model cache; restart the process or clear
`infer._MODEL_CACHE` to force a reload). The checkpoint stores `classes` so the
output-index order can never drift from training.

## Files

| File          | Role                                                              |
|---------------|------------------------------------------------------------------|
| `infer.py`    | `infer(crop, part)` — stable contract, checkpoint + fallback     |
| `model.py`    | small 4-block CNN (lazy torch), `build_model(n_classes)`         |
| `train.py`    | manifest → trained checkpoint (class-weighted, early-stop, seed) |
| `registry.py` | part→model map, checkpoint paths, class sets, template loading   |
```
