# Face-Atlas — stable-resting-pose discovery per part

A rigid part dropped onto a flat table comes to rest in a **finite set of stable
faces**. Each face fixes the two out-of-plane rotations and the rest height; only
the in-plane position `(x, y)` and the yaw remain free. The Face-Atlas module
discovers those faces **automatically from the CAD model** — load a part, see how
many physical faces it has, how likely each is, and how it looks resting on each.

This is the foundation of the planar pose pipeline:

- **what faces exist** → the class set `k` for the per-part face classifier,
- **how likely each is** → the natural prior the classifier is trained against,
- and (via the empirical path) the **6D ground-truth** read back from each settle.

For a near-cylindrical part (the motor `Anker`) that means *one* face (lying on
its side); for an asymmetric part it surfaces several distinct rest configs.

## Two paths, one core

| Path | Engine | Speed | Probabilities |
|------|--------|-------|---------------|
| **analytic** (`face_atlas.py`) | `trimesh.compute_stable_poses` | seconds, CPU, no GPU | quasi-static energy model |
| **empirical** (`discover_faces.py` → `atlas_from_drops.py`) | Isaac Sim physics drops | ~2 min / 1000 drops on the RTX box | **measured drop frequencies** |

Both feed the same de-bloating + collage core (`atlas_core.py`), so the atlas
looks identical regardless of engine — only the probabilities differ.

> **Why both:** the analytic model is a free instant baseline, but it
> **over-weights standing/edge poses that a real drop never reaches**. Measured:
> `Poltopf_kurz` is analytic 57.5 % lying + 7.9 % standing + …, but empirically
> **100 % lying** (1000/1000). The empirical distribution is the real prior the
> classifier sees, so it is the source of truth; analytic is the sanity check.

## De-bloating — same face is never counted twice

Raw stable poses (or raw settles) are collapsed into *physical* faces in three
stages, so numeric/system splits never inflate the face count:

1. **g-merge** — single-linkage cluster by the gravity direction in the body
   frame (yaw-invariant, a point on S²). Chains a cylinder's continuous ring of
   side-rolls and tessellation duplicates of one flat face into one cluster.
2. **symmetry-merge** — clusters with the same yaw-invariant contact signature
   (footprint area + rest height), or equiprobable at the same height, merge —
   the N equal sides of a prism, or ring arcs torn apart by the angular
   threshold, collapse to one face.
3. **probability-floor** — clusters below `--min-prob` are unstable / noise and
   are reported as one "selten" note, never as their own face.

All thresholds are CLI-tunable: `--g-merge-deg` (35°), `--sig-tol` (0.10),
`--min-prob` (0.02).

## Usage

Analytic (anywhere with the `faces` venv: `usd-core trimesh matplotlib scipy
networkx`):

```bash
python faces/face_atlas.py data/SDG/IsaacSim/USD-Files/Anker_Lang.usd \
    --out faces_out/Anker_Lang_atlas.png
```

Empirical (discovery runs in the Isaac venv on the GPU box; the atlas render runs
in the `faces` venv):

```bash
# 1) drop N times, read back settled poses  (Isaac venv, GPU box)
python sim_code/discover_faces.py \
    --part data/SDG/IsaacSim/USD-Files/Anker_Lang.usd \
    --out  faces_out/drops_Anker_Lang.jsonl --num 1000 --per-scene 60

# 2) cluster + collage from the settles  (faces venv)
python faces/atlas_from_drops.py \
    data/SDG/IsaacSim/USD-Files/Anker_Lang.usd \
    faces_out/drops_Anker_Lang.jsonl \
    --out faces_out/Anker_Lang_empirical_atlas.png
```

## Files

| File | Role |
|------|------|
| `usd_mesh.py` | extract one welded, world-baked triangle mesh from a USD |
| `atlas_core.py` | shared 3-stage de-bloat clustering + collage rendering |
| `face_atlas.py` | analytic path (trimesh stable poses) |
| `atlas_from_drops.py` | empirical path (cluster Isaac settles) |
| `../sim_code/discover_faces.py` | Isaac Sim N-drop discovery → `drops_<part>.jsonl` |

## Drop-record format (`drops_<part>.jsonl`)

One JSON object per settle:

```json
{"R": [9 floats, column-convention world = R · body], "t": [x, y, z], "g_body": [gx, gy, gz]}
```

`R` is also the per-settle 6D rotation ground-truth — the same readback feeds 6D
pose GT for the eval harness and the face label for classifier training.
