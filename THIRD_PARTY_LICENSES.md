# Third-Party Licenses

This project's own code is MIT-licensed (see [`LICENSE`](LICENSE)). It builds on
the following third-party components, which are **installed/cloned separately**
(not vendored in this repository) and keep their own licenses. Trained weights
(`*.pt`, `*.pth`) are project artefacts and are git-ignored, not redistributed
here.

| Component | Role in POSE | License | Source |
|---|---|---|---|
| GDRNPP | Primary pose estimator (Track B) | **Apache-2.0** | https://github.com/shanice-l/gdrnpp_bop2022 |
| CNOS | CAD-based detection/segmentation (Track A) | **MIT** | https://github.com/nv-nguyen/cnos |
| GigaPose | Coarse novel-object pose (Track A) | **MIT** | https://github.com/nv-nguyen/gigapose |
| MegaPose | Render-and-compare refinement (Track A) | **Apache-2.0** | https://github.com/megapose6d/megapose6d |
| bop_toolkit | BOP format + eval metrics | **MIT** | https://github.com/thodan/bop_toolkit |
| Segment Anything (SAM) | Segmentor used by CNOS | **Apache-2.0** | https://github.com/facebookresearch/segment-anything |
| DINOv2 | Descriptor backbone (CNOS) | **Apache-2.0** | https://github.com/facebookresearch/dinov2 |
| Three.js | 3D viewer (vendored in `project/frontend/vendor/`) | **MIT** | https://github.com/mrdoob/three.js |
| NVIDIA Isaac Sim | Synthetic data generation | NVIDIA license (research/eval) | https://developer.nvidia.com/isaac/sim |
| **Ultralytics YOLOv8** | OBB object detector (`detector.pt`) | **AGPL-3.0** ⚠️ | https://github.com/ultralytics/ultralytics |

## ⚠️ Important: YOLOv8 detector is AGPL-3.0

The object detector backbone (`yolov8s-obb`) is licensed **AGPL-3.0** (strong
copyleft). This is fine for research, evaluation and internal use, but matters
for **closed/commercial redistribution**:

- AGPL-3.0 requires that anyone who interacts with the software over a network
  can obtain the corresponding source.
- If you intend to ship POSE as part of a **closed product**, either
  (a) obtain a commercial license from Ultralytics, or
  (b) replace the detector with a permissively-licensed one (e.g. RT-DETR,
  RTMDet, or GDRNPP's own YOLOX detector under Apache-2.0).
- This caveat affects **only the detection stage**. The BOP pose stack
  (GDRNPP / CNOS / GigaPose / MegaPose / bop_toolkit / SAM) is MIT/Apache
  throughout, and GDRNPP accepts detections from any detector.

## Rejected on license grounds

- **FastSAM** (AGPL-3.0) — CNOS is configured to use SAM (Apache-2.0) instead.
- **FoundationPose** (NVIDIA Source Code License, non-commercial) — rejected
  (license + hard depth dependency on shiny metal). See ADR-018.

## Project license note

The project's own original code is offered under MIT for maximum reuse. If the
downstream industry use prefers Apache-2.0 (explicit patent grant), the project
LICENSE can be swapped to Apache-2.0 without affecting any dependency — all
dependencies above are Apache/MIT-compatible.
