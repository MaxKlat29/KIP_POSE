# References

> The methods, repositories and benchmarks this project builds on. Paper titles,
> authors, venues and URLs were verified against arXiv / the source repos on
> 2026-05-23. Licenses for the code dependencies are tracked separately in
> [`../../THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).

---

## Pose estimation methods

**GDRNPP** — primary pose estimator (Gleis B), per-object, RGB-only.
> X. Liu, R. Zhang, C. Zhang, G. Wang, J. Tang, Z. Li, X. Ji.
> *GDRNPP: A Geometry-guided and Fully Learning-based Object Pose Estimator.*
> IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025.
> Extends GDR-Net (CVPR 2021). arXiv: https://arxiv.org/abs/2102.12145 ·
> Code: https://github.com/shanice-l/gdrnpp_bop2022 (Apache-2.0)

**CNOS** — CAD-based zero-shot detection/segmentation (Gleis A, stage 1).
> V. N. Nguyen, T. Groueix, G. Ponimatkin, V. Lepetit, T. Hodaň.
> *CNOS: A Strong Baseline for CAD-based Novel Object Segmentation.*
> ICCV 2023 Workshop (R6D). arXiv: https://arxiv.org/abs/2307.11067 ·
> Code: https://github.com/nv-nguyen/cnos (MIT)

**GigaPose** — fast novel-object coarse pose via one correspondence (Gleis A, stage 2).
> V. N. Nguyen, T. Groueix, M. Salzmann, V. Lepetit.
> *GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence.*
> CVPR 2024. arXiv: https://arxiv.org/abs/2311.14155 ·
> Code: https://github.com/nv-nguyen/gigapose (MIT)
> · KIP-Fork: https://github.com/yannicd03/gigapose

**FoundationPose** — unified 6D pose estimation/tracking of novel objects (RGB-D-Pfad, `fp-svc`).
> B. Wen, W. Yang, J. Kautz, S. Birchfield.
> *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects.*
> CVPR 2024 (Highlight). arXiv: https://arxiv.org/abs/2312.08344 ·
> Code: https://github.com/NVlabs/FoundationPose (NVIDIA Source Code License)
> · KIP-Fork: https://github.com/yannicd03/FoundationPose

**MegaPose** — render-and-compare pose refinement for novel objects (Gleis A, stage 3).
> Y. Labbé, L. Manuelli, A. Mousavian, S. Tyree, S. Birchfield, J. Tremblay,
> J. Carpentier, M. Aubry, D. Fox, J. Sivic.
> *MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare.*
> CoRL 2022. arXiv: https://arxiv.org/abs/2212.06870 ·
> Code: https://github.com/megapose6d/megapose6d (Apache-2.0)

---

## Foundation models

**Segment Anything (SAM)** — the segmentor CNOS uses (the Apache `sam` config,
**not** the AGPL FastSAM config).
> A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao,
> S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick.
> *Segment Anything.* ICCV 2023. arXiv: https://arxiv.org/abs/2304.02643 ·
> Code: https://github.com/facebookresearch/segment-anything (Apache-2.0)

> CNOS also uses **DINOv2** (Oquab et al., 2023) as its descriptor backbone,
> loaded on demand via `torch.hub` from
> https://github.com/facebookresearch/dinov2 (Apache-2.0).

---

## Benchmark, format and tooling

**BOP Challenge 2024** — the benchmark, datasets and evaluation methodology this
project conforms to.
> V. N. Nguyen, S. Tyree, A. Guo, M. Fourmy, A. Gouda, T. Lee, S. Moon, H. Son,
> L. Ranftl, J. Tremblay, E. Brachmann, B. Drost, V. Lepetit, C. Rother,
> S. Birchfield, J. Matas, Y. Labbé, M. Sundermeyer, T. Hodaň.
> *BOP Challenge 2024 on Model-Based and Model-Free 6D Object Pose Estimation.*
> 2025. arXiv: https://arxiv.org/abs/2504.02812 ·
> Leaderboards: https://bop.felk.cvut.cz/leaderboards/

**bop_toolkit** — the BOP dataset format, the symmetry-aware metrics
(VSD/MSSD/MSPD, ADD/ADI) and `get_symmetry_transformations`.
> T. Hodaň et al. *BOP Toolkit.*
> Code: https://github.com/thodan/bop_toolkit (MIT)

---

## Detector

**YOLOv8 (Ultralytics)** — the OBB detector backbone (`detector.pt`,
`yolov8s-obb`).
> Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics (AGPL-3.0)

> Note: Ultralytics ships under AGPL-3.0. The trained `detector.pt` weights are a
> project artefact; if the detector is to be redistributed as part of a closed
> product, review the Ultralytics licensing terms (a commercial license is
> available) or retrain on a permissively-licensed detector. This does not affect
> the BOP pose stack, which is MIT/Apache throughout. See
> [`../../THIRD_PARTY_LICENSES.md`](../../THIRD_PARTY_LICENSES.md).

---

## Simulation

**NVIDIA Isaac Sim** — the synthetic-data generator (arm-visible top-down PBR
frames). https://developer.nvidia.com/isaac/sim

---

## Related
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where each method sits in the pipeline.
- ADR-018 — the method-selection decision (why these, why the rejected ones).
