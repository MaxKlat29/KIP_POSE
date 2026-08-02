# GDRNPP Checkpoints — Handover

> Trained 6D-pose checkpoints for the BOP track (ADR-018). They live **on the GPU
> box only** (4.6 G total, RGB-only ConvNeXt-Base, 100 epochs each). They are
> **not** in git — pull them with the rsync command below when you need to run
> inference locally, or run inference on the box directly.

## What was trained

Per-object single-object (SO) GDRNPP models, RGB-only (`in_chans=3`,
DEPTH_BACKBONE disabled, `XYZ_ONLINE=True`, no depth-refine). Symmetry from
`models_info.json` is active in both the training loss (`PM_LOSS_SYM`) and the
eval metric. Three of the six parts are trained in this round.

| obj_id | part | checkpoint (on box) | size |
|---|---|---|---|
| 1 | Anker_Kurz | `output/gdrn/poseIsaacPbrSO/anker_kurz/model_final.pth` | 1.6 G |
| 2 | Anker_Lang | `output/gdrn/poseIsaacPbrSO/anker_lang/model_final.pth` | 1.6 G |
| 6 | Zahnrad   | `output/gdrn/poseIsaacPbrSO/zahnrad/model_final.pth`    | 1.6 G |

Box root: `/mnt/data/bop/repos/gdrnpp`
So the full path of e.g. Anker_Kurz is
`/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO/anker_kurz/model_final.pth`.

Per-object configs sit next to them under
`/mnt/data/bop/repos/gdrnpp/configs/gdrn/poseIsaacPbrSO/<part>.py`
(`base_so.py` is the shared base). Each `model_final.pth` is the **best/last**
checkpoint of a 100-epoch run; intermediate `model_00NNNNN.pth` are kept too
(checkpoint every 5 epochs).

Parts **not** trained this round: Buerstenhalter_2polig (3),
Getriebegehaeuse_typ4 (4), Ringmagnet (5) — no checkpoint yet.

## Accuracy (val split, symmetry-aware) — see `docs/PROJECT_REPORT.md` §4

| obj | part | AR | trans median (mm) | rot median (°, sym-resolved) |
|---|---|---|---|---|
| 1 | Anker_Kurz | 0.42 | 36.6 | 8.1 |
| 2 | Anker_Lang | 0.45 | 39.8 | 5.4 |
| 6 | Zahnrad   | 0.28 | 27.2 | 87.1 (rotation not converged) |

Beats the in-house baseline (91° / 186 mm) on Anker median rotation (11–17×) and
on translation for all three (~5×). Honest caveats in the report.

## How `infer.ipynb` / `e2e_infer.py` use a checkpoint

The pose backend is selected by the `--checkpoint` flag (laptop) or the
`GDRNPP_CHECKPOINT` variable (notebook). No checkpoint → MOCK backend; a real
`.pth` → real GDRNPP inference (the `_gdrnpp_real()` injection point in
`e2e_infer.py`).

```bash
# laptop, with a checkpoint pulled to e.g. project/models/anker_kurz.pth
python3 project/e2e_infer.py --image project/input/<scene>.png \
    --checkpoint project/models/anker_kurz.pth --serve
```

```python
# infer.ipynb
GDRNPP_CHECKPOINT = "project/models/anker_kurz.pth"   # or None -> MOCK
run(image, out_path, cfg=GdrnppConfig(checkpoint=GDRNPP_CHECKPOINT))
```

> Note: `_gdrnpp_real()` is the wiring point. Running real GDRNPP inference needs
> the full GDRNPP stack (torch + mmcv-full 1.7.2 + the patched repo), which is set
> up on the box, not on the laptop. For laptop demos use the MOCK backend; for
> real numbers run inference on the box (below).

## Pull the checkpoints to the laptop (when needed)

```bash
# all three (4.6 G):
rsync -azP \
  $GPU_HOST:'/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO/{anker_kurz,anker_lang,zahnrad}/model_final.pth' \
  ./project/models/

# or one, renamed:
rsync -azP \
  $GPU_HOST:/mnt/data/bop/repos/gdrnpp/output/gdrn/poseIsaacPbrSO/anker_kurz/model_final.pth \
  ./project/models/anker_kurz.pth
```

`*.pth` under `project/models/` should stay out of git (it is regenerable and
large). Keep them local-only.

## Reproduce the eval / a real pose_result on the box

```bash
# 1) combine the per-object GDRNPP prediction CSVs the training run wrote
/mnt/data/bop/bop-venv/bin/python /mnt/data/kip_pose/box_src/combine_preds.py \
  /mnt/data/bop/results/preds_all.csv \
  output/gdrn/poseIsaacPbrSO/anker_kurz/inference_/*/*.csv \
  output/gdrn/poseIsaacPbrSO/anker_lang/inference_/*/*.csv \
  output/gdrn/poseIsaacPbrSO/zahnrad/inference_/*/*.csv      # run from gdrnpp root

# 2) symmetry-aware BOP eval -> report.txt / report.json
/mnt/data/bop/bop-venv/bin/python /mnt/data/kip_pose/box_src/eval_bop.py \
  --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac \
  --split val --preds /mnt/data/bop/results/preds_all.csv \
  --out /mnt/data/bop/results/gdrnpp_val

# 3) a real (non-mock) pose_result.json + detection overlay for one frame
/mnt/data/bop/bop-venv/bin/python /mnt/data/kip_pose/box_src/real_pose_result.py \
  --dataset-dir /mnt/data/kip_pose/project/bop/pose_isaac --split val \
  --scene 1 --im 92 --preds /mnt/data/bop/results/preds_all.csv \
  --out-json /mnt/data/bop/results/pose_result.json \
  --out-overlay /mnt/data/bop/results/det_overlay.png
```

## Related

- `docs/PROJECT_REPORT.md` §4 (full numbers + honest verdict)
- `docs/EVAL.md` / `box_src/eval_bop.py` (the symmetry-aware harness)
- `bop_adapter.py` (the production camera→world adapter both tracks share)
- ADR-018 (BOP pivot) · ADR-017 (`pose_result` contract)
