"""GDRNPP per-object (SO) base config for pose_isaac — RGB-ONLY (Max HARD).

Deployed to configs/gdrn/poseIsaacPbrSO/base_so.py. Per-object thin configs
inherit from this and only set OUTPUT_DIR + DATASETS.{TRAIN,TEST}.

Modeled on configs/gdrn/itoddPbrSO/.../1.py (ITODD = the closest analogue:
RGB-only synthetic-PBR metal parts). RGB-only is the GDRN default — the net
backbone is in_chans=3 and DEPTH_BACKBONE is disabled in gdrn_base. Depth in the
BOP dataset is used ONLY to backproject XYZ crops during preprocessing; it is
never fed to the network and there is no depth-refine stage (test_gdrn.sh, not
test_gdrn_depth_refine.sh).

Eval: unlike itodd_test (no GT), our `val` split HAS GT, so we run normal pose
eval (ad/rete/proj + AR via bop_toolkit) on the held-out 200-frame val.
Symmetries come from models_info.json via get_symmetry_transformations -> the
PM_LOSS_SYM and the metric both handle the continuous/discrete ambiguity
(the analytic fix for the 120°/91° rotation problem, adr.md §2).
"""
_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/poseIsaacPbrSO/base_so"

INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=True,
    # [T-068] CHANGE_BG_PROB=0.0: our Isaac-PBR renders ALREADY contain a
    # realistic full scene (robot arm occluder + clutter + table). The GDRNPP
    # default 0.5 pastes random VOC2012 images behind the object, which (a)
    # requires the 2GB VOCdevkit download (the loader asserts BG ROOT exists ->
    # AssertionError "datasets/VOCdevkit/VOC2012/ does not exist") and (b) is the
    # WRONG augmentation for us — it would destroy the arm/clutter context that
    # is the whole point of the arm-visible dataset (ADR-018). Backgrounds stay
    # as rendered.
    CHANGE_BG_PROB=0.0,
    # [T-050 / PHASE2] 0.8 -> 0.9. Texture-less specular metal has NO real albedo
    # signal — the network must become shape-biased, not texture-biased (Thalhammer
    # et al. 2024, arXiv:2402.04878: randomized-texture / texture-destroying aug
    # gives +18.6% pose AR on ITODD metal). Pushing COLOR_AUG_PROB up + the strong
    # Grayscale/contrast/brightness chain below forces geometry-from-shading cues.
    # This is a generalizable training-data lever, not a per-part tweak. The
    # heavy lifting is in the DR-5k render (material/light randomization); this aug
    # is the on-the-fly complement.
    COLOR_AUG_PROB=0.9,
    MIN_SIZE_TRAIN=720,
    MAX_SIZE_TRAIN=1280,
    MIN_SIZE_TEST=720,
    MAX_SIZE_TEST=1280,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
        "], random_order=True)"
    ),
)

SOLVER = dict(
    # [T-050 / PHASE2] Batch 16 -> 24 because (a) the new sdg_armvis_dr5k DR set
    # roughly doubles usable instances after the visib>0.20 filter, and (b) AMP
    # leaves headroom on the 3090. KEEP 16 if the INPUT_RES bump (below) is also
    # enabled — higher crop res costs VRAM; do not raise both blindly. GPU
    # smoke-test required (see PHASE2_PLAN.md §Retrain validation).
    IMS_PER_BATCH=16,            # conservative for one 3090 24GB (adr.md §5-R1)
    # [T-050 / PHASE2] 100 -> 160 epochs. The Zahnrad never converged at 100
    # (median 87°, 1% <5°); texture-less small-feature in-plane orientation needs
    # more iterations. Anker was already good at 100, but the bigger DR set means
    # more variety per epoch — longer schedule amortises it. flat_and_anneal
    # keeps the LR flat then cosine-anneals at ANNEAL_POINT, so extending is safe.
    TOTAL_EPOCHS=160,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    AMP=dict(ENABLED=True),       # mixed precision -> lower VRAM (adr.md §5-R1)
    # [T-068 — OVERFITTING GUARD: checkpoint SPREAD for best-by-val selection]
    # GDRNPP has NO in-training validation (TEST.EVAL_PERIOD=0 below — its internal
    # eval crashes with `KeyError: pose_isaac` and would kill the run, see W4),
    # NO early-stopping, and the chain blindly deploys `model_final.pth` = the LAST
    # epoch (160), which on a 160-epoch RGB-only run on a small synthetic set is a
    # prime overfitting candidate. The common_base.py defaults (PERIOD=5,
    # MAX_TO_KEEP=5) keep ONLY the final ~25 epochs (5 ckpts × every-5th, all in
    # ep136-160) — useless for picking a non-overfit checkpoint because there is no
    # EARLY checkpoint to compare against. We instead keep a WIDE, EVEN spread over
    # the whole schedule so select_best_ckpt.py can score each on the held-out val
    # and pick the true AR-best (which may well be an earlier epoch).
    #   PERIOD=10, BY_EPOCH=True  -> a checkpoint at epochs 10,20,...,160 = 16 ckpts
    #   MAX_TO_KEEP=20            -> all 16 + model_final survive (no early deletion)
    # Disk cost: 16 × 1.6GB × 3 objects ≈ 77GB; /mnt/data has 3.2T free (9% used) →
    # trivial. If disk ever got tight, PERIOD=16 -> 10 ckpts is the fallback.
    CHECKPOINT_PERIOD=10,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=20,
)

DATASETS = dict(
    # [T-050 / PHASE2] The new DR-heavy set sdg_armvis_dr5k (5000 scenes,
    # --dr-strong: per-object roughness/metallic/tint, lights 120-2200, cam
    # jitter +-0.08 + roll +-12°, focal 14-24, clutter 8-16) is the Sim2Real +
    # shape-bias lever (see PHASE2_PLAN.md §Fix-2). The data-strang converts it
    # with `isaac_to_bop.py --min-visib 0.20`. WIRING (pick one — owned by the
    # data-strang's BOP layout, validate at GPU-free time):
    #   (A) MERGE: append the DR scenes into pose_isaac/train_pbr/ (renumbered
    #       scene dirs). Then the existing TRAIN names below already cover them —
    #       no config change. PREFERRED (keeps SO names stable).
    #   (B) SEPARATE split: register `pose_isaac_<obj>_train_dr` in
    #       ref_pose_isaac.py + pose_isaac_pbr.py and set TRAIN to the TUPLE
    #       ("pose_isaac_<obj>_train_pbr", "pose_isaac_<obj>_train_dr") — GDRNPP
    #       concatenates multiple train dataset names natively.
    TRAIN=("pose_isaac_anker_kurz_train_pbr",),   # overridden per-object
    TEST=("pose_isaac_anker_kurz_val",),          # overridden per-object; HAS GT
    # GT-box training (adr.md): TEST_BBOX_TYPE=gt below uses scene_gt_info boxes.
    DET_TOPK_PER_OBJ=100,
    DET_THR=0.1,
)

DATALOADER = dict(
    NUM_WORKERS=8,
    # [T-050 / PHASE2] 0.3 -> 0.2 to MATCH the uniform THR=0.20 visibility gate
    # that T-038 applied at the BOP-GT level (filter_visibility.py / --min-visib
    # 0.20). The data is already filtered at <=0.20 on disk; aligning the loader
    # threshold to 0.20 (not 0.30) keeps the 0.20-0.30 visibility band as TRAIN
    # signal instead of silently dropping it twice. A part 20-30% visible is still
    # pose-able and is exactly the arm-occluded regime we must learn.
    FILTER_VISIB_THR=0.2,
)

MODEL = dict(
    LOAD_DETS_TEST=False,         # use GT boxes for val (we have GT); det-bridge is for real inference
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        # [T-068] XYZ_ONLINE stays True. We investigated offline XYZ precompute as
        # a training-speed fix (online EGL renders the dense XYZ target once per
        # instance per iteration in the main loop, engine_utils.batch_data_train_online).
        # MEASURED (GPU-exclusive, anker_kurz): offline XYZ = 3.022 s/iter vs online
        # ~2.8-3.0 s/iter — IDENTICAL. The 16 per-iter EGL renders cost ~20 ms
        # (787 inst/s probe), <1% of the 2800 ms iter. The real per-iter cost is the
        # convnext_base forward/backward at INPUT_RES=320 (107M params) — the run is
        # COMPUTE-BOUND, not render- or IO-bound. Batch 16->24 confirmed it:
        # 3.02 s/iter@b16 (783 s/epoch) vs 4.38 s/iter@b24 (759 s/epoch), ~linear
        # scaling => no fixed overhead to amortize, batch increase doesn't help
        # either. There is NO cheap per-iter win without lowering INPUT_RES (the
        # forbidden Zahnrad small-feature fix). So we keep the exact proven config
        # and just relaunch. The offline path (pose_isaac_pbr_gen_xyz.py + the
        # precomputed <split>/xyz_crop pkls) is correct and committed for the record,
        # but NOT activated — switching the data path for zero speed gain adds risk
        # to an unattended multi-day run. See team/kai-ml/learnings.md (2026-05-25 T-068).
        XYZ_ONLINE=True,
        # [T-050 / PHASE2 — HIGHEST-IMPACT FIX for Zahnrad, GPU-GATED]
        # gdrn_base default is INPUT_RES=256 / OUTPUT_RES=64. The Zahnrad
        # (diameter 49.9mm, fine C_7 teeth) is crushed into a 256px crop and the
        # XYZ/region head only emits 64px — the per-tooth orientation cue that the
        # C_7 rotation depends on is below the output-grid resolution. This is the
        # concrete reason the gear's rotation never converged (median 87°, 1%
        # <5°, MSSD 0.043) while the metric self-test proves the C_7 symmetry
        # handling is correct. Raising to 320/80 restores small-feature detail and
        # generalizes to any small/fine-feature part (not a per-part hack).
        #
        # VRAM/shape RISK: convnext_base out_indices=(3,) feeds GEO_HEAD
        # in_dim=1024; the feature-map size scales with INPUT_RES. Must run a
        # 1-iter GPU smoke-test (CUDA_VISIBLE_DEVICES=0, --num-gpus 1, few iters)
        # to confirm no shape mismatch and that batch 16 still fits before the
        # multi-day run. If 320 OOMs, drop IMS_PER_BATCH to 12. See
        # PHASE2_PLAN.md §Retrain validation.
        INPUT_RES=320,
        OUTPUT_RES=80,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,          # RGB-only
                features_only=True,
                out_indices=(3,),
            ),
        ),
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(type="TopDownDoubleMaskXyzRegionHead", in_dim=1024),
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            XYZ_LOSS_TYPE="L1", XYZ_LOSS_MASK_GT="visib", XYZ_LW=1.0,
            MASK_LOSS_TYPE="L1", MASK_LOSS_GT="trunc", MASK_LW=1.0,
            FULL_MASK_LOSS_TYPE="L1", FULL_MASK_LW=1.0,
            REGION_LOSS_TYPE="CE", REGION_LOSS_MASK_GT="visib", REGION_LW=1.0,
            PM_LOSS_SYM=True,        # symmetry-aware pose loss (adr.md §2)
            PM_R_ONLY=True, PM_LW=1.0,
            CENTROID_LOSS_TYPE="L1", CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1", Z_LW=1.0,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="pose_isaac",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_pose_isaac_val.json",
    ERROR_TYPES="ad,reS,teS,proj",   # AR + ADD/ADI(sym), translation/rotation
    RENDERER_TYPE="cpp",
    SPLIT="val",
    SPLIT_TYPE="",
    N_TOP=-1,
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=True,
    SAVE_BOP_CSV_ONLY=False,          # val HAS GT -> compute scores
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="gt")  # GT boxes for val
