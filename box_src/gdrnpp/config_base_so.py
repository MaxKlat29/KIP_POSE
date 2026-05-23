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
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
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
    IMS_PER_BATCH=16,            # conservative for one 3090 24GB (adr.md §5-R1)
    TOTAL_EPOCHS=100,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    AMP=dict(ENABLED=True),       # mixed precision -> lower VRAM (adr.md §5-R1)
)

DATASETS = dict(
    TRAIN=("pose_isaac_anker_kurz_train_pbr",),   # overridden per-object
    TEST=("pose_isaac_anker_kurz_val",),          # overridden per-object; HAS GT
    # GT-box training (adr.md): TEST_BBOX_TYPE=gt below uses scene_gt_info boxes.
    DET_TOPK_PER_OBJ=100,
    DET_THR=0.1,
)

DATALOADER = dict(
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.3,         # skip near-fully-arm-occluded instances in train
)

MODEL = dict(
    LOAD_DETS_TEST=False,         # use GT boxes for val (we have GT); det-bridge is for real inference
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        XYZ_ONLINE=True,
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
