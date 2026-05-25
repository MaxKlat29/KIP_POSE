#!/usr/bin/env python3
"""GDRNPP ConvPnPNet final_spatial_size <- OUTPUT_RES patch (T-068 / PHASE-2).

WHY
----
Phase-2 raises MODEL.POSE_NET.INPUT_RES 256->320 (OUTPUT_RES 64->80) to give the
Zahnrad's small inner-spline teeth more feature-map resolution (the Zahnrad
rotation never converged at 256). But GDRNPP's ConvPnPNet hard-codes its flatten
dimension to an 8x8 final feature map:

    # core/gdrn_modeling/models/heads/conv_pnp_net.py
    def __init__(self, ..., num_stride2_layers=3, final_spatial_size=(8, 8), ...):
        final_h, final_w = final_spatial_size
        fc_in_dim = {"flatten": featdim * final_h * final_w, ...}[spatial_pooltype]
        self.fc1 = nn.Linear(fc_in_dim, 1024)

ConvPnPNet applies `num_stride2_layers=3` stride-2 convs, i.e. it downsamples the
geo-head maps by /8. So the final spatial size is OUTPUT_RES/8:
    OUTPUT_RES=64 -> 8x8  -> fc_in_dim = 128*8*8  = 8192   (the hard-coded default)
    OUTPUT_RES=80 -> 10x10-> fc_in_dim = 128*10*10= 12800

`get_pnp_net()` (core/gdrn_modeling/models/model_utils.py) builds the ConvPnPNet
init cfg but NEVER passes final_spatial_size, so at OUTPUT_RES=80 the conv stack
emits 16x12800 into the 8192-wide fc1:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x12800 and 8192x1024)

This is NOT an OOM and NOT an argparse error (T-068 smoke discrimination correctly
hard-stopped instead of laddering down). It's a genuine architecture wiring gap
for any OUTPUT_RES != 64.

FIX
----
Make get_pnp_net derive final_spatial_size from OUTPUT_RES // 8 and pass it into
the ConvPnPNet init cfg. At OUTPUT_RES=64 this is (8,8) == the old default, so
the patch is a NO-OP for the legacy resolution and ONLY changes behaviour for the
new 320/80 (and any future res). Idempotent + anchor-guarded.

Run ON the box:
  /mnt/data/bop/gdrnpp-venv/bin/python box_src/gdrnpp/patch_pnp_spatial_size.py
"""
import os

GDRN = os.environ.get("GDRN", "/mnt/data/bop/repos/gdrnpp")
MODEL_UTILS = os.path.join(GDRN, "core/gdrn_modeling/models/model_utils.py")

# Anchor: the ConvPnPNet/ConvPnPNetCls branch of get_pnp_net that builds the
# init cfg. We INSERT a final_spatial_size kwarg derived from OUTPUT_RES.
ANCHOR = (
    '    if pnp_head_type in ["ConvPnPNet", "ConvPnPNetCls"]:\n'
    "        pnp_net_init_cfg.update(\n"
    "            nIn=pnp_net_in_channel,\n"
    "            rot_dim=rot_dim,\n"
    "            num_regions=g_head_cfg.NUM_REGIONS,\n"
    "            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,\n"
    "        )"
)

REPLACEMENT = (
    '    if pnp_head_type in ["ConvPnPNet", "ConvPnPNetCls"]:\n'
    "        # [T-068/PHASE-2] ConvPnPNet downsamples the geo-head maps by /8\n"
    "        # (num_stride2_layers=3) before flattening into fc1. Its default\n"
    "        # final_spatial_size=(8,8) only matches OUTPUT_RES=64. Derive it\n"
    "        # from OUTPUT_RES so a raised INPUT_RES/OUTPUT_RES (320/80 for the\n"
    "        # Zahnrad small-feature fix) doesn't crash with a 12800-vs-8192\n"
    "        # mat-mul shape mismatch. At OUTPUT_RES=64 this is (8,8) == the old\n"
    "        # default, so this is a no-op for the legacy resolution.\n"
    "        _out_res = int(net_cfg.OUTPUT_RES)\n"
    "        _num_s2 = int(pnp_net_init_cfg.get('num_stride2_layers', 3))\n"
    "        _final_spatial = max(1, _out_res // (2 ** _num_s2))\n"
    "        pnp_net_init_cfg.update(\n"
    "            nIn=pnp_net_in_channel,\n"
    "            rot_dim=rot_dim,\n"
    "            num_regions=g_head_cfg.NUM_REGIONS,\n"
    "            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,\n"
    "            final_spatial_size=(_final_spatial, _final_spatial),\n"
    "        )"
)

MARKER = "[T-068/PHASE-2] ConvPnPNet downsamples the geo-head maps by /8"


def patch_model_utils():
    if not os.path.isfile(MODEL_UTILS):
        return f"model_utils.py: WARN not found at {MODEL_UTILS}"
    s = open(MODEL_UTILS).read()
    if MARKER in s:
        return "get_pnp_net final_spatial_size: already patched"
    if ANCHOR not in s:
        return ("get_pnp_net final_spatial_size: WARN anchor not found "
                "(ConvPnPNet update block changed upstream?) — NOT patched")
    open(MODEL_UTILS, "w").write(s.replace(ANCHOR, REPLACEMENT, 1))
    return "get_pnp_net final_spatial_size: patched"


def main():
    print("  ", patch_model_utils())
    print("PNP_SPATIAL_SIZE_PATCH_DONE")


if __name__ == "__main__":
    main()
