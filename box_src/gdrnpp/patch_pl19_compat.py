#!/usr/bin/env python3
"""GDRNPP <-> pytorch-lightning 1.9 compatibility patch (T-068 blocker fix).

GDRNPP was tested on PL 1.6.0.dev0. The gdrnpp-venv runs torch 2.5.1+cu121, for
which PL 1.6 is too old, so we pin PL==1.9.0. PL 1.9 keeps `LightningLite`
(works as-is at the 3 `from pytorch_lightning.lite import LightningLite` sites)
but RENAMED the lite wrappers: `pytorch_lightning.lite.wrappers._LiteModule` ->
`lightning_fabric.wrappers._FabricModule`.

Two GDRNPP sites break against PL 1.9 (idempotent, guarded patches):

  1. core/utils/my_checkpoint.py imports `_LiteModule`. PL 1.9 renamed the lite
     wrapper: `pytorch_lightning.lite.wrappers._LiteModule` ->
     `lightning_fabric.wrappers._FabricModule`. Rewritten with a try/except
     fallback (no behaviour change — used only in an isinstance() unwrap).

  2. core/gdrn_modeling/engine/engine.py:223 accesses `self._precision_plugin`
     (PL<=1.6 Lite attr) to grab the AMP GradScaler for the checkpoint dict.
     PL 1.9 removed that attr; the precision plugin now lives at
     `self._strategy.precision` (a MixedPrecision plugin with `.scaler`). Without
     this fix do_train raises `AttributeError: 'Lite' object has no attribute
     '_precision_plugin'` AFTER the renderer loads (T-068: this was the next
     blocker once the EGL crash was fixed) -> 0 checkpoints. Rewritten to a
     version-robust getattr that prefers the PL 1.9 path and falls back to 1.6.

Run ON the box:
  /mnt/data/bop/gdrnpp-venv/bin/python box_src/gdrnpp/patch_pl19_compat.py
"""
import os

GDRN = os.environ.get("GDRN", "/mnt/data/bop/repos/gdrnpp")
CKPT = os.path.join(GDRN, "core/utils/my_checkpoint.py")
ENGINE = os.path.join(GDRN, "core/gdrn_modeling/engine/engine.py")

CKPT_OLD = "from pytorch_lightning.lite.wrappers import _LiteModule"
CKPT_NEW = (
    "try:\n"
    "    from pytorch_lightning.lite.wrappers import _LiteModule  # PL<=1.6\n"
    "except ImportError:\n"
    "    from lightning_fabric.wrappers import _FabricModule as _LiteModule  # PL 1.9 rename\n"
)

ENGINE_OLD = '        if hasattr(self._precision_plugin, "scaler"):\n            extra_ckpt_dict["gradscaler"] = self._precision_plugin.scaler'
ENGINE_NEW = (
    "        # [T-068] PL 1.9: _precision_plugin removed; plugin is at\n"
    "        # self._strategy.precision (MixedPrecision, has .scaler). Fall back\n"
    "        # to PL<=1.6 self._precision_plugin if present.\n"
    "        _prec_plugin = getattr(getattr(self, \"_strategy\", None), \"precision\", None)\n"
    "        if _prec_plugin is None:\n"
    "            _prec_plugin = getattr(self, \"_precision_plugin\", None)\n"
    '        if hasattr(_prec_plugin, "scaler"):\n'
    '            extra_ckpt_dict["gradscaler"] = _prec_plugin.scaler'
)


def patch_ckpt():
    s = open(CKPT).read()
    if "lightning_fabric.wrappers import _FabricModule as _LiteModule" in s:
        return "my_checkpoint _LiteModule: already patched"
    if CKPT_OLD not in s:
        return "my_checkpoint _LiteModule: WARN anchor not found"
    open(CKPT, "w").write(s.replace(CKPT_OLD, CKPT_NEW, 1))
    return "my_checkpoint _LiteModule: patched"


def patch_engine_precision():
    s = open(ENGINE).read()
    if "[T-068] PL 1.9: _precision_plugin removed" in s:
        return "engine _precision_plugin: already patched"
    if ENGINE_OLD not in s:
        return "engine _precision_plugin: WARN anchor not found"
    open(ENGINE, "w").write(s.replace(ENGINE_OLD, ENGINE_NEW, 1))
    return "engine _precision_plugin: patched"


def main():
    for fn in (patch_ckpt, patch_engine_precision):
        print("  ", fn())
    print("PL19_PATCH_DONE")


if __name__ == "__main__":
    main()
