#!/usr/bin/env python3
"""GDRNPP <-> pytorch-lightning 1.9 compatibility patch (T-068 blocker fix).

GDRNPP was tested on PL 1.6.0.dev0. The gdrnpp-venv runs torch 2.5.1+cu121, for
which PL 1.6 is too old, so we pin PL==1.9.0. PL 1.9 keeps `LightningLite`
(works as-is at the 3 `from pytorch_lightning.lite import LightningLite` sites)
but RENAMED the lite wrappers: `pytorch_lightning.lite.wrappers._LiteModule` ->
`lightning_fabric.wrappers._FabricModule`.

Only ONE GDRNPP file breaks: core/utils/my_checkpoint.py imports `_LiteModule`.
This idempotently rewrites that import to the PL 1.9 path (aliased back to the
name the rest of the file uses), with a try/except fallback so it also still
works if anyone reverts PL. No behaviour change — `_FabricModule` is the renamed
`_LiteModule`, used only in an isinstance() unwrap.

Run ON the box:
  /mnt/data/bop/gdrnpp-venv/bin/python box_src/gdrnpp/patch_pl19_compat.py
"""
import os

GDRN = "/mnt/data/bop/repos/gdrnpp"
TARGET = os.path.join(GDRN, "core/utils/my_checkpoint.py")

OLD = "from pytorch_lightning.lite.wrappers import _LiteModule"
NEW = (
    "try:\n"
    "    from pytorch_lightning.lite.wrappers import _LiteModule  # PL<=1.6\n"
    "except ImportError:\n"
    "    from lightning_fabric.wrappers import _FabricModule as _LiteModule  # PL 1.9 rename\n"
)


def main():
    s = open(TARGET).read()
    if "lightning_fabric.wrappers import _FabricModule as _LiteModule" in s:
        print("PL19_PATCH already applied")
        return
    if OLD not in s:
        print(f"PL19_PATCH WARN: anchor not found in {TARGET} (already patched or moved?)")
        return
    s = s.replace(OLD, NEW, 1)
    open(TARGET, "w").write(s)
    print(f"PL19_PATCH applied to {TARGET}")


if __name__ == "__main__":
    main()
