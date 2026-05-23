#!/usr/bin/env python3
"""GDRNPP source patches for Python 3.11 + modern stack (T-068 blocker fixes).

GDRNPP targets Python 3.6-3.8 / numpy<1.24 / PL 1.6. On the box it runs Python
3.11 + torch 2.5.1. These idempotent source patches close the gaps that pip-pins
alone cannot. Run ON the box (any venv python):

  python box_src/gdrnpp/patch_gdrnpp_py311.py

Patches (each guarded, safe to re-run):
  1. core/utils/data_utils.py      : `from collections import Sequence` ->
                                     `from collections.abc import Sequence`
                                     (Sequence moved to collections.abc in 3.10)
  2. lib/utils/is_binary_file.py   : remove the Python-2 `unicode()` fallback +
                                     guard `encoding=None` (chardet returns None)
                                     so load_ply works on binary PLYs in 3.11
  3. lib/egl_renderer/cpp/egl_renderer.cpp : add `#include <iostream>` (gcc 13
                                     no longer transitively includes it -> std::cout)

NOTE: the PL 1.9 `_LiteModule`->`_FabricModule` rename is handled separately by
patch_pl19_compat.py (different file, different mechanism).
"""
import os

GDRN = "/mnt/data/bop/repos/gdrnpp"


def patch_collections():
    p = os.path.join(GDRN, "core/utils/data_utils.py")
    s = open(p).read()
    old = "from collections import Sequence, defaultdict, deque"
    new = "from collections import defaultdict, deque\nfrom collections.abc import Sequence"
    if "from collections.abc import Sequence" in s:
        return "collections: already patched"
    if old in s:
        open(p, "w").write(s.replace(old, new, 1))
        return "collections: patched"
    return "collections: WARN anchor not found"


def patch_is_binary():
    p = os.path.join(GDRN, "lib/utils/is_binary_file.py")
    s = open(p).read()
    old = (
        '    if detected_encoding["confidence"] > 0.9 and detected_encoding["encoding"] != "ascii":\n'
        "        try:\n"
        "            try:\n"
        '                bytes_to_check.decode(encoding=detected_encoding["encoding"])\n'
        "            except TypeError:\n"
        "                # happens only on Python 2.6\n"
        '                unicode(bytes_to_check, encoding=detected_encoding["encoding"])  # noqa\n'
        "            decodable_as_unicode = True"
    )
    new = (
        '    if (detected_encoding["confidence"] > 0.9 and detected_encoding["encoding"] not in ("ascii", None)):\n'
        "        try:\n"
        '            bytes_to_check.decode(encoding=detected_encoding["encoding"])\n'
        "            decodable_as_unicode = True"
    )
    if 'not in ("ascii", None)' in s:
        return "is_binary_file: already patched"
    if old in s:
        open(p, "w").write(s.replace(old, new, 1))
        return "is_binary_file: patched"
    return "is_binary_file: WARN anchor not found"


def patch_egl_iostream():
    p = os.path.join(GDRN, "lib/egl_renderer/cpp/egl_renderer.cpp")
    s = open(p).read()
    if "#include <iostream>" in s:
        return "egl iostream: already present"
    if "#include <cstdint>" in s:
        open(p, "w").write(s.replace("#include <cstdint>", "#include <cstdint>\n#include <iostream>", 1))
        return "egl iostream: patched"
    return "egl iostream: WARN anchor not found"


if __name__ == "__main__":
    for fn in (patch_collections, patch_is_binary, patch_egl_iostream):
        print("  ", fn())
    print("PATCH_GDRNPP_PY311_DONE")
