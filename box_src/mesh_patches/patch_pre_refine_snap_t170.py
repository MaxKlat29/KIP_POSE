"""T-170: depth-init Translations-Prior VOR dem MegaPose-Refiner (pre-refine snap).

Hypothese (Max-Direktive, feasibler Versuch):
  Der GigaPose-Adapter snappt die Translation auf die Median-Maskentiefe (GP_DEPTH_SNAP)
  + ICP erst NACH _refine (post-MegaPose). Die coarse-Translation kommt aus
  Template-Scale-Matching (unzuverlaessig, cm-bis-m-Fehler) und geht UNGESNAPPT in den
  MegaPose-Refiner. Wenn der Refiner schon mit einem metrisch korrekten 3D-Startpunkt
  beginnt (coarse-T entlang des Kamerastrahls auf die beobachtete Median-Tiefe unter der
  Maske gezogen), sollte er weniger lateral-rauschen (Theo T-166: t-Fehler ~36mm random).

  Projektions-erhaltend: T[:3,3] *= median(z)/Zc laesst (x/z, y/z) unveraendert, korrigiert
  nur den groben Tiefen-Float, bevor MegaPose iteriert. Reiner Numpy-Schritt, 0 VRAM.

Env:
  GP_PRE_REFINE_SNAP (default 0)  1 = coarse-T vor _refine auf Median-Maskentiefe snappen.

Idempotent: prueft auf den env-Marker und no-op't, wenn schon gepatcht.
"""
import io
import sys

PATH = "/workspace/GigaPose/gigapose_infer.py"
src = io.open(PATH, encoding="utf-8").read()

if "GP_PRE_REFINE_SNAP" in src:
    print("[patch_pre_refine_snap_t170] already patched -> no-op")
    sys.exit(0)

# 1) Helfer-Methode direkt vor der oeffentlichen estimate()-Methode einfuegen.
anchor = "    # ------------------------------------------------------------------- public\n    @torch.no_grad()\n    def estimate(self, class_name, K, rgb, mask, depth=None, iterations=5, hypotheses=5, kabsch=False):"
assert anchor in src, "estimate anchor not found"

helper = (
    "    # --------------------------------------------------------- pre-refine snap\n"
    "    def _pre_refine_snap(self, coarse_poses, K, depth_m, mask):\n"
    "        \"\"\"T-170: coarse-Translation entlang des Kamerastrahls auf die beobachtete\n"
    "        Median-Maskentiefe ziehen, BEVOR der MegaPose-Refiner laeuft. Gibt dem\n"
    "        Refiner einen metrisch korrekten 3D-Startpunkt statt des verrauschten\n"
    "        Template-Scale-Werts. Projektions-erhaltend (x/z, y/z bleiben), env-gated.\"\"\"\n"
    "        valid = (mask > 0) & (depth_m > 0)\n"
    "        ys, xs = np.nonzero(valid)\n"
    "        if xs.size < 50:\n"
    "            return coarse_poses\n"
    "        z_med = float(np.median(depth_m[ys, xs]))\n"
    "        if not (z_med > 1e-6):\n"
    "            return coarse_poses\n"
    "        cp = coarse_poses.clone()\n"
    "        for i in range(cp.shape[0]):\n"
    "            Zc = float(cp[i, 2, 3])\n"
    "            if Zc > 1e-6:\n"
    "                cp[i, :3, 3] = cp[i, :3, 3] * (z_med / Zc)\n"
    "        return cp\n"
    "\n"
)
src = src.replace(anchor, helper + anchor, 1)

# 2) In estimate(): nach _coarse, vor _refine die coarse_poses snappen, wenn aktiviert.
#    depth ist in estimate() als Param vorhanden (RGB-D-Pfad). Wir snappen NUR wenn
#    depth da ist (RGB-Pfad bleibt byte-identisch).
old_refine_block = (
    "        if self.pose_estimator is not None:\n"
    "            try:\n"
    "                Tr, sc = self._refine(obj_id, rgb, K, coarse_poses[:h], iterations)\n"
)
new_refine_block = (
    "        # T-170 depth-init: coarse-T vor dem Refiner auf Median-Maskentiefe snappen\n"
    "        # (env-gated, nur RGB-D). Gibt MegaPose einen metrischen 3D-Start.\n"
    "        if depth is not None and os.environ.get('GP_PRE_REFINE_SNAP', '0') == '1':\n"
    "            coarse_poses = self._pre_refine_snap(\n"
    "                coarse_poses, K, np.asarray(depth, dtype=np.float64), mask)\n"
    "        if self.pose_estimator is not None:\n"
    "            try:\n"
    "                Tr, sc = self._refine(obj_id, rgb, K, coarse_poses[:h], iterations)\n"
)
assert old_refine_block in src, "refine block not found"
src = src.replace(old_refine_block, new_refine_block, 1)

io.open(PATH, "w", encoding="utf-8").write(src)
print("[patch_pre_refine_snap_t170] patched: GP_PRE_REFINE_SNAP wired (pre-refine median-depth snap)")

import py_compile
py_compile.compile(PATH, doraise=True)
print("[patch_pre_refine_snap_t170] py_compile OK")
