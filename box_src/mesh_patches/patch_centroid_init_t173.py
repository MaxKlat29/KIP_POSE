"""T-173: LATERALER depth-init Translations-Prior fuer GigaPose-3D coarse-T.

Abgrenzung zu T-170 (verworfen):
  T-170 (GP_PRE_REFINE_SNAP) snappte die coarse-T nur RADIAL entlang des Kamerastrahls
  auf die Median-Maskentiefe (T[:3,3] *= z_med/Zc). Das ist projektions-erhaltend
  (x/z, y/z bleiben) und korrigiert NUR den Tiefen-Float. Theo T-166 hat aber gezeigt:
  der RGB-D-t-Fehler ist random LATERAL (~33mm in x/y), nicht radial. Ein radialer
  Snap kann einen lateralen Drift per Definition NICHT fangen -> T-170 minimal schlechter.

Dieser Hebel (NEU, von T-170 nie getestet):
  Ersetze die coarse-Translation VOLLSTAENDIG durch die Backprojektion des Masken-
  Centroids — exakt FoundationPose's guess_translation:
      uc,vc = BBox-Mitte der Maske,  z = Median-Tiefe unter der Maske
      t = K^-1 @ [uc, vc, 1] * z      (lateral X=(uc-cx)z/fx, Y=(vc-cy)z/fy + radial Z=z)
  Rotation der coarse-Pose bleibt unberuehrt (Theo: Rotation top ~5 Grad). Das setzt
  einen metrisch UND lateral korrekten 3D-Startpunkt fuer den MegaPose-Refiner — der
  einzige Mechanismus der den lateralen Drift fangen koennte (FP nutzt genau diesen
  Init und trifft die Translation deutlich besser als GigaPose's Template-Scale-coarse).

  Reiner Numpy-Schritt, 0 zusaetzliches VRAM. env-gated, default 0 (byte-identisch).

Env:
  GP_CENTROID_INIT (default 0)  1 = coarse-T vor _refine durch Masken-Centroid-
                                Backprojektion ersetzen (lateral+radial depth-init).

Idempotent: prueft auf den env-Marker und no-op't, wenn schon gepatcht.
"""
import io
import sys

PATH = "/workspace/GigaPose/gigapose_infer.py"
src = io.open(PATH, encoding="utf-8").read()

if "GP_CENTROID_INIT" in src:
    print("[patch_centroid_init_t173] already patched -> no-op")
    sys.exit(0)

# Voraussetzung: der T-170-Patch ist drin (liefert den estimate-Hook-Kontext).
assert "GP_PRE_REFINE_SNAP" in src, (
    "expected T-170 patch (GP_PRE_REFINE_SNAP) to be applied first")

# 1) Helfer-Methode direkt nach _pre_refine_snap einfuegen (gleiche Stelle, vor 'public').
anchor = "    # ------------------------------------------------------------------- public\n    @torch.no_grad()\n    def estimate(self, class_name, K, rgb, mask, depth=None, iterations=5, hypotheses=5, kabsch=False):"
assert anchor in src, "estimate anchor not found"

helper = (
    "    # ------------------------------------------------- centroid depth-init\n"
    "    def _centroid_init_snap(self, coarse_poses, K, depth_m, mask):\n"
    "        \"\"\"T-173: coarse-Translation VOLLSTAENDIG durch die Backprojektion des\n"
    "        Masken-Centroids ersetzen (lateral X=(uc-cx)z/fx, Y=(vc-cy)z/fy + radial\n"
    "        Z=z_med) — exakt FoundationPose's guess_translation. Faengt Theos lateralen\n"
    "        t-Drift, den der radiale T-170-Snap nicht fangen konnte. Rotation bleibt.\"\"\"\n"
    "        ys, xs = np.nonzero(mask > 0)\n"
    "        if xs.size < 50:\n"
    "            return coarse_poses\n"
    "        uc = (float(xs.min()) + float(xs.max())) / 2.0\n"
    "        vc = (float(ys.min()) + float(ys.max())) / 2.0\n"
    "        valid = (mask > 0) & (depth_m > 1e-3)\n"
    "        if not valid.any():\n"
    "            return coarse_poses\n"
    "        z_med = float(np.median(depth_m[valid]))\n"
    "        if not (z_med > 1e-6):\n"
    "            return coarse_poses\n"
    "        Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))\n"
    "        t = (Kinv @ np.asarray([uc, vc, 1.0]).reshape(3, 1)).reshape(3) * z_med\n"
    "        cp = coarse_poses.clone()\n"
    "        import torch as _torch\n"
    "        t_t = _torch.tensor(t, dtype=cp.dtype, device=cp.device)\n"
    "        for i in range(cp.shape[0]):\n"
    "            cp[i, :3, 3] = t_t\n"
    "        return cp\n"
    "\n"
)
src = src.replace(anchor, helper + anchor, 1)

# 2) Im estimate(): unmittelbar nach dem T-170-Snap-Block den Centroid-Init einhaengen.
#    Reihenfolge: GP_CENTROID_INIT gewinnt (ueberschreibt komplett), wenn beide an
#    waeren. In der Praxis testen wir genau EINEN Schalter zur Zeit (A/B).
old_block = (
    "        # T-170 depth-init: coarse-T vor dem Refiner auf Median-Maskentiefe snappen\n"
    "        # (env-gated, nur RGB-D). Gibt MegaPose einen metrischen 3D-Start.\n"
    "        if depth is not None and os.environ.get('GP_PRE_REFINE_SNAP', '0') == '1':\n"
    "            coarse_poses = self._pre_refine_snap(\n"
    "                coarse_poses, K, np.asarray(depth, dtype=np.float64), mask)\n"
)
new_block = old_block + (
    "        # T-173 LATERALER depth-init: coarse-T vor dem Refiner durch Masken-Centroid-\n"
    "        # Backprojektion ersetzen (lateral+radial, env-gated, nur RGB-D). Der eine\n"
    "        # Mechanismus der Theos lateralen t-Drift fangen koennte (FP nutzt genau das).\n"
    "        if depth is not None and os.environ.get('GP_CENTROID_INIT', '0') == '1':\n"
    "            coarse_poses = self._centroid_init_snap(\n"
    "                coarse_poses, K, np.asarray(depth, dtype=np.float64), mask)\n"
)
assert old_block in src, "T-170 snap block not found (apply patch_pre_refine_snap_t170 first)"
src = src.replace(old_block, new_block, 1)

io.open(PATH, "w", encoding="utf-8").write(src)
print("[patch_centroid_init_t173] patched: GP_CENTROID_INIT wired (lateral centroid backproject)")

import py_compile
py_compile.compile(PATH, doraise=True)
print("[patch_centroid_init_t173] py_compile OK")
