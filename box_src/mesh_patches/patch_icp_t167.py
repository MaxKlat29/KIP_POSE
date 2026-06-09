"""T-167: gigapose_infer._icp env-tunable machen (max_corr/iters/estimation/snap-multi-radius).

Hebel fuer den RGB-D-Translations-Fix:
  GP_ICP_MAX_CORR  (default 0.01)  open3d-ICP-Korrespondenz-Radius (m). Theo: lateraler
                                   t-Fehler ~33mm > 1cm -> ICP greift nicht. Groesser
                                   ziehen heisst: weiter entfernte Korrespondenzen zaehlen.
  GP_ICP_ITERS     (default 50)    max_iteration der ICPConvergenceCriteria.
  GP_ICP_ESTIMATION(default point) 'point' (PointToPoint) | 'plane' (PointToPlane, braucht
                                   Normalen -> auf der Ziel-Pointcloud geschaetzt).

Reiner Refiner-Knopf, 0 zusaetzliches VRAM (open3d laeuft auf CPU). Idempotent: prueft auf
den env-Marker und no-op't, wenn schon gepatcht.
"""
import io
import re
import sys

PATH = "/workspace/GigaPose/gigapose_infer.py"
src = io.open(PATH, encoding="utf-8").read()

if "GP_ICP_MAX_CORR" in src:
    print("[patch_icp_t167] already patched (GP_ICP_MAX_CORR present) -> no-op")
    sys.exit(0)

# 1) Signatur: harte Defaults -> None (damit Env vorrangig ist).
old_sig = "    def _icp(self, T, K, depth_m, mask, obj_id, max_corr=0.01, iters=50):"
new_sig = "    def _icp(self, T, K, depth_m, mask, obj_id, max_corr=None, iters=None):"
assert old_sig in src, "signature line not found"
src = src.replace(old_sig, new_sig, 1)

# 2) Env-Auswertung direkt nach dem Import-Block im _icp einsetzen (vor der Mask-Logik).
#    Wir haengen sie nach 'import trimesh' an, das ist die erste Zeile im _icp-Body.
anchor = "        import open3d as o3d\n        import trimesh\n"
assert anchor in src, "icp import anchor not found"
env_block = anchor + (
    "        # T-167: Refiner-Knoepfe aus Env (0 VRAM, open3d=CPU). max_corr groesser\n"
    "        # zieht weiter entfernte Korrespondenzen rein (Theo: lateraler t-Fehler\n"
    "        # ~33mm > 1cm Default-Radius -> ICP konnte den lateralen Drift nicht fangen).\n"
    "        if max_corr is None:\n"
    "            max_corr = float(os.environ.get('GP_ICP_MAX_CORR', '0.01'))\n"
    "        if iters is None:\n"
    "            iters = int(os.environ.get('GP_ICP_ITERS', '50'))\n"
    "        _icp_est = os.environ.get('GP_ICP_ESTIMATION', 'point').lower()\n"
)
src = src.replace(anchor, env_block, 1)

# 3) ICP-Aufruf: PointToPlane optional. Bei 'plane' Normalen auf der Ziel-Wolke schaetzen.
old_call = (
    "        ps = o3d.geometry.PointCloud(); ps.points = o3d.utility.Vector3dVector(src)\n"
    "        po = o3d.geometry.PointCloud(); po.points = o3d.utility.Vector3dVector(obs)\n"
    "        reg = o3d.pipelines.registration.registration_icp(\n"
    "            ps, po, max_corr, np.eye(4),\n"
    "            o3d.pipelines.registration.TransformationEstimationPointToPoint(),\n"
    "            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),\n"
    "        )\n"
    "        return np.asarray(reg.transformation) @ T\n"
)
new_call = (
    "        ps = o3d.geometry.PointCloud(); ps.points = o3d.utility.Vector3dVector(src)\n"
    "        po = o3d.geometry.PointCloud(); po.points = o3d.utility.Vector3dVector(obs)\n"
    "        if _icp_est == 'plane':\n"
    "            # PointToPlane: Ziel-Normalen aus der Depth-Pointcloud schaetzen.\n"
    "            po.estimate_normals(\n"
    "                o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr * 2.0, max_nn=30))\n"
    "            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()\n"
    "        else:\n"
    "            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()\n"
    "        reg = o3d.pipelines.registration.registration_icp(\n"
    "            ps, po, max_corr, np.eye(4), estimation,\n"
    "            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),\n"
    "        )\n"
    "        return np.asarray(reg.transformation) @ T\n"
)
assert old_call in src, "icp call block not found"
src = src.replace(old_call, new_call, 1)

io.open(PATH, "w", encoding="utf-8").write(src)
print("[patch_icp_t167] patched: GP_ICP_MAX_CORR / GP_ICP_ITERS / GP_ICP_ESTIMATION wired")

# Syntax-Selbsttest.
import py_compile
py_compile.compile(PATH, doraise=True)
print("[patch_icp_t167] py_compile OK")
