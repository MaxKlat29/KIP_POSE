#!/usr/bin/env python3
"""GDRNPP EGL renderer mesh-normal fix (T-068 training crash).

ROOT CAUSE
----------
The GDRNPP EGL renderer crashed for all 3 objects at
`lib/egl_renderer/egl_renderer_v3.py:729 load_object` ->
`glutils/meshutil.py calc_normals`:

  * anker_kurz / anker_lang : Segmentation fault rc=139 (core dumped)
  * zahnrad                 : IndexError 'index 2771 out of bounds, size 2771'

`calc_normals(vertices)` assumes `vertices` is a *flat triangle soup*
(it loops `for i in range(0, N-1, 3): v1,v2,v3 = vertices[i:i+3]`).
But our BOP models are INDEXED meshes exported by trimesh:

  element vertex N   (unique vertices, NO nx/ny/nz normals)
  element face   M   (property list uchar int vertex_indices)

So `recalculate_normals or "normals" not in model` is always True ->
`calc_normals` runs on the *unique* vertex array:
  * zahnrad obj6 = 2771 verts -> last i=2769 -> vertices[2771] -> IndexError
  * anker    = 23241/25060 verts -> garbage normals + soup-strided write ->
               EGL/GL upload reads bad memory -> segfault.

This is NOT a headless-EGL problem (EGL inits fine, GL extensions enumerate).
It is a mesh-loader bug. Note: offline-XYZ precompute (itodd_pbr_1_gen_xyz.py)
uses the SAME EGLRenderer.load_object -> load_mesh_sixd -> calc_normals path,
so switching XYZ_ONLINE=False alone would crash identically. The correct fix
is to compute per-vertex normals from the FACE indices.

FIX
---
Rewrite `calc_normals` to accept optional `faces`:

  * faces given (indexed mesh): proper area-weighted per-vertex normals
    accumulated over the triangles, then normalized. Vectorized, O(M).
  * faces None (legacy soup callers): old behaviour but out-of-bounds-safe
    (`range(0, N - N % 3, 3)`), so a non-multiple-of-3 length can never
    IndexError again.

Then feed `faces` at the two indexed call sites:
  * load_mesh_sixd   : faces = np.asarray(model["faces"])
  * load_mesh_pyassimp: faces = mesh.faces

Idempotent — guarded, safe to re-run. Run ON the box (any venv python):

  python box_src/gdrnpp/patch_egl_calc_normals.py

After patching, the mesh cache must be cleared once (a crash never wrote a
cache, but be safe): rm -rf /mnt/data/bop/repos/gdrnpp/.cache
(handled by the chain / smoke wrapper, not here).
"""
import os

GDRN = os.environ.get("GDRN", "/mnt/data/bop/repos/gdrnpp")
MESHUTIL = os.path.join(GDRN, "lib/egl_renderer/glutils/meshutil.py")

PATCH_MARKER = "# [T-068] calc_normals: indexed-mesh aware"

NEW_CALC_NORMALS = '''def calc_normals(vertices, faces=None):
    {marker}
    # Correct per-vertex normals for INDEXED meshes (faces given): accumulate
    # area-weighted face normals onto their 3 vertices, then normalize. The old
    # implementation assumed `vertices` was a flat triangle soup (stride-3),
    # which is wrong for trimesh-exported BOP PLYs (unique verts + face list) and
    # caused IndexError / EGL segfaults at load_object (T-068).
    vertices = np.asarray(vertices, dtype=np.float64)
    if faces is not None:
        faces = np.asarray(faces).reshape(-1, 3).astype(np.int64)
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)  # area-weighted (length == 2*area)
        for k in range(3):
            np.add.at(normals, faces[:, k], fn)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        normals = normals / lens
        return normals.astype(vertices.dtype)
    # Legacy fallback: flat triangle soup, but out-of-bounds-safe.
    normals = np.empty_like(vertices)
    N = vertices.shape[0]
    for i in range(0, N - (N % 3), 3):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        v3 = vertices[i + 2]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        normal = np.zeros(3) if norm == 0 else normal / norm
        normals[i] = normal
        normals[i + 1] = normal
        normals[i + 2] = normal
    return normals'''.format(marker=PATCH_MARKER)

OLD_CALC_NORMALS = '''def calc_normals(vertices):
    normals = np.empty_like(vertices)
    N = vertices.shape[0]
    for i in range(0, N - 1, 3):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        v3 = vertices[i + 2]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        normal = np.zeros(3) if norm == 0 else normal / norm
        normals[i] = normal
        normals[i + 1] = normal
        normals[i + 2] = normal
    return normals'''


def patch_calc_normals(s):
    if PATCH_MARKER in s:
        return s, "calc_normals: already patched"
    if OLD_CALC_NORMALS in s:
        return s.replace(OLD_CALC_NORMALS, NEW_CALC_NORMALS, 1), "calc_normals: patched"
    return s, "calc_normals: WARN anchor not found"


def patch_sixd_call(s):
    # load_mesh_sixd: feed faces (model["faces"]) to calc_normals.
    if 'normals = calc_normals(vertices, faces=np.asarray(model["faces"]))' in s:
        return s, "sixd call: already patched"
    old = ('    if recalculate_normals or "normals" not in model:\n'
           '        normals = calc_normals(vertices)\n'
           '    else:\n'
           '        normals = np.array(model["normals"]).astype(np.float32)')
    new = ('    if recalculate_normals or "normals" not in model:\n'
           '        normals = calc_normals(vertices, faces=np.asarray(model["faces"]))\n'
           '    else:\n'
           '        normals = np.array(model["normals"]).astype(np.float32)')
    if old in s:
        return s.replace(old, new, 1), "sixd call: patched"
    return s, "sixd call: WARN anchor not found"


SIXD_MATERIAL_OLD = '''    result = load_mesh_pyassimp(
        model_path,
        recalculate_normals=recalculate_normals,
        vertex_scale=vertex_scale,
        is_textured=False,
        use_cache=False,
        verbose=False,
    )
    attributes.update(
        uMatDiffuse=result["uMatDiffuse"],
        uMatSpecular=result["uMatSpecular"],
        uMatAmbient=result["uMatAmbient"],
        uMatShininess=result["uMatShininess"],
    )
    mmcv.dump(attributes, cache_file)
    return attributes'''

SIXD_MATERIAL_NEW = '''    # [T-068] pyassimp material re-load REMOVED — it segfaults (pyassimp 4.1.4
    # wrapper vs system libassimp 5.3.1 ABI mismatch: core.py:134 _init reads C
    # structs with stale offsets -> SIGSEGV on these trimesh PLYs). Geometry is
    # already loaded above via inout.load_ply; pyassimp was only fetching
    # material properties, which our untextured CAD PLYs do not contain anyway.
    # Substitute the exact defaults load_mesh_pyassimp returns for a
    # material-less mesh (diffuse 0.8, specular 0.5, ambient 0, shininess 1).
    attributes.update(
        uMatDiffuse=[0.8, 0.8, 0.8],
        uMatSpecular=[0.5, 0.5, 0.5],
        uMatAmbient=[0.0, 0.0, 0.0],
        uMatShininess=1,
    )
    mmcv.dump(attributes, cache_file)
    return attributes'''


def patch_sixd_material(s):
    # load_mesh_sixd: drop the segfaulting pyassimp material re-load.
    if "[T-068] pyassimp material re-load REMOVED" in s:
        return s, "sixd material: already patched"
    if SIXD_MATERIAL_OLD in s:
        return s.replace(SIXD_MATERIAL_OLD, SIXD_MATERIAL_NEW, 1), "sixd material: patched"
    return s, "sixd material: WARN anchor not found"


def patch_pyassimp_call(s):
    # load_mesh_pyassimp: feed mesh.faces to both calc_normals calls.
    old = ('    vertices = mesh.vertices * vertex_scale\n'
           '    if recalculate_normals:\n'
           '        normals = calc_normals(vertices)\n'
           '    else:\n'
           '        normals = mesh.normals\n'
           '    if sum(normals.shape) == 0:\n'
           '        normals = calc_normals(vertices)')
    new = ('    vertices = mesh.vertices * vertex_scale\n'
           '    if recalculate_normals:\n'
           '        normals = calc_normals(vertices, faces=mesh.faces)\n'
           '    else:\n'
           '        normals = mesh.normals\n'
           '    if sum(normals.shape) == 0:\n'
           '        normals = calc_normals(vertices, faces=mesh.faces)')
    if 'normals = calc_normals(vertices, faces=mesh.faces)' in s:
        return s, "pyassimp call: already patched"
    if old in s:
        return s.replace(old, new, 1), "pyassimp call: patched"
    return s, "pyassimp call: WARN anchor not found"


if __name__ == "__main__":
    s = open(MESHUTIL).read()
    results = []
    for fn in (patch_calc_normals, patch_sixd_call, patch_sixd_material, patch_pyassimp_call):
        s, msg = fn(s)
        results.append(msg)
    open(MESHUTIL, "w").write(s)
    for r in results:
        print("  ", r)
    print("PATCH_EGL_CALC_NORMALS_DONE")
