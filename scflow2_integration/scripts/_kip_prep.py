"""Convert our FoundationPose-format frames into a minimal SCFlow2 RefineTestDataset
layout (NO GT needed — forward-pass de-risk). Units: depth PNG already mm (depth_scale=1.0,
verified), FP poses metres->mm (x1000), meshes metres->mm (x1000)."""
import os, json, shutil, glob
import numpy as np
import cv2
import trimesh

REPO = '/home/age/Desktop/Gruppe3/KIP_POSE'
OUT = os.path.join(REPO, 'scflow2/data/kip')
# FoundationPose init poses (fp_output) only exist for Anker_Kurz, so that's the only
# scene we can refine in the forward-pass de-risk. Meshes for both ids are still emitted.
PARTS = [('Anker_Kurz', 1)]                              # obj_id
MESH_SRC = {1: f'{REPO}/data/meshes/Anker_Kurz.obj', 2: f'{REPO}/data/meshes/Anker_Lang.obj'}

def aabb_corners(mesh):
    lo, hi = mesh.bounds
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)

# ---- meshes (mm) + keypoints + diameters ----
os.makedirs(f'{OUT}/models', exist_ok=True)
keypoints, diameters, models_info = [], [], {}
for obj_id in (1, 2):
    m = trimesh.load(MESH_SRC[obj_id], force='mesh')
    m.apply_scale(1000.0)                                # m -> mm
    # pytorch3d renderer (Phong) needs vertex textures; CAD has none -> flat gray.
    m.visual.vertex_colors = np.tile(np.array([180, 180, 180, 255], np.uint8), (len(m.vertices), 1))
    m.export(f'{OUT}/models/obj_{obj_id:06d}.ply')
    keypoints.append(aabb_corners(m).tolist())           # (8,3) mm
    v = m.vertices
    lo, hi = v.min(0), v.max(0)
    diam = float(np.linalg.norm(hi - lo))                # bbox-diagonal approx (mm)
    diameters.append(round(diam, 3))
    # Anker parts: continuous rotational symmetry about body-Y (per CLAUDE.md)
    models_info[str(obj_id)] = {
        'diameter': diam,
        'min_x': float(lo[0]), 'min_y': float(lo[1]), 'min_z': float(lo[2]),
        'size_x': float(hi[0]-lo[0]), 'size_y': float(hi[1]-lo[1]), 'size_z': float(hi[2]-lo[2]),
        'symmetries_continuous': [{'axis': [0, 1, 0], 'offset': [0, 0, 0]}],
    }
    print(f'obj {obj_id}: verts={len(v)} diameter~{diam:.1f}mm bbox(mm)={(hi-lo).round(1)}')
json.dump(keypoints, open(f'{OUT}/keypoints.json', 'w'))
json.dump(models_info, open(f'{OUT}/models/models_info.json', 'w'))
print('diameters(mm) =', diameters)

# ---- per-part scenes (one scene per part: Anker_Kurz=000000, Anker_Lang=000001) ----
image_list = []
for scene_idx, (part, obj_id) in enumerate(PARTS):
    seq = f'{scene_idx:06d}'
    frames = sorted(glob.glob(f'{REPO}/data/fp_input/{part}/frame_*'))
    test_dir = f'{OUT}/test/{seq}'
    for sub in ('rgb', 'depth', 'mask_visib'):
        os.makedirs(f'{test_dir}/{sub}', exist_ok=True)
    os.makedirs(f'{OUT}/init/{seq}', exist_ok=True)
    scene_camera, init_gt = {}, {}
    for i, fr in enumerate(frames):
        # rgb (drop alpha -> 3ch BGR as-is)
        rgb = cv2.imread(f'{fr}/rgb/0.png', cv2.IMREAD_UNCHANGED)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        cv2.imwrite(f'{test_dir}/rgb/{i:06d}.png', rgb)
        shutil.copy(f'{fr}/depth/0.png', f'{test_dir}/depth/{i:06d}.png')           # uint16 mm
        shutil.copy(f'{fr}/masks/0.png', f'{test_dir}/mask_visib/{i:06d}_000000.png')
        K = np.loadtxt(f'{fr}/cam_K.txt', dtype=np.float64).reshape(3, 3)
        scene_camera[str(i)] = {'cam_K': K.flatten().tolist(), 'depth_scale': 1.0}
        T = np.loadtxt(f'{REPO}/data/fp_output/{part}/{os.path.basename(fr)}/ob_in_cam/0.txt', dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3] * 1000.0                                          # m -> mm
        init_gt[str(i)] = [{'cam_R_m2c': R.flatten().tolist(), 'cam_t_m2c': t.tolist(),
                            'obj_id': obj_id, 'score': 1.0}]
        image_list.append(f'{seq}/rgb/{i:06d}.png')
    json.dump(scene_camera, open(f'{test_dir}/scene_camera.json', 'w'))
    json.dump(init_gt, open(f'{OUT}/init/{seq}/scene_gt.json', 'w'))
    print(f'{part}: scene {seq}, {len(frames)} frames')

open(f'{OUT}/image_list.txt', 'w').write('\n'.join(image_list) + '\n')
print('image_list:', len(image_list), 'entries ->', f'{OUT}/image_list.txt')
