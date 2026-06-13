"""GT-free quality signal: does the SCFlow2-refined pose fit the OBSERVED mask/depth
better than the FoundationPose init pose? Projects mesh verts at each pose and measures:
  - mask_inlier%: fraction of projected verts landing inside the object mask (2D fit)
  - depth_MAE(mm): median |proj_z - sensor_depth| over verts on mask+valid-depth (3D fit)
Caveat: depth fit is partly what the refiner optimizes (not fully independent); mask IoU is
a more independent 2D check. This confirms direction, not absolute accuracy (needs GT)."""
import json, numpy as np, cv2, trimesh

ROOT = 'data/kip'
mesh = trimesh.load(f'{ROOT}/models/obj_000001.ply')      # mm
V = np.asarray(mesh.vertices, np.float64)
rng = np.random.default_rng(0)
V = V[rng.choice(len(V), min(30000, len(V)), replace=False)]

cam = json.load(open(f'{ROOT}/test/000000/scene_camera.json'))
init = json.load(open(f'{ROOT}/init/000000/scene_gt.json'))
ref = json.load(open('results/kip_refined/000000/scene_gt.json'))

def metrics(R, t, K, mask, depth):
    Vc = V @ R.T + t                                       # cam mm
    z = Vc[:, 2]
    u = (K[0, 0] * Vc[:, 0] / z + K[0, 2])
    v = (K[1, 1] * Vc[:, 1] / z + K[1, 2])
    H, W = mask.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inb = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    in_mask = mask[vi[inb], ui[inb]] > 0
    mask_frac = 100.0 * in_mask.mean()
    sensor = depth[vi[inb], ui[inb]].astype(np.float64)
    pick = in_mask & (sensor > 0)
    depth_mae = float(np.median(np.abs(z[inb][pick] - sensor[pick]))) if pick.sum() > 20 else float('nan')
    return mask_frac, depth_mae

print('frame | mask_inlier%% init->ref | depth_MAE(mm) init->ref')
print('-' * 64)
win_mask = win_depth = 0
for k in sorted(init, key=lambda x: int(x)):
    K = np.array(cam[k]['cam_K'], np.float64).reshape(3, 3)
    mask = cv2.imread(f'{ROOT}/test/000000/mask_visib/{int(k):06d}_000000.png', cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(f'{ROOT}/test/000000/depth/{int(k):06d}.png', cv2.IMREAD_UNCHANGED).astype(np.float64)
    res = {}
    for name, src in (('init', init), ('ref', ref)):
        R = np.array(src[k][0]['cam_R_m2c']).reshape(3, 3)
        t = np.array(src[k][0]['cam_t_m2c'])
        res[name] = metrics(R, t, K, mask, depth)
    mi, mr = res['init'][0], res['ref'][0]
    di, dr = res['init'][1], res['ref'][1]
    win_mask += mr > mi
    win_depth += dr < di
    print(f'  {k:>3} | {mi:6.1f} -> {mr:6.1f}  ({mr-mi:+5.1f}) | {di:7.1f} -> {dr:7.1f}  ({dr-di:+6.1f})')
print('-' * 64)
print(f'refined better: mask {win_mask}/5 frames,  depth {win_depth}/5 frames')
