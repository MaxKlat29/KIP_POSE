"""Diagnose the degradation: (1) mesh centering, (2) per-frame mask validity,
(3) decompose init->refined rotation into body-Y(symmetry-axis) vs perpendicular."""
import json, numpy as np, cv2, trimesh

m = trimesh.load('data/kip/models/obj_000001.ply')
lo, hi = m.bounds
print('mesh bounds (mm): lo', lo.round(1), 'hi', hi.round(1), '-> center', ((lo+hi)/2).round(2),
      '(AABB-centered?', np.allclose((lo+hi)/2, 0, atol=1.0), ')')
print('long axis = argmax extent:', 'XYZ'[np.argmax(hi-lo)], '(symmetry axis should be body-Y)')

init = json.load(open('data/kip/init/000000/scene_gt.json'))
ref = json.load(open('results/kip_refined/000000/scene_gt.json'))

def axis_angle(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    if ang < 1e-3:
        return np.array([0, 1., 0]), 0.0
    ax = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    return ax/ (np.linalg.norm(ax)+1e-9), ang

print('\nframe | maskpx | dR_total | rot-axis(body) | |axis.Y| (1=pure symmetry-axis rot)')
print('-'*78)
for k in sorted(init, key=lambda x: int(x)):
    mask = cv2.imread(f'data/kip/test/000000/mask_visib/{int(k):06d}_000000.png', cv2.IMREAD_GRAYSCALE)
    npx = int((mask > 0).sum())
    Ri = np.array(init[k][0]['cam_R_m2c']).reshape(3, 3)
    Rr = np.array(ref[k][0]['cam_R_m2c']).reshape(3, 3)
    R_rel_body = Ri.T @ Rr                       # relative rotation in BODY frame
    ax, ang = axis_angle(R_rel_body)
    dotY = abs(float(ax[1]))
    print(f'  {k:>3} | {npx:6d} | {ang:7.2f}  | [{ax[0]:+.2f} {ax[1]:+.2f} {ax[2]:+.2f}] | {dotY:.2f}')
