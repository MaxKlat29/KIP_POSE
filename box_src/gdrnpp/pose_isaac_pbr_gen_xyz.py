"""Offline XYZ precompute for pose_isaac (BOP) — kills per-iteration online EGL render.

GDRNPP's online XYZ path (engine_utils.batch_data_train_online) renders the dense
XYZ coordinate *target* with EGLRenderer once **per instance per iteration** in the
main training loop (bs sequential single-instance renders, each with a GPU<->CPU
sync). At INPUT_RES=320/OUTPUT_RES=80 this is the dominant per-iter cost
(~2.8 s/iter vs ~0.6 s/iter at 256/64). Over TOTAL_EPOCHS=160 every instance is
re-rendered ~160x.

This script renders every GT instance **once** at full image resolution and saves
`<xyz_root>/<scene>/<im>_<anno>-xyz.pkl` ({xyz_crop: float16, xyxy}) — the exact
path + format that pose_isaac_pbr.py already populates as `xyz_path` and that the
non-online loader (data_loader.GDRN_DatasetFromList) consumes. Flip
XYZ_ONLINE=False and the training loop does ZERO EGL rendering.

Pose / scale conventions match the online renderer (engine_utils.get_renderer)
and the loader (pose_isaac_pbr.py) EXACTLY:
  R = cam_R_m2c ; t = cam_t_m2c / 1000 (meters) ; pose = [R|t]
  vertex_scale = 0.001 (PLY mm -> m) ; per-frame K from scene_camera.json
  obj render index = position of obj_id within the sorted model_paths list.

S-051 EGL fixes (calc_normals per-vertex from face indices, pyassimp material
reload removed) are already applied in lib/egl_renderer — the renderer loads
headless cleanly.

Usage:
  python pose_isaac_pbr_gen_xyz.py --split train_pbr [--scenes 0,1,2] [--limit N]
                                   [--gpu 0] [--time-only] [--vis-check]
"""
import argparse
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import time

import mmcv
import numpy as np

cur_dir = osp.abspath(osp.dirname(__file__))
# Resolve GDRNPP repo root: the deployed script lives in <repo>/core/... or is
# called from box_src; default to the canonical box path, override via --repo.
DEFAULT_REPO = "/mnt/data/bop/repos/gdrnpp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GDRNPP repo root")
    ap.add_argument("--dataset-root", default=None,
                    help="pose_isaac BOP root (default: <repo>/datasets/BOP_DATASETS/pose_isaac)")
    ap.add_argument("--split", default="train_pbr", help="train_pbr | val")
    ap.add_argument("--scenes", default=None, help="comma list of scene ints; default all")
    ap.add_argument("--limit", type=int, default=0, help="stop after N instances (0=all)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--time-only", action="store_true",
                    help="render but do NOT write pkls; report throughput")
    ap.add_argument("--vis-check", action="store_true",
                    help="print xyz stats for the first few instances")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
    from lib.pysixd import misc

    droot = args.dataset_root or osp.join(args.repo, "datasets/BOP_DATASETS/pose_isaac")
    droot = osp.normpath(droot)
    split_root = osp.join(droot, args.split)
    model_dir = osp.join(droot, "models")
    # xyz_root MUST match pose_isaac_pbr.py:_cfg_for() exactly:
    #   xyz_root = <dataset_root>/<split_dir>/xyz_crop   (split_dir = train_pbr | val)
    # NOT a shared top-level dir — train_pbr and val reuse scene-IDs 000000.. and
    # would collide. Per-split xyz_crop keeps them separate.
    xyz_root = osp.join(split_root, "xyz_crop")
    assert osp.isdir(split_root), split_root

    # obj_id -> render index (position within sorted model_paths), matching
    # EGLRenderer's internal object indexing. We register ALL 6 models so any
    # obj_id maps correctly; SO-training filters at the loader, not here.
    all_ids = sorted(int(f.split("_")[1].split(".")[0])
                     for f in os.listdir(model_dir) if f.startswith("obj_") and f.endswith(".ply"))
    model_paths = [osp.join(model_dir, f"obj_{i:06d}.ply") for i in all_ids]
    id2renderidx = {oid: i for i, oid in enumerate(all_ids)}
    print(f"[gen_xyz] models: {all_ids}  vertex_scale=0.001  full-res target")

    IM_W, IM_H = 1280, 720
    near, far = 0.01, 6.5
    device = f"cuda:{args.gpu}"

    ren = EGLRenderer(
        model_paths,
        texture_paths=None,
        vertex_scale=0.001,
        height=IM_H,
        width=IM_W,
        znear=near,
        zfar=far,
        use_cache=True,
        gpu_id=args.gpu,
    )
    seg_tensor = torch.cuda.FloatTensor(IM_H, IM_W, 4, device=device).detach()
    pc_cam_tensor = torch.cuda.FloatTensor(IM_H, IM_W, 4, device=device).detach()

    if args.scenes:
        want = set(int(x) for x in args.scenes.split(","))
        scenes = [s for s in sorted(os.listdir(split_root))
                  if s.isdigit() and int(s) in want]
    else:
        scenes = [s for s in sorted(os.listdir(split_root)) if s.isdigit()]

    n_done = n_empty = n_skip = 0
    t0 = time.time()
    render_secs = 0.0
    for scene in scenes:
        scene_root = osp.join(split_root, scene)
        gt = mmcv.load(osp.join(scene_root, "scene_gt.json"))
        cam = mmcv.load(osp.join(scene_root, "scene_camera.json"))
        for str_im_id, annos in gt.items():
            int_im_id = int(str_im_id)
            K = np.array(cam[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            for anno_i, anno in enumerate(annos):
                if args.limit and (n_done + n_empty) >= args.limit:
                    _finish(t0, render_secs, n_done, n_empty, n_skip, args)
                    return
                obj_id = anno["obj_id"]
                save_path = osp.join(xyz_root, scene, f"{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                if (not args.overwrite) and osp.exists(save_path) and osp.getsize(save_path) > 0:
                    n_skip += 1
                    continue
                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                pose = np.hstack([R, t.reshape(3, 1)])
                R_th = torch.tensor(R, dtype=torch.float32, device=device)
                t_th = torch.tensor(t, dtype=torch.float32, device=device)
                K_th = torch.tensor(K, dtype=torch.float32, device=device)

                tr = time.time()
                ren.render([id2renderidx[obj_id]], [pose], K=K,
                           seg_tensor=seg_tensor, pc_cam_tensor=pc_cam_tensor)
                torch.cuda.synchronize()
                render_secs += time.time() - tr

                mask = (seg_tensor[:, :, 0] > 0).to(torch.uint8)
                if int(mask.sum()) == 0:
                    xyz_info = {"xyz_crop": np.zeros((IM_H, IM_W, 3), dtype=np.float16),
                                "xyxy": [0, 0, IM_W - 1, IM_H - 1]}
                    n_empty += 1
                else:
                    ys_xs = mask.nonzero(as_tuple=False)
                    ys, xs = ys_xs[:, 0], ys_xs[:, 1]
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    depth_th = pc_cam_tensor[:, :, 2].detach()
                    xyz_th = misc.calc_xyz_bp_torch(depth_th, R_th, t_th, K_th)
                    xyz_crop = xyz_th[y1:y2 + 1, x1:x2 + 1].cpu().numpy()
                    xyz_info = {"xyz_crop": xyz_crop.astype("float16"), "xyxy": [x1, y1, x2, y2]}
                    n_done += 1
                    if args.vis_check and (n_done <= 3):
                        nz = xyz_crop[(xyz_crop != 0).any(-1)]
                        print(f"[vis] obj {obj_id} scene {scene} im {int_im_id} anno {anno_i} "
                              f"crop {xyz_crop.shape} nz_min {nz.min():.4f} nz_max {nz.max():.4f} "
                              f"xyxy {[x1, y1, x2, y2]}")

                if not args.time_only:
                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.dump(xyz_info, save_path)

    _finish(t0, render_secs, n_done, n_empty, n_skip, args)


def _finish(t0, render_secs, n_done, n_empty, n_skip, args):
    wall = time.time() - t0
    n_rendered = n_done + n_empty
    print("=" * 60)
    print(f"[gen_xyz] DONE split={args.split} time_only={args.time_only}")
    print(f"  rendered instances : {n_rendered}  (visible {n_done}, empty {n_empty})")
    print(f"  skipped (existing) : {n_skip}")
    print(f"  wall               : {wall:.1f}s")
    print(f"  pure render        : {render_secs:.1f}s")
    if n_rendered:
        print(f"  render throughput  : {n_rendered / max(render_secs, 1e-6):.1f} inst/s "
              f"({1000 * render_secs / n_rendered:.2f} ms/inst)")
        print(f"  wall throughput    : {n_rendered / max(wall, 1e-6):.1f} inst/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
