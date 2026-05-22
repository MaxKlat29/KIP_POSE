#!/usr/bin/env python3
"""Generalised face-view dataset renderer — for ANY part.

Drops one part with physics, then renders the top-down camera view (RGB + depth
+ part mask) right after settling, crops to the part's bounding box, and records
the settled 6D rotation. Repeats N times. The crops are exactly the inference
snippets; clustering them (faces/cluster_views.py) yields the distinct top-views
(= CNN classes) AND the labelled training crops, plus the pose per crop for
registration. No per-part geometry tuning — the face identity is the rendered
appearance itself.

    /mnt/data/isaacsim-venv/bin/python -u sim_code/render_dataset.py \\
        --part .../Zahnrad_Typ7.usdz --out .../faceset_Zahnrad --num 160
"""
import argparse
import json
import os
import time

import numpy as np


def log(m):
    print(f"[ds {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=160)
    p.add_argument("--scale", type=float, default=0.001)
    p.add_argument("--cam-h", type=float, default=0.16)
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--drop-h", type=float, default=0.16)
    p.add_argument("--settle", type=int, default=160)
    p.add_argument("--res", type=int, default=900)
    p.add_argument("--pad", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.makedirs(args.out, exist_ok=True)

    log(f"booting SimulationApp ... (N={args.num})")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.no_headless})

    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline
    import carb
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf
    from PIL import Image

    _s = carb.settings.get_settings()
    _s.set("/app/asyncRendering", False)
    _s.set("/omni/replicator/asyncRendering", False)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 1.0
    ground.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)
    # ground gets NO semantics -> only the part is segmented / bboxed

    coll = UsdGeom.Cube.Define(stage, "/World/_Ground")
    coll.GetSizeAttr().Set(1.0)
    cx = UsdGeom.Xformable(coll); cx.ClearXformOpOrder()
    cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5)); cx.AddScaleOp().Set(Gf.Vec3f(8, 8, 1))
    UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
    UsdGeom.Imageable(coll.GetPrim()).MakeInvisible()

    ps = UsdPhysics.Scene.Define(stage, "/World/_Phys")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); ps.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

    cam = UsdGeom.Camera.Define(stage, "/World/TopCam")
    cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, args.cam_h))
    cam.CreateFocalLengthAttr(args.focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))
    for _ in range(20):
        app.update()

    rp = rep.create.render_product("/World/TopCam", (args.res, args.res))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    a_sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    for a in (a_rgb, a_dep, a_sem):
        a.attach([rp])

    timeline = omni.timeline.get_timeline_interface()
    rng = np.random.default_rng(args.seed)
    down = np.array([0.0, 0.0, -1.0])
    manifest = []

    for i in range(args.num):
        for prim in list(stage.Traverse()):
            if prim.GetPath().pathString.startswith("/World/Obj"):
                stage.RemovePrim(prim.GetPath())
        prim = stage.DefinePrim("/World/Obj", "Xform")
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        x = float(rng.uniform(-0.015, 0.015)); y = float(rng.uniform(-0.015, 0.015))
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, args.drop_h))
        rx, ry, rz = (float(rng.uniform(0, 360)) for _ in range(3))
        xf.AddOrientOp().Set(dg.euler_to_quaternion(rx, ry, rz))
        xf.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))
        prim.GetReferences().AddReference(args.part)
        dg.add_update_semantics(prim, "part")
        dg.apply_rigid_body(prim); dg.apply_collision(prim, stage)

        for _ in range(10):
            app.update()
        timeline.play()
        for _ in range(args.settle):
            app.update()
        timeline.pause()
        for _ in range(5):
            app.update()
        rep.orchestrator.step(rt_subframes=16)

        rgb = np.asarray(a_rgb.get_data())[:, :, :3]
        dep = np.asarray(a_dep.get_data())
        sem = a_sem.get_data()
        sdata = np.asarray(sem["data"]) if isinstance(sem, dict) else np.asarray(sem)
        if sdata.ndim > 2:
            sdata = sdata[..., 0]
        mask = sdata != 0                                  # part vs unlabelled ground
        if mask.sum() < 200:
            log(f"{i}: empty mask, skip"); continue
        ys, xs = np.nonzero(mask)
        H, W = mask.shape
        pw, ph = xs.max() - xs.min(), ys.max() - ys.min()
        px, py = int(pw * args.pad) + 2, int(ph * args.pad) + 2
        x0 = max(0, xs.min() - px); x1 = min(W, xs.max() + px)
        y0 = max(0, ys.min() - py); y1 = min(H, ys.max() + py)

        # settled rotation (column convention world = R @ body)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=float).reshape(4, 4)
        rows = M[:3, :3]; rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        R = rows.T
        g_body = (R.T @ down)

        rgb_c = rgb[y0:y1, x0:x1]
        dep_c = dep[y0:y1, x0:x1].astype(np.float32)
        msk_c = mask[y0:y1, x0:x1]
        Image.fromarray(rgb_c).save(os.path.join(args.out, f"rgb_{i:04d}.png"))
        np.save(os.path.join(args.out, f"dep_{i:04d}.npy"), dep_c)
        np.save(os.path.join(args.out, f"msk_{i:04d}.npy"), msk_c)
        manifest.append({"i": i, "rgb": f"rgb_{i:04d}.png", "dep": f"dep_{i:04d}.npy",
                         "msk": f"msk_{i:04d}.npy",
                         "R": R.flatten().round(6).tolist(),
                         "g_body": g_body.round(6).tolist()})
        if i % 20 == 0:
            log(f"{i}/{args.num}")

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    log(f"done -> {args.out} ({len(manifest)} crops)")
    app.close()


if __name__ == "__main__":
    main()
