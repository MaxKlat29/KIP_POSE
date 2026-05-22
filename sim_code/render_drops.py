#!/usr/bin/env python3
"""Generalised drop-and-render — for ANY part.

Drops one part with physics (random full-SO(3) orientation, from a height, real
gravity + ground collider), lets it settle, then renders the actual top-down
camera image *right after the drop*. Repeats N times and writes the frames plus
a montage — to see what the real settled data looks like, not a reconstruction.

    /mnt/data/isaacsim-venv/bin/python -u sim_code/render_drops.py \\
        --part /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/Zahnrad_Typ7.usdz \\
        --out  /mnt/data/kip_pose/data/output/drops_Zahnrad --num 16
"""
import argparse
import os
import time


def log(m):
    print(f"[drops {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--scale", type=float, default=0.001, help="USD mm -> m")
    p.add_argument("--cam-h", type=float, default=0.22, help="camera height above origin (m)")
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--drop-h", type=float, default=0.16, help="drop height (m)")
    p.add_argument("--settle", type=int, default=160, help="physics steps")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    log("booting SimulationApp ...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.no_headless})
    log("SimulationApp ready")

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline
    import carb
    from pxr import UsdGeom, UsdLux, UsdPhysics, Gf
    from PIL import Image

    _s = carb.settings.get_settings()
    _s.set("/app/asyncRendering", False)
    _s.set("/omni/replicator/asyncRendering", False)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── scene: Z-up metres, ground (visual + collider), light, top-down cam ──
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 1.0
    ground.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)

    # invisible solid collider so the part lands cleanly on z=0
    coll = UsdGeom.Cube.Define(stage, "/World/_Ground")
    coll.GetSizeAttr().Set(1.0)
    cx = UsdGeom.Xformable(coll); cx.ClearXformOpOrder()
    cx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    cx.AddScaleOp().Set(Gf.Vec3f(8.0, 8.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
    UsdGeom.Imageable(coll.GetPrim()).MakeInvisible()

    ps = UsdPhysics.Scene.Define(stage, "/World/_Phys")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

    cam_path = "/World/TopCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, args.cam_h))     # looks down -Z
    cam.CreateFocalLengthAttr(args.focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))

    for _ in range(20):
        app.update()

    render_product = rep.create.render_product(cam_path, (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])

    timeline = omni.timeline.get_timeline_interface()
    frames = []
    for i in range(args.num):
        for prim in list(stage.Traverse()):
            if prim.GetPath().pathString.startswith("/World/Obj_"):
                stage.RemovePrim(prim.GetPath())
        # spawn one part: small xy jitter, random full orientation, drop height
        x = float(rng.uniform(-0.02, 0.02)); y = float(rng.uniform(-0.02, 0.02))
        rx, ry, rz = (float(rng.uniform(0, 360)) for _ in range(3))
        prim = stage.DefinePrim("/World/Obj_00", "Xform")
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, args.drop_h))
        xf.AddOrientOp().Set(dg.euler_to_quaternion(rx, ry, rz))
        xf.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))
        prim.GetReferences().AddReference(args.part)
        dg.apply_rigid_body(prim)
        dg.apply_collision(prim, stage)

        for _ in range(10):
            app.update()
        timeline.play()
        for _ in range(args.settle):
            app.update()
        timeline.pause()
        for _ in range(5):
            app.update()

        rep.orchestrator.step(rt_subframes=16)
        arr = rgb.get_data()
        img = Image.fromarray(arr).convert("RGB")
        fp = os.path.join(args.out, f"drop_{i:02d}.png")
        img.save(fp)
        frames.append(img)
        log(f"{i + 1}/{args.num} rendered -> {fp}")

    timeline.stop()

    # ── 4-col montage ──
    if frames:
        cols = 4
        rows = (len(frames) + cols - 1) // cols
        w, h = frames[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for idx, im in enumerate(frames):
            r, c = divmod(idx, cols)
            sheet.paste(im, (c * w, r * h))
        mp = os.path.join(args.out, "montage.png")
        sheet.save(mp)
        log(f"montage -> {mp}")

    log(f"done -> {args.out}")
    app.close()


if __name__ == "__main__":
    main()
