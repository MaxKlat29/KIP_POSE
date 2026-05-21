#!/usr/bin/env python3
"""Empirical stable-face discovery — drop ONE part N times in Isaac Sim.

Spawns the part at random full-SO(3) orientation onto a flat ground collider,
lets physics settle it, then reads back the settled 6D pose. No rendering — this
is the cheap "where does it actually come to rest" pass. Writes one JSONL row
per settle ``{R (9, column-convention world=R·body), t, g_body}`` consumed by
``faces/atlas_from_drops.py`` for clustering + the empirical atlas collage.

    /mnt/data/isaacsim-venv/bin/python -u sim_code/discover_faces.py \
        --part   /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/Anker_Lang.usd \
        --out    /mnt/data/kip_pose/faces_out/drops_Anker_Lang.jsonl \
        --num 1000 --per-scene 50
"""
import argparse
import json
import os
import time


def log(m):
    print(f"[discover {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--per-scene", type=int, default=50)
    p.add_argument("--settle", type=int, default=160, help="physics steps per batch")
    p.add_argument("--drop-h", type=float, default=0.18, help="drop height (m)")
    p.add_argument("--spacing", type=float, default=0.22, help="spawn grid spacing (m)")
    p.add_argument("--scale", type=float, default=0.001, help="USD->m scale (Marc's OBJECT_SCALE)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    log("booting SimulationApp ...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    log("SimulationApp ready")

    import numpy as np
    import carb
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdGeom, UsdPhysics, Gf

    s = carb.settings.get_settings()
    s.set("/app/asyncRendering", False)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(20):
        app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # physics + invisible ground collider with its top at z=0
    ps = UsdPhysics.Scene.Define(stage, "/World/_Phys")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    gp = UsdGeom.Cube.Define(stage, "/World/_Ground")
    gp.GetSizeAttr().Set(1.0)
    gx = UsdGeom.Xformable(gp)
    gx.ClearXformOpOrder()
    gx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    gx.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
    UsdGeom.Imageable(gp.GetPrim()).MakeInvisible()
    for _ in range(5):
        app.update()

    rng = np.random.default_rng(args.seed)
    timeline = omni.timeline.get_timeline_interface()
    grid = max(1, int(np.ceil(np.sqrt(args.per_scene))))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w")
    written = 0
    down = np.array([0.0, 0.0, -1.0])

    while written < args.num:
        m = min(args.per_scene, args.num - written)
        paths = []
        for i in range(m):
            x = (i % grid) * args.spacing
            y = (i // grid) * args.spacing
            pth = f"/World/Obj_{i:03d}"
            prim = stage.DefinePrim(pth, "Xform")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, args.drop_h))
            rx, ry, rz = (float(rng.uniform(0, 360)) for _ in range(3))
            xf.AddOrientOp().Set(dg.euler_to_quaternion(rx, ry, rz))
            xf.AddScaleOp().Set(Gf.Vec3f(args.scale, args.scale, args.scale))
            prim.GetReferences().AddReference(args.part)
            dg.apply_rigid_body(prim)
            dg.apply_collision(prim, stage)
            paths.append(pth)
        for _ in range(10):
            app.update()

        timeline.play()
        for _ in range(args.settle):
            app.update()
        timeline.pause()
        for _ in range(5):
            app.update()

        for pth in paths:
            prim = stage.GetPrimAtPath(pth)
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), dtype=float).reshape(4, 4)
            rows = M[:3, :3]                              # world images of body axes (scaled)
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            R = rows.T                                    # column convention: world = R @ body
            g_body = (R.T @ down)
            t = M[3, :3]
            fout.write(json.dumps({"R": R.flatten().round(6).tolist(),
                                   "t": t.round(6).tolist(),
                                   "g_body": g_body.round(6).tolist()}) + "\n")
            written += 1
        fout.flush()
        for pth in paths:
            stage.RemovePrim(pth)
        for _ in range(5):
            app.update()
        log(f"{written}/{args.num} settled")

    fout.close()
    log(f"done -> {args.out} ({written} drops)")
    app.close()


if __name__ == "__main__":
    main()
