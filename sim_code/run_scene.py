#!/usr/bin/env python3
"""
Render an assembled scene (table/cart + LARA5 robot + tray) from the real Zivid
camera — the actual pick-and-place situation, top-down.

Optionally spawns the Anker parts into the tray (tray / random mode, reusing the
helpers in datagenerationscript.py) and writes RGB + 2D bbox + semantic +
instance + depth.

    /mnt/data/isaacsim-venv/bin/python -u sim_code/run_scene.py \\
        --scene  /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd \\
        --usd-dir /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files \\
        --output /mnt/data/kip_pose/data/output/scene \\
        --camera /World/Zivid --spawn tray --num-objects 6
"""
import argparse
import json
import math
import os
import time


def compute_oriented_boxes(inst, sem, sem_id2label, part_classes,
                           bbox_data=None, min_pixels=80):
    """Oriented 2D boxes per part instance via PCA on the segmentation masks.

    Aligns each box to the part's principal (long) axis, so a part lying at an
    angle gets a rotated box hugging it — not the axis-aligned enclosing box.
    Uses only the visible instance pixels, so it is occlusion-robust.

    If ``bbox_data`` (the bounding_box_2d_tight result) is given, each oriented
    box is tagged with the occlusionRatio of the nearest same-class tight box.
    """
    import numpy as np
    boxes = []
    if inst is None or sem is None:
        return boxes
    inst = np.asarray(inst)
    sem = np.asarray(sem)
    if inst.ndim > 2:
        inst = inst[..., 0]
    if sem.ndim > 2:
        sem = sem[..., 0]

    # class -> [(cx, cy, occlusion)] from the tight boxes, for occlusion lookup
    tight = {}
    if isinstance(bbox_data, dict):
        b_lab = bbox_data.get("info", {}).get("idToLabels", {})
        for row in bbox_data.get("data", []):
            cls = b_lab.get(str(int(row[0])), {}).get("class", "").lower()
            occ = float(row[5]) if len(row) > 5 else 0.0
            tight.setdefault(cls, []).append(((row[1] + row[3]) / 2.0,
                                              (row[2] + row[4]) / 2.0, occ))
    for iid in np.unique(inst):
        if int(iid) == 0:
            continue
        mask = inst == iid
        n = int(mask.sum())
        if n < min_pixels:
            continue
        sids, counts = np.unique(sem[mask], return_counts=True)
        sid = int(sids[counts.argmax()])
        cls = sem_id2label.get(str(sid), {}).get("class", "")
        if cls.lower() not in part_classes:
            continue
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        c = pts.mean(0)
        d = pts - c
        cov = (d.T @ d) / max(len(d) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(evals.argmax())]
        minor = evecs[:, int(evals.argmin())]
        pa, pi = d @ major, d @ minor
        a0, a1 = float(pa.min()), float(pa.max())
        i0, i1 = float(pi.min()), float(pi.max())
        corners = [c + major * a0 + minor * i0, c + major * a1 + minor * i0,
                   c + major * a1 + minor * i1, c + major * a0 + minor * i1]
        occ = 0.0
        cands = tight.get(cls.lower(), [])
        if cands:
            occ = min(cands, key=lambda t: (t[0] - c[0]) ** 2 + (t[1] - c[1]) ** 2)[2]
        boxes.append({
            "class": cls,
            "corners": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in corners],
            "center": [round(float(c[0]), 1), round(float(c[1]), 1)],
            "angle_deg": round(math.degrees(math.atan2(float(major[1]), float(major[0]))), 1),
            "length_px": round(a1 - a0, 1),
            "width_px": round(i1 - i0, 1),
            "area_px": n,
            "occlusion": round(float(occ), 3),
        })
    return boxes


def log(m):
    print(f"[scene {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--usd-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--camera", default="/World/Zivid")
    p.add_argument("--cam-pos", default=None,
                   help="x,y,z eye of a custom camera (metres); enables it")
    p.add_argument("--cam-look", default=None,
                   help="x,y,z look-at target; if omitted the camera looks straight down")
    p.add_argument("--cam-focal", type=float, default=20.0,
                   help="focal length (mm) of the custom camera")
    p.add_argument("--spawn", choices=["none", "tray", "random"], default="none")
    p.add_argument("--spawn-box", default=None,
                   help="static placement: cx,cy,cz,half — drop parts randomly in this box")
    p.add_argument("--spawn-bounds", default=None,
                   help="restrict random-mode spawn to xmin,xmax,ymin,ymax (inner table)")
    p.add_argument("--physics-z", type=float, default=None,
                   help="enable physics: gravity + an invisible ground collider with top at z")
    p.add_argument("--num-objects", type=int, default=6)
    p.add_argument("--num-renders", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--add-light", action="store_true",
                   help="add a fallback dome light (use if the scene renders dark)")
    p.add_argument("--hide", default=None,
                   help="comma-separated name substrings; matching prims are made invisible "
                        "(e.g. the robot arm so the camera sees the table)")
    p.add_argument("--no-headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["SDG_USD_DIR"] = args.usd_dir
    os.environ["SDG_OUTPUT_DIR"] = args.output
    os.environ["SDG_CAMERA_PATH"] = args.camera

    log("booting SimulationApp ...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": not args.no_headless})
    log("SimulationApp ready")

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline
    import carb
    from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

    _s = carb.settings.get_settings()
    _s.set("/app/asyncRendering", False)
    _s.set("/omni/replicator/asyncRendering", False)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    dg.SPAWN_MODE = args.spawn if args.spawn != "none" else dg.SPAWN_MODE
    dg.NUM_ANKERS_IN_TRAY = args.num_objects
    dg.NUM_OBJECTS = args.num_objects
    if args.spawn_bounds:
        xm, xx, ym, yx = (float(v) for v in args.spawn_bounds.split(","))
        b = dict(dg.SPAWN_BOUNDS)          # keep Marc's high drop-height z range
        b["x"], b["y"] = (xm, xx), (ym, yx)
        dg.SPAWN_BOUNDS = b

    # open the assembled scene
    ctx = omni.usd.get_context()
    log(f"opening scene: {args.scene}")
    _r = ctx.open_stage(args.scene)
    ok = _r[0] if isinstance(_r, (tuple, list)) else _r   # API returns bool or (bool, str)
    if not ok:
        log(f"FAILED to open stage: {args.scene}")
        simulation_app.close(); sys.exit(1)
    for _ in range(80):
        simulation_app.update()
    stage = ctx.get_stage()
    log(f"scene loaded ({len(list(stage.Traverse()))} prims)")

    if args.hide:
        subs = [s.strip().lower() for s in args.hide.split(",") if s.strip()]
        hidden = 0
        for prim in stage.Traverse():
            if any(sub in prim.GetName().lower() for sub in subs):
                img = UsdGeom.Imageable(prim)
                if img:
                    img.MakeInvisible()
                    hidden += 1
        log(f"hid {hidden} prim(s) matching {subs}")
        for _ in range(5):
            simulation_app.update()

    if args.add_light:
        UsdLux.DomeLight.Define(stage, "/World/_FallbackDome").CreateIntensityAttr(500.0)
        for _ in range(5):
            simulation_app.update()

    # custom camera (free eye + optional look-at target)
    if args.cam_pos:
        eye = Gf.Vec3d(*[float(v) for v in args.cam_pos.split(",")])
        tc = UsdGeom.Camera.Define(stage, "/World/_Cam")
        if args.cam_look:
            target = Gf.Vec3d(*[float(v) for v in args.cam_look.split(",")])
            view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))  # Z-up
            tc.AddTransformOp().Set(view.GetInverse())
        else:
            tc.AddTranslateOp().Set(eye)            # identity = straight down (-Z) in Z-up
        tc.CreateFocalLengthAttr(args.cam_focal)
        tc.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        args.camera = "/World/_Cam"
        log(f"custom camera eye={tuple(eye)} look={args.cam_look} focal={args.cam_focal}")
        for _ in range(5):
            simulation_app.update()

    # validate camera
    cam = stage.GetPrimAtPath(args.camera)
    if not cam.IsValid():
        log(f"camera not found: {args.camera}; available:")
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Camera):
                log(f"   {p.GetPath()}")
        simulation_app.close(); sys.exit(1)

    if args.physics_z is not None:
        ps = UsdPhysics.Scene.Define(stage, "/World/_PhysicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        ps.CreateGravityMagnitudeAttr(9.81)
        gp = UsdGeom.Cube.Define(stage, "/World/_PhysGround")
        gp.GetSizeAttr().Set(1.0)
        gx = UsdGeom.Xformable(gp)
        gx.ClearXformOpOrder()
        gx.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.1, args.physics_z - 0.01))
        gx.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))
        UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
        UsdGeom.Imageable(gp.GetPrim()).MakeInvisible()
        log(f"physics enabled: gravity + invisible ground top at z={args.physics_z}")
        for _ in range(5):
            simulation_app.update()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    timeline = omni.timeline.get_timeline_interface()

    render_product = rep.create.render_product(args.camera, (args.width, args.height))
    annots = {
        "rgb":          rep.AnnotatorRegistry.get_annotator("rgb"),
        "bbox_2d":      rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        "semantic_seg": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
        "instance_seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
        "depth":        rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
    }
    for a in annots.values():
        a.attach([render_product])

    SCALE = 0.001
    box = None
    if args.spawn_box:
        bx, by, bz, bhalf = (float(v) for v in args.spawn_box.split(","))
        box = (bx, by, bz, bhalf)

    for ridx in range(args.num_renders):
        log(f"render {ridx + 1}/{args.num_renders} (spawn={'box' if box else args.spawn})")
        if box is not None:
            # static placement of real parts in a calibrated box (no physics)
            for prim in list(stage.Traverse()):
                if prim.GetPath().pathString.startswith("/World/Obj_"):
                    stage.RemovePrim(prim.GetPath())
            bx, by, bz, bhalf = box
            for i in range(args.num_objects):
                asset = dg.ASSETS[int(rng.integers(0, len(dg.ASSETS)))]
                x = float(rng.uniform(bx - bhalf, bx + bhalf))
                y = float(rng.uniform(by - bhalf, by + bhalf))
                yaw = float(rng.uniform(0, 360))
                pth = f"/World/Obj_{i:02d}"
                prim = stage.DefinePrim(pth, "Xform")
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(x, y, bz))
                xf.AddOrientOp().Set(dg.euler_to_quaternion(0, 0, yaw))
                xf.AddScaleOp().Set(Gf.Vec3f(SCALE, SCALE, SCALE))
                prim.GetReferences().AddReference(asset["path"])
                dg.add_update_semantics(prim, asset["label"])
            for _ in range(15):
                simulation_app.update()
        elif args.spawn != "none":
            timeline.stop()
            for _ in range(5):
                simulation_app.update()
            dg.cleanup_spawned_objects(stage)
            for _ in range(5):
                simulation_app.update()
            if args.spawn == "tray":
                dg.spawn_tray_mode(stage, rng)
            else:
                dg.spawn_random_mode(stage, rng)
            for _ in range(10):
                simulation_app.update()
            # let physics settle the dropped parts
            timeline.play()
            for _ in range(dg.PHYSICS_SETTLE_STEPS):
                simulation_app.update()
            timeline.pause()
            for _ in range(5):
                simulation_app.update()
        else:
            for _ in range(15):
                simulation_app.update()

        rep.orchestrator.step(rt_subframes=16)
        data = {k: a.get_data() for k, a in annots.items()}
        dg.save_output(data, ridx, args.output)

        # oriented 2D boxes (PCA on the instance masks) -> obb_2d_<idx>.json
        inst = data["instance_seg"]
        seg = data["semantic_seg"]
        obbs = compute_oriented_boxes(
            inst["data"] if isinstance(inst, dict) else None,
            seg["data"] if isinstance(seg, dict) else None,
            seg["info"].get("idToLabels", {}) if isinstance(seg, dict) else {},
            {"anker_kurz", "anker_lang"},
            bbox_data=data.get("bbox_2d"),
        )
        with open(os.path.join(args.output, f"obb_2d_{ridx:04d}.json"), "w") as f:
            json.dump({"boxes": obbs}, f, indent=2)
        log(f"  obb: {len(obbs)} oriented boxes")

    timeline.stop()
    log(f"done -> {args.output}")
    simulation_app.close()


if __name__ == "__main__":
    main()
