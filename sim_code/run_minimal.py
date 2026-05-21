#!/usr/bin/env python3
"""
Minimal end-to-end SDG run — proves the pipeline against the real part USDs.

Builds a clean, self-contained scene (ground + dome light + top-down camera),
references the real Anker parts (scaled mm -> m), places a handful at random
positions/orientations, and renders one annotated sample (RGB + 2D bbox +
semantic + instance + depth). No tray, no physics, no Marc-scene calibration —
the smallest thing that produces a real labelled image.

Run inside the Isaac Sim venv on the workstation:
    /mnt/data/isaacsim-venv/bin/python -u sim_code/run_minimal.py \\
        --usd-dir /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files \\
        --output  /mnt/data/kip_pose/data/output/minimal \\
        --num-renders 3 --num-objects 6
"""
import argparse
import os
import time


def log(m):
    print(f"[minimal {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--usd-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-renders", type=int, default=3)
    p.add_argument("--num-objects", type=int, default=6)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    log("booting SimulationApp ...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": not args.no_headless})
    log("SimulationApp ready")

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import carb
    from pxr import UsdGeom, UsdLux, Gf, Sdf

    # synchronous rendering (async can deadlock a headless capture)
    _s = carb.settings.get_settings()
    _s.set("/app/asyncRendering", False)
    _s.set("/omni/replicator/asyncRendering", False)

    # reuse the shared helpers (semantics, quaternion, saving)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output, exist_ok=True)

    PARTS = [
        (os.path.join(args.usd_dir, "Anker_Kurz.usd"), "/Anker_kurz", "Anker_Kurz"),
        (os.path.join(args.usd_dir, "Anker_Lang.usd"), "/Anker_lang", "Anker_Lang"),
    ]
    SCALE = 0.001          # parts are in mm -> scale to metres
    REST_Z = 0.013         # rod cross-section ~24mm -> half ~12mm, sit on ground
    AREA = 0.16            # spawn within +/-0.16 m square
    PART_LABELS = [lbl for *_, lbl in PARTS]

    # ── build the scene ───────────────────────────────────────
    log("building scene (Z-up metres: ground + dome light + camera)")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    # ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 2.0
    ground.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)
    dg.add_update_semantics(ground.GetPrim(), "Ground")

    # dome light
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    # top-down camera framing the spawn area
    cam_path = "/World/TopCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.8))   # 0.8 m above origin, looking down -Z
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    for _ in range(20):
        simulation_app.update()

    # ── lighting randomization (branch marc) ──────────────────
    # run_minimal builds its own scene so we override the light position
    # bounds to match the small spawn area (±AREA metres around origin).
    dg.LIGHT_CONFIG = dict(dg.LIGHT_CONFIG)
    dg.LIGHT_CONFIG["pos_min"] = (-AREA, -AREA, 0.4)
    dg.LIGHT_CONFIG["pos_max"] = ( AREA,  AREA, 1.2)
    dg.setup_randomized_lights()

    # ── render loop ───────────────────────────────────────────
    render_product = rep.create.render_product(cam_path, (args.width, args.height))
    annots = {
        "rgb":          rep.AnnotatorRegistry.get_annotator("rgb"),
        "bbox_2d":      rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        "semantic_seg": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
        "instance_seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
        "depth":        rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
        # 6-DoF pose labels (branch marc)
        "bbox_3d":      rep.AnnotatorRegistry.get_annotator("bounding_box_3d"),
        "obj_pose":     rep.AnnotatorRegistry.get_annotator("pose"),
    }
    for a in annots.values():
        a.attach([render_product])

    for ridx in range(args.num_renders):
        log(f"render {ridx + 1}/{args.num_renders}: placing {args.num_objects} parts")
        # clear previous
        for prim in list(stage.Traverse()):
            if prim.GetPath().pathString.startswith("/World/Obj_"):
                stage.RemovePrim(prim.GetPath())

        for i in range(args.num_objects):
            asset_path, def_prim, label = PARTS[int(rng.integers(0, len(PARTS)))]
            x = float(rng.uniform(-AREA, AREA))
            y = float(rng.uniform(-AREA, AREA))
            yaw = float(rng.uniform(0, 360))
            p = f"/World/Obj_{i:02d}"
            prim = stage.DefinePrim(p, "Xform")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, REST_Z))
            xf.AddOrientOp().Set(dg.euler_to_quaternion(0, 0, yaw))
            xf.AddScaleOp().Set(Gf.Vec3f(SCALE, SCALE, SCALE))
            prim.GetReferences().AddReference(asset_path, primPath=def_prim)
            dg.add_update_semantics(prim, label)

        for _ in range(15):
            simulation_app.update()

        rep.orchestrator.step(rt_subframes=16)
        data = {k: a.get_data() for k, a in annots.items()}
        dg.save_output(data, ridx, args.output)

    log(f"done -> {args.output}")
    simulation_app.close()


if __name__ == "__main__":
    main()
