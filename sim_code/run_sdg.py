#!/usr/bin/env python3
"""
Standalone, headless runner for the synthetic-data-generation pipeline.

Marc's original ``datagenerationscript.py`` targets the Isaac Sim *Script
Editor* and drives the app through the editor's async update loop. This runner
boots a headless ``SimulationApp`` instead, opens a USD scene, and re-uses the
exact spawn / physics / save helpers from ``datagenerationscript.py`` so there
is a single source of truth for the generation logic — only the orchestration
loop is re-implemented synchronously here.

Run inside the Isaac Sim venv on the workstation, e.g.:

    SDG_USD_DIR=/mnt/data/kip_pose/usd \\
    /mnt/data/isaacsim-venv/bin/python sim_code/run_sdg.py \\
        --scene  /mnt/data/kip_pose/usd/scene.usd \\
        --output /mnt/data/kip_pose/output \\
        --num-renders 50 --mode tray

NOTE: This has been written against the Isaac Sim 5.1 API but cannot be fully
verified until the USD assets (parts + tray scene + Zivid camera) are provided.
Annotator names and the ``rep.orchestrator.step`` signature are the most likely
spots to need a tweak — see inline TODO markers.
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Headless Isaac Sim SDG runner")
    p.add_argument("--scene", required=True,
                   help="USD scene to open (contains tray + Zivid camera).")
    p.add_argument("--usd-dir", default=None,
                   help="Directory with part USDs (sets SDG_USD_DIR).")
    p.add_argument("--output", default=None,
                   help="Output directory (sets SDG_OUTPUT_DIR).")
    p.add_argument("--camera", default=None,
                   help="Camera prim path (sets SDG_CAMERA_PATH).")
    p.add_argument("--num-renders", type=int, default=None)
    p.add_argument("--num-objects", type=int, default=None)
    p.add_argument("--mode", choices=["tray", "random"], default=None)
    p.add_argument("--no-headless", action="store_true",
                   help="Show the GUI (default is headless).")
    return p.parse_args()


def main():
    args = parse_args()

    # Pass CLI config to datagenerationscript via env (read at its import time).
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if args.usd_dir:
        os.environ["SDG_USD_DIR"] = args.usd_dir
    if args.output:
        os.environ["SDG_OUTPUT_DIR"] = args.output
    if args.camera:
        os.environ["SDG_CAMERA_PATH"] = args.camera

    # 1. Boot the simulator FIRST — every omni.* import below depends on it.
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": not args.no_headless})

    import omni.usd
    import omni.replicator.core as rep
    import omni.timeline

    # 2. Now safe to import the shared generation helpers.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg

    # 3. Apply remaining overrides onto the module config.
    if args.mode:
        dg.SPAWN_MODE = args.mode
    if args.num_renders is not None:
        dg.NUM_RENDERS = args.num_renders
    if args.num_objects is not None:
        dg.NUM_OBJECTS = args.num_objects
        dg.NUM_ANKERS_IN_TRAY = args.num_objects

    import numpy as np

    # 4. Open the scene and let it load.
    ctx = omni.usd.get_context()
    print(f"[run_sdg] opening scene: {args.scene}")
    _r = ctx.open_stage(args.scene)
    ok = _r[0] if isinstance(_r, (tuple, list)) else _r   # API returns bool or (bool, str)
    if not ok:
        print(f"[run_sdg] FAILED to open stage: {args.scene}")
        simulation_app.close()
        sys.exit(1)
    for _ in range(60):
        simulation_app.update()

    stage = ctx.get_stage()
    rng = np.random.default_rng(dg.RANDOM_SEED)
    timeline = omni.timeline.get_timeline_interface()
    os.makedirs(dg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SDG (headless standalone)")
    print(f"  Mode    : {dg.SPAWN_MODE}")
    print(f"  Assets  : {[a['label'] for a in dg.ASSETS]}")
    print(f"  Renders : {dg.NUM_RENDERS}")
    print(f"  Output  : {dg.OUTPUT_DIR}")
    print("=" * 60)

    # 5. Validate camera.
    cam = stage.GetPrimAtPath(dg.ZIVID_CAMERA_PATH)
    if not cam.IsValid():
        print(f"[run_sdg] camera not found at {dg.ZIVID_CAMERA_PATH}; available cameras:")
        from pxr import UsdGeom
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                print(f"    -> {prim.GetPath().pathString}")
        simulation_app.close()
        sys.exit(1)

    # 6. Annotators (same set as the editor script).
    render_product = rep.create.render_product(
        dg.ZIVID_CAMERA_PATH, resolution=(dg.IMAGE_WIDTH, dg.IMAGE_HEIGHT)
    )
    annots = {
        "rgb":          rep.AnnotatorRegistry.get_annotator("rgb"),
        "bbox_2d":      rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        "semantic_seg": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
        "instance_seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
        "depth":        rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
    }
    for a in annots.values():
        a.attach([render_product])

    # 7. Synchronous generation loop (editor's `await next_update_async()` -> update()).
    for render_idx in range(dg.NUM_RENDERS):
        print(f"\n[Render {render_idx + 1}/{dg.NUM_RENDERS}]")

        timeline.stop()
        for _ in range(5):
            simulation_app.update()

        dg.cleanup_spawned_objects(stage)
        for _ in range(5):
            simulation_app.update()

        if dg.SPAWN_MODE == "tray":
            dg.spawn_tray_mode(stage, rng)
        else:
            dg.spawn_random_mode(stage, rng)

        for _ in range(10):
            simulation_app.update()

        timeline.play()
        for step in range(dg.PHYSICS_SETTLE_STEPS):
            simulation_app.update()
        timeline.pause()
        for _ in range(5):
            simulation_app.update()

        # TODO(verify): step signature on 5.1 — `delta_time` may be required/forbidden.
        rep.orchestrator.step(rt_subframes=8)

        data = {k: a.get_data() for k, a in annots.items()}
        dg.save_output(data, render_idx, dg.OUTPUT_DIR)

    timeline.stop()
    for a in annots.values():
        a.detach()

    print(f"\n[run_sdg] done — {dg.NUM_RENDERS} samples -> {dg.OUTPUT_DIR}")
    simulation_app.close()


if __name__ == "__main__":
    main()
