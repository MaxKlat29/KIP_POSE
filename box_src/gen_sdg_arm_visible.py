#!/usr/bin/env python3
"""Arm-VISIBLE top-down SDG for the BOP/GDRNPP foundation (Story S-202 / T-061).

Renders the real GST cell from a top-down Zivid-like camera with the **LARA5
arm and the work-cart left VISIBLE** — the arm is an OCCLUDER, never hidden, never
a BOP object. Parts (focus: Anker_Kurz, Anker_Lang, Zahnrad; others as
distractors) are dropped with physics, settle on the calibrated cart surface,
and are captured together with everything the Isaac->BOP converter needs:

  per frame, written to <output>/:
    rgb_{idx:04d}.png            RGB
    depth_{idx:04d}.npy          metric distance-to-camera (float32, METRES)
    instance_{idx:04d}.png       instance-seg (uint16/uint8 ids)  -> mask_visib source
    instance_labels_{idx:04d}.json  idToLabels for instance-seg (id -> {class, instanceId/primPath})
    semantic_{idx:04d}.png       semantic-seg (uint8 class ids)
    semantic_labels_{idx:04d}.json  idToLabels for semantic-seg (id -> {class})
    gt_raw_{idx:04d}.json        camera intrinsics + cam c2w + per-instance prim world transform + class
    obb_2d_{idx:04d}.json        oriented 2D boxes (debug / detector reuse)

The converter (isaac_to_bop.py) turns this raw bundle into the BOP-format
contract (scene_camera/scene_gt/scene_gt_info + mask_visib). NOTHING in here
writes BOP files directly — clean separation (Isaac venv vs bop venv).

DESIGN NOTES (Viktor adr.md cross-refs):
  * R5 — Arm = occluder: the arm prim is NOT given a semantic label, so it never
    appears as an instance/obj_id. It DOES occlude parts -> shows up implicitly as
    reduced visible pixels of the parts behind it (visib_fract lever).
  * R4 — the camera transform written here is the **camera-to-world** matrix
    (Isaac/USD convention); the converter inverts it to w2c. Documented in
    gt_raw so the converter never has to guess direction.
  * mm/scale — parts are spawned at OBJECT_SCALE=1e-3 (USD geometry is in mm,
    scaled to scene-metres). The PLY models are exported in mm; the GT translation
    written here is in scene-metres and the converter multiplies by 1000.

Self-contained: imports datagenerationscript helpers + run_scene.compute_oriented_boxes.
"""
import argparse, json, os, sys, time

def log(m): print(f"[armgen {time.strftime('%H:%M:%S')}] {m}", flush=True)

# Focus parts first (weighted heavier in spawning), distractors after.
# Poltopf_kurz_centered is EXCLUDED (degenerate ~50um mesh, adr.md R6).
PART_FILES = [
    ("Anker_Kurz",            "Anker_Kurz.usd"),           # obj_id 1  FOCUS
    ("Anker_Lang",            "Anker_Lang.usd"),           # obj_id 2  FOCUS
    ("Buerstenhalter_2polig", "Buerstenhalter_2polig.usd"),# obj_id 3  distractor
    ("Getriebegehaeuse_typ4", "Getriebegehaeuse_typ4.usdz"),# obj_id 4 distractor
    ("Ringmagnet",            "Ringmagnet.usd"),           # obj_id 5  distractor
    ("Zahnrad",               "Zahnrad_Typ7.usdz"),        # obj_id 6  FOCUS
]
FOCUS_LABELS = {"Anker_Kurz", "Anker_Lang", "Zahnrad"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--usd-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-scenes", type=int, default=20)
    p.add_argument("--min-obj", type=int, default=6)
    p.add_argument("--max-obj", type=int, default=14)
    p.add_argument("--dr-strong", action="store_true",
                   help="T-038 Phase-2: STRONGER domain randomization — per-object "
                        "material roughness/metallic/tint, denser clutter, wider "
                        "camera jitter, randomized background/dome.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--start", type=int, default=0)
    # calibrated GST_Scene cart/tray surface (measured -0.007 m)
    p.add_argument("--table-z", type=float, default=-0.007)
    p.add_argument("--settle", type=int, default=200)
    # spawn-region: the open work area in front of / under the arm so the arm
    # genuinely occludes some parts. Keeps the calibrated on-surface bounds.
    p.add_argument("--focus-frac", type=float, default=0.6,
                   help="fraction of spawned objects forced to be focus parts")
    # spawn region (metres, world). Default = open front-left work area in front
    # of the LARA5 arm so MOST parts are camera-visible, while the arm still
    # occludes a fraction (occlusion training signal). Calibrated on-surface.
    p.add_argument("--spawn-x", default="0.18,0.52", help="xmin,xmax (m)")
    p.add_argument("--spawn-y", default="0.08,0.50", help="ymin,ymax (m)")
    p.add_argument("--spawn-z", default="0.30,0.55", help="zmin,zmax drop height (m)")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["SDG_USD_DIR"] = a.usd_dir
    os.environ["SDG_OUTPUT_DIR"] = a.output
    log("booting SimulationApp ...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True}); log("ready")
    import numpy as np, omni.usd, omni.replicator.core as rep, omni.timeline, carb
    from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics, PhysxSchema, UsdShade, Sdf
    s = carb.settings.get_settings()
    s.set("/app/asyncRendering", False); s.set("/omni/replicator/asyncRendering", False)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # the helpers live in the box sim_code dir; allow import from there too
    sys.path.insert(0, "/mnt/data/kip_pose/sim_code")
    import datagenerationscript as dg
    from run_scene import compute_oriented_boxes

    U = a.usd_dir
    dg.ASSETS = [{"path": os.path.join(U, fn), "label": lbl} for lbl, fn in PART_FILES]
    all_labels = {lbl.lower() for lbl, _ in PART_FILES}
    focus_idx = [i for i, (lbl, _) in enumerate(PART_FILES) if lbl in FOCUS_LABELS]
    distract_idx = [i for i, (lbl, _) in enumerate(PART_FILES) if lbl not in FOCUS_LABELS]
    log(f"assets: {[x['label'] for x in dg.ASSETS]} | focus={[PART_FILES[i][0] for i in focus_idx]}")
    dg.PHYSICS_SETTLE_STEPS = a.settle
    # drop ABOVE the cart-top into the open work-area in front of the arm
    _xy = lambda s: tuple(float(v) for v in s.split(","))
    dg.SPAWN_BOUNDS = {"x": _xy(a.spawn_x), "y": _xy(a.spawn_y), "z": _xy(a.spawn_z)}
    log(f"spawn bounds: {dg.SPAWN_BOUNDS}")

    ctx = omni.usd.get_context(); _r = ctx.open_stage(a.scene)
    ok = _r[0] if isinstance(_r, (tuple, list)) else _r
    if not ok: log("FAILED open"); app.close(); sys.exit(1)
    for _ in range(80): app.update()
    stage = ctx.get_stage(); log(f"scene loaded ({len(list(stage.Traverse()))} prims)")

    # ── static colliders on cart + trays so parts rest on REAL surfaces ──────
    STATIC_ROOTS = ["/World/Basiswagen", "/World/Basiswagen_01", "/World/Basiswagen_02",
                    "/World/Anker_Tray", "/World/Poltopf_Tray"]
    n_coll = 0
    for root in STATIC_ROOTS:
        rp_ = stage.GetPrimAtPath(root)
        if not rp_.IsValid(): continue
        for prim in Usd.PrimRange(rp_):
            if prim.IsA(UsdGeom.Mesh):
                try:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mc.CreateApproximationAttr().Set("none")
                    n_coll += 1
                except Exception: pass
    log(f"{n_coll} static collider meshes (cart+trays)")

    # ── ARM STAYS VISIBLE (R5). Give the arm + cart NO semantic label so they
    #    never become a BOP instance, but they ARE rendered + DO occlude. ─────
    ARM_PATHS = ["/World/NEURA_LARA5_Pose_Zivid_Detection", "/World/lara5",
                 "/World/Greifer_mit_Fingern"]
    arm_present = []
    for ap in ARM_PATHS:
        if stage.GetPrimAtPath(ap).IsValid():
            arm_present.append(ap)
            # ensure the arm has a collider too, so a dropped part can rest on/against it
            for prim in Usd.PrimRange(stage.GetPrimAtPath(ap)):
                if prim.IsA(UsdGeom.Mesh):
                    try:
                        UsdPhysics.CollisionAPI.Apply(prim)
                        mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mc.CreateApproximationAttr().Set("convexHull")
                    except Exception: pass
    log(f"ARM VISIBLE (occluder, no obj_id): {arm_present or 'NONE FOUND — check scene!'}")
    if not arm_present:
        log("WARNING: no arm prim found — arm-visibility cannot be guaranteed!")

    # ── DR lights ────────────────────────────────────────────────────────────
    dome = UsdLux.DomeLight.Define(stage, "/World/_DRDome")
    dome_int = dome.CreateIntensityAttr(500.0); dome_col = dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))
    dist = UsdLux.DistantLight.Define(stage, "/World/_DRDistant")
    dist_int = dist.CreateIntensityAttr(800.0)
    dist_rot = UsdGeom.Xformable(dist.GetPrim()).AddRotateXYZOp()

    ps = UsdPhysics.Scene.Define(stage, "/World/_PhysicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); ps.CreateGravityMagnitudeAttr(9.81)
    gp = UsdGeom.Cube.Define(stage, "/World/_PhysGround"); gp.GetSizeAttr().Set(1.0)
    gx = UsdGeom.Xformable(gp); gx.ClearXformOpOrder()
    gx.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.1, a.table_z - 0.01)); gx.AddScaleOp().Set(Gf.Vec3f(4, 4, 0.02))
    UsdPhysics.CollisionAPI.Apply(gp.GetPrim()); UsdGeom.Imageable(gp.GetPrim()).MakeInvisible()
    for _ in range(5): app.update()

    # ── DR camera (top-down with jitter) ─────────────────────────────────────
    cam = UsdGeom.Camera.Define(stage, "/World/_DRCam"); cam_xf = UsdGeom.Xformable(cam.GetPrim())
    SENSOR_W = float(cam.GetHorizontalApertureAttr().Get() or 20.955)  # mm, USD default
    cam_focal = cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    cam_path = "/World/_DRCam"
    os.makedirs(a.output, exist_ok=True)
    rp = rep.create.render_product(cam_path, (a.width, a.height))
    annots = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
              "bbox_2d": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
              "semantic_seg": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
              "instance_seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
              "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera")}
    for x in annots.values(): x.attach([rp])
    tl = omni.timeline.get_timeline_interface(); rng = np.random.default_rng(a.seed)
    base_eye = np.array([0.46, 0.27, 0.95]); t0 = time.time(); done = 0

    def part_world_transforms(stage):
        """{prim_path: {'T_w': 4x4 row-major (object->world, scene-metres)}} for
        every settled spawned part. Row-major so it matches BOP/numpy convention."""
        out = {}
        for prim in stage.Traverse():
            pp = prim.GetPath().pathString
            if pp.startswith("/World/SpawnedObject_") and prim.GetParent().GetPath().pathString == "/World":
                try:
                    M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()), float).reshape(4, 4)
                    # USD stores transforms ROW-vector convention (point row * M).
                    # Convert to the column-vector / row-major convention used by
                    # BOP (p' = M @ p): that is the TRANSPOSE of the USD matrix.
                    out[pp] = M.T.tolist()
                except Exception: pass
        return out

    def cam_to_world(cam_xf):
        """Camera object->world 4x4 (row-major, column-vector convention)."""
        M = np.array(UsdGeom.Xformable(cam_xf.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), float).reshape(4, 4)
        return M.T.tolist()

    def randomize_material(stage, prim_path, rng):
        """T-038 DR-strong: bind a randomized UsdPreviewSurface to a spawned part so
        the network sees a wide span of metallic/roughness/tint instead of the one
        baked CAD look. Sim2real lever — real metal parts vary in finish + lighting.
        """
        try:
            mat_path = prim_path + "/_DRMat"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            # metallic parts: bias metallic high but keep a fraction dielectric
            metallic = float(rng.uniform(0.0, 1.0))
            rough = float(rng.uniform(0.15, 0.85))
            # neutral-grey base with a small random tint (steel/brass/zinc-ish)
            g = float(rng.uniform(0.45, 0.85))
            tint = np.array([g, g, g]) * np.array(
                [rng.uniform(0.92, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.85, 1.0)])
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(c) for c in tint]))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(
                float(rng.uniform(0.3, 0.8)))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)
        except Exception as e:
            pass  # DR is best-effort; never crash the render over a material

    def spawn_weighted(stage, rng, n_obj, focus_frac):
        """Like dg.spawn_random_mode but forces a focus_frac of focus parts.
        Returns {prim_path: label}."""
        existing = dg.get_existing_objects(stage)
        spawned = 0; attempt = 0
        path2label = {}
        max_attempts = n_obj * dg.MAX_SPAWN_ATTEMPTS * 2
        n_focus = int(round(n_obj * focus_frac))
        plan = ([rng.choice(focus_idx) for _ in range(n_focus)] +
                [rng.choice(distract_idx) for _ in range(n_obj - n_focus)])
        rng.shuffle(plan)
        while spawned < n_obj and attempt < max_attempts:
            attempt += 1
            x = rng.uniform(*dg.SPAWN_BOUNDS["x"]); y = rng.uniform(*dg.SPAWN_BOUNDS["y"])
            z = rng.uniform(*dg.SPAWN_BOUNDS["z"])
            half = dg.OBJECT_SIZE / 2.0 + 0.02
            nmin = (x - half, y - half, z - half); nmax = (x + half, y + half, z + half)
            overlap = False
            for obj in existing:
                ss = obj["size"] + 0.02
                om = tuple(p - ss for p in obj["pos"]); ox = tuple(p + ss for p in obj["pos"])
                if (nmax[0] > om[0] and nmin[0] < ox[0] and nmax[1] > om[1] and
                        nmin[1] < ox[1] and nmax[2] > om[2] and nmin[2] < ox[2]):
                    overlap = True; break
            if overlap: continue
            asset = dg.ASSETS[plan[spawned]]
            rx = float(rng.uniform(0, 360)); ry = float(rng.uniform(0, 360)); rz = float(rng.uniform(0, 360))
            pp = f"/World/SpawnedObject_{spawned:03d}"
            dg.spawn_single_object(stage, pp, asset, x, y, z, rx, ry, rz)
            if a.dr_strong:
                randomize_material(stage, pp, rng)
            existing.append({"path": pp, "pos": (x, y, z), "size": dg.OBJECT_SIZE / 2})
            path2label[pp] = asset["label"]
            spawned += 1
        return path2label

    for sidx in range(a.start, a.num_scenes):
        # DR: light intensity / colour / direction. dr_strong widens every range +
        # randomizes the dome colour fully (not just warm) for harder lighting.
        if a.dr_strong:
            dome_int.Set(float(rng.uniform(120, 1300)))
            dome_col.Set(Gf.Vec3f(float(rng.uniform(0.75, 1)), float(rng.uniform(0.75, 1)),
                                  float(rng.uniform(0.75, 1))))
            dist_int.Set(float(rng.uniform(150, 2200)))
            dist_rot.Set(Gf.Vec3f(float(rng.uniform(-75, -10)), float(rng.uniform(-60, 60)),
                                  float(rng.uniform(0, 360))))
        else:
            dome_int.Set(float(rng.uniform(250, 900)))
            dome_col.Set(Gf.Vec3f(1.0, float(rng.uniform(0.85, 1)), float(rng.uniform(0.85, 1))))
            dist_int.Set(float(rng.uniform(300, 1400)))
            dist_rot.Set(Gf.Vec3f(float(rng.uniform(-60, -20)), float(rng.uniform(-40, 40)),
                                  float(rng.uniform(0, 360))))
        # top-down camera jitter + roll (wider span under dr_strong)
        if a.dr_strong:
            jx, jy = rng.uniform(-0.08, 0.08, 2); h = 0.95 + rng.uniform(-0.10, 0.14)
            roll = float(rng.uniform(-12, 12))
        else:
            jx, jy = rng.uniform(-0.04, 0.04, 2); h = 0.95 + rng.uniform(-0.05, 0.08)
            roll = 0.0
        eye = np.array([base_eye[0] + jx, base_eye[1] + jy, h])
        look = np.array([base_eye[0] + jx, base_eye[1] + jy, a.table_z])
        # roll the up-vector around the view axis for in-plane camera rotation DR
        up = Gf.Vec3d(float(np.sin(np.radians(roll))), float(np.cos(np.radians(roll))), 0.0)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), up)
        cam_xf.ClearXformOpOrder(); cam_xf.AddTransformOp().Set(view.GetInverse())
        focal = float(rng.uniform(14, 24) if a.dr_strong else rng.uniform(16, 22))
        cam_focal.Set(focal)

        n_obj = int(rng.integers(a.min_obj, a.max_obj + 1))
        tl.stop()
        for _ in range(5): app.update()
        dg.cleanup_spawned_objects(stage)
        for _ in range(5): app.update()
        path2label = spawn_weighted(stage, rng, n_obj, a.focus_frac)
        for _ in range(10): app.update()
        tl.play()
        for _ in range(dg.PHYSICS_SETTLE_STEPS): app.update()
        tl.pause()
        for _ in range(5): app.update()

        rep.orchestrator.step(rt_subframes=16)
        data = {k: x.get_data() for k, x in annots.items()}

        # ── save raw isaac bundle ────────────────────────────────────────────
        from PIL import Image
        Image.fromarray(data["rgb"]).save(os.path.join(a.output, f"rgb_{sidx:04d}.png"))
        dd = data.get("depth"); arr = dd["data"] if isinstance(dd, dict) else dd
        if arr is not None:
            np.save(os.path.join(a.output, f"depth_{sidx:04d}.npy"), np.asarray(arr, np.float32))
        for key, fn in (("instance_seg", "instance"), ("semantic_seg", "semantic")):
            d = data[key]
            if isinstance(d, dict):
                # keep full id resolution (uint16) — many instances + the arm
                np.save(os.path.join(a.output, f"{fn}_{sidx:04d}.npy"),
                        np.asarray(d["data"]).astype(np.uint32))
                json.dump(dg.convert_numpy(d.get("info", {})),
                          open(os.path.join(a.output, f"{fn}_labels_{sidx:04d}.json"), "w"), indent=2)

        # ── GT: camera intrinsics + cam c2w + per-instance world transform ───
        gt_raw = {
            "image_id": sidx,
            "width": a.width, "height": a.height,
            "focal_length_mm": focal,
            "horizontal_aperture_mm": SENSOR_W,
            # cam_K computed in the converter from focal/aperture/W/H
            "cam_c2w": cam_to_world(cam_xf),     # camera->world (R4: converter inverts)
            "object_scale": list(dg.OBJECT_SCALE),
            "table_z_m": a.table_z,
            "instances": [],                      # filled below
        }
        world_T = part_world_transforms(stage)
        for pp, label in path2label.items():
            T = world_T.get(pp)
            if T is None: continue
            gt_raw["instances"].append({
                "prim_path": pp,
                "label": label,
                "T_obj2world": T,                 # 4x4 row-major, scene-metres
            })
        json.dump(dg.convert_numpy(gt_raw),
                  open(os.path.join(a.output, f"gt_raw_{sidx:04d}.json"), "w"), indent=2)

        # ── oriented 2D boxes (debug / detector reuse) ───────────────────────
        inst = data["instance_seg"]; seg = data["semantic_seg"]
        obbs = compute_oriented_boxes(inst["data"] if isinstance(inst, dict) else None,
                                      seg["data"] if isinstance(seg, dict) else None,
                                      seg["info"].get("idToLabels", {}) if isinstance(seg, dict) else {},
                                      all_labels, bbox_data=data.get("bbox_2d"))
        json.dump({"boxes": obbs}, open(os.path.join(a.output, f"obb_2d_{sidx:04d}.json"), "w"), indent=2)

        done += 1
        if done % 5 == 0 or done <= 3:
            r = done / max(1e-6, time.time() - t0)
            log(f"scene {sidx}: {len(path2label)} parts, {len(obbs)} boxes | "
                f"{r:.2f}/s | ETA {(a.num_scenes - sidx - 1)/max(1e-6, r)/60:.1f}min")

    tl.stop()
    log(f"DONE {done} scenes -> {a.output} | {(time.time()-t0)/60:.1f}min")
    app.close()


if __name__ == "__main__":
    main()
