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
    # T-038 Phase-2: 0..10 parts INCLUDING EMPTY (0) and sparse (1-3). The model
    # must also recognise empty / nearly-empty cells, not just crowded trays.
    # n_obj is sampled inclusive [min,max]; min=0 => some scenes render the bare
    # cell (0 GT, arm + cart only) which the converter writes as a 0-GT frame.
    p.add_argument("--min-obj", type=int, default=0)
    p.add_argument("--max-obj", type=int, default=10)
    p.add_argument("--equalize-bright-materials", action="store_true",
                   help="T-112: rebind a darker metallic material onto spawned parts "
                        "whose baked CAD material is near-white (diffuseColor mean > "
                        "--bright-mat-thresh), so a pale part (Zahnrad_Typ7.usdz carries "
                        "diffuseColor 0.828 'SmoothWhite') renders with contrast like the "
                        "dark-metal Anker instead of washing out against the cell. "
                        "Default OFF — enabling CHANGES the rendered appearance and "
                        "REQUIRES a retrain (the shipped model was trained on the bright "
                        "look). Independent of --dr-strong.")
    p.add_argument("--bright-mat-thresh", type=float, default=0.70,
                   help="T-112: mean baked diffuseColor above this counts as 'too bright' "
                        "for --equalize-bright-materials (Zahnrad SmoothWhite = 0.828).")
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
    p.add_argument("--focus-frac", type=float, default=0.6,
                   help="fraction of spawned objects forced to be focus parts")
    # T-038 Phase-2: spawn over the WHOLE TABLE, not the narrow inner-table zone.
    # Defaults mirror Marc's datagenerationscript.py SPAWN_BOUNDS (random mode) —
    # the full calibrated cart-surface extent. Parts may land anywhere on the
    # table; the LARA5 arm still occludes whatever ends up behind it (occlusion
    # training signal preserved). x: 0.726 m wide, y: 0.495 m deep.
    p.add_argument("--spawn-x", default="0.05732,0.78332", help="xmin,xmax (m) — full table")
    p.add_argument("--spawn-y", default="0.04157,0.53698", help="ymin,ymax (m) — full table")
    p.add_argument("--spawn-z", default="0.30,0.55", help="zmin,zmax drop height (m)")
    # T-182: keep-out — no part spawns within --arm-clear (m, XY radius) of the arm
    # foundation, so dropped parts can't clip into the static arm base. The center is
    # auto-derived from the arm prim in the scene; --arm-base x,y overrides it.
    # Default 0.18 = LARA5 pedestal (~0.08 m radius) + halbe Teilelaenge (~0.05 m)
    # + 0.05 m Puffer. Der Wert gilt fuer die ABWURF- UND die Endposition.
    p.add_argument("--arm-clear", type=float, default=0.18,
                   help="XY keep-out radius (m) around the arm foundation; 0 disables")
    p.add_argument("--arm-base", default="",
                   help="override arm-foundation center 'x,y' (m); empty = derive from scene")
    # Smoke/debug only: force an explicit per-scene part-count sequence instead of
    # sampling [min,max]. e.g. "0,8,1,10,3" guarantees an empty scene + a full
    # one for the smoke proof. Empty when omitted (normal RNG sampling).
    p.add_argument("--force-counts", default="",
                   help="comma list of n_obj per scene (smoke proof; overrides min/max)")
    p.add_argument("--force-each-focus", action="store_true",
                   help="T-119: garantiere >=1 gespawntes Objekt JEDER Fokus-Klasse "
                        "(Anker_Kurz/Anker_Lang/Zahnrad), sofern Platz (n_focus>=3). "
                        "Vom Live-Website-Sim genutzt; aus fuer Dataset-/Eval-Gen.")
    # T-183: persistenter warmer Worker. Statt pro Sim-Klick SimulationApp neu zu
    # booten (~25s cold-start, Swap-Thrash-Quelle auf der 15Gi-Box) bootet der
    # Prozess EINMAL und rendert danach je HTTP-POST /render eine Szene (~2-5s).
    p.add_argument("--serve", action="store_true",
                   help="T-183: warm-worker — Isaac einmal booten, dann HTTP /render bedienen")
    p.add_argument("--serve-port", type=int, default=8091)  # 8090 = mesh-gateway!
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

    # ── PATCH: Isaac-Sim 5.1 deprecated `add_update_semantics` — bbox_2d_tight
    # erkennt damit unsere Spawn-Parts nicht mehr. Wir wrappen den Call und
    # registrieren parallel die NEUE label-API (`add_labels`) damit der
    # Replicator-Annotator die Parts wieder pro-Frame als Boxen liefert.
    # (Bug 2026-05-26: smoke v3 zeigte 4-9 spawned parts aber 1 box im output.)
    _orig_aus = dg.add_update_semantics
    try:
        from isaacsim.core.utils.semantics import (
            add_labels as _add_labels,
            upgrade_prim_semantics_to_labels as _upgrade_labels,
        )
        def _patched_aus(prim, label):
            _orig_aus(prim, label)
            try:
                _upgrade_labels(prim)
            except Exception:
                pass
            try:
                _add_labels(prim, labels=[label], instance_name="class")
            except TypeError:
                try: _add_labels(prim, labels=[label])
                except Exception: pass
            except Exception:
                pass
        dg.add_update_semantics = _patched_aus
        log("semantics-API patched: add_update_semantics + add_labels (Isaac-5.1)")
    except ImportError:
        log("WARN: new semantics-API not found — falling back to deprecated only")

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

    # ── REMOVE baked-in part PRIMS, KEEP tray structures —
    # GST_Scene.usd hat:
    #   /World/Anker_Tray/Tray_Anker_leer    ← die Plastik-Schale (BEHALTEN)
    #   /World/Anker_Tray/Anker_Lang         ← baked-in Anker (RAUS)
    #   /World/Anker_Tray/Anker_Kurz         ← baked-in Anker (RAUS)
    #   /World/Poltopf_Tray/Tray_Poltopf_leer ← Schale (BEHALTEN)
    # Spawn-Bounds X=[0.057, 0.783] vermeiden eh die Tray-XY-Bereiche
    # (Anker_Tray X=[0.806, 1.021], Poltopf_Tray X=[-0.204, -0.004]).
    # NIEMALS Substring-Match auf Children-Namen — würde "Tray_Anker_leer" treffen.
    BAKED_PART_PATHS = [
        "/World/Anker_Tray/Anker_Kurz",
        "/World/Anker_Tray/Anker_Lang",
    ]
    # Exakte Part-Klassen-Namen für VARIANTEN ("Anker_Kurz_1" usw.) — Tray-Schale
    # ("Tray_Anker_leer") matcht NICHT, weil sie mit "Tray_" beginnt.
    PART_CLASS_PREFIXES = ("Anker_Kurz", "Anker_Lang", "Zahnrad",
                            "Buerstenhalter", "Getriebe", "Ringmagnet", "Poltopf_")
    for tray in ["/World/Anker_Tray", "/World/Poltopf_Tray"]:
        tray_prim = stage.GetPrimAtPath(tray)
        if not tray_prim.IsValid():
            continue
        for child in tray_prim.GetChildren():
            name = child.GetName()
            # Tray-Schale hat Präfix "Tray_" — die NICHT.
            if name.startswith("Tray_"):
                continue
            if any(name.startswith(p) for p in PART_CLASS_PREFIXES):
                BAKED_PART_PATHS.append(child.GetPath().pathString)
    n_removed = 0
    for p in sorted(set(BAKED_PART_PATHS)):
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            stage.RemovePrim(p)
            n_removed += 1
    log(f"removed {n_removed} baked tray parts (trays themselves intact)")

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
    # KEINE Collision auf dem Arm (Fix 2026-05-28): mit convexHull-Collider blieben
    # gedroppte Teile AUF dem Arm liegen — in real-life unmöglich (würde fallen).
    # Ohne Collision fallen alle Teile durch den Arm auf den Tisch; der Arm bleibt
    # voll sichtbar + occludiert die Teile die unter ihm landen (Occlusion-Signal
    # bleibt erhalten). So liegt NIE ein Teil auf dem Arm.
    ARM_PATHS = ["/World/NEURA_LARA5_Pose_Zivid_Detection", "/World/lara5",
                 "/World/Greifer_mit_Fingern"]
    arm_present = [ap for ap in ARM_PATHS if stage.GetPrimAtPath(ap).IsValid()]
    log(f"ARM VISIBLE (occluder, NO collision, no obj_id): {arm_present or 'NONE FOUND — check scene!'}")
    if not arm_present:
        log("WARNING: no arm prim found — arm-visibility cannot be guaranteed!")

    # ── Arm keep-out center (T-182): derive the foundation XY from the arm prim so
    #    spawned parts never clip into the static base. Self-correcting if the scene
    #    moves; --arm-base overrides. Shared by BOTH sim types — normal + MoE use
    #    this same generator (MoE only adds IR-shadow masking AFTER spawn). ────────
    arm_base_xy = None
    if arm_present:
        try:
            _at = UsdGeom.Xformable(stage.GetPrimAtPath(arm_present[0])
                  ).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            arm_base_xy = (float(_at[0]), float(_at[1]))
        except Exception as _e:
            log(f"WARN: could not read arm transform ({_e}) — keep-out disabled")
    if a.arm_base.strip():
        _bx, _by = (float(v) for v in a.arm_base.split(","))
        arm_base_xy = (_bx, _by)
    ARM_CLEAR_R = float(a.arm_clear)
    if arm_base_xy and ARM_CLEAR_R > 0:
        log(f"arm keep-out: center=({arm_base_xy[0]:.4f},{arm_base_xy[1]:.4f}) "
            f"radius={ARM_CLEAR_R:.3f} m")
    else:
        log("arm keep-out: DISABLED")

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

    # ── Camera = Marc's fixed Zivid (sim_trainingsdatageneration_pipeline) ──
    # Aus GST_Scene.usd /World/Zivid (focal=15.5mm, hap=vap=15mm, pos in scene).
    # KEIN _DRCam, KEIN jitter, KEIN random focal — eine feste deployment-ähnliche
    # Kamera (Zivid Top-Down). DR macht den Rest (lighting, materials, spawn).
    cam_path = "/World/Zivid"
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid() or not cam_prim.IsA(UsdGeom.Camera):
        log(f"FATAL: {cam_path} not found in scene {a.scene} — needs GST_Scene.usd with /World/Zivid")
        app.close(); sys.exit(1)
    cam = UsdGeom.Camera(cam_prim)
    cam_xf = UsdGeom.Xformable(cam_prim)
    SENSOR_W = float(cam.GetHorizontalApertureAttr().Get() or 15.0)
    cam_focal = cam.GetFocalLengthAttr()
    log(f"using fixed Zivid camera: focal={cam_focal.Get()}mm  hap={SENSOR_W}mm")
    os.makedirs(a.output, exist_ok=True)
    rp = rep.create.render_product(cam_path, (a.width, a.height))
    # Annotator-Set:
    #   rgb / bbox_2d / semantic / instance / depth — Pflicht für GDRNPP+Detector.
    #   bbox_3d — Future-Proofing für RGB-only 3D-Detection (CenterPose, FCOS3D).
    # NORMALS RAUS (2026-05-26): wir haben bei Inference KEINE Depth-Kamera,
    # daher kann man normals nicht aus Depth ableiten → kein PVN3D/FFB6D möglich.
    # GDRNPP-RGB-only ist und bleibt der operative Path.
    annots = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
              "bbox_2d": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
              "bbox_3d": rep.AnnotatorRegistry.get_annotator("bounding_box_3d"),
              "semantic_seg": rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
              "instance_seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
              "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera")}
    for x in annots.values(): x.attach([rp])
    tl = omni.timeline.get_timeline_interface(); rng = np.random.default_rng(a.seed)
    base_eye = np.array([0.46, 0.27, 0.95]); t0 = time.time(); done = 0
    force_counts = [int(v) for v in a.force_counts.split(",") if v.strip() != ""]
    if force_counts:
        log(f"FORCE-COUNTS (smoke): per-scene n_obj = {force_counts}")

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

    def _mean_bound_diffuse(stage, prim_path):
        """Return the mean diffuseColor of the BRIGHTEST UsdPreviewSurface bound under
        prim_path, or None if no preview-surface diffuseColor is found. Used to detect a
        baked near-white CAD material (T-112: Zahnrad SmoothWhite = (0.828,0.828,0.828))."""
        try:
            best = None
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                if not prim.IsA(UsdShade.Shader):
                    continue
                sh = UsdShade.Shader(prim)
                if sh.GetIdAttr().Get() != "UsdPreviewSurface":
                    continue
                inp = sh.GetInput("diffuseColor")
                v = inp.Get() if inp else None
                if v is None:
                    continue
                m = float((v[0] + v[1] + v[2]) / 3.0)
                if best is None or m > best:
                    best = m
            return best
        except Exception:
            return None

    def equalize_bright_material(stage, prim_path, rng, thresh):
        """T-112: if a spawned part's baked CAD material is near-white (mean diffuseColor
        > thresh), rebind a darker neutral-metal UsdPreviewSurface so it renders with
        contrast like the dark-metal Anker (whose meshes carry no preview-surface and use
        dark MDL looks). NON-DESTRUCTIVE to other parts — only touches bright ones. Returns
        True if a part was darkened. Default-OFF flag; enabling REQUIRES a retrain."""
        m = _mean_bound_diffuse(stage, prim_path)
        if m is None or m <= thresh:
            return False
        try:
            mat_path = prim_path + "/_EqMat"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            # dark neutral metal, matched to the Anker tonal range (renders gray ~50-90,
            # not ~175). Small random tint so the network still sees finish variation.
            g = float(rng.uniform(0.18, 0.40))
            tint = np.array([g, g, g]) * np.array(
                [rng.uniform(0.92, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.85, 1.0)])
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(c) for c in tint]))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(rng.uniform(0.5, 1.0)))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rng.uniform(0.25, 0.6)))
            shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(float(rng.uniform(0.3, 0.7)))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)
            return True
        except Exception:
            return False  # best-effort; never crash the render over a material

    def spawn_weighted(stage, rng, n_obj, focus_frac):
        """Like dg.spawn_random_mode but forces a focus_frac of focus parts.
        Returns {prim_path: label}."""
        existing = dg.get_existing_objects(stage)
        spawned = 0; attempt = 0
        path2label = {}
        max_attempts = n_obj * dg.MAX_SPAWN_ATTEMPTS * 2
        n_focus = int(round(n_obj * focus_frac))
        # T-119: jede Fokus-Klasse mehrfach seeden, sofern Platz. >=2 wenn moeglich
        # (robust gegen Upright-Removal der rotsym. Anker: ein Exemplar darf aufrecht
        # wegfallen, das zweite bleibt → praktisch immer >=1 von jedem Typ), sonst
        # >=1. Rest zufaellig aus den Fokus-Klassen. Sonst rein zufaellig (alt).
        nf = len(focus_idx)
        if getattr(a, "force_each_focus", False) and n_focus >= nf:
            reps = 2 if n_focus >= 2 * nf else 1
            seeded = list(focus_idx) * reps
            focus_plan = seeded + [rng.choice(focus_idx)
                                   for _ in range(n_focus - len(seeded))]
        else:
            focus_plan = [rng.choice(focus_idx) for _ in range(n_focus)]
        plan = (focus_plan +
                [rng.choice(distract_idx) for _ in range(n_obj - n_focus)])
        rng.shuffle(plan)
        while spawned < n_obj and attempt < max_attempts:
            attempt += 1
            x = rng.uniform(*dg.SPAWN_BOUNDS["x"]); y = rng.uniform(*dg.SPAWN_BOUNDS["y"])
            z = rng.uniform(*dg.SPAWN_BOUNDS["z"])
            # T-182 arm keep-out (XY): reject drops within ARM_CLEAR_R of the arm
            # foundation so parts can't clip into the static arm base. Both sim
            # types (normal + MoE) share this spawner.
            if arm_base_xy is not None and ARM_CLEAR_R > 0:
                _dx = x - arm_base_xy[0]; _dy = y - arm_base_xy[1]
                if _dx * _dx + _dy * _dy < ARM_CLEAR_R * ARM_CLEAR_R:
                    continue
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
            if a.equalize_bright_materials:
                # T-112: darken near-white baked materials (e.g. Zahnrad SmoothWhite)
                # AFTER any DR bind, so the equalizer wins for a still-bright part.
                if equalize_bright_material(stage, pp, rng, a.bright_mat_thresh):
                    log(f"  [eq-bright] darkened {asset['label']} @ {pp}")
            existing.append({"path": pp, "pos": (x, y, z), "size": dg.OBJECT_SIZE / 2})
            path2label[pp] = asset["label"]
            spawned += 1
        return path2label

    def render_one(sidx, out_dir, forced_n, rng):
        """Render EINE Szene nach out_dir (Dateien *_{sidx:04d}.*). forced_n=None →
        n_obj aus [min,max] sampeln (Batch); sonst exakt forced_n Teile (Live/Smoke).
        rng wird als Parameter uebergeben, damit der Batch-Pfad seinen geteilten,
        seed-reproduzierbaren rng behaelt und der Serve-Pfad je Request frisch seedet.
        Rueckgabe: (n_parts, n_boxes)."""
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
        # Marc's fixed Zivid Kamera — KEIN jitter, KEIN random focal mehr.
        # Die Kamera bleibt frame-für-frame an ihrer USD-position. Alle DR jetzt
        # in lighting + materials + spawn (Anker fallen physikbasiert per scene).

        if forced_n is not None:
            n_obj = forced_n
        else:
            # inclusive [min,max]; min=0 => empty scenes appear (bare cell, 0 GT)
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

        # ── Filter physikalisch unrealistische Settled-Posen ──────────────────
        # Rotationssymmetrische Teile (Anker_Kurz/Lang, Zahnrad, Ringmagnet)
        # koennen nach Physics-Settle auf einer Endflaeche/Kante balancieren
        # (Zahnrad auf Zaehnen, Anker hochkant). Solche Posen sind real-world
        # so selten, dass sie das GT verzerren und im Demo-Viewer unrealistisch
        # wirken. Wir messen die body-Y-Achse (Laengs-/Rotationsachse) im
        # Welt-Frame: ist deren z-Komponente > 0.5 (Teil mehr aufrecht als
        # liegend), entfernen wir den Prim aus der Szene bevor der Render-Step
        # die Annotatoren-Daten zieht. Resultat: weniger Teile im Bild, dafuer
        # nur stabile Posen.
        ROTSYM_LABELS = ("Anker_Kurz", "Anker_Lang", "Zahnrad", "Ringmagnet")
        # T-119: im Live-Demo-Modus (--force-each-focus) das Zahnrad NICHT upright-
        # entfernen. Zahnrad ist C7 (diskrete Achse → aufrecht pose-bar), anders als
        # die KONTINUIERLICH rotsym. Anker/Ringmagnet (aufrecht echt mehrdeutig). So
        # verschwindet das garantierte Zahnrad nicht, wenn es auf der Kante landet.
        if getattr(a, "force_each_focus", False):
            ROTSYM_LABELS = tuple(l for l in ROTSYM_LABELS if l != "Zahnrad")
        def _is_upright_unstable(prim_path, label):
            if label not in ROTSYM_LABELS:
                return False
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return False
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), float).reshape(4, 4)
            # USD: row-vector convention; transpose -> column-vector (BOP).
            # WICHTIG: die 3x3-Block enthaelt R * OBJECT_SCALE (1e-3, da USD-
            # Geometrie in mm und Szene in m). Wir muessen die Spalte
            # normalisieren, sonst sind alle Komponenten ~0.001 und der
            # Schwellwert 0.5 wird NIE erreicht (Filter greift dann nie).
            T = M.T
            by_raw = T[:3, 1]                      # body-Y axis in world (skaliert)
            mag = float(np.linalg.norm(by_raw))
            if mag == 0:
                return False
            by_world_z = float(by_raw[2]) / mag    # normalisierte z-Komponente
            # |z-component| > 0.5: body-Y points more "up" than horizontal
            # → Teil steht/lehnt aufrecht (= instabil bei rotationssym. Teilen).
            return abs(by_world_z) > 0.5

        def _in_arm_keepout(prim_path):
            """True, wenn das Teil NACH dem Settle im Sperrkreis um die Armbasis
            liegt. Der Abwurf-Test greift nur fuer die Startposition; beim Fallen
            rollen und rutschen die Teile und landen sonst am Sockel (T-190)."""
            if arm_base_xy is None or ARM_CLEAR_R <= 0:
                return False
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return False
            try:
                t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()).ExtractTranslation()
            except Exception:
                return False
            dx = float(t[0]) - arm_base_xy[0]; dy = float(t[1]) - arm_base_xy[1]
            return dx * dx + dy * dy < ARM_CLEAR_R * ARM_CLEAR_R

        in_keepout = [pp for pp in path2label if _in_arm_keepout(pp)]
        for pp in in_keepout:
            try:
                stage.RemovePrim(pp)
            except Exception:
                pass
            path2label.pop(pp, None)
        if in_keepout:
            log(f"  [keep-out] {len(in_keepout)} Teil(e) am Armsockel entfernt")

        unstable = [pp for pp, lbl in path2label.items() if _is_upright_unstable(pp, lbl)]
        for pp in unstable:
            try:
                stage.RemovePrim(pp)
            except Exception:
                pass
            path2label.pop(pp, None)
        if unstable:
            log(f"removed {len(unstable)} instabile Pose(n) (rotsym. Teil aufrecht): {unstable}")
            # Settle nochmal kurz, damit die verbleibenden Teile nicht mit den
            # geloeschten Stuetzen mitfallen oder Sprung-Artefakte zeigen.
            tl.play()
            for _ in range(40): app.update()
            tl.pause()
            for _ in range(3): app.update()

        rep.orchestrator.step(rt_subframes=16)
        data = {k: x.get_data() for k, x in annots.items()}

        # ── save raw isaac bundle ────────────────────────────────────────────
        from PIL import Image
        Image.fromarray(data["rgb"]).save(os.path.join(out_dir, f"rgb_{sidx:04d}.png"))
        dd = data.get("depth"); arr = dd["data"] if isinstance(dd, dict) else dd
        if arr is not None:
            np.save(os.path.join(out_dir, f"depth_{sidx:04d}.npy"), np.asarray(arr, np.float32))
        for key, fn in (("instance_seg", "instance"), ("semantic_seg", "semantic")):
            d = data[key]
            if isinstance(d, dict):
                # keep full id resolution (uint16) — many instances + the arm
                np.save(os.path.join(out_dir, f"{fn}_{sidx:04d}.npy"),
                        np.asarray(d["data"]).astype(np.uint32))
                json.dump(dg.convert_numpy(d.get("info", {})),
                          open(os.path.join(out_dir, f"{fn}_labels_{sidx:04d}.json"), "w"), indent=2)

        # ── bbox_3d: AABB pro Instanz im Camera-Frame (für 3D-Detector / Volume-IoU)
        bb3 = data.get("bbox_3d")
        if isinstance(bb3, dict):
            arr = bb3.get("data")
            info = bb3.get("info", {})
            entries = []
            if arr is not None:
                for row in np.asarray(arr):
                    # Replicator-Schema: (semId, x_min,y_min,z_min, x_max,y_max,z_max, transform[4x4], occlusion)
                    try:
                        sem_id = int(row["semanticId"])
                    except Exception:
                        sem_id = int(row[0]) if hasattr(row, "__getitem__") else -1
                    rec = {"semantic_id": sem_id}
                    for k in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max",
                              "transform", "occlusion_ratio"):
                        try: rec[k] = dg.convert_numpy(row[k])
                        except Exception: pass
                    entries.append(rec)
            json.dump({"boxes": entries, "info": dg.convert_numpy(info)},
                      open(os.path.join(out_dir, f"bbox_3d_{sidx:04d}.json"), "w"), indent=2)


        # ── GT: camera intrinsics + cam c2w + per-instance world transform ───
        gt_raw = {
            "image_id": sidx,
            "width": a.width, "height": a.height,
            "focal_length_mm": float(cam_focal.Get()),
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
                  open(os.path.join(out_dir, f"gt_raw_{sidx:04d}.json"), "w"), indent=2)

        # ── oriented 2D boxes (debug / detector reuse) ───────────────────────
        inst = data["instance_seg"]; seg = data["semantic_seg"]
        obbs = compute_oriented_boxes(inst["data"] if isinstance(inst, dict) else None,
                                      seg["data"] if isinstance(seg, dict) else None,
                                      seg["info"].get("idToLabels", {}) if isinstance(seg, dict) else {},
                                      all_labels, bbox_data=data.get("bbox_2d"))
        json.dump({"boxes": obbs}, open(os.path.join(out_dir, f"obb_2d_{sidx:04d}.json"), "w"), indent=2)

        return len(path2label), len(obbs)

    # ── T-183 SERVE: Isaac bleibt gebootet, rendert je HTTP-POST /render eine Szene.
    # HTTPServer bedient Requests auf dem Serving-(Main-)Thread → Isaac bleibt
    # single-threaded, Renders sind by-construction serialisiert (kein Lock noetig).
    if a.serve:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        _scene = [0]

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/healthz":
                    self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                if self.path != "/render":
                    self.send_response(404); self.end_headers(); return
                try:
                    ln = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(ln) or b"{}")
                    out_dir = req["output"]; n_obj = int(req.get("n_obj", 8))
                    seed = int(req.get("seed", 0))
                    os.makedirs(out_dir, exist_ok=True)
                    t = time.time()
                    # sidx IMMER 0: jeder Request rendert EINE Szene in sein eigenes
                    # frisches out_dir → Dateien heissen *_0000.* (Konverter erwartet
                    # rgb_0000.png). _scene ist nur ein Log-Zaehler.
                    n_parts, n_boxes = render_one(
                        0, out_dir, n_obj, np.random.default_rng(seed))
                    _scene[0] += 1
                    log(f"served #{_scene[0]} -> {out_dir}: {n_parts} parts, "
                        f"{n_boxes} boxes | {time.time()-t:.1f}s")
                    body = json.dumps(
                        {"ok": True, "n_parts": n_parts, "n_boxes": n_boxes}).encode()
                    self.send_response(200)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    body = json.dumps({"ok": False, "error": str(e)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

        srv = HTTPServer(("127.0.0.1", a.serve_port), _H)
        log(f"SERVE ready 127.0.0.1:{a.serve_port} — warm Isaac worker (DONE booting)")
        try:
            srv.serve_forever()
        finally:
            app.close()
        return

    for sidx in range(a.start, a.num_scenes):
        forced_n = (force_counts[(sidx - a.start) % len(force_counts)]
                    if force_counts else None)
        n_parts, n_boxes = render_one(sidx, a.output, forced_n, rng)
        done += 1
        if n_parts == 0 or done % 5 == 0 or done <= 3:
            r = done / max(1e-6, time.time() - t0)
            tag = " [EMPTY 0-GT]" if n_parts == 0 else ""
            log(f"scene {sidx}: {n_parts} parts, {n_boxes} boxes{tag} | "
                f"{r:.2f}/s | ETA {(a.num_scenes - sidx - 1)/max(1e-6, r)/60:.1f}min")

    tl.stop()
    log(f"DONE {done} scenes -> {a.output} | {(time.time()-t0)/60:.1f}min")
    app.close()


if __name__ == "__main__":
    main()
