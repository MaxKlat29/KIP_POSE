"""
Im folgenden sind die Gitterkonstanten, d.h. der Abstand zwischen zwei benachbarten Mittelpunkten der Steckplätze des “Tray_Anker_Leer” angegeben. Dadurch wird eine relative Navigation im Grid des Trays ermöglicht. In jeweils posistive x- bzw. y-Richtung lauten die Gitterkonstanten:

g_x = 0,03442
g_y = 0,033887

Nutzungsbeispiel: 

Ein Anker_Kurz befindet sich auf einem Steckplatz mit der x-Koordinate 0,70938. Um ihn in den benachbarten Steckplatz (in positive x-Richtung) zu verschieben, addieren wir einfach auf die aktuelle x-Koordinate 1-fach die Gitterkonstante g_x hinzu. Die neue Koordinate ergibt sich somit zu: x_neu = 0,70938 + g_x = 0,7438. 
"""

import numpy as np
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.usd
import omni.replicator.core as rep
import omni.timeline
import omni.kit.app
import asyncio
import os
import json
from PIL import Image
# Isaac Sim 5.x renamed the extension namespace (omni.isaac.* -> isaacsim.*).
# Try the new path first, fall back to the legacy one for older installs.
try:
    from isaacsim.core.utils.semantics import add_update_semantics
except ImportError:  # pragma: no cover - depends on Isaac Sim version
    from omni.isaac.core.utils.semantics import add_update_semantics

# ── ASSET-KONFIGURATION ────────────────────────────────────────
# USD-Assets liegen standardmaessig unter ../data/usd, koennen aber per
# Env-Variable SDG_USD_DIR ueberschrieben werden (z.B. auf der Workstation).
_USD_DIR = os.environ.get(
    "SDG_USD_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "usd"),
)
ASSETS = [
    {"path": os.path.join(_USD_DIR, "Anker_Kurz.usd"), "label": "Anker_Kurz"},
    {"path": os.path.join(_USD_DIR, "Anker_Lang.usd"), "label": "Anker_Lang"},
]

# ── KAMERA-KONFIGURATION ───────────────────────────────────────
ZIVID_CAMERA_PATH = os.environ.get("SDG_CAMERA_PATH", "/World/Zivid")

# ── OUTPUT-KONFIGURATION ───────────────────────────────────────
OUTPUT_DIR   = os.environ.get(
    "SDG_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output"),
)
NUM_RENDERS  = 2
IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720

# ── PHYSIK-SETTLING ────────────────────────────────────────────
PHYSICS_SETTLE_STEPS = 120

# ── SPAWN-MODUS ────────────────────────────────────────────────
# "tray"   → Objekte in Tray-Steckplätzen
# "random" → Objekte fallen zufällig im Bereich
SPAWN_MODE = "tray"

# ── RANDOM SPAWN KONFIGURATION ─────────────────────────────────
SPAWN_BOUNDS = {
    "x": (0.05732, 0.78332),
    "y": (0.04157, 0.53698),
    "z": (0.3,     0.74184),
}

# ── TRAY GRID KONFIGURATION ────────────────────────────────────
_REF_X, _REF_Y, _REF_Z = 0.63989, 0.05617, 0.08009
_REF_COL, _REF_ROW     = 2, 1

ANKER_TRAY = {
    "origin_x": _REF_X - _REF_COL * 0.03442,
    "origin_y": _REF_Y - _REF_ROW * 0.033887,
    "z_tray":   _REF_Z,
    "z_spawn":  _REF_Z + 0.08,
    "g_x":      0.03442,
    "g_y":      0.033887,
    "num_x":    7,
    "num_y":    5,
    "excluded_slots": {
        (0, 0), (6, 0),
        (0, 4), (6, 4),
    },
}

NUM_ANKERS_IN_TRAY = 5

# ── ROTATIONS-KONFIGURATION ────────────────────────────────────
TRAY_ROTATION = {
    # Feste Basis-Orientierung wie in der Szene (90, 0, 0)
    "base_rx": 90.0,
    "base_ry": 0.0,
    # Zufällige Z-Rotation damit verschiedene Ausrichtungen entstehen
    "random_rz": True,
    "rz_range":  (0.0, 360.0),
}

RANDOM_ROTATION = {
    "random": True,
    "x": (0.0, 360.0),
    "y": (0.0, 360.0),
    "z": (0.0, 360.0),
}

# ── OBJEKT KONFIGURATION ───────────────────────────────────────
NUM_OBJECTS  = 5
OBJECT_SCALE = (0.001, 0.001, 0.001)
OBJECT_SIZE  = 0.03
RANDOM_SEED  = 42
MAX_SPAWN_ATTEMPTS = 100

# ── LIGHTING RANDOMIZATION ────────────────────────────────────
# Controls the Replicator-driven light randomization added on branch marc.
# Set LIGHT_RANDOMIZE = False to keep the scene lights unchanged (legacy behaviour).
LIGHT_RANDOMIZE = True
LIGHT_CONFIG = {
    # Number of randomized sphere lights added per render.
    "count": 3,
    # Intensity range (lux).  Metallic anchor parts are sensitive to this.
    "intensity": (200_000, 2_000_000),
    # Colour temperature range expressed as (R,G,B) float tuples.
    "color_min": (0.8, 0.65, 0.45),   # warm / tungsten tint
    "color_max": (1.0, 1.00, 1.00),   # neutral white
    # Sphere radius range (m) – larger = softer shadows.
    "radius":    (0.05, 0.25),
    # Position bounding box (m) around the tray area, above it.
    "pos_min": (-0.1, -0.1,  0.4),
    "pos_max": ( 0.9,  0.6,  1.2),
}

# ── RIGID BODY KONFIGURATION ───────────────────────────────────
RIGID_BODY = {
    "enabled":          True,
    "mass":             1.0,
    "linear_damping":   0.1,
    "angular_damping":  0.1,
    "initial_velocity": (0.0, 0.0, 0.0),
}

# ── KOLLISIONS-KONFIGURATION ───────────────────────────────────
COLLISION = {
    "enabled":        True,
    "approximation":  "convexHull",
    "contact_offset": 0.005,
    "rest_offset":    0.0,
}


# ══════════════════════════════════════════════════════════════
#  GRID HELPER
# ══════════════════════════════════════════════════════════════

def get_valid_tray_slots(tray_cfg):
    slots = []
    for row in range(tray_cfg["num_y"]):
        for col in range(tray_cfg["num_x"]):
            if (col, row) in tray_cfg["excluded_slots"]:
                continue
            x = tray_cfg["origin_x"] + col * tray_cfg["g_x"]
            y = tray_cfg["origin_y"] + row * tray_cfg["g_y"]
            z = tray_cfg["z_spawn"]
            slots.append((x, y, z, col, row))
    return slots


def print_tray_grid(tray_cfg, slot_assignments):
    """slot_assignments: dict {(col,row): asset_label}"""
    print("\n  Tray Grid (K=Kurz, L=Lang, O=frei, .=kein Slot):")
    for row in range(tray_cfg["num_y"]):
        line = "  "
        for col in range(tray_cfg["num_x"]):
            if (col, row) in tray_cfg["excluded_slots"]:
                line += ".  "
            elif (col, row) in slot_assignments:
                label = slot_assignments[(col, row)]
                line += ("K  " if "Kurz" in label else "L  ")
            else:
                line += "O  "
        print(line)
    print()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def euler_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    roll  = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw   = np.radians(yaw_deg)
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def cleanup_spawned_objects(stage):
    to_remove = []
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/SpawnedObject_"):
            to_remove.append(prim.GetPath())
    if to_remove:
        print(f"  [Cleanup] Lösche {len(to_remove)} vorherige Objekte...")
        for path in to_remove:
            stage.RemovePrim(path)


def get_prim_position(prim):
    try:
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        translate_op = xformable.GetTranslateOp()
        if translate_op:
            pos = translate_op.Get()
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    except:
        pass
    return None


def get_prim_size(prim):
    if prim.IsA(UsdGeom.Cube):
        size_attr = UsdGeom.Cube(prim).GetSizeAttr()
        if size_attr:
            return float(size_attr.Get()) / 2.0
    if prim.IsA(UsdGeom.Sphere):
        radius_attr = UsdGeom.Sphere(prim).GetRadiusAttr()
        if radius_attr:
            return float(radius_attr.Get())
    return OBJECT_SIZE


def get_existing_objects(stage):
    objects = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path in ["/", "/World"] or path.startswith("/World/SpawnedObject"):
            continue
        pos = get_prim_position(prim)
        if pos is None:
            continue
        objects.append({"path": path, "pos": pos, "size": get_prim_size(prim)})
    return objects


def apply_rigid_body(prim):
    if not RIGID_BODY["enabled"]:
        return False
    try:
        rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        vx, vy, vz = RIGID_BODY["initial_velocity"]
        rb_api.CreateVelocityAttr().Set(Gf.Vec3f(vx, vy, vz))
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(RIGID_BODY["mass"])
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.CreateLinearDampingAttr().Set(RIGID_BODY["linear_damping"])
        physx_rb.CreateAngularDampingAttr().Set(RIGID_BODY["angular_damping"])
        return True
    except Exception as e:
        print(f"  ⚠ RigidBody Fehler: {e}")
        return False


def apply_collision(prim, stage):
    if not COLLISION["enabled"]:
        return 0
    mesh_count = 0
    def apply_to_mesh(p):
        nonlocal mesh_count
        if p.IsA(UsdGeom.Mesh):
            try:
                UsdPhysics.CollisionAPI.Apply(p)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(p)
                mesh_api.CreateApproximationAttr().Set(COLLISION["approximation"])
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(p)
                physx_col.CreateContactOffsetAttr().Set(COLLISION["contact_offset"])
                physx_col.CreateRestOffsetAttr().Set(COLLISION["rest_offset"])
                mesh_count += 1
            except Exception as e:
                print(f"  ⚠ Collision Fehler: {e}")
        for child in p.GetChildren():
            apply_to_mesh(child)
    apply_to_mesh(prim)
    return mesh_count


def spawn_single_object(stage, prim_path, asset, x, y, z, rx, ry, rz):
    """Spawnt ein Objekt mit gegebenem Asset-Dict {path, label}"""
    prim = stage.DefinePrim(prim_path, "Xform")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    xform.AddOrientOp().Set(euler_to_quaternion(rx, ry, rz))
    xform.AddScaleOp().Set(Gf.Vec3f(*OBJECT_SCALE))

    prim.GetReferences().AddReference(asset["path"])
    add_update_semantics(prim, asset["label"])
    apply_rigid_body(prim)
    apply_collision(prim, stage)
    return prim


# ══════════════════════════════════════════════════════════════
#  SPAWN MODES
# ══════════════════════════════════════════════════════════════

def spawn_tray_mode(stage, rng):
    """Spawnt zufällig Anker_Kurz & Anker_Lang in Tray-Slots"""
    all_slots    = get_valid_tray_slots(ANKER_TRAY)
    num_to_place = min(NUM_ANKERS_IN_TRAY, len(all_slots))

    chosen_indices = rng.choice(len(all_slots), size=num_to_place, replace=False)
    chosen_slots   = [all_slots[i] for i in chosen_indices]

    # Zufällig Asset-Typ pro Slot zuweisen
    asset_assignments  = rng.integers(0, len(ASSETS), size=num_to_place)
    slot_assignments   = {}
    spawned_paths      = []
    asset_counts       = {a["label"]: 0 for a in ASSETS}

    for i, (x, y, z, col, row) in enumerate(chosen_slots):
        asset = ASSETS[asset_assignments[i]]

        # Feste Basis-Orientierung (90, 0, 0) + zufällige Z-Rotation
        rx = TRAY_ROTATION["base_rx"]
        ry = TRAY_ROTATION["base_ry"]
        rz = float(rng.uniform(*TRAY_ROTATION["rz_range"])) \
             if TRAY_ROTATION["random_rz"] else 0.0

        prim_path = f"/World/SpawnedObject_{i:03d}"
        spawn_single_object(stage, prim_path, asset, x, y, z, rx, ry, rz)

        slot_assignments[(col, row)] = asset["label"]
        spawned_paths.append(prim_path)
        asset_counts[asset["label"]] += 1

        print(f"  ✓ {asset['label']:<12} → Slot ({col},{row}) "
              f"@ ({x:.5f}, {y:.5f}, {z:.5f}) "
              f"| Rot({rx:.0f}°, {ry:.0f}°, {rz:.0f}°)")

    print_tray_grid(ANKER_TRAY, slot_assignments)
    print(f"  Verteilung: " +
          " | ".join(f"{k}: {v}" for k, v in asset_counts.items()))

    return spawned_paths


def spawn_random_mode(stage, rng):
    """Spawnt zufällig Anker_Kurz & Anker_Lang im freien Bereich"""
    existing_objects = get_existing_objects(stage)
    spawned_paths    = []
    spawned = 0
    attempt = 0
    max_total_attempts = NUM_OBJECTS * MAX_SPAWN_ATTEMPTS * 2

    while spawned < NUM_OBJECTS and attempt < max_total_attempts:
        attempt += 1
        x = rng.uniform(*SPAWN_BOUNDS["x"])
        y = rng.uniform(*SPAWN_BOUNDS["y"])
        z = rng.uniform(*SPAWN_BOUNDS["z"])

        half = OBJECT_SIZE / 2.0 + 0.02
        new_min = (x - half, y - half, z - half)
        new_max = (x + half, y + half, z + half)
        overlap = False
        for obj in existing_objects:
            s  = obj["size"] + 0.02
            om = tuple(p - s for p in obj["pos"])
            ox = tuple(p + s for p in obj["pos"])
            if (new_max[0]>om[0] and new_min[0]<ox[0] and
                new_max[1]>om[1] and new_min[1]<ox[1] and
                new_max[2]>om[2] and new_min[2]<ox[2]):
                overlap = True
                break
        if overlap:
            continue

        asset = ASSETS[rng.integers(0, len(ASSETS))]
        rx = float(rng.uniform(*RANDOM_ROTATION["x"]))
        ry = float(rng.uniform(*RANDOM_ROTATION["y"]))
        rz = float(rng.uniform(*RANDOM_ROTATION["z"]))

        prim_path = f"/World/SpawnedObject_{spawned:03d}"
        spawn_single_object(stage, prim_path, asset, x, y, z, rx, ry, rz)
        existing_objects.append({"path": prim_path, "pos": (x,y,z), "size": OBJECT_SIZE/2})
        spawned_paths.append(prim_path)

        print(f"  ✓ {asset['label']:<12} @ ({x:.4f}, {y:.4f}, {z:.4f}) "
              f"| Rot({rx:.0f}°,{ry:.0f}°,{rz:.0f}°)")
        spawned += 1

    return spawned_paths


# ══════════════════════════════════════════════════════════════
#  SPEICHERN
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  LIGHTING HELPERS
# ══════════════════════════════════════════════════════════════

def setup_randomized_lights():
    """Register a Replicator randomizer that re-rolls N sphere lights on every
    rep.orchestrator.step() call.  Returns the light group prim so the caller
    can detach / remove it after the run.

    Must be called *after* SimulationApp is up and rep is imported.
    Respects LIGHT_RANDOMIZE — is a no-op when disabled.
    """
    if not LIGHT_RANDOMIZE:
        return None

    cfg = LIGHT_CONFIG
    lights = rep.create.light(
        light_type="Sphere",
        count=cfg["count"],
        intensity=rep.distribution.uniform(*cfg["intensity"]),
        color=rep.distribution.uniform(cfg["color_min"], cfg["color_max"]),
        position=rep.distribution.uniform(cfg["pos_min"], cfg["pos_max"]),
        scale=rep.distribution.uniform(*cfg["radius"]),
    )

    # Register so Replicator re-randomizes on every orchestrator step.
    with lights:
        rep.modify.attribute(
            "intensity",
            rep.distribution.uniform(*cfg["intensity"]),
        )
        rep.modify.attribute(
            "color",
            rep.distribution.uniform(cfg["color_min"], cfg["color_max"]),
        )
        rep.modify.pose(
            position=rep.distribution.uniform(cfg["pos_min"], cfg["pos_max"]),
        )

    print(f"  [Lights] {cfg['count']} randomized sphere lights registered "
          f"(intensity {cfg['intensity'][0]/1e3:.0f}k–{cfg['intensity'][1]/1e3:.0f}k lux, "
          f"LIGHT_RANDOMIZE={LIGHT_RANDOMIZE})")
    return lights


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):return [convert_numpy(i) for i in obj]
    return obj


def save_output(data, render_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if data["rgb"] is not None:
        path = os.path.join(output_dir, f"rgb_{render_idx:04d}.png")
        Image.fromarray(data["rgb"]).save(path)
        print(f"     rgb             → {path}")

    if data["bbox_2d"] is not None:
        path = os.path.join(output_dir, f"bbox_2d_{render_idx:04d}.json")
        with open(path, "w") as f:
            json.dump(convert_numpy(data["bbox_2d"]), f, indent=2)
        print(f"     bbox_2d         → {path}")

    if data["semantic_seg"] is not None:
        path = os.path.join(output_dir, f"semantic_{render_idx:04d}.png")
        Image.fromarray(data["semantic_seg"]["data"].astype(np.uint8)).save(path)
        mapping_path = os.path.join(output_dir, f"semantic_labels_{render_idx:04d}.json")
        with open(mapping_path, "w") as f:
            json.dump(convert_numpy(data["semantic_seg"]["info"]), f, indent=2)
        print(f"     semantic        → {path}")

    if data["instance_seg"] is not None:
        path = os.path.join(output_dir, f"instance_{render_idx:04d}.png")
        Image.fromarray(data["instance_seg"]["data"].astype(np.uint8)).save(path)
        print(f"     instance        → {path}")

    if data["depth"] is not None:
        path = os.path.join(output_dir, f"depth_{render_idx:04d}.png")
        depth = data["depth"]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        Image.fromarray((depth_norm * 65535).astype(np.uint16)).save(path)
        print(f"     depth           → {path}")

    # ── 3-D bounding boxes (world-space AABB per instance) ──────
    if data.get("bbox_3d") is not None:
        path = os.path.join(output_dir, f"bbox_3d_{render_idx:04d}.json")
        with open(path, "w") as f:
            json.dump(convert_numpy(data["bbox_3d"]), f, indent=2)
        print(f"     bbox_3d         → {path}")

    # ── 6-DoF object poses (4×4 transform matrix per instance) ──
    # Format: { "data": [ { "semanticId": int,
    #                        "localToWorldTransform": [[...], ...] }, ... ],
    #           "info": { "idToLabels": { "1": {"class": "Anker_Kurz"}, ... } } }
    if data.get("obj_pose") is not None:
        path = os.path.join(output_dir, f"pose_{render_idx:04d}.json")
        with open(path, "w") as f:
            json.dump(convert_numpy(data["obj_pose"]), f, indent=2)
        print(f"     obj_pose        → {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

async def run_data_generation():
    stage    = omni.usd.get_context().get_stage()
    rng      = np.random.default_rng(RANDOM_SEED)
    timeline = omni.timeline.get_timeline_interface()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SDG - Synthetic Data Generation")
    print(f"  Modus   : {SPAWN_MODE.upper()}")
    print(f"  Assets  : {[a['label'] for a in ASSETS]}")
    print(f"  Renders : {NUM_RENDERS}")
    print(f"  Settling: {PHYSICS_SETTLE_STEPS} Frames")
    print(f"  Output  : {OUTPUT_DIR}")
    if SPAWN_MODE == "tray":
        print(f"  Rotation: base({TRAY_ROTATION['base_rx']:.0f}°,"
              f"{TRAY_ROTATION['base_ry']:.0f}°) + random Z")
        print(f"  Grid    : {ANKER_TRAY['num_x']}x{ANKER_TRAY['num_y']} "
              f"Origin({ANKER_TRAY['origin_x']:.5f}, {ANKER_TRAY['origin_y']:.5f})")
    print("=" * 60)

    # Kamera validieren
    camera_prim = stage.GetPrimAtPath(ZIVID_CAMERA_PATH)
    if not camera_prim.IsValid():
        print(f"⚠ Kamera nicht gefunden: {ZIVID_CAMERA_PATH}")
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                print(f"    → {prim.GetPath().pathString}")
        return

    # Lighting randomization (Replicator-driven sphere lights)
    setup_randomized_lights()

    # Annotators
    render_product = rep.create.render_product(
        ZIVID_CAMERA_PATH, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT)
    )
    rgb_annot      = rep.AnnotatorRegistry.get_annotator("rgb")
    bbox_annot     = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    semantic_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    instance_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
    depth_annot    = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    # 6-DoF pose labels — critical for pose estimation training
    bbox_3d_annot  = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    pose_annot     = rep.AnnotatorRegistry.get_annotator("pose")

    all_annots = [rgb_annot, bbox_annot, semantic_annot, instance_annot,
                  depth_annot, bbox_3d_annot, pose_annot]
    for annot in all_annots:
        annot.attach([render_product])

    print(f"[Setup] Annotators bereit (inkl. bbox_3d + pose)\n")

    for render_idx in range(NUM_RENDERS):
        print(f"\n{'─' * 50}")
        print(f"[Render {render_idx + 1}/{NUM_RENDERS}]")

        timeline.stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        cleanup_spawned_objects(stage)
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        # Spawn
        if SPAWN_MODE == "tray":
            spawned_paths = spawn_tray_mode(stage, rng)
        else:
            spawned_paths = spawn_random_mode(stage, rng)

        print(f"  → {len(spawned_paths)} Objekte gespawnt")

        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        # Simulation
        timeline.play()
        for step in range(PHYSICS_SETTLE_STEPS):
            await omni.kit.app.get_app().next_update_async()
            if step % 40 == 0:
                print(f"     Settling {step}/{PHYSICS_SETTLE_STEPS}...")

        timeline.pause()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        print(f"  → Capture...")

        await rep.orchestrator.step_async(rt_subframes=8, delta_time=0.0)

        data = {
            "rgb":          rgb_annot.get_data(),
            "bbox_2d":      bbox_annot.get_data(),
            "semantic_seg": semantic_annot.get_data(),
            "instance_seg": instance_annot.get_data(),
            "depth":        depth_annot.get_data(),
            "bbox_3d":      bbox_3d_annot.get_data(),
            "obj_pose":     pose_annot.get_data(),
        }

        print(f"  → Speichere:")
        save_output(data, render_idx, OUTPUT_DIR)

    timeline.stop()
    for annot in all_annots:
        annot.detach()

    print(f"\n{'=' * 60}")
    print(f"✅ Fertig! {NUM_RENDERS} Datensätze generiert")
    print(f"   Labels : {[a['label'] for a in ASSETS]}")
    print(f"   Output : {OUTPUT_DIR}")
    print(f"{'=' * 60}")


# Auto-run when pasted/executed directly in the Isaac Sim Script Editor.
# Guarded so the standalone runner (run_sdg.py) can import the helpers above
# without triggering a second generation pass.
if __name__ == "__main__":
    asyncio.ensure_future(run_data_generation())
