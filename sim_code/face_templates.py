#!/usr/bin/env python3
"""Render the canonical camera-accurate Face-View template per face.

Reads the registration table ``faces_<part>.json`` (Face N -> R_face) and, for
each face, places the part in that canonical resting pose under the same Isaac
top-down camera that produces the synthetic data, renders, and crops to the
part's tight 2D bounding box. The crop is exactly what an inference snippet looks
like (the bbox cut out of the camera image), so the learned Face-Views match the
data the classifier sees. Writes one template per face + a labelled montage.

    /mnt/data/isaacsim-venv/bin/python -u sim_code/face_templates.py \\
        --part  .../USD-Files/Zahnrad_Typ7.usdz \\
        --faces .../faces_out/faces_Zahnrad_Typ7.json \\
        --out   .../faces_out/templates_Zahnrad_Typ7
"""
import argparse
import json
import os
import time


def log(m):
    print(f"[tmpl {time.strftime('%T')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--faces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=0.001)
    p.add_argument("--cam-h", type=float, default=0.18)
    p.add_argument("--focal", type=float, default=24.0)
    p.add_argument("--snippet", type=int, default=224)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--pad", type=float, default=0.12, help="bbox crop padding (frac)")
    p.add_argument("--no-headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    faces = json.load(open(args.faces))["faces"]
    os.makedirs(args.out, exist_ok=True)

    log(f"booting SimulationApp ... ({len(faces)} faces)")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.no_headless})

    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import carb
    from pxr import Usd, UsdGeom, UsdLux, Gf
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
    g = 1.0
    ground.CreatePointsAttr([(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)])
    ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)
    # ground intentionally has NO semantics, so only the part gets a 2D bbox

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

    cam = UsdGeom.Camera.Define(stage, "/World/TopCam")
    cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, args.cam_h))
    cam.CreateFocalLengthAttr(args.focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))

    for _ in range(20):
        app.update()

    rp = rep.create.render_product("/World/TopCam", (args.res, args.res))
    rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
    bb_a = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    rgb_a.attach([rp]); bb_a.attach([rp])

    def set_pose(prim, R, scale, t):
        M = Gf.Matrix4d(
            scale * R[0][0], scale * R[1][0], scale * R[2][0], 0.0,
            scale * R[0][1], scale * R[1][1], scale * R[2][1], 0.0,
            scale * R[0][2], scale * R[1][2], scale * R[2][2], 0.0,
            t[0], t[1], t[2], 1.0)
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(M)

    tmpls = []
    for f in faces:
        R = np.asarray(f["R"], float).reshape(3, 3)
        for prim in list(stage.Traverse()):
            if prim.GetPath().pathString.startswith("/World/Obj"):
                stage.RemovePrim(prim.GetPath())
        prim = stage.DefinePrim("/World/Obj", "Xform")
        prim.GetReferences().AddReference(args.part)
        dg.add_update_semantics(prim, "part")
        # pass 1: rotation+scale at origin, measure world bbox
        set_pose(prim, R, args.scale, (0.0, 0.0, 0.0))
        for _ in range(3):
            app.update()
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = bbc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        log(f"  {f['name']} world-bbox size={tuple(round(mx[k]-mn[k],4) for k in range(3))}")
        cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        # pass 2: centre under camera, rest on z=0
        set_pose(prim, R, args.scale, (-cx, -cy, -mn[2]))
        for _ in range(8):
            app.update()

        rep.orchestrator.step(rt_subframes=16)
        rgb = np.asarray(rgb_a.get_data())[:, :, :3]
        bb = bb_a.get_data()
        data = bb["data"] if isinstance(bb, dict) else bb
        if data is None or len(data) == 0:
            log(f"  {f['name']}: no bbox, skip"); continue
        row = data[0]
        x0, y0, x1, y1 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
        pw, ph = x1 - x0, y1 - y0
        px, py = int(pw * args.pad), int(ph * args.pad)
        H, W = rgb.shape[:2]
        x0 = max(0, x0 - px); y0 = max(0, y0 - py)
        x1 = min(W, x1 + px); y1 = min(H, y1 + py)
        crop = rgb[y0:y1, x0:x1]

        # square-pad (keep aspect) then resize to the snippet size
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        sq = np.full((side, side, 3), 128, np.uint8)
        sq[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = crop
        im = Image.fromarray(sq).resize((args.snippet, args.snippet))
        fp = os.path.join(args.out, f"template_{f['name'].replace(' ', '')}.png")
        im.save(fp); tmpls.append((f, im))
        log(f"  {f['name']} ({f['tag']}, p={f['prob']*100:.0f}%) bbox={pw}x{ph} -> {fp}")

    # labelled montage
    if tmpls:
        from PIL import ImageDraw
        s = args.snippet; lab = 26
        sheet = Image.new("RGB", (len(tmpls) * s, s + lab), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, (f, im) in enumerate(tmpls):
            sheet.paste(im, (i * s, lab))
            d.text((i * s + 4, 6), f'{f["name"]} {f["tag"]} {f["prob"]*100:.0f}%', fill=(0, 0, 0))
        mp = os.path.join(args.out, "templates_montage.png")
        sheet.save(mp); log(f"montage -> {mp}")

    log(f"done -> {args.out}")
    app.close()


if __name__ == "__main__":
    main()
