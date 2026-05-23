#!/usr/bin/env python3
"""P5 — Eval scene generator WITH ground-truth 6D poses (camera-consistent).

Renders multi-part scenes from a FIXED top-down-ish camera (_DRCam, same family
the detector was trained on) over the cart work-surface (P1 static colliders +
calibrated spawn region). Reads back each settled part's 6D pose (R column-conv
world=R@body, t) + label. Writes per scene:
    rgb_<idx>.png  depth_<idx>.npy  eval_gt_<idx>.json
eval_gt = {table_z, camera:{focal_mm,aperture_mm,render_w,render_h,cam_t,R_wc},
           instances:[{instance_id, part, R(9), t(3), bbox_2d, min_z}]}.
The recorded camera params let the pipeline backproject metrically-consistently.
"""
import argparse, json, os, sys, time
def log(m): print(f"[eval-gt {time.strftime('%H:%M:%S')}] {m}", flush=True)
PART_FILES = [
    ("Anker_Kurz","Anker_Kurz.usd"), ("Anker_Lang","Anker_Lang.usd"),
    ("Buerstenhalter_2polig","Buerstenhalter_2polig.usd"),
    ("Getriebegehaeuse_typ4","Getriebegehaeuse_typ4.usdz"),
    ("Zahnrad","Zahnrad_Typ7.usdz"), ("Ringmagnet","Ringmagnet.usd"),
]
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--scene", required=True); p.add_argument("--usd-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-scenes", type=int, default=6)
    p.add_argument("--min-obj", type=int, default=5); p.add_argument("--max-obj", type=int, default=9)
    p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--table-z", type=float, default=-0.007)
    p.add_argument("--settle", type=int, default=200)
    return p.parse_args()
def main():
    a=parse_args()
    os.environ["OMNI_KIT_ACCEPT_EULA"]="YES"; os.environ["SDG_USD_DIR"]=a.usd_dir
    os.environ["SDG_OUTPUT_DIR"]=a.output
    log("booting SimulationApp ..."); from isaacsim import SimulationApp
    app=SimulationApp({"headless":True}); log("ready")
    import numpy as np, omni.usd, omni.replicator.core as rep, omni.timeline, carb
    from pxr import Usd, UsdGeom, UsdLux, Gf, UsdPhysics
    s=carb.settings.get_settings(); s.set("/app/asyncRendering",False); s.set("/omni/replicator/asyncRendering",False)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datagenerationscript as dg
    from run_scene import compute_oriented_boxes
    U=a.usd_dir; dg.ASSETS=[{"path":os.path.join(U,fn),"label":lbl} for lbl,fn in PART_FILES]
    label_by_base={os.path.basename(fn):lbl for lbl,fn in PART_FILES}
    all_labels={lbl.lower() for lbl,_ in PART_FILES}
    dg.PHYSICS_SETTLE_STEPS=a.settle
    dg.SPAWN_BOUNDS={"x":(0.30,0.62),"y":(0.07,0.47),"z":(0.30,0.55)}

    ctx=omni.usd.get_context(); _r=ctx.open_stage(a.scene)
    ok=_r[0] if isinstance(_r,(tuple,list)) else _r
    if not ok: log("FAILED open"); app.close(); sys.exit(1)
    for _ in range(80): app.update()
    stage=ctx.get_stage()
    for root in ["/World/Basiswagen","/World/Basiswagen_01","/World/Basiswagen_02","/World/Anker_Tray","/World/Poltopf_Tray"]:
        rp_=stage.GetPrimAtPath(root)
        if not rp_.IsValid(): continue
        for prim in Usd.PrimRange(rp_):
            if prim.IsA(UsdGeom.Mesh):
                try: UsdPhysics.CollisionAPI.Apply(prim); UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
                except Exception: pass
    # P0 TOP-DOWN: the LARA5 robot arm sits directly over the tray and occludes a
    # straight-down view -> hide it (+ duplicate carts) so the camera sees the parts.
    for occ in ["/World/NEURA_LARA5_Pose_Zivid_Detection",
                "/World/Basiswagen_01","/World/Basiswagen_02"]:
        op=stage.GetPrimAtPath(occ)
        if op.IsValid():
            try: UsdGeom.Imageable(op).MakeInvisible()
            except Exception: pass
    UsdLux.DomeLight.Define(stage,"/World/_DRDome").CreateIntensityAttr(650.0)
    ps=UsdPhysics.Scene.Define(stage,"/World/_PhysicsScene"); ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1)); ps.CreateGravityMagnitudeAttr(9.81)
    gp=UsdGeom.Cube.Define(stage,"/World/_PhysGround"); gp.GetSizeAttr().Set(1.0)
    gx=UsdGeom.Xformable(gp); gx.ClearXformOpOrder(); gx.AddTranslateOp().Set(Gf.Vec3d(0.5,0.1,a.table_z-0.01)); gx.AddScaleOp().Set(Gf.Vec3f(4,4,0.02))
    UsdPhysics.CollisionAPI.Apply(gp.GetPrim()); UsdGeom.Imageable(gp.GetPrim()).MakeInvisible()
    for _ in range(5): app.update()

    # FIXED _DRCam over the tray (no jitter) — record its pose for the pipeline.
    FOCAL=18.0; APERTURE=20.955
    cam=UsdGeom.Camera.Define(stage,"/World/_DRCam"); cam_xf=UsdGeom.Xformable(cam.GetPrim())
    cam.CreateFocalLengthAttr(FOCAL); cam.CreateHorizontalApertureAttr(APERTURE); cam.CreateVerticalApertureAttr(APERTURE*a.height/a.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01,100.0)); cam_path="/World/_DRCam"
    # TRUE TOP-DOWN: camera straight above the tray centre looking straight down (-Z).
    # up=(0,1,0) so world +Y maps to image-up -> matches the top-down template-bank
    # convention (build_template_bank rasterize_topdown: +Y up). This is the P0 fix:
    # the detector + template banks are top-down, so the eval render must be too.
    cx=0.5*(dg.SPAWN_BOUNDS["x"][0]+dg.SPAWN_BOUNDS["x"][1])
    cy=0.5*(dg.SPAWN_BOUNDS["y"][0]+dg.SPAWN_BOUNDS["y"][1])
    eye=Gf.Vec3d(cx,cy,0.95); look=Gf.Vec3d(cx,cy,a.table_z)
    view=Gf.Matrix4d().SetLookAt(eye,look,Gf.Vec3d(0,1,0))
    cam_xf.ClearXformOpOrder(); cam_xf.AddTransformOp().Set(view.GetInverse())
    for _ in range(10): app.update()
    # camera world transform -> R_wc (rows) + t for the pipeline backprojection
    M=np.array(UsdGeom.Xformable(cam.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).reshape(4,4)
    cam_t=[float(M[3,0]),float(M[3,1]),float(M[3,2])]
    Rwc=M[:3,:3]; Rwc=Rwc/np.linalg.norm(Rwc,axis=1,keepdims=True)
    cam_params={"focal_mm":FOCAL,"aperture_h_mm":APERTURE,"aperture_v_mm":float(APERTURE*a.height/a.width),
                "render_w":a.width,"render_h":a.height,"cam_t":cam_t,"R_wc":[[float(v) for v in r] for r in Rwc.tolist()]}

    os.makedirs(a.output,exist_ok=True)
    rp=rep.create.render_product(cam_path,(a.width,a.height))
    annots={"rgb":rep.AnnotatorRegistry.get_annotator("rgb"),
            "bbox_2d":rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
            "semantic_seg":rep.AnnotatorRegistry.get_annotator("semantic_segmentation"),
            "instance_seg":rep.AnnotatorRegistry.get_annotator("instance_segmentation"),
            "depth":rep.AnnotatorRegistry.get_annotator("distance_to_camera")}
    for x in annots.values(): x.attach([rp])
    tl=omni.timeline.get_timeline_interface(); rng=np.random.default_rng(a.seed)

    def readback(prim):
        Mm=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).reshape(4,4)
        rows=Mm[:3,:3]; rows=rows/np.linalg.norm(rows,axis=1,keepdims=True)
        return rows.T, Mm[3,:3]

    def part_label(prim):
        # resolve via the prim's authored references (composition arcs)
        try:
            for spec in prim.GetPrimStack():
                for ref in (spec.referenceList.prependedItems if spec.referenceList else []):
                    base=os.path.basename(ref.assetPath)
                    if base in label_by_base: return label_by_base[base]
        except Exception: pass
        return None

    # part_meta (prim-origin -> centroid offset, body frame, m): the 2D tight bbox
    # encloses the visible GEOMETRY (~centroid), not the often far-away prim origin
    # (Getriebe: 26cm). Project the CENTROID, not t, to assign the matching bbox.
    _PMETA = {}
    for _cand in (os.path.join(a.usd_dir, "..", "part_meta.json"),
                  "/mnt/data/pose_eval/project/models/part_meta.json",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "part_meta.json")):
        if os.path.exists(_cand):
            try: _PMETA = json.load(open(_cand)); break
            except Exception: pass

    def _centroid_world(t, R, part):
        m = _PMETA.get(part)
        if not m or "centroid_offset_m" not in m: return np.asarray(t, float)
        off = np.asarray(m["centroid_offset_m"], float)
        return np.asarray(t, float) + np.asarray(R, float).reshape(3, 3) @ off

    def _project(pw):
        fx = cam_params["focal_mm"]/cam_params["aperture_h_mm"]*cam_params["render_w"]
        fy = cam_params["focal_mm"]/cam_params["aperture_v_mm"]*cam_params["render_h"]
        cx, cy = cam_params["render_w"]/2.0, cam_params["render_h"]/2.0
        rel = np.asarray(pw, float) - np.asarray(cam_params["cam_t"], float)
        d = np.asarray(cam_params["R_wc"], float) @ rel
        if abs(d[2]) < 1e-9: return None
        return (cx + fx*(d[0]/-d[2]), cy - fy*(d[1]/-d[2]))

    for sidx in range(a.num_scenes):
        tl.stop()
        for _ in range(5): app.update()
        dg.cleanup_spawned_objects(stage)
        for _ in range(5): app.update()
        dg.NUM_OBJECTS=int(rng.integers(a.min_obj,a.max_obj+1))
        dg.spawn_random_mode(stage,rng)
        for _ in range(10): app.update()
        tl.play()
        for _ in range(a.settle): app.update()
        tl.pause()
        for _ in range(5): app.update()
        rep.orchestrator.step(rt_subframes=16); data={k:x.get_data() for k,x in annots.items()}
        Image=_Im
        rgb=data["rgb"]; Image.fromarray(rgb).convert("RGB").save(os.path.join(a.output,f"rgb_{sidx:04d}.png"))
        try:
            dd=data.get("depth"); arr=dd["data"] if isinstance(dd,dict) else dd
            if arr is not None: np.save(os.path.join(a.output,f"depth_{sidx:04d}.npy"), np.asarray(arr,np.float32))
        except Exception: pass
        # 2D bbox per class from tight annotator
        bb=data.get("bbox_2d"); boxes_by_class={}
        if isinstance(bb,dict):
            rows=bb["data"]; binfo=bb.get("info",{}).get("idToLabels",{})
            for row in rows:
                try:
                    cls=(binfo.get(str(int(row[0])),{}) or {}).get("class","").lower()
                    x0,x1=sorted((int(row[1]),int(row[3]))); y0,y1=sorted((int(row[2]),int(row[4])))
                    if (x1-x0)>=4 and (y1-y0)>=4: boxes_by_class.setdefault(cls,[]).append([x0,y0,x1,y1])
                except Exception: pass
        # collect settled instances first, then assign bboxes by PROJECTED CENTROID
        # nearest-match (greedy per class). The old `cands.pop(0)` assigned bboxes in
        # arbitrary order -> instances of the same class got each other's bbox, which
        # made the eval backprojection ray miss by up to 680mm (a GT-data bug, not a
        # model error). Matching by projection guarantees bbox-center ~ pose-projection.
        raw=[]
        for prim in stage.Traverse():
            pp=prim.GetPath().pathString
            if not (pp.startswith("/World/SpawnedObject_") and prim.GetParent().GetPath().pathString=="/World"): continue
            lbl=part_label(prim)
            if lbl is None: continue
            R,t=readback(prim)
            if float(t[2])<a.table_z-0.05: continue  # floor-faller, skip
            raw.append({"part":lbl,"R":R,"t":t})
        used={cls:[False]*len(v) for cls,v in boxes_by_class.items()}
        insts=[]; iid=0
        for it in raw:
            lbl=it["part"]; R=it["R"]; t=it["t"]
            cands=boxes_by_class.get(lbl.lower(),[])
            bbox=None
            pr=_project(_centroid_world(t,R,lbl))
            if cands and pr is not None:
                best,bd=None,1e18
                for k,b in enumerate(cands):
                    if used[lbl.lower()][k]: continue
                    bc=((b[0]+b[2])/2.0,(b[1]+b[3])/2.0)
                    dd=(bc[0]-pr[0])**2+(bc[1]-pr[1])**2
                    if dd<bd: bd,best=dd,k
                # only assign when the bbox center is actually near the projection
                # (<=120px); else it is an annotator ghost -> instance gets no bbox
                # (drops out of eval matching instead of counting as a huge error).
                if best is not None and bd<=(120.0**2):
                    bbox=cands[best]; used[lbl.lower()][best]=True
            insts.append({"instance_id":iid,"part":lbl,"R":[round(float(v),6) for v in R.flatten()],
                          "t":[round(float(v),6) for v in t],"bbox_2d":bbox,"min_z":float(t[2])})
            iid+=1
        json.dump({"table_z":a.table_z,"camera_params":cam_params,"instances":insts},
                  open(os.path.join(a.output,f"eval_gt_{sidx:04d}.json"),"w"), indent=2)
        log(f"scene {sidx}: {len(insts)} GT instances ({sum(1 for i in insts if i['bbox_2d'])} with bbox)")
    tl.stop(); log(f"DONE {a.num_scenes} eval scenes -> {a.output}"); app.close()
if __name__=="__main__": main()
