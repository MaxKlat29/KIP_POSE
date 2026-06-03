#!/usr/bin/env python3
"""export_cell_glb.py — GST_Scene.usd (4433 meshes, 401 mats) → hochwertiges GLB
für den Three.js-Viewer. Ersetzt das alte 4-mesh-decimated cell.glb.

Was es macht:
  • iteriert ALLE UsdGeom.Mesh-Prims, transformiert ihre Punkte in den Welt-Frame
    (ComputeLocalToWorldTransform), trianguliert die Faces
  • zieht pro Mesh die DIFFUSE-Farbe (UsdShade Material → diffuseColor, sonst
    displayColor-Primvar, sonst Default-Grau) → echte Farben statt verloren
  • gruppiert Dreiecke nach (quantisierter) Farbe → 1 trimesh.Mesh pro Farbe
    (statt 4433 Einzel-Meshes → weniger Draw-Calls im Browser)
  • optional Decimation pro Farb-Mesh wenn target-faces gesetzt
  • DECAL-SCHUTZ (T-106): feine flache Label-/Logo-Meshes (wbk/KIT/Warn-Schilder)
    werden erkannt (dünne Platte: kleinste Bbox-Dim < --decal-thin-mm, breit
    > --decal-broad-mm) und in EIGENE, NIE dezimierte Gruppen gelegt. Das Tri-
    Budget (--max-total-tris) gilt nur fürs Bulk → kleine Datei + scharfe Decals.
  • exportiert EIN GLB mit PBR-baseColor pro Gruppe

CPU-only — kein GPU, läuft gefahrlos parallel zum GDRNPP-Training.

Einheiten: USD in mm → ×0.001 auf Meter. Z-up bleibt (matcht pipeline-Frame).

    /mnt/data/faces-venv/bin/python box_src/export_cell_glb.py \
        --usd /mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd \
        --out /mnt/data/kip_pose/project/frontend/assets/cell_hq.glb \
        --scale 0.001
"""
import argparse, time, sys
import numpy as np


def log(m):
    print(f"[cell-glb {time.strftime('%T')}] {m}", flush=True)


def triangulate(counts, indices):
    """fan-triangulate polygon faces -> (M,3) int array."""
    tris = []
    off = 0
    for c in counts:
        if c == 3:
            tris.append((indices[off], indices[off + 1], indices[off + 2]))
        elif c == 4:
            a, b, cc, d = indices[off:off + 4]
            tris.append((a, b, cc)); tris.append((a, cc, d))
        elif c > 4:
            for k in range(1, c - 1):
                tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), np.int64)


def main():
    from pxr import Usd, UsdGeom, UsdShade, Gf
    import trimesh

    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.001)
    ap.add_argument("--max-total-tris", type=int, default=2_500_000,
                    help="globales Tri-Budget; Gruppen werden proportional dezimiert (0=voll)")
    # DECAL-SCHUTZ (T-106): feine flache Label-/Logo-Meshes (wbk/KIT/Warn-Schilder
    # auf den Zell-Türen) sind dünne Platten-Geometrie. Die proportionale
    # Farbgruppen-Dezimierung zerschießt ihre filigranen Buchstaben-/Logo-Kanten
    # (jagged Polygon-Artefakte). Lösung: solche Meshes als EIGENE, NIE dezimierte
    # Gruppen exportieren — der grobe Rest bleibt im Tri-Budget, die Decals scharf.
    ap.add_argument("--protect-decals", type=int, default=1,
                    help="1=feine flache Label/Logo-Meshes voll-tesselliert behalten (nie dezimieren)")
    ap.add_argument("--decal-thin-mm", type=float, default=8.0,
                    help="max. Dicke (kleinste Bbox-Dim, mm) damit ein Mesh als Decal/Label gilt")
    ap.add_argument("--decal-broad-mm", type=float, default=12.0,
                    help="min. Breite (2.-kleinste Bbox-Dim, mm) eines Decals — schließt winzige Splitter aus")
    ap.add_argument("--decal-aspect", type=float, default=4.0,
                    help="min. broad/thin-Verhältnis — trennt flache Schilder/Logos von 3D-Boxen "
                         "(ein eingeprägtes Schild ist viel breiter als dick; ein Kasten nicht)")
    ap.add_argument("--decal-max-tris", type=int, default=20_000,
                    help="Decals über diesem Tri-Count sind eher Bleche/Wände → NICHT schützen (in Budget)")
    a = ap.parse_args()

    stage = Usd.Stage.Open(a.usd)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Named-MDL-Library-Materialien ohne diffuse_color_constant → sinnvolle Defaults.
    NAMED = {
        "metal_white_paint": (0.86, 0.86, 0.84), "polycarbonate_opaque": (0.80, 0.82, 0.85),
        "polycarbonate_cloudy_rough": (0.74, 0.77, 0.80), "aluminum": (0.78, 0.80, 0.82),
        "steel": (0.62, 0.64, 0.67), "rubber": (0.12, 0.12, 0.13), "black": (0.10, 0.10, 0.11),
    }
    _cache = {}

    def diffuse_color(prim):
        """diffuse RGB (0..1): child-Shader diffuse_color_constant (Omniverse-MDL),
        sonst Material-Namen-Heuristik, sonst displayColor, sonst Stahl-Grau."""
        try:
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat:
                mp = mat.GetPath().pathString
                if mp in _cache:
                    return _cache[mp]
                name = mat.GetPrim().GetName().lower()
                # 1) Child-Shader diffuse_color_constant (greift für ~alle CAD-Mats)
                for child in mat.GetPrim().GetChildren():
                    if child.IsA(UsdShade.Shader):
                        sh = UsdShade.Shader(child)
                        for nm in ("diffuse_color_constant", "diffuseColor", "baseColor"):
                            inp = sh.GetInput(nm)
                            if inp and inp.Get() is not None:
                                v = inp.Get()
                                col = (float(v[0]), float(v[1]), float(v[2]))
                                _cache[mp] = col; return col
                # 2) named MDL library material
                for k, c in NAMED.items():
                    if k in name:
                        _cache[mp] = c; return c
                _cache[mp] = (0.62, 0.64, 0.67); return _cache[mp]
        except Exception:
            pass
        try:
            dc = UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()
            if dc: return (float(dc[0][0]), float(dc[0][1]), float(dc[0][2]))
        except Exception:
            pass
        return (0.62, 0.64, 0.67)

    # Baked-in Teile (Anker/Zahnrad/… die in der USD-Szene in den Trays liegen)
    # NICHT mit-exportieren — die sind keine echten Detektions-Objekte und würden
    # als "Geister-Teile" in der Maschine erscheinen. (Fix 2026-05-28)
    BAKED_SUBSTR = ("/Anker_Kurz", "/Anker_Lang", "/Zahnrad", "/Buerstenhalter",
                    "/Getriebe", "/Ringmagnet", "/Poltopf_")
    def is_baked_part(prim):
        p = prim.GetPath().pathString
        # Tray-SCHALEN behalten (heißen "Tray_..."), nur die Part-Prims raus.
        for s in BAKED_SUBSTR:
            if s in p and "/Tray_" not in p:
                return True
        return False

    # DECAL-/LABEL-Erkennung (T-106): ein Aufkleber/Logo/Warn-Schild ist eine
    # DÜNNE FLACHE Platte — eine Bbox-Dim ist nahe 0 (Folien-/Schilddicke), die
    # beiden anderen sind nennenswert breit. Solche Meshes tragen die feinen
    # Buchstaben-/Logo-Kanten und dürfen NICHT dezimiert werden. Sehr große
    # Bleche/Wände (über --decal-max-tris) sind keine Decals → bleiben im Budget.
    def is_decal(F_count, ext_mm):
        if not a.protect_decals:
            return False
        if F_count > a.decal_max_tris:
            return False
        se = sorted(float(e) for e in ext_mm)   # [thin, broad, broader]
        thin, broad = se[0], se[1]
        if thin >= a.decal_thin_mm or broad <= a.decal_broad_mm:
            return False
        # Aspekt-Kriterium: ein flaches Schild/Logo ist VIEL breiter als dick.
        # Trennt eingeprägte Labels (thin~5mm, broad~35mm → Aspekt 7) von kleinen
        # 3D-Klötzen/Knöpfen (thin~6mm, broad~14mm → Aspekt 2.3, KEIN Decal).
        return (broad / max(thin, 1e-3)) >= a.decal_aspect

    # accumulate per quantized color (bulk) + per-mesh dedizierte Decal-Gruppen
    groups = {}        # color_key -> [vertsList, facesList, vcount]   (dezimierbar)
    decal_groups = []  # [(Vw, F)]  je Decal-Mesh eine eigene, NIE dezimierte Gruppe
    nmesh = ntri = nskip = ndecal = 0
    decal_tris = 0
    t0 = time.time()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if is_baked_part(prim):
            nskip += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        # WICHTIG: KEIN Pre-Scale! Die Punkte sind in mm, aber die
        # LocalToWorldTransform M enthält den mm→m-Scale (0.001) bereits.
        # raw(mm) @ M = korrekte Welt-Meter. (Bug 2026-05-28: *0.001 war doppelt
        # → Geometrie kollabierte auf den Translations-Punkt → unsichtbar.)
        V = np.array(pts, dtype=np.float64)
        ext_mm = V.max(0) - V.min(0)   # LOKALE Bbox in mm — Flachheit ist frame-invariant
        M = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64).reshape(4, 4)
        Vw = (np.c_[V, np.ones(len(V))] @ M)[:, :3]   # USD row-vector: p' = p @ M
        F = triangulate(list(counts), list(idx))
        if len(F) == 0:
            continue
        col = diffuse_color(prim)
        key = tuple(int(round(c * 24)) for c in col)   # quantize ~24 buckets/channel (feiner)
        if is_decal(len(F), ext_mm):
            # Eigene Gruppe → wird NIE in die Bulk-Farbgruppe gemischt, NIE dezimiert.
            decal_groups.append((Vw, F, key))
            ndecal += 1; decal_tris += len(F)
        else:
            g = groups.setdefault(key, [[], [], 0])
            g[0].append(Vw)
            g[1].append(F + g[2])
            g[2] += len(Vw)
        nmesh += 1; ntri += len(F)
        if nmesh % 500 == 0:
            log(f"  processed {nmesh} meshes, {ntri/1e6:.1f}M tris, {time.time()-t0:.0f}s")

    bulk_tris = ntri - decal_tris
    log(f"collected {nmesh} meshes ({nskip} baked parts skipped) -> "
        f"{len(groups)} color-groups + {ndecal} protected decals ({decal_tris/1e3:.0f}k tris), "
        f"{ntri/1e6:.2f}M tris total")

    # Das Tri-Budget gilt NUR fürs dezimierbare Bulk — die geschützten Decals
    # zählen nicht mit (sie bleiben voll). So bleibt die Datei klein UND scharf.
    ratio = 1.0
    if a.max_total_tris and bulk_tris > a.max_total_tris:
        ratio = a.max_total_tris / bulk_tris
        log(f"decimating bulk: {bulk_tris/1e6:.2f}M -> ~{a.max_total_tris/1e6:.2f}M tris "
            f"(ratio {ratio:.2f}); {ndecal} Decals voll erhalten")

    scene = trimesh.Scene()
    for ci, (key, (vl, fl, _)) in enumerate(groups.items()):
        V = np.vstack(vl); F = np.vstack(fl)
        m = trimesh.Trimesh(vertices=V, faces=F, process=False)
        if ratio < 1.0 and len(m.faces) > 128:
            tgt = max(64, int(len(m.faces) * ratio))
            done = False
            for call in (
                lambda: m.simplify_quadric_decimation(face_count=tgt),
                lambda: m.simplify_quadric_decimation(percent=ratio),
                lambda: m.simplify_quadric_decimation(1.0 - ratio),  # target_reduction
            ):
                try:
                    m = call(); done = True; break
                except Exception:
                    continue
            if not done and ci < 3:
                log(f"  decimation API mismatch group {ci} — keeping full")
        # WICHTIG: Normalen erzwingen — ohne sie rendert MeshStandardMaterial
        # im Three.js unbeleuchtet/unsichtbar (Bug 2026-05-28: Cell war komplett
        # unsichtbar trotz korrekter Geometrie+Farbe).
        try:
            m.fix_normals()
            _ = m.vertex_normals          # lazy-compute erzwingen
        except Exception:
            pass
        rgb = [min(1.0, max(0.0, key[i] / 24.0)) for i in range(3)]   # glTF baseColor ist 0..1!
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
            metallicFactor=0.15, roughnessFactor=0.65)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(m, geom_name=f"grp_{ci}")

    # GESCHÜTZTE DECALS: voll-tesselliert, NIE dezimiert. Nach Farbe gemergt
    # (eine Gruppe je Decal-Farbe), damit die Draw-Calls niedrig bleiben — aber
    # in EIGENEN Gruppen, getrennt von den (dezimierten) Bulk-Farbgruppen, sonst
    # würde der proportionale Pass sie wieder mitzerschießen.
    decal_by_color = {}
    for Vw, F, key in decal_groups:
        d = decal_by_color.setdefault(key, [[], [], 0])
        d[0].append(Vw); d[1].append(F + d[2]); d[2] += len(Vw)
    for di, (key, (vl, fl, _)) in enumerate(decal_by_color.items()):
        V = np.vstack(vl); F = np.vstack(fl)
        m = trimesh.Trimesh(vertices=V, faces=F, process=False)   # KEINE Dezimierung
        try:
            m.fix_normals()
            _ = m.vertex_normals
        except Exception:
            pass
        rgb = [min(1.0, max(0.0, key[i] / 24.0)) for i in range(3)]
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
            metallicFactor=0.15, roughnessFactor=0.65)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(m, geom_name=f"decal_{di}")
    if decal_by_color:
        log(f"  + {len(decal_by_color)} Decal-Gruppen voll-tesselliert ({decal_tris/1e3:.0f}k tris)")

    log(f"exporting GLB ({len(scene.geometry)} groups) ...")
    data = scene.export(file_type="glb")
    with open(a.out, "wb") as f:
        f.write(data)
    log(f"wrote {a.out}  ({len(data)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
