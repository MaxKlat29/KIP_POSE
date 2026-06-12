"""composed.py — ComposedPipeline: die 2-stufige (Seg→Pose) PipelineAdapter-Auspraegung.

S-003 / T-129. Das Rueckgrat der komponierbaren Multi-Pipeline-Plattform: ein
`SegAdapter` (Stage 1, liefert Masken/OBBs) + ein `PoseAdapter` (Stage 2, liefert
`T_cam_obj`) werden zu EINER `PipelineAdapter` verschmolzen, die das bestehende
byte-kompatible `infer(image_path, camera, table_origin) -> pose_result`-Interface
erfuellt. Damit laufen `compare_pipelines.py` + `/api/pipelines` ohne Anpassung weiter.

Der Live-Pfad (`gdrnpp`-Monolith = Pipeline A) bleibt UNBERUEHRT — das ist ein eigener
Adapter (`gdrnpp_adapter.py`), der hier NICHT angefasst wird.

DIE ZWEI FALLEN (CONTRACT.md §1 + §6), exakt adressiert:

  1. **Koordinaten-Frame** (§1): die Pose-Stage liefert `T_cam_obj` = OpenCV-cam, Meter,
     mesh→cam, 4×4 row-major. Das ist NICHT der pose_result-Welt-Frame. Die Umrechnung
     ist `bop_adapter.bop_pose_to_world` (mit t in **mm**!) — der inverse Round-Trip
     `compare_pipelines.world_to_bop_cam` ist getestet (test_composed_pipeline.py).
     Vorzeichen-/Transpose-Falle: `D_FLIP=diag(1,-1,-1,1)` ist USD→OpenCV, NICHT cam↔world.

  2. **Klassen-Mapping** (§6): das Mesh nutzt lowercase (`anker_kurz`), unser
     `bop_adapter` CamelCase (`Anker_Kurz`). NIE direkt `PART_SYMMETRY.get("anker_kurz")`
     (→ None → Symmetrie still uebersprungen → Eval-Muell). IMMER ueber obj_id:
     `part = bop_adapter.part_for_obj_id(CLASS_TO_OBJ_ID[cls])`. Danach
     `canonicalize_rotation` (Anker = continuous-Y/sym) → kein Yaw-Flackern im Overlay.

Lokal lauffaehig OHNE GPU/echte Services: `post_json` der Adapter ist die Mock-Naht
(Tests injizieren einen Mock-Service). Der Boden-Snap ist best-effort — fehlt das CAD-
Mesh lokal, wird er still uebersprungen (wie der gdrnpp-Mock-Fallback).
"""
from __future__ import annotations

import base64
import io
import pathlib
import sys

from .base import PipelineAdapter
from . import contract
from .seg_base import SegAdapter
from .pose_base import PoseAdapter

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ── §6: lowercase Mesh-Klasse → BOP obj_id (= numerisch konsistent ueber das Mesh) ──
# yolo:{0:anker_kurz,1:anker_lang}+1 == gigapose CLASS_TO_OBJ_ID == bop_adapter obj_id.
# zahnrad (obj 6) seit T-178c im VIEWER-/Live-Scope (GDRNPP-Checkpoint existiert,
# T-115). Die EVAL-Hauptmetrik bleibt bewusst D1 = 2 Klassen — batch_eval traegt
# dafuer seine EIGENE CLASS_TO_OBJ_ID-Kopie (active_class_parts haengt daran).
CLASS_TO_OBJ_ID = {"anker_kurz": 1, "anker_lang": 2, "zahnrad": 6}


def tcamobj_to_world_entry(*, cls: str, T_cam_obj, R_w2c, t_w2c, table_origin,
                           confidence: float, bbox_2d, instance_id: int,
                           snap: bool = True, warn=None) -> "dict | None":
    """DER eine T_cam_obj → Welt-Frame-Mapping-Schritt (S-003 §1+§6). Genutzt von der
    ComposedPipeline UND vom kip_server-Gateway-Proxy (gateway_proxy.py, S-013), damit
    beide Pfade byte-gleiche pose_result-Eintraege liefern (Eval-Konsistenz).

    Erledigt: §6-Klassen-Mapping (lowercase → obj_id → CamelCase-Part), §1-Einheit
    (T ist Meter → mm fuer bop_pose_to_world), Boden-Snap (best-effort), Symmetrie-
    Kanonisierung (gegen Yaw-Flackern). Gibt einen assemble_doc-Eintrag zurueck, oder
    None wenn die Klasse nicht im 2-Klassen-Scope ist (§6, still gefiltert)."""
    import numpy as np
    from bop_adapter import (
        bop_pose_to_world, planar_z_snap, canonicalize_rotation, part_for_obj_id,
    )
    if cls not in CLASS_TO_OBJ_ID:
        return None
    obj_id = CLASS_TO_OBJ_ID[cls]
    part = part_for_obj_id(obj_id)                      # §6: obj_id → CamelCase-Part

    T = np.asarray(T_cam_obj, float).reshape(4, 4)
    R_m2c = T[:3, :3]
    t_m2c_mm = T[:3, 3] * 1000.0                        # §1: Meter → mm

    R_world, t_world = bop_pose_to_world(R_m2c, t_m2c_mm, R_w2c, t_w2c, table_origin)

    if snap:
        verts = _mesh_verts_for(obj_id, warn or (lambda *a, **k: None))
        if verts is not None:
            t_world, _dz = planar_z_snap(R_world, t_world, verts, table_z=0.0)

    R_world = canonicalize_rotation(R_world, part)      # §6: Yaw-Flackern vermeiden

    return {
        "instance_id": int(instance_id),
        "part": part,
        "R_world": [float(x) for x in np.asarray(R_world, float).reshape(9)],
        "t_world": [float(x) for x in np.asarray(t_world, float).reshape(3)],
        "confidence": float(confidence),
        "bbox_2d": list(bbox_2d),
    }


def _mask_b64_to_bbox(mask_b64: str) -> "list[int]":
    """Voll-Bild-Maske (base64-PNG 0/255) → achsenparallele bbox [x0,y0,x1,y1].

    `assemble_doc`/das frozen Schema verlangen `bbox_2d`. Die Pose-Stage liefert keine
    Box, also leiten wir sie aus der Init-Maske ab (numpy+PIL). Leere Maske → 0-Box.
    """
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(mask_b64)
    m = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
    ys, xs = np.where(m > 127)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _png_to_b64(path: str) -> str:
    """Bild-Datei → base64-PNG-String (RGB). Stage-Input-Encoding."""
    from PIL import Image

    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _depth_png_to_b64(path: str) -> "str | None":
    """Optionale Depth-PNG (uint16 mm) → base64. None wenn die Datei fehlt."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


class ComposedPipeline(PipelineAdapter):
    """Seg→Pose verschmolzen zu einer vergleichbaren PipelineAdapter.

    Konstruktion (combos.py erzeugt diese pro Whitelist-Kombi):
        ComposedPipeline(
            combo_id="yolo_seg__foundationpose",
            name="yolo-seg → FoundationPose (RGB-D)",
            seg=SegAdapter(base_url=...),
            pose=PoseAdapter(base_url=..., needs_depth=True),
        )

    `needs_depth` wird vom Pose-Adapter geerbt (Single-Source = die Kombi-Whitelist §5).
    """

    def __init__(self, *, combo_id: str, name: str, seg: SegAdapter, pose: PoseAdapter,
                 description: str = "", seg_prompts: dict | None = None,
                 iterations: int | None = None, hypotheses: int | None = None,
                 table_origin=None, warn=None):
        if not isinstance(seg, SegAdapter):
            raise TypeError(f"seg muss ein SegAdapter sein, nicht {type(seg)!r}")
        if not isinstance(pose, PoseAdapter):
            raise TypeError(f"pose muss ein PoseAdapter sein, nicht {type(pose)!r}")
        self.id = combo_id
        self.name = name
        self.description = description or f"Composed: {seg.id} → {pose.id}"
        self.seg = seg
        self.pose = pose
        self.seg_prompts = seg_prompts
        self.iterations = iterations
        self.hypotheses = hypotheses
        self._table_origin = (
            tuple(table_origin) if table_origin is not None
            else tuple(contract.TABLE_ORIGIN_SCENE)
        )
        self._warn = warn or (lambda *a, **k: None)

    # needs_depth ist die Single-Source aus der Kombi (Pose-Adapter, §5).
    @property
    def needs_depth(self) -> bool:
        return bool(self.pose.needs_depth)

    @property
    def available(self) -> bool:
        """Der Seam selbst ist immer instanziierbar (Mock-faehig). Auf der Box wird
        `available` echt am `/health` der beiden Services gemessen (S-013) — hier
        defensiv True, weil die Mock-Naht den Seam lokal lauffaehig macht. Eine
        fehlende Service-URL macht ihn NICHT 'unavailable' (das pruefen die Gates)."""
        return True

    # ── die 2-stufige Komposition ──
    def infer(self, image_path: str, camera: dict, table_origin: list) -> dict:
        """Stage-1 (segment) → Stage-2 (pose) → T_cam_obj→Welt → assemble_doc.

        camera: {cam_K[9], cam_R_w2c[9], cam_t_w2c[3] mm} (ein BOP scene_camera-Eintrag).
        table_origin: [x,y,z] Welt-Nullpunkt (Meter). Faellt auf den Kombi-Default zurueck.
        """
        cam = camera or {}
        K = cam.get("cam_K")
        R_w2c = cam.get("cam_R_w2c")
        t_w2c = cam.get("cam_t_w2c")
        if K is None or R_w2c is None or t_w2c is None:
            raise ValueError(
                "ComposedPipeline.infer braucht camera={cam_K,cam_R_w2c,cam_t_w2c}. "
                "Die Seg→Pose-Stages koennen die Extrinsics NICHT selbst aufloesen "
                "(anders als die gdrnpp-Referenz, die e2e_infer fuer alles nutzt)."
            )
        to = list(table_origin) if table_origin is not None else list(self._table_origin)

        # Stage-Input-Encoding.
        rgb_b64 = _png_to_b64(image_path)
        depth_b64 = _depth_png_to_b64(_depth_path_for(image_path)) if self.needs_depth else None

        # ── Stage 1: segment ──
        dets = self.seg.segment(rgb_b64, K=K, prompts=self.seg_prompts)
        # Nur Klassen behalten, die wir auf obj_id mappen koennen (2-Klassen-Scope §6).
        # Unbekannte Klassen werden hier still gefiltert (wie das Mesh skip-Verhalten §3).
        dets = [d for d in dets if d.cls in CLASS_TO_OBJ_ID]
        if not dets:
            return contract.assemble_doc(image_path, [], table_origin=to)

        # ── Stage 2: pose ──
        poses = self.pose.estimate(
            rgb_b64, K, dets,
            depth_b64=depth_b64, iterations=self.iterations, hypotheses=self.hypotheses,
        )

        # Init-Masken per id fuer bbox + Konfidenz aus Stage 1 zuruecklegen (§3: das
        # Gateway mergt conf+mask per id; die Pose-Liste kann kuerzer sein).
        det_by_id = {d.id: d for d in dets}

        entries = []
        for p in poses:
            d = det_by_id.get(p.id)
            if d is None:
                # Pose ohne zugehoerige Detection (id-Drift) — defensiv ueberspringen.
                self._warn(f"[composed] Pose id={p.id} ohne Seg-Detection — uebersprungen")
                continue
            # DER eine Mapping-Schritt (§1+§6) — geteilt mit dem kip_server-Gateway-Proxy.
            entry = tcamobj_to_world_entry(
                cls=d.cls, T_cam_obj=p.T_cam_obj, R_w2c=R_w2c, t_w2c=t_w2c,
                table_origin=to, instance_id=p.id,
                confidence=(p.score if p.score is not None else d.conf),
                bbox_2d=_mask_b64_to_bbox(d.mask_b64), snap=True, warn=self._warn,
            )
            if entry is not None:                       # None = Klasse out-of-scope (§6)
                entries.append(entry)

        # assemble_doc leitet face/upright analytisch ab + validiert gegen das frozen
        # Schema (additionalProperties:false). Liefert ein byte-kompatibles pose_result.
        return contract.assemble_doc(image_path, entries, table_origin=to)

    def __repr__(self) -> str:
        return (f"<ComposedPipeline {self.id!r} seg={self.seg.id!r} pose={self.pose.id!r} "
                f"needs_depth={self.needs_depth}>")


# ── Hilfen (best-effort, lokal-lauffaehig) ────────────────────────────────────
def _mesh_verts_for(obj_id: int, warn):
    """CAD-Body-Vertices (Meter) fuer den Boden-Snap. None wenn kein CAD lokal da
    (dann wird der Snap uebersprungen — der Seam bleibt lauffaehig)."""
    try:
        from e2e_infer import _load_mesh_verts_m
        return _load_mesh_verts_m(int(obj_id), warn=warn)
    except Exception:  # noqa: BLE001 — CAD-Mesh ist best-effort
        return None


def _depth_path_for(image_path: str) -> str:
    """Depth-Datei neben dem RGB (BOP/Zivid-Konvention) — fuer den STANDALONE
    `infer(image_path,...)`-Pfad. (Das echte FE-Wiring S-013 reicht Depth explizit
    durchs Gateway durch, NICHT ueber Disk.) Probiert zwei Konventionen:
      * `rgb`→`depth` im Namen (z.B. `rgb_000001.png` → `depth_000001.png`),
      * `<stem>_depth.png`.
    Existiert keine, gibt _depth_png_to_b64 spaeter None zurueck (needs_depth-Pfad
    sendet dann kein Depth — der Service erwartet/skippt das)."""
    p = pathlib.Path(image_path)
    candidates = (p.parent / f"{p.stem.replace('rgb', 'depth')}.png",
                  p.parent / f"{p.stem}_depth.png")
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[-1])
