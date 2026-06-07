"""combos.py — die 7-Kombi-Whitelist der Multi-Pipeline-Plattform (S-003 / T-129).

Seg-Source × Pose-Source. Es gibt GENAU 7 valide Kombis (NICHT 12) — die GDRNPP-
Kopplungsregel (CONTRACT.md §4) verbietet die uebrigen. Quelle der Wahrheit fuer die
Whitelist + `needs_depth` + `pipeline`-Flag ist CONTRACT.md §5 (Viktor, frozen v1):

  # | Seg       | Pose          | pose_source-id  | pipeline | needs_depth
  --+-----------+---------------+-----------------+----------+------------
  1 | yolo-obb  | GDRNPP        | gdrnpp          | —        | nein   = Pipeline A (Monolith!)
  2 | yolo-seg  | FoundationPose| foundationpose  | —        | JA
  3 | sam3      | FoundationPose| foundationpose  | —        | JA
  4 | yolo-seg  | GigaPose-3D   | gigapose_rgbd   | rgbd     | JA
  5 | yolo-seg  | GigaPose-2D   | gigapose_rgb    | rgb      | nein
  6 | sam3      | GigaPose-3D   | gigapose_rgbd   | rgbd     | JA
  7 | sam3      | GigaPose-2D   | gigapose_rgb    | rgb      | nein

**Kombi 1 ist KEINE ComposedPipeline.** GDRNPP ist hart an yolo-obb gekoppelt (liest
`obb`, per-Objekt-Checkpoints, §4) und IST die bestehende Pipeline A — der Live-Monolith
(`gdrnpp_adapter.GdrnppReferenceAdapter`), der byte-identisch + unveraendert bleibt. Wir
bauen sie hier NICHT nach; sie ist schon in der Registry (registry._autoload_reference).
combos.py registriert nur die **6 NICHT-A-Kombis** als 2-stufige ComposedPipelines.

Registrierung ist DEFENSIV (wie registry._autoload_reference): ein Fehler beim Bauen
einer Kombi darf `import pipelines` NIE brechen. Service-URLs kommen aus Env (Gateway-
Adressen) mit Defaults; lokal nicht erreichbar zu sein ist OK — `available` prueft das
nicht, die Gates (S-016) + das echte Wiring (S-013) tun es. Die Unit-Tests injizieren
einen Mock-Service ueber `seg.post_json` / `pose.post_json`.
"""
from __future__ import annotations

import os

from .seg_base import SegAdapter
from .pose_base import PoseAdapter
from .composed import ComposedPipeline
from . import registry


# ── Seg-Service-Endpunkte (Env-overridebar; Default = lokales Mesh-Gateway-Routing) ──
def _seg_url(name: str) -> str:
    return os.environ.get(f"POSE_SEG_URL_{name.upper().replace('-', '_')}",
                          f"http://localhost:8090/seg/{name}")


def _pose_url(name: str) -> str:
    return os.environ.get(f"POSE_POSE_URL_{name.upper().replace('-', '_')}",
                          f"http://localhost:8090/pose/{name}")


# ── Seg-Adapter-Fabriken (drop-in pro Quelle, Contract §2) ────────────────────
def make_yolo_seg() -> SegAdapter:
    s = SegAdapter(_seg_url("yolo-seg"))
    s.id, s.name, s.needs_prompts = "yolo-seg", "YOLO26m-seg", False
    return s


def make_sam3() -> SegAdapter:
    s = SegAdapter(_seg_url("sam3"))
    s.id, s.name, s.needs_prompts = "sam3", "SAM3 (prompt)", True
    return s


# ── Pose-Adapter-Fabriken (drop-in pro Estimator, Contract §3) ────────────────
def make_foundationpose() -> PoseAdapter:
    p = PoseAdapter(_pose_url("foundationpose"))
    p.id, p.name = "foundationpose", "FoundationPose (RGB-D)"
    p.needs_depth = True                      # §5: FP ist RGB-D-only (Pflicht)
    return p


def make_gigapose_rgbd() -> PoseAdapter:
    p = PoseAdapter(_pose_url("gigapose"))    # gigapose_2D/3D teilen EINEN Load (§5)
    p.id, p.name = "gigapose_rgbd", "GigaPose-3D (rgbd+Kabsch)"
    p.needs_depth, p.pipeline = True, "rgbd"  # §5: Kabsch auf Depth-Pointcloud
    return p


def make_gigapose_rgb() -> PoseAdapter:
    p = PoseAdapter(_pose_url("gigapose"))
    p.id, p.name = "gigapose_rgb", "GigaPose-2D (rgb)"
    p.needs_depth, p.pipeline = False, "rgb"  # §5: kein Depth, kein Kabsch
    return p


# ── Die 6 NICHT-A-ComposedPipelines (Kombi 2..7) ──────────────────────────────
# (combo_id, name, seg-Fabrik, pose-Fabrik, seg_prompts?)  — Reihenfolge = §5 #2..#7.
_SAM3_PROMPTS = {
    "anker_kurz": "short anchor metal part",
    "anker_lang": "long anchor metal part",
}

_COMBO_SPECS = [
    # #2
    ("yolo_seg__foundationpose", "yolo-seg → FoundationPose (RGB-D)",
     make_yolo_seg, make_foundationpose, None),
    # #3
    ("sam3__foundationpose", "sam3 → FoundationPose (RGB-D)",
     make_sam3, make_foundationpose, _SAM3_PROMPTS),
    # #4
    ("yolo_seg__gigapose_rgbd", "yolo-seg → GigaPose-3D (rgbd)",
     make_yolo_seg, make_gigapose_rgbd, None),
    # #5
    ("yolo_seg__gigapose_rgb", "yolo-seg → GigaPose-2D (rgb)",
     make_yolo_seg, make_gigapose_rgb, None),
    # #6
    ("sam3__gigapose_rgbd", "sam3 → GigaPose-3D (rgbd)",
     make_sam3, make_gigapose_rgbd, _SAM3_PROMPTS),
    # #7
    ("sam3__gigapose_rgb", "sam3 → GigaPose-2D (rgb)",
     make_sam3, make_gigapose_rgb, _SAM3_PROMPTS),
]


def build_combos() -> "list[ComposedPipeline]":
    """Baut die 6 NICHT-A-ComposedPipelines (Kombi 2..7). Reine Konstruktion (kein
    Netz). Wird von `register_combos` + von den Tests genutzt."""
    out = []
    for combo_id, name, seg_f, pose_f, prompts in _COMBO_SPECS:
        seg, pose = seg_f(), pose_f()
        out.append(ComposedPipeline(
            combo_id=combo_id, name=name, seg=seg, pose=pose, seg_prompts=prompts,
            description=f"Kombi {combo_id}: {seg.id} → {pose.id} "
                        f"(needs_depth={pose.needs_depth}"
                        + (f", pipeline={pose.pipeline}" if pose.pipeline else "") + ")",
        ))
    return out


# ── Whitelist-Metadaten (FE-Gating + needs_depth, CONTRACT.md §5) ─────────────
# Alle 7 inkl. Kombi 1 (Pipeline A = der gdrnpp-Monolith). Single-Source fuers FE
# (S-010) + die Gates. needs_depth steuert das Depth-Forwarding im Gateway (§5).
COMBO_WHITELIST = [
    {"n": 1, "seg": "yolo-obb", "pose": "GDRNPP", "id": "gdrnpp",
     "pipeline": None, "needs_depth": False, "is_pipeline_a": True},
    {"n": 2, "seg": "yolo-seg", "pose": "FoundationPose", "id": "yolo_seg__foundationpose",
     "pipeline": None, "needs_depth": True, "is_pipeline_a": False},
    {"n": 3, "seg": "sam3", "pose": "FoundationPose", "id": "sam3__foundationpose",
     "pipeline": None, "needs_depth": True, "is_pipeline_a": False},
    {"n": 4, "seg": "yolo-seg", "pose": "GigaPose-3D", "id": "yolo_seg__gigapose_rgbd",
     "pipeline": "rgbd", "needs_depth": True, "is_pipeline_a": False},
    {"n": 5, "seg": "yolo-seg", "pose": "GigaPose-2D", "id": "yolo_seg__gigapose_rgb",
     "pipeline": "rgb", "needs_depth": False, "is_pipeline_a": False},
    {"n": 6, "seg": "sam3", "pose": "GigaPose-3D", "id": "sam3__gigapose_rgbd",
     "pipeline": "rgbd", "needs_depth": True, "is_pipeline_a": False},
    {"n": 7, "seg": "sam3", "pose": "GigaPose-2D", "id": "sam3__gigapose_rgb",
     "pipeline": "rgb", "needs_depth": False, "is_pipeline_a": False},
]


def register_combos() -> None:
    """Registriert die 6 NICHT-A-Kombis in der Pipeline-Registry. DEFENSIV: ein Fehler
    beim Bauen/Registrieren einer Kombi darf `import pipelines` NIE brechen (wie
    registry._autoload_reference). Doppel-Registrierung wird still uebersprungen.

    Kombi 1 (gdrnpp = Pipeline A) wird hier NICHT registriert — das ist der bestehende
    Monolith-Adapter (registry._autoload_reference)."""
    try:
        combos = build_combos()
    except Exception:  # noqa: BLE001 — Bauen ist best-effort beim Import
        return
    for cp in combos:
        try:
            if not registry.has(cp.id):
                registry.register(cp)
        except Exception:  # noqa: BLE001 — eine kaputte Kombi blockiert die anderen nicht
            continue
