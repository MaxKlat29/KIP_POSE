"""gateway_proxy.py — pure (fastapi-/httpx-freie) Logik fuer den kip_server-Proxy (S-013).

`kip_server:8077` bleibt der **Live-Pipeline-A-Pfad** (`pipeline=gdrnpp` = byte-identisch,
Guard returnt frueh). Fuer die **6 NICHT-A-Kombis** proxyt kip_server zum `gateway:8000`,
damit das FE EINE Origin behaelt (Caddy `/KIP`, kein zweiter Tunnel). Dieses Modul ist die
**testbare Naht** dafuer — bewusst OHNE fastapi/httpx-Import (die liegen nur in der Box-venv;
die lokalen pipelines-Tests laufen stdlib+numpy):

  * `COMBO_TO_GATEWAY`   — Map der 6 Kombi-ids → Gateway-`(seg_source, pose_source)`.
                           (kip_server-Kombi-ids ≠ Gateway-source-ids; siehe unten.)
  * `combo_to_gateway`   — Kombi-id → {seg_source, pose_source, needs_depth} (4xx bei ungueltig).
  * `gateway_predict_to_pose_result` — Gateway-`/predict`-Antwort (`instances[].T_cam_obj`)
                           → frozen pose_result-Doc. Reused die EXAKTE composed.py-Mapping-
                           Mathematik (bop_pose_to_world mit t in **mm**, Boden-Snap,
                           Symmetrie-Kanonisierung ueber den CamelCase-Part, assemble_doc).
  * `pipelines_status`   — die 7-Kombi-`/api/pipelines`-Payload mit `available` +
                           **`unavailable_reason`** (enum, Mia S-010 §12 Pflicht-Feld) +
                           seg/pose/needs_depth/is_pipeline_a. Health-Probe injizierbar.
  * `unavailable_reason` — die Enum-Logik (service_down | training | None).

Quelle der Wahrheit: `combos.COMBO_WHITELIST` (combo-Metadaten) + `mesh/CONTRACT.md §5`
(7 Kombis, needs_depth) + `mesh/gateway/app.py` (INFER_SOURCES × POSE_SOURCES).
"""
from __future__ import annotations

from typing import Callable, Optional

from .combos import COMBO_WHITELIST, FEASIBLE_COMBOS

# ── unavailable_reason-Enum (Mia S-010 §12) ───────────────────────────────────
# Das FE muss „Dienst gerade nicht aktiv" von „Modell trainiert noch" unterscheiden
# koennen (sonst kann es Available-disabled nicht von Gating-disabled trennen).
REASON_SERVICE_DOWN = "service_down"   # /health down/evicted → Option disabled „Dienst nicht aktiv"
REASON_TRAINING = "training"           # Modell trainiert noch (S-008 yolo-seg Langlaeufer)
UNAVAILABLE_REASONS = (REASON_SERVICE_DOWN, REASON_TRAINING)


# ── Kombi-id → Gateway-source-ids ─────────────────────────────────────────────
# WICHTIG: die kip_server/combos-Kombi-ids sind NICHT die Gateway-source-ids.
#   * combos.py-seg-Namen:  yolo-seg, sam3        (CONTRACT.md §2/§5)
#   * Gateway INFER_SOURCES: "yolo", "sam3"       (gateway/app.py:51-54)  → yolo-seg == "yolo"
#   * combos.py-pose-ids:    foundationpose, gigapose_rgbd, gigapose_rgb
#   * Gateway POSE_SOURCES:  "foundationpose","gigapose_rgbd","gigapose_rgb" (1:1, §5)
# Die `pipeline`-Flagge (rgb/rgbd) traegt das Gateway selbst pro pose_source (§5) — wir
# muessen sie NICHT mitschicken. Kombi 1 (gdrnpp/yolo-obb = Pipeline A) ist NICHT hier:
# sie laeuft NIE ueber das Gateway (Live-Monolith, §4).
_SEG_COMBO_TO_GATEWAY = {"yolo-seg": "yolo", "sam3": "sam3"}

COMBO_TO_GATEWAY: "dict[str, dict]" = {}
for _c in COMBO_WHITELIST:
    if _c["is_pipeline_a"]:
        continue  # Kombi 1 = Pipeline A, kein Gateway-Proxy
    COMBO_TO_GATEWAY[_c["id"]] = {
        "seg_source": _SEG_COMBO_TO_GATEWAY[_c["seg"]],
        "pose_source": _c["id"].split("__", 1)[1],   # combo-id-Suffix == Gateway-pose-id
        "needs_depth": bool(_c["needs_depth"]),
    }


# Die Pipeline-A-Identitaet (Live-Pfad). Sowohl die Kombi-id `gdrnpp` als auch das
# explizite Achsen-Paar (seg=yolo-obb, pose=gdrnpp) zeigen darauf.
PIPELINE_A_ID = "gdrnpp"
PIPELINE_A_SEG = "yolo-obb"
PIPELINE_A_POSE = "gdrnpp"


class InvalidCombo(ValueError):
    """Ungueltige (nicht-Whitelist) Kombi → kip_server mappt das auf HTTP 400."""


def is_pipeline_a(*, pipeline: str | None = None,
                  seg: str | None = None, pose: str | None = None) -> bool:
    """True wenn die Anfrage Pipeline A (= Live-Pfad) meint.

    Akzeptiert beide FE-Aufruf-Formen (Mission AK 1):
      * `pipeline=gdrnpp`                         (heutiger Default-Param)
      * `seg=yolo-obb & pose=gdrnpp`              (neue 2-Achsen-Form S-010)
    Ein leerer/fehlender pipeline-Param ohne seg/pose ist ebenfalls Pipeline A
    (Default, byte-identisch zu currentPipeline()→"gdrnpp")."""
    pid = (pipeline or "").strip()
    if seg or pose:
        return (seg or "").strip() == PIPELINE_A_SEG and (pose or "").strip() == PIPELINE_A_POSE
    return pid in ("", PIPELINE_A_ID)


def resolve_combo_id(*, pipeline: str | None = None,
                     seg: str | None = None, pose: str | None = None) -> str:
    """Loest die Anfrage auf eine Kombi-id auf. Pipeline A → 'gdrnpp'. Die 6 NICHT-A
    werden entweder direkt per `pipeline=<combo_id>` ODER per `seg=&pose=` adressiert.

    Wirft InvalidCombo wenn (seg,pose) keine der 7 Whitelist-Kombis ist (→ 400 im
    kip_server, „niemals 12 Kombis"-Invariante, Mia S-010)."""
    if is_pipeline_a(pipeline=pipeline, seg=seg, pose=pose):
        return PIPELINE_A_ID

    pid = (pipeline or "").strip()
    if pid and pid in COMBO_TO_GATEWAY:
        return pid

    # 2-Achsen-Form: (seg, pose) → Kombi-id ueber die Whitelist nachschlagen.
    if seg or pose:
        s, p = (seg or "").strip(), (pose or "").strip()
        for c in COMBO_WHITELIST:
            if c["is_pipeline_a"]:
                continue
            # Whitelist-pose ist ein Label ("FoundationPose"); Gateway-pose-id steckt
            # im combo-id-Suffix. Wir matchen tolerant auf beides.
            gw = COMBO_TO_GATEWAY[c["id"]]
            if c["seg"] == s and (p == gw["pose_source"] or _pose_label_matches(c, p)):
                return c["id"]
        raise InvalidCombo(
            f"Ungueltige Kombi (seg={seg!r}, pose={pose!r}). Erlaubt sind genau die "
            f"7 Whitelist-Kombis (CONTRACT.md §5)."
        )

    raise InvalidCombo(
        f"Unbekannte Pipeline {pipeline!r}. Erlaubt: 'gdrnpp' (Pipeline A) oder eine der "
        f"6 NICHT-A-Kombi-ids {sorted(COMBO_TO_GATEWAY)}."
    )


def _pose_label_matches(combo: dict, pose: str) -> bool:
    """Tolerantes Match der FE-Pose-Achse gegen die Whitelist-Pose. Akzeptiert das
    Label ('FoundationPose','GigaPose-2D','GigaPose-3D') ODER die Gateway-id."""
    if not pose:
        return False
    label = combo["pose"].lower().replace(" ", "").replace("-", "")
    cand = pose.lower().replace(" ", "").replace("-", "").replace("_", "")
    # GigaPose-3D == gigapose_rgbd, GigaPose-2D == gigapose_rgb (pipeline-Flag im Suffix).
    aliases = {
        "gigapose3d": "gigaposergbd", "gigapose2d": "gigaposergb",
    }
    cand = aliases.get(cand, cand)
    label = aliases.get(label, label)
    return cand == label or cand == COMBO_TO_GATEWAY[combo["id"]]["pose_source"].replace("_", "")


def combo_to_gateway(combo_id: str) -> dict:
    """Kombi-id → Gateway-Routing {seg_source, pose_source, needs_depth}.

    Wirft InvalidCombo wenn combo_id keine der 6 NICHT-A-Kombis ist (Pipeline A
    laeuft NICHT ueber das Gateway — das prueft der Aufrufer vorher per is_pipeline_a)."""
    gw = COMBO_TO_GATEWAY.get(combo_id)
    if gw is None:
        raise InvalidCombo(
            f"Kombi {combo_id!r} ist keine Gateway-Kombi. NICHT-A-Kombis: "
            f"{sorted(COMBO_TO_GATEWAY)} (Pipeline A 'gdrnpp' laeuft ueber den Live-Pfad)."
        )
    return dict(gw)


# ── Gateway-/predict-Antwort → frozen pose_result-Doc ─────────────────────────
def gateway_predict_to_pose_result(gateway_resp: dict, *, camera: dict,
                                   table_origin, source_image: str,
                                   warn: Optional[Callable] = None,
                                   snap: bool = True) -> dict:
    """Gateway-`/predict`-Antwort → frozen pose_result-Doc (S-003-Contract).

    Das Gateway hat Seg→Pose schon gefahren und liefert `instances:[{id,class,conf,
    T_cam_obj,mask_b64}]` (T_cam_obj = OpenCV-cam, Meter, mesh→cam, §1). Wir machen NUR
    noch den Welt-Frame-Schritt — und zwar ueber denselben `composed.tcamobj_to_world_entry`,
    den auch die ComposedPipeline nutzt. So liefert der Proxy-Pfad byte-gleiche Docs wie
    der direkte Composed-Pfad (Eval-Konsistenz S-012, KEINE zweite Mapping-Implementierung).

    Args:
      gateway_resp : die JSON-Antwort von `POST gateway/predict`.
      camera       : {cam_K[9], cam_R_w2c[9], cam_t_w2c[3] mm} (ein BOP scene_camera).
      table_origin : [x,y,z] Welt-Nullpunkt (Meter).
      source_image : Bild-Kennung fuers Doc-meta.
      snap         : Boden-Snap anwenden (default True; best-effort, fehlt CAD → skip).
    """
    from . import contract
    from .composed import tcamobj_to_world_entry, _mask_b64_to_bbox

    cam = camera or {}
    R_w2c = cam.get("cam_R_w2c")
    t_w2c = cam.get("cam_t_w2c")
    if R_w2c is None or t_w2c is None:
        raise ValueError(
            "gateway_predict_to_pose_result braucht camera={cam_R_w2c,cam_t_w2c} "
            "(Welt-Extrinsics) — die Pose-Stage liefert nur den Cam-Frame T_cam_obj."
        )
    to = list(table_origin)

    entries = []
    for inst in gateway_resp.get("instances", []):
        cls, T = inst.get("class"), inst.get("T_cam_obj")
        if cls is None or T is None:
            continue                         # Instanz ohne Pose (Gateway-Skip §3) → weg
        mask = inst.get("mask_b64")
        conf = inst.get("conf")
        entry = tcamobj_to_world_entry(      # §1+§6, geteilt mit ComposedPipeline
            cls=cls, T_cam_obj=T, R_w2c=R_w2c, t_w2c=t_w2c, table_origin=to,
            instance_id=int(inst.get("id", len(entries))),
            confidence=(conf if conf is not None else 1.0),
            bbox_2d=(_mask_b64_to_bbox(mask) if mask else [0, 0, 0, 0]),
            snap=snap, warn=warn,
        )
        if entry is not None:                # None = Klasse out-of-scope (§6)
            entries.append(entry)

    return contract.assemble_doc(source_image, entries, table_origin=to)


# ── /api/pipelines — die 7-Kombi-Payload mit unavailable_reason ───────────────
def unavailable_reason(available: bool, *, service_up: bool,
                       training: bool) -> "str | None":
    """Die unavailable_reason-Enum-Logik (Mia S-010 §12).

    Rueckgabe:
      None             → Kombi available (kein Grund).
      'training'       → Service-seitig „Modell trainiert noch" (S-008 yolo-seg).
                         Hat Vorrang vor service_down: das FE soll „trainiert noch"
                         zeigen statt „Dienst nicht aktiv", auch wenn der Service
                         (mangels Modell) noch nicht up ist.
      'service_down'   → Service /health down/evicted.
    """
    if available:
        return None
    if training:
        return REASON_TRAINING
    return REASON_SERVICE_DOWN


def pipelines_status(*, gateway_health: Optional[dict] = None,
                     live_pipeline_a_available: bool = True,
                     training_segs: Optional[set] = None) -> list:
    """Baut die volle Feasibility-Matrix (~12 Kombis) fuer `/api/pipelines`.

    T-147-RELAX (Max 2026-06-07): das FE-Gating soll ALLE feasible Kombis erlauben,
    nicht nur die kuratierten 7. Quelle der Wahrheit ist jetzt `combos.FEASIBLE_COMBOS`
    (registry × feasibility-Kreuzprodukt). Die kuratierten 7 (`COMBO_WHITELIST`)
    bleiben das `recommended`-Highlight-Subset. Die GDRNPP↔yolo-obb-Kopplung wird NICHT
    mehr hart hardcoded — sie faellt aus der Matrix (GDRNPP koppelt jetzt mit allen 3
    Seg; mit yolo-seg/sam3 = `degraded`, AABB-aus-Maske).

    Pro Kombi (Mia S-010 §12 + T-147-Flags):
      id, name, description, seg, pose, needs_depth, is_pipeline_a,
      available (bool), unavailable_reason (service_down|training|None),
      recommended (bool), degraded (bool), degraded_reason (str|None),
      class_ambiguity (bool).

    Back-compat: die 7 recommended-Kombis tragen exakt dieselben Felder/ids wie zuvor;
    FE-Code, der nur die alten Felder liest, funktioniert weiter. Die 5 zusaetzlichen
    Kombis sind durch `recommended=False` + Flags markiert.

    Args:
      gateway_health : die `gateway/health`-Antwort {ok, yolo, fp, gigapose, sam3}
                       (gateway/app.py:292-311). None → Gateway nicht erreichbar →
                       alle Mesh-Kombis service_down. Pipeline A bleibt UNBERUEHRT.
      live_pipeline_a_available : ob der :8077/:8078-Live-Pfad lebt. Mia §6: solange
                       er lebt, ist Pipeline A (und der GDRNPP-Pose-Pfad) available.
      training_segs  : Set der Seg-Quellen, deren Modell gerade trainiert (S-008).
                       Diese Kombis → unavailable_reason 'training' statt 'service_down'.
    """
    training_segs = training_segs or set()
    svc_up = _gateway_service_up(gateway_health)

    out = []
    for c in FEASIBLE_COMBOS:
        seg_id, pose_id = c["seg_id"], c["pose_id"]
        training = seg_id in training_segs

        if c["is_pipeline_a"]:
            # Kombi 1 (yolo-obb → GDRNPP) = Live-Monolith, haengt NICHT am Mesh.
            avail = bool(live_pipeline_a_available)
            reason = unavailable_reason(avail, service_up=avail, training=False)
            name = "GDRNPP (yolo-obb, Pipeline A)"
        elif pose_id == "gdrnpp":
            # GDRNPP-degraded (yolo-seg/sam3 → GDRNPP, AABB-aus-Maske). Der GDRNPP-Pose-
            # Pfad haengt am Live-Worker (wie Pipeline A); zusaetzlich braucht es den
            # Seg-Service (yolo/sam3) up — die Maske kommt aus dem Mesh.
            seg_up = svc_up.get(_SEG_COMBO_TO_GATEWAY.get(seg_id, seg_id), False)
            pose_up = bool(live_pipeline_a_available)
            avail = bool(seg_up and pose_up and not training)
            reason = unavailable_reason(avail, service_up=(seg_up and pose_up),
                                        training=training)
            name = f"{c['seg']} → {c['pose']}"
        else:
            # Reine Mesh-Kombi (FP/GigaPose). Routing kommt aus COMBO_TO_GATEWAY (sofern
            # registriert); fehlt es (sollte nicht), faellt sie defensiv auf service_down.
            gw = COMBO_TO_GATEWAY.get(c["id"])
            if gw is not None:
                seg_up = svc_up.get(gw["seg_source"], False)
                pose_up = svc_up.get(gw["pose_source"], False)
            else:
                seg_up = svc_up.get(_SEG_COMBO_TO_GATEWAY.get(seg_id, seg_id), False)
                pose_up = svc_up.get(pose_id, False)
            avail = bool(seg_up and pose_up and not training)
            reason = unavailable_reason(avail, service_up=(seg_up and pose_up),
                                        training=training)
            name = f"{c['seg']} → {c['pose']}"

        desc = (f"Kombi {c['n']}: {c['seg']} → {c['pose']} "
                f"(needs_depth={c['needs_depth']}"
                + (f", pipeline={c['pipeline']}" if c.get('pipeline') else "")
                + (", degraded=AABB-aus-Maske" if c["degraded"] else "")
                + (", class-ambig" if c["class_ambiguity"] else "") + ")")

        out.append({
            "id": c["id"],
            "name": name,
            "description": desc,
            "seg": c["seg"],
            "pose": c["pose"],
            "needs_depth": bool(c["needs_depth"]),
            "is_pipeline_a": bool(c["is_pipeline_a"]),
            "available": avail,
            "unavailable_reason": reason,
            # T-147-Flags (FE-Gating-Relax): recommended-7 vs degraded/ambig-5.
            "recommended": bool(c["recommended"]),
            "degraded": bool(c["degraded"]),
            "degraded_reason": c["degraded_reason"],
            "class_ambiguity": bool(c["class_ambiguity"]),
        })
    return out


def _gateway_service_up(gateway_health: Optional[dict]) -> dict:
    """Aus der gateway/health-Antwort die per-Service-up-Map ableiten.

    gateway/health (gateway/app.py:292-311) hat die Form
      {ok, yolo:{ok}, fp:{ok}, gigapose:{...}, sam3:{ok}}.
    Wir mappen das auf die source-ids, die COMBO_TO_GATEWAY nutzt:
      seg:  'yolo' (yolo-svc), 'sam3' (sam3-svc)
      pose: 'foundationpose' (fp-svc), 'gigapose_rgbd'+'gigapose_rgb' (beide gigapose-svc).
    """
    if not gateway_health:
        return {}
    def _ok(node):
        if isinstance(node, dict):
            # gigapose nutzt status:'ok'|'loading'|'error'; yolo/fp/sam3 nutzen ok:bool.
            if "ok" in node:
                return bool(node.get("ok"))
            return str(node.get("status", "")).lower() == "ok"
        return False
    yolo = _ok(gateway_health.get("yolo"))
    fp = _ok(gateway_health.get("fp"))
    giga = _ok(gateway_health.get("gigapose"))
    sam3 = _ok(gateway_health.get("sam3"))
    return {
        "yolo": yolo,
        "sam3": sam3,
        "foundationpose": fp,
        "gigapose_rgbd": giga,
        "gigapose_rgb": giga,
    }


def available_combo_ids(pipelines: list) -> list:
    """Aus einer pipelines_status()-Liste die available-Kombi-ids (fuer /api/compare-Gating)."""
    return [p["id"] for p in pipelines if p.get("available")]
