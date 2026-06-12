"""gateway_proxy.py — pure (fastapi-/httpx-freie) Logik fuer den kip_server-Proxy (S-013).

`kip_server:8077` bleibt der **Live-Pipeline-A-Pfad** (`pipeline=gdrnpp` = byte-identisch,
Guard returnt frueh). Fuer ALLE **NICHT-A feasible-Kombis** (T-155: 11 = 12 FEASIBLE_COMBOS
minus Pipeline A, inkl. der 3 yolo-obb-Mesh + 2 gdrnpp-degraded) proxyt kip_server zum
`gateway:8000`, damit das FE EINE Origin behaelt (Caddy `/KIP`, kein zweiter Tunnel). Dieses
Modul ist die **testbare Naht** dafuer — bewusst OHNE fastapi/httpx-Import (die liegen nur in
der Box-venv; die lokalen pipelines-Tests laufen stdlib+numpy):

  * `COMBO_TO_GATEWAY`   — Map ALLER NICHT-A-Kombi-ids → Gateway-`(seg_source, pose_source,
                           needs_depth, degraded)`. (kip_server-Kombi-ids ≠ Gateway-source-ids.)
  * `combo_to_gateway`   — Kombi-id → {seg_source, pose_source, needs_depth, degraded} (4xx
                           bei ungueltig). `degraded=true` nur fuer die 2 gdrnpp-degraded.
  * `gateway_predict_to_pose_result` — Gateway-`/predict`-Antwort (`instances[].T_cam_obj`)
                           → frozen pose_result-Doc. Reused die EXAKTE composed.py-Mapping-
                           Mathematik (bop_pose_to_world mit t in **mm**, Boden-Snap,
                           Symmetrie-Kanonisierung ueber den CamelCase-Part, assemble_doc).
  * `pipelines_status`   — die 12-Kombi-`/api/pipelines`-Payload mit `available` +
                           **`unavailable_reason`** (enum, Mia S-010 §12 Pflicht-Feld) +
                           seg/pose/needs_depth/is_pipeline_a/degraded. Health-Probe injizierbar.
  * `unavailable_reason` — die Enum-Logik (service_down | training | None).

Quelle der Wahrheit: `combos.FEASIBLE_COMBOS` (alle ~12, registry-Kreuzprodukt) +
`mesh/gateway/app.py` (INFER_SOURCES × POSE_SOURCES). `COMBO_WHITELIST` bleibt das
kuratierte `recommended`-7-Subset.
"""
from __future__ import annotations

from typing import Callable, Optional

from .combos import FEASIBLE_COMBOS

# ── unavailable_reason-Enum (Mia S-010 §12) ───────────────────────────────────
# Das FE muss „Dienst gerade nicht aktiv" von „Modell trainiert noch" unterscheiden
# koennen (sonst kann es Available-disabled nicht von Gating-disabled trennen).
REASON_SERVICE_DOWN = "service_down"   # /health down/evicted → Option disabled „Dienst nicht aktiv"
REASON_TRAINING = "training"           # Modell trainiert noch (S-008 yolo-seg Langlaeufer)
UNAVAILABLE_REASONS = (REASON_SERVICE_DOWN, REASON_TRAINING)


# ── Kombi-id → Gateway-source-ids ─────────────────────────────────────────────
# WICHTIG: die kip_server/combos-Kombi-ids sind NICHT die Gateway-source-ids.
#   * combos.py-seg-ids:     yolo-obb, yolo-seg, sam3   (SEG_SOURCES / CONTRACT.md §2)
#   * Gateway INFER_SOURCES:  "yolo", "sam3", "yolo-obb" (gateway/app.py:53-59)
#       - yolo-seg == "yolo" (YOLO26m-seg, mask-emittierend)
#       - yolo-obb → "yolo": die 3 yolo-obb-MESH-Kombis (FP/GigaPose nutzen die Maske)
#         fahren ueber den mask-emittierenden yolo-seg-Service. Identisch zum Runner
#         (`batch_eval._SEG_SOURCE = {"yolo-obb": "yolo", ...}`) → byte-gleiche
#         Gateway-Anfrage Runner↔Proxy (Konsistenz T-155 §4). Der Gateway lehnt
#         seg_source="yolo-obb" mit non-gdrnpp-pose hart ab (§4-Inverse-Guard,
#         app.py:527) — "yolo" ist der korrekte mask-Pfad fuer FP/GigaPose.
#   * combos.py-pose-ids:    foundationpose, gigapose_rgbd, gigapose_rgb, gdrnpp
#   * Gateway POSE_SOURCES:  1:1 (gateway/app.py:77-) → pose-id == combo-id-Suffix
# Die `pipeline`-Flagge (rgb/rgbd) traegt das Gateway selbst pro pose_source (§5) — wir
# muessen sie NICHT mitschicken. Kombi 1 (gdrnpp/yolo-obb = Pipeline A) ist NICHT hier:
# sie laeuft NIE ueber das Gateway (Live-Monolith, §4).
_SEG_COMBO_TO_GATEWAY = {"yolo-obb": "yolo", "yolo-seg": "yolo", "sam3": "sam3"}

# Quelle der Wahrheit ist jetzt FEASIBLE_COMBOS (alle ~12 feasible Kombis, T-155) statt
# nur der kuratierten 7-Whitelist. Sonst fehlen 5 Kombis (3 yolo-obb-Mesh + 2 gdrnpp-
# degraded) → resolve_combo_id/combo_to_gateway werfen InvalidCombo bei direktem
# Inferenz/Eval (Lena+Sam T-154-Befund). Pipeline A bleibt ausgenommen (Live-Pfad).
# Ein neues Seg-/Pose-Modul in combos.SEG_SOURCES/POSE_SOURCES taucht damit
# AUTOMATISCH auf — der 12-Kombi-Invarianten-Test (test_gateway_proxy) faengt, falls
# `_SEG_COMBO_TO_GATEWAY` dann nicht mitwaechst.
COMBO_TO_GATEWAY: "dict[str, dict]" = {}
for _c in FEASIBLE_COMBOS:
    if _c["is_pipeline_a"]:
        continue  # Kombi 1 = Pipeline A, kein Gateway-Proxy (Live-Monolith)
    COMBO_TO_GATEWAY[_c["id"]] = {
        "seg_source": _SEG_COMBO_TO_GATEWAY[_c["seg_id"]],
        "pose_source": _c["pose_id"],               # == combo-id-Suffix == Gateway-pose-id
        "needs_depth": bool(_c["needs_depth"]),
        # GDRNPP-degraded (yolo-seg/sam3 → gdrnpp): das Gateway muss `degraded=true`
        # bekommen, damit es den seg→yolo-obb-Force fuer pose=gdrnpp UEBERSPRINGT und
        # die Maske erhaelt → gdrnpp-svc AABB-aus-Maske-Fallback (T-153-Gateway-Opt-in,
        # default-off; deckt sich mit `batch_eval.http_predict`). Reine Mesh-Kombis
        # (FP/GigaPose) + die yolo-obb-Mesh-Kombis setzen es NICHT.
        "degraded": bool(_c["degraded"]),
    }


# Nur die NICHT-A feasible-Kombis (Metadaten fuer resolve_combo_id 2-Achsen-Form):
# {id, seg (= seg_id == Mesh-/FE-Name), pose (= FE-Label)}. Pipeline A ist hier NICHT
# drin — die wird vom is_pipeline_a-Guard vorab erledigt.
_NON_A_COMBOS = [
    {"id": _c["id"], "seg": _c["seg_id"], "pose": _c["pose"]}
    for _c in FEASIBLE_COMBOS if not _c["is_pipeline_a"]
]

# ── MoE (T-178): kuratierte Spezial-Pipeline, eigener FE-Screen ───────────────
# BEWUSST NICHT Teil der Seg×Pose-Matrix (FEASIBLE_COMBOS bleibt 12; Dropdowns,
# Eval und Invarianten-Tests unveraendert). seg=yolo-obb NATIV: der gdrnpp-RGB-
# Zweig braucht die orientierte Box, der RGB-D-Zweig nutzt die rasterisierte
# yolo-obb-Maske. pose='moe' = Gateway-Router: pro Instanz Anker-Pixel ->
# Sehstrahl auf die Tischebene -> Punkt-im-IR-Schatten-Polygon (T-178-
# Raytracing-Sim, project/config/moe_shadow.json) -> RGB- bzw. RGB-D-Zweig.
# Adressierung NUR per `pipeline=moe` (id-Form); die 2-Achsen-Form bleibt
# matrix-exklusiv (_NON_A_COMBOS unveraendert).
MOE_COMBO_ID = "moe"
COMBO_TO_GATEWAY[MOE_COMBO_ID] = {
    "seg_source": "yolo-obb",
    "pose_source": "moe",
    "needs_depth": True,
    "degraded": False,
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
      * `seg=yolo-obb & pose=gdrnpp` ODER `pose=GDRNPP`  (2-Achsen-Form S-010; das FE
        kann auf der pose-Achse die Gateway-id `gdrnpp` ODER das Label `GDRNPP` schicken)
    Ein leerer/fehlender pipeline-Param ohne seg/pose ist ebenfalls Pipeline A
    (Default, byte-identisch zu currentPipeline()→"gdrnpp")."""
    pid = (pipeline or "").strip()
    if seg or pose:
        # pose-Achse case-insensitiv: id 'gdrnpp' und Label 'GDRNPP' sind beide Pipeline A
        # (sonst faellt seg=yolo-obb&pose=GDRNPP faelschlich durch → InvalidCombo, weil
        # yolo-obb→gdrnpp NICHT in COMBO_TO_GATEWAY steht — es IST der Live-Monolith).
        return ((seg or "").strip() == PIPELINE_A_SEG
                and (pose or "").strip().lower() == PIPELINE_A_POSE)
    return pid in ("", PIPELINE_A_ID)


def resolve_combo_id(*, pipeline: str | None = None,
                     seg: str | None = None, pose: str | None = None) -> str:
    """Loest die Anfrage auf eine Kombi-id auf. Pipeline A → 'gdrnpp'. Die NICHT-A-Kombis
    werden entweder direkt per `pipeline=<combo_id>` ODER per `seg=&pose=` adressiert.

    Quelle der Wahrheit ist `COMBO_TO_GATEWAY` (= alle NICHT-A feasible-Kombis, T-155).
    Wirft InvalidCombo wenn (seg,pose) keine feasible-Kombi ist (→ 400 im kip_server)."""
    if is_pipeline_a(pipeline=pipeline, seg=seg, pose=pose):
        return PIPELINE_A_ID

    pid = (pipeline or "").strip()
    if pid and pid in COMBO_TO_GATEWAY:
        return pid

    # 2-Achsen-Form: (seg, pose) → Kombi-id ueber die feasible-Kombis nachschlagen.
    if seg or pose:
        s, p = (seg or "").strip(), (pose or "").strip()
        for c in _NON_A_COMBOS:
            # c["seg"] ist die Mesh-/FE-Seg-id (yolo-obb/yolo-seg/sam3); c["pose"] das
            # FE-Label ("FoundationPose"). Die Gateway-pose-id steckt im combo-id-Suffix.
            # Wir matchen tolerant auf Label ODER Gateway-id.
            gw = COMBO_TO_GATEWAY[c["id"]]
            if c["seg"] == s and (p == gw["pose_source"] or _pose_label_matches(c, p)):
                return c["id"]
        raise InvalidCombo(
            f"Ungueltige Kombi (seg={seg!r}, pose={pose!r}). Erlaubt sind die feasiblen "
            f"Kombis (combos.FEASIBLE_COMBOS) bzw. Pipeline A (seg=yolo-obb&pose=gdrnpp)."
        )

    raise InvalidCombo(
        f"Unbekannte Pipeline {pipeline!r}. Erlaubt: 'gdrnpp' (Pipeline A) oder eine der "
        f"NICHT-A-Kombi-ids {sorted(COMBO_TO_GATEWAY)}."
    )


def _pose_label_matches(combo: dict, pose: str) -> bool:
    """Tolerantes Match der FE-Pose-Achse gegen die feasible-Pose. Akzeptiert das Label
    ('FoundationPose','GigaPose-2D','GigaPose-3D','GDRNPP') ODER die Gateway-id."""
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
    """Kombi-id → Gateway-Routing {seg_source, pose_source, needs_depth, degraded}.

    `degraded=True` nur fuer die 2 GDRNPP-degraded-Kombis (yolo-seg/sam3 → gdrnpp);
    der Aufrufer (kip_server) MUSS es dann als Form-Feld `degraded=true` ans Gateway
    durchreichen (T-153-Opt-in: Maske bleibt → gdrnpp-svc AABB-aus-Maske-Fallback).

    Wirft InvalidCombo wenn combo_id keine NICHT-A-Kombi ist (Pipeline A laeuft NICHT
    ueber das Gateway — das prueft der Aufrufer vorher per is_pipeline_a)."""
    gw = COMBO_TO_GATEWAY.get(combo_id)
    if gw is None:
        raise InvalidCombo(
            f"Kombi {combo_id!r} ist keine Gateway-Kombi. NICHT-A-Kombis: "
            f"{sorted(COMBO_TO_GATEWAY)} (Pipeline A 'gdrnpp' laeuft ueber den Live-Pfad)."
        )
    return dict(gw)


# ── Gateway-/predict-Antwort → Welt-Frame-Eintraege (geteilte Iteration) ──────
def _gateway_world_entries(gateway_resp: dict, *, R_w2c, t_w2c, table_origin,
                           with_bbox: bool, start_inst_id: int = 0,
                           snap: bool = True, warn=None) -> list:
    """DER eine Iterations-/Mapping-Schritt der Gateway-`instances` → Welt-Frame-Eintraege.

    Geteilt von `gateway_predict_to_pose_result` (Contract-Doc, Eval) UND
    `gateway_instances_to_viewer_preds` (Viewer-Overlay, Sim/Real) — beide brauchen
    GENAU diese Schleife (Skip ohne Pose §3, §6-Scope-Filter, tcamobj_to_world_entry),
    nur die Ziel-Form unterscheidet sich. So gibt es EINE Stelle fuer die Cam→Welt-Naht.

    with_bbox: True → bbox_2d aus mask_b64 ableiten (Contract braucht es); False → [0,0,0,0]
    (der Viewer nutzt keine 2D-Box auf der Pred). start_inst_id: fortlaufende iids ab hier
    (der Sim/Viewer haengt Preds hinter die GT-iids)."""
    from .composed import tcamobj_to_world_entry, _mask_b64_to_bbox

    entries, iid = [], start_inst_id
    for inst in gateway_resp.get("instances", []):
        cls, T = inst.get("class"), inst.get("T_cam_obj")
        if cls is None or T is None:
            continue                         # Instanz ohne Pose (Gateway-Skip §3) → weg
        conf = inst.get("conf")
        mask = inst.get("mask_b64")
        entry = tcamobj_to_world_entry(      # §1+§6, geteilt mit ComposedPipeline
            cls=cls, T_cam_obj=T, R_w2c=R_w2c, t_w2c=t_w2c, table_origin=table_origin,
            instance_id=iid,
            confidence=(conf if conf is not None else 1.0),
            bbox_2d=(_mask_b64_to_bbox(mask) if (with_bbox and mask) else [0, 0, 0, 0]),
            snap=snap, warn=warn,
        )
        if entry is not None:                # None = Klasse out-of-scope (§6) → still weg
            entries.append(entry)
            iid += 1
    return entries


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

    cam = camera or {}
    R_w2c = cam.get("cam_R_w2c")
    t_w2c = cam.get("cam_t_w2c")
    if R_w2c is None or t_w2c is None:
        raise ValueError(
            "gateway_predict_to_pose_result braucht camera={cam_R_w2c,cam_t_w2c} "
            "(Welt-Extrinsics) — die Pose-Stage liefert nur den Cam-Frame T_cam_obj."
        )
    to = list(table_origin)
    entries = _gateway_world_entries(
        gateway_resp, R_w2c=R_w2c, t_w2c=t_w2c, table_origin=to,
        with_bbox=True, snap=snap, warn=warn)
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

    gateway/health (gateway/app.py:319-340) hat die Form
      {ok, yolo:{ok}, fp:{ok}, gigapose:{...}, sam3:{ok}, yolo_obb:{ok}, gdrnpp:{ok}}.
    Wir mappen ALLE Health-Knoten auf die source-ids, die COMBO_TO_GATEWAY +
    pipelines_status nutzen — alle 4 Seg-Quellen UND alle 4 Pose-Modelle (T-155):
      seg:  'yolo' (yolo-svc), 'sam3' (sam3-svc), 'yolo-obb' (yolo-obb-svc, OBB-Detektor)
      pose: 'foundationpose' (fp-svc), 'gigapose_rgbd'+'gigapose_rgb' (beide gigapose-svc),
            'gdrnpp' (gdrnpp-svc, nativer GDRNPP-Pose-Service fuer die degraded-Kombis).

    Vorher fehlten 'yolo-obb' + 'gdrnpp' → die 3 yolo-obb-Mesh-Kombis fielen faelschlich
    auf service_down, obwohl yolo-obb-svc im Gateway-/health ok war (Sam T-154-Befund).
    Der Gateway-Health-Knoten heisst `yolo_obb` (Underscore); unsere source-id ist
    `yolo-obb` (Bindestrich, == SEG_SOURCES-id / Mesh-Name) — hier explizit gemappt.
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
    yolo_obb = _ok(gateway_health.get("yolo_obb"))
    gdrnpp = _ok(gateway_health.get("gdrnpp"))
    return {
        # Seg-Quellen (4):
        "yolo": yolo,
        "sam3": sam3,
        "yolo-obb": yolo_obb,
        # Pose-Modelle (4):
        "foundationpose": fp,
        "gigapose_rgbd": giga,
        "gigapose_rgb": giga,
        "gdrnpp": gdrnpp,
    }


def available_combo_ids(pipelines: list) -> list:
    """Aus einer pipelines_status()-Liste die available-Kombi-ids (fuer /api/compare-Gating)."""
    return [p["id"] for p in pipelines if p.get("available")]


# ── Result-Metadaten: welche Kombi/welches Modell wurde gefahren (S-014 / T-140) ──
# Lena (T-164) zeigt im Result an, was wirklich lief. Single-Source = FEASIBLE_COMBOS;
# funktioniert auch fuer Pipeline A ('gdrnpp', der Live-Monolith) → die Sim/Real/Upload-
# Pfade haengen used_combo/used_seg/used_pose/modality an JEDES Infer-Result, egal ob
# Live-Pfad oder Gateway-Proxy.
_COMBOS_BY_ID = {c["id"]: c for c in FEASIBLE_COMBOS}


def _modality(needs_depth: bool) -> str:
    """Input-Modalitaet (FE-Spalte, T-159): RGBD wenn die Pose Depth braucht, sonst RGB.
    Identisch zu batch_eval._modality (Single-Source = needs_depth, CONTRACT.md §5)."""
    return "RGBD" if needs_depth else "RGB"


def combo_result_meta(combo_id: str) -> dict:
    """Die `used_*`-Result-Felder einer gefahrenen Kombi (S-014 / T-140, Lena T-164).

    -> {used_combo, used_seg, used_pose, modality, needs_depth, degraded}.
      * used_combo : Kombi-id ('gdrnpp' = Pipeline A, sonst die Gateway-Kombi-id).
      * used_seg   : Seg-Quelle (FE-Name: yolo-obb | yolo-seg | sam3).
      * used_pose  : Pose-Modell (FE-Label: GDRNPP | FoundationPose | GigaPose-2D/3D).
      * modality   : "RGB" | "RGBD" (aus needs_depth; FE Input-Spalte, T-159).
      * needs_depth, degraded : die Routing-Flags (Single-Source FEASIBLE_COMBOS).

    Wirft InvalidCombo wenn combo_id keine feasible-Kombi ist (Pipeline A inklusive)."""
    if combo_id == MOE_COMBO_ID:
        # MoE ist KEINE Matrix-Kombi (s.o.) — eigene Meta. modality benennt
        # beide Zweige ehrlich (FE Input-Spalte / Inferiert-mit-Zeile).
        return {"used_combo": MOE_COMBO_ID, "used_seg": "yolo-obb",
                "used_pose": "MoE (RGB im IR-Schatten / GigaPose-3D)",
                "modality": "RGB+RGBD", "needs_depth": True, "degraded": False}
    c = _COMBOS_BY_ID.get(combo_id)
    if c is None:
        raise InvalidCombo(
            f"Unbekannte Kombi {combo_id!r}. Erlaubt: feasible-Kombi-ids "
            f"{sorted(_COMBOS_BY_ID)} (inkl. 'gdrnpp' = Pipeline A)."
        )
    return {
        "used_combo": c["id"],
        "used_seg": c["seg_id"],
        "used_pose": c["pose"],          # FE-Label (GDRNPP | FoundationPose | GigaPose-…)
        "modality": _modality(bool(c["needs_depth"])),
        "needs_depth": bool(c["needs_depth"]),
        "degraded": bool(c["degraded"]),
    }


# ── Gateway-/predict-Antwort → Viewer-OVERLAY-Pred-Eintraege (color="pred") ──────
# Der Sim/Real-Web-Viewer nutzt NICHT das frozen Contract-Doc (das ist die Eval-Form),
# sondern die OVERLAY-Form (kip_server._bop_pose_to_result): {instance_id, part, face,
# confidence, t_world, R_world, upright, color}. Fuer die NICHT-A-Kombis kommt die Pose
# vom Gateway (T_cam_obj) statt vom :8078-Worker — wir mappen sie ueber DIESELBE
# Cam→Welt-Mathematik (composed.tcamobj_to_world_entry) in genau diese Overlay-Form,
# damit der Viewer Gateway-Pred und Live-Pred byte-gleich rendert (rot). GT (blau) bleibt
# der Worker-/scene_gt-Pfad — das Gateway liefert NIE GT (image-only, ADR-020).
def gateway_instances_to_viewer_preds(gateway_resp: dict, *, R_w2c, t_w2c,
                                      table_origin, start_inst_id: int = 1,
                                      snap: bool = True, warn=None) -> list:
    """Gateway-`/predict`-Instanzen (T_cam_obj) → Viewer-Overlay-Pred-Eintraege (rot).

    Jeder Eintrag hat dieselbe Form wie kip_server._bop_pose_to_result(..., color="pred"):
    {instance_id, part, face:"—", confidence, t_world, R_world, upright, color:"pred"}.
    Klassen ausserhalb des 2-Klassen-Scope werden still gefiltert (§6, None aus
    tcamobj_to_world_entry). Reused die ComposedPipeline-Mathematik → keine zweite
    Mapping-Implementierung, byte-gleich zum Eval-/Composed-Pfad.

    R_w2c/t_w2c: Welt-Extrinsics als 3x3 / 3-Vektor (mm), wie scene_camera.json. Sim/Real
    liefern die feste Zivid-Kamera (bzw. die Live-Isaac-Kamera des Frames)."""
    import numpy as np

    # Geteilte Cam→Welt-Iteration (with_bbox=False: der Viewer braucht keine 2D-Box).
    entries = _gateway_world_entries(
        gateway_resp, R_w2c=R_w2c, t_w2c=t_w2c, table_origin=table_origin,
        with_bbox=False, start_inst_id=start_inst_id, snap=snap, warn=warn)
    preds = []
    for entry in entries:
        Rw = np.asarray(entry["R_world"], float).reshape(3, 3)
        preds.append({
            "instance_id": int(entry["instance_id"]),
            "part": entry["part"],
            "face": "—",
            "confidence": float(entry["confidence"]),
            "t_world": [float(x) for x in entry["t_world"]],
            "R_world": [float(x) for x in Rw.reshape(9)],
            # upright-Heuristik identisch zu _bop_pose_to_result: Objekt-Y-Achse ~ Welt-Z.
            "upright": bool(abs(float(Rw[2, 1])) > 0.6),
            "color": "pred",
        })
    return preds
