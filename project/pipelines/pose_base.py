"""pose_base.py — PoseAdapter: dünner HTTP-Client gegen einen Pose-Service (`POST /pose`).

Stage 2 der 2-stufigen Multi-Pipeline (S-003). Ein Pose-Service (gdrnpp, foundationpose,
gigapose) erfüllt EXAKT den frozen `POST /pose`-Contract (`project/mesh/CONTRACT.md` §3)
und gibt `T_cam_obj` aus (§1: OpenCV-cam, Meter, mesh→cam, 4×4 row-major). Dieser Adapter
spricht den Contract; er rechnet NICHT in den Welt-Frame um — das ist Sache der
ComposedPipeline (composed.py).

Design (identisch zu seg_base):
  * stdlib-only Transport (`urllib.request`), kein `requests`-Pin.
  * `post_json` ist die einzige Mock-Naht.
  * `needs_depth` steuert, ob `depth_b64` mitgesendet wird (Gateway-Gating, §3/§5).
  * `pipeline` ("rgbd"|"rgb") nur für GigaPose (Kabsch-Gate, §5); `gdrnpp` liest `obb`
    statt mask (§4) — der Adapter reicht beides durch (`Detection.to_instance`).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


# ── Wert-Typ: eine geschätzte Pose (Contract §3 Response) ─────────────────────
@dataclass
class Pose:
    """Eine Pose aus `POST /pose`. Felder = frozen Contract §3.

    id        : int = instances[].id, durchgereicht.
    cls       : str (lowercase Mesh-Klasse, §6). Attribut `cls` (nicht Keyword `class`).
    T_cam_obj : 4×4 row-major (Liste von 4 Listen à 4 floats). §1: OpenCV-cam, Meter,
                mesh→cam. DER einzige Pose-Frame im Mesh.
    score     : optional float (fp lässt es weg; gigapose liefert es).
    stage     : optional str — NUR gigapose ∈ {coarse, refined, refined+kabsch}.
    """
    id: int
    cls: str
    T_cam_obj: "list[list[float]]"
    score: "float | None" = None
    stage: "str | None" = None

    @classmethod
    def from_wire(cls, d: dict) -> "Pose":
        T = d["T_cam_obj"]
        # Defensiv: Form 4×4 erzwingen (Contract §1). Ein Service der eine flache 16er-
        # Liste liefert ist contract-fehlerhaft, aber wir reshapen tolerant.
        rows = _as_4x4(T)
        return cls(
            id=int(d["id"]),
            cls=str(d.get("class", d.get("class_name", ""))),   # gigapose akzeptiert class_name
            T_cam_obj=rows,
            score=(float(d["score"]) if d.get("score") is not None else None),
            stage=(str(d["stage"]) if d.get("stage") is not None else None),
        )


def _as_4x4(T) -> "list[list[float]]":
    """Normalisiert T_cam_obj auf 4 Zeilen à 4 floats (row-major). Akzeptiert sowohl
    [[..],[..],[..],[..]] als auch flat-16; lehnt alles andere ab."""
    if isinstance(T, (list, tuple)) and len(T) == 4 and all(
        isinstance(r, (list, tuple)) and len(r) == 4 for r in T
    ):
        return [[float(x) for x in row] for row in T]
    flat = list(T)
    if len(flat) == 16:
        return [[float(flat[r * 4 + c]) for c in range(4)] for r in range(4)]
    raise ValueError(f"T_cam_obj muss 4×4 oder flat-16 sein, got len={len(flat)}")


class PoseAdapter:
    """Pose-Service-Client. Spricht `POST /pose` (Contract §3), drop-in pro Estimator.

    Unterklassen/Instanzen setzen `base_url`, `needs_depth`, ggf. `pipeline` + `uses_obb`.
    `estimate()` ist der einzige Einstieg; `post_json()` die einzige Transport-Naht.
    """

    # ── Metadaten (überschreibbar) ──
    id: str = "pose-abstract"
    name: str = "Abstract Pose"
    needs_depth: bool = False              # §5: FP immer, GigaPose-3D(rgbd); sonst False
    pipeline: "str | None" = None          # NUR gigapose: "rgbd"|"rgb" (Kabsch-Gate, §5)
    uses_obb: bool = False                 # NUR gdrnpp: liest obb statt mask (§4)

    def __init__(self, base_url: str = "", *, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    # ── Transport (die EINZIGE Naht für Mocks) ──
    def post_json(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    # ── Contract §3: POST /pose ──
    def estimate(self, rgb_b64: str, K, detections, *,
                 depth_b64: str | None = None,
                 iterations: int | None = None,
                 hypotheses: int | None = None) -> "list[Pose]":
        """`POST /pose` → Liste von Posen (Contract §3).

        Args:
          rgb_b64    : base64-PNG uint8 RGB. PFLICHT.
          K          : [fx,0,cx,0,fy,cy,0,0,1] flat 9, row-major. PFLICHT (§3).
          detections : Liste von seg_base.Detection (Stage-1-Output). Wird zu
                       instances[] gemappt (mask UND obb durchgereicht, §4).
          depth_b64  : base64-PNG uint16 mm. Nur gesendet wenn `self.needs_depth`
                       (Gateway-Gating §3/§5). RGB-only-Pfad sendet KEIN Depth.
          iterations : optional Refiner-Iterationen (default Service-seitig 5).
          hypotheses : optional coarse top-K (NUR gigapose, default 5).

        Skip-Verhalten (frozen §3): unbekannte Klasse / leere Maske → Instanz wird
        Service-seitig still übersprungen → die Pose-Liste kann KÜRZER sein als
        `detections`. Das ist KEIN Fehler.
        """
        instances = [d.to_instance() for d in detections]
        if not instances:
            return []                       # nichts zu posieren — leer ist valide

        payload: dict = {
            "rgb_b64": rgb_b64,
            "K": list(K),
            "instances": instances,
        }
        # depth NUR wenn dieser Estimator es braucht (needs_depth-Gating, §5).
        if self.needs_depth and depth_b64 is not None:
            payload["depth_b64"] = depth_b64
        if self.pipeline is not None:
            payload["pipeline"] = self.pipeline             # gigapose rgbd|rgb
        if iterations is not None:
            payload["iterations"] = int(iterations)
        if hypotheses is not None:
            payload["hypotheses"] = int(hypotheses)

        resp = self.post_json("/pose", payload)
        return [Pose.from_wire(p) for p in resp.get("poses", [])]

    def health(self) -> dict:
        url = self.base_url + "/health"
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def __repr__(self) -> str:
        return (f"<PoseAdapter {self.id!r} @ {self.base_url!r} "
                f"needs_depth={self.needs_depth} pipeline={self.pipeline!r}>")
