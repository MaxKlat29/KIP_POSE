"""seg_base.py — SegAdapter: dünner HTTP-Client gegen einen Seg-Service (`POST /segment`).

Stage 1 der 2-stufigen Multi-Pipeline (S-003). Ein Seg-Service (yolo-seg, yolo-obb,
sam3) ist ein drop-in austauschbarer Maskenlieferant; ALLE erfüllen EXAKT den frozen
`POST /segment`-Contract (`project/mesh/CONTRACT.md` §2). Dieser Adapter spricht diesen
Contract — er kennt KEINE Service-Interna.

Design:
  * **stdlib-only Transport** (`urllib.request`). `requests` ist bewusst NICHT in
    project/requirements.txt (die schweren Box-Stacks haben eigene venvs) — ein neuer
    Pin würde `import pipelines` auf der lokalen venv brechen. urllib reicht für einen
    JSON-POST völlig.
  * **`post_json` ist überschreibbar** — die Unit-Tests (S-003) injizieren einen
    Mock-Service ohne echten Socket/GPU.
  * Detection ist ein schmaler Wert-Typ (id/class/conf/mask_b64 + optional obb), exakt
    die `detections[]`-Form aus dem Contract.

KEINE Symmetrie/Welt-Logik hier — das ist Sache der ComposedPipeline (composed.py).
Der Seg-Adapter reicht nur durch.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


# ── Wert-Typ: eine Seg-Detection (Contract §2 Response) ───────────────────────
@dataclass
class Detection:
    """Eine Instanz aus `POST /segment`. Felder = frozen Contract §2.

    id       : int, stabil über den Request (das Gateway/die Pose-Stage mergt per id).
    cls      : str ∈ {anker_kurz, anker_lang} — **lowercase** (Mesh-Konvention, §6).
               (Attribut heißt `cls`, NICHT `class` — `class` ist ein Python-Keyword.)
    conf     : float 0..1.
    mask_b64 : base64-PNG single-channel 0/255, VOLLE-Bild-Maske (kein Crop). PFLICHT.
    obb      : optional [cx,cy,w,h,theta] (θ rad) — NUR yolo-obb liefert es (§4).
    """
    id: int
    cls: str
    conf: float
    mask_b64: str
    obb: "list[float] | None" = None

    @classmethod
    def from_wire(cls, d: dict) -> "Detection":
        """Aus einem rohen detections[]-dict (Contract §2). Defensiv: fehlt mask_b64,
        ist das ein Contract-Verstoß des Service (nicht still tolerieren)."""
        if "mask_b64" not in d or not d["mask_b64"]:
            raise ValueError(
                f"Seg-Detection ohne mask_b64 verletzt Contract §2 (id={d.get('id')!r}): "
                "mask_b64 ist die einzige Pose-Init."
            )
        obb = d.get("obb")
        return cls(
            id=int(d["id"]),
            cls=str(d["class"]),                 # Wire-Feld heißt "class" (lowercase Mesh-Name)
            conf=float(d.get("conf", 1.0)),
            mask_b64=str(d["mask_b64"]),
            obb=[float(x) for x in obb] if obb is not None else None,
        )

    def to_instance(self) -> dict:
        """Als Pose-Stage-Instance-dict (Contract §3 instances[]). Reicht mask UND obb
        durch — der Pose-Adapter entscheidet je nach Modell, was er nutzt (§4)."""
        inst = {"id": self.id, "class": self.cls, "mask_b64": self.mask_b64}
        if self.obb is not None:
            inst["obb"] = list(self.obb)
        return inst


class SegAdapter:
    """Seg-Service-Client. Spricht `POST /segment` (Contract §2), drop-in pro Quelle.

    Unterklassen/Instanzen setzen `base_url` + Metadaten. `segment()` ist der einzige
    Einstieg; `post_json()` ist die einzige Transport-Naht (für Mocks überschreibbar).
    """

    # ── Metadaten (überschreibbar) ──
    id: str = "seg-abstract"
    name: str = "Abstract Seg"
    needs_prompts: bool = False     # nur sam3 nutzt `prompts` (§2)

    def __init__(self, base_url: str = "", *, timeout: float = 60.0):
        # Trailing-Slash normalisieren, damit base_url + "/segment" sauber bleibt.
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    # ── Transport (die EINZIGE Naht für Mocks) ──
    def post_json(self, path: str, payload: dict) -> dict:
        """JSON-POST gegen `base_url + path`. stdlib-urllib, kein `requests`-Pin.

        Tests überschreiben diese Methode (Mock-Service) — dann fließt KEIN echter
        Socket. Produktiv ist es ein simpler JSON-Roundtrip."""
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (intern, kein User-Input)
            return json.loads(resp.read().decode("utf-8"))

    # ── Contract §2: POST /segment ──
    def segment(self, rgb_b64: str, K=None, *, prompts: dict | None = None,
                depth_b64: str | None = None) -> "list[Detection]":
        """`POST /segment` → Liste von Detections (Contract §2).

        Args:
          rgb_b64 : base64-PNG uint8 RGB. PFLICHT.
          K       : optional [fx,0,cx,0,fy,cy,0,0,1] (flat 9) — nur Quellen die K
                    brauchen (sam3 depth-PCA-Upgrade). Sonst weglassen.
          prompts : optional {class: "concept text"} — NUR sam3 (yolo ignoriert es).
          depth_b64: optional base64-PNG uint16 mm — NUR sam3 2-pass-Upgrade.

        Nur gesetzte optionale Felder werden gesendet (frozen Contract: optionale
        Felder sind weglassbar — ältere yolo/sam3-Services lesen sie sowieso nicht).
        """
        payload: dict = {"rgb_b64": rgb_b64}
        if prompts is not None and self.needs_prompts:
            payload["prompts"] = prompts
        if K is not None:
            payload["K"] = list(K)
        if depth_b64 is not None:
            payload["depth_b64"] = depth_b64

        resp = self.post_json("/segment", payload)
        dets = resp.get("detections", [])
        return [Detection.from_wire(d) for d in dets]

    def health(self) -> dict:
        """`GET /health` → {"ok": true} (Contract §2). Best-effort, wirft bei Netzfehler."""
        url = self.base_url + "/health"
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def __repr__(self) -> str:
        return f"<SegAdapter {self.id!r} @ {self.base_url!r}>"
