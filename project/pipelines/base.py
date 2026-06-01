"""base.py — PipelineAdapter: die pluggable Schnittstelle fuer den Pipeline-Vergleich.

Eine Pipeline ist ein KOMPLETTES, eigenstaendiges System, das aus einem Szenen-Bild
6D-Posen macht. Wir fixieren nur den *Contract an der Grenze*, nicht die Interna.

CONTRACT (eingefroren, Max-Entscheidung 2026-06-01):
  input :  image_path    — Pfad zum Szenen-RGB (Zivid-Format)
           camera        — dict {cam_K[9], cam_R_w2c[9], cam_t_w2c[3] (mm)}
                           (= ein BOP scene_camera-Eintrag). Pipelines die ihre
                           eigene Kamera aufloesen (z.B. die Referenz via e2e_infer)
                           duerfen das ignorieren.
           table_origin  — [x,y,z] Welt-Nullpunkt der Tischebene in Metern
  output:  ein pose_result-dict, das gegen project/pose_result.schema.json validiert
           (instance_id, part, face, R_world[9], t_world[3], confidence, bbox_2d[4],
            upright + meta). Pruefung via pipelines.contract.validate(doc).

Jede fremde Pipeline wird als ROHES Python-Projekt unter ``<id>/vendor/`` abgelegt
und durch eine adapter.py (Subklasse von PipelineAdapter) auf diesen Contract
gemappt. Felder die das fremde System nicht kennt (face/upright) werden analytisch
abgeleitet (pipelines.contract.assemble_doc -> bop_adapter.face_and_upright_from_R).
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Output einer Pipeline fuer EIN Bild + die operative Telemetrie, die der
    Vergleichs-Harness fuer Latenz / Coverage / Robustheit braucht."""
    pipeline_id: str
    doc: dict | None          # schema-valides pose_result, oder None bei Fehler
    latency_ms: float
    n_results: int
    ok: bool
    error: str = ""

    def as_summary(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "n_results": self.n_results,
            "error": self.error,
        }


class PipelineAdapter(abc.ABC):
    """Basisklasse fuer eine vergleichbare Pose-Pipeline.

    Unterklassen ueberschreiben die Metadaten-Klassenattribute + `available` +
    `infer`. Der Harness ruft `run_timed`, nie `infer` direkt.
    """

    # ── Metadaten (als Klassenattribute ueberschreiben) ──
    id: str = "abstract"
    name: str = "Abstract Pipeline"
    description: str = ""

    @property
    def available(self) -> bool:
        """True nur wenn diese Pipeline tatsaechlich lauffaehig ist (Deps +
        Artefakte vorhanden). Leere Stubs / noch-nicht-gelieferte Pipelines geben
        False -> im Dropdown disabled, der Harness ueberspringt sie mit explizitem
        Log (KEIN stilles Weglassen)."""
        return False

    @abc.abstractmethod
    def infer(self, image_path: str, camera: dict, table_origin: list) -> dict:
        """Pipeline auf EIN Bild anwenden. MUSS ein pose_result-dict liefern, das
        pipelines.contract.validate() besteht. Bei Fehler: Exception werfen."""
        raise NotImplementedError

    def run_timed(self, image_path: str, camera: dict, table_origin: list) -> PipelineResult:
        """infer() + Wall-Clock-Latenz + Coverage + Crash-Capture. Der Harness
        ruft DIESE Methode (nie infer direkt) — so wird ein Crash einer Pipeline
        zu einem ok=False-Datenpunkt statt den ganzen Vergleich abzubrechen."""
        t0 = time.perf_counter()
        try:
            doc = self.infer(image_path, camera, table_origin)
            dt = (time.perf_counter() - t0) * 1000.0
            # Output gegen den eingefrorenen Contract pruefen: ein Adapter, der kein
            # dict / ein schema-invalides Doc liefert, ist ein Robustheits-Datenpunkt
            # (ok=False) — NICHT ein falsch-positives ok=True, das den Vergleich
            # verfaelscht. Lazy-Import haelt base leichtgewichtig + zyklenfrei.
            from . import contract
            errs = contract.validate(doc) if isinstance(doc, dict) else ["infer() lieferte kein dict"]
            if errs:
                return PipelineResult(self.id, None, dt, 0, False, "contract: " + "; ".join(errs[:3]))
            return PipelineResult(self.id, doc, dt, len(doc.get("results", [])), True, "")
        except Exception as e:  # noqa: BLE001 — bewusst breit: ein Pipeline-Crash
            dt = (time.perf_counter() - t0) * 1000.0  # ist ein Robustheits-Datenpunkt
            return PipelineResult(self.id, None, dt, 0, False, repr(e))

    def __repr__(self) -> str:
        return f"<PipelineAdapter {self.id!r} available={self.available}>"
