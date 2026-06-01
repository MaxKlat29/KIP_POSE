"""adapter.py — TEMPLATE fuer die Anbindung einer KOMPLETT ANDEREN Pose-Pipeline.

So gehst du pro Pipeline vor:
  1. Kopiere das ganze ``_template/``-Verzeichnis nach ``pipelines/<deine_id>/``.
  2. Lege das ROHE, unformatierte externe Python-Projekt (von Max) in ``<id>/vendor/`` ab.
  3. Pinne dessen (isolierte) Dependencies in ``<id>/requirements.txt`` — fremde
     Projekte bringen oft kollidierende Deps mit; die laufen in einer EIGENEN venv
     (siehe README), nicht in project/requirements.txt.
  4. Implementiere `infer()` unten: rufe in vendor/ hinein, mappe dessen rohen Output
     auf den EINGEFRORENEN pose_result-Contract. Nutze pipelines.contract.assemble_doc
     (leitet face/upright analytisch ab, falls die fremde Pipeline kein face-Konzept hat).
  5. Aktiviere die Registrierung am Dateiende (register-Zeile entkommentieren) — dann
     erscheint die Pipeline in /api/pipelines, im Dropdown und im Vergleichs-Harness.

Solange keine echte Pipeline angebunden ist, wirft `infer()` bewusst NotImplementedError
("noch nix reinsetzen" — die Pipelines kommen von Max). `available` bleibt False.

Vollstaendige Anleitung: ../../docs/PIPELINE_INTEGRATION.md
"""
from __future__ import annotations

import pathlib

from pipelines.base import PipelineAdapter
from pipelines import contract  # noqa: F401  (contract.assemble_doc / assert_valid beim Implementieren nutzen)

_HERE = pathlib.Path(__file__).resolve().parent
VENDOR = _HERE / "vendor"   # <- hier das rohe externe Python-Projekt ablegen


class TemplatePipeline(PipelineAdapter):
    id = "template"                                  # <- eindeutige id setzen (z.B. "pipeline_x")
    name = "Template (leer)"                         # <- Label fuers Dropdown
    description = "Platzhalter. Hier wird ein von Max geliefertes rohes Python-Projekt angebunden."

    @property
    def available(self) -> bool:
        # Auf einen echten Check umstellen sobald angebunden, z.B.:
        #   return (VENDOR / "<entrypoint>.py").exists()
        return False

    def infer(self, image_path: str, camera: dict, table_origin: list) -> dict:
        # ── BEIM IMPLEMENTIEREN ungefaehr so: ─────────────────────────────────
        #   import sys; sys.path.insert(0, str(VENDOR))
        #   from <vendor_entrypoint> import run_my_pipeline      # rohes Projekt
        #   raw = run_my_pipeline(image_path, camera)            # was auch immer es liefert
        #   entries = [{
        #       "instance_id": i,
        #       "part": <CAD/Registry-Name, z.B. "Anker_Lang">,
        #       "R_world": <9 floats, world = R @ body>,
        #       "t_world": <3 floats, Meter, rel. Tisch-Nullpunkt>,
        #       "confidence": <0..1>,
        #       "bbox_2d": <[x0,y0,x1,y1] ints>,
        #   } for i, det in enumerate(raw)]
        #   return contract.assemble_doc(image_path, entries, table_origin)
        # assemble_doc leitet face/upright ab + validiert gegen das frozen Schema.
        raise NotImplementedError(
            "Diese Pipeline ist noch nicht angebunden. Max liefert das rohe Python-"
            "Projekt -> in vendor/ ablegen, hier im Adapter auf den pose_result-"
            "Contract mappen (siehe docs/PIPELINE_INTEGRATION.md)."
        )


# Erst registrieren wenn implementiert (available=True) — verhindert, dass leere
# Stubs Dropdown + Vergleichs-Matrix fluten:
#
# from pipelines.registry import register
# register(TemplatePipeline())
