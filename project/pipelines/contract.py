"""contract.py — das eingefrorene pose_result-Contract-Gate fuer Pipelines.

Max-Entscheidung (2026-06-01): fremde Pipelines muessen das EXAKTE bestehende
pose_result.schema.json erfuellen (keine Aufweichung, additionalProperties:false,
face/upright Pflicht). Wir verwenden den autoritativen Validator wieder, der bereits
in e2e_infer.py liegt, statt ihn neu zu implementieren.

⚠️ Runtime-Abweichung (wichtig fuer Adapter-Autoren): der LIVE-Web-Viewer
(kip_server._bop_pose_to_result) emittiert eine *Viewer-Runtime*-Variante des Docs,
die ein `color`-Feld (gt/pred) ergaenzt und bbox_2d weglaesst. Das ist die OVERLAY-
Form des Viewers, NICHT der eingefrorene Contract. Ein Adapter fuer den Vergleichs-
Harness muss gegen das FROZEN Schema konform sein (validate() unten). Die gt/pred-
Einfaerbung ist eine Vergleichs-Overlay-Sache, die der Harness/Viewer separat
behandelt — nicht der Pipeline-Contract.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# Autoritative Gates + Doc-Builder aus e2e_infer wiederverwenden (DRY).
from e2e_infer import (  # noqa: E402
    check_pose_result,
    _check_with_jsonschema,
    jsonschema_available,
    build_pose_result,
    SCHEMA_VERSION,
    COORD_CONVENTION,
    TABLE_ORIGIN_SCENE,
)

__all__ = [
    "validate", "assert_valid", "assemble_doc", "is_valid",
    "jsonschema_available", "SCHEMA_VERSION", "COORD_CONVENTION", "TABLE_ORIGIN_SCENE",
]


def validate(doc) -> list:
    """Liste der Contract-Verstoesse (leer = gueltig). stdlib-Gate immer +
    jsonschema-Gate wenn verfuegbar (Draft 2020-12, additionalProperties:false)."""
    return check_pose_result(doc) + _check_with_jsonschema(doc)


def is_valid(doc) -> bool:
    return not validate(doc)


def assert_valid(doc) -> dict:
    """Wirft ValueError mit allen Verstoessen, sonst gibt doc zurueck."""
    errs = validate(doc)
    if errs:
        raise ValueError("pose_result verletzt den Contract:\n  - " + "\n  - ".join(errs))
    return doc


def assemble_doc(source_image: str, entries: list, table_origin=TABLE_ORIGIN_SCENE) -> dict:
    """Baut ein frozen-schema-valides pose_result-Doc aus Welt-Frame-Eintraegen.

    Komfort fuer fremde Adapter, die nur Welt-Posen liefern: face/upright werden
    analytisch via bop_adapter.face_and_upright_from_R abgeleitet (das fremde
    System muss unser Registry-'face'-Konzept nicht kennen).

    entries: Liste von dicts mit Keys:
      instance_id:int, part:str (muss CAD/Registry-Name treffen),
      R_world:(9 floats, world=R@body), t_world:(3 floats, m rel. Tisch),
      confidence:float 0..1, bbox_2d:(4 ints [x0,y0,x1,y1]).
      Optional face:str / upright:bool ueberschreiben die Ableitung.

    Hinweis: confidence wird defensiv auf 0..1 geklemmt (das frozen Schema verlangt
    den Bereich). Liefert ein Adapter Werte ausserhalb, korrigiert assemble_doc sie
    still — pruefe deinen Adapter, wenn du saubere Konfidenzen erwartest.
    """
    import numpy as np
    from bop_adapter import face_and_upright_from_R

    aligned = []
    for e in entries:
        R = np.asarray(e["R_world"], float).reshape(3, 3)
        face_d, upright_d = face_and_upright_from_R(R, part=e.get("part"))
        aligned.append({
            "instance_id": int(e["instance_id"]),
            "part": e["part"],
            "face_name": e.get("face", face_d),
            "R_world": [float(x) for x in np.asarray(e["R_world"], float).reshape(9)],
            "t_world": [float(x) for x in e["t_world"]],
            "confidence": float(max(0.0, min(1.0, e["confidence"]))),
            "bbox_2d": [int(x) for x in e["bbox_2d"]],
            "upright": bool(e.get("upright", upright_d)),
        })
    return build_pose_result(source_image, aligned, table_origin=table_origin)
