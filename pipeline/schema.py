#!/usr/bin/env python3
"""Validate a ``pose_result`` dict against the frozen contract (S-006).

The pipeline asserts its own output is contract-valid before writing it. Two
gates, both run when available:

  1. a pure-stdlib validator (:func:`check_pose_result`) implementing the
     contract invariants directly — always available, so the E2E run is green on
     a bare ``python3`` with no extra deps;
  2. the JSON-Schema in ``docs/pose_result.schema.json`` via ``jsonschema``
     Draft202012Validator — used only if ``jsonschema`` is importable (e.g. under
     ``uv run --with jsonschema``), giving the strict schema gate as a bonus.

Both must pass when both are available. ``validate_or_raise`` raises
:class:`PoseResultInvalid` listing every problem; ``run_pipeline`` calls it
before persisting.
"""
from __future__ import annotations

import json
import os

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "pose_result.schema.json",
)

SCHEMA_VERSION = "1.0.0"


class PoseResultInvalid(ValueError):
    """Raised when a pose_result fails the contract. Carries the error list."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("pose_result is not contract-valid:\n  - " + "\n  - ".join(errors))


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check_pose_result(doc: dict) -> list[str]:
    """Return a (possibly empty) list of contract violations — never raises.

    Mirrors the frozen invariants in docs/pose_result.schema.json:
      meta{source_image, table_origin[3], units="m", coordinate_convention,
           schema_version=semver}
      results[]{instance_id>=0 int, part str, face str, R_world[9 floats],
                t_world[3 floats], confidence 0..1, bbox_2d[4 ints, x0<=x1,
                y0<=y1, >=0], upright bool}
    """
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["top level must be an object"]

    for key in ("meta", "results"):
        if key not in doc:
            errors.append(f"missing required top-level key '{key}'")

    meta = doc.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
    else:
        sm = meta.get("source_image")
        if not isinstance(sm, str) or not sm:
            errors.append("meta.source_image must be a non-empty string")
        to = meta.get("table_origin")
        if not (isinstance(to, list) and len(to) == 3 and all(_is_number(v) for v in to)):
            errors.append("meta.table_origin must be 3 numbers")
        if meta.get("units") != "m":
            errors.append("meta.units must be 'm'")
        if not isinstance(meta.get("coordinate_convention"), str) or not meta.get("coordinate_convention"):
            errors.append("meta.coordinate_convention must be a non-empty string")
        sv = meta.get("schema_version")
        if not (isinstance(sv, str) and sv.count(".") == 2 and all(p.isdigit() for p in sv.split("."))):
            errors.append("meta.schema_version must be semver 'X.Y.Z'")

    results = doc.get("results")
    if not isinstance(results, list):
        errors.append("results must be an array")
        return errors

    for i, r in enumerate(results):
        p = f"results[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{p} must be an object")
            continue
        if not (_is_int(r.get("instance_id")) and r["instance_id"] >= 0):
            errors.append(f"{p}.instance_id must be an int >= 0")
        if not (isinstance(r.get("part"), str) and r["part"]):
            errors.append(f"{p}.part must be a non-empty string")
        if not (isinstance(r.get("face"), str) and r["face"]):
            errors.append(f"{p}.face must be a non-empty string")
        R = r.get("R_world")
        if not (isinstance(R, list) and len(R) == 9 and all(_is_number(v) for v in R)):
            errors.append(f"{p}.R_world must be 9 numbers (row-major 3x3)")
        t = r.get("t_world")
        if not (isinstance(t, list) and len(t) == 3 and all(_is_number(v) for v in t)):
            errors.append(f"{p}.t_world must be 3 numbers")
        c = r.get("confidence")
        if not (_is_number(c) and 0.0 <= c <= 1.0):
            errors.append(f"{p}.confidence must be a number in [0, 1]")
        b = r.get("bbox_2d")
        if not (isinstance(b, list) and len(b) == 4 and all(_is_int(v) and v >= 0 for v in b)):
            errors.append(f"{p}.bbox_2d must be 4 ints >= 0")
        elif not (b[0] <= b[2] and b[1] <= b[3]):
            errors.append(f"{p}.bbox_2d must satisfy x0<=x1 and y0<=y1")
        if not isinstance(r.get("upright"), bool):
            errors.append(f"{p}.upright must be a boolean")
    return errors


def _check_with_jsonschema(doc: dict) -> list[str]:
    """Strict JSON-Schema gate; empty list if jsonschema is unavailable."""
    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return []
    if not os.path.exists(SCHEMA_PATH):
        return []
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in v.iter_errors(doc)]


def validate_or_raise(doc: dict) -> None:
    """Validate ``doc`` against the contract; raise :class:`PoseResultInvalid` on failure.

    Runs the stdlib validator always and the jsonschema validator when present.
    """
    errors = check_pose_result(doc)
    errors += [f"[jsonschema] {m}" for m in _check_with_jsonschema(doc)]
    if errors:
        raise PoseResultInvalid(errors)


def jsonschema_available() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except Exception:
        return False
