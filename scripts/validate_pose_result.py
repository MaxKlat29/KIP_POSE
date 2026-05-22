#!/usr/bin/env python3
"""Validate a pose_result.json against the contract schema; print the part count.

Used by scripts/run_e2e.sh as an independent re-validation of the file the
pipeline wrote (the runner already gates before writing — this is belt +
suspenders and keeps the smoke gate honest if the file is edited by hand).

    python3 scripts/validate_pose_result.py data/output/pose_result.json \
        docs/pose_result.schema.json

Exit codes: 0 ok (prints part count), 2 bad args, 3 stdlib gate fail,
4 jsonschema gate fail. Green on bare python3 (jsonschema optional).
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: validate_pose_result.py <pose_result.json> [schema.json]",
              file=sys.stderr)
        return 2
    out_path = argv[1]
    schema_path = argv[2] if len(argv) > 2 else "docs/pose_result.schema.json"

    with open(out_path) as f:
        doc = json.load(f)

    # Project's own gate first — keeps this consistent with the runner.
    try:
        from pipeline.schema import validate_or_raise
        validate_or_raise(doc)
    except Exception as e:  # noqa: BLE001
        print(f"VALIDATION_ERROR (stdlib gate): {e}", file=sys.stderr)
        return 3

    # jsonschema cross-check when available; never weakens the stdlib gate.
    try:
        import jsonschema
        with open(schema_path) as f:
            schema = json.load(f)
        jsonschema.validate(doc, schema)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"JSONSCHEMA_ERROR: {e}", file=sys.stderr)
        return 4

    print(len(doc.get("results", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
