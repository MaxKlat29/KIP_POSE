"""eval — Batch-Eval-Runner fuer die Multi-Pipeline-Plattform (S-012 / T-138).

Faehrt die 7 ComposedPipeline-Kombis (CONTRACT.md §5) ueber N SDG-Seeds MIT GT,
schickt jede Szene pro Config durch das Mesh-Gateway-/predict, rechnet die Cam-Frame-
Posen in den pose_result-Welt-Frame zurueck (round-trip-getestet, S-003), baut die
BOP-results-CSV, scort sie via eval_bop.py --icbin (sym-aware AR IC-BIN) und aggregiert
mean/median/std/coverage/crash ueber die Seeds — Max' "was-klappt-am-besten"-Tabelle.

Das Kernmodul `batch_eval` ist FASTAPI-FREI + mock-injizierbar (predict_fn/eval_fn),
die FastAPI-Endpoints (/api/eval/*) liegen duenn in kip_server.py.
"""
from __future__ import annotations

from . import batch_eval

__all__ = ["batch_eval"]
