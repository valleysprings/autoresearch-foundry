from __future__ import annotations

import importlib.util
from pathlib import Path


_LEGACY_PATH = Path(__file__).resolve().parents[2] / "planning_verified" / "planbench" / "verifier.py"
_SPEC = importlib.util.spec_from_file_location("_legacy_planbench_verifier", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load legacy PlanBench verifier from {_LEGACY_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

evaluate_candidate = _MODULE.evaluate_candidate
