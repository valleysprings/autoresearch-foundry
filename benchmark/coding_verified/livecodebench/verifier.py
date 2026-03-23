from __future__ import annotations

import ast
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CASE_TIMEOUT_S = 5.0


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"livecodebench_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem_file(task: dict[str, Any]) -> Path:
    item = dict(task.get("question_item") or {})
    metadata = dict(item.get("metadata") or {})
    relative_path = str(metadata.get("problem_file") or "").strip()
    if not relative_path:
        raise ValueError("LiveCodeBench question metadata is missing problem_file.")
    candidate_path = Path(relative_path)
    path = candidate_path if candidate_path.is_absolute() else ROOT / "data" / relative_path
    if not path.exists():
        raise FileNotFoundError(f"LiveCodeBench cached problem file was not found: {path}")
    return path


def _load_problem(task: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_problem_file(task).read_text())


def _literal_or_text(text: Any) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        return ast.literal_eval(raw)
    except Exception:  # noqa: BLE001
        return raw


def _functional_args(text: Any) -> list[Any]:
    raw = str(text or "")
    if not raw.strip():
        return []
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [_literal_or_text(raw)]
    return [_literal_or_text(line) for line in lines]


def _normalize_output_text(value: Any) -> str:
    lines = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    return "\n".join(line.rstrip() for line in lines if line is not None).strip()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _literal_or_text(value)
        if parsed != value:
            return _normalize_value(parsed)
        return value.strip()
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, float):
        return float(value)
    return value


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:  # noqa: BLE001
            return False
    if type(left) is not type(right):
        if isinstance(left, list) and isinstance(right, tuple):
            return _values_match(left, list(right))
        if isinstance(left, tuple) and isinstance(right, list):
            return _values_match(list(left), right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_values_match(l_item, r_item) for l_item, r_item in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_values_match(left[key], right[key]) for key in left)
    return left == right


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return repr(value)


def _resolve_functional_callable(module: Any, function_name: str):
    solution_class = getattr(module, "Solution", None)
    if solution_class is not None:
        solution = solution_class()
        method = getattr(solution, function_name, None)
        if callable(method):
            return method
    function = getattr(module, function_name, None)
    if callable(function):
        return function
    raise ValueError(f"Candidate did not define Solution.{function_name} or top-level {function_name}().")


def _run_functional_case(callable_obj: Any, case: dict[str, Any]) -> dict[str, Any]:
    args = _functional_args(case.get("input"))
    expected = _normalize_value(case.get("output"))
    try:
        actual_raw = callable_obj(*args)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": case.get("name") or "functional-case",
            "expected": _display_value(expected),
            "actual": "",
            "actual_raw": "",
            "passed": False,
            "error": str(exc),
        }
    actual = _normalize_value(actual_raw)
    return {
        "name": case.get("name") or "functional-case",
        "expected": _display_value(expected),
        "actual": _display_value(actual),
        "actual_raw": _display_value(actual_raw),
        "passed": _values_match(actual, expected),
        "error": None,
    }


def _run_stdin_case(candidate_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    stdin_payload = str(case.get("input") or "")
    if stdin_payload and not stdin_payload.endswith("\n"):
        stdin_payload += "\n"
    expected = _normalize_output_text(case.get("output"))
    try:
        completed = subprocess.run(
            [sys.executable, str(candidate_path)],
            cwd=candidate_path.parent,
            input=stdin_payload,
            text=True,
            capture_output=True,
            timeout=CASE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": case.get("name") or "stdin-case",
            "expected": expected,
            "actual": "",
            "actual_raw": "",
            "passed": False,
            "error": f"timed out after {CASE_TIMEOUT_S:.1f}s",
        }
    actual = _normalize_output_text(completed.stdout)
    error = None
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}"
    passed = error is None and actual == expected
    return {
        "name": case.get("name") or "stdin-case",
        "expected": expected,
        "actual": actual,
        "actual_raw": completed.stdout,
        "passed": passed,
        "error": error,
    }


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    started = time.perf_counter()
    problem = _load_problem(task)
    public_cases = list(problem.get("public_test_cases") or [])
    private_cases = list(problem.get("private_test_cases") or [])
    cases = public_cases + private_cases
    if not cases:
        raise ValueError("LiveCodeBench cached item did not contain any test cases.")

    evaluation_mode = str(problem.get("evaluation_mode") or "stdin")
    rows: list[dict[str, Any]] = []

    if evaluation_mode == "functional":
        function_name = str(problem.get("function_name") or "").strip()
        if not function_name:
            raise ValueError("Functional LiveCodeBench item is missing function_name.")
        module = _load_module(candidate_path)
        callable_obj = _resolve_functional_callable(module, function_name)
        rows = [_run_functional_case(callable_obj, case) for case in cases]
    else:
        rows = [_run_stdin_case(candidate_path, case) for case in cases]

    passed = sum(1 for row in rows if row["passed"])
    total = len(rows)
    pass_rate = passed / total if total else 0.0
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    status = "pass" if passed == total else "fail"
    return {
        "status": status,
        "verifier_status": status,
        "correctness": round(pass_rate, 6),
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": round(pass_rate, 6),
        "objective_score": round(pass_rate, 6),
        "objective_signal": round(pass_rate, 6),
        "stability": 1.0,
        "error": None,
        "platform": problem.get("platform"),
        "evaluation_mode": evaluation_mode,
        "test_results": rows,
    }
