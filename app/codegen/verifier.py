from __future__ import annotations

import importlib.util
import re
import statistics
import textwrap
import time
from pathlib import Path
from typing import Any

from app.configs.codegen import (
    BENCHMARK_SAMPLE_COUNT,
    COMPLEXITY_BASE,
    COMPLEXITY_FOR_COST,
    COMPLEXITY_IF_COST,
    COMPLEXITY_IMPORT_COST,
    COMPLEXITY_LINE_DIVISOR,
    COMPLEXITY_MAX,
    COMPLEXITY_WHILE_COST,
    FORBIDDEN_NETWORK_PATTERNS,
    LINE_COUNT_NORMALIZER,
    SPEED_SCORE_CAP,
)
from app.codegen.selection import compute_tie_break_score, evaluate_gate


def _line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def _estimate_complexity(source_code: str) -> float:
    line_cost = _line_count(source_code) / COMPLEXITY_LINE_DIVISOR
    import_cost = source_code.count("\nimport ") * COMPLEXITY_IMPORT_COST + source_code.count("\nfrom ") * COMPLEXITY_IMPORT_COST
    branch_cost = (
        source_code.count(" for ") * COMPLEXITY_FOR_COST
        + source_code.count(" while ") * COMPLEXITY_WHILE_COST
        + source_code.count(" if ") * COMPLEXITY_IF_COST
    )
    return round(min(COMPLEXITY_BASE + line_cost + import_cost + branch_cost, COMPLEXITY_MAX), 2)


def _objective_direction(task: dict[str, Any]) -> str:
    objective_spec = task.get("objective_spec") or {}
    return str(objective_spec.get("direction") or task.get("objective_direction") or "max")


def _objective_score(task: dict[str, Any], objective: float) -> float:
    direction = _objective_direction(task)
    return objective if direction == "max" else -objective


def _clamp01(value: float) -> float:
    return max(0.0, min(value, 1.0))


def indent_function_body(function_body: str) -> str:
    normalized = function_body.strip("\n")
    if not normalized.strip():
        raise ValueError("function_body must not be empty.")
    return textwrap.indent(normalized, "    ")


def build_candidate_source(task: dict[str, Any], imports: list[str], function_body: str) -> str:
    sections: list[str] = []
    normalized_imports = [line.strip() for line in imports if line.strip()]
    if normalized_imports:
        sections.append("\n".join(dict.fromkeys(normalized_imports)))
    sections.append(f"{task['function_signature']}\n{indent_function_body(function_body)}")
    return "\n\n".join(sections).rstrip() + "\n"


def normalize_file_body(file_body: str) -> str:
    normalized = file_body.strip("\n")
    if not normalized.strip():
        raise ValueError("file_body must not be empty.")
    return normalized.rstrip() + "\n"


def materialize_candidate(
    *,
    task: dict[str, Any],
    workspace_root: Path,
    candidate_id: str,
    file_body: str | None = None,
    imports: list[str] | None = None,
    function_body: str | None = None,
) -> tuple[Path, str]:
    candidate_dir = workspace_root / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    filename = str(task.get("editable_filename") or Path(str(task.get("editable_file") or "candidate.py")).name)
    source_path = candidate_dir / filename
    if file_body is None:
        source_code = build_candidate_source(task, imports or [], function_body or "")
    else:
        source_code = normalize_file_body(file_body)
    source_path.write_text(source_code)
    return source_path, source_code


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_callable_from_path(path: Path, symbol: str):
    module_name = f"candidate_{path.parent.name}_{path.stem}".replace("-", "_")
    module = _load_module_from_path(path, module_name)
    function = getattr(module, symbol, None)
    if not callable(function):
        raise ValueError(f"{symbol} was not defined by the candidate")
    return function


def _benchmark_args(kind: str) -> tuple[Any, ...]:
    if kind == "contains_duplicates":
        values = list(range(1200))
        return (values,)
    if kind == "first_repeated_value":
        values = list(range(1400))
        values[1120] = values[240]
        return (values,)
    if kind == "has_overlap":
        left = list(range(400))
        right = list(range(2400, 3600))
        right[960] = 215
        return (left, right)
    if kind == "most_frequent_item":
        values = [index % 17 for index in range(1800)]
        values.extend([7] * 180)
        return (values,)
    if kind == "deduplicate_preserve_order":
        values = [index % 45 for index in range(1600)]
        return (values,)
    if kind == "missing_number":
        missing = 777
        values = [index for index in range(1600) if index != missing]
        return (values,)
    if kind == "count_primes_up_to":
        return (3500,)
    if kind == "count_change_ways":
        return (48, [1, 2, 5, 10, 20, 50])
    if kind == "count_n_queens":
        return (9,)
    raise ValueError(f"Unknown benchmark kind: {kind}")


def finalize_candidate_metrics(
    *,
    task: dict[str, Any],
    source_code: str,
    memory_applied: bool,
    raw_metrics: dict[str, Any],
) -> dict[str, Any]:
    selection_spec = dict(task.get("selection_spec") or {})
    status = str(raw_metrics.get("status") or "pass")
    verifier_status = str(raw_metrics.get("verifier_status") or status)
    objective = float(raw_metrics.get("objective") or 0.0)
    objective_score = float(raw_metrics.get("objective_score") or _objective_score(task, objective))
    objective_signal = float(
        raw_metrics.get(
            "objective_signal",
            raw_metrics.get("speed_score", _clamp01(objective_score if objective_score >= 0 else 0.0)),
        )
    )
    correctness = float(raw_metrics.get("correctness") or 0.0)
    stability = float(raw_metrics.get("stability") or (1.0 if status == "pass" else 0.0))
    complexity = float(raw_metrics.get("complexity") or _estimate_complexity(source_code))
    line_count = int(raw_metrics.get("line_count") or _line_count(source_code))
    computed_values = {
        "status": status,
        "verifier_status": verifier_status,
        "objective": objective,
        "objective_score": objective_score,
        "objective_signal": objective_signal,
        "correctness": correctness,
        "stability": stability,
        "complexity": complexity,
        "line_count": line_count,
        "line_count_normalized": line_count / LINE_COUNT_NORMALIZER,
        "memory_applied": 1.0 if memory_applied else 0.0,
        "benchmark_ms": raw_metrics.get("benchmark_ms"),
    }
    for key, value in raw_metrics.items():
        if key not in computed_values:
            computed_values[key] = value
    gate_passed = evaluate_gate(selection_spec, computed_values)
    primary_score = float(raw_metrics.get("primary_score") or objective_score)
    tie_break_score = float(raw_metrics.get("tie_break_score") or compute_tie_break_score(selection_spec, computed_values))

    metrics = {
        "status": status,
        "verifier_status": verifier_status,
        "gate_passed": gate_passed,
        "correctness": correctness,
        "passed_tests": int(raw_metrics.get("passed_tests") or 0),
        "total_tests": int(raw_metrics.get("total_tests") or 0),
        "benchmark_ms": raw_metrics.get("benchmark_ms"),
        "benchmark_samples_ms": list(raw_metrics.get("benchmark_samples_ms") or []),
        "speedup_vs_baseline": raw_metrics.get("speedup_vs_baseline", 0.0),
        "speed_score": raw_metrics.get("speed_score", 0.0),
        "stability": round(stability, 3),
        "complexity": round(complexity, 3),
        "line_count": line_count,
        "objective": round(objective, 6),
        "objective_score": round(objective_score, 6),
        "objective_signal": round(objective_signal, 6),
        "primary_score": round(primary_score, 6),
        "tie_break_score": round(tie_break_score, 6),
        "error": raw_metrics.get("error"),
        "test_results": list(raw_metrics.get("test_results") or []),
    }
    for key, value in raw_metrics.items():
        if key not in metrics:
            metrics[key] = value
    return metrics


def error_candidate_metrics(
    *,
    task: dict[str, Any],
    source_code: str,
    error: str,
) -> dict[str, Any]:
    return finalize_candidate_metrics(
        task=task,
        source_code=source_code,
        memory_applied=False,
        raw_metrics={
            "status": "error",
            "verifier_status": "error",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": 0,
            "benchmark_ms": None,
            "benchmark_samples_ms": [],
            "speedup_vs_baseline": 0.0,
            "speed_score": 0.0,
            "stability": 0.0,
            "objective": 0.0,
            "objective_score": 0.0,
            "objective_signal": 0.0,
            "error": error,
            "test_results": [],
        },
    )


def _network_access_error(task: dict[str, Any], source_code: str) -> str | None:
    if bool(task.get("allow_browsing", False)):
        return None
    for pattern in FORBIDDEN_NETWORK_PATTERNS:
        if re.search(pattern, source_code):
            return "Browsing and external network access are disabled for this task."
    return None


def evaluate_python_function_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    baseline_metrics: dict[str, Any] | None,
    memory_applied: bool,
) -> dict[str, Any]:
    network_error = _network_access_error(task, source_code)
    if network_error is not None:
        return error_candidate_metrics(task=task, source_code=source_code, error=network_error)
    try:
        function = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    except Exception as exc:  # noqa: BLE001
        return error_candidate_metrics(task=task, source_code=source_code, error=str(exc))

    verifier_config = dict(task.get("data") or {})
    tests = list(verifier_config.get("tests") or [])
    results: list[dict[str, Any]] = []
    try:
        for test in tests:
            actual = function(*test["args"])
            results.append(
                {
                    "name": test["name"],
                    "expected": test["expected"],
                    "actual": actual,
                    "passed": actual == test["expected"],
                }
            )
    except Exception as exc:  # noqa: BLE001
        return error_candidate_metrics(task=task, source_code=source_code, error=str(exc))

    passed_tests = sum(1 for result in results if result["passed"])
    total_tests = len(results)
    correctness = 1.0 if total_tests and passed_tests == total_tests else 0.0
    if correctness == 0.0:
        return finalize_candidate_metrics(
            task=task,
            source_code=source_code,
            memory_applied=memory_applied,
            raw_metrics={
                "status": "fail",
                "verifier_status": "fail",
                "correctness": 0.0,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "benchmark_ms": None,
                "benchmark_samples_ms": [],
                "speedup_vs_baseline": 0.0,
                "speed_score": 0.0,
                "stability": 0.0,
                "objective": 0.0,
                "objective_score": 0.0,
                "objective_signal": 0.0,
                "error": None,
                "test_results": results,
            },
        )

    benchmark = dict(verifier_config.get("benchmark") or {})
    repeats = int(benchmark.get("repeats", 1))
    benchmark_args = _benchmark_args(str(benchmark["kind"]))
    samples: list[float] = []
    try:
        for _ in range(BENCHMARK_SAMPLE_COUNT):
            started = time.perf_counter()
            for _ in range(repeats):
                function(*benchmark_args)
            samples.append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        return error_candidate_metrics(task=task, source_code=source_code, error=str(exc))

    benchmark_ms = statistics.median(samples)
    baseline_ms = None if baseline_metrics is None else baseline_metrics.get("benchmark_ms")
    speedup = float(baseline_ms) / benchmark_ms if baseline_ms and benchmark_ms > 0 else 1.0
    speed_score = min(speedup, SPEED_SCORE_CAP) / SPEED_SCORE_CAP
    stability = min(samples) / max(samples) if max(samples) > 0 else 1.0
    return finalize_candidate_metrics(
        task=task,
        source_code=source_code,
        memory_applied=memory_applied,
        raw_metrics={
            "status": "pass",
            "verifier_status": "pass",
            "correctness": correctness,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "benchmark_ms": round(benchmark_ms, 3),
            "benchmark_samples_ms": [round(sample, 3) for sample in samples],
            "speedup_vs_baseline": round(speedup, 3),
            "speed_score": round(speed_score, 3),
            "stability": round(stability, 3),
            "objective": round(speedup, 3),
            "objective_score": round(_objective_score(task, round(speedup, 3)), 3),
            "objective_signal": round(speed_score, 3),
            "error": None,
            "test_results": results,
        },
    )


def _load_task_verifier(task: dict[str, Any]):
    verifier_path = Path(str(task["verifier_path"]))
    module_name = f"task_verifier_{task['id'].replace('-', '_')}"
    module = _load_module_from_path(verifier_path, module_name)
    evaluator = getattr(module, "evaluate_candidate", None)
    if not callable(evaluator):
        raise ValueError(f"{verifier_path} must export callable evaluate_candidate().")
    return evaluator


def evaluate_materialized_candidate(
    *,
    task: dict[str, Any],
    source_path: Path,
    source_code: str,
    baseline_metrics: dict[str, Any] | None,
    memory_applied: bool,
) -> dict[str, Any]:
    network_error = _network_access_error(task, source_code)
    if network_error is not None:
        return error_candidate_metrics(task=task, source_code=source_code, error=network_error)
    try:
        evaluator = _load_task_verifier(task)
        raw_metrics = evaluator(
            task=task,
            candidate_path=source_path,
            source_code=source_code,
            baseline_metrics=baseline_metrics,
            memory_applied=memory_applied,
        )
        if not isinstance(raw_metrics, dict):
            raise ValueError("Task verifier must return a metrics dict.")
        return finalize_candidate_metrics(
            task=task,
            source_code=source_code,
            memory_applied=memory_applied,
            raw_metrics=raw_metrics,
        )
    except Exception as exc:  # noqa: BLE001
        return error_candidate_metrics(task=task, source_code=source_code, error=str(exc))
