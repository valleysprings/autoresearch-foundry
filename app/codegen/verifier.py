from __future__ import annotations

import statistics
import textwrap
import time
from pathlib import Path
from typing import Any


def _line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def _estimate_complexity(source_code: str, imports: list[str]) -> float:
    line_cost = _line_count(source_code) / 28.0
    import_cost = len(imports) * 0.04
    branch_cost = source_code.count(" for ") * 0.03 + source_code.count(" while ") * 0.04 + source_code.count(" if ") * 0.02
    return round(min(0.12 + line_cost + import_cost + branch_cost, 0.95), 2)


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


def materialize_candidate(
    *,
    task: dict[str, Any],
    workspace_root: Path,
    candidate_id: str,
    imports: list[str],
    function_body: str,
) -> tuple[Path, str]:
    candidate_dir = workspace_root / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    source_path = candidate_dir / "candidate.py"
    source_code = build_candidate_source(task, imports, function_body)
    source_path.write_text(source_code)
    return source_path, source_code


def _load_function_from_path(path: Path, function_name: str):
    namespace: dict[str, Any] = {}
    source = path.read_text()
    exec(compile(source, str(path), "exec"), namespace)
    function = namespace.get(function_name)
    if not callable(function):
        raise ValueError(f"{function_name} was not defined by the candidate")
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
    raise ValueError(f"Unknown benchmark kind: {kind}")


def _run_tests(function, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
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
    return results


def evaluate_materialized_candidate(
    *,
    task: dict[str, Any],
    source_path: Path,
    source_code: str,
    imports: list[str],
    baseline_ms: float | None,
    memory_applied: bool,
) -> dict[str, Any]:
    complexity = _estimate_complexity(source_code, imports)
    try:
        function = _load_function_from_path(source_path, task["function_name"])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "verifier_status": "error",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": len(task["tests"]),
            "benchmark_ms": None,
            "benchmark_samples_ms": [],
            "speedup_vs_baseline": 0.0,
            "speed_score": 0.0,
            "stability": 0.0,
            "complexity": complexity,
            "line_count": _line_count(source_code),
            "objective": 0.0,
            "error": str(exc),
            "test_results": [],
            "J": -1.0,
        }

    test_results = _run_tests(function, task["tests"])
    passed_tests = sum(1 for result in test_results if result["passed"])
    total_tests = len(test_results)
    correctness = 1.0 if passed_tests == total_tests else 0.0
    line_count = _line_count(source_code)

    if correctness == 0.0:
        return {
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
            "complexity": complexity,
            "line_count": line_count,
            "objective": 0.0,
            "error": None,
            "test_results": test_results,
            "J": round(0.45 - 0.25 * complexity, 4),
        }

    samples: list[float] = []
    repeats = int(task["benchmark"]["repeats"])
    benchmark_args = _benchmark_args(task["benchmark"]["kind"])
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(repeats):
            function(*benchmark_args)
        samples.append((time.perf_counter() - started) * 1000.0)

    benchmark_ms = statistics.median(samples)
    speedup = baseline_ms / benchmark_ms if baseline_ms is not None and benchmark_ms > 0 else 1.0
    speed_score = min(speedup, 8.0) / 8.0
    stability = min(samples) / max(samples) if max(samples) > 0 else 1.0
    memory_bonus = 1.0 if memory_applied else 0.0
    score = (
        1.20 * correctness
        + 0.95 * speed_score
        + 0.20 * memory_bonus
        + 0.15 * stability
        - 0.18 * complexity
        - 0.05 * (line_count / 10.0)
    )
    return {
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
        "complexity": complexity,
        "line_count": line_count,
        "objective": round(speedup, 3),
        "error": None,
        "test_results": test_results,
        "J": round(score, 4),
    }
