from __future__ import annotations

import statistics
import time
from typing import Any


def _compile_function(code: str, function_name: str):
    namespace: dict[str, Any] = {}
    exec(compile(code, f"<{function_name}>", "exec"), namespace)
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
        args = test["args"]
        actual = function(*args)
        results.append(
            {
                "name": test["name"],
                "expected": test["expected"],
                "actual": actual,
                "passed": actual == test["expected"],
            }
        )
    return results


def evaluate_program(
    *,
    code: str,
    function_name: str,
    tests: list[dict[str, Any]],
    benchmark: dict[str, Any],
    baseline_ms: float | None,
    complexity: float,
    memory_applied: bool,
) -> dict[str, Any]:
    try:
        function = _compile_function(code, function_name)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": len(tests),
            "benchmark_ms": None,
            "benchmark_samples_ms": [],
            "speedup_vs_baseline": 0.0,
            "speed_score": 0.0,
            "stability": 0.0,
            "complexity": round(complexity, 2),
            "line_count": _line_count(code),
            "error": str(exc),
            "J": -1.0,
        }

    test_results = _run_tests(function, tests)
    passed_tests = sum(1 for item in test_results if item["passed"])
    total_tests = len(test_results)
    correctness = 1.0 if passed_tests == total_tests else 0.0

    if not correctness:
        return {
            "status": "fail",
            "correctness": correctness,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "benchmark_ms": None,
            "benchmark_samples_ms": [],
            "speedup_vs_baseline": 0.0,
            "speed_score": 0.0,
            "stability": 0.0,
            "complexity": round(complexity, 2),
            "line_count": _line_count(code),
            "test_results": test_results,
            "error": None,
            "J": round(0.45 - 0.25 * complexity, 4),
        }

    samples = []
    benchmark_args = _benchmark_args(benchmark["kind"])
    repeats = int(benchmark.get("repeats", 10))
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            function(*benchmark_args)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed_ms)

    benchmark_ms = statistics.median(samples)
    speedup = (
        baseline_ms / benchmark_ms
        if baseline_ms is not None and benchmark_ms > 0
        else 1.0
    )
    stability = min(samples) / max(samples) if max(samples) > 0 else 1.0
    line_count = _line_count(code)
    speed_score = min(speedup, 8.0) / 8.0
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
        "correctness": correctness,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "benchmark_ms": round(benchmark_ms, 3),
        "benchmark_samples_ms": [round(sample, 3) for sample in samples],
        "speedup_vs_baseline": round(speedup, 3),
        "speed_score": round(speed_score, 3),
        "stability": round(stability, 3),
        "complexity": round(complexity, 2),
        "line_count": line_count,
        "test_results": test_results,
        "error": None,
        "J": round(score, 4),
    }


def _line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())
