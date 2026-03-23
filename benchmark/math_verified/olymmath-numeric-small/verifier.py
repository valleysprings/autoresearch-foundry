from __future__ import annotations

import time

from app.codegen.benchmark_support import canonical_numeric_text
from app.codegen.verifier import load_callable_from_path


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    rows = []
    for item in task["data"]["items"]:
        actual = canonical_numeric_text(solver(item["problem"]))
        expected = canonical_numeric_text(item["answer"])
        rows.append(
            {
                "name": item["name"],
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    exact_rate = passed / total if total else 0.0
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": exact_rate,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": exact_rate,
        "objective_score": exact_rate,
        "objective_signal": exact_rate,
        "error": None,
        "test_results": rows,
    }
