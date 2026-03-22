from __future__ import annotations

import time

from app.codegen.benchmark_support import canonical_text, public_question_payload
from app.codegen.verifier import load_callable_from_path


def _normalize_plan(value: object) -> list[str]:
    if isinstance(value, list):
        raw_steps = [str(item) for item in value]
    else:
        raw_text = str(value or "").replace(";", "\n")
        raw_steps = raw_text.splitlines()
    normalized: list[str] = []
    for step in raw_steps:
        compact = canonical_text(step, lowercase=True).replace("(", " ").replace(")", " ")
        compact = " ".join(compact.split())
        if compact:
            normalized.append(compact)
    return normalized


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("PlanBench dataset task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_steps = _normalize_plan(raw_actual)
    expected_steps = _normalize_plan(item["expected_answer"])
    passed = actual_steps == expected_steps
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item["expected_answer"],
        "actual": "\n".join(actual_steps),
        "actual_raw": raw_actual,
        "passed": passed,
    }
    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "error": None,
        "test_results": [row],
    }
