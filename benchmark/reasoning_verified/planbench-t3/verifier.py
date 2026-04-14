from __future__ import annotations

import time

from app.bench.benchmark_support import choice_answer_matches, choice_response_display, public_question_payload
from app.codegen.verifier import load_callable_from_path


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("PlanBench Task 3 dataset task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    passed, actual = choice_answer_matches(
        raw_actual,
        expected=item["expected_answer"],
        choices=item.get("choices") or [],
        answer_alias_list=item.get("metadata", {}).get("answer_aliases", []),
        correct_choice_index=item.get("metadata", {}).get("correct_choice_index"),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item["expected_answer"],
        "actual": actual,
        "actual_display": choice_response_display(
            actual,
            raw_actual=raw_actual,
            choices=item.get("choices") or [],
            preferred_choice_index=item.get("metadata", {}).get("correct_choice_index") if passed else None,
        ),
        "actual_raw": str(raw_actual or ""),
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
