from __future__ import annotations

import time

from app.bench.benchmark_support import public_question_payload
from app.bench.planbench_support import (
    PlanExtractionError,
    display_plan,
    extract_plan,
    plan_step_count,
    validate_plan,
)
from app.codegen.verifier import load_callable_from_path


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("PlanBench dataset task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    parse_error: str | None = None
    try:
        extracted_plan = extract_plan(raw_actual, item)
    except PlanExtractionError as exc:
        extracted_plan = ""
        parse_error = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not extracted_plan.strip():
        row = {
            "name": item.get("name") or item["item_id"],
            "expected": item.get("expected_answer"),
            "actual": "",
            "actual_raw": raw_actual,
            "passed": False,
            "reason": parse_error or "plan extraction failed",
        }
        return {
            "status": "fail",
            "verifier_status": "fail",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": 1,
            "benchmark_ms": round(elapsed_ms, 3),
            "benchmark_samples_ms": [round(elapsed_ms, 3)],
            "objective": 0.0,
            "objective_score": 0.0,
            "objective_signal": 0.0,
            "plan_steps": 0,
            "avg_plan_steps": 0.0,
            "error": None,
            "test_results": [row],
        }

    passed, validator_output = validate_plan(item, extracted_plan)
    step_count = plan_step_count(extracted_plan)
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item.get("expected_answer"),
        "actual": display_plan(extracted_plan),
        "actual_raw": raw_actual,
        "passed": passed,
        "validator_output": validator_output,
        "actual_plan_pddl": extracted_plan,
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
        "plan_steps": step_count,
        "avg_plan_steps": float(step_count),
        "error": None,
        "test_results": [row],
    }
