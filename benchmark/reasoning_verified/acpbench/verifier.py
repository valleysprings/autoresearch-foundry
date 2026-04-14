from __future__ import annotations

import re
import time

from app.bench.benchmark_support import (
    choice_answer_matches,
    choice_response_display,
    normalize_answer_text,
    public_question_payload,
)
from app.codegen.verifier import load_callable_from_path


BOOLEAN_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _match_boolean(item: dict[str, object], raw_actual: object) -> tuple[bool, str]:
    expected = normalize_answer_text(item.get("expected_answer"))
    text = str(raw_actual or "").strip()
    if not text:
        return False, ""
    normalized = normalize_answer_text(text)
    if normalized in {"yes", "no"}:
        return normalized == expected, normalized
    tokens = [match.group(1).lower() for match in BOOLEAN_TOKEN_RE.finditer(text)]
    unique_tokens = list(dict.fromkeys(tokens))
    if len(unique_tokens) == 1:
        actual = unique_tokens[0]
        return actual == expected, actual
    return False, normalized


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    answer_format = str(item.get("metadata", {}).get("answer_format") or "").strip().lower()
    if answer_format == "bool":
        passed, actual = _match_boolean(item, raw_actual)
        actual_display = actual or str(raw_actual or "")
    else:
        passed, actual = choice_answer_matches(
            raw_actual,
            expected=item["expected_answer"],
            choices=item.get("choices") or [],
            answer_alias_list=item.get("metadata", {}).get("answer_aliases", []),
            correct_choice_index=item.get("metadata", {}).get("correct_choice_index"),
        )
        actual_display = choice_response_display(
            actual,
            raw_actual=raw_actual,
            choices=item.get("choices") or [],
            preferred_choice_index=item.get("metadata", {}).get("correct_choice_index") if passed else None,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item["expected_answer"],
        "actual": actual,
        "actual_display": actual_display,
        "actual_raw": str(raw_actual or ""),
        "passed": passed,
    }
    objective = 1.0 if passed else 0.0
    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": objective,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "error": None,
        "test_results": [row],
    }
