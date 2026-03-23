from __future__ import annotations

import contextlib
import io
import time
from typing import Any

from math_verify import parse, verify

from app.codegen.benchmark_support import canonical_numeric_text, choice_answer_matches, public_question_payload
from app.codegen.verifier import load_callable_from_path

VALID_MATH_ANSWER_FORMATS = {"symbolic", "numeric", "choice"}


def _latex_wrap(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("$") and text.endswith("$"):
        return text
    return f"${text}$"


def _normalize_symbolic_answer(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    replacements = {
        "pi": "\\pi",
        "acos": "\\arccos",
        "asin": "\\arcsin",
        "atan": "\\arctan",
    }
    normalized = text.replace("**", "^")
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return _latex_wrap(normalized)


def _parse_quiet(value: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return parse(value, parsing_timeout=None)


def _verify_quiet(gold, answer) -> bool:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return verify(gold, answer, strict=True, timeout_seconds=None)


def math_answer_format(item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    answer_format = str(metadata.get("answer_format") or "").strip().lower()
    if answer_format not in VALID_MATH_ANSWER_FORMATS:
        raise ValueError(f"Math question {item.get('item_id') or item.get('name') or '<unknown>'} has invalid answer_format={answer_format!r}.")
    return answer_format


def grade_math_answer(item: dict[str, Any], raw_actual: object) -> tuple[bool, str]:
    answer_format = math_answer_format(item)
    metadata = dict(item.get("metadata") or {})

    if answer_format == "choice":
        passed, actual = choice_answer_matches(
            raw_actual,
            expected=item["expected_answer"],
            choices=item.get("choices") or [],
            answer_alias_list=metadata.get("answer_aliases", []),
            correct_choice_index=metadata.get("correct_choice_index"),
        )
        return passed, actual

    if answer_format == "numeric":
        actual = canonical_numeric_text(raw_actual) or ""
        expected = canonical_numeric_text(item["expected_answer"]) or ""
        return bool(actual) and actual == expected, actual

    expected = _latex_wrap(item["expected_answer"])
    actual = _normalize_symbolic_answer(raw_actual)
    gold = _parse_quiet(expected)
    answer = _parse_quiet(actual)
    passed = bool(gold) and bool(answer) and _verify_quiet(gold, answer)
    return passed, actual


def evaluate_math_dataset_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    passed, actual = grade_math_answer(item, raw_actual)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": str(item["expected_answer"]),
        "actual": actual,
        "actual_raw": str(raw_actual or ""),
        "answer_format": math_answer_format(item),
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
