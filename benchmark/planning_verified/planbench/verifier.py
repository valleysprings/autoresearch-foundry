from __future__ import annotations

import re
import time
from typing import Any, Callable

from app.codegen.benchmark_support import canonical_text, public_question_payload
from app.codegen.verifier import load_callable_from_path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_-]+")
PAREN_STEP_PATTERN = re.compile(r"\(([^()]+)\)")
PLAN_BLOCK_PATTERN = re.compile(r"\[plan\](.*?)(?:\[plan end\]|$)", re.IGNORECASE | re.DOTALL)

DOMAIN_ACTION_ARITIES: dict[str, dict[str, int]] = {
    "obfuscated_deceptive_logistics": {
        "paltry": 3,
        "sip": 3,
        "clip": 3,
        "wretched": 4,
        "memory": 3,
        "tightfisted": 3,
    },
    "logistics": {
        "load": 3,
        "unload": 3,
        "drive": 3,
        "fly": 3,
        "load-airplane": 3,
        "load-truck": 3,
        "unload-airplane": 3,
        "unload-truck": 3,
        "drive-truck": 3,
        "fly-airplane": 3,
    },
    "depots": {
        "drive": 3,
        "lift": 4,
        "load": 4,
        "unload": 4,
        "drop": 4,
    },
    "blocksworld": {
        "pick-up": 1,
        "put-down": 1,
        "stack": 2,
        "unstack": 2,
    },
    "blocksworld_3": {
        "pick-up": 1,
        "put-down": 1,
        "stack": 2,
        "unstack": 2,
    },
    "mystery_blocksworld": {
        "attack": 1,
        "feast": 2,
        "succumb": 1,
        "overcome": 2,
    },
    "mystery_blocksworld_3": {
        "attack": 1,
        "feast": 2,
        "succumb": 1,
        "overcome": 2,
    },
}


def _domain_name(item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    raw_context = item.get("raw_context")
    if isinstance(raw_context, dict):
        domain = raw_context.get("domain")
        if isinstance(domain, str) and domain.strip():
            return domain.strip().lower()
    domain = metadata.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain.strip().lower()
    return ""


def _plan_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _extract_plan_region(text: str) -> str:
    matches = PLAN_BLOCK_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return text


def _normalize_symbol(token: str) -> str:
    normalized = canonical_text(token, lowercase=True).strip(".,;:[]{}")
    if not normalized:
        return ""
    match = re.fullmatch(r"object_(\d+)", normalized)
    if match:
        return f"o{match.group(1)}"
    match = re.fullmatch(r"package_(\d+)", normalized)
    if match:
        return f"p{match.group(1)}"
    match = re.fullmatch(r"airplane_(\d+)", normalized)
    if match:
        return f"a{match.group(1)}"
    match = re.fullmatch(r"truck_(\d+)", normalized)
    if match:
        return f"t{match.group(1)}"
    match = re.fullmatch(r"location_(\d+)_(\d+)", normalized)
    if match:
        return f"l{match.group(1)}-{match.group(2)}"
    match = re.fullmatch(r"city_(\d+)", normalized)
    if match:
        return f"c{match.group(1)}"
    return normalized


def _canonical_step(step: str) -> str:
    tokens = [_normalize_symbol(token) for token in TOKEN_PATTERN.findall(step)]
    normalized = [token for token in tokens if token]
    return " ".join(normalized)


def _parenthesized_steps(text: str) -> list[str]:
    return [normalized for match in PAREN_STEP_PATTERN.findall(text) if (normalized := _canonical_step(match))]


def _expected_steps(item: dict[str, Any]) -> list[str]:
    return _parenthesized_steps(str(item.get("expected_answer") or ""))


def _digit_like_arg(raw: str, normalized: str) -> bool:
    if raw.startswith("object_"):
        return True
    return bool(re.search(r"\d", raw) or re.fullmatch(r"[a-z]+\d+(?:-\d+)?", normalized))


def _obfuscated_arg(raw: str, normalized: str) -> bool:
    return raw.startswith("object_") or bool(re.fullmatch(r"o\d+", normalized))


def _vocab_arg(vocabulary: set[str]) -> Callable[[str, str], bool]:
    return lambda _raw, normalized: normalized in vocabulary


def _scan_steps(
    text: str,
    *,
    action_arities: dict[str, int],
    is_arg_token: Callable[[str, str], bool],
    normalize_action: Callable[[str, list[str]], str | None] | None = None,
) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    steps: list[str] = []
    index = 0
    while index < len(tokens):
        action = tokens[index]
        if action not in action_arities:
            index += 1
            continue
        arity = action_arities[action]
        args: list[str] = []
        cursor = index + 1
        while cursor < len(tokens) and len(args) < arity:
            raw = tokens[cursor]
            normalized = _normalize_symbol(raw)
            if normalized and is_arg_token(raw, normalized):
                args.append(normalized)
            cursor += 1
        if len(args) == arity:
            canonical_action = normalize_action(action, args) if normalize_action is not None else action
            if canonical_action:
                steps.append(f"{canonical_action} {' '.join(args)}".strip())
                index = cursor
                continue
        index += 1
    return steps


def _normalize_logistics_action(action: str, args: list[str]) -> str | None:
    if action in {"load-airplane", "load-truck", "unload-airplane", "unload-truck", "drive-truck", "fly-airplane"}:
        return action
    if action in {"load", "unload"}:
        vehicle = args[1] if len(args) > 1 else ""
        if vehicle.startswith("a"):
            return f"{action}-airplane"
        if vehicle.startswith("t"):
            return f"{action}-truck"
        return None
    if action == "drive":
        return "drive-truck"
    if action == "fly":
        return "fly-airplane"
    return action


def _normalize_plan(value: object, item: dict[str, Any]) -> list[str]:
    text = _extract_plan_region(_plan_text(value))
    parenthesized = _parenthesized_steps(text)
    if parenthesized:
        return parenthesized

    domain = _domain_name(item)
    normalized_text = canonical_text(text, lowercase=True)
    normalized_text = normalized_text.replace("pick up", "pick-up").replace("put down", "put-down")

    if domain == "obfuscated_deceptive_logistics":
        steps = _scan_steps(
            normalized_text,
            action_arities=DOMAIN_ACTION_ARITIES[domain],
            is_arg_token=_obfuscated_arg,
        )
        if steps:
            return steps

    if domain == "logistics":
        steps = _scan_steps(
            normalized_text,
            action_arities=DOMAIN_ACTION_ARITIES[domain],
            is_arg_token=_digit_like_arg,
            normalize_action=_normalize_logistics_action,
        )
        if steps:
            return steps

    if domain == "depots":
        steps = _scan_steps(
            normalized_text,
            action_arities=DOMAIN_ACTION_ARITIES[domain],
            is_arg_token=_digit_like_arg,
        )
        if steps:
            return steps

    if domain in {"blocksworld", "blocksworld_3", "mystery_blocksworld", "mystery_blocksworld_3"}:
        vocabulary = {token for step in _expected_steps(item) for token in step.split()[1:]}
        steps = _scan_steps(
            normalized_text,
            action_arities=DOMAIN_ACTION_ARITIES[domain],
            is_arg_token=_vocab_arg(vocabulary),
        )
        if steps:
            return steps

    raw_steps = text.replace(";", "\n").splitlines()
    normalized: list[str] = []
    for step in raw_steps:
        candidate = _canonical_step(step)
        if candidate:
            normalized.append(candidate)
    return normalized


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("PlanBench dataset task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_steps = _normalize_plan(raw_actual, item)
    expected_steps = _expected_steps(item)
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
