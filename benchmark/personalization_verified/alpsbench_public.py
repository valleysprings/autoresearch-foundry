from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
for candidate in (ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.bench.benchmark_support import preview_display_text, public_question_payload
from app.bench.personalization_support import serialize_dialogue_history, write_manifest
from app.codegen.verifier import load_callable_from_path


DATASET_REPO = "Cosineyx/Alpsbench"
CACHE_DIR = ROOT / "_downloads" / "alpsbench"
_WS_RE = re.compile(r"\s+")

TASK1_DATASET_ID = "alpsbench_task1_validation"
TASK2_DATASET_ID = "alpsbench_task2_validation"
TASK3_DATASET_ID = "alpsbench_task3_validation"
TASK4_DATASET_ID = "alpsbench_task4_validation"

TASK1_SIZE = 466
TASK2_SIZE = 469
TASK3_SIZE = 476 * 5
TASK4_SIZE = 577

TASK3_DISTRACTOR_PATHS: tuple[str, ...] = ("d100", "d300", "d500", "d700", "d1000")
TASK4_ABILITY_PATHS: tuple[str, ...] = ("ability1", "ability2", "ability3", "ability4", "ability5")

TASK1_PROMPT = (
    "Extract stable long-term memories about the user from the dialogue. "
    'Return ONLY JSON, either as {"memory_items": [...]} or as a raw JSON list of memory items.'
)
TASK2_PROMPT = (
    "Update the user's long-term memory using the existing memories plus the new dialogue. "
    'Return ONLY JSON, either as {"memory_items": [...]} or as a raw JSON list of memory items.'
)
TASK3_PROMPT = (
    "Answer the query using the best supporting candidate memory. "
    'Return ONLY JSON with keys "answer", "reason", and "selected_memory_id".'
)
TASK4_PROMPT = (
    "Write a personalized response grounded in the relevant memory. "
    'Return ONLY JSON with keys "answer" and "used_memory_fact".'
)


def add_items_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser


def requested_count(total: int, items: int | None) -> int:
    return total if items is None else max(1, min(int(items), total))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def dataset_file(filename: str) -> Path:
    if hf_hub_download is not None:
        return Path(hf_hub_download(DATASET_REPO, filename=filename, repo_type="dataset"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / filename.replace("/", "__")
    if destination.exists():
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    url = f"https://huggingface.co/datasets/{quote(DATASET_REPO, safe='/')}/resolve/main/{quote(filename, safe='/')}"
    with urllib.request.urlopen(url, timeout=120) as response:
        temp_path.write_bytes(response.read())
    temp_path.replace(destination)
    return destination


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else _compact_json(value)
    return _WS_RE.sub(" ", text.strip().lower())


def _round(value: float) -> float:
    return round(float(value), 6)


def _coerce_memory_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _canonical_memory_item(item: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        normalize_text(item.get("label")),
        normalize_text(item.get("value")),
        normalize_text(item.get("type")),
        normalize_text(item.get("preference_attitude")),
        normalize_text(item.get("time_scope")),
        normalize_text(item.get("emotion")),
    )


def _multiset_metrics(pred_items: list[dict[str, Any]], ref_items: list[dict[str, Any]]) -> dict[str, Any]:
    pred_counter = Counter(_canonical_memory_item(item) for item in pred_items)
    ref_counter = Counter(_canonical_memory_item(item) for item in ref_items)
    overlap = pred_counter & ref_counter
    true_positive = sum(overlap.values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())

    if pred_total == 0 and ref_total == 0:
        precision = recall = f1 = 1.0
    else:
        precision = true_positive / pred_total if pred_total else 0.0
        recall = true_positive / ref_total if ref_total else 0.0
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "exact_match": 1.0 if pred_counter == ref_counter else 0.0,
        "pred_count": pred_total,
        "reference_count": ref_total,
        "main_score": _round(f1),
    }


def parse_solver_payload(raw_actual: Any) -> Any:
    if isinstance(raw_actual, (dict, list)):
        return raw_actual
    text = str(raw_actual or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None


def _task1_prediction_payload(parsed: Any, item_id: str) -> Mapping[str, Any] | None:
    if isinstance(parsed, list):
        return {"benchmark_id": item_id, "prediction": parsed}
    if isinstance(parsed, Mapping):
        payload = dict(parsed)
        payload.setdefault("benchmark_id", item_id)
        return payload
    return None


def _task2_prediction_payload(parsed: Any, item_id: str) -> Mapping[str, Any] | None:
    return _task1_prediction_payload(parsed, item_id)


def _task3_prediction_payload(parsed: Any, item_id: str) -> Mapping[str, Any] | None:
    if not isinstance(parsed, Mapping):
        return None
    payload = dict(parsed)
    payload.setdefault("benchmark_id", item_id)
    return payload


def _task4_prediction_payload(parsed: Any, item_id: str) -> Mapping[str, Any] | None:
    if not isinstance(parsed, Mapping):
        return None
    payload = dict(parsed)
    payload.setdefault("benchmark_id", item_id)
    return payload


def score_task1(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    payload = prediction.get("prediction") if "prediction" in prediction else prediction
    if isinstance(payload, list):
        pred_items = _coerce_memory_items(payload)
    elif isinstance(payload, Mapping):
        pred_items = _coerce_memory_items(payload.get("memory_items"))
    else:
        pred_items = []
    ref_items = _coerce_memory_items(reference.get("gold", {}).get("memory_items"))
    return _multiset_metrics(pred_items, ref_items)


def score_task2(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    payload = prediction.get("prediction") if "prediction" in prediction else prediction
    if isinstance(payload, list):
        pred_items = _coerce_memory_items(payload)
    elif isinstance(payload, Mapping):
        pred_items = _coerce_memory_items(payload.get("answer") or payload.get("memory_items"))
    else:
        pred_items = []
    ref_items = _coerce_memory_items(reference.get("gold", {}).get("answer"))
    return _multiset_metrics(pred_items, ref_items)


def score_task3(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    ref_gold = reference.get("gold", {})
    ref_memory = ref_gold.get("selected_memory") or {}
    ref_id = normalize_text(ref_gold.get("selected_memory_id"))
    ref_value = normalize_text(ref_memory.get("value"))
    pred_id = normalize_text(prediction.get("selected_memory_id") or prediction.get("memory_key"))
    pred_memory = prediction.get("selected_memory") or {}
    pred_value = normalize_text(pred_memory.get("value") or prediction.get("selected_memory_value"))
    id_match = 1.0 if ref_id and pred_id == ref_id else 0.0
    value_match = 1.0 if ref_value and pred_value == ref_value else 0.0
    accuracy = 1.0 if id_match or value_match else 0.0
    return {
        "accuracy": accuracy,
        "memory_id_match": id_match,
        "memory_value_match": value_match,
        "main_score": accuracy,
    }


def score_task4(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    ref_memory = reference.get("gold", {}).get("selected_memory") or {}
    ref_value = normalize_text(ref_memory.get("value"))
    pred_fact = normalize_text(prediction.get("used_memory_fact"))
    response_present = 1.0 if normalize_text(prediction.get("answer")) else 0.0
    fact_match = 0.0
    if ref_value and pred_fact and (pred_fact == ref_value or pred_fact in ref_value or ref_value in pred_fact):
        fact_match = 1.0
    return {
        "grounding_accuracy": fact_match,
        "used_memory_fact_match": fact_match,
        "response_present": response_present,
        "main_score": fact_match,
    }


def _response_row(name: str, expected: object, actual: object, *, passed: bool, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "name": name,
        "expected": expected,
        "actual": actual,
        "actual_raw": actual if isinstance(actual, str) else _compact_json(actual),
        "passed": passed,
    }
    if extras:
        row.update(extras)
    return row


def _payload_result(
    *,
    item: dict[str, Any],
    raw_actual: Any,
    parsed: Any,
    metrics: dict[str, Any] | None,
    elapsed_ms: float,
    parse_expectation: str,
) -> dict[str, Any]:
    if parsed is None or metrics is None:
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
            "error": None,
            "test_results": [
                _response_row(
                    item.get("name") or item["item_id"],
                    parse_expectation,
                    str(raw_actual or ""),
                    passed=False,
                )
            ],
        }

    objective = float(metrics.get("main_score") or 0.0)
    row = _response_row(
        item.get("name") or item["item_id"],
        item.get("expected_answer"),
        parsed,
        passed=True,
        extras={key: value for key, value in metrics.items() if key != "main_score"},
    )
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": 1,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "error": None,
        "test_results": [row],
    }


def evaluate_structured_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    parse_expectation: str,
    payload_builder,
    score_fn,
) -> dict[str, Any]:
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = __import__("time").perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    parsed = parse_solver_payload(raw_actual)
    payload = payload_builder(parsed, str(item["item_id"]))
    metrics = None if payload is None else score_fn(payload, dict(item.get("raw_reference") or {}))
    elapsed_ms = (__import__("time").perf_counter() - started) * 1000.0
    return _payload_result(
        item=item,
        raw_actual=raw_actual,
        parsed=parsed,
        metrics=metrics,
        elapsed_ms=elapsed_ms,
        parse_expectation=parse_expectation,
    )


def _extraction_context(dialogue: Sequence[dict[str, Any]]) -> str:
    return f"Dialogue:\n{serialize_dialogue_history(dialogue)}"


def _update_context(old_dialogue: Sequence[dict[str, Any]], new_dialogue: Sequence[dict[str, Any]], memory: Sequence[dict[str, Any]]) -> str:
    memory_lines = []
    for item in memory:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        value = str(item.get("value") or "").strip()
        if label and value:
            memory_lines.append(f"- {label}: {value}")
    return "\n\n".join(
        block
        for block in (
            "Existing memories:\n" + ("\n".join(memory_lines) if memory_lines else "(none)"),
            "Old dialogue:\n" + serialize_dialogue_history(old_dialogue),
            "New dialogue:\n" + serialize_dialogue_history(new_dialogue),
        )
        if block.strip()
    )


def _retrieval_context(query: str, candidate_memories: Sequence[dict[str, Any]]) -> str:
    lines = [f"Query:\n{query}", "", "Candidate memories:"]
    for memory in candidate_memories:
        if not isinstance(memory, dict):
            continue
        lines.append(
            f"- {memory.get('memory_id')}: {str(memory.get('label') or '').strip()} | {str(memory.get('value') or '').strip()}"
        )
    return "\n".join(lines).strip()


def _utilization_context(*, ability: str, query: str, selected_memory: dict[str, Any] | None, history: Sequence[dict[str, Any]]) -> str:
    if ability == "ability1":
        memory_text = str((selected_memory or {}).get("value") or "").strip()
        return "\n\n".join(block for block in (f"Selected memory:\n{memory_text}", f"Current query:\n{query}") if block)
    return "\n\n".join(
        block
        for block in (
            "Dialogue history:\n" + serialize_dialogue_history(history),
            f"Current query:\n{query}",
        )
        if block.strip()
    )


def _base_item(
    *,
    benchmark: str,
    item_id: str,
    name: str,
    prompt: str,
    context: str,
    raw_context: Any,
    expected_answer: Any,
    raw_reference: dict[str, Any],
    metadata_extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "benchmark": benchmark,
        "benchmark_category": "user_persona_personalization",
        "interaction_mode": "single_turn",
        **metadata_extra,
    }
    return {
        "item_id": item_id,
        "name": name,
        "prompt": prompt,
        "context": context,
        "raw_context": raw_context,
        "expected_answer": expected_answer,
        "raw_reference": raw_reference,
        "metadata": metadata,
    }


def build_task1_items(*, benchmark: str, limit: int | None = None) -> list[dict[str, Any]]:
    input_rows = _load_jsonl(dataset_file("dataset/validation/task1/model_input.jsonl"))
    reference_rows = _load_jsonl(dataset_file("dataset/validation/task1/reference_output.jsonl"))
    if len(input_rows) != len(reference_rows):
        raise ValueError("AlpsBench task1 input/output row counts differ.")
    requested = requested_count(len(input_rows), limit)
    items: list[dict[str, Any]] = []
    for input_row, reference_row in zip(input_rows[:requested], reference_rows[:requested], strict=True):
        item_id = str(input_row.get("benchmark_id") or input_row.get("canonical_id") or "").strip()
        input_payload = dict(input_row.get("input") or {})
        dialogue = [turn for turn in list(input_payload.get("dialogue") or []) if isinstance(turn, dict)]
        gold_memory = list(dict(reference_row.get("gold") or {}).get("memory_items") or [])
        if not item_id or not dialogue:
            raise ValueError(f"Incomplete AlpsBench task1 row for {item_id!r}.")
        items.append(
            _base_item(
                benchmark=benchmark,
                item_id=item_id,
                name=f"AlpsBench Task 1 {item_id}",
                prompt=TASK1_PROMPT,
                context=_extraction_context(dialogue),
                raw_context={"dialogue": dialogue},
                expected_answer={"memory_items": gold_memory},
                raw_reference=reference_row,
                metadata_extra={
                    "task": "task1",
                    "task_shape": "agentic_open_ended",
                    "scoring_mode": "rubric_score",
                    "stratum": str(input_row.get("stratum") or "").strip(),
                },
            )
        )
    if limit is None and len(items) != TASK1_SIZE:
        raise ValueError(f"Expected {TASK1_SIZE} AlpsBench task1 rows, found {len(items)}.")
    return items


def build_task2_items(*, benchmark: str, limit: int | None = None) -> list[dict[str, Any]]:
    input_rows = _load_jsonl(dataset_file("dataset/validation/task2/model_input.jsonl"))
    reference_rows = _load_jsonl(dataset_file("dataset/validation/task2/reference_output.jsonl"))
    if len(input_rows) != len(reference_rows):
        raise ValueError("AlpsBench task2 input/output row counts differ.")
    requested = requested_count(len(input_rows), limit)
    items: list[dict[str, Any]] = []
    for input_row, reference_row in zip(input_rows[:requested], reference_rows[:requested], strict=True):
        item_id = str(input_row.get("benchmark_id") or input_row.get("canonical_id") or "").strip()
        input_payload = dict(input_row.get("input") or {})
        old_dialogue = [turn for turn in list(input_payload.get("old_dialogue") or []) if isinstance(turn, dict)]
        new_dialogue = [turn for turn in list(input_payload.get("new_dialogue") or []) if isinstance(turn, dict)]
        memory = [entry for entry in list(input_payload.get("memory") or []) if isinstance(entry, dict)]
        gold_memory = list(dict(reference_row.get("gold") or {}).get("answer") or [])
        prompt = str(input_payload.get("query") or "").strip() or TASK2_PROMPT
        if not item_id or not memory:
            raise ValueError(f"Incomplete AlpsBench task2 row for {item_id!r}.")
        items.append(
            _base_item(
                benchmark=benchmark,
                item_id=item_id,
                name=f"AlpsBench Task 2 {item_id}",
                prompt=prompt,
                context=_update_context(old_dialogue, new_dialogue, memory),
                raw_context={
                    "existing_memories": memory,
                    "old_dialogue": old_dialogue,
                    "new_dialogue": new_dialogue,
                },
                expected_answer={"memory_items": gold_memory},
                raw_reference=reference_row,
                metadata_extra={
                    "task": "task2",
                    "task_shape": "agentic_open_ended",
                    "scoring_mode": "rubric_score",
                    "stratum": str(input_row.get("stratum") or "").strip(),
                },
            )
        )
    if limit is None and len(items) != TASK2_SIZE:
        raise ValueError(f"Expected {TASK2_SIZE} AlpsBench task2 rows, found {len(items)}.")
    return items


def build_task3_items(*, benchmark: str, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    remaining = limit
    for distractors in TASK3_DISTRACTOR_PATHS:
        input_rows = _load_jsonl(dataset_file(f"dataset/validation/task3/{distractors}/model_input.jsonl"))
        reference_rows = _load_jsonl(dataset_file(f"dataset/validation/task3/{distractors}/reference_output.jsonl"))
        if len(input_rows) != len(reference_rows):
            raise ValueError(f"AlpsBench task3 {distractors} input/output row counts differ.")
        for input_row, reference_row in zip(input_rows, reference_rows, strict=True):
            item_id = str(input_row.get("benchmark_id") or input_row.get("canonical_id") or "").strip()
            input_payload = dict(input_row.get("input") or {})
            candidate_memories = [entry for entry in list(input_payload.get("candidate_memories") or []) if isinstance(entry, dict)]
            query = str(input_payload.get("query") or "").strip()
            gold = dict(reference_row.get("gold") or {})
            gold_memory = dict(gold.get("selected_memory") or {})
            if not item_id or not query or not candidate_memories or not gold_memory:
                raise ValueError(f"Incomplete AlpsBench task3 row for {item_id!r}.")
            expected_answer = {
                "answer": str(gold_memory.get("value") or "").strip(),
                "reason": "Uses the selected memory.",
                "selected_memory_id": str(gold.get("selected_memory_id") or gold_memory.get("memory_id") or "").strip(),
            }
            items.append(
                _base_item(
                    benchmark=benchmark,
                    item_id=item_id,
                    name=f"AlpsBench Task 3 {distractors} {item_id}",
                    prompt=TASK3_PROMPT,
                    context=_retrieval_context(query, candidate_memories),
                    raw_context={"query": query, "candidate_memories": candidate_memories},
                    expected_answer=expected_answer,
                    raw_reference=reference_row,
                    metadata_extra={
                        "task": "task3",
                        "task_shape": "agentic_open_ended",
                        "scoring_mode": "rubric_score",
                        "distractors": distractors,
                        "stratum": str(input_row.get("stratum") or "").strip(),
                        "runtime_split_tags": [f"distractors:{distractors}"],
                    },
                )
            )
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return items
    if limit is None and len(items) != TASK3_SIZE:
        raise ValueError(f"Expected {TASK3_SIZE} AlpsBench task3 rows, found {len(items)}.")
    return items


def build_task4_items(*, benchmark: str, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    remaining = limit
    for ability in TASK4_ABILITY_PATHS:
        input_rows = _load_jsonl(dataset_file(f"dataset/validation/task4_{ability}/model_input.jsonl"))
        reference_rows = _load_jsonl(dataset_file(f"dataset/validation/task4_{ability}/reference_output.jsonl"))
        if len(input_rows) != len(reference_rows):
            raise ValueError(f"AlpsBench task4 {ability} input/output row counts differ.")
        for input_row, reference_row in zip(input_rows, reference_rows, strict=True):
            item_id = str(input_row.get("benchmark_id") or input_row.get("canonical_id") or "").strip()
            input_payload = dict(input_row.get("input") or {})
            audit_context = dict(input_payload.get("audit_context") or {})
            history = [turn for turn in list(audit_context.get("conversation") or []) if isinstance(turn, dict)]
            query = str(input_payload.get("query") or "").strip()
            model_input = dict(input_payload.get("model_input") or {})
            selected_memory = dict(model_input.get("selected_memory") or {})
            gold = dict(reference_row.get("gold") or {})
            gold_memory = dict(gold.get("selected_memory") or {})
            if not item_id or not query or not gold_memory:
                raise ValueError(f"Incomplete AlpsBench task4 row for {item_id!r}.")
            if ability == "ability1":
                raw_context = {"selected_memory": selected_memory, "latest_query": query}
            else:
                raw_context = {"dialogue_history": history, "latest_query": query}
            items.append(
                _base_item(
                    benchmark=benchmark,
                    item_id=item_id,
                    name=f"AlpsBench Task 4 {ability} {item_id}",
                    prompt=TASK4_PROMPT,
                    context=_utilization_context(
                        ability=ability,
                        query=query,
                        selected_memory=selected_memory or gold_memory,
                        history=history,
                    ),
                    raw_context=raw_context,
                    expected_answer={
                        "answer": "Grounded response.",
                        "used_memory_fact": str(gold_memory.get("value") or "").strip(),
                    },
                    raw_reference=reference_row,
                    metadata_extra={
                        "task": "task4",
                        "task_shape": "dialogue_judgement",
                        "scoring_mode": "rubric_score",
                        "ability": ability,
                        "stratum": str(input_row.get("stratum") or "").strip(),
                        "runtime_split_tags": [f"ability:{ability}"],
                    },
                )
            )
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return items
    if limit is None and len(items) != TASK4_SIZE:
        raise ValueError(f"Expected {TASK4_SIZE} AlpsBench task4 rows, found {len(items)}.")
    return items


def write_task_manifest(
    *,
    manifest_path: Path,
    dataset_id: str,
    split: str,
    items: Sequence[dict[str, Any]],
    dataset_size: int,
) -> dict[str, Any]:
    return write_manifest(
        manifest_path,
        dataset_id=dataset_id,
        split=split,
        items=items,
        dataset_size=dataset_size,
    )


def default_task1_solver(_: dict[str, Any]) -> dict[str, Any]:
    return {"memory_items": []}


def default_task2_solver(_: dict[str, Any]) -> dict[str, Any]:
    return {"memory_items": []}


def default_task3_solver(question: dict[str, Any]) -> dict[str, Any]:
    context = dict(question.get("raw_context") or question.get("context") or {})
    candidate_memories = [entry for entry in list(context.get("candidate_memories") or []) if isinstance(entry, dict)]
    first = candidate_memories[0] if candidate_memories else {}
    return {
        "answer": str(first.get("value") or ""),
        "reason": "Selected the first candidate memory.",
        "selected_memory_id": str(first.get("memory_id") or ""),
    }


def default_task4_solver(question: dict[str, Any]) -> dict[str, Any]:
    context = dict(question.get("raw_context") or question.get("context") or {})
    selected_memory = dict(context.get("selected_memory") or {})
    used_memory_fact = str(selected_memory.get("value") or "").strip()
    return {
        "answer": "Stub personalized response.",
        "used_memory_fact": used_memory_fact,
    }


def preview_json(value: Any) -> str:
    preview = preview_display_text(_compact_json(value))
    return preview or _compact_json(value)
