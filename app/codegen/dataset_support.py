from __future__ import annotations

from collections import Counter
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.codegen.benchmark_support import canonical_text
from app.configs.codegen import (
    DATASET_NETWORK_ACCESS_INSTRUCTION,
    DATASET_SINGLE_QUESTION_INSTRUCTION,
    QUESTION_PREVIEW_LIMIT,
)
from app.codegen.task_contracts import infer_runtime_backend

VALID_MATH_ANSWER_FORMATS = {"symbolic", "numeric", "choice"}


def is_dataset_task(task: dict[str, Any]) -> bool:
    return bool(task.get("local_dataset_only")) and infer_runtime_backend(task) == "dataset"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "item"


def _preview(text: str, *, limit: int = QUESTION_PREVIEW_LIMIT) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _prepare_datasets_hint(task: dict[str, Any]) -> str:
    task_id = str(task.get("id") or "").strip()
    if task_id:
        return f"Run `python benchmark/prepare_datasets.py --task-id {task_id}` first."
    return "Run `python benchmark/prepare_datasets.py` first."


def _task_char_limit(task: dict[str, Any], key: str) -> int | None:
    value = task.get(key)
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _stringify_context(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _truncate_text(text: str, *, limit: int, note: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"... {note}"


def _dedupe_item_id(base_item_id: str, raw_item: dict[str, Any], index: int) -> str:
    metadata = dict(raw_item.get("metadata") or {})
    name = str(raw_item.get("name") or "").strip()
    parts = [base_item_id]
    if name and name != base_item_id:
        parts.append(name)
    for key in ("domain", "prompt_type", "config", "source_split", "source_index"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            parts.append(str(value))
    parts.append(str(index))
    return _slugify("-".join(parts))


def _hydrate_manifest_item(raw_item: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    hydrated = dict(raw_item)
    item_file = raw_item.get("item_file")
    if not isinstance(item_file, str) or not item_file.strip():
        return hydrated

    item_path = manifest_path.parent / item_file
    if not item_path.exists():
        raise FileNotFoundError(f"Question item file not found: {item_path}")

    payload = json.loads(item_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Question item file must contain an object: {item_path}")

    raw_metadata = dict(hydrated.get("metadata") or {})
    payload_metadata = dict(payload.get("metadata") or {})
    if raw_metadata or payload_metadata:
        hydrated["metadata"] = {**raw_metadata, **payload_metadata}

    for key, value in payload.items():
        if key == "metadata":
            continue
        hydrated[key] = value
    return hydrated


def load_question_manifest(task: dict[str, Any], min_items: int | None = None) -> list[dict[str, Any]]:
    manifest_path_raw = task.get("item_manifest_path")
    if not isinstance(manifest_path_raw, str) or not manifest_path_raw.strip():
        raise FileNotFoundError(f"Task {task.get('id') or '<unknown>'} is missing item_manifest_path.")

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.exists() and bool(task.get("lazy_item_manifest")):
        _run_task_prepare(task, min_items=min_items)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Question manifest not found: {manifest_path}. {_prepare_datasets_hint(task)}")

    raw_items = _load_manifest_items(manifest_path)

    if bool(task.get("lazy_item_manifest")):
        default_requested_items = task.get("dataset_size") if min_items is None else min_items
        requested_items = max(0, int(default_requested_items or 0))
        if requested_items > 0 and len(raw_items) < requested_items:
            _run_task_prepare(task, min_items=requested_items)
            raw_items = _load_manifest_items(manifest_path)
        if requested_items > 0 and len(raw_items) < requested_items:
            raise ValueError(
                f"Task {task.get('id') or '<unknown>'} prepared only {len(raw_items)} items "
                f"but {requested_items} were requested."
            )

    hydrated_items = [
        _hydrate_manifest_item(raw_item, manifest_path=manifest_path)
        for raw_item in raw_items
        if isinstance(raw_item, dict)
    ]
    if len(hydrated_items) != len(raw_items):
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                raise ValueError(f"Question manifest item {index} must be an object: {manifest_path}")

    base_item_ids = [
        _slugify(str(raw_item.get("item_id") or raw_item.get("name") or f"item-{index}"))
        for index, raw_item in enumerate(hydrated_items, start=1)
    ]
    duplicate_ids = {item_id for item_id, count in Counter(base_item_ids).items() if count > 1}
    used_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw_item in enumerate(hydrated_items, start=1):
        prompt = raw_item.get("prompt")
        expected = raw_item.get("expected_answer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Question manifest item {index} is missing prompt: {manifest_path}")
        if expected is None:
            raise ValueError(f"Question manifest item {index} is missing expected_answer: {manifest_path}")
        item_id = str(raw_item.get("item_id") or raw_item.get("name") or f"item-{index}")
        normalized_item_id = _slugify(item_id)
        if normalized_item_id in duplicate_ids:
            normalized_item_id = _dedupe_item_id(item_id, raw_item, index)
        while normalized_item_id in used_ids:
            normalized_item_id = f"{normalized_item_id}-{index}"
        used_ids.add(normalized_item_id)
        raw_context = raw_item.get("context")
        raw_choices = list(raw_item.get("choices") or [])
        metadata = dict(raw_item.get("metadata") or {})
        if str(task.get("track") or "") == "math_verified":
            answer_format = str(metadata.get("answer_format") or "").strip().lower()
            if answer_format not in VALID_MATH_ANSWER_FORMATS:
                raise ValueError(
                    f"Math question manifest item {index} must declare metadata.answer_format in "
                    f"{sorted(VALID_MATH_ANSWER_FORMATS)}: {manifest_path}"
                )
            if answer_format == "choice" and not isinstance(metadata.get("correct_choice_index"), int):
                raise ValueError(
                    f"Choice-form math question manifest item {index} must declare metadata.correct_choice_index: {manifest_path}"
                )
        normalized.append(
            {
                "id": normalized_item_id,
                "item_id": normalized_item_id,
                "question_id": normalized_item_id,
                "raw_item_id": item_id,
                "name": str(raw_item.get("name") or item_id),
                "prompt": canonical_text(prompt),
                "raw_prompt": str(prompt).strip(),
                "context": canonical_text(raw_context) if raw_context is not None else None,
                "raw_context": raw_context,
                "choices": [canonical_text(choice) for choice in raw_choices],
                "raw_choices": raw_choices,
                "expected_answer": canonical_text(expected),
                "raw_expected_answer": expected,
                "metadata": metadata,
            }
        )
    return normalized


def _load_manifest_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    raw_items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(raw_items, list):
        raise ValueError(f"Question manifest must contain a list of items: {path}")
    return raw_items


def _run_task_prepare(task: dict[str, Any], *, min_items: int | None) -> None:
    task_dir = Path(str(task.get("task_dir") or "")).resolve()
    prepare_path = task_dir / "prepare.py"
    if not prepare_path.exists():
        raise FileNotFoundError(
            f"Task {task.get('id') or '<unknown>'} is missing prepare.py for lazy manifest loading. "
            f"{_prepare_datasets_hint(task)}"
        )

    args = [sys.executable, str(prepare_path)]
    if isinstance(min_items, int) and min_items > 0:
        args.extend(["--items", str(min_items)])
    completed = subprocess.run(
        args,
        cwd=task_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"returncode={completed.returncode}"
        raise RuntimeError(
            f"Task {task.get('id') or '<unknown>'} prepare.py failed: {details}. {_prepare_datasets_hint(task)}"
        )


def micro_task_id(dataset_task_id: str, item_id: str) -> str:
    return f"{dataset_task_id}-{_slugify(item_id)}"


def question_prompt_context(task: dict[str, Any], item: dict[str, Any]) -> str:
    sections = [str(task.get("prompt_context") or "").strip()]
    sections.append(f"Dataset question id: {item['item_id']}")
    raw_prompt = str(item.get("raw_prompt") or "").strip()
    prompt = str(item.get("prompt") or "").strip()
    if raw_prompt:
        sections.append("Question raw prompt:")
        sections.append(raw_prompt)
    elif prompt:
        sections.append(f"Question prompt: {prompt}")
    context = item.get("raw_context") if item.get("raw_context") is not None else item.get("context")
    if context:
        context_text = _stringify_context(context)
        context_limit = _task_char_limit(task, "prompt_context_max_chars")
        if context_limit is not None:
            context_text = _truncate_text(
                context_text,
                limit=context_limit,
                note=(
                    f"[truncated from {len(context_text)} chars; "
                    "the full context is still available to solve(question) at runtime]"
                ),
            )
        sections.append(f"Question context: {context_text}")
    choices = item.get("choices") or []
    if choices:
        sections.append(f"Choices: {json.dumps(choices, ensure_ascii=True)}")
    if not bool(task.get("allow_browsing", False)):
        sections.append(DATASET_NETWORK_ACCESS_INSTRUCTION)
    sections.append(DATASET_SINGLE_QUESTION_INSTRUCTION)
    return "\n".join(section for section in sections if section)


def build_micro_task(task: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    micro_task = dict(task)
    micro_id = micro_task_id(str(task["id"]), str(item["item_id"]))
    preview = _preview(item["prompt"])
    micro_task.update(
        {
            "id": micro_id,
            "dataset_task_id": task["id"],
            "dataset_task_title": task["title"],
            "title": f"{task['title']} / {item['name']}",
            "description": preview,
            "question_item": item,
            "prompt_context": question_prompt_context(task, item),
        }
    )
    return micro_task


def aggregate_dataset_metrics(item_runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_items = len(item_runs)
    baseline_passed = sum(1 for item in item_runs if item["baseline"]["metrics"]["verifier_status"] == "pass")
    winner_passed = sum(1 for item in item_runs if item["winner"]["metrics"]["verifier_status"] == "pass")
    baseline_objective = sum(float(item["baseline"]["metrics"]["objective"] or 0.0) for item in item_runs)
    winner_objective = sum(float(item["winner"]["metrics"]["objective"] or 0.0) for item in item_runs)
    avg_delta_primary_score = sum(
        float(item.get("run_delta_primary_score") or item.get("delta_primary_score") or 0.0)
        for item in item_runs
    )
    failure_count = total_items - winner_passed

    if total_items:
        baseline_objective /= total_items
        winner_objective /= total_items
        avg_delta_primary_score /= total_items

    return {
        "total_items": total_items,
        "baseline_passed": baseline_passed,
        "winner_passed": winner_passed,
        "failure_count": failure_count,
        "solved_ratio": round((winner_passed / total_items) if total_items else 0.0, 6),
        "avg_baseline_objective": round(baseline_objective, 6),
        "avg_winner_objective": round(winner_objective, 6),
        "avg_delta_primary_score": round(avg_delta_primary_score, 6),
    }


def _aggregate_verifier_status(statuses: list[object]) -> str:
    normalized = [str(status or "").strip().lower() for status in statuses if str(status or "").strip()]
    if not normalized:
        return "not-run"
    unique = set(normalized)
    if len(unique) == 1:
        return normalized[0]
    return "mixed"


def aggregate_candidate(role: str, item_runs: list[dict[str, Any]], objective_label: str) -> dict[str, Any]:
    objective_total = sum(float(item[role]["metrics"]["objective"] or 0.0) for item in item_runs)
    objective_score_total = sum(float(item[role]["metrics"].get("objective_score") or 0.0) for item in item_runs)
    primary_score_total = sum(float(item[role]["metrics"].get("primary_score") or 0.0) for item in item_runs)
    tie_break_score_total = sum(float(item[role]["metrics"].get("tie_break_score") or 0.0) for item in item_runs)
    total_items = len(item_runs)
    aggregate_status = _aggregate_verifier_status(
        [item[role]["metrics"].get("verifier_status") or item[role]["metrics"].get("status") for item in item_runs]
    )
    gate_passed = bool(item_runs) and all(bool(item[role]["metrics"].get("gate_passed")) for item in item_runs)
    if total_items:
        objective_total /= total_items
        objective_score_total /= total_items
        primary_score_total /= total_items
        tie_break_score_total /= total_items
    label = "Dataset baseline aggregate" if role == "baseline" else "Dataset winner aggregate"
    return {
        "agent": role,
        "label": label,
        "strategy": f"Aggregate {role} summary across dataset questions.",
        "rationale": f"Dataset-level aggregate for {objective_label}.",
        "candidate_summary": label,
        "proposal_model": None,
        "source_code": "",
        "metrics": {
            "objective": round(objective_total, 6),
            "objective_score": round(objective_score_total, 6),
            "primary_score": round(primary_score_total, 6),
            "tie_break_score": round(tie_break_score_total, 6),
            "gate_passed": gate_passed,
            "verifier_status": aggregate_status,
            "status": aggregate_status,
        },
    }
