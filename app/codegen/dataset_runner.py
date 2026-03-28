from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from app.codegen.catalog import seed_strategy_experiences
from app.codegen.errors import AutoresearchError, ConfigError
from app.configs.codegen import ITEM_MEMORY_JSON_NAME, ITEM_MEMORY_MD_NAME, QUESTION_PREVIEW_LIMIT
from app.codegen.dataset_support import (
    aggregate_candidate,
    aggregate_dataset_metrics,
    build_micro_task,
    is_dataset_task,
    load_question_manifest,
)
from app.codegen.llm import ProposalRuntime
from app.codegen.task_contracts import infer_optimization_scope, infer_runtime_backend, infer_task_mode
from app.codegen.trainer import run_codegen_task
from app.memory.store import MemoryStore


ProgressCallback = Callable[[dict[str, Any]], None]


def _question_preview(prompt: str, *, limit: int = QUESTION_PREVIEW_LIMIT) -> str:
    normalized = " ".join(str(prompt or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _memory_store_for_item(memory_root: Path, dataset_task_id: str, item_id: str) -> MemoryStore:
    item_dir = memory_root / dataset_task_id / item_id
    return MemoryStore(
        item_dir / ITEM_MEMORY_JSON_NAME,
        markdown_path=item_dir / ITEM_MEMORY_MD_NAME,
        title=f"{dataset_task_id}:{item_id} Strategy Memory",
    )


def _task_char_limit(task: dict[str, Any], key: str) -> int | None:
    value = task.get(key)
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _serialize_context_for_result(task: dict[str, Any], context: object) -> object:
    limit = _task_char_limit(task, "result_context_max_chars")
    if context is None or limit is None:
        return context
    if isinstance(context, (dict, list)):
        text = json.dumps(context, ensure_ascii=True)
    else:
        text = str(context)
    if len(text) <= limit:
        return context
    return text[:limit].rstrip() + f"... [truncated from {len(text)} chars in run artifact]"


def _question_payload_for_result(task: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["item_id"],
        "item_id": item["item_id"],
        "question_id": item["item_id"],
        "name": item.get("name") or item["item_id"],
        "prompt": item["prompt"],
        "raw_prompt": item.get("raw_prompt"),
        "context": _serialize_context_for_result(task, item.get("context")),
        "raw_context": _serialize_context_for_result(task, item.get("raw_context")),
        "choices": list(item.get("choices") or []),
        "raw_choices": list(item.get("raw_choices") or []),
        "expected_answer": item.get("expected_answer"),
        "raw_expected_answer": item.get("raw_expected_answer"),
        "metadata": dict(item.get("metadata") or {}),
    }


def _item_selector_aliases(item: dict[str, Any], *, position: int | None = None) -> set[str]:
    metadata = dict(item.get("metadata") or {})
    aliases = {
        str(item.get("item_id") or "").strip().lower(),
        str(item.get("raw_item_id") or "").strip().lower(),
        str(item.get("question_id") or "").strip().lower(),
        str(item.get("name") or "").strip().lower(),
        str(metadata.get("source_id") or "").strip().lower(),
    }
    source_index = metadata.get("source_index")
    if isinstance(source_index, int) and source_index >= 0:
        aliases.add(str(source_index + 1))
    if isinstance(position, int) and position > 0:
        aliases.add(str(position))
    return {alias for alias in aliases if alias}


def _select_requested_items(items: list[dict[str, Any]], selected_item_ids: list[str] | None) -> list[dict[str, Any]]:
    if not selected_item_ids:
        return items

    alias_map: dict[str, list[dict[str, Any]]] = {}
    for position, item in enumerate(items, start=1):
        for alias in _item_selector_aliases(item, position=position):
            alias_map.setdefault(alias, []).append(item)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    unmatched: list[str] = []
    ambiguous: list[str] = []
    for raw_token in selected_item_ids:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        matches = alias_map.get(token, [])
        unique_matches: list[dict[str, Any]] = []
        unique_ids: set[str] = set()
        for item in matches:
            item_id = str(item["item_id"])
            if item_id not in unique_ids:
                unique_ids.add(item_id)
                unique_matches.append(item)
        if not unique_matches:
            unmatched.append(raw_token)
            continue
        if len(unique_matches) > 1:
            ambiguous.append(f"{raw_token} -> {', '.join(str(item['item_id']) for item in unique_matches)}")
            continue
        item = unique_matches[0]
        item_id = str(item["item_id"])
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            selected.append(item)

    if unmatched or ambiguous:
        fragments: list[str] = []
        if unmatched:
            fragments.append(f"unknown item ids: {', '.join(unmatched)}")
        if ambiguous:
            fragments.append(f"ambiguous item ids: {'; '.join(ambiguous)}")
        raise ConfigError("Invalid item selection: " + ". ".join(fragments))
    return selected


def _progress_wrapper(
    *,
    progress_callback: ProgressCallback | None,
    dataset_task_id: str,
    item_id: str,
    item_name: str,
    item_prompt: str,
    expected_answer: object,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None

    item_brief = _question_preview(item_prompt)
    expected_answer_text = str(expected_answer).strip()

    def emit(event: dict[str, Any]) -> None:
        message = event.get("message")
        progress_callback(
            {
                **event,
                "task_id": dataset_task_id,
                "question_task_id": event.get("task_id"),
                "item_id": item_id,
                "item_name": item_name,
                "item_brief": item_brief,
                "expected_answer": expected_answer_text,
                "message": f"[{item_id}] {message}" if isinstance(message, str) and message else message,
            }
        )

    return emit


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AutoresearchError):
        return exc.as_payload()
    return {
        "terminal": True,
        "error_type": "runtime_error",
        "error": str(exc),
        "model": None,
    }


def _failed_candidate(
    *,
    role: str,
    summary: str,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "agent": role,
        "label": summary,
        "strategy": summary,
        "rationale": error_message or summary,
        "candidate_summary": summary,
        "proposal_model": None,
        "source_code": "",
        "metrics": {
            "objective": 0.0,
            "objective_score": 0.0,
            "primary_score": 0.0,
            "tie_break_score": 0.0,
            "gate_passed": False,
            "verifier_status": status,
            "status": status,
            "error": error_message,
            "test_results": [],
        },
    }


def _failed_item_result(
    *,
    task: dict[str, Any],
    item: dict[str, Any],
    proposal_runtime: ProposalRuntime,
    memory_before_count: int,
    memory_after_count: int,
    exc: Exception,
) -> dict[str, Any]:
    error_payload = _error_payload(exc)
    error_message = str(error_payload.get("error") or exc)
    error_trace: dict[str, Any] = {
        "phase": "item_failed",
        "item_id": item["item_id"],
        "selected_model": error_payload.get("model") or proposal_runtime.active_model,
        "error_type": error_payload.get("error_type"),
        "error": error_message,
    }
    details = error_payload.get("details")
    if isinstance(details, dict):
        error_trace.update(details)
    return {
        "run_mode": "llm-required",
        "active_model": proposal_runtime.active_model,
        "selection_spec": dict(task.get("selection_spec") or {}),
        "baseline": _failed_candidate(
            role="baseline",
            summary="Baseline not completed",
            status="not-run",
            error_message=error_message,
        ),
        "winner": _failed_candidate(
            role="winner",
            summary="Question run failed",
            status="error",
            error_message=error_message,
        ),
        "generations": [],
        "objective_curve": [],
        "llm_traces": [error_trace],
        "memory_before_count": memory_before_count,
        "memory_after_count": memory_after_count,
        "positive_experiences_added": 0,
        "negative_experiences_added": 0,
        "added_experiences": [],
        "memory_markdown": "",
        "delta_primary_score": 0.0,
        "run_delta_primary_score": 0.0,
        "run_delta_objective": 0.0,
        "selection_reason": f"Question run failed before completion: {error_message}",
        "error_payload": error_payload,
        "dataset_task_id": task["id"],
        "item_id": item["item_id"],
        "item_name": item.get("name") or item["item_id"],
        "question": _question_payload_for_result(task, item),
    }


def _run_item(
    *,
    task: dict[str, Any],
    item: dict[str, Any],
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    memory_root: Path,
    session_id: str,
    progress_callback: ProgressCallback | None,
    pace_ms: int,
) -> dict[str, Any]:
    micro_task = build_micro_task(task, item)
    store = _memory_store_for_item(memory_root, str(task["id"]), str(item["item_id"]))
    store.ensure_seed_records(seed_strategy_experiences())
    before_count = store.count()
    item_progress = _progress_wrapper(
        progress_callback=progress_callback,
        dataset_task_id=str(task["id"]),
        item_id=str(item["item_id"]),
        item_name=str(item.get("name") or item["item_id"]),
        item_prompt=str(item.get("raw_prompt") or item["prompt"]),
        expected_answer=item.get("raw_expected_answer") or item.get("expected_answer"),
    )
    try:
        result = run_codegen_task(
            micro_task,
            store,
            proposal_runtime=proposal_runtime,
            workspace_root=workspace_root / task["id"] / "item_runs" / item["item_id"],
            session_id=f"{session_id}-{item['item_id']}",
            progress_callback=item_progress,
            pace_ms=pace_ms,
        )
    except Exception as exc:
        after_count = store.count()
        if item_progress is not None:
            payload = _error_payload(exc)
            item_progress(
                {
                    "phase": "item_failed",
                    "selected_model": payload.get("model") or proposal_runtime.active_model,
                    "error_type": payload.get("error_type"),
                    "error": payload.get("error"),
                    "message": f"Question run failed: {payload.get('error') or exc}",
                }
            )
        return _failed_item_result(
            task=task,
            item=item,
            proposal_runtime=proposal_runtime,
            memory_before_count=before_count,
            memory_after_count=after_count,
            exc=exc,
        )
    after_count = store.count()
    result["memory_before_count"] = before_count
    result["memory_after_count"] = after_count
    result["memory_markdown"] = store.load_markdown()
    result["dataset_task_id"] = task["id"]
    result["item_id"] = item["item_id"]
    result["item_name"] = item.get("name") or item["item_id"]
    result["question"] = _question_payload_for_result(task, item)
    return result


def run_dataset_task(
    task: dict[str, Any],
    *,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    memory_root: Path,
    session_id: str,
    max_items: int | None = None,
    selected_item_ids: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    if not is_dataset_task(task):
        raise ValueError(f"Task {task['id']} does not use the dataset runtime backend.")

    requested_items = max_items if isinstance(max_items, int) and max_items > 0 else int(task.get("dataset_size") or 0) or None
    items = load_question_manifest(task, min_items=requested_items)
    if selected_item_ids:
        items = _select_requested_items(items, selected_item_ids)
    elif isinstance(max_items, int) and max_items > 0:
        items = items[:max_items]
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "dataset_loaded",
                "task_id": task["id"],
                "message": f"Loaded dataset {task['id']} with {len(items)} questions.",
            }
        )

    configured_workers = int(task.get("item_workers") or 20)
    max_workers = max(1, min(configured_workers, proposal_runtime.config.llm_concurrency, len(items)))
    if len(items) <= 1:
        item_runs = [
            _run_item(
                task=task,
                item=items[0],
                proposal_runtime=proposal_runtime,
                workspace_root=workspace_root,
                memory_root=memory_root,
                session_id=session_id,
                progress_callback=progress_callback,
                pace_ms=pace_ms,
            )
        ] if items else []
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_item: dict[Any, dict[str, Any]] = {}
        item_runs = []
        try:
            future_to_item = {
                executor.submit(
                    _run_item,
                    task=task,
                    item=item,
                    proposal_runtime=proposal_runtime,
                    workspace_root=workspace_root,
                    memory_root=memory_root,
                    session_id=session_id,
                    progress_callback=progress_callback,
                    pace_ms=pace_ms,
                ): item
                for item in items
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    item_runs.append(future.result())
                except Exception as exc:
                    item_runs.append(
                        _failed_item_result(
                            task=task,
                            item=item,
                            proposal_runtime=proposal_runtime,
                            memory_before_count=0,
                            memory_after_count=0,
                            exc=exc,
                        )
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    item_runs.sort(key=lambda item_run: str(item_run["item_id"]))
    summary = aggregate_dataset_metrics(item_runs)
    positive_experiences_added = sum(int(item_run.get("positive_experiences_added") or 0) for item_run in item_runs)
    negative_experiences_added = sum(int(item_run.get("negative_experiences_added") or 0) for item_run in item_runs)
    added_experiences = [
        {**experience, "item_id": item_run["item_id"]}
        for item_run in item_runs
        for experience in item_run.get("added_experiences", [])
    ]
    generations_total = sum(len(item_run.get("generations", [])) for item_run in item_runs)
    average_delta_primary_score = summary["avg_delta_primary_score"]
    average_objective_delta = round(
        summary["avg_winner_objective"] - summary["avg_baseline_objective"],
        6,
    )

    return {
        "run_mode": "llm-required",
        "active_model": proposal_runtime.active_model,
        "selection_spec": dict(task["selection_spec"]),
        "benchmark_tier": task["benchmark_tier"],
        "track": task["track"],
        "dataset_id": task["dataset_id"],
        "included_in_main_comparison": task["included_in_main_comparison"],
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "entry_symbol": task["entry_symbol"],
            "editable_file": task["editable_file"],
            "answer_metric": task["answer_metric"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "objective_spec": task["objective_spec"],
            "selection_spec": task["selection_spec"],
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "branching_factor": task["branching_factor"],
            "item_workers": configured_workers,
            "runtime_backend": infer_runtime_backend(task),
            "task_mode": infer_task_mode(task),
            "optimization_scope": infer_optimization_scope(task),
            "benchmark_tier": task["benchmark_tier"],
            "track": task["track"],
            "dataset_id": task["dataset_id"],
            "dataset_size": task["dataset_size"] or len(items),
            "local_dataset_only": task["local_dataset_only"],
            "split": task.get("split"),
            "included_in_main_comparison": task["included_in_main_comparison"],
            "run_baseline_verifier": bool(task.get("run_baseline_verifier", True)),
        },
        "baseline": aggregate_candidate("baseline", item_runs, task["objective_label"]),
        "winner": aggregate_candidate("winner", item_runs, task["objective_label"]),
        "dataset_summary": summary,
        "item_runs": item_runs,
        "generations": [],
        "objective_curve": [],
        "llm_traces": [
            trace
            for item_run in item_runs
            for trace in item_run.get("llm_traces", [])
        ],
        "memory_markdown": "",
        "memory_before_count": sum(int(item_run.get("memory_before_count") or 0) for item_run in item_runs),
        "memory_after_count": sum(int(item_run.get("memory_after_count") or 0) for item_run in item_runs),
        "positive_experiences_added": positive_experiences_added,
        "negative_experiences_added": negative_experiences_added,
        "added_experiences": added_experiences,
        "delta_primary_score": average_delta_primary_score,
        "run_delta_primary_score": average_delta_primary_score,
        "run_delta_objective": average_objective_delta,
        "selection_reason": (
            f"Dataset {task['id']} aggregated {summary['winner_passed']}/{summary['total_items']} solved questions "
            f"with average {task['objective_label']}={summary['avg_winner_objective']}."
        ),
        "total_generations": generations_total,
    }
