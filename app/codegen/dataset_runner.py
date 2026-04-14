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
from app.codegen.task_contracts import infer_interaction_mode, infer_task_mode
from app.codegen.trainer import run_codegen_task
from app.memory.store import MemoryStore


ProgressCallback = Callable[[dict[str, Any]], None]


def _question_preview(prompt: str, *, limit: int = QUESTION_PREVIEW_LIMIT) -> str:
    normalized = " ".join(str(prompt or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _raw_context_brief(item: dict[str, Any]) -> str | None:
    raw_context = item.get("raw_context")
    if not isinstance(raw_context, dict):
        return None

    benchmark = str(raw_context.get("benchmark") or "").strip().lower()
    prompt_text = ""
    for key in ("query", "latest_query", "question_text", "latest_user_message", "target_utterance", "user_message", "question", "instruction"):
        candidate = str(raw_context.get(key) or "").strip()
        if candidate:
            prompt_text = candidate
            break
    if not prompt_text:
        for key in ("dialogue", "new_dialogue", "old_dialogue"):
            turns = raw_context.get(key)
            if not isinstance(turns, list) or not turns:
                continue
            snippets: list[str] = []
            for turn in turns[-3:]:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or "speaker").strip()
                text = str(turn.get("text") or turn.get("content") or turn.get("value") or "").strip()
                if text:
                    snippets.append(f"{speaker}: {text}")
            if snippets:
                prompt_text = " ".join(snippets)
                break
    if not prompt_text:
        return None

    if benchmark == "socialbench":
        role_name = str(raw_context.get("role_name") or "").strip()
        if role_name:
            return f"{role_name}: {prompt_text}"
    return prompt_text


def _item_brief(item: dict[str, Any]) -> str:
    contextual = _raw_context_brief(item)
    if contextual:
        return _question_preview(contextual)
    return _question_preview(str(item.get("raw_prompt") or item.get("prompt") or ""))


def _question_field_names(item: dict[str, Any]) -> list[str]:
    field_names: list[str] = []

    prompt = str(item.get("raw_prompt") or item.get("prompt") or "").strip()
    if prompt:
        field_names.append("prompt")

    context = item.get("raw_context") if item.get("raw_context") is not None else item.get("context")
    if context not in (None, "", [], {}):
        field_names.append("context")

    raw_choices = item.get("raw_choices")
    choices = raw_choices if isinstance(raw_choices, list) and raw_choices else item.get("choices")
    if isinstance(choices, list) and choices:
        field_names.append("choices")

    return field_names


def _dataset_loaded_message(task: dict[str, Any], items: list[dict[str, Any]]) -> str:
    unit = "episodes" if str(task.get("interaction_mode") or "") == "multi_turn" else "questions"
    message = f"Loaded dataset {task['id']} with {len(items)} {unit}."
    if not items:
        return message

    field_names = _question_field_names(items[0])
    if len(field_names) < 2:
        return message

    return f"{message} Question fields ({len(field_names)}): {', '.join(field_names)}."


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


def _item_source_index(item: dict[str, Any]) -> int | None:
    metadata = dict(item.get("metadata") or {})
    raw_index = metadata.get("source_index")
    if isinstance(raw_index, bool):
        return None
    try:
        source_index = int(raw_index)
    except (TypeError, ValueError):
        return None
    return source_index if source_index >= 0 else None


def _item_run_source_index(item_run: dict[str, Any]) -> int | None:
    raw_index = item_run.get("item_source_index")
    if isinstance(raw_index, bool):
        raw_index = None
    if raw_index is not None:
        try:
            source_index = int(raw_index)
        except (TypeError, ValueError):
            source_index = None
        else:
            if source_index >= 0:
                return source_index
    question = item_run.get("question")
    if isinstance(question, dict):
        return _item_source_index(question)
    return None


def _item_run_sort_key(item_run: dict[str, Any]) -> tuple[bool, int, str]:
    source_index = _item_run_source_index(item_run)
    return (
        source_index is None,
        source_index if source_index is not None else 0,
        str(item_run["item_id"]),
    )


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
    item_source_index: int | None,
    item: dict[str, Any],
    expected_answer: object,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None

    item_brief = _item_brief(item)
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
                "item_source_index": item_source_index,
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
        "item_source_index": _item_source_index(item),
        "item_brief": _item_brief(item),
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
        item_source_index=_item_source_index(item),
        item=item,
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
    result["item_source_index"] = _item_source_index(item)
    result["item_brief"] = _item_brief(item)
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
    max_episodes: int | None = None,
    eval_model: str | None = None,
    selected_item_ids: list[str] | None = None,
    suite_config: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    if not is_dataset_task(task):
        raise ValueError(f"Task {task['id']} is not configured as a local dataset-backed task.")

    task_for_run = {
        **task,
        "eval_model": eval_model,
        "runtime_model_override": proposal_runtime.active_model,
    }
    uses_episode_limit = str(task_for_run.get("interaction_mode") or "") == "multi_turn"
    requested_limit = max_episodes if uses_episode_limit else max_items
    requested_items = requested_limit if isinstance(requested_limit, int) and requested_limit > 0 else int(task_for_run.get("dataset_size") or 0) or None
    items = load_question_manifest(task_for_run, min_items=requested_items, suite_config=suite_config)
    selected_runtime_split = None
    runtime_split_selector = task_for_run.get("runtime_split_selector")
    if isinstance(runtime_split_selector, dict):
        selected_runtime_split = str((suite_config or {}).get("split") or runtime_split_selector.get("default_value") or "").strip() or None
    if selected_item_ids:
        items = _select_requested_items(items, selected_item_ids)
    elif isinstance(requested_limit, int) and requested_limit > 0:
        items = items[:requested_limit]
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "dataset_loaded",
                "task_id": task_for_run["id"],
                "message": _dataset_loaded_message(task_for_run, items),
            }
        )

    configured_workers = int(task_for_run.get("item_workers") or 20)
    max_workers = max(1, min(configured_workers, len(items)))
    if len(items) <= 1:
        item_runs = [
            _run_item(
                task=task_for_run,
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
                    task=task_for_run,
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
                            task=task_for_run,
                            item=item,
                            proposal_runtime=proposal_runtime,
                            memory_before_count=0,
                            memory_after_count=0,
                            exc=exc,
                        )
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    item_runs.sort(key=_item_run_sort_key)
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
        "selection_spec": dict(task_for_run["selection_spec"]),
        "benchmark_tier": task_for_run["benchmark_tier"],
        "track": task_for_run["track"],
        "dataset_id": task_for_run["dataset_id"],
        "included_in_main_comparison": task_for_run["included_in_main_comparison"],
        "task": {
            "id": task_for_run["id"],
            "title": task_for_run["title"],
            "description": task_for_run["description"],
            "family": task_for_run["family"],
            "function_name": task_for_run["function_name"],
            "entry_symbol": task_for_run["entry_symbol"],
            "editable_file": task_for_run["editable_file"],
            "answer_metric": task_for_run["answer_metric"],
            "objective_label": task_for_run["objective_label"],
            "objective_direction": task_for_run["objective_direction"],
            "objective_spec": task_for_run["objective_spec"],
            "selection_spec": task_for_run["selection_spec"],
            "generation_budget": task_for_run["generation_budget"],
            "candidate_budget": task_for_run["candidate_budget"],
            "branching_factor": task_for_run["branching_factor"],
            "item_workers": configured_workers,
            "task_mode": infer_task_mode(task_for_run),
            "interaction_mode": infer_interaction_mode(task_for_run),
            "benchmark_tier": task_for_run["benchmark_tier"],
            "track": task_for_run["track"],
            "dataset_id": task_for_run["dataset_id"],
            "dataset_size": task_for_run["dataset_size"] or len(items),
            "local_dataset_only": task_for_run["local_dataset_only"],
            "split": task_for_run.get("split"),
            "research_line": task_for_run.get("research_line"),
            "personalization_category": task_for_run.get("personalization_category"),
            "personalization_focus": task_for_run.get("personalization_focus"),
            "safety_category": task_for_run.get("safety_category"),
            "safety_focus": task_for_run.get("safety_focus"),
            "included_in_main_comparison": task_for_run["included_in_main_comparison"],
            "supports_eval_model": bool(task_for_run.get("supports_eval_model")),
            "requires_eval_model": bool(task_for_run.get("requires_eval_model")),
            "default_eval_model": task_for_run.get("default_eval_model"),
            "run_baseline_verifier": bool(task_for_run.get("run_baseline_verifier", True)),
            "supports_runtime_config": isinstance(task_for_run.get("runtime_suite_config"), dict),
            "suite_run_config": task_for_run.get("runtime_suite_config"),
            "runtime_split_selector": task_for_run.get("runtime_split_selector"),
            "selected_runtime_split": selected_runtime_split,
            "supports_max_items": not uses_episode_limit,
            "default_max_items": (task_for_run["dataset_size"] or len(items)) if not uses_episode_limit else None,
            "supports_max_episodes": uses_episode_limit,
            "default_max_episodes": (task_for_run["dataset_size"] or len(items)) if uses_episode_limit else None,
        },
        "baseline": aggregate_candidate("baseline", item_runs, task_for_run["objective_label"]),
        "winner": aggregate_candidate("winner", item_runs, task_for_run["objective_label"]),
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
            f"Dataset {task_for_run['id']} aggregated {summary['winner_passed']}/{summary['total_items']} solved "
            f"{'episodes' if uses_episode_limit else 'questions'} "
            f"with average {task_for_run['objective_label']}={summary['avg_winner_objective']}."
        ),
        "total_generations": generations_total,
    }
