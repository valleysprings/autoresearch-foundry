from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from app.codegen.catalog import seed_strategy_experiences
from app.configs.codegen import ITEM_MEMORY_JSON_NAME, ITEM_MEMORY_MD_NAME, QUESTION_PREVIEW_LIMIT
from app.codegen.dataset_support import (
    aggregate_candidate,
    aggregate_dataset_metrics,
    build_micro_task,
    is_dataset_task,
    load_question_manifest,
)
from app.codegen.llm import ProposalRuntime
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
    result = run_codegen_task(
        micro_task,
        store,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root / task["id"] / "item_runs" / item["item_id"],
        session_id=f"{session_id}-{item['item_id']}",
        progress_callback=_progress_wrapper(
            progress_callback=progress_callback,
            dataset_task_id=str(task["id"]),
            item_id=str(item["item_id"]),
            item_name=str(item.get("name") or item["item_id"]),
            item_prompt=str(item.get("raw_prompt") or item["prompt"]),
            expected_answer=item.get("raw_expected_answer") or item.get("expected_answer"),
        ),
        pace_ms=pace_ms,
    )
    after_count = store.count()
    result["memory_before_count"] = before_count
    result["memory_after_count"] = after_count
    result["memory_markdown"] = store.load_markdown()
    result["dataset_task_id"] = task["id"]
    result["item_id"] = item["item_id"]
    result["item_name"] = item.get("name") or item["item_id"]
    result["question"] = {
        "id": item["item_id"],
        "item_id": item["item_id"],
        "question_id": item["item_id"],
        "name": item.get("name") or item["item_id"],
        "prompt": item["prompt"],
        "raw_prompt": item.get("raw_prompt"),
        "context": item.get("context"),
        "raw_context": item.get("raw_context"),
        "choices": list(item.get("choices") or []),
        "raw_choices": list(item.get("raw_choices") or []),
        "expected_answer": item.get("expected_answer"),
        "raw_expected_answer": item.get("raw_expected_answer"),
        "metadata": dict(item.get("metadata") or {}),
    }
    return result


def run_dataset_task(
    task: dict[str, Any],
    *,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    memory_root: Path,
    session_id: str,
    max_items: int | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    if not is_dataset_task(task):
        raise ValueError(f"Task {task['id']} is not a dataset-task.")

    requested_items = max_items if isinstance(max_items, int) and max_items > 0 else int(task.get("dataset_size") or 0) or None
    items = load_question_manifest(task, min_items=requested_items)
    if isinstance(max_items, int) and max_items > 0:
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
    max_workers = max(1, min(configured_workers, len(items)))
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
        futures = []
        try:
            futures = [
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
                )
                for item in items
            ]
            item_runs = [future.result() for future in as_completed(futures)]
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
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
    average_delta_j = summary["avg_delta_J"]
    average_objective_delta = round(
        summary["avg_winner_objective"] - summary["avg_baseline_objective"],
        6,
    )

    return {
        "run_mode": "llm-required",
        "active_model": proposal_runtime.active_model,
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
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "branching_factor": task["branching_factor"],
            "item_workers": configured_workers,
            "source_type": task["source_type"],
            "benchmark_tier": task["benchmark_tier"],
            "track": task["track"],
            "dataset_id": task["dataset_id"],
            "dataset_size": task["dataset_size"] or len(items),
            "local_dataset_only": task["local_dataset_only"],
            "split": task.get("split"),
            "included_in_main_comparison": task["included_in_main_comparison"],
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
        "delta_J": average_delta_j,
        "run_delta_J": average_delta_j,
        "run_delta_objective": average_objective_delta,
        "selection_reason": (
            f"Dataset {task['id']} aggregated {summary['winner_passed']}/{summary['total_items']} solved questions "
            f"with average {task['objective_label']}={summary['avg_winner_objective']}."
        ),
        "total_generations": generations_total,
    }
