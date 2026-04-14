from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks, seed_strategy_experiences, task_summary
from app.codegen.errors import ConfigError
from app.codegen.task_contracts import interaction_mode_summary, task_mode_summary
from app.configs.codegen import (
    DEFAULT_SESSION_ID,
    DELTA_FORMULA,
    FLYWHEEL_STEPS,
    ITEM_MEMORY_DIR_NAME,
    OBJECTIVE_FORMULA,
    PRIMARY_FORMULA,
    RUN_DELTA_FORMULA,
    TIE_BREAK_FORMULA,
    WORKING_MEMORY_MD_NAME,
    WORKING_MEMORY_NAME,
    WORKING_MEMORY_TITLE,
)
from app.configs.paths import ROOT, RUNS_ROOT
from app.codegen.dataset_runner import run_dataset_task
from app.codegen.dataset_support import is_dataset_task
from app.codegen.llm import ProposalRuntime
from app.codegen.trainer import run_codegen_task
from app.memory.skills import (
    annotate_task_catalog_with_skills,
    annotate_task_summary_with_skills,
    append_distilled_skill_prompt_context,
    distill_dataset_skill,
    load_task_skill_markdown,
)
from app.memory.store import MemoryStore

RUNS = RUNS_ROOT
ProgressCallback = Callable[[dict[str, Any]], None]
CLI_COMMANDS = frozenset(
    {
        "tasks",
        "runtime",
        "latest-run",
        "run-task",
        "run-sequence",
        "prepare-datasets",
        "audit-datasets",
        "plan-dataset-smoke",
        "smoke-test-datasets",
    }
)
DATASET_SMOKE_PLACEHOLDER_HINTS = (
    "placeholder",
    "proxy",
    "scaffold",
    "hidden",
    "gated",
    "planned task",
)

def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_remote(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return result.stdout.strip() or "unknown"


def _validate_runtime_dependencies(tasks: list[dict[str, Any]]) -> None:
    math_tasks = [str(task["id"]) for task in tasks if str(task.get("track") or "") == "math_verified"]
    if not math_tasks:
        return
    if importlib.util.find_spec("math_verify") is not None:
        return
    task_list = ", ".join(sorted(math_tasks))
    raise ConfigError(
        "math_verified tasks require the 'math-verify' package in the active Python interpreter. "
        f"Current interpreter: {sys.executable}. "
        f"Affected tasks: {task_list}. "
        "Install it in this interpreter and restart the server."
    )


def generate_discrete_payload(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
    workspace_root: Path | None = None,
    session_id: str | None = None,
    generation_budget: int | None = None,
    candidate_budget: int | None = None,
    branching_factor: int | None = None,
    item_workers: int | None = None,
    max_items: int | None = None,
    max_episodes: int | None = None,
    selected_item_ids: list[str] | None = None,
    suite_config: dict[str, Any] | None = None,
    eval_model: str | None = None,
    record_skill: bool = False,
    skill_item_limit: int | None = None,
    selected_skill_id: str | None = None,
) -> dict[str, Any]:
    active_runs_root = runs_root or RUNS
    active_workspace_root = workspace_root or (active_runs_root / "workspace" / "current")
    active_runs_root.mkdir(parents=True, exist_ok=True)
    if task_id is not None:
        tasks = load_codegen_tasks(task_id)
        if not tasks:
            raise ValueError(f"Unknown task id: {task_id}")
    else:
        tasks = load_codegen_tasks(included_in_main_comparison=True)
    if generation_budget is not None or candidate_budget is not None or branching_factor is not None or item_workers is not None:
        overridden_tasks: list[dict[str, Any]] = []
        for task in tasks:
            patched = dict(task)
            if generation_budget is not None:
                patched["generation_budget"] = generation_budget
            if candidate_budget is not None:
                patched["candidate_budget"] = candidate_budget
            if branching_factor is not None:
                patched["branching_factor"] = branching_factor
            if item_workers is not None:
                patched["item_workers"] = item_workers
            overridden_tasks.append(patched)
        tasks = overridden_tasks
    if suite_config is not None:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("suite_config requires running exactly one task.")
        task = tasks[0]
        supports_suite_config = isinstance(task.get("runtime_suite_config"), dict)
        supports_runtime_split = (
            is_dataset_task(task)
            and str(task.get("interaction_mode") or "") == "single_turn"
            and isinstance(task.get("runtime_split_selector"), dict)
        )
        if not supports_suite_config and not supports_runtime_split:
            raise ConfigError(
                "suite_config is only supported for tasks that declare runtime_suite_config "
                "or single-turn dataset tasks that declare runtime_split_selector."
            )
        if supports_runtime_split and set(suite_config) - {"split"}:
            raise ConfigError("Dataset runtime split selection only supports suite_config.split.")
        if supports_suite_config:
            tasks = [{**task, "runtime_suite_config": dict(suite_config)}]
        else:
            tasks = [{**task, "suite_config": dict(suite_config)}]
    if selected_item_ids is not None:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("item_ids requires running exactly one task.")
        if not is_dataset_task(tasks[0]):
            raise ConfigError("item_ids is only supported for dataset tasks.")
    if record_skill:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("record_skill requires running exactly one task.")
        if not is_dataset_task(tasks[0]):
            raise ConfigError("record_skill is only supported for dataset tasks.")
    if selected_skill_id is not None:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("selected_skill_id requires running exactly one task.")
        selected_skill_markdown = load_task_skill_markdown(str(tasks[0]["id"]), selected_skill_id, runs_root=active_runs_root)
        tasks = [
            {
                **tasks[0],
                "prompt_context": append_distilled_skill_prompt_context(
                    str(tasks[0].get("prompt_context") or ""),
                    skill_markdown=selected_skill_markdown,
                    skill_label=selected_skill_id,
                ),
            }
        ]
    _validate_runtime_dependencies(tasks)
    legacy_store = MemoryStore(
        active_runs_root / WORKING_MEMORY_NAME,
        markdown_path=active_runs_root / WORKING_MEMORY_MD_NAME,
        title=WORKING_MEMORY_TITLE,
    )
    initial_memories = legacy_store.ensure_seed_records(seed_strategy_experiences())
    runtime = proposal_runtime or ProposalRuntime.from_env(env_root or ROOT)

    runs = []
    write_backs = 0
    total_generations = 0
    experiment_write_backs = 0
    total_run_count = 0
    experiment_run_count = 0
    total_memory_before = 0
    total_memory_after = 0
    for task in tasks:
        if is_dataset_task(task):
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=active_workspace_root,
                memory_root=active_runs_root / ITEM_MEMORY_DIR_NAME,
                session_id=session_id or DEFAULT_SESSION_ID,
                max_items=max_items,
                max_episodes=max_episodes,
                eval_model=eval_model,
                selected_item_ids=selected_item_ids,
                suite_config=suite_config,
                progress_callback=progress_callback,
                pace_ms=pace_ms,
            )
            if record_skill:
                source_count = skill_item_limit or len(result.get("item_runs", []))
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "skill_distillation_started",
                            "task_id": task["id"],
                            "message": f"Distilling reusable skill from the first {source_count} runtime-memory traces.",
                        }
                    )
                try:
                    generated_skill = distill_dataset_skill(
                        runtime,
                        task=task,
                        item_runs=list(result.get("item_runs") or []),
                        skill_item_limit=skill_item_limit,
                        session_id=session_id or DEFAULT_SESSION_ID,
                        runs_root=active_runs_root,
                    )
                    result["generated_skill"] = generated_skill
                    if progress_callback is not None and generated_skill is not None:
                        progress_callback(
                            {
                                "phase": "skill_distillation_completed",
                                "task_id": task["id"],
                                "message": (
                                    f"Saved distilled skill {generated_skill['filename']} "
                                    f"from {generated_skill['source_items']} items."
                                ),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    result["generated_skill_error"] = {
                        "error_type": "skill_distillation_error",
                        "error": str(exc),
                        "model": runtime.active_model,
                    }
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "skill_distillation_failed",
                                "task_id": task["id"],
                                "message": f"Skill distillation failed: {exc}",
                            }
                        )
            before_count = int(result.get("memory_before_count") or 0)
            after_count = int(result.get("memory_after_count") or 0)
            delta = after_count - before_count
        else:
            before_count = legacy_store.count()
            result = run_codegen_task(
                task,
                legacy_store,
                proposal_runtime=runtime,
                workspace_root=active_workspace_root,
                session_id=session_id or DEFAULT_SESSION_ID,
                progress_callback=progress_callback,
                pace_ms=pace_ms,
            )
            after_count = legacy_store.count()
            delta = after_count - before_count
            result["memory_markdown"] = legacy_store.load_markdown()

        total_run_count += 1
        if result["included_in_main_comparison"]:
            write_backs += delta
            total_generations += int(result.get("total_generations") or len(result["generations"]))
        else:
            experiment_run_count += 1
            experiment_write_backs += delta
        total_memory_before += before_count
        total_memory_after += after_count
        result["memory_before_count"] = before_count
        result["memory_after_count"] = after_count
        result["policy_model"] = str(result.get("active_model") or runtime.active_model)
        result["eval_model"] = eval_model
        runs.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "run_completed",
                    "task_id": task["id"],
                    "candidate": result["winner"]["agent"],
                    "generation": len(result["generations"]),
                    "message": (
                        f"Completed {task['id']} with objective={result['winner']['metrics']['objective']} "
                        f"and delta_primary_score={result['delta_primary_score']}"
                    ),
                    "delta_primary_score": result["delta_primary_score"],
                }
            )

    winners = Counter(run["winner"]["agent"] for run in runs if run["included_in_main_comparison"])
    task_catalog = annotate_task_catalog_with_skills(list_codegen_task_summaries(), runs_root=active_runs_root)
    if len(tasks) == 1 and task_id is not None:
        current_summary = annotate_task_summary_with_skills(task_summary(tasks[0]), runs_root=active_runs_root)
        task_catalog = [current_summary if task["id"] == current_summary["id"] else task for task in task_catalog]
    return {
        "run_mode": "llm-required",
        "summary": {
            "project": "autoresearch-foundry",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "source_repo": git_remote(ROOT),
            "run_mode": "llm-required",
            "active_model": runtime.active_model,
            "policy_model": runtime.active_model,
            "eval_model": eval_model,
            "num_tasks": len([run for run in runs if run["included_in_main_comparison"]]),
            "total_runs": total_run_count,
            "experiment_runs": experiment_run_count,
            "total_generations": total_generations,
            "initial_memory_count": total_memory_before or len(initial_memories),
            "memory_size_after_run": total_memory_after or legacy_store.count(),
            "write_backs": write_backs,
            "experiment_write_backs": experiment_write_backs,
            "winner_candidates": dict(winners),
            "proposal_engine": runtime.describe(),
            "flywheel": FLYWHEEL_STEPS,
        },
        "formulas": {
            "objective": OBJECTIVE_FORMULA,
            "primary_score": PRIMARY_FORMULA,
            "tie_break_score": TIE_BREAK_FORMULA,
            "delta_primary_score": DELTA_FORMULA,
            "run_delta_primary_score": RUN_DELTA_FORMULA,
        },
        "audit": {
            "workspace_root": _relative(active_workspace_root),
            "policy_model": runtime.active_model,
            "eval_model": eval_model,
            "max_items": max_items,
            "max_episodes": max_episodes,
            "selected_item_ids": list(selected_item_ids) if selected_item_ids is not None else None,
        },
        "task_catalog": task_catalog,
        "memory_markdown": legacy_store.load_markdown(),
        "runs": runs,
    }


def empty_discrete_payload(
    *,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
    eval_model: str | None = None,
) -> dict[str, Any]:
    active_runs_root = runs_root or RUNS
    runtime = proposal_runtime or ProposalRuntime.from_env(env_root or ROOT)
    workspace_root = active_runs_root / "workspace" / "current"
    return {
        "run_mode": "llm-required",
        "summary": {
            "project": "autoresearch-foundry",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "source_repo": git_remote(ROOT),
            "run_mode": "llm-required",
            "active_model": runtime.active_model,
            "policy_model": runtime.active_model,
            "eval_model": eval_model,
            "num_tasks": 0,
            "total_runs": 0,
            "experiment_runs": 0,
            "total_generations": 0,
            "initial_memory_count": 0,
            "memory_size_after_run": 0,
            "write_backs": 0,
            "experiment_write_backs": 0,
            "winner_candidates": {},
            "proposal_engine": runtime.describe(),
            "flywheel": FLYWHEEL_STEPS,
        },
        "formulas": {
            "objective": OBJECTIVE_FORMULA,
            "primary_score": PRIMARY_FORMULA,
            "tie_break_score": TIE_BREAK_FORMULA,
            "delta_primary_score": DELTA_FORMULA,
            "run_delta_primary_score": RUN_DELTA_FORMULA,
        },
        "audit": {
            "workspace_root": _relative(workspace_root),
            "policy_model": runtime.active_model,
            "eval_model": eval_model,
            "session_id": None,
        },
        "task_catalog": annotate_task_catalog_with_skills(list_codegen_task_summaries(), runs_root=active_runs_root),
        "memory_markdown": "",
        "runs": [],
    }


def load_cached_discrete_payload(
    *,
    task_id: str | None = None,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
) -> dict[str, Any]:
    active_runs_root = runs_root or RUNS
    candidate_paths: list[Path] = []
    if task_id is not None:
        candidate_paths.append(active_runs_root / f"codegen-{task_id}.json")
    candidate_paths.append(active_runs_root / "latest_run.json")
    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text())
    return empty_discrete_payload(
        proposal_runtime=proposal_runtime,
        runs_root=active_runs_root,
        env_root=env_root,
    )


def write_discrete_artifacts(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
    generation_budget: int | None = None,
    candidate_budget: int | None = None,
    branching_factor: int | None = None,
    item_workers: int | None = None,
    max_items: int | None = None,
    max_episodes: int | None = None,
    selected_item_ids: list[str] | None = None,
    suite_config: dict[str, Any] | None = None,
    eval_model: str | None = None,
    record_skill: bool = False,
    skill_item_limit: int | None = None,
    selected_skill_id: str | None = None,
) -> Path:
    active_runs_root = runs_root or RUNS
    session_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    active_workspace_root = active_runs_root / "workspace" / session_id

    def progress(event: dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": session_id,
            "event_type": event.get("phase", "unknown"),
            **event,
        }
        if progress_callback is not None:
            progress_callback(enriched)

    payload = generate_discrete_payload(
        task_id=task_id,
        progress_callback=progress,
        pace_ms=pace_ms,
        proposal_runtime=proposal_runtime,
        runs_root=active_runs_root,
        env_root=env_root,
        workspace_root=active_workspace_root,
        session_id=session_id,
        generation_budget=generation_budget,
        candidate_budget=candidate_budget,
        branching_factor=branching_factor,
        item_workers=item_workers,
        max_items=max_items,
        max_episodes=max_episodes,
        selected_item_ids=selected_item_ids,
        suite_config=suite_config,
        eval_model=eval_model,
        record_skill=record_skill,
        skill_item_limit=skill_item_limit,
        selected_skill_id=selected_skill_id,
    )
    payload["audit"]["session_id"] = session_id
    generated_at = str(payload["summary"]["generated_at"])
    for run in payload["runs"]:
        run["session_id"] = session_id
        run["generated_at"] = generated_at
    active_runs_root.mkdir(parents=True, exist_ok=True)
    out_name = f"codegen-{task_id}.json" if task_id else "codegen-latest.json"
    out = active_runs_root / out_name
    out.write_text(json.dumps(payload, indent=2))
    (active_runs_root / "latest_run.json").write_text(json.dumps(payload, indent=2))
    return out


def _add_common_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_task_id: bool,
    allow_suite_config: bool,
) -> None:
    if require_task_id:
        parser.add_argument("--task-id", required=True, help="Task id from benchmark/registry.json.")
    parser.add_argument("--model", help="Override the active proposal model for this run.")
    parser.add_argument("--llm-concurrency", type=int, help="Override in-flight LLM request concurrency for this run.")
    parser.add_argument("--generation-budget", type=int, help="Override generation budget for this run.")
    parser.add_argument("--candidate-budget", type=int, help="Override candidate budget for this run.")
    parser.add_argument("--branching-factor", type=int, help="Override branching factor for this run.")
    parser.add_argument("--item-workers", type=int, help="Override dataset item worker count for this run.")
    parser.add_argument("--max-items", type=int, help="Run only the first N items from the selected task or sequence.")
    parser.add_argument("--max-episodes", type=int, help="Run only the first N episodes from the selected multi-turn task.")
    parser.add_argument("--eval-model", help="Optional judge/eval model passed through to dataset verifiers that support it.")
    if allow_suite_config:
        parser.add_argument(
            "--suite-config",
            help="JSON object used as runtime suite_config, matching the server-side /api/run-task body.",
        )
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")


def _matching_task_summaries(
    *,
    task_id: str | None = None,
    track: str | None = None,
    tier: str | None = None,
    mode: str | None = None,
    main_only: bool = False,
) -> list[dict[str, Any]]:
    tasks = list_codegen_task_summaries()
    if task_id:
        tasks = [task for task in tasks if task["id"] == task_id]
    if track:
        tasks = [task for task in tasks if task["track"] == track]
    if tier:
        tasks = [task for task in tasks if task["benchmark_tier"] == tier]
    if mode:
        tasks = [task for task in tasks if task["task_mode"] == mode]
    if main_only:
        tasks = [task for task in tasks if task["included_in_main_comparison"]]
    return tasks


def _stringify_cli_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _render_kv_rows(rows: list[tuple[str, Any]]) -> str:
    if not rows:
        return ""
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label.ljust(width)} : {_stringify_cli_value(value)}" for label, value in rows)


def _print_task_table(tasks: list[dict[str, Any]]) -> None:
    if not tasks:
        print("No tasks matched the requested filters.")
        return
    columns = [
        ("id", "id"),
        ("tier", "benchmark_tier"),
        ("mode", "task_mode"),
        ("track", "track"),
        ("metric", "answer_metric"),
    ]
    widths = {
        header: max(len(header), *(len(str(task[key])) for task in tasks))
        for header, key in columns
    }
    header = "  ".join(header.ljust(widths[header]) for header, _ in columns)
    divider = "  ".join("-" * widths[header] for header, _ in columns)
    print(header)
    print(divider)
    for task in tasks:
        print("  ".join(str(task[key]).ljust(widths[header]) for header, key in columns))


def _print_task_detail(summary: dict[str, Any]) -> None:
    selection_spec = dict(summary.get("selection_spec") or {})
    objective_spec = dict(summary.get("objective_spec") or {})
    rows = [
        ("id", summary["id"]),
        ("title", summary["title"]),
        ("benchmark_tier", summary["benchmark_tier"]),
        ("included_in_main_comparison", summary["included_in_main_comparison"]),
        ("track", summary["track"]),
        ("task_mode", summary["task_mode"]),
        ("task_mode_summary", task_mode_summary(str(summary["task_mode"]))),
        ("interaction_mode", summary["interaction_mode"]),
        ("interaction_mode_summary", interaction_mode_summary(str(summary["interaction_mode"]))),
        ("dataset_id", summary["dataset_id"]),
        ("dataset_size", summary["dataset_size"]),
        ("split", summary["split"]),
        ("answer_metric", summary["answer_metric"]),
        ("objective_name", objective_spec.get("display_name")),
        ("objective_direction", objective_spec.get("direction")),
        ("objective_unit", objective_spec.get("unit")),
        ("objective_formula", objective_spec.get("formula")),
        ("primary_score", selection_spec.get("primary_formula")),
        ("gate", selection_spec.get("gate_summary")),
        ("tie_break", selection_spec.get("tie_break_formula")),
        ("archive_features", selection_spec.get("archive_summary")),
        ("generation_budget", summary["generation_budget"]),
        ("candidate_budget", summary["candidate_budget"]),
        ("branching_factor", summary["branching_factor"]),
        ("item_workers", summary["item_workers"]),
        ("supports_max_items", summary["supports_max_items"]),
        ("default_max_items", summary["default_max_items"]),
        ("supports_max_episodes", summary.get("supports_max_episodes")),
        ("default_max_episodes", summary.get("default_max_episodes")),
        ("supports_runtime_config", summary["supports_runtime_config"]),
    ]
    print(_render_kv_rows(rows))
    print()
    print("description")
    print(summary["description"])
    suite_run_config = summary.get("suite_run_config")
    if suite_run_config is not None:
        print()
        print("suite_run_config")
        print(json.dumps(suite_run_config, indent=2, sort_keys=False))


def _print_cached_run_summary(payload: dict[str, Any]) -> None:
    summary = dict(payload.get("summary") or {})
    audit = dict(payload.get("audit") or {})
    rows = [
        ("generated_at", summary.get("generated_at")),
        ("policy_model", summary.get("policy_model") or summary.get("active_model")),
        ("eval_model", summary.get("eval_model")),
        ("num_tasks", summary.get("num_tasks")),
        ("total_runs", summary.get("total_runs")),
        ("total_generations", summary.get("total_generations")),
        ("write_backs", summary.get("write_backs")),
        ("experiment_runs", summary.get("experiment_runs")),
        ("session_id", audit.get("session_id")),
        ("workspace_root", audit.get("workspace_root")),
        ("max_items", audit.get("max_items")),
        ("max_episodes", audit.get("max_episodes")),
    ]
    print(_render_kv_rows(rows))
    runs = list(payload.get("runs") or [])
    if not runs:
        print()
        print("runs")
        print("No cached runs were found.")
        return
    print()
    print("runs")
    for run in runs:
        task = dict(run.get("task") or {})
        winner = dict(run.get("winner") or {})
        metrics = dict(winner.get("metrics") or {})
        line = (
            f"- {task.get('id', '<unknown>')}: "
            f"objective={metrics.get('objective', '-')}, "
            f"primary_score={metrics.get('primary_score', '-')}, "
            f"delta_primary_score={run.get('delta_primary_score', '-')}"
        )
        print(line)


def _load_prepare_datasets_module() -> Any:
    script_path = ROOT / "benchmark" / "prepare_datasets.py"
    spec = importlib.util.spec_from_file_location("benchmark_prepare_datasets", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import benchmark prepare helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_tasks_for_smoke(
    *,
    task_ids: list[str] | None = None,
    tracks: list[str] | None = None,
    main_only: bool = False,
) -> list[dict[str, Any]]:
    tasks = [task for task in load_codegen_tasks() if bool(task.get("local_dataset_only"))]
    if task_ids:
        allowed = set(task_ids)
        tasks = [task for task in tasks if str(task["id"]) in allowed]
    if tracks:
        allowed_tracks = set(tracks)
        tasks = [task for task in tasks if str(task.get("track") or "") in allowed_tracks]
    if main_only:
        tasks = [task for task in tasks if bool(task.get("included_in_main_comparison"))]
    return tasks


def _is_placeholder_dataset_task(task: dict[str, Any]) -> bool:
    dataset_size = int(task.get("dataset_size") or task.get("prepared_item_count") or 0)
    if dataset_size >= 100:
        return False
    searchable = " ".join(
        str(task.get(field) or "").lower()
        for field in ("description", "baseline_summary", "split", "dataset_id", "title")
    )
    if not any(token in searchable for token in DATASET_SMOKE_PLACEHOLDER_HINTS):
        return False
    return not bool(task.get("included_in_main_comparison", True))


def _dataset_size_for_smoke(task: dict[str, Any]) -> int:
    return int(task.get("dataset_size") or 0)


def _dataset_prepared_count(task: dict[str, Any]) -> int:
    return int(task.get("prepared_item_count") or 0)


def _dataset_smoke_row(
    task: dict[str, Any],
    *,
    max_items_cap: int,
    skip_placeholders: bool,
) -> dict[str, Any]:
    dataset_size = _dataset_size_for_smoke(task)
    prepared_count = _dataset_prepared_count(task)
    placeholder = _is_placeholder_dataset_task(task)
    action = "run"
    reason = f"dataset_size<={max_items_cap}; using full dataset"
    max_items = dataset_size
    if dataset_size <= 0:
        action = "skip"
        reason = "dataset_size is missing or zero"
        max_items = 0
    elif prepared_count > 0 and prepared_count < dataset_size:
        reason = f"local manifest incomplete ({prepared_count}/{dataset_size}); capping to prepared rows"
        max_items = min(prepared_count, max_items_cap)
    elif placeholder and skip_placeholders:
        action = "skip"
        reason = "placeholder/proxy dataset under smoke threshold"
        max_items = 0
    elif dataset_size > max_items_cap:
        reason = f"dataset_size>{max_items_cap}; capping smoke run"
        max_items = max_items_cap
    return {
        "task_id": str(task["id"]),
        "title": str(task.get("title") or task["id"]),
        "track": str(task.get("track") or ""),
        "dataset_size": dataset_size,
        "prepared_count": prepared_count if prepared_count > 0 else None,
        "max_items": max_items,
        "action": action,
        "reason": reason,
        "requires_eval_model": bool(task.get("requires_eval_model")),
        "default_eval_model": task.get("default_eval_model"),
        "included_in_main_comparison": bool(task.get("included_in_main_comparison")),
        "placeholder": placeholder,
    }


def _build_dataset_smoke_plan(
    *,
    task_ids: list[str] | None = None,
    tracks: list[str] | None = None,
    main_only: bool = False,
    max_items_cap: int = 100,
    skip_placeholders: bool = True,
) -> dict[str, Any]:
    tasks = _dataset_tasks_for_smoke(task_ids=task_ids, tracks=tracks, main_only=main_only)
    rows = [
        _dataset_smoke_row(task, max_items_cap=max_items_cap, skip_placeholders=skip_placeholders)
        for task in tasks
    ]
    return {
        "max_items_cap": max_items_cap,
        "skip_placeholders": skip_placeholders,
        "rows": rows,
    }


def _print_dataset_smoke_plan(plan: dict[str, Any]) -> None:
    rows = list(plan.get("rows") or [])
    if not rows:
        print("No dataset tasks matched the requested smoke-test filters.")
        return
    columns = [
        ("task_id", "task_id"),
        ("track", "track"),
        ("size", "dataset_size"),
        ("prepared", "prepared_count"),
        ("max_items", "max_items"),
        ("action", "action"),
        ("eval", "requires_eval_model"),
    ]
    widths = {
        header: max(len(header), *(len(_stringify_cli_value(row[key])) for row in rows))
        for header, key in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column, _ in columns)
    divider = "  ".join("-" * widths[column] for column, _ in columns)
    print(header)
    print(divider)
    for row in rows:
        print("  ".join(_stringify_cli_value(row[key]).ljust(widths[column]) for column, key in columns))
        print(f"reason={row['reason']}")


def _add_dataset_smoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[], help="Limit smoke testing to one or more dataset task ids.")
    parser.add_argument("--track", action="append", dest="tracks", default=[], help="Limit smoke testing to one or more tracks.")
    parser.add_argument("--main-only", action="store_true", help="Include only tasks in the main comparison set.")
    parser.add_argument("--max-items-cap", type=int, default=100, help="Smoke-test cap for larger datasets. Smaller real datasets run in full.")
    parser.add_argument(
        "--include-placeholders",
        action="store_true",
        help="Run small placeholder/proxy datasets too instead of skipping them.",
    )
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")


def _audit_dataset_tasks(
    *,
    task_ids: list[str] | None = None,
    tracks: list[str] | None = None,
    main_only: bool = False,
) -> dict[str, Any]:
    tasks = _dataset_tasks_for_smoke(task_ids=task_ids, tracks=tracks, main_only=main_only)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_dir = Path(str(task["task_dir"]))
        verifier_path = Path(str(task["verifier_path"]))
        item_manifest = str(task.get("item_manifest") or "").strip()
        manifest_path = task_dir / item_manifest if item_manifest else None
        manifest_exists = manifest_path.exists() if manifest_path is not None else False
        prepared_count = _dataset_prepared_count(task)
        declared_size = _dataset_size_for_smoke(task)
        size_status = "ok"
        if declared_size <= 0:
            size_status = "missing_dataset_size"
        elif not manifest_exists:
            size_status = "missing_manifest"
        elif prepared_count != declared_size:
            size_status = "count_mismatch"
        verifier_compile = True
        verifier_import = True
        verifier_error: str | None = None
        try:
            importlib.util.cache_from_source(str(verifier_path))
            import py_compile

            py_compile.compile(str(verifier_path), doraise=True)
        except Exception as exc:  # noqa: BLE001
            verifier_compile = False
            verifier_error = f"{type(exc).__name__}: {exc}"
        try:
            spec = importlib.util.spec_from_file_location(f"audit_{task['id'].replace('-', '_')}", verifier_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to create import spec for {verifier_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            verifier_import = False
            verifier_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "task_id": str(task["id"]),
                "track": str(task.get("track") or ""),
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "manifest_exists": manifest_exists,
                "dataset_size": declared_size,
                "prepared_count": prepared_count if prepared_count > 0 else None,
                "size_status": size_status,
                "task_json_exists": Path(str(task["task_path"])).exists(),
                "readme_exists": bool(task.get("readme_path")) and Path(str(task["readme_path"])).exists(),
                "prepare_exists": (task_dir / "prepare.py").exists(),
                "editable_exists": Path(str(task["editable_path"])).exists(),
                "verifier_exists": verifier_path.exists(),
                "verifier_compile": verifier_compile,
                "verifier_import": verifier_import,
                "verifier_error": verifier_error,
            }
        )
    return {
        "summary": {
            "dataset_tasks": len(rows),
            "missing_manifests": [row["task_id"] for row in rows if not row["manifest_exists"]],
            "count_mismatches": [row["task_id"] for row in rows if row["size_status"] == "count_mismatch"],
            "verifier_compile_failures": [row["task_id"] for row in rows if not row["verifier_compile"]],
            "verifier_import_failures": [row["task_id"] for row in rows if not row["verifier_import"]],
        },
        "rows": rows,
    }


def _print_dataset_audit(payload: dict[str, Any]) -> None:
    summary = dict(payload.get("summary") or {})
    print(_render_kv_rows(list(summary.items())))
    rows = list(payload.get("rows") or [])
    if not rows:
        return
    print()
    columns = [
        ("task_id", "task_id"),
        ("track", "track"),
        ("size", "dataset_size"),
        ("prepared", "prepared_count"),
        ("size_status", "size_status"),
        ("compile", "verifier_compile"),
        ("import", "verifier_import"),
    ]
    widths = {
        header: max(len(header), *(len(_stringify_cli_value(row[key])) for row in rows))
        for header, key in columns
    }
    print("  ".join(column.ljust(widths[column]) for column, _ in columns))
    print("  ".join("-" * widths[column] for column, _ in columns))
    for row in rows:
        print("  ".join(_stringify_cli_value(row[key]).ljust(widths[column]) for column, key in columns))
        if row["size_status"] != "ok":
            print(f"note=manifest={row['manifest_path']}")
        if row.get("verifier_error"):
            print(f"verifier_error={row['verifier_error']}")


def _parse_suite_config_arg(raw_value: str | None) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--suite-config must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--suite-config must decode to a JSON object.")
    return dict(parsed)


def _runtime_for_cli(
    model: str | None = None,
    llm_concurrency: int | None = None,
    item_workers: int | None = None,
) -> ProposalRuntime:
    runtime = ProposalRuntime.from_env()
    runtime = runtime.with_llm_concurrency(llm_concurrency if llm_concurrency is not None else item_workers)
    return runtime.with_model(model) if model else runtime


def _payload_from_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _handle_tasks_command(args: argparse.Namespace) -> None:
    tasks = _matching_task_summaries(
        task_id=args.task_id,
        track=args.track,
        tier=args.tier,
        mode=args.mode,
        main_only=args.main_only,
    )
    if args.pretty:
        if args.task_id and len(tasks) == 1:
            _print_task_detail(tasks[0])
            return
        _print_task_table(tasks)
        return
    _print_json({"tasks": tasks})


def _handle_runtime_command(args: argparse.Namespace) -> None:
    runtime = _runtime_for_cli()
    payload = runtime.describe()
    if args.pretty:
        print(_render_kv_rows(sorted(payload.items())))
        return
    _print_json(payload)


def _handle_run_task_command(args: argparse.Namespace) -> None:
    runtime = _runtime_for_cli(args.model, args.llm_concurrency, args.item_workers)
    suite_config = _parse_suite_config_arg(args.suite_config)
    out = write_discrete_artifacts(
        task_id=args.task_id,
        proposal_runtime=runtime,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
        branching_factor=args.branching_factor,
        item_workers=args.item_workers,
        max_items=args.max_items,
        max_episodes=args.max_episodes,
        suite_config=suite_config,
        eval_model=args.eval_model,
    )
    payload = _payload_from_artifact(out)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _handle_latest_run_command(args: argparse.Namespace) -> None:
    payload = load_cached_discrete_payload(task_id=args.task_id)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _handle_run_sequence_command(args: argparse.Namespace) -> None:
    runtime = _runtime_for_cli(args.model, args.llm_concurrency, args.item_workers)
    out = write_discrete_artifacts(
        task_id=None,
        proposal_runtime=runtime,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
        branching_factor=args.branching_factor,
        item_workers=args.item_workers,
        max_items=args.max_items,
        max_episodes=args.max_episodes,
        eval_model=args.eval_model,
    )
    payload = _payload_from_artifact(out)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _handle_prepare_datasets_command(args: argparse.Namespace) -> None:
    module = _load_prepare_datasets_module()
    forwarded_argv: list[str] = []
    if args.benchmark_root:
        forwarded_argv.extend(["--benchmark-root", args.benchmark_root])
    if args.registry:
        forwarded_argv.extend(["--registry", args.registry])
    for task_id in args.task_ids:
        forwarded_argv.extend(["--task-id", task_id])
    if args.python:
        forwarded_argv.extend(["--python", args.python])
    if args.list:
        forwarded_argv.append("--list")
    if args.debug:
        forwarded_argv.append("--debug")
    if args.dry_run:
        forwarded_argv.append("--dry-run")
    if args.continue_on_error:
        forwarded_argv.append("--continue-on-error")
    raise SystemExit(int(module.main(forwarded_argv)))


def _handle_audit_datasets_command(args: argparse.Namespace) -> None:
    payload = _audit_dataset_tasks(
        task_ids=list(args.task_ids),
        tracks=list(args.tracks),
        main_only=bool(args.main_only),
    )
    if args.pretty:
        _print_dataset_audit(payload)
        return
    _print_json(payload)


def _handle_plan_dataset_smoke_command(args: argparse.Namespace) -> None:
    plan = _build_dataset_smoke_plan(
        task_ids=list(args.task_ids),
        tracks=list(args.tracks),
        main_only=bool(args.main_only),
        max_items_cap=args.max_items_cap,
        skip_placeholders=not bool(args.include_placeholders),
    )
    if args.pretty:
        _print_dataset_smoke_plan(plan)
        return
    _print_json(plan)


def _handle_smoke_test_datasets_command(args: argparse.Namespace) -> None:
    plan = _build_dataset_smoke_plan(
        task_ids=list(args.task_ids),
        tracks=list(args.tracks),
        main_only=bool(args.main_only),
        max_items_cap=args.max_items_cap,
        skip_placeholders=not bool(args.include_placeholders),
    )
    runtime = _runtime_for_cli(args.model, args.llm_concurrency)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in plan["rows"]:
        if row["action"] != "run":
            results.append({**row, "status": "skipped"})
            continue
        eval_model = args.eval_model or row.get("default_eval_model")
        if row["requires_eval_model"] and not eval_model:
            message = f"{row['task_id']}: requires --eval-model for smoke testing"
            failures.append(message)
            results.append({**row, "status": "failed", "error": message})
            if not args.continue_on_error:
                break
            continue
        if args.dry_run:
            results.append({**row, "status": "planned", "eval_model": eval_model})
            continue
        try:
            artifact_path = write_discrete_artifacts(
                task_id=row["task_id"],
                proposal_runtime=runtime,
                generation_budget=args.generation_budget,
                candidate_budget=args.candidate_budget,
                branching_factor=args.branching_factor,
                item_workers=args.item_workers,
                max_items=int(row["max_items"]),
                max_episodes=args.max_episodes,
                eval_model=eval_model,
            )
            payload = _payload_from_artifact(artifact_path)
            results.append(
                {
                    **row,
                    "status": "ok",
                    "eval_model": eval_model,
                    "artifact_path": str(artifact_path),
                    "generated_at": payload.get("summary", {}).get("generated_at"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{row['task_id']}: {exc}"
            failures.append(message)
            results.append({**row, "status": "failed", "error": str(exc), "eval_model": eval_model})
            if not args.continue_on_error:
                break
    payload = {
        "plan": plan,
        "summary": {
            "planned": len(plan["rows"]),
            "ran": sum(1 for row in results if row.get("status") == "ok"),
            "skipped": sum(1 for row in results if row.get("status") == "skipped"),
            "failed": len(failures),
            "dry_run": bool(args.dry_run),
        },
        "results": results,
    }
    if args.pretty:
        print(_render_kv_rows(list(payload["summary"].items())))
        print()
        for row in results:
            line = f"- {row['task_id']}: status={row['status']}, max_items={row['max_items']}, reason={row['reason']}"
            if row.get("error"):
                line = f"{line}, error={row['error']}"
            print(line)
        return
    _print_json(payload)
    if failures:
        raise SystemExit(1)


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Workbench CLI for task discovery, single-task/sequence runs, and benchmark-local dataset "
            "preparation or smoke testing."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    tasks_parser = subparsers.add_parser("tasks", help="Mirror /api/tasks.")
    tasks_parser.add_argument("--task-id", help="Filter down to one task id.")
    tasks_parser.add_argument("--track", help="Filter by track such as coding_verified or science_verified.")
    tasks_parser.add_argument("--tier", choices=["comparable", "experiment"], help="Filter by benchmark tier metadata.")
    tasks_parser.add_argument("--mode", choices=["answer", "artifact"], help="Filter by task contract mode.")
    tasks_parser.add_argument("--main-only", action="store_true", help="Show only tasks included in the active benchmark task set.")
    tasks_parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    tasks_parser.set_defaults(handler=_handle_tasks_command)

    runtime_parser = subparsers.add_parser("runtime", help="Mirror /api/runtime.")
    runtime_parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    runtime_parser.set_defaults(handler=_handle_runtime_command)

    latest_run_parser = subparsers.add_parser("latest-run", help="Mirror /api/latest-run.")
    latest_run_parser.add_argument("--task-id", help="Prefer the cached payload for a single task.")
    latest_run_parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    latest_run_parser.set_defaults(handler=_handle_latest_run_command)

    run_task_parser = subparsers.add_parser("run-task", help="Mirror /api/run-task.")
    _add_common_run_arguments(run_task_parser, require_task_id=True, allow_suite_config=True)
    run_task_parser.set_defaults(handler=_handle_run_task_command)

    run_sequence_parser = subparsers.add_parser("run-sequence", help="Mirror /api/run-sequence.")
    _add_common_run_arguments(run_sequence_parser, require_task_id=False, allow_suite_config=False)
    run_sequence_parser.set_defaults(handler=_handle_run_sequence_command)

    prepare_parser = subparsers.add_parser("prepare-datasets", help="Run benchmark/prepare_datasets.py through the main CLI.")
    prepare_parser.add_argument("--benchmark-root", help="Override benchmark root; mainly useful for tests.")
    prepare_parser.add_argument("--registry", help="Override registry path; mainly useful for tests.")
    prepare_parser.add_argument("--task-id", action="append", dest="task_ids", default=[], help="Prepare one or more task ids.")
    prepare_parser.add_argument("--python", default=sys.executable, help="Python executable used for task-local prepare.py scripts.")
    prepare_parser.add_argument("--list", action="store_true", help="List dataset prepare readiness without executing prepare.py.")
    prepare_parser.add_argument("--debug", action="store_true", help="Print detailed local dataset readiness information.")
    prepare_parser.add_argument("--dry-run", action="store_true", help="Print prepare commands without executing them.")
    prepare_parser.add_argument("--continue-on-error", action="store_true", help="Keep preparing later datasets after a failure.")
    prepare_parser.set_defaults(handler=_handle_prepare_datasets_command)

    audit_parser = subparsers.add_parser(
        "audit-datasets",
        help="Audit enabled dataset tasks for manifest completeness and verifier health.",
    )
    audit_parser.add_argument("--task-id", action="append", dest="task_ids", default=[], help="Audit one or more task ids.")
    audit_parser.add_argument("--track", action="append", dest="tracks", default=[], help="Audit one or more tracks.")
    audit_parser.add_argument("--main-only", action="store_true", help="Audit only dataset tasks in the main comparison set.")
    audit_parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    audit_parser.set_defaults(handler=_handle_audit_datasets_command)

    smoke_plan_parser = subparsers.add_parser(
        "plan-dataset-smoke",
        help="Plan smoke-test coverage across dataset tasks with the limit-100 policy.",
    )
    _add_dataset_smoke_arguments(smoke_plan_parser)
    smoke_plan_parser.set_defaults(handler=_handle_plan_dataset_smoke_command)

    smoke_run_parser = subparsers.add_parser(
        "smoke-test-datasets",
        help="Run dataset tasks with the shared smoke-test policy.",
    )
    _add_dataset_smoke_arguments(smoke_run_parser)
    smoke_run_parser.add_argument("--model", help="Override the active proposal model for this smoke run.")
    smoke_run_parser.add_argument("--llm-concurrency", type=int, help="Override in-flight LLM request concurrency.")
    smoke_run_parser.add_argument("--generation-budget", type=int, help="Override generation budget for each smoke-tested task.")
    smoke_run_parser.add_argument("--candidate-budget", type=int, help="Override candidate budget for each smoke-tested task.")
    smoke_run_parser.add_argument("--branching-factor", type=int, help="Override branching factor for each smoke-tested task.")
    smoke_run_parser.add_argument("--item-workers", type=int, help="Override item worker count for each smoke-tested task.")
    smoke_run_parser.add_argument("--max-episodes", type=int, help="Optional episode cap forwarded to multi-turn smoke-tested tasks.")
    smoke_run_parser.add_argument("--eval-model", help="Judge/eval model passed to tasks that support or require eval_model.")
    smoke_run_parser.add_argument("--dry-run", action="store_true", help="Plan smoke runs without executing them.")
    smoke_run_parser.add_argument("--continue-on-error", action="store_true", help="Keep running later smoke tests after a failure.")
    smoke_run_parser.set_defaults(handler=_handle_smoke_test_datasets_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_main_parser()
    if not args:
        parser.print_help()
        return
    if args[0] not in CLI_COMMANDS:
        parser.parse_args(args)
        return
    parsed = parser.parse_args(args)
    handler = getattr(parsed, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(parsed)


if __name__ == "__main__":
    main()
