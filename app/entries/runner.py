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

from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks, seed_strategy_experiences
from app.codegen.errors import ConfigError
from app.configs.codegen import (
    DEFAULT_SESSION_ID,
    DELTA_FORMULA,
    DISCRETE_DEMO_J_SPEC,
    FLYWHEEL_STEPS,
    ITEM_MEMORY_DIR_NAME,
    J_FORMULA,
    OBJECTIVE_FORMULA,
    RUN_DELTA_FORMULA,
    WORKING_MEMORY_MD_NAME,
    WORKING_MEMORY_NAME,
    WORKING_MEMORY_TITLE,
)
from app.configs.paths import ROOT, RUNS_ROOT
from app.codegen.dataset_runner import run_dataset_task
from app.codegen.dataset_support import is_dataset_task
from app.codegen.llm import ProposalRuntime
from app.codegen.trainer import run_codegen_task
from app.memory.store import MemoryStore

RUNS = RUNS_ROOT
ProgressCallback = Callable[[dict[str, Any]], None]


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
) -> dict[str, Any]:
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
    _validate_runtime_dependencies(tasks)

    active_runs_root = runs_root or RUNS
    active_workspace_root = workspace_root or (active_runs_root / "workspace" / "current")
    active_runs_root.mkdir(parents=True, exist_ok=True)
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
                progress_callback=progress_callback,
                pace_ms=pace_ms,
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
                        f"and delta_J={result['delta_J']}"
                    ),
                    "delta_J": result["delta_J"],
                }
            )

    winners = Counter(run["winner"]["agent"] for run in runs if run["included_in_main_comparison"])
    return {
        "run_mode": "llm-required",
        "summary": {
            "project": "autoresearch-foundry",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "source_repo": git_remote(ROOT),
            "run_mode": "llm-required",
            "active_model": runtime.active_model,
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
            "J": J_FORMULA,
            "objective": OBJECTIVE_FORMULA,
            "delta_J": DELTA_FORMULA,
            "run_delta_J": RUN_DELTA_FORMULA,
        },
        "j_spec": dict(DISCRETE_DEMO_J_SPEC),
        "audit": {
            "workspace_root": _relative(active_workspace_root),
            "max_items": max_items,
        },
        "task_catalog": list_codegen_task_summaries(),
        "memory_markdown": legacy_store.load_markdown(),
        "runs": runs,
    }


def empty_discrete_payload(
    *,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
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
            "J": J_FORMULA,
            "objective": OBJECTIVE_FORMULA,
            "delta_J": DELTA_FORMULA,
            "run_delta_J": RUN_DELTA_FORMULA,
        },
        "j_spec": dict(DISCRETE_DEMO_J_SPEC),
        "audit": {
            "workspace_root": _relative(workspace_root),
            "session_id": None,
        },
        "task_catalog": list_codegen_task_summaries(),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch tasks from the CLI.")
    parser.add_argument("--task", help="Run only one task id from the codegen catalog.")
    parser.add_argument("--list-tasks", action="store_true", help="List available task ids.")
    parser.add_argument("--generation-budget", type=int, help="Override generation budget for this run.")
    parser.add_argument("--candidate-budget", type=int, help="Override candidate budget for this run.")
    parser.add_argument("--branching-factor", type=int, help="Override branching factor for this run.")
    parser.add_argument("--item-workers", type=int, help="Override dataset item worker count for this run.")
    parser.add_argument("--max-items", type=int, help="Run only the first N items from each dataset task.")
    args = parser.parse_args(argv)

    if args.list_tasks:
        for task in list_codegen_task_summaries():
            print(f"{task['id']}: {task['title']}")
        return

    out = write_discrete_artifacts(
        task_id=args.task,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
        branching_factor=args.branching_factor,
        item_workers=args.item_workers,
        max_items=args.max_items,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
