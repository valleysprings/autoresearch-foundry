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
from app.codegen.external import is_external_task, run_external_task
from app.codegen.errors import ConfigError
from app.codegen.task_contracts import optimization_scope_summary, task_mode_summary
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
from app.memory.store import MemoryStore

RUNS = RUNS_ROOT
ProgressCallback = Callable[[dict[str, Any]], None]
CLI_COMMANDS = frozenset({"tasks", "runtime", "latest-run", "run-task", "run-sequence"})


def _external_task_uses_codegen_loop(task: dict[str, Any]) -> bool:
    if not is_external_task(task):
        return False
    return int(task.get("generation_budget") or 0) > 0 and int(task.get("candidate_budget") or 0) > 0


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
    selected_item_ids: list[str] | None = None,
    external_config: dict[str, Any] | None = None,
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
    if external_config is not None:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("external_config requires running exactly one task.")
        if not is_external_task(tasks[0]):
            raise ConfigError("external_config is only supported for external benchmark tasks.")
        tasks = [{**tasks[0], "runtime_external_config": dict(external_config)}]
    if selected_item_ids is not None:
        if task_id is None or len(tasks) != 1:
            raise ConfigError("item_ids requires running exactly one task.")
        if not is_dataset_task(tasks[0]):
            raise ConfigError("item_ids is only supported for dataset tasks.")
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
        if _external_task_uses_codegen_loop(task):
            task = {
                **task,
                "runtime_model_override": runtime.active_model,
                "runtime_max_items": max_items,
            }
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
        elif is_external_task(task):
            before_count = legacy_store.count()
            result = run_external_task(
                task,
                proposal_runtime=runtime,
                workspace_root=active_workspace_root / task["id"],
                session_id=session_id or DEFAULT_SESSION_ID,
                max_items=max_items,
                progress_callback=progress_callback,
                pace_ms=pace_ms,
            )
            after_count = legacy_store.count()
            delta = after_count - before_count
        elif is_dataset_task(task):
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=active_workspace_root,
                memory_root=active_runs_root / ITEM_MEMORY_DIR_NAME,
                session_id=session_id or DEFAULT_SESSION_ID,
                max_items=max_items,
                selected_item_ids=selected_item_ids,
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
                        f"and delta_primary_score={result['delta_primary_score']}"
                    ),
                    "delta_primary_score": result["delta_primary_score"],
                }
            )

    winners = Counter(run["winner"]["agent"] for run in runs if run["included_in_main_comparison"])
    task_catalog = list_codegen_task_summaries()
    if len(tasks) == 1 and task_id is not None:
        current_summary = task_summary(tasks[0])
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
            "max_items": max_items,
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
            "objective": OBJECTIVE_FORMULA,
            "primary_score": PRIMARY_FORMULA,
            "tie_break_score": TIE_BREAK_FORMULA,
            "delta_primary_score": DELTA_FORMULA,
            "run_delta_primary_score": RUN_DELTA_FORMULA,
        },
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
    selected_item_ids: list[str] | None = None,
    external_config: dict[str, Any] | None = None,
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
        selected_item_ids=selected_item_ids,
        external_config=external_config,
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
    allow_external_config: bool,
) -> None:
    if require_task_id:
        parser.add_argument("--task-id", required=True, help="Task id from benchmark/registry.json.")
    parser.add_argument("--model", help="Override the active proposal model for this run.")
    parser.add_argument("--generation-budget", type=int, help="Override generation budget for this run.")
    parser.add_argument("--candidate-budget", type=int, help="Override candidate budget for this run.")
    parser.add_argument("--branching-factor", type=int, help="Override branching factor for this run.")
    parser.add_argument("--item-workers", type=int, help="Override dataset item worker count for this run.")
    parser.add_argument("--max-items", type=int, help="Run only the first N items from the selected task or sequence.")
    if allow_external_config:
        parser.add_argument(
            "--external-config",
            help="JSON object used as runtime external_config, matching the server-side /api/run-task body.",
        )
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")


def _matching_task_summaries(
    *,
    task_id: str | None = None,
    track: str | None = None,
    tier: str | None = None,
    mode: str | None = None,
    backend: str | None = None,
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
    if backend:
        tasks = [task for task in tasks if task["runtime_backend"] == backend]
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
        ("backend", "runtime_backend"),
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
        ("runtime_backend", summary["runtime_backend"]),
        ("task_mode", summary["task_mode"]),
        ("task_mode_summary", task_mode_summary(str(summary["task_mode"]))),
        ("optimization_scope", summary["optimization_scope"]),
        ("optimization_scope_summary", optimization_scope_summary(str(summary["optimization_scope"]))),
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
        ("supports_runtime_config", summary["supports_runtime_config"]),
    ]
    print(_render_kv_rows(rows))
    print()
    print("description")
    print(summary["description"])
    external_run_config = summary.get("external_run_config")
    if external_run_config is not None:
        print()
        print("external_run_config")
        print(json.dumps(external_run_config, indent=2, sort_keys=False))


def _print_cached_run_summary(payload: dict[str, Any]) -> None:
    summary = dict(payload.get("summary") or {})
    audit = dict(payload.get("audit") or {})
    rows = [
        ("generated_at", summary.get("generated_at")),
        ("active_model", summary.get("active_model")),
        ("num_tasks", summary.get("num_tasks")),
        ("total_runs", summary.get("total_runs")),
        ("total_generations", summary.get("total_generations")),
        ("write_backs", summary.get("write_backs")),
        ("experiment_runs", summary.get("experiment_runs")),
        ("session_id", audit.get("session_id")),
        ("workspace_root", audit.get("workspace_root")),
        ("max_items", audit.get("max_items")),
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


def _parse_external_config_arg(raw_value: str | None) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--external-config must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--external-config must decode to a JSON object.")
    return dict(parsed)


def _runtime_for_cli(model: str | None = None) -> ProposalRuntime:
    runtime = ProposalRuntime.from_env()
    return runtime.with_model(model) if model else runtime


def _payload_from_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _handle_tasks_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Mirror the /api/tasks surface on the CLI.")
    parser.add_argument("--task-id", help="Filter down to one task id.")
    parser.add_argument("--track", help="Filter by track such as coding_verified or agent_verified.")
    parser.add_argument("--tier", choices=["comparable", "experiment"], help="Filter by benchmark tier.")
    parser.add_argument("--mode", choices=["answer", "artifact", "agent"], help="Filter by task contract mode.")
    parser.add_argument("--backend", choices=["dataset", "external"], help="Filter by runtime backend.")
    parser.add_argument("--main-only", action="store_true", help="Show only comparable tasks included in the main comparison.")
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    args = parser.parse_args(argv)
    tasks = _matching_task_summaries(
        task_id=args.task_id,
        track=args.track,
        tier=args.tier,
        mode=args.mode,
        backend=args.backend,
        main_only=args.main_only,
    )
    if args.pretty:
        if args.task_id and len(tasks) == 1:
            _print_task_detail(tasks[0])
            return
        _print_task_table(tasks)
        return
    _print_json({"tasks": tasks})


def _handle_runtime_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Mirror the /api/runtime surface on the CLI.")
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    args = parser.parse_args(argv)
    runtime = _runtime_for_cli()
    payload = runtime.describe()
    if args.pretty:
        print(_render_kv_rows(sorted(payload.items())))
        return
    _print_json(payload)


def _handle_run_task_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Mirror the /api/run-task surface on the CLI.")
    _add_common_run_arguments(parser, require_task_id=True, allow_external_config=True)
    args = parser.parse_args(argv)
    runtime = _runtime_for_cli(args.model)
    external_config = _parse_external_config_arg(args.external_config)
    out = write_discrete_artifacts(
        task_id=args.task_id,
        proposal_runtime=runtime,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
        branching_factor=args.branching_factor,
        item_workers=args.item_workers,
        max_items=args.max_items,
        external_config=external_config,
    )
    payload = _payload_from_artifact(out)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _handle_latest_run_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Mirror the /api/latest-run surface on the CLI.")
    parser.add_argument("--task-id", help="Prefer the cached payload for a single task.")
    parser.add_argument("--pretty", action="store_true", help="Render a human-readable summary instead of JSON.")
    args = parser.parse_args(argv)
    payload = load_cached_discrete_payload(task_id=args.task_id)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _handle_run_sequence_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Mirror the /api/run-sequence surface on the CLI.")
    _add_common_run_arguments(parser, require_task_id=False, allow_external_config=False)
    args = parser.parse_args(argv)
    runtime = _runtime_for_cli(args.model)
    out = write_discrete_artifacts(
        task_id=None,
        proposal_runtime=runtime,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
        branching_factor=args.branching_factor,
        item_workers=args.item_workers,
        max_items=args.max_items,
    )
    payload = _payload_from_artifact(out)
    if args.pretty:
        _print_cached_run_summary(payload)
        return
    _print_json(payload)


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="CLI surfaces that mirror the workbench API: tasks, runtime, latest-run, run-task, and run-sequence.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("tasks", help="Mirror /api/tasks.")
    subparsers.add_parser("runtime", help="Mirror /api/runtime.")
    subparsers.add_parser("latest-run", help="Mirror /api/latest-run.")
    subparsers.add_parser("run-task", help="Mirror /api/run-task.")
    subparsers.add_parser("run-sequence", help="Mirror /api/run-sequence.")
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

    command = args[0]
    command_argv = args[1:]
    if command == "tasks":
        _handle_tasks_command(command_argv)
        return
    if command == "runtime":
        _handle_runtime_command(command_argv)
        return
    if command == "latest-run":
        _handle_latest_run_command(command_argv)
        return
    if command == "run-task":
        _handle_run_task_command(command_argv)
        return
    if command == "run-sequence":
        _handle_run_sequence_command(command_argv)
        return


if __name__ == "__main__":
    main()
