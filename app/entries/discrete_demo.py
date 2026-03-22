from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks, seed_strategy_experiences
from app.codegen.handoff import UPSTREAM_TARGET, git_commit, git_remote, write_json, write_jsonl
from app.codegen.llm import ProposalRuntime
from app.codegen.reporting import build_improvement_table, write_improvement_report_svg
from app.codegen.trainer import run_codegen_task
from app.memory.store import MemoryStore


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
WORKING_MEMORY_NAME = "codegen_working_memory.json"
WORKING_MEMORY_MD_NAME = "codegen_working_memory.md"
ProgressCallback = Callable[[dict[str, Any]], None]

J_FORMULA = (
    "J = 1.20 * correctness + 0.95 * speed_score + 0.20 * memory_bonus "
    "+ 0.15 * stability - 0.18 * complexity - 0.05 * (line_count / 10)"
)
OBJECTIVE_FORMULA = "objective = speedup_vs_baseline"
DELTA_FORMULA = "delta_J = J(winner) - J(previous_best)"
FLYWHEEL_STEPS = [
    "load strict llm config from shell env or repo-root .env",
    "retrieve strategy memory fragments",
    "ask the configured model for candidate function bodies",
    "materialize candidates into an ignored workspace",
    "run deterministic tests and benchmarks",
    "select winners and write back reusable strategy experience",
    "emit payload, memory ledger, trace, and llm_trace artifacts",
]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
) -> dict[str, Any]:
    tasks = load_codegen_tasks()
    if task_id is not None:
        tasks = [task for task in tasks if task["id"] == task_id]
        if not tasks:
            raise ValueError(f"Unknown task id: {task_id}")
    if generation_budget is not None or candidate_budget is not None:
        overridden_tasks: list[dict[str, Any]] = []
        for task in tasks:
            patched = dict(task)
            if generation_budget is not None:
                patched["generation_budget"] = generation_budget
            if candidate_budget is not None:
                patched["candidate_budget"] = candidate_budget
            overridden_tasks.append(patched)
        tasks = overridden_tasks

    active_runs_root = runs_root or RUNS
    active_workspace_root = workspace_root or (active_runs_root / "workspace" / "current")
    active_runs_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(
        active_runs_root / WORKING_MEMORY_NAME,
        markdown_path=active_runs_root / WORKING_MEMORY_MD_NAME,
        title="Codegen Strategy Memory",
    )
    initial_memories = store.ensure_seed_records(seed_strategy_experiences())
    runtime = proposal_runtime or ProposalRuntime.from_env(env_root or ROOT)

    runs = []
    write_backs = 0
    total_generations = 0
    for task in tasks:
        before_count = store.count()
        result = run_codegen_task(
            task,
            store,
            proposal_runtime=runtime,
            workspace_root=active_workspace_root,
            session_id=session_id or "session-current",
            progress_callback=progress_callback,
            pace_ms=pace_ms,
        )
        after_count = store.count()
        write_backs += after_count - before_count
        total_generations += len(result["generations"])
        result["memory_before_count"] = before_count
        result["memory_after_count"] = after_count
        result["memory_markdown"] = store.load_markdown()
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

    winners = Counter(run["winner"]["agent"] for run in runs)
    return {
        "run_mode": "llm-required",
        "summary": {
            "project": "autoresearch-with-experience-replay",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "source_repo": git_remote(ROOT),
            "upstream_target": UPSTREAM_TARGET,
            "run_mode": "llm-required",
            "active_model": runtime.active_model,
            "num_tasks": len(runs),
            "total_generations": total_generations,
            "initial_memory_count": len(initial_memories),
            "memory_size_after_run": store.count(),
            "write_backs": write_backs,
            "winner_candidates": dict(winners),
            "proposal_engine": runtime.describe(),
            "flywheel": FLYWHEEL_STEPS,
        },
        "formulas": {
            "J": J_FORMULA,
            "objective": OBJECTIVE_FORMULA,
            "delta_J": DELTA_FORMULA,
        },
        "audit": {
            "upstream_target": UPSTREAM_TARGET,
            "workspace_root": _relative(active_workspace_root),
        },
        "task_catalog": list_codegen_task_summaries(),
        "memory_markdown": store.load_markdown(),
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
            "project": "autoresearch-with-experience-replay",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "source_repo": git_remote(ROOT),
            "upstream_target": UPSTREAM_TARGET,
            "run_mode": "llm-required",
            "active_model": runtime.active_model,
            "num_tasks": 0,
            "total_generations": 0,
            "initial_memory_count": 0,
            "memory_size_after_run": 0,
            "write_backs": 0,
            "winner_candidates": {},
            "proposal_engine": runtime.describe(),
            "flywheel": FLYWHEEL_STEPS,
        },
        "formulas": {
            "J": J_FORMULA,
            "objective": OBJECTIVE_FORMULA,
            "delta_J": DELTA_FORMULA,
        },
        "audit": {
            "upstream_target": UPSTREAM_TARGET,
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


def _write_handoff_bundle(
    *,
    payload: dict[str, Any],
    artifact_path: Path,
    events_by_task: dict[str, list[dict[str, Any]]],
    handoff_root: Path,
) -> None:
    handoff_root.mkdir(parents=True, exist_ok=True)
    session_id = str(payload["audit"].get("session_id") or "session-unknown")
    generated_at = str(payload["summary"]["generated_at"])
    for run in payload["runs"]:
        task_id = run["task"]["id"]
        bundle_dir = handoff_root / session_id / task_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        objective_curve_path = bundle_dir / "objective_curve.json"
        trace_path = bundle_dir / "trace.jsonl"
        llm_trace_path = bundle_dir / "llm_trace.jsonl"
        memory_path = bundle_dir / "memory.md"
        report_svg_path = bundle_dir / "improvement_report.svg"
        improvement_table_path = bundle_dir / "improvement_table.json"
        manifest_path = bundle_dir / "manifest.json"

        write_json(objective_curve_path, run["objective_curve"])
        write_jsonl(trace_path, events_by_task.get(task_id, []))
        write_jsonl(llm_trace_path, run["llm_traces"])
        memory_path.write_text(run["memory_markdown"])
        improvement_table = build_improvement_table(run)
        write_json(improvement_table_path, improvement_table)
        write_improvement_report_svg(run, report_svg_path)

        artifact_paths = {
            "payload": _relative(artifact_path),
            "objective_curve": _relative(objective_curve_path),
            "trace": _relative(trace_path),
            "memory_markdown": _relative(memory_path),
            "llm_trace_jsonl": _relative(llm_trace_path),
            "report_svg": _relative(report_svg_path),
            "improvement_table_json": _relative(improvement_table_path),
        }
        manifest = {
            "source_repo": payload["summary"]["source_repo"],
            "upstream_target": payload["summary"]["upstream_target"],
            "generated_at": generated_at,
            "git_commit": payload["summary"]["git_commit"],
            "session_id": session_id,
            "task_id": task_id,
            "function_name": run["task"]["function_name"],
            "objective_label": run["task"]["objective_label"],
            "objective_direction": run["task"]["objective_direction"],
            "run_mode": "llm-required",
            "active_model": payload["summary"]["active_model"],
            "baseline_objective": run["baseline"]["metrics"]["objective"],
            "winner_objective": run["winner"]["metrics"]["objective"],
            "delta_J": run["delta_J"],
            "winner_candidate": run["winner"]["agent"],
            "winner_strategy_label": run["winner"]["label"],
            "artifact_paths": artifact_paths,
        }
        write_json(manifest_path, manifest)
        run["session_id"] = session_id
        run["generated_at"] = generated_at
        run["improvement_table"] = improvement_table
        run["handoff_bundle"] = {
            "manifest": manifest,
            "manifest_path": _relative(manifest_path),
            "trace_event_count": len(events_by_task.get(task_id, [])),
        }


def write_discrete_artifacts(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
    proposal_runtime: ProposalRuntime | None = None,
    runs_root: Path | None = None,
    env_root: Path | None = None,
    generation_budget: int | None = None,
    candidate_budget: int | None = None,
) -> Path:
    active_runs_root = runs_root or RUNS
    session_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    active_workspace_root = active_runs_root / "workspace" / session_id
    captured_events: list[dict[str, Any]] = []

    def progress(event: dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": session_id,
            "event_type": event.get("phase", "unknown"),
            **event,
        }
        captured_events.append(enriched)
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
    )
    payload["audit"]["session_id"] = session_id
    active_runs_root.mkdir(parents=True, exist_ok=True)
    out_name = f"codegen-{task_id}.json" if task_id else "codegen-latest.json"
    out = active_runs_root / out_name
    out.write_text(json.dumps(payload, indent=2))
    (active_runs_root / "latest_run.json").write_text(json.dumps(payload, indent=2))

    events_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in captured_events:
        task_key = event.get("task_id")
        if isinstance(task_key, str):
            events_by_task[task_key].append(event)

    _write_handoff_bundle(
        payload=payload,
        artifact_path=out,
        events_by_task=events_by_task,
        handoff_root=active_runs_root / "handoff",
    )
    out.write_text(json.dumps(payload, indent=2))
    (active_runs_root / "latest_run.json").write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict LLM-required codegen demo.")
    parser.add_argument("--task", help="Run only one task id from the codegen catalog.")
    parser.add_argument("--list-tasks", action="store_true", help="List available task ids.")
    parser.add_argument("--generation-budget", type=int, help="Override generation budget for this run.")
    parser.add_argument("--candidate-budget", type=int, help="Override candidate budget for this run.")
    args = parser.parse_args()

    if args.list_tasks:
        for task in list_codegen_task_summaries():
            print(f"{task['id']}: {task['title']}")
        return

    out = write_discrete_artifacts(
        task_id=args.task,
        generation_budget=args.generation_budget,
        candidate_budget=args.candidate_budget,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
