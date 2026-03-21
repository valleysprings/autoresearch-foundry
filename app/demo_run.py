from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.engine import run_task
from app.memory_store import MemoryStore
from app.task_catalog import list_task_summaries, load_tasks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
WORKING_MEMORY = RUNS / "working_memory.json"
ProgressCallback = Callable[[dict[str, Any]], None]

J_FORMULA = (
    "J = 1.20 * correctness + 0.95 * speed_score + 0.20 * memory_bonus "
    "+ 0.15 * stability - 0.18 * complexity - 0.05 * (line_count / 10)"
)
DELTA_FORMULA = "delta_J = J(winner) - J(baseline)"
SPEED_FORMULA = "speed_score = min(speedup_vs_baseline, 8) / 8"


def generate_demo_payload(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    tasks = load_tasks()
    if task_id is not None:
        tasks = [task for task in tasks if task["id"] == task_id]
        if not tasks:
            raise ValueError(f"Unknown task id: {task_id}")

    RUNS.mkdir(exist_ok=True)
    store = MemoryStore(WORKING_MEMORY)
    seed_memories = store.seed_from(DATA / "experiences.json")

    all_runs = []
    write_backs = 0
    for task in tasks:
        before_count = store.count()
        memories = store.retrieve(
            task_signature=task["task_signature"],
            family=task["family"],
            top_k=3,
        )
        result = run_task(
            task,
            memories,
            progress_callback=progress_callback,
            pace_ms=pace_ms,
        )
        if result["should_write_memory"] and result["new_experience"] is not None:
            if store.append(result["new_experience"]):
                write_backs += 1
        result["memory_before_count"] = before_count
        result["memory_after_count"] = store.count()
        all_runs.append(result)

    winners = Counter(run["winner"]["agent"] for run in all_runs)
    return {
        "summary": {
            "project": "autoresearch-with-experience-replay",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "num_tasks": len(all_runs),
            "initial_memory_count": len(seed_memories),
            "memory_size_after_run": store.count(),
            "write_backs": write_backs,
            "winner_agents": dict(winners),
            "flywheel": [
                "load baseline",
                "retrieve experience",
                "mutate candidate programs",
                "run fixed tests",
                "benchmark passing variants",
                "select winner and write memory",
            ],
            "karpathy_alignment": {
                "matches": [
                    "baseline-first comparison",
                    "multiple candidate architectures per task",
                    "deterministic keep/discard metric",
                    "experience logged only after measured improvement",
                ],
                "gaps": [
                    "not yet mutating train.py with an LLM",
                    "not yet an overnight autonomous loop",
                    "not yet training a real model architecture",
                ],
            },
        },
        "formulas": {
            "J": J_FORMULA,
            "speed_score": SPEED_FORMULA,
            "delta_J": DELTA_FORMULA,
        },
        "task_catalog": list_task_summaries(),
        "runs": all_runs,
    }


def write_demo_artifacts(
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> Path:
    payload = generate_demo_payload(
        task_id=task_id,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    out_name = f"{task_id}.json" if task_id else "latest_run.json"
    out = RUNS / out_name
    out.write_text(json.dumps(payload, indent=2))
    if task_id:
        (RUNS / "latest_run.json").write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local evolve demo.")
    parser.add_argument("--task", help="Run only one task id from the catalog.")
    parser.add_argument("--list-tasks", action="store_true", help="List available task ids.")
    args = parser.parse_args()

    if args.list_tasks:
        for task in list_task_summaries():
            print(f"{task['id']}: {task['title']}")
        return

    out = write_demo_artifacts(task_id=args.task)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
