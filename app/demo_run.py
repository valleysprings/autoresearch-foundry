from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.engine import run_task
from app.memory_store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
WORKING_MEMORY = RUNS / "working_memory.json"


def generate_demo_payload() -> dict[str, Any]:
    tasks = json.loads((DATA / "tasks.json").read_text())
    RUNS.mkdir(exist_ok=True)
    store = MemoryStore(WORKING_MEMORY)
    seed_memories = store.seed_from(DATA / "experiences.json")

    all_runs = []
    write_backs = 0

    for task in tasks:
        before_count = store.count()
        memories = store.retrieve(
            task_signature=task["task_signature"],
            target_device=task["profile"]["target_device"],
            top_k=3,
        )
        result = run_task(task, memories)
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
                "task intake",
                "experience retrieval",
                "candidate proposal generation",
                "deterministic evaluation",
                "winner selection",
                "selective write-back",
            ],
        },
        "roadmap": [
            {
                "phase": "Local search loop",
                "status": "implemented",
                "detail": "macOS-first proposal selection with MPS-safe kernels and deterministic evaluation.",
            },
            {
                "phase": "Experience replay",
                "status": "implemented",
                "detail": "task-signature retrieval and delta_J-gated memory consolidation.",
            },
            {
                "phase": "H200 handoff",
                "status": "demo-ready",
                "detail": "bridge local winning strategies into cluster-ready handoff bundles.",
            },
        ],
        "reference_mapping": [
            {
                "source": "karpathy/autoresearch",
                "takeaway": "fixed-budget experiment loop, results logging, and branchable research programs",
            },
            {
                "source": "miolini/autoresearch-macos",
                "takeaway": "Apple Silicon first constraints, MPS-safe attention choices, and small local runs",
            },
            {
                "source": "algorithmicsuperintelligence/openevolve",
                "takeaway": "candidate diversity, mutation lanes, and selection pressure over proposals",
            },
        ],
        "runs": all_runs,
    }


def write_demo_artifacts() -> Path:
    payload = generate_demo_payload()
    out = RUNS / "latest_run.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    out = write_demo_artifacts()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
