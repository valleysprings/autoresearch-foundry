from __future__ import annotations

import time
from typing import Any, Callable

from app.evaluator import evaluate_program

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(progress_callback: ProgressCallback | None, pace_ms: int, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
    if pace_ms > 0:
        time.sleep(pace_ms / 1000.0)


def planner(task: dict[str, Any], memories: list[dict[str, Any]]) -> dict[str, Any]:
    active_rules = sorted({rule for item in memories for rule in item.get("reusable_rules", [])})
    return {
        "task_id": task["id"],
        "objective": task["description"],
        "family": task["family"],
        "selection_policy": "run fixed tests first, benchmark only passing candidates, then maximize J",
        "priority_checks": [
            "preserve correctness under fixed unit cases",
            "improve runtime over the baseline implementation",
            "reuse retrieved memory only when it maps to a real code idiom",
            "write back experience only if delta_J clears epsilon",
        ],
        "active_rules": active_rules,
        "lanes": [spec["agent"] for spec in task["candidate_specs"]],
        "memory_count": len(memories),
    }


def _memory_support(candidate_spec: dict[str, Any], memories: list[dict[str, Any]]) -> list[str]:
    if not candidate_spec.get("uses_memory"):
        return []

    required_rules = set(candidate_spec.get("required_rules", []))
    if not required_rules:
        return [item["experience_id"] for item in memories]

    support_ids = []
    for item in memories:
        available_rules = set(item.get("reusable_rules", []))
        if required_rules & available_rules:
            support_ids.append(item["experience_id"])
    return support_ids


def _baseline_candidate(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": "baseline",
        "label": "Initial program",
        "architecture_family": "quadratic-baseline",
        "strategy": "Use the checked-in baseline implementation without any optimization.",
        "code": task["baseline_code"],
        "notes": [
            "This is the starting point that every candidate must beat.",
            "The baseline is executed exactly like the mutated candidates.",
        ],
        "uses_memory": False,
        "supporting_memory_ids": [],
        "reusable_rules": [],
        "complexity": 0.30,
    }


def _candidate_from_spec(spec: dict[str, Any], memories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "agent": spec["agent"],
        "label": spec["label"],
        "architecture_family": spec.get("architecture_family", "unknown"),
        "strategy": spec["strategy"],
        "code": spec["code"],
        "notes": spec.get("notes", []),
        "uses_memory": spec.get("uses_memory", False),
        "supporting_memory_ids": _memory_support(spec, memories),
        "reusable_rules": spec.get("reusable_rules", []),
        "complexity": float(spec.get("complexity", 0.3)),
    }


def _evaluate_candidate(
    candidate: dict[str, Any],
    task: dict[str, Any],
    baseline_ms: float | None,
) -> dict[str, Any]:
    enriched = dict(candidate)
    enriched["metrics"] = evaluate_program(
        code=candidate["code"],
        function_name=task["function_name"],
        tests=task["tests"],
        benchmark=task["benchmark"],
        baseline_ms=baseline_ms,
        complexity=float(candidate.get("complexity", 0.3)),
        memory_applied=bool(candidate.get("supporting_memory_ids")),
    )
    return enriched


def _build_experience(
    task: dict[str, Any],
    winner: dict[str, Any],
    baseline: dict[str, Any],
    delta_j: float,
) -> dict[str, Any]:
    return {
        "experience_id": f"exp-{task['id']}",
        "source_task": task["id"],
        "family": task["family"],
        "task_signature": task["task_signature"],
        "failure_pattern": baseline["strategy"],
        "successful_strategy": winner["strategy"],
        "tool_trace_summary": (
            f"{winner['agent']} passed {winner['metrics']['passed_tests']}/{winner['metrics']['total_tests']} tests "
            f"and ran in {winner['metrics']['benchmark_ms']} ms"
        ),
        "delta_J": delta_j,
        "reusable_rules": winner.get("reusable_rules", []),
        "supporting_memory_ids": winner.get("supporting_memory_ids", []),
        "code_pattern": winner["label"],
    }


def run_task(
    task: dict[str, Any],
    memories: list[dict[str, Any]],
    epsilon: float = 0.20,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    plan = planner(task, memories)

    _emit(progress_callback, pace_ms, phase="task_loaded", task_id=task["id"], message=f"Loaded task {task['id']}")

    baseline = _baseline_candidate(task)
    _emit(progress_callback, pace_ms, phase="baseline_started", task_id=task["id"], message="Evaluating baseline")
    baseline = _evaluate_candidate(baseline, task, baseline_ms=None)
    baseline_ms = baseline["metrics"]["benchmark_ms"]
    _emit(
        progress_callback,
        pace_ms,
        phase="baseline_finished",
        task_id=task["id"],
        message=f"Baseline finished in {baseline_ms} ms",
        metrics=baseline["metrics"],
    )

    candidates = []
    for spec in task["candidate_specs"]:
        candidate = _candidate_from_spec(spec, memories)
        _emit(
            progress_callback,
            pace_ms,
            phase="candidate_started",
            task_id=task["id"],
            candidate=candidate["agent"],
            architecture=candidate["architecture_family"],
            message=f"Running {candidate['agent']}",
        )
        evaluated = _evaluate_candidate(candidate, task, baseline_ms=baseline_ms)
        candidates.append(evaluated)
        _emit(
            progress_callback,
            pace_ms,
            phase="candidate_finished",
            task_id=task["id"],
            candidate=evaluated["agent"],
            architecture=evaluated["architecture_family"],
            message=f"{evaluated['agent']} J={evaluated['metrics']['J']}",
            metrics=evaluated["metrics"],
        )

    candidates.sort(key=lambda item: item["metrics"]["J"], reverse=True)
    winner = candidates[0]
    delta_j = round(winner["metrics"]["J"] - baseline["metrics"]["J"], 4)
    should_write = winner["metrics"]["status"] == "pass" and delta_j > epsilon
    new_experience = _build_experience(task, winner, baseline, delta_j) if should_write else None

    _emit(
        progress_callback,
        pace_ms,
        phase="winner_selected",
        task_id=task["id"],
        candidate=winner["agent"],
        architecture=winner["architecture_family"],
        message=f"Selected {winner['agent']} with delta_J={delta_j}",
        delta_J=delta_j,
    )

    if should_write and new_experience is not None:
        _emit(
            progress_callback,
            pace_ms,
            phase="memory_writeback",
            task_id=task["id"],
            candidate=winner["agent"],
            message=f"Writing experience {new_experience['experience_id']}",
            experience_id=new_experience["experience_id"],
        )

    architectures = [
        {
            "agent": baseline["agent"],
            "label": baseline["label"],
            "family": baseline["architecture_family"],
            "J": baseline["metrics"]["J"],
            "benchmark_ms": baseline["metrics"]["benchmark_ms"],
            "speedup_vs_baseline": baseline["metrics"]["speedup_vs_baseline"],
        }
    ] + [
        {
            "agent": candidate["agent"],
            "label": candidate["label"],
            "family": candidate["architecture_family"],
            "J": candidate["metrics"]["J"],
            "benchmark_ms": candidate["metrics"]["benchmark_ms"],
            "speedup_vs_baseline": candidate["metrics"]["speedup_vs_baseline"],
        }
        for candidate in candidates
    ]

    return {
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "baseline_path": task["baseline_path"],
            "task_signature": task["task_signature"],
        },
        "plan": plan,
        "retrieved_memories": memories,
        "baseline": baseline,
        "candidates": candidates,
        "architectures": architectures,
        "winner": winner,
        "delta_J": delta_j,
        "should_write_memory": should_write,
        "selection_reason": (
            f"{winner['agent']} won with J={winner['metrics']['J']} and "
            f"{winner['metrics']['speedup_vs_baseline']}x speedup while preserving correctness."
        ),
        "new_experience": new_experience,
    }
