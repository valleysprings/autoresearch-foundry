from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from app.codegen.llm import ProposalRuntime, propose_code_candidates, reflect_strategy_experience
from app.codegen.verifier import evaluate_materialized_candidate, materialize_candidate
from app.memory.store import MemoryStore


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(progress_callback: ProgressCallback | None, pace_ms: int, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
    if pace_ms > 0:
        time.sleep(pace_ms / 1000.0)


def _baseline_candidate(task: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    source_path, source_code = materialize_candidate(
        task=task,
        workspace_root=workspace_root,
        candidate_id="baseline",
        imports=task.get("baseline_imports", []),
        function_body=task["baseline_body"],
    )
    metrics = evaluate_materialized_candidate(
        task=task,
        source_path=source_path,
        source_code=source_code,
        imports=task.get("baseline_imports", []),
        baseline_ms=None,
        memory_applied=False,
    )
    return {
        "candidate_id": f"{task['id']}-baseline",
        "agent": "baseline",
        "label": "Checked-in baseline",
        "strategy": "Use the embedded baseline implementation without any mutation.",
        "rationale": "This is the deterministic reference point that every candidate must beat.",
        "imports": task.get("baseline_imports", []),
        "function_body": task["baseline_body"],
        "source_code": source_code,
        "baseline_source": source_code,
        "candidate_summary": task["baseline_summary"],
        "workspace_path": str(source_path),
        "run_mode": "llm-required",
        "proposal_model": None,
        "metrics": metrics,
        "supporting_memory_ids": [],
    }


def _build_experience(
    task: dict[str, Any],
    generation: int,
    previous_best: dict[str, Any],
    winner: dict[str, Any],
    delta_j: float,
    reflection: dict[str, str],
) -> dict[str, Any]:
    return {
        "experience_id": f"exp-{task['id']}-g{generation}-{winner['agent']}",
        "experience_type": "strategy_experience",
        "source_task": task["id"],
        "family": task["family"],
        "task_signature": task["task_signature"],
        "failure_pattern": reflection["failure_pattern"],
        "strategy_hypothesis": reflection["strategy_hypothesis"],
        "successful_strategy": reflection["successful_strategy"],
        "prompt_fragment": reflection["prompt_fragment"],
        "tool_trace_summary": reflection["tool_trace_summary"],
        "delta_J": delta_j,
        "proposal_model": winner["proposal_model"],
        "candidate_summary": winner["candidate_summary"],
        "supporting_memory_ids": winner["supporting_memory_ids"],
    }


def run_codegen_task(
    task: dict[str, Any],
    store: MemoryStore,
    *,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    _emit(progress_callback, pace_ms, phase="task_loaded", task_id=task["id"], message=f"Loaded task {task['id']}")
    baseline = _baseline_candidate(task, workspace_root / task["id"])
    baseline_ms = baseline["metrics"]["benchmark_ms"]
    current_best = baseline
    candidate_history: list[dict[str, Any]] = []
    llm_traces: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    objective_curve = [
        {
            "generation": 0,
            "objective": baseline["metrics"]["objective"],
            "candidate_objective": baseline["metrics"]["objective"],
            "J": baseline["metrics"]["J"],
            "candidate_J": baseline["metrics"]["J"],
            "candidate_agent": baseline["agent"],
            "candidate_label": baseline["label"],
            "accepted": True,
            "proposal_model": None,
            "run_mode": "llm-required",
        }
    ]
    initial_retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)

    for generation in range(1, int(task["generation_budget"]) + 1):
        retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_started",
            task_id=task["id"],
            generation=generation,
            message=f"Generation {generation} retrieved {len(retrieved)} memories",
        )
        candidate_specs, proposal_trace = propose_code_candidates(
            proposal_runtime,
            task=task,
            generation=generation,
            current_best=current_best,
            candidate_history=candidate_history,
            memories=retrieved,
        )
        llm_traces.append(
            {
                "task_id": task["id"],
                "generation": generation,
                "phase": "proposal_generation",
                **proposal_trace,
            }
        )
        _emit(
            progress_callback,
            pace_ms,
            phase="proposal_generated",
            task_id=task["id"],
            generation=generation,
            candidate=proposal_trace["selected_model"],
            message=f"Generated {len(candidate_specs)} candidates with {proposal_trace['selected_model']}",
        )

        evaluated_candidates: list[dict[str, Any]] = []
        for index, spec in enumerate(candidate_specs, start=1):
            candidate_id = f"{task['id']}-g{generation}-c{index}"
            source_path, source_code = materialize_candidate(
                task=task,
                workspace_root=workspace_root / task["id"] / f"generation-{generation}",
                candidate_id=candidate_id,
                imports=spec["imports"],
                function_body=spec["function_body"],
            )
            metrics = evaluate_materialized_candidate(
                task=task,
                source_path=source_path,
                source_code=source_code,
                imports=spec["imports"],
                baseline_ms=baseline_ms,
                memory_applied=bool(retrieved),
            )
            candidate = {
                **spec,
                "candidate_id": candidate_id,
                "generation": generation,
                "active_model": proposal_runtime.active_model,
                "workspace_path": str(source_path),
                "source_code": source_code,
                "baseline_source": baseline["source_code"],
                "run_mode": "llm-required",
                "supporting_memory_ids": [memory["experience_id"] for memory in retrieved],
                "metrics": metrics,
                "verifier_status": metrics["verifier_status"],
            }
            candidate_history.append(candidate)
            evaluated_candidates.append(candidate)
            _emit(
                progress_callback,
                pace_ms,
                phase="candidate_verified",
                task_id=task["id"],
                generation=generation,
                candidate=candidate["agent"],
                message=(
                    f"{candidate['agent']} status={candidate['verifier_status']} "
                    f"objective={candidate['metrics']['objective']} J={candidate['metrics']['J']}"
                ),
            )

        evaluated_candidates.sort(key=lambda item: item["metrics"]["J"], reverse=True)
        generation_winner = evaluated_candidates[0]
        previous_best = current_best
        accepted = False
        if (
            generation_winner["metrics"]["status"] == "pass"
            and generation_winner["metrics"]["objective"] > previous_best["metrics"]["objective"]
        ):
            current_best = generation_winner
            accepted = True

        delta_j = round(generation_winner["metrics"]["J"] - previous_best["metrics"]["J"], 4)
        wrote_memory = False
        new_experience = None
        if (
            generation_winner["metrics"]["status"] == "pass"
            and generation_winner["metrics"]["objective"] > previous_best["metrics"]["objective"]
            and delta_j > float(task["epsilon"])
        ):
            reflection, reflection_trace = reflect_strategy_experience(
                proposal_runtime,
                task=task,
                generation=generation,
                previous_best=previous_best,
                winner=generation_winner,
                delta_j=delta_j,
            )
            llm_traces.append(
                {
                    "task_id": task["id"],
                    "generation": generation,
                    "phase": "memory_reflection",
                    **reflection_trace,
                }
            )
            new_experience = _build_experience(task, generation, previous_best, generation_winner, delta_j, reflection)
            wrote_memory = store.append(new_experience)
            if wrote_memory:
                memory_events.append(
                    {
                        "generation": generation,
                        "experience_id": new_experience["experience_id"],
                        "delta_J": delta_j,
                        "prompt_fragment": new_experience["prompt_fragment"],
                    }
                )
                _emit(
                    progress_callback,
                    pace_ms,
                    phase="memory_writeback",
                    task_id=task["id"],
                    generation=generation,
                    candidate=generation_winner["agent"],
                    message=f"Wrote {new_experience['experience_id']}",
                )

        objective_curve.append(
            {
                "generation": generation,
                "objective": current_best["metrics"]["objective"],
                "candidate_objective": generation_winner["metrics"]["objective"],
                "J": current_best["metrics"]["J"],
                "candidate_J": generation_winner["metrics"]["J"],
                "candidate_agent": generation_winner["agent"],
                "candidate_label": generation_winner["label"],
                "accepted": accepted,
                "active_model": proposal_runtime.active_model,
                "proposal_model": generation_winner["proposal_model"],
                "run_mode": "llm-required",
            }
        )
        generations.append(
            {
                "generation": generation,
                "run_mode": "llm-required",
                "active_model": proposal_runtime.active_model,
                "retrieved_memories": retrieved,
                "candidates": evaluated_candidates,
                "winner": generation_winner,
                "winner_accepted": accepted,
                "best_after_generation": current_best,
                "delta_J": delta_j,
                "wrote_memory": wrote_memory,
                "new_experience": new_experience,
            }
        )
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_finished",
            task_id=task["id"],
            generation=generation,
            candidate=current_best["agent"],
            message=f"Best objective after generation {generation}: {current_best['metrics']['objective']}",
        )

    run_delta_j = round(current_best["metrics"]["J"] - baseline["metrics"]["J"], 4)
    return {
        "run_mode": "llm-required",
        "active_model": proposal_runtime.active_model,
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "function_signature": task["function_signature"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "source_type": task["source_type"],
        },
        "baseline": baseline,
        "initial_retrieved_memories": initial_retrieved,
        "generations": generations,
        "winner": current_best,
        "delta_J": run_delta_j,
        "objective_curve": objective_curve,
        "memory_events": memory_events,
        "llm_traces": llm_traces,
        "proposal_engine": proposal_runtime.describe(),
        "selection_reason": (
            f"{current_best['label']} reached objective={current_best['metrics']['objective']} "
            f"with J={current_best['metrics']['J']} after {len(generations)} generations."
        ),
    }
