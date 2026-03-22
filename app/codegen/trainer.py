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
    session_id: str,
    generation: int,
    winner: dict[str, Any],
    delta_j: float,
    reflection: dict[str, str],
    *,
    outcome: str,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "experience_id": f"exp-{task['id']}-{session_id}-g{generation}-{winner['agent']}-{outcome}",
        "experience_type": "strategy_experience",
        "experience_outcome": outcome,
        "generation": generation,
        "source_task": task["id"],
        "source_session_id": session_id,
        "family": task["family"],
        "task_signature": task["task_signature"],
        "verifier_status": winner["metrics"]["verifier_status"],
        "rejection_reason": rejection_reason or "",
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


def _failure_reason(previous_best: dict[str, Any], winner: dict[str, Any]) -> str:
    verifier_status = winner["metrics"]["verifier_status"]
    if verifier_status == "error":
        return f"Candidate errored before verification completed: {winner['metrics'].get('error') or 'unknown error'}"
    if verifier_status == "fail":
        failed_tests = [result["name"] for result in winner["metrics"].get("test_results", []) if not result.get("passed")]
        if failed_tests:
            return "Candidate failed deterministic tests: " + ", ".join(failed_tests)
        return "Candidate failed deterministic correctness checks."
    if winner["metrics"]["objective"] <= previous_best["metrics"]["objective"]:
        return (
            "Candidate passed verification but did not improve the incumbent objective "
            f"({winner['metrics']['objective']} <= {previous_best['metrics']['objective']})."
        )
    return "Candidate was not accepted by the deterministic verifier."


def _objective(candidate: dict[str, Any]) -> float:
    return float(candidate.get("metrics", {}).get("objective") or 0.0)


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float]:
    return _objective(candidate), float(candidate.get("metrics", {}).get("J") or -1.0)


def _select_generation_winner(evaluated_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in evaluated_candidates if candidate["metrics"]["status"] == "pass"]
    if passing:
        return max(passing, key=_candidate_rank)
    return max(evaluated_candidates, key=lambda candidate: (float(candidate["metrics"]["J"]), _objective(candidate)))


def _select_parent(frontier: list[dict[str, Any]], current_best: dict[str, Any], generation: int) -> dict[str, Any]:
    if len(frontier) == 1:
        return current_best

    sorted_frontier = sorted(frontier, key=_candidate_rank, reverse=True)
    recent_non_best = [item for item in reversed(frontier) if item["candidate_id"] != current_best["candidate_id"]]
    strategy_index = (generation - 1) % 3
    if strategy_index == 0:
        return current_best
    if strategy_index == 1 and recent_non_best:
        return recent_non_best[0]
    if len(sorted_frontier) > 1:
        return sorted_frontier[1]
    return current_best


def _extend_frontier(frontier: list[dict[str, Any]], candidate: dict[str, Any], max_size: int = 6) -> list[dict[str, Any]]:
    if any(item.get("source_code") == candidate.get("source_code") for item in frontier):
        return frontier
    updated = frontier + [candidate]
    if len(updated) <= max_size:
        return updated
    best = max(updated, key=_candidate_rank)
    keep_ids = {best["candidate_id"]}
    kept: list[dict[str, Any]] = [best]
    for item in reversed(updated):
        candidate_id = item["candidate_id"]
        if candidate_id in keep_ids:
            continue
        kept.append(item)
        keep_ids.add(candidate_id)
        if len(kept) >= max_size:
            break
    kept.reverse()
    return kept


def _failure_reason_against_parent(parent_candidate: dict[str, Any], winner: dict[str, Any], epsilon: float) -> str:
    verifier_status = winner["metrics"]["verifier_status"]
    if verifier_status == "error":
        return f"Candidate errored before verification completed: {winner['metrics'].get('error') or 'unknown error'}"
    if verifier_status == "fail":
        failed_tests = [result["name"] for result in winner["metrics"].get("test_results", []) if not result.get("passed")]
        if failed_tests:
            return "Candidate failed deterministic tests: " + ", ".join(failed_tests)
        return "Candidate failed deterministic correctness checks."
    winner_objective = _objective(winner)
    parent_objective = _objective(parent_candidate)
    return (
        "Candidate passed verification but did not improve the selected parent objective "
        f"({winner_objective:.3f} <= {parent_objective:.3f} + epsilon {epsilon:.3f})."
    )


def _should_write_failure_memory(winner: dict[str, Any]) -> bool:
    return winner["metrics"]["verifier_status"] in {"fail", "error"}


def run_codegen_task(
    task: dict[str, Any],
    store: MemoryStore,
    *,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    session_id: str = "session-current",
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    _emit(progress_callback, pace_ms, phase="task_loaded", task_id=task["id"], message=f"Loaded task {task['id']}")
    baseline = _baseline_candidate(task, workspace_root / task["id"])
    baseline_ms = baseline["metrics"]["benchmark_ms"]
    current_best = baseline
    frontier = [baseline]
    candidate_history: list[dict[str, Any]] = []
    llm_traces: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    epsilon = float(task.get("epsilon", 0.0))
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
        parent_candidate = _select_parent(frontier, current_best, generation)
        retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_started",
            task_id=task["id"],
            generation=generation,
            candidate=parent_candidate["agent"],
            message=(
                f"Generation {generation} retrieved {len(retrieved)} memories "
                f"and selected parent {parent_candidate['agent']}"
            ),
        )
        candidate_specs, proposal_trace = propose_code_candidates(
            proposal_runtime,
            task=task,
            generation=generation,
            parent_candidate=parent_candidate,
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
                "parent_candidate_id": parent_candidate["candidate_id"],
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

        evaluated_candidates.sort(key=lambda item: float(item["metrics"]["J"]), reverse=True)
        generation_winner = _select_generation_winner(evaluated_candidates)
        previous_best = current_best
        accepted = False
        improved_global_best = False
        if generation_winner["metrics"]["status"] == "pass" and _objective(generation_winner) > _objective(parent_candidate) + epsilon:
            accepted = True
            frontier = _extend_frontier(frontier, generation_winner)
            if _objective(generation_winner) > _objective(previous_best) + epsilon:
                current_best = generation_winner
                improved_global_best = True

        delta_j = round(generation_winner["metrics"]["J"] - parent_candidate["metrics"]["J"], 4)
        global_best_delta_j = round(generation_winner["metrics"]["J"] - previous_best["metrics"]["J"], 4)
        experience_outcome = "success" if accepted else "failure"
        rejection_reason = None if accepted else _failure_reason_against_parent(parent_candidate, generation_winner, epsilon)
        wrote_memory = False
        new_experience = None
        should_reflect = accepted or _should_write_failure_memory(generation_winner)
        if should_reflect:
            reflection, reflection_trace = reflect_strategy_experience(
                proposal_runtime,
                task=task,
                generation=generation,
                previous_best=parent_candidate,
                winner=generation_winner,
                delta_j=delta_j,
                outcome=experience_outcome,
                rejection_reason=rejection_reason,
            )
            llm_traces.append(
                {
                    "task_id": task["id"],
                    "generation": generation,
                    "phase": "memory_reflection",
                    "experience_outcome": experience_outcome,
                    **reflection_trace,
                }
            )
            new_experience = _build_experience(
                task,
                session_id,
                generation,
                generation_winner,
                delta_j,
                reflection,
                outcome=experience_outcome,
                rejection_reason=rejection_reason,
            )
            wrote_memory = store.append(new_experience)
            if wrote_memory:
                memory_events.append(
                    {
                        "generation": generation,
                        "experience_id": new_experience["experience_id"],
                        "experience_outcome": experience_outcome,
                        "verifier_status": generation_winner["metrics"]["verifier_status"],
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
                    message=f"Wrote {new_experience['experience_id']} ({experience_outcome})",
                )
        else:
            _emit(
                progress_callback,
                pace_ms,
                phase="memory_skipped",
                task_id=task["id"],
                generation=generation,
                candidate=generation_winner["agent"],
                message="Skipped write-back for a passing but non-improving candidate.",
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
                "improved_global_best": improved_global_best,
                "parent_candidate": parent_candidate["agent"],
                "active_model": proposal_runtime.active_model,
                "proposal_model": generation_winner["proposal_model"],
                "experience_outcome": experience_outcome,
                "run_mode": "llm-required",
            }
        )
        generations.append(
            {
                "generation": generation,
                "run_mode": "llm-required",
                "active_model": proposal_runtime.active_model,
                "parent_candidate": parent_candidate,
                "retrieved_memories": retrieved,
                "candidates": evaluated_candidates,
                "winner": generation_winner,
                "winner_accepted": accepted,
                "winner_improved_global_best": improved_global_best,
                "best_after_generation": current_best,
                "delta_J": delta_j,
                "global_best_delta_J": global_best_delta_j,
                "experience_outcome": experience_outcome,
                "wrote_memory": wrote_memory,
                "new_experience": new_experience,
                "frontier_size": len(frontier),
            }
        )
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_finished",
            task_id=task["id"],
            generation=generation,
            candidate=current_best["agent"],
            message=(
                f"Best objective after generation {generation}: {current_best['metrics']['objective']} "
                f"(frontier={len(frontier)})"
            ),
        )

    run_delta_j = round(current_best["metrics"]["J"] - baseline["metrics"]["J"], 4)
    added_experiences = [
        generation["new_experience"]
        for generation in generations
        if generation.get("wrote_memory") and generation.get("new_experience")
    ]
    positive_experiences_added = sum(
        1 for experience in added_experiences if experience.get("experience_outcome") == "success"
    )
    negative_experiences_added = sum(
        1 for experience in added_experiences if experience.get("experience_outcome") == "failure"
    )
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
        "added_experiences": added_experiences,
        "positive_experiences_added": positive_experiences_added,
        "negative_experiences_added": negative_experiences_added,
        "memory_events": memory_events,
        "llm_traces": llm_traces,
        "proposal_engine": proposal_runtime.describe(),
        "selection_reason": (
            f"{current_best['label']} reached objective={current_best['metrics']['objective']} "
            f"with J={current_best['metrics']['J']} after {len(generations)} generations."
        ),
    }
