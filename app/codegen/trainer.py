from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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


def _objective_direction(task: dict[str, Any]) -> str:
    objective_spec = task.get("objective_spec") or {}
    return str(objective_spec.get("direction") or task.get("objective_direction") or "max")


def _objective_label(task: dict[str, Any]) -> str:
    objective_spec = task.get("objective_spec") or {}
    return str(objective_spec.get("display_name") or task.get("objective_label") or "objective")


def _objective_value(candidate: dict[str, Any]) -> float:
    return float(candidate.get("metrics", {}).get("objective") or 0.0)


def _objective_score(candidate: dict[str, Any]) -> float:
    metrics = candidate.get("metrics", {})
    if "objective_score" in metrics:
        return float(metrics.get("objective_score") or 0.0)
    return _objective_value(candidate)


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float]:
    return _objective_score(candidate), float(candidate.get("metrics", {}).get("J") or -1.0)


def _candidate_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "agent": candidate.get("agent"),
        "label": candidate.get("label"),
        "candidate_summary": candidate.get("candidate_summary"),
        "proposal_model": candidate.get("proposal_model"),
        "metrics": {
            "objective": candidate.get("metrics", {}).get("objective"),
            "objective_score": candidate.get("metrics", {}).get("objective_score"),
            "J": candidate.get("metrics", {}).get("J"),
            "verifier_status": candidate.get("metrics", {}).get("verifier_status"),
            "status": candidate.get("metrics", {}).get("status"),
        },
    }


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


def _select_generation_winner(evaluated_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in evaluated_candidates if candidate["metrics"]["status"] == "pass"]
    if passing:
        return max(passing, key=_candidate_rank)
    return max(evaluated_candidates, key=lambda candidate: (float(candidate["metrics"]["J"]), _objective_score(candidate)))


def _extend_frontier(frontier: list[dict[str, Any]], candidate: dict[str, Any], max_size: int = 8) -> list[dict[str, Any]]:
    if any(item.get("source_code") == candidate.get("source_code") for item in frontier):
        return frontier
    updated = frontier + [candidate]
    if len(updated) <= max_size:
        return updated
    best = max(updated, key=_candidate_rank)
    keep_ids = {best["candidate_id"]}
    kept: list[dict[str, Any]] = [best]
    ranked = sorted(updated, key=_candidate_rank, reverse=True)
    recent = list(reversed(updated))
    for pool in (ranked, recent):
        for item in pool:
            candidate_id = item["candidate_id"]
            if candidate_id in keep_ids:
                continue
            kept.append(item)
            keep_ids.add(candidate_id)
            if len(kept) >= max_size:
                break
        if len(kept) >= max_size:
            break
    return kept


def _failure_reason_against_parent(task: dict[str, Any], parent_candidate: dict[str, Any], winner: dict[str, Any], epsilon: float) -> str:
    verifier_status = winner["metrics"]["verifier_status"]
    if verifier_status == "error":
        return f"Candidate errored before verification completed: {winner['metrics'].get('error') or 'unknown error'}"
    if verifier_status == "fail":
        failed_tests = [result["name"] for result in winner["metrics"].get("test_results", []) if not result.get("passed")]
        if failed_tests:
            return "Candidate failed deterministic tests: " + ", ".join(failed_tests)
        return "Candidate failed deterministic correctness checks."
    winner_objective = _objective_value(winner)
    parent_objective = _objective_value(parent_candidate)
    direction = _objective_direction(task)
    comparison = "higher" if direction == "max" else "lower"
    return (
        f"Candidate passed verification but did not produce a {comparison} { _objective_label(task) } "
        f"than the selected parent ({winner_objective:.3f} vs {parent_objective:.3f}; epsilon {epsilon:.3f})."
    )


def _should_write_failure_memory(winner: dict[str, Any]) -> bool:
    return winner["metrics"]["verifier_status"] in {"fail", "error"}


def _select_branch_parents(
    frontier: list[dict[str, Any]],
    current_best: dict[str, Any],
    accepted_history: list[dict[str, Any]],
    branching_factor: int,
) -> list[dict[str, Any]]:
    if not frontier:
        return [current_best]

    limit = max(1, min(branching_factor, len(frontier)))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(candidate: dict[str, Any] | None) -> None:
        if candidate is None:
            return
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in selected_ids:
            return
        selected.append(candidate)
        selected_ids.add(candidate_id)

    add(current_best)

    stepping_stone = next(
        (
            candidate
            for candidate in reversed(accepted_history)
            if candidate["candidate_id"] != current_best["candidate_id"]
        ),
        None,
    )
    add(stepping_stone)

    ranked = [candidate for candidate in sorted(frontier, key=_candidate_rank, reverse=True) if candidate["candidate_id"] not in selected_ids]
    recent = [candidate for candidate in reversed(frontier) if candidate["candidate_id"] not in selected_ids]
    use_recent = True
    while len(selected) < limit and (ranked or recent):
        pool = recent if use_recent and recent else ranked
        if not pool:
            pool = recent if recent else ranked
        candidate = pool.pop(0)
        add(candidate)
        ranked = [item for item in ranked if item["candidate_id"] != candidate["candidate_id"]]
        recent = [item for item in recent if item["candidate_id"] != candidate["candidate_id"]]
        use_recent = not use_recent
    return selected


def _branch_message(branch_id: str, parent_candidate: dict[str, Any], retrieved_count: int) -> str:
    return f"{branch_id} parent={parent_candidate['agent']} retrieved_memories={retrieved_count}"


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
    accepted_history: list[dict[str, Any]] = []
    candidate_history: list[dict[str, Any]] = []
    llm_traces: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    epsilon = float(task.get("epsilon", 0.0))
    branching_factor = int(task.get("branching_factor", 1))
    objective_curve = [
        {
            "generation": 0,
            "objective": baseline["metrics"]["objective"],
            "objective_score": baseline["metrics"]["objective_score"],
            "candidate_objective": baseline["metrics"]["objective"],
            "candidate_objective_score": baseline["metrics"]["objective_score"],
            "J": baseline["metrics"]["J"],
            "candidate_J": baseline["metrics"]["J"],
            "candidate_agent": baseline["agent"],
            "candidate_label": baseline["label"],
            "accepted": True,
            "accepted_count": 1,
            "memory_delta": 0,
            "proposal_model": None,
            "run_mode": "llm-required",
        }
    ]
    initial_retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)

    for generation in range(1, int(task["generation_budget"]) + 1):
        parents = _select_branch_parents(frontier, current_best, accepted_history, branching_factor)
        generation_best_before = current_best
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_started",
            task_id=task["id"],
            generation=generation,
            parent_candidate=current_best["agent"],
            accepted_to_frontier=False,
            improved_global_best=False,
            memory_delta=0,
            message=f"Generation {generation} spawned {len(parents)} branches from frontier={len(frontier)}",
        )

        branch_inputs: list[dict[str, Any]] = []
        for branch_index, parent_candidate in enumerate(parents, start=1):
            retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)
            branch_id = f"g{generation}-b{branch_index}"
            branch_input = {
                "branch_id": branch_id,
                "branch_index": branch_index,
                "parent_candidate": parent_candidate,
                "retrieved_memories": retrieved,
            }
            branch_inputs.append(branch_input)
            _emit(
                progress_callback,
                pace_ms,
                phase="branch_started",
                task_id=task["id"],
                generation=generation,
                branch_id=branch_id,
                branch_index=branch_index,
                parent_candidate=parent_candidate["agent"],
                candidate=parent_candidate["agent"],
                accepted_to_frontier=False,
                improved_global_best=False,
                memory_delta=0,
                message=_branch_message(branch_id, parent_candidate, len(retrieved)),
            )

        if len(branch_inputs) == 1:
            candidate_specs, proposal_trace = propose_code_candidates(
                proposal_runtime,
                task=task,
                generation=generation,
                parent_candidate=branch_inputs[0]["parent_candidate"],
                current_best=generation_best_before,
                candidate_history=candidate_history,
                memories=branch_inputs[0]["retrieved_memories"],
            )
            branch_inputs[0]["candidate_specs"] = candidate_specs
            branch_inputs[0]["proposal_trace"] = proposal_trace
        else:
            with ThreadPoolExecutor(max_workers=len(branch_inputs)) as executor:
                futures = {
                    executor.submit(
                        propose_code_candidates,
                        proposal_runtime,
                        task=task,
                        generation=generation,
                        parent_candidate=branch_input["parent_candidate"],
                        current_best=generation_best_before,
                        candidate_history=candidate_history,
                        memories=branch_input["retrieved_memories"],
                    ): branch_input
                    for branch_input in branch_inputs
                }
                for future, branch_input in futures.items():
                    candidate_specs, proposal_trace = future.result()
                    branch_input["candidate_specs"] = candidate_specs
                    branch_input["proposal_trace"] = proposal_trace

        for branch_input in branch_inputs:
            proposal_trace = dict(branch_input["proposal_trace"])
            branch_id = branch_input["branch_id"]
            llm_traces.append(
                {
                    "task_id": task["id"],
                    "generation": generation,
                    "branch_id": branch_id,
                    "branch_index": branch_input["branch_index"],
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
                branch_id=branch_id,
                branch_index=branch_input["branch_index"],
                parent_candidate=branch_input["parent_candidate"]["agent"],
                candidate=proposal_trace["selected_model"],
                accepted_to_frontier=False,
                improved_global_best=False,
                memory_delta=0,
                message=f"{branch_id} generated {len(branch_input['candidate_specs'])} candidates with {proposal_trace['selected_model']}",
            )

        branch_results: list[dict[str, Any]] = []
        for branch_input in branch_inputs:
            parent_candidate = branch_input["parent_candidate"]
            branch_id = branch_input["branch_id"]
            branch_index = branch_input["branch_index"]
            retrieved = branch_input["retrieved_memories"]
            evaluated_candidates: list[dict[str, Any]] = []

            for index, spec in enumerate(branch_input["candidate_specs"], start=1):
                candidate_id = f"{task['id']}-g{generation}-b{branch_index}-c{index}"
                source_path, source_code = materialize_candidate(
                    task=task,
                    workspace_root=workspace_root / task["id"] / f"generation-{generation}" / branch_id,
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
                    "branch_id": branch_id,
                    "branch_index": branch_index,
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
                    branch_id=branch_id,
                    branch_index=branch_index,
                    parent_candidate=parent_candidate["agent"],
                    candidate=candidate["agent"],
                    accepted_to_frontier=False,
                    improved_global_best=False,
                    memory_delta=0,
                    message=(
                        f"{branch_id} {candidate['agent']} status={candidate['verifier_status']} "
                        f"objective={candidate['metrics']['objective']} score={candidate['metrics']['objective_score']} "
                        f"J={candidate['metrics']['J']}"
                    ),
                )

            evaluated_candidates.sort(key=lambda item: float(item["metrics"]["J"]), reverse=True)
            branch_winner = _select_generation_winner(evaluated_candidates)
            accepted_to_frontier = (
                branch_winner["metrics"]["status"] == "pass"
                and _objective_score(branch_winner) > _objective_score(parent_candidate) + epsilon
            )
            improved_global_best = accepted_to_frontier and (
                _objective_score(branch_winner) > _objective_score(generation_best_before) + epsilon
            )
            delta_j = round(branch_winner["metrics"]["J"] - parent_candidate["metrics"]["J"], 4)
            global_best_delta_j = round(branch_winner["metrics"]["J"] - generation_best_before["metrics"]["J"], 4)
            experience_outcome = "success" if accepted_to_frontier else "failure"
            rejection_reason = None if accepted_to_frontier else _failure_reason_against_parent(task, parent_candidate, branch_winner, epsilon)
            wrote_memory = False
            memory_delta = 0
            new_experience = None
            should_reflect = accepted_to_frontier or _should_write_failure_memory(branch_winner)
            if should_reflect:
                reflection, reflection_trace = reflect_strategy_experience(
                    proposal_runtime,
                    task=task,
                    generation=generation,
                    previous_best=parent_candidate,
                    winner=branch_winner,
                    delta_j=delta_j,
                    outcome=experience_outcome,
                    rejection_reason=rejection_reason,
                )
                llm_traces.append(
                    {
                        "task_id": task["id"],
                        "generation": generation,
                        "branch_id": branch_id,
                        "branch_index": branch_index,
                        "phase": "memory_reflection",
                        "experience_outcome": experience_outcome,
                        **reflection_trace,
                    }
                )
                new_experience = _build_experience(
                    task,
                    session_id,
                    generation,
                    branch_winner,
                    delta_j,
                    reflection,
                    outcome=experience_outcome,
                    rejection_reason=rejection_reason,
                )
                wrote_memory = store.append(new_experience)
                if wrote_memory:
                    memory_delta = 1 if experience_outcome == "success" else -1
                    memory_event = {
                        "task_id": task["id"],
                        "generation": generation,
                        "branch_id": branch_id,
                        "branch_index": branch_index,
                        "experience_id": new_experience["experience_id"],
                        "experience_outcome": experience_outcome,
                        "verifier_status": branch_winner["metrics"]["verifier_status"],
                        "delta_J": delta_j,
                        "memory_delta": memory_delta,
                        "prompt_fragment": new_experience["prompt_fragment"],
                    }
                    memory_events.append(memory_event)
                    _emit(
                        progress_callback,
                        pace_ms,
                        phase="memory_writeback",
                        task_id=task["id"],
                        generation=generation,
                        branch_id=branch_id,
                        branch_index=branch_index,
                        parent_candidate=parent_candidate["agent"],
                        candidate=branch_winner["agent"],
                        accepted_to_frontier=accepted_to_frontier,
                        improved_global_best=improved_global_best,
                        memory_delta=memory_delta,
                        message=f"{branch_id} wrote {new_experience['experience_id']} ({experience_outcome})",
                    )
            else:
                _emit(
                    progress_callback,
                    pace_ms,
                    phase="memory_skipped",
                    task_id=task["id"],
                    generation=generation,
                    branch_id=branch_id,
                    branch_index=branch_index,
                    parent_candidate=parent_candidate["agent"],
                    candidate=branch_winner["agent"],
                    accepted_to_frontier=accepted_to_frontier,
                    improved_global_best=improved_global_best,
                    memory_delta=0,
                    message=f"{branch_id} skipped write-back for a passing but non-improving candidate.",
                )

            branch_results.append(
                {
                    "branch_id": branch_id,
                    "branch_index": branch_index,
                    "parent_candidate": parent_candidate,
                    "retrieved_memories": retrieved,
                    "candidates": evaluated_candidates,
                    "winner": branch_winner,
                    "winner_accepted": accepted_to_frontier,
                    "winner_improved_global_best": improved_global_best,
                    "delta_J": delta_j,
                    "global_best_delta_J": global_best_delta_j,
                    "experience_outcome": experience_outcome,
                    "wrote_memory": wrote_memory,
                    "memory_delta": memory_delta,
                    "new_experience": new_experience,
                    "rejection_reason": rejection_reason,
                }
            )

        accepted_candidates = [branch["winner"] for branch in branch_results if branch["winner_accepted"]]
        for accepted_candidate in sorted(accepted_candidates, key=_candidate_rank, reverse=True):
            frontier = _extend_frontier(frontier, accepted_candidate)
            accepted_history.append(accepted_candidate)
        current_best = max([generation_best_before, *accepted_candidates], key=_candidate_rank)
        generation_winner = _select_generation_winner([branch["winner"] for branch in branch_results])
        winner_branch = next(branch for branch in branch_results if branch["winner"]["candidate_id"] == generation_winner["candidate_id"])
        positive_writebacks = sum(1 for branch in branch_results if branch["memory_delta"] > 0)
        negative_writebacks = sum(1 for branch in branch_results if branch["memory_delta"] < 0)
        memory_delta = positive_writebacks - negative_writebacks

        objective_curve.append(
            {
                "generation": generation,
                "objective": current_best["metrics"]["objective"],
                "objective_score": current_best["metrics"]["objective_score"],
                "candidate_objective": generation_winner["metrics"]["objective"],
                "candidate_objective_score": generation_winner["metrics"]["objective_score"],
                "J": current_best["metrics"]["J"],
                "candidate_J": generation_winner["metrics"]["J"],
                "candidate_agent": generation_winner["agent"],
                "candidate_label": generation_winner["label"],
                "accepted": winner_branch["winner_accepted"],
                "accepted_count": len(accepted_candidates),
                "improved_global_best": any(branch["winner_improved_global_best"] for branch in branch_results),
                "parent_candidate": winner_branch["parent_candidate"]["agent"],
                "active_model": proposal_runtime.active_model,
                "proposal_model": generation_winner["proposal_model"],
                "experience_outcome": winner_branch["experience_outcome"],
                "memory_delta": memory_delta,
                "run_mode": "llm-required",
            }
        )
        generations.append(
            {
                "generation": generation,
                "run_mode": "llm-required",
                "active_model": proposal_runtime.active_model,
                "parent_candidate": winner_branch["parent_candidate"],
                "parents": [_candidate_snapshot(parent) for parent in parents],
                "retrieved_memories": winner_branch["retrieved_memories"],
                "candidates": [candidate for branch in branch_results for candidate in branch["candidates"]],
                "winner": generation_winner,
                "winner_accepted": winner_branch["winner_accepted"],
                "winner_improved_global_best": any(branch["winner_improved_global_best"] for branch in branch_results),
                "best_after_generation": current_best,
                "delta_J": winner_branch["delta_J"],
                "global_best_delta_J": winner_branch["global_best_delta_J"],
                "experience_outcome": winner_branch["experience_outcome"],
                "wrote_memory": any(branch["wrote_memory"] for branch in branch_results),
                "new_experience": winner_branch["new_experience"],
                "frontier_size": len(frontier),
                "branches": branch_results,
                "accepted_count": len(accepted_candidates),
                "memory_delta": memory_delta,
                "positive_writebacks": positive_writebacks,
                "negative_writebacks": negative_writebacks,
            }
        )
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_finished",
            task_id=task["id"],
            generation=generation,
            branch_id=winner_branch["branch_id"],
            branch_index=winner_branch["branch_index"],
            parent_candidate=winner_branch["parent_candidate"]["agent"],
            candidate=current_best["agent"],
            accepted_to_frontier=winner_branch["winner_accepted"],
            improved_global_best=winner_branch["winner_improved_global_best"],
            memory_delta=memory_delta,
            message=(
                f"Best { _objective_label(task) } after generation {generation}: "
                f"{current_best['metrics']['objective']} (accepts={len(accepted_candidates)}, frontier={len(frontier)})"
            ),
        )

    run_delta_j = round(current_best["metrics"]["J"] - baseline["metrics"]["J"], 4)
    run_delta_objective = round(_objective_value(current_best) - _objective_value(baseline), 4)
    added_experiences = [
        branch["new_experience"]
        for generation in generations
        for branch in generation.get("branches", [])
        if branch.get("wrote_memory") and branch.get("new_experience")
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
        "j_spec": {
            "display_name": "Internal selection score J",
            "direction": "max",
            "summary_template": "J is the always-max internal selection score used to rank verified candidates across tasks.",
            "formula": "J = 1.20 * correctness + 0.95 * speed_score + 0.20 * memory_bonus + 0.15 * stability - 0.18 * complexity - 0.05 * (line_count / 10)",
            "delta_template": "delta_J compares the generation winner against the selected parent; run_delta_J compares the final winner against the baseline.",
        },
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "function_signature": task["function_signature"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "objective_spec": task["objective_spec"],
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "branching_factor": branching_factor,
            "source_type": task["source_type"],
        },
        "baseline": baseline,
        "initial_retrieved_memories": initial_retrieved,
        "generations": generations,
        "winner": current_best,
        "delta_J": run_delta_j,
        "run_delta_J": run_delta_j,
        "run_delta_objective": run_delta_objective,
        "objective_curve": objective_curve,
        "added_experiences": added_experiences,
        "positive_experiences_added": positive_experiences_added,
        "negative_experiences_added": negative_experiences_added,
        "memory_events": memory_events,
        "llm_traces": llm_traces,
        "proposal_engine": proposal_runtime.describe(),
        "selection_reason": (
            f"{current_best['label']} reached { _objective_label(task) }={current_best['metrics']['objective']} "
            f"with J={current_best['metrics']['J']} after {len(generations)} generations."
        ),
    }
