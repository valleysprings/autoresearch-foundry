from __future__ import annotations

from typing import Any

from app.evaluator import evaluate


def planner(task: dict[str, Any], memories: list[dict[str, Any]]) -> dict[str, Any]:
    reusable_rules = sorted(
        {rule for item in memories for rule in item.get("reusable_rules", [])}
    )
    return {
        "task_id": task["id"],
        "objective": task["description"],
        "selection_policy": "maximize J(candidate; x) under deterministic evaluation and budget constraints",
        "priority_checks": [
            f"fit {task['profile']['target_device']} budget",
            "keep deterministic evaluator as the source of truth",
            "reuse retrieved experience when it improves reliability",
            "only write memory back when delta_J clears epsilon",
        ],
        "active_rules": reusable_rules,
        "lanes": _candidate_lane_names(task["id"]),
        "memory_count": len(memories),
    }


def _candidate_lane_names(task_id: str) -> list[str]:
    if task_id == "h200-handoff":
        return ["replay-synthesizer", "scale-bridge", "evolution-scout"]
    return ["local-optimizer", "replay-synthesizer", "evolution-scout"]


def _baseline_candidate(task: dict[str, Any]) -> dict[str, Any]:
    baseline = task["baseline"]
    return {
        "agent": "baseline",
        "label": "Current baseline",
        "strategy": baseline["strategy"],
        "steps": baseline["steps"],
        "novelty": baseline["novelty"],
        "supporting_memory_ids": [],
        "proposal": {
            "attention_backend": baseline["attention_backend"],
            "deterministic_eval": baseline["deterministic_eval"],
            "uses_memory": baseline["uses_memory"],
            "results_logging": baseline["results_logging"],
            "selective_writeback": baseline["selective_writeback"],
            "handoff_bundle": baseline["handoff_bundle"],
            "program_patch": baseline["program_patch"],
            "artifacts": baseline["artifacts"],
        },
        "traits": {
            "expected_gain": baseline["expected_gain"],
            "memory_gb": baseline["memory_gb"],
            "runtime_min": baseline["runtime_min"],
            "reproducibility": baseline["reproducibility"],
            "complexity": baseline["complexity"],
        },
    }


def _candidate(
    agent: str,
    label: str,
    strategy: str,
    steps: int,
    novelty: float,
    supporting_memory_ids: list[str],
    proposal: dict[str, Any],
    traits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "agent": agent,
        "label": label,
        "strategy": strategy,
        "steps": steps,
        "novelty": novelty,
        "supporting_memory_ids": supporting_memory_ids,
        "proposal": proposal,
        "traits": traits,
    }


def solver_candidates(task: dict[str, Any], memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory_ids = [item["experience_id"] for item in memories]

    if task["id"] == "local-mps-bootstrap":
        return [
            _candidate(
                "local-optimizer",
                "MPS-safe trim",
                "Replace unsupported kernels, shrink the search space, and keep the Mac loop simple enough to run every night.",
                4,
                0.24,
                ["exp-deterministic-loop"] if "exp-deterministic-loop" in memory_ids else [],
                {
                    "attention_backend": "sdpa",
                    "deterministic_eval": True,
                    "uses_memory": False,
                    "results_logging": True,
                    "selective_writeback": True,
                    "handoff_bundle": False,
                    "program_patch": [
                        "swap FlashAttention-3 for SDPA on MPS",
                        "cap sequence length at 512 for the local run",
                        "write results.tsv entries on every experiment",
                    ],
                    "artifacts": ["results.tsv", "run.log", "program.md diff"],
                },
                {
                    "expected_gain": 0.52,
                    "memory_gb": 16.6,
                    "runtime_min": 4.9,
                    "reproducibility": 0.79,
                    "complexity": 0.34,
                },
            ),
            _candidate(
                "replay-synthesizer",
                "Replay-guided local loop",
                "Fuse the fixed-budget logging pattern with an MPS-safe kernel plan so the next experiment starts with validated priors instead of guessing.",
                3,
                0.41,
                ["exp-deterministic-loop"] if "exp-deterministic-loop" in memory_ids else [],
                {
                    "attention_backend": "sdpa",
                    "deterministic_eval": True,
                    "uses_memory": True,
                    "results_logging": True,
                    "selective_writeback": True,
                    "handoff_bundle": False,
                    "program_patch": [
                        "reuse the fixed five-minute evaluation harness",
                        "retrieve prior experience before mutating the program",
                        "export delta_J so memory growth stays selective",
                    ],
                    "artifacts": ["results.tsv", "run.log", "replay_summary.json"],
                },
                {
                    "expected_gain": 0.63,
                    "memory_gb": 15.4,
                    "runtime_min": 4.3,
                    "reproducibility": 0.89,
                    "complexity": 0.39,
                },
            ),
            _candidate(
                "evolution-scout",
                "Aggressive mutation sweep",
                "Try a wider search frontier and more ambitious kernels immediately, accepting the risk that the local device will reject them.",
                6,
                0.78,
                [],
                {
                    "attention_backend": "flash-attn3",
                    "deterministic_eval": False,
                    "uses_memory": False,
                    "results_logging": True,
                    "selective_writeback": False,
                    "handoff_bundle": False,
                    "program_patch": [
                        "jump straight to GPU-first kernels",
                        "raise batch size without adapting for Metal memory",
                        "skip replay conditioning to maximize novelty",
                    ],
                    "artifacts": ["run.log"],
                },
                {
                    "expected_gain": 0.76,
                    "memory_gb": 22.0,
                    "runtime_min": 5.9,
                    "reproducibility": 0.52,
                    "complexity": 0.76,
                },
            ),
        ]

    if task["id"] == "local-replay-tighten":
        replay_support = [memory_id for memory_id in memory_ids if memory_id in {"exp-deterministic-loop", "exp-local-mps-bootstrap"}]
        return [
            _candidate(
                "local-optimizer",
                "Lean second-pass tune",
                "Apply the local-safe defaults but avoid the extra replay machinery, aiming for a simpler second iteration.",
                3,
                0.20,
                ["exp-local-mps-bootstrap"] if "exp-local-mps-bootstrap" in memory_ids else [],
                {
                    "attention_backend": "sdpa",
                    "deterministic_eval": True,
                    "uses_memory": False,
                    "results_logging": True,
                    "selective_writeback": True,
                    "handoff_bundle": False,
                    "program_patch": [
                        "keep the MPS-safe attention backend",
                        "lower the search depth for the second run",
                        "reuse the results logging schema only",
                    ],
                    "artifacts": ["results.tsv", "run.log"],
                },
                {
                    "expected_gain": 0.58,
                    "memory_gb": 14.2,
                    "runtime_min": 4.1,
                    "reproducibility": 0.82,
                    "complexity": 0.31,
                },
            ),
            _candidate(
                "replay-synthesizer",
                "Experience replay flywheel",
                "Start from the first winning Mac memory, reduce mutation churn, and keep only changes that improve the deterministic scoreboard.",
                2,
                0.33,
                replay_support,
                {
                    "attention_backend": "sdpa",
                    "deterministic_eval": True,
                    "uses_memory": True,
                    "results_logging": True,
                    "selective_writeback": True,
                    "handoff_bundle": False,
                    "program_patch": [
                        "retrieve the new Mac-specific experience before proposing mutations",
                        "bias the planner toward low-step candidates on MPS",
                        "write back only when delta_J beats the baseline by a real margin",
                    ],
                    "artifacts": ["results.tsv", "run.log", "replay_summary.json", "memory_diff.json"],
                },
                {
                    "expected_gain": 0.71,
                    "memory_gb": 13.1,
                    "runtime_min": 3.9,
                    "reproducibility": 0.93,
                    "complexity": 0.37,
                },
            ),
            _candidate(
                "evolution-scout",
                "Novelty-heavy fork",
                "Push more aggressive mutations and a larger search frontier, even if it threatens the local budget and makes review harder.",
                5,
                0.82,
                replay_support,
                {
                    "attention_backend": "sdpa",
                    "deterministic_eval": True,
                    "uses_memory": True,
                    "results_logging": True,
                    "selective_writeback": True,
                    "handoff_bundle": False,
                    "program_patch": [
                        "increase mutation variety for the second run",
                        "explore higher-memory kernels without widening the budget",
                        "keep replay on, but accept higher complexity",
                    ],
                    "artifacts": ["results.tsv", "run.log", "candidate_pool.json"],
                },
                {
                    "expected_gain": 0.79,
                    "memory_gb": 16.8,
                    "runtime_min": 4.5,
                    "reproducibility": 0.63,
                    "complexity": 0.71,
                },
            ),
        ]

    replay_support = [
        memory_id
        for memory_id in memory_ids
        if memory_id in {"exp-deterministic-loop", "exp-local-mps-bootstrap", "exp-local-replay-tighten"}
    ]
    return [
        _candidate(
            "replay-synthesizer",
            "Conservative cluster bridge",
            "Carry the local replay policy forward, but stay close to the Mac setup and avoid changing the runtime envelope too much.",
            4,
            0.29,
            replay_support,
            {
                "attention_backend": "sdpa",
                "deterministic_eval": True,
                "uses_memory": True,
                "results_logging": True,
                "selective_writeback": True,
                "handoff_bundle": False,
                "program_patch": [
                    "reuse the local replay recipe on the cluster",
                    "keep the evaluator and results logging fixed",
                    "avoid packaging a full handoff bundle yet",
                ],
                "artifacts": ["results.tsv", "run.log", "replay_summary.json"],
            },
            {
                "expected_gain": 0.66,
                "memory_gb": 54.0,
                "runtime_min": 4.8,
                "reproducibility": 0.91,
                "complexity": 0.35,
            },
        ),
        _candidate(
            "scale-bridge",
            "Local-to-H200 handoff",
            "Package the winning local pattern into a cluster-ready experiment bundle, swap in H200-capable kernels, and preserve the same deterministic keep/discard policy.",
            4,
            0.51,
            replay_support,
            {
                "attention_backend": "flash-attn3",
                "deterministic_eval": True,
                "uses_memory": True,
                "results_logging": True,
                "selective_writeback": True,
                "handoff_bundle": True,
                "program_patch": [
                    "export the validated local strategy as a cluster handoff bundle",
                    "switch to FlashAttention-3 only on the H200 lane",
                    "keep replay memory and results.tsv compatible across devices",
                ],
                "artifacts": ["results.tsv", "run.log", "handoff_bundle.tar", "cluster_plan.yaml"],
            },
            {
                "expected_gain": 0.83,
                "memory_gb": 76.0,
                "runtime_min": 4.7,
                "reproducibility": 0.90,
                "complexity": 0.48,
            },
        ),
        _candidate(
            "evolution-scout",
            "Max-novelty cluster leap",
            "Aim for the biggest jump on H200 by increasing mutation breadth and heavier kernels immediately, even if the experiment becomes harder to trust.",
            6,
            0.88,
            [],
            {
                "attention_backend": "flash-attn3",
                "deterministic_eval": False,
                "uses_memory": False,
                "results_logging": False,
                "selective_writeback": False,
                "handoff_bundle": True,
                "program_patch": [
                    "expand the mutation frontier aggressively",
                    "skip replay conditioning and deterministic gating",
                    "optimize only for upside on the H200 lane",
                ],
                "artifacts": ["cluster_plan.yaml"],
            },
            {
                "expected_gain": 0.92,
                "memory_gb": 82.0,
                "runtime_min": 5.3,
                "reproducibility": 0.74,
                "complexity": 0.83,
            },
        ),
    ]


def _build_experience(
    task: dict[str, Any],
    winner: dict[str, Any],
    baseline: dict[str, Any],
    delta_j: float,
) -> dict[str, Any]:
    reusable_rules = [
        rule
        for rule, passed in winner["metrics"]["required_checks"].items()
        if passed
    ]
    return {
        "experience_id": f"exp-{task['id']}",
        "source_task": task["id"],
        "target_device": task["profile"]["target_device"],
        "task_signature": task["task_signature"],
        "failure_pattern": baseline["strategy"],
        "successful_strategy": winner["strategy"],
        "tool_trace_summary": f"planner -> {winner['agent']} -> deterministic evaluator -> selective write-back",
        "delta_J": delta_j,
        "reusable_rules": reusable_rules,
        "supporting_memory_ids": winner.get("supporting_memory_ids", []),
    }


def run_task(
    task: dict[str, Any],
    memories: list[dict[str, Any]],
    epsilon: float = 0.35,
) -> dict[str, Any]:
    plan = planner(task, memories)
    baseline = _baseline_candidate(task)
    baseline["metrics"] = evaluate(baseline, task)

    scored = []
    for candidate in solver_candidates(task, memories):
        enriched = dict(candidate)
        enriched["metrics"] = evaluate(candidate, task)
        scored.append(enriched)

    scored.sort(key=lambda item: item["metrics"]["J"], reverse=True)
    winner = scored[0]
    delta_j = round(winner["metrics"]["J"] - baseline["metrics"]["J"], 4)
    should_write = winner["metrics"]["success"] == 1.0 and delta_j > epsilon
    new_experience = _build_experience(task, winner, baseline, delta_j) if should_write else None

    return {
        "task": task,
        "plan": plan,
        "retrieved_memories": memories,
        "baseline": baseline,
        "candidates": scored,
        "winner": winner,
        "delta_J": delta_j,
        "should_write_memory": should_write,
        "selection_reason": (
            f"{winner['agent']} wins because it achieves the highest J while satisfying "
            f"{sum(winner['metrics']['required_checks'].values())}/{len(winner['metrics']['required_checks'])} "
            "required checks."
        ),
        "new_experience": new_experience,
    }
