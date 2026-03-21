from __future__ import annotations

from typing import Any


DEVICE_BACKENDS = {
    "mps": {"sdpa", "metal-sdpa"},
    "h200": {"sdpa", "flash-attn3"},
    "cpu": {"sdpa"},
}


def _supports(rule: str, candidate: dict[str, Any], task: dict[str, Any]) -> bool:
    proposal = candidate["proposal"]
    target_device = task["profile"]["target_device"]

    checks = {
        "deterministic_eval": proposal.get("deterministic_eval", False),
        "replay_memory": proposal.get("uses_memory", False),
        "mps_safe_attention": proposal.get("attention_backend") in DEVICE_BACKENDS["mps"],
        "results_logging": proposal.get("results_logging", False),
        "handoff_bundle": proposal.get("handoff_bundle", False),
        "selective_writeback": proposal.get("selective_writeback", False),
        "device_compatible": proposal.get("attention_backend") in DEVICE_BACKENDS.get(target_device, {"sdpa"}),
    }
    return bool(checks.get(rule, False))


def evaluate(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    profile = task["profile"]
    traits = candidate["traits"]
    proposal = candidate["proposal"]

    memory_budget = float(profile["memory_budget_gb"])
    runtime_budget = float(profile["runtime_budget_min"])
    required_rules = profile.get("must_include", [])

    compatibility = proposal.get("attention_backend") in DEVICE_BACKENDS.get(profile["target_device"], {"sdpa"})
    budget_pass = traits["memory_gb"] <= memory_budget and traits["runtime_min"] <= runtime_budget
    required_checks = {rule: _supports(rule, candidate, task) for rule in required_rules}
    must_include_score = (
        sum(1.0 for passed in required_checks.values() if passed) / len(required_checks)
        if required_checks
        else 1.0
    )

    replay_alignment = 1.0 if proposal.get("uses_memory") and candidate.get("supporting_memory_ids") else 0.0
    scale_readiness = 1.0 if proposal.get("handoff_bundle") else 0.25
    expected_gain = float(traits["expected_gain"])
    reproducibility = float(traits["reproducibility"])
    novelty = float(candidate.get("novelty", 0.0))
    complexity = float(traits["complexity"])
    steps = float(candidate.get("steps", 1.0))
    cost = ((traits["memory_gb"] / memory_budget) + (traits["runtime_min"] / runtime_budget)) / 2.0
    unsupported_penalty = 0.0 if compatibility else 1.0
    step_penalty = steps / 10.0

    success = 1.0 if budget_pass and compatibility and proposal.get("deterministic_eval") else 0.0
    score = (
        1.20 * success
        + 0.90 * must_include_score
        + 0.80 * expected_gain
        + 0.45 * reproducibility
        + 0.35 * replay_alignment
        + 0.25 * scale_readiness
        + 0.20 * novelty
        - 0.40 * cost
        - 0.25 * complexity
        - 0.10 * step_penalty
        - 0.60 * unsupported_penalty
    )

    return {
        "success": round(success, 2),
        "test_pass": round(must_include_score, 2),
        "budget_pass": round(1.0 if budget_pass else 0.0, 2),
        "compatibility": round(1.0 if compatibility else 0.0, 2),
        "replay_alignment": round(replay_alignment, 2),
        "reproducibility": round(reproducibility, 2),
        "scale_readiness": round(scale_readiness, 2),
        "expected_gain": round(expected_gain, 2),
        "novelty": round(novelty, 2),
        "cost": round(cost, 2),
        "complexity": round(complexity, 2),
        "steps": round(steps, 2),
        "memory_gb": round(float(traits["memory_gb"]), 2),
        "runtime_min": round(float(traits["runtime_min"]), 2),
        "required_checks": required_checks,
        "J": round(score, 4),
    }
