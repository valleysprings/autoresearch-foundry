from __future__ import annotations

from app.codegen.verifier import evaluate_python_function_candidate


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    return evaluate_python_function_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        baseline_metrics=baseline_metrics,
        memory_applied=memory_applied,
    )
