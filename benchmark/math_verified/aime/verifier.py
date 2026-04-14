from __future__ import annotations

from app.bench.math_grading import evaluate_math_dataset_candidate


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    return evaluate_math_dataset_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        baseline_metrics=baseline_metrics,
        memory_applied=memory_applied,
    )
