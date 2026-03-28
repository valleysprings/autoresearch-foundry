from __future__ import annotations

from app.codegen.co_benchmarks import evaluate_co_bench_candidate


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    return evaluate_co_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
    )
