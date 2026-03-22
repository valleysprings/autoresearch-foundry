from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.codegen.catalog import load_codegen_tasks
from app.codegen.verifier import build_candidate_source, evaluate_materialized_candidate, materialize_candidate


class CodegenVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = next(task for task in load_codegen_tasks() if task["id"] == "contains-duplicates")

    def test_materializer_builds_callable_source(self) -> None:
        source = build_candidate_source(
            self.task,
            [],
            "seen = set()\nfor value in values:\n    if value in seen:\n        return True\n    seen.add(value)\nreturn False",
        )
        self.assertIn("def contains_duplicates(values):", source)
        self.assertIn("return False", source)

    def test_compile_error_marks_candidate_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path, source_code = materialize_candidate(
                task=self.task,
                workspace_root=Path(tmp_dir),
                candidate_id="broken",
                imports=[],
                function_body="return values[",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=source_path,
                source_code=source_code,
                imports=[],
                baseline_ms=1.0,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "error")
            self.assertLess(metrics["J"], 0.0)

    def test_test_failure_marks_candidate_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path, source_code = materialize_candidate(
                task=self.task,
                workspace_root=Path(tmp_dir),
                candidate_id="wrong",
                imports=[],
                function_body="return False",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=source_path,
                source_code=source_code,
                imports=[],
                baseline_ms=1.0,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "fail")
            self.assertEqual(metrics["correctness"], 0.0)

    def test_runtime_exception_is_captured_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path, source_code = materialize_candidate(
                task=self.task,
                workspace_root=Path(tmp_dir),
                candidate_id="raises",
                imports=[],
                function_body="raise RuntimeError('boom')",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=source_path,
                source_code=source_code,
                imports=[],
                baseline_ms=1.0,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "error")
            self.assertIn("boom", metrics["error"])

    def test_passing_candidate_benchmarks_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_path, baseline_source = materialize_candidate(
                task=self.task,
                workspace_root=Path(tmp_dir),
                candidate_id="baseline",
                imports=[],
                function_body=self.task["baseline_body"],
            )
            baseline_metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=baseline_path,
                source_code=baseline_source,
                imports=[],
                baseline_ms=None,
                memory_applied=False,
            )
            candidate_path, candidate_source = materialize_candidate(
                task=self.task,
                workspace_root=Path(tmp_dir),
                candidate_id="fast",
                imports=[],
                function_body="return len(values) != len(set(values))",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=candidate_path,
                source_code=candidate_source,
                imports=[],
                baseline_ms=baseline_metrics["benchmark_ms"],
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)
            self.assertGreater(metrics["J"], baseline_metrics["J"])

    def test_math_task_benchmark_kind_executes(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "count-primes-up-to")
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_path, baseline_source = materialize_candidate(
                task=task,
                workspace_root=Path(tmp_dir),
                candidate_id="baseline-math",
                imports=[],
                function_body=task["baseline_body"],
            )
            baseline_metrics = evaluate_materialized_candidate(
                task=task,
                source_path=baseline_path,
                source_code=baseline_source,
                imports=[],
                baseline_ms=None,
                memory_applied=False,
            )
            candidate_path, candidate_source = materialize_candidate(
                task=task,
                workspace_root=Path(tmp_dir),
                candidate_id="sieve",
                imports=[],
                function_body=(
                    "if limit < 2:\n"
                    "    return 0\n"
                    "sieve = [True] * (limit + 1)\n"
                    "sieve[0] = False\n"
                    "sieve[1] = False\n"
                    "candidate = 2\n"
                    "while candidate * candidate <= limit:\n"
                    "    if sieve[candidate]:\n"
                    "        step = candidate * candidate\n"
                    "        while step <= limit:\n"
                    "            sieve[step] = False\n"
                    "            step += candidate\n"
                    "    candidate += 1\n"
                    "return sum(1 for is_prime in sieve if is_prime)"
                ),
            )
            metrics = evaluate_materialized_candidate(
                task=task,
                source_path=candidate_path,
                source_code=candidate_source,
                imports=[],
                baseline_ms=baseline_metrics["benchmark_ms"],
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)


if __name__ == "__main__":
    unittest.main()
