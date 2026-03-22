from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.codegen.catalog import load_codegen_tasks
from app.codegen.dataset_support import build_micro_task, load_question_manifest
from app.codegen.verifier import evaluate_materialized_candidate, materialize_candidate


class CodegenVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        tasks = load_codegen_tasks()
        self.task = next(task for task in tasks if task["id"] == "contains-duplicates")

    def _materialize(self, task: dict, file_body: str) -> tuple[Path, str, str]:
        temp_dir = tempfile.TemporaryDirectory()
        workspace = Path(temp_dir.name)
        source_path, source_code = materialize_candidate(
            task=task,
            workspace_root=workspace,
            candidate_id="candidate",
            file_body=file_body,
        )
        self.addCleanup(temp_dir.cleanup)
        return workspace, source_path, source_code

    def test_materializer_writes_full_editable_file(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return len(values) != len(set(values))\n",
        )
        self.assertEqual(source_path.name, "editable.py")
        self.assertIn("def contains_duplicates(values):", source_code)
        self.assertTrue(source_code.endswith("\n"))

    def test_compile_error_marks_candidate_as_error(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return values[\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "error")
        self.assertLess(metrics["J"], 0.0)

    def test_test_failure_marks_candidate_as_fail(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return False\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "fail")
        self.assertEqual(metrics["correctness"], 0.0)

    def test_runtime_exception_is_captured_as_error(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    raise RuntimeError('boom')\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "error")
        self.assertIn("boom", metrics["error"])

    def test_passing_candidate_benchmarks_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            baseline_source = Path(self.task["editable_path"]).read_text()
            baseline_path, baseline_code = materialize_candidate(
                task=self.task,
                workspace_root=workspace,
                candidate_id="baseline",
                file_body=baseline_source,
            )
            baseline_metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=baseline_path,
                source_code=baseline_code,
                baseline_metrics=None,
                memory_applied=False,
            )
            candidate_path, candidate_code = materialize_candidate(
                task=self.task,
                workspace_root=workspace,
                candidate_id="fast",
                file_body="def contains_duplicates(values):\n    return len(values) != len(set(values))\n",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=candidate_path,
                source_code=candidate_code,
                baseline_metrics=baseline_metrics,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)
            self.assertGreater(metrics["J"], baseline_metrics["J"])

    def test_math_experiment_candidate_improves_benchmark(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "count-primes-up-to")
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            baseline_source = Path(task["editable_path"]).read_text()
            baseline_path, baseline_code = materialize_candidate(
                task=task,
                workspace_root=workspace,
                candidate_id="baseline-math",
                file_body=baseline_source,
            )
            baseline_metrics = evaluate_materialized_candidate(
                task=task,
                source_path=baseline_path,
                source_code=baseline_code,
                baseline_metrics=None,
                memory_applied=False,
            )
            candidate_path, candidate_code = materialize_candidate(
                task=task,
                workspace_root=workspace,
                candidate_id="sieve",
                file_body=(
                    "def count_primes_up_to(limit):\n"
                    "    if limit < 2:\n"
                    "        return 0\n"
                    "    sieve = [True] * (limit + 1)\n"
                    "    sieve[0] = False\n"
                    "    sieve[1] = False\n"
                    "    candidate = 2\n"
                    "    while candidate * candidate <= limit:\n"
                    "        if sieve[candidate]:\n"
                    "            step = candidate * candidate\n"
                    "            while step <= limit:\n"
                    "                sieve[step] = False\n"
                    "                step += candidate\n"
                    "        candidate += 1\n"
                    "    return sum(1 for is_prime in sieve if is_prime)\n"
                ),
            )
            metrics = evaluate_materialized_candidate(
                task=task,
                source_path=candidate_path,
                source_code=candidate_code,
                baseline_metrics=baseline_metrics,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)

    def test_legacy_comparable_tracks_are_not_in_active_research_lane(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        comparable_tasks = [task for task in comparable_tasks if not task.get("local_dataset_only")]
        self.assertEqual(comparable_tasks, [])

    def test_dataset_question_microtasks_generate_item_level_records(self) -> None:
        dataset_tasks = [task for task in load_codegen_tasks(included_in_main_comparison=True) if task.get("local_dataset_only")]
        self.assertEqual({task["id"] for task in dataset_tasks}, {"olymmath", "sciq"})
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            for task in dataset_tasks:
                items = load_question_manifest(task)
                micro_task = build_micro_task(task, items[0])
                source = Path(task["editable_path"]).read_text()
                candidate_path, candidate_code = materialize_candidate(
                    task=micro_task,
                    workspace_root=workspace / task["id"],
                    candidate_id="baseline",
                    file_body=source,
                )
                metrics = evaluate_materialized_candidate(
                    task=micro_task,
                    source_path=candidate_path,
                    source_code=candidate_code,
                    baseline_metrics=None,
                    memory_applied=False,
                )
                self.assertEqual(metrics["total_tests"], 1)
                self.assertEqual(len(metrics["test_results"]), 1)
                self.assertEqual(metrics["test_results"][0]["name"], items[0]["name"])


if __name__ == "__main__":
    unittest.main()
