from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CO_BENCH_ROOT = ROOT / "benchmark" / "or_verified" / "co-bench"
if str(CO_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(CO_BENCH_ROOT))

from co_bench_support import (
    build_co_bench_manifest,
    _canonical_problem_name,
    _co_bench_data_dir,
    _co_bench_evaluation_dir,
    _controller_task_names,
    co_bench_problem_names,
    evaluate_co_bench_candidate,
)


class CoBenchmarksTest(unittest.TestCase):
    def test_problem_aliases_match_dataset_folder_names(self) -> None:
        self.assertEqual(_canonical_problem_name("TSP"), "Travelling salesman problem")
        self.assertEqual(_canonical_problem_name("MIS"), "Maximal independent set")
        self.assertEqual(_canonical_problem_name("Assignment problem"), "Assignment problem")

    def test_default_paths_live_under_task_directory(self) -> None:
        task = {"task_dir": "/tmp/benchmark/or_verified/co-bench"}
        self.assertEqual(_co_bench_data_dir(task, {}), Path("/tmp/benchmark/or_verified/co-bench/data"))
        self.assertEqual(
            _co_bench_evaluation_dir(task, {}),
            Path("/tmp/benchmark/or_verified/co-bench/evaluation"),
        )

    def test_controller_task_names_are_normalized_and_deduplicated(self) -> None:
        class FakeController:
            TASK_LIST = ["Aircraft landing", "TSP", "MIS", "TSP"]

        self.assertEqual(
            _controller_task_names(FakeController),
            ["Aircraft landing", "Travelling salesman problem", "Maximal independent set"],
        )

    def test_manifest_generation_uses_checked_in_problem_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir) / "co-bench"
            evaluation_dir = task_dir / "evaluation"
            data_dir = task_dir / "data"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            (evaluation_dir / "controller.py").write_text(
                "TASK_LIST = ['Aircraft landing', 'TSP']\n"
            )
            aircraft_dir = data_dir / "Aircraft landing"
            tsp_dir = data_dir / "Travelling salesman problem"
            aircraft_dir.mkdir(parents=True, exist_ok=True)
            tsp_dir.mkdir(parents=True, exist_ok=True)
            (aircraft_dir / "config.py").write_text(
                "DESCRIPTION = 'Aircraft landing description.'\n\n"
                "def load_data(path):\n"
                "    del path\n"
                "    return [{'id': 1}]\n\n"
                "def solve(**kwargs):\n"
                "    return {'schedule': {}}\n"
            )
            (aircraft_dir / "case.txt").write_text("case")
            (tsp_dir / "config.py").write_text(
                "DESCRIPTION = 'Travelling salesman description.'\n\n"
                "def load_data(path):\n"
                "    del path\n"
                "    return [{'id': 1}, {'id': 2}]\n\n"
                "def solve(**kwargs):\n"
                "    return {'tour': []}\n"
            )
            (tsp_dir / "case.txt").write_text("case")

            self.assertEqual(
                co_bench_problem_names(task_dir),
                ["Aircraft landing", "Travelling salesman problem"],
            )
            manifest = build_co_bench_manifest(
                task_dir=task_dir,
                data_dir=data_dir,
                problem_names=["TSP"],
            )

            self.assertEqual(manifest["dataset_size"], 2)
            self.assertEqual(manifest["prepared_count"], 1)
            item = manifest["items"][0]
            self.assertEqual(item["name"], "Travelling salesman problem")
            self.assertEqual(item["metadata"]["problem_name"], "Travelling salesman problem")
            self.assertEqual(item["metadata"]["case_count"], 1)
            self.assertEqual(item["metadata"]["instance_count"], 2)
            self.assertEqual(item["metadata"]["runtime_split_tags"], ["problem:travelling-salesman-problem"])
            self.assertIn("Travelling salesman description.", item["context"])
            self.assertIn("def solve(**kwargs):", item["context"])

    def test_candidate_source_is_scored_directly_without_nested_codegen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir) / "co-bench"
            evaluation_dir = task_dir / "evaluation"
            data_dir = task_dir / "data" / "Assignment problem"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            (evaluation_dir / "__init__.py").write_text("")
            (evaluation_dir / "controller.py").write_text(
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class Data:\n"
                "    problem_description: str\n\n"
                "def get_data(task, src_dir='data'):\n"
                "    del task, src_dir\n"
                "    return Data(problem_description='Assignment problem description.')\n"
            )
            (data_dir / "config.py").write_text(
                "DESCRIPTION = 'Assignment problem description.'\n\n"
                "def solve(**kwargs):\n"
                "    return {'assignment': []}\n"
            )
            candidate_path = task_dir / "editable.py"
            source_code = "def solve(**kwargs):\n    return {'assignment': []}\n"
            candidate_path.write_text(source_code)
            task = {
                "id": "co-bench",
                "task_dir": str(task_dir),
                "question_item": {
                    "item_id": "assignment-problem",
                    "name": "Assignment problem",
                    "metadata": {"problem_name": "Assignment problem"},
                },
            }

            def fake_evaluate(data, python_code, *, timeout_s):  # noqa: ANN001
                self.assertEqual(data.problem_description, "Assignment problem description.")
                self.assertEqual(python_code, source_code)
                self.assertEqual(timeout_s, 10)
                return {
                    "score": 0.5,
                    "dev_score": 0.5,
                    "test_score": 0.5,
                    "feedback": "ok",
                    "dev_feedback": "ok",
                    "test_feedback": "ok",
                    "results": {},
                }

            with patch("co_bench_support._evaluate_with_official_scoring", side_effect=fake_evaluate):
                metrics = evaluate_co_bench_candidate(
                    task=task,
                    candidate_path=candidate_path,
                    source_code=source_code,
                )

            self.assertEqual(metrics["objective"], 0.5)
            self.assertEqual(metrics["test_results"][0]["actual"], 0.5)
            self.assertTrue(metrics["test_results"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
