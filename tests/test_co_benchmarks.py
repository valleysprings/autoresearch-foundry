from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.codegen.co_benchmarks import (
    build_co_bench_manifest,
    _canonical_problem_name,
    _co_bench_data_dir,
    _co_bench_evaluation_dir,
    _controller_task_names,
    co_bench_problem_names,
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
                "def solve(**kwargs):\n"
                "    return {'schedule': {}}\n"
            )
            (tsp_dir / "config.py").write_text(
                "DESCRIPTION = 'Travelling salesman description.'\n\n"
                "def solve(**kwargs):\n"
                "    return {'tour': []}\n"
            )

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
            self.assertIn("Travelling salesman description.", item["context"])
            self.assertIn("def solve(**kwargs):", item["context"])


if __name__ == "__main__":
    unittest.main()
