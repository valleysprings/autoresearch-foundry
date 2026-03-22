from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen import catalog
from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks


class CodegenCatalogTest(unittest.TestCase):
    def test_small_experiments_and_comparable_tracks_are_classified(self) -> None:
        tasks = load_codegen_tasks()
        experiment_tasks = [task for task in tasks if task["benchmark_tier"] == "experiment"]
        comparable_tasks = [task for task in tasks if task["benchmark_tier"] == "comparable"]

        self.assertTrue(experiment_tasks)
        self.assertTrue(comparable_tasks)
        self.assertTrue(all(task["track"] == "small_experiments" for task in experiment_tasks))
        self.assertTrue(all(not task["included_in_main_comparison"] for task in experiment_tasks))
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            {"math_verified", "planning_verified", "science_verified", "multihop_qa_snapshot", "terminal_verified"},
        )
        self.assertTrue(all(task["included_in_main_comparison"] for task in comparable_tasks))
        self.assertEqual([task["id"] for task in comparable_tasks[:4]], ["olymmath", "math-500", "aime", "amc"])

    def test_main_comparison_filter_returns_only_comparable_tasks(self) -> None:
        main_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertTrue(main_tasks)
        self.assertTrue(all(task["benchmark_tier"] == "comparable" for task in main_tasks))
        self.assertNotIn("small_experiments", {task["track"] for task in main_tasks})

    def test_task_summaries_include_benchmark_metadata(self) -> None:
        summaries = list_codegen_task_summaries()
        contains_duplicates = next(task for task in summaries if task["id"] == "contains-duplicates")
        olymmath = next(task for task in summaries if task["id"] == "olymmath")
        math_500 = next(task for task in summaries if task["id"] == "math-500")
        amc = next(task for task in summaries if task["id"] == "amc")
        planbench = next(task for task in summaries if task["id"] == "planbench")
        sciq = next(task for task in summaries if task["id"] == "sciq")
        qasc = next(task for task in summaries if task["id"] == "qasc")
        scienceqa = next(task for task in summaries if task["id"] == "scienceqa")
        planbench_lite = next(task for task in summaries if task["id"] == "planbench-lite")

        self.assertEqual(contains_duplicates["benchmark_tier"], "experiment")
        self.assertEqual(contains_duplicates["track"], "small_experiments")
        self.assertEqual(contains_duplicates["answer_metric"], "speedup_vs_baseline")
        self.assertFalse(contains_duplicates["included_in_main_comparison"])

        self.assertTrue(olymmath["local_dataset_only"])
        self.assertEqual(olymmath["dataset_size"], 100)
        self.assertEqual(olymmath["split"], "en-hard:test")
        self.assertEqual(math_500["track"], "math_verified")
        self.assertEqual(math_500["split"], "test")
        self.assertEqual(amc["dataset_size"], 4)
        self.assertTrue(planbench["local_dataset_only"])
        self.assertEqual(planbench["dataset_size"], 2270)
        self.assertEqual(planbench["track"], "planning_verified")
        self.assertTrue(planbench["included_in_main_comparison"])
        self.assertEqual(planbench["split"], "task_1_plan_generation:train")
        self.assertEqual(sciq["dataset_size"], 1000)
        self.assertEqual(sciq["track"], "science_verified")
        self.assertEqual(sciq["split"], "validation")
        self.assertEqual(qasc["dataset_size"], 926)
        self.assertEqual(qasc["split"], "validation")
        self.assertEqual(scienceqa["dataset_size"], 768)
        self.assertEqual(scienceqa["split"], "validation:natural-science:text-only:biology-chemistry-physics")
        self.assertEqual(planbench_lite["dataset_size"], 4)
        self.assertEqual(planbench_lite["track"], "small_experiments")
        self.assertFalse(planbench_lite["included_in_main_comparison"])

    def test_missing_local_benchmark_assets_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "missing-task", "path": "missing-task", "enabled": True},
                        ]
                    }
                )
            )
            with (
                patch.object(catalog, "BENCHMARK_ROOT", root),
                patch.object(catalog, "REGISTRY_PATH", registry_path),
            ):
                self.assertEqual(load_codegen_tasks(), [])
                self.assertEqual(list_codegen_task_summaries(), [])


if __name__ == "__main__":
    unittest.main()
