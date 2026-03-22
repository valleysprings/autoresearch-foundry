from __future__ import annotations

import unittest

from app.codegen.catalog import load_codegen_tasks


class BenchmarkComparisonTest(unittest.TestCase):
    def test_active_benchmark_tasks_include_math_first_dataset_and_legacy_verified_tasks(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertEqual(
            {task["id"] for task in comparable_tasks},
            {
                "olymmath",
                "math-500",
                "aime",
                "amc",
                "planbench",
                "sciq",
                "qasc",
                "scienceqa",
                "multihop-snapshot-small",
                "tbench-lite",
            },
        )
        dataset_tasks = [task for task in comparable_tasks if task.get("local_dataset_only")]
        self.assertEqual(
            {task["id"] for task in dataset_tasks},
            {"olymmath", "math-500", "aime", "amc", "planbench", "sciq", "qasc", "scienceqa"},
        )
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            {"math_verified", "planning_verified", "science_verified", "multihop_qa_snapshot", "terminal_verified"},
        )
        self.assertEqual([task["id"] for task in comparable_tasks[:4]], ["olymmath", "math-500", "aime", "amc"])


if __name__ == "__main__":
    unittest.main()
