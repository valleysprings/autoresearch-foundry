from __future__ import annotations

import unittest

from app.codegen.catalog import load_codegen_tasks


class BenchmarkComparisonTest(unittest.TestCase):
    def test_active_main_comparison_lane_includes_dataset_and_legacy_verified_tasks(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertEqual(
            {task["id"] for task in comparable_tasks},
            {"olymmath", "sciq", "planbench-lite", "multihop-snapshot-small", "tbench-lite"},
        )
        dataset_tasks = [task for task in comparable_tasks if task.get("local_dataset_only")]
        self.assertEqual({task["id"] for task in dataset_tasks}, {"olymmath", "sciq"})
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            {"math_verified", "science_verified", "planning_verified", "multihop_qa_snapshot", "terminal_verified"},
        )


if __name__ == "__main__":
    unittest.main()
