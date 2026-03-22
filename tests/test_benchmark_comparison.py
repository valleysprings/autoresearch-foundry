from __future__ import annotations

import unittest

from app.codegen.catalog import load_codegen_tasks


class BenchmarkComparisonTest(unittest.TestCase):
    def test_active_main_comparison_lane_uses_dataset_research_tasks(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        self.assertEqual({task["id"] for task in comparable_tasks}, {"olymmath", "sciq"})
        self.assertTrue(all(task.get("local_dataset_only") for task in comparable_tasks))
        self.assertEqual({task["track"] for task in comparable_tasks}, {"math_verified", "science_verified"})


if __name__ == "__main__":
    unittest.main()
