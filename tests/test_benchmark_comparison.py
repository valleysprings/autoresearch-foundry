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
                "aime-2024",
                "aime-2025",
                "aime-2026",
                "planbench",
                "arc-challenge",
                "longbench-v2",
                "sciq",
                "qasc",
                "scienceqa",
                "openbookqa",
                "livecodebench",
                "tbench-lite",
            },
        )
        dataset_tasks = [task for task in comparable_tasks if task.get("local_dataset_only")]
        self.assertEqual(
            {task["id"] for task in dataset_tasks},
            {
                "olymmath",
                "math-500",
                "aime-2024",
                "aime-2025",
                "aime-2026",
                "planbench",
                "arc-challenge",
                "longbench-v2",
                "sciq",
                "qasc",
                "scienceqa",
                "openbookqa",
                "livecodebench",
            },
        )
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            {"math_verified", "reasoning_verified", "longcontext_verified", "science_verified", "terminal_verified", "coding_verified"},
        )
        self.assertEqual([task["id"] for task in comparable_tasks[:5]], ["olymmath", "math-500", "aime-2024", "aime-2025", "aime-2026"])


if __name__ == "__main__":
    unittest.main()
