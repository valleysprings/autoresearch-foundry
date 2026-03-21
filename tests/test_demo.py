from __future__ import annotations

import unittest

from app.demo_run import generate_demo_payload


class DemoPayloadTest(unittest.TestCase):
    def test_single_task_payload_runs_task1(self) -> None:
        payload = generate_demo_payload(task_id="contains-duplicates")
        self.assertEqual(payload["summary"]["num_tasks"], 1)
        self.assertEqual(payload["runs"][0]["task"]["id"], "contains-duplicates")
        self.assertGreater(payload["runs"][0]["winner"]["metrics"]["speedup_vs_baseline"], 1.0)
        self.assertTrue(payload["runs"][0]["should_write_memory"])

    def test_sequence_replays_memory(self) -> None:
        payload = generate_demo_payload()
        runs = payload["runs"]
        self.assertLess(runs[0]["memory_before_count"], runs[0]["memory_after_count"])
        self.assertLessEqual(runs[0]["memory_after_count"], runs[-1]["memory_after_count"])
        self.assertTrue(runs[1]["retrieved_memories"])


if __name__ == "__main__":
    unittest.main()
