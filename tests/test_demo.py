from __future__ import annotations

import unittest

from app.demo_run import generate_demo_payload


class DemoPayloadTest(unittest.TestCase):
    def test_demo_payload_contains_three_runs_and_writebacks(self) -> None:
        payload = generate_demo_payload()
        self.assertEqual(payload["summary"]["num_tasks"], 3)
        self.assertGreaterEqual(payload["summary"]["write_backs"], 2)
        self.assertEqual(payload["runs"][0]["winner"]["agent"], "replay-synthesizer")
        self.assertEqual(payload["runs"][-1]["winner"]["agent"], "scale-bridge")

    def test_replay_memory_grows_over_time(self) -> None:
        payload = generate_demo_payload()
        runs = payload["runs"]
        self.assertLess(runs[0]["memory_before_count"], runs[0]["memory_after_count"])
        self.assertLessEqual(runs[0]["memory_after_count"], runs[-1]["memory_after_count"])
        self.assertTrue(runs[1]["retrieved_memories"])


if __name__ == "__main__":
    unittest.main()
