from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
TASK_DIRS = {
    "planbench-t1": ROOT / "benchmark" / "reasoning_verified" / "planbench-t1",
    "planbench-t2": ROOT / "benchmark" / "reasoning_verified" / "planbench-t2",
    "planbench-t3": ROOT / "benchmark" / "reasoning_verified" / "planbench-t3",
}


def _load_prepare_module(task_id: str):
    script_path = TASK_DIRS[task_id] / "prepare.py"
    spec = importlib.util.spec_from_file_location(f"{task_id.replace('-', '_')}_prepare", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlanBenchPrepareTest(unittest.TestCase):
    def test_task1_prepare_writes_plan_items(self) -> None:
        module = _load_prepare_module("planbench-t1")
        rows = [
            {
                "domain": "logistics",
                "prompt_type": "oneshot",
                "instance_id": 2,
                "query": "Plan a route.",
                "ground_truth_plan": "(load-truck p0 t0 l0-0)\n",
                "task": "task_1_plan_generation",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "questions.json"
            with (
                patch.object(sys, "argv", ["prepare.py", "--output", str(output)]),
                patch("app.bench.planbench_prepare.load_rows", return_value=rows),
            ):
                module.main()
            payload = json.loads(output.read_text())
            item = payload["items"][0]
            self.assertEqual(item["item_id"], "planbench-t1-logistics-oneshot-00002")
            self.assertEqual(item["prompt"], "Plan a route.")
            self.assertEqual(item["expected_answer"], "(load-truck p0 t0 l0-0)")
            self.assertEqual(item["metadata"]["config"], "task_1_plan_generation")
            self.assertEqual(item["metadata"]["source_index"], 0)
            self.assertEqual(item["metadata"]["runtime_split_tags"], ["domain:logistics", "prompt_type:oneshot"])

    def test_task2_prepare_records_optimal_plan_steps(self) -> None:
        module = _load_prepare_module("planbench-t2")
        rows = [
            {
                "domain": "logistics",
                "prompt_type": "oneshot",
                "instance_id": 7,
                "query": "Plan optimally.",
                "ground_truth_plan": "(load-truck p0 t0 l0-0)\n(drive-truck t0 l0-0 l0-1 c0)\n",
                "task": "task_2_plan_optimality",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "questions.json"
            with (
                patch.object(sys, "argv", ["prepare.py", "--output", str(output)]),
                patch("app.bench.planbench_prepare.load_rows", return_value=rows),
            ):
                module.main()
            payload = json.loads(output.read_text())
            item = payload["items"][0]
            self.assertEqual(item["item_id"], "planbench-t2-logistics-oneshot-00007")
            self.assertEqual(item["metadata"]["optimal_plan_steps"], 2)
            self.assertEqual(item["metadata"]["config"], "task_2_plan_optimality")
            self.assertEqual(item["metadata"]["runtime_split_tags"], ["domain:logistics", "prompt_type:oneshot"])

    def test_task3_prepare_normalizes_binary_verdicts(self) -> None:
        module = _load_prepare_module("planbench-t3")
        rows = [
            {
                "domain": "mystery_blocksworld_3",
                "prompt_type": "oneshot",
                "instance_id": 1,
                "query": "Verify the plan.",
                "ground_truth_plan": "The above plan is invalid. This is the unmet goal condition:\nobject a craves object b",
                "task": "task_3_plan_verification",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "questions.json"
            with (
                patch.object(sys, "argv", ["prepare.py", "--output", str(output)]),
                patch("app.bench.planbench_prepare.load_rows", return_value=rows),
            ):
                module.main()
            payload = json.loads(output.read_text())
            item = payload["items"][0]
            self.assertEqual(item["item_id"], "planbench-t3-mystery_blocksworld_3-oneshot-00001")
            self.assertEqual(item["expected_answer"], "no")
            self.assertEqual(item["choices"], ["yes", "no"])
            self.assertEqual(item["metadata"]["correct_choice_index"], 1)
            self.assertIn("invalid", item["metadata"]["answer_aliases"])
            self.assertEqual(item["metadata"]["runtime_split_tags"], ["domain:mystery_blocksworld_3", "prompt_type:oneshot"])
            self.assertTrue(item["prompt"].endswith("Answer only yes or no: is the plan valid?"))

    def test_task3_prepare_derives_missing_verdict_from_semantics(self) -> None:
        module = _load_prepare_module("planbench-t3")
        rows = [
            {
                "domain": "blocksworld_3",
                "prompt_type": "oneshot",
                "instance_id": 2,
                "query": "Verify the final plan.\n\n[PLAN]\npick up the red block\n[PLAN END]",
                "ground_truth_plan": None,
                "task": "task_3_plan_verification_with_llm_plans",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "questions.json"
            with (
                patch.object(sys, "argv", ["prepare.py", "--output", str(output)]),
                patch("app.bench.planbench_prepare.load_rows", return_value=rows),
                patch("app.bench.planbench_prepare.derive_verification_verdict_from_query", return_value=("yes", "Plan valid")),
            ):
                module.main()
            payload = json.loads(output.read_text())
            item = payload["items"][0]
            self.assertEqual(item["item_id"], "planbench-t3-blocksworld_3-oneshot-00002-task_3_plan_verification_with_llm_plans")
            self.assertEqual(item["expected_answer"], "yes")
            self.assertEqual(item["metadata"]["verdict_source"], "semantic_derivation")
            self.assertEqual(item["metadata"]["semantic_verification_detail"], "Plan valid")
            self.assertIsNone(item["metadata"]["official_verification"])


if __name__ == "__main__":
    unittest.main()
