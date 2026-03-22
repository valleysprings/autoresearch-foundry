from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.codegen.catalog import load_codegen_tasks
from app.codegen.verifier import evaluate_materialized_candidate, materialize_candidate


HANDCRAFTED_CANDIDATES = {
    "olymmath-numeric-small": """def solve(problem: str) -> str:
    text = problem.lower()
    if "remainder 1 when divided by 2, 3, 4, 5, and 6" in text:
        return "301"
    if "13, 14, and 15" in text:
        return "84"
    if "positive divisors does 360 have" in text:
        return "24"
    if "x + 1/x = 3" in text:
        return "7"
    return "0"
""",
    "planbench-lite": """from collections import deque

def solve(problem: dict) -> list[str]:
    roads = {}
    for left, right in problem["roads"]:
        roads.setdefault(left, set()).add(right)
        roads.setdefault(right, set()).add(left)

    start = problem["start"]
    package_start = problem["package_start"]
    goal = problem["goal"]
    initial = (start, package_start, False)
    queue = deque([(initial, [])])
    seen = {initial}

    while queue:
        (robot, package, holding), path = queue.popleft()
        if package == goal and not holding:
            return path
        for target in roads.get(robot, set()):
            state = (target, package, holding)
            if state not in seen:
                seen.add(state)
                queue.append((state, path + [f"drive {robot} {target}"]))
        if not holding and robot == package:
            state = (robot, package, True)
            if state not in seen:
                seen.add(state)
                queue.append((state, path + [f"pickup package {robot}"]))
        if holding:
            state = (robot, robot, False)
            if state not in seen:
                seen.add(state)
                queue.append((state, path + [f"drop package {robot}"]))
    return []
""",
    "multihop-snapshot-small": """def solve(question: str, documents: list[dict]):
    question = question.lower()
    if "aurora museum" in question or "lyra venn" in question:
        return "lunaport"
    if "southern country" in question or "mirel" in question:
        return "brightbay"
    if "all capital cities" in question:
        return ["brightbay", "lunaport"]
    return ""
""",
    "tbench-lite": """def build_commands(case: dict) -> list[str]:
    if "errors.txt" in case["expected_files"]:
        return ["grep '^ERROR' app.log > errors.txt"]
    return ["grep '^TASK ' notes.txt | sort -u > todo.txt"]
""",
}


class BenchmarkComparisonTest(unittest.TestCase):
    def test_handcrafted_candidates_outperform_baselines_on_comparable_tracks(self) -> None:
        tasks = load_codegen_tasks(included_in_main_comparison=True)
        if not tasks:
            self.skipTest("No local benchmark task assets are available.")

        expected_ids = set(HANDCRAFTED_CANDIDATES)
        actual_ids = {task["id"] for task in tasks}
        self.assertTrue(expected_ids.issubset(actual_ids))

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for task in tasks:
                if task["id"] not in HANDCRAFTED_CANDIDATES:
                    continue
                baseline_source = Path(task["editable_path"]).read_text()
                baseline_path, baseline_code = materialize_candidate(
                    task=task,
                    workspace_root=root / task["id"] / "baseline",
                    candidate_id="baseline",
                    file_body=baseline_source,
                )
                baseline_metrics = evaluate_materialized_candidate(
                    task=task,
                    source_path=baseline_path,
                    source_code=baseline_code,
                    baseline_metrics=None,
                    memory_applied=False,
                )
                candidate_path, candidate_code = materialize_candidate(
                    task=task,
                    workspace_root=root / task["id"] / "candidate",
                    candidate_id="candidate",
                    file_body=HANDCRAFTED_CANDIDATES[task["id"]],
                )
                candidate_metrics = evaluate_materialized_candidate(
                    task=task,
                    source_path=candidate_path,
                    source_code=candidate_code,
                    baseline_metrics=baseline_metrics,
                    memory_applied=False,
                )
                self.assertEqual(candidate_metrics["status"], "pass", msg=task["id"])
                self.assertGreater(candidate_metrics["objective"], baseline_metrics["objective"], msg=task["id"])
                self.assertGreater(candidate_metrics["J"], baseline_metrics["J"], msg=task["id"])


if __name__ == "__main__":
    unittest.main()
