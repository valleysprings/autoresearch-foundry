from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.benchmark_support import public_question_payload
from app.codegen.catalog import load_codegen_tasks
from app.codegen.dataset_support import build_micro_task, load_question_manifest
from app.codegen.verifier import evaluate_materialized_candidate, materialize_candidate
from tests.helpers import load_fixture_codegen_tasks


ROOT = Path(__file__).resolve().parents[1]
PLANBENCH_TEST_LAYOUT = {
    "obfuscated_deceptive_logistics": (
        "obfuscated_deceptive_logistics/generated_domain.pddl",
        "obfuscated_deceptive_logistics/generated_basic",
    ),
    "logistics": (
        "logistics/generated_domain.pddl",
        "logistics/generated_basic",
    ),
}


def canonical_plan_steps(raw_plan: object) -> list[str]:
    text = str(raw_plan or "")
    parenthesized = [match.strip() for match in re.findall(r"\(([^()]+)\)", text)]
    if parenthesized:
        return parenthesized
    steps: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().strip("()")
        if stripped:
            steps.append(stripped)
    return steps


def canonical_plan_text(raw_plan: object) -> str:
    return "\n".join(canonical_plan_steps(raw_plan))


def logistics_symbol(symbol: str) -> str:
    if symbol.startswith("p"):
        return f"package_{symbol[1:]}"
    if symbol.startswith("a"):
        return f"airplane_{symbol[1:]}"
    if symbol.startswith("t"):
        return f"truck_{symbol[1:]}"
    if symbol.startswith("l"):
        left, right = symbol[1:].split("-", 1)
        return f"location_{left}_{right}"
    if symbol.startswith("c"):
        return f"city_{symbol[1:]}"
    return symbol


def logistics_line(step: str) -> str:
    action, *args = step.split()
    if action == "load-airplane":
        package, airplane, location = map(logistics_symbol, args)
        return f"load {package} into {airplane} at {location}"
    if action == "load-truck":
        package, truck, location = map(logistics_symbol, args)
        return f"load {package} into {truck} at {location}"
    if action == "unload-airplane":
        package, airplane, location = map(logistics_symbol, args)
        return f"unload {package} from {airplane} at {location}"
    if action == "unload-truck":
        package, truck, location = map(logistics_symbol, args)
        return f"unload {package} from {truck} at {location}"
    if action == "drive-truck":
        truck, source, target = map(logistics_symbol, args)
        return f"drive {truck} from {source} to {target}"
    if action == "fly-airplane":
        airplane, source, target = map(logistics_symbol, args)
        return f"fly {airplane} from {source} to {target}"
    raise ValueError(f"Unsupported logistics action: {action}")


class CodegenVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        tasks = load_fixture_codegen_tasks()
        self.task = next(task for task in tasks if task["id"] == "contains-duplicates")

    def _materialize(self, task: dict, file_body: str) -> tuple[Path, str, str]:
        temp_dir = tempfile.TemporaryDirectory()
        workspace = Path(temp_dir.name)
        source_path, source_code = materialize_candidate(
            task=task,
            workspace_root=workspace,
            candidate_id="candidate",
            file_body=file_body,
        )
        self.addCleanup(temp_dir.cleanup)
        return workspace, source_path, source_code

    def _stub_planbench_assets(self, item: dict[str, object], *, expected_plan: str) -> None:
        raw_context = item.get("raw_context")
        if not isinstance(raw_context, dict):
            raise ValueError("PlanBench test item must provide raw_context.")
        domain = str(raw_context["domain"])
        instance_id = int(raw_context["instance_id"])
        if domain not in PLANBENCH_TEST_LAYOUT:
            raise ValueError(f"Missing PlanBench test layout for domain {domain!r}.")
        domain_file, instance_dir = PLANBENCH_TEST_LAYOUT[domain]

        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        domain_path = root / "instances" / domain_file
        instance_path = root / "instances" / instance_dir / f"instance-{instance_id}.pddl"
        validator_path = root / "validate_stub.py"

        domain_path.parent.mkdir(parents=True, exist_ok=True)
        domain_path.write_text("(define (domain stub))\n")
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        instance_path.write_text("(define (problem stub))\n")
        validator_path.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "plan = Path(sys.argv[3]).read_text().strip()\n"
            "expected = os.environ.get('PLANBENCH_STUB_EXPECTED_PLAN', '').strip()\n"
            "if plan == expected:\n"
            "    print('Plan valid')\n"
            "    raise SystemExit(0)\n"
            "print('Plan invalid')\n"
            "raise SystemExit(1)\n"
        )
        validator_path.chmod(0o755)

        normalized_expected_plan = "\n".join(f"({step})" for step in canonical_plan_steps(expected_plan))
        env_patch = patch.dict(
            os.environ,
            {
                "PLANBENCH_OFFICIAL_ROOT": str(root),
                "PLANBENCH_VAL_BINARY": str(validator_path),
                "PLANBENCH_STUB_EXPECTED_PLAN": normalized_expected_plan,
            },
            clear=False,
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.addCleanup(temp_dir.cleanup)

    def test_materializer_writes_full_editable_file(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return len(values) != len(set(values))\n",
        )
        self.assertEqual(source_path.name, "editable.py")
        self.assertIn("def contains_duplicates(values):", source_code)
        self.assertTrue(source_code.endswith("\n"))

    def test_compile_error_marks_candidate_as_error(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return values[\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "error")
        self.assertFalse(metrics["gate_passed"])
        self.assertEqual(metrics["primary_score"], 0.0)

    def test_test_failure_marks_candidate_as_fail(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    return False\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "fail")
        self.assertEqual(metrics["correctness"], 0.0)

    def test_runtime_exception_is_captured_as_error(self) -> None:
        _, source_path, source_code = self._materialize(
            self.task,
            "def contains_duplicates(values):\n    raise RuntimeError('boom')\n",
        )
        metrics = evaluate_materialized_candidate(
            task=self.task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "error")
        self.assertIn("boom", metrics["error"])

    def test_passing_candidate_benchmarks_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            baseline_source = Path(self.task["editable_path"]).read_text()
            baseline_path, baseline_code = materialize_candidate(
                task=self.task,
                workspace_root=workspace,
                candidate_id="baseline",
                file_body=baseline_source,
            )
            baseline_metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=baseline_path,
                source_code=baseline_code,
                baseline_metrics=None,
                memory_applied=False,
            )
            candidate_path, candidate_code = materialize_candidate(
                task=self.task,
                workspace_root=workspace,
                candidate_id="fast",
                file_body="def contains_duplicates(values):\n    return len(values) != len(set(values))\n",
            )
            metrics = evaluate_materialized_candidate(
                task=self.task,
                source_path=candidate_path,
                source_code=candidate_code,
                baseline_metrics=baseline_metrics,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)
            self.assertGreater(metrics["primary_score"], baseline_metrics["primary_score"])

    def test_math_experiment_candidate_improves_benchmark(self) -> None:
        task = next(task for task in load_fixture_codegen_tasks() if task["id"] == "count-primes-up-to")
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            baseline_source = Path(task["editable_path"]).read_text()
            baseline_path, baseline_code = materialize_candidate(
                task=task,
                workspace_root=workspace,
                candidate_id="baseline-math",
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
                workspace_root=workspace,
                candidate_id="sieve",
                file_body=(
                    "def count_primes_up_to(limit):\n"
                    "    if limit < 2:\n"
                    "        return 0\n"
                    "    sieve = [True] * (limit + 1)\n"
                    "    sieve[0] = False\n"
                    "    sieve[1] = False\n"
                    "    candidate = 2\n"
                    "    while candidate * candidate <= limit:\n"
                    "        if sieve[candidate]:\n"
                    "            step = candidate * candidate\n"
                    "            while step <= limit:\n"
                    "                sieve[step] = False\n"
                    "                step += candidate\n"
                    "        candidate += 1\n"
                    "    return sum(1 for is_prime in sieve if is_prime)\n"
                ),
            )
            metrics = evaluate_materialized_candidate(
                task=task,
                source_path=candidate_path,
                source_code=candidate_code,
                baseline_metrics=baseline_metrics,
                memory_applied=False,
            )
            self.assertEqual(metrics["status"], "pass")
            self.assertGreater(metrics["speedup_vs_baseline"], 1.0)

    def test_no_standalone_non_dataset_comparable_tracks_remain_in_active_research_lane(self) -> None:
        comparable_tasks = load_codegen_tasks(included_in_main_comparison=True)
        comparable_tasks = [task for task in comparable_tasks if not task.get("local_dataset_only")]
        self.assertEqual(comparable_tasks, [])

    def test_dataset_question_microtasks_generate_item_level_records(self) -> None:
        dataset_tasks = [task for task in load_codegen_tasks(included_in_main_comparison=True) if task.get("local_dataset_only")]
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
        eager_dataset_tasks = [task for task in dataset_tasks if not task.get("lazy_item_manifest")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            for task in eager_dataset_tasks:
                items = load_question_manifest(task)
                micro_task = build_micro_task(task, items[0])
                source = Path(task["editable_path"]).read_text()
                candidate_path, candidate_code = materialize_candidate(
                    task=micro_task,
                    workspace_root=workspace / task["id"],
                    candidate_id="baseline",
                    file_body=source,
                )
                metrics = evaluate_materialized_candidate(
                    task=micro_task,
                    source_path=candidate_path,
                    source_code=candidate_code,
                    baseline_metrics=None,
                    memory_applied=False,
                )
                self.assertEqual(metrics["total_tests"], 1)
                self.assertEqual(len(metrics["test_results"]), 1)
                self.assertEqual(metrics["test_results"][0]["name"], items[0]["name"])

    def test_livecodebench_lazy_manifest_can_prepare_on_demand(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "livecodebench")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_dir = tmp / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            prepare_path = tmp / "prepare.py"
            prepare_path.write_text(
                (
                    "import json\n"
                    "from pathlib import Path\n\n"
                    "root = Path(__file__).resolve().parent\n"
                    "data_dir = root / 'data'\n"
                    "data_dir.mkdir(parents=True, exist_ok=True)\n"
                    "(data_dir / 'questions.json').write_text(json.dumps({'items': [{\n"
                    "  'item_id': 'lazy-livecodebench-sample',\n"
                    "  'name': 'lazy-livecodebench-sample',\n"
                    "  'prompt': 'Write a solver.',\n"
                    "  'context': 'Synthetic coding task.',\n"
                    "  'expected_answer': 'Pass all public and private tests.',\n"
                    "  'metadata': {'problem_file': 'problems/lazy-livecodebench-sample.json'}\n"
                    "}]}, indent=2))\n"
                )
            )
            lazy_task = {
                **task,
                "task_dir": str(tmp),
                "item_manifest_path": str(data_dir / "questions.json"),
                "editable_path": task["editable_path"],
                "verifier_path": task["verifier_path"],
            }
            items = load_question_manifest(lazy_task, min_items=1)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["item_id"], "lazy-livecodebench-sample")

    def test_livecodebench_stdin_candidate_passes_cached_problem(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "livecodebench")
        item = {
            "id": "livecodebench-stdin-sample",
            "item_id": "livecodebench-stdin-sample",
            "question_id": "livecodebench-stdin-sample",
            "raw_item_id": "livecodebench-stdin-sample",
            "name": "sum-two-integers",
            "prompt": "Read two integers and print their sum.",
            "raw_prompt": "Read two integers and print their sum.",
            "context": "Platform: atcoder. Evaluation mode: stdin.",
            "raw_context": "Platform: atcoder. Evaluation mode: stdin.",
            "choices": [],
            "raw_choices": [],
            "expected_answer": "Pass all public and private tests.",
            "raw_expected_answer": "Pass all public and private tests.",
            "metadata": {
                "problem_file": str(ROOT / "tests" / "fixtures" / "livecodebench" / "problems" / "stdin_problem.json"),
                "platform": "atcoder",
                "evaluation_mode": "stdin",
            },
        }
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "from __future__ import annotations\n\n"
                "import sys\n\n"
                "def solve() -> None:\n"
                "    left, right = map(int, sys.stdin.read().split())\n"
                "    print(left + right)\n\n"
                "class Solution:\n"
                "    pass\n\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["total_tests"], 2)
        self.assertEqual(metrics["test_results"][0]["actual"], "5")

    def test_livecodebench_functional_candidate_passes_cached_problem(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "livecodebench")
        item = {
            "id": "livecodebench-functional-sample",
            "item_id": "livecodebench-functional-sample",
            "question_id": "livecodebench-functional-sample",
            "raw_item_id": "livecodebench-functional-sample",
            "name": "add-two",
            "prompt": "Implement Solution.addTwo(a, b) and return the sum.",
            "raw_prompt": "Implement Solution.addTwo(a, b) and return the sum.",
            "context": "Platform: leetcode. Evaluation mode: functional. Required method: Solution.addTwo.",
            "raw_context": "Platform: leetcode. Evaluation mode: functional. Required method: Solution.addTwo.",
            "choices": [],
            "raw_choices": [],
            "expected_answer": "Pass all public and private tests.",
            "raw_expected_answer": "Pass all public and private tests.",
            "metadata": {
                "problem_file": str(ROOT / "tests" / "fixtures" / "livecodebench" / "problems" / "functional_problem.json"),
                "platform": "leetcode",
                "evaluation_mode": "functional",
                "function_name": "addTwo",
            },
        }
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "from __future__ import annotations\n\n"
                "def solve() -> None:\n"
                "    return None\n\n"
                "class Solution:\n"
                "    def addTwo(self, left: int, right: int) -> int:\n"
                "        return left + right\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["total_tests"], 2)
        self.assertEqual(metrics["test_results"][0]["actual"], "3")

    def test_sciq_question_id_alias_is_available_to_solver(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "sciq")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "def solve(question: dict) -> str:\n"
                "    if question.get('id') == 'sciq-validation-0':\n"
                "        return 'darwin'\n"
                "    return ''\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], "darwin")

    def test_sciq_choice_letter_answer_is_accepted(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "sciq")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        correct_choice_index = int(item["metadata"]["correct_choice_index"])
        correct_choice_letter = chr(ord("A") + correct_choice_index)
        _, source_path, source_code = self._materialize(
            micro_task,
            f"def solve(question: dict) -> str:\n    return {correct_choice_letter!r}\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], item["expected_answer"])

    def test_qasc_choice_letter_answer_is_accepted(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "qasc")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        correct_choice_index = int(item["metadata"]["correct_choice_index"])
        correct_choice_letter = chr(ord("A") + correct_choice_index)
        _, source_path, source_code = self._materialize(
            micro_task,
            f"def solve(question: dict) -> str:\n    return {correct_choice_letter!r}\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], item["expected_answer"])

    def test_aime_2026_numeric_answer_is_normalized_before_match(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "aime-2026")
        item = next(item for item in load_question_manifest(task) if item["item_id"] == "aime-2026-01")
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            "def solve(question: dict) -> str:\n    return '0277'\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], "277")

    def test_aime_numeric_answer_is_normalized_before_match(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "aime-2024")
        item = next(item for item in load_question_manifest(task) if item["item_id"] == "aime-2024-02")
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            "def solve(question: dict) -> str:\n    return '0113'\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], "113")

    def test_dataset_solver_cannot_read_expected_answer_from_question_payload(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "sciq")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "def solve(question: dict) -> str:\n"
                "    answer = question.get('expected_answer')\n"
                "    alias = question.get('metadata', {}).get('answer_aliases')\n"
                "    if answer:\n"
                "        return str(answer)\n"
                "    if alias:\n"
                "        return str(alias[0])\n"
                "    return ''\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "fail")
        self.assertNotEqual(metrics["test_results"][0]["actual"], "darwin")

    def test_math_dataset_solver_cannot_read_expected_answer_from_question_payload(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "math-500")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "def solve(question: dict) -> str:\n"
                "    answer = question.get('expected_answer')\n"
                "    metadata = question.get('metadata', {})\n"
                "    if answer:\n"
                "        return str(answer)\n"
                "    if metadata.get('correct_choice_index') is not None:\n"
                "        return str(metadata['correct_choice_index'])\n"
                "    return ''\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "fail")
        self.assertNotEqual(metrics["test_results"][0]["actual"], "5")

    def test_network_access_is_rejected_for_dataset_tasks(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "sciq")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            (
                "import urllib.request\n\n"
                "def solve(question: dict) -> str:\n"
                "    return urllib.request.urlopen('https://example.com').read().decode()\n"
            ),
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "error")
        self.assertIn("disabled", metrics["error"])

    def test_olymmath_math_verify_accepts_symbolic_equivalent_answer(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "olymmath")
        item = next(item for item in load_question_manifest(task) if item["raw_item_id"] == "OlymMATH-HARD-4-EN")
        micro_task = build_micro_task(task, item)
        _, source_path, source_code = self._materialize(
            micro_task,
            "def solve(question: dict) -> str:\n    return r'$44*sqrt(30)+241$'\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertIn("44*sqrt(30)+241", metrics["test_results"][0]["actual"])

    def test_planbench_manifest_item_ids_are_unique(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "planbench")
        items = load_question_manifest(task)
        item_ids = [item["item_id"] for item in items]
        self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_planbench_public_question_payload_preserves_raw_prompt_and_context(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "planbench")
        item = load_question_manifest(task)[0]
        micro_task = build_micro_task(task, item)
        payload = public_question_payload(item)
        self.assertEqual(payload["prompt"], item["raw_prompt"])
        self.assertEqual(payload["context"], item["raw_context"])
        self.assertIn("Question raw prompt:", micro_task["prompt_context"])
        self.assertIn("[PLAN]", micro_task["prompt_context"])

    def test_planbench_obfuscated_single_line_object_format_is_accepted(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "planbench")
        item = next(item for item in load_question_manifest(task) if item["metadata"].get("domain") == "obfuscated_deceptive_logistics")
        micro_task = build_micro_task(task, item)
        self._stub_planbench_assets(item, expected_plan=str(item["expected_answer"]))
        raw_plan = " ".join(
            re.sub(r"\bo(\d+)\b", r"object_\1", step)
            for step in canonical_plan_steps(item["expected_answer"])
        )
        _, source_path, source_code = self._materialize(
            micro_task,
            f"def solve(question: dict) -> str:\n    return {raw_plan!r}\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], canonical_plan_text(item["expected_answer"]))
        self.assertIn("Plan valid", metrics["test_results"][0]["validator_output"])

    def test_planbench_logistics_natural_language_plan_is_accepted(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "planbench")
        item = next(item for item in load_question_manifest(task) if item["metadata"].get("domain") == "logistics")
        micro_task = build_micro_task(task, item)
        self._stub_planbench_assets(item, expected_plan=str(item["expected_answer"]))
        natural_plan = "\n".join(logistics_line(step) for step in canonical_plan_steps(item["expected_answer"]))
        _, source_path, source_code = self._materialize(
            micro_task,
            f"def solve(question: dict) -> str:\n    return {natural_plan!r}\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], canonical_plan_text(item["expected_answer"]))
        self.assertIn("Plan valid", metrics["test_results"][0]["validator_output"])

    def test_planbench_parenthesized_pddl_plan_is_accepted(self) -> None:
        task = next(task for task in load_codegen_tasks() if task["id"] == "planbench")
        item = next(item for item in load_question_manifest(task) if item["metadata"].get("domain") == "obfuscated_deceptive_logistics")
        micro_task = build_micro_task(task, item)
        self._stub_planbench_assets(item, expected_plan=str(item["expected_answer"]))
        plan_steps = [f"({step})" for step in canonical_plan_steps(item["expected_answer"])]
        _, source_path, source_code = self._materialize(
            micro_task,
            f"def solve(question: dict) -> list[str]:\n    return {plan_steps!r}\n",
        )
        metrics = evaluate_materialized_candidate(
            task=micro_task,
            source_path=source_path,
            source_code=source_code,
            baseline_metrics=None,
            memory_applied=False,
        )
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["test_results"][0]["actual"], canonical_plan_text(item["expected_answer"]))


if __name__ == "__main__":
    unittest.main()
