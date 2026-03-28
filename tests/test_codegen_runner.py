from __future__ import annotations

import json
import os
import tempfile
import textwrap
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.codegen.errors import ConfigError, LlmResponseError, LlmTransportError
from app.codegen.catalog import load_codegen_tasks, seed_strategy_experiences
from app.codegen.dataset_runner import _question_payload_for_result, run_dataset_task
from app.codegen.dataset_support import build_micro_task, load_question_manifest
from app.codegen.llm import _proposal_prompt
from app.codegen.config import RuntimeConfig
from app.codegen.llm import ProposalRuntime
from app.codegen.trainer import run_codegen_task
from app.entries.runner import generate_discrete_payload, write_discrete_artifacts
from app.memory.store import MemoryStore
from tests.helpers import chat_response, load_fixture_codegen_tasks, make_runtime, patch_runner_fixture_catalog


ROOT = Path(__file__).resolve().parents[1]


def _file_with_entry(symbol: str, args: str, body: str) -> str:
    return f"def {symbol}({args}):\n{textwrap.indent(body.strip(), '    ')}\n"


def _fixture_task(task_id: str) -> dict[str, object]:
    return dict(next(item for item in load_fixture_codegen_tasks() if item["id"] == task_id))


def raw_content_response(
    content: str,
    *,
    model: str = "deepseek-chat",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> str:
    return json.dumps(
        {
            "id": "resp-raw",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "model": model,
        }
    )


PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Set cardinality",
            "strategy": "Use a set cardinality check.",
            "rationale": "The runtime can detect duplicates by comparing lengths.",
            "imports": [],
            "file_body": _file_with_entry("contains_duplicates", "values", "return len(values) != len(set(values))"),
            "candidate_summary": "Single-pass set cardinality duplicate detection.",
        },
        {
            "name": "Seen set",
            "strategy": "Stream through the list with a seen set.",
            "rationale": "Early exit preserves correctness and still removes quadratic scans.",
            "imports": [],
            "file_body": _file_with_entry(
                "contains_duplicates",
                "values",
                "seen = set()\nfor value in values:\n    if value in seen:\n        return True\n    seen.add(value)\nreturn False",
            ),
            "candidate_summary": "Streaming duplicate detection with early exit.",
        },
        {
            "name": "Sorted scan",
            "strategy": "Sort then scan neighbors.",
            "rationale": "Sorting avoids quadratic pair comparisons.",
            "imports": [],
            "file_body": _file_with_entry(
                "contains_duplicates",
                "values",
                "ordered = sorted(values)\nfor index in range(1, len(ordered)):\n    if ordered[index] == ordered[index - 1]:\n        return True\nreturn False",
            ),
            "candidate_summary": "Sort-based duplicate detection.",
        },
    ]
}

PARALLEL_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Seen set",
            "strategy": "Stream through the list with a seen set.",
            "rationale": "Early exit preserves correctness and removes quadratic scans.",
            "imports": [],
            "file_body": _file_with_entry(
                "contains_duplicates",
                "values",
                "seen = set()\nfor value in values:\n    if value in seen:\n        return True\n    seen.add(value)\nreturn False",
            ),
            "candidate_summary": "Streaming duplicate detection with early exit.",
        }
    ]
}

FAILURE_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Always false",
            "strategy": "Return a constant false value.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "file_body": _file_with_entry("contains_duplicates", "values", "return False"),
            "candidate_summary": "Incorrect constant-false duplicate detector.",
        },
        {
            "name": "Always true",
            "strategy": "Return a constant true value.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "file_body": _file_with_entry("contains_duplicates", "values", "return True"),
            "candidate_summary": "Incorrect constant-true duplicate detector.",
        },
        {
            "name": "Length gate",
            "strategy": "Use an obviously wrong shortcut.",
            "rationale": "This is intentionally incorrect and should fail deterministic tests.",
            "imports": [],
            "file_body": _file_with_entry("contains_duplicates", "values", "return len(values) > 3"),
            "candidate_summary": "Incorrect heuristic duplicate detector.",
        },
    ]
}

REFLECTION_PAYLOAD = {
    "failure_pattern": "Quadratic duplicate detection wasted time on repeated scans.",
    "strategy_hypothesis": "Hash-based membership checks will dominate the nested scan on this task family.",
    "successful_strategy": "Use a set cardinality check to detect duplicates.",
    "prompt_fragment": "Prefer a hash-based duplicate detector that preserves semantics and reduces the scan to near-linear work.",
    "tool_trace_summary": "The set-cardinality candidate passed all tests and outpaced the baseline benchmark.",
}

FAILURE_REFLECTION_PAYLOAD = {
    "failure_pattern": "The mutation preserved correctness but stalled because it did not improve on the incumbent benchmark.",
    "strategy_hypothesis": "Repeatedly proposing the same architecture wastes generations once the best set-based path is already in place.",
    "successful_strategy": "Shift to a materially different architecture only when it can beat the incumbent, otherwise avoid repeating the same plan.",
    "prompt_fragment": "Do not spend another generation on a semantic no-op that merely restates the incumbent strategy without a measurable gain.",
    "tool_trace_summary": "candidate passed tests, matched or trailed the incumbent benchmark, and was rejected as a non-improving repeat.",
}

NON_IMPROVING_PASS_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Sorted scan",
            "strategy": "Sort then scan neighbors.",
            "rationale": "This stays correct but should trail the already accepted set-based winner.",
            "imports": [],
            "file_body": _file_with_entry(
                "contains_duplicates",
                "values",
                "ordered = sorted(values)\nfor index in range(1, len(ordered)):\n    if ordered[index] == ordered[index - 1]:\n        return True\nreturn False",
            ),
            "candidate_summary": "Correct but weaker sorted duplicate detection.",
        }
    ]
}

FULL_FILE_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Full editable file",
            "strategy": "Return the entire editable file as required by the editable-file benchmark contract.",
            "rationale": "The runtime should accept a full file_body payload without any extra wrapping logic.",
            "imports": [],
            "file_body": _file_with_entry("contains_duplicates", "values", "return len(values) != len(set(values))"),
            "candidate_summary": "Set-cardinality duplicate detector returned as a full editable file.",
        }
    ]
}

QUESTION_SOLVER_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "Prompt matcher",
            "strategy": "Use prompt keywords from the normalized question schema.",
            "rationale": "Each dataset question run is independent, so a prompt-targeted full-file rewrite can solve the active question.",
            "imports": [],
            "file_body": (
                "def solve(question: dict) -> str:\n"
                "    prompt = str(question.get('prompt') or '').lower()\n"
                "    if 'remainder 1 when divided by 2, 3, 4, 5, and 6' in prompt:\n"
                "        return '301'\n"
                "    if '13, 14, and 15' in prompt:\n"
                "        return '84'\n"
                "    if 'positive divisors does 360 have' in prompt:\n"
                "        return '24'\n"
                "    if 'x + 1/x = 3' in prompt:\n"
                "        return '7'\n"
                "    if 'natural selection' in prompt:\n"
                "        return 'darwin'\n"
                "    if 'linear sequence' in prompt and 'acids' in prompt:\n"
                "        return 'amino'\n"
                "    if 'frameshift mutation' in prompt:\n"
                "        return 'nucleotides'\n"
                "    choices = question.get('choices') or []\n"
                "    return str(choices[0]) if choices else ''\n"
            ),
            "candidate_summary": "Keyword-based solver over the normalized question dict.",
        }
    ]
}

PLANBENCH_PROPOSAL_PAYLOAD = {
    "candidates": [
        {
            "name": "BFS planner",
            "strategy": "Search over robot, package, and carrying state with breadth-first search.",
            "rationale": "The planning benchmark is small and deterministic, so BFS can return a shortest valid action list.",
            "imports": [],
            "file_body": (
                "from collections import deque\n\n"
                "def solve(problem: dict) -> list[str]:\n"
                "    graph: dict[str, set[str]] = {}\n"
                "    for left, right in problem.get('roads', []):\n"
                "        graph.setdefault(left, set()).add(right)\n"
                "        graph.setdefault(right, set()).add(left)\n"
                "    start = (problem['start'], problem['package_start'], False)\n"
                "    queue = deque([(start, [])])\n"
                "    seen = {start}\n"
                "    goal = problem['goal']\n"
                "    while queue:\n"
                "        (robot, package, holding), path = queue.popleft()\n"
                "        if not holding and package == goal:\n"
                "            return path\n"
                "        for neighbor in sorted(graph.get(robot, set())):\n"
                "            next_state = (neighbor, package, holding)\n"
                "            if next_state not in seen:\n"
                "                seen.add(next_state)\n"
                "                queue.append((next_state, path + [f'drive {robot} {neighbor}']))\n"
                "        if not holding and robot == package:\n"
                "            next_state = (robot, package, True)\n"
                "            if next_state not in seen:\n"
                "                seen.add(next_state)\n"
                "                queue.append((next_state, path + [f'pickup pkg {robot}']))\n"
                "        if holding:\n"
                "            next_state = (robot, robot, False)\n"
                "            if next_state not in seen:\n"
                "                seen.add(next_state)\n"
                "                queue.append((next_state, path + [f'drop pkg {robot}']))\n"
                "    return []\n"
            ),
            "candidate_summary": "Breadth-first delivery planner that emits valid drive/pickup/drop actions.",
        }
    ]
}


class CodegenRunnerTest(unittest.TestCase):
    def test_missing_runtime_config_fails_before_run(self) -> None:
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                generate_discrete_payload(task_id="contains-duplicates", runs_root=Path(tmp_dir), env_root=Path(tmp_dir))

    def test_external_benchmark_task_runs_without_codegen_loop(self) -> None:
        runtime = make_runtime([])
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_discrete_payload(
                task_id="external-score",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
            )
        self.assertEqual(payload["summary"]["total_runs"], 1)
        run = payload["runs"][0]
        self.assertEqual(run["run_mode"], "external-benchmark")
        self.assertEqual(run["winner"]["agent"], "fixture-agent")
        self.assertEqual(run["winner"]["metrics"]["objective"], 0.75)
        self.assertEqual(run["winner"]["metrics"]["verifier_status"], "pass")
        self.assertEqual(run["baseline"]["metrics"]["verifier_status"], "not-run")
        self.assertEqual(run["selection_reason"], "External fixture finished with 3/4 successful cases.")
        self.assertEqual(run["external_summary"], {"passed": 3, "total": 4})
        self.assertEqual(run["generations"], [])
        task_summary = next(task for task in payload["task_catalog"] if task["id"] == "external-score")
        self.assertTrue(task_summary["supports_runtime_config"])
        self.assertEqual(task_summary["external_run_config"]["cases"], 4)
        self.assertTrue(task_summary["supports_max_items"])
        self.assertEqual(task_summary["default_max_items"], 4)

    def test_external_config_override_updates_external_task_without_editing_file(self) -> None:
        runtime = make_runtime([])
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_discrete_payload(
                task_id="external-score",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
                external_config={"agent_name": "override-agent", "cases": 5, "passed": 4},
            )
        run = payload["runs"][0]
        self.assertEqual(run["winner"]["agent"], "override-agent")
        self.assertEqual(run["winner"]["metrics"]["objective"], 0.8)
        self.assertIn("'cases': 5", run["winner"]["source_code"])
        task_summary = next(task for task in payload["task_catalog"] if task["id"] == "external-score")
        self.assertEqual(task_summary["external_run_config"]["cases"], 5)
        self.assertEqual(task_summary["default_max_items"], 5)

    def test_external_task_can_enter_codegen_search_loop_when_budgets_are_positive(self) -> None:
        runtime = make_runtime(
            [
                chat_response(
                    {
                        "candidates": [
                            {
                                "name": "perfect fixture wrapper",
                                "strategy": "Increase the passed count to the full case count.",
                                "rationale": "The fixture objective is passed / total, so making both values equal reaches 1.0.",
                                "candidate_summary": "Wrapper with a perfect synthetic pass rate.",
                                "file_body": (
                                    "def build_run_config() -> dict:\n"
                                    "    return {\n"
                                    "        'agent_name': 'evolved-agent',\n"
                                    "        'cases': 5,\n"
                                    "        'passed': 5,\n"
                                    "    }\n\n"
                                    "RUN_CONFIG = build_run_config()\n"
                                ),
                            }
                        ]
                    }
                ),
                chat_response(
                    {
                        "failure_pattern": "The baseline leaves synthetic cases unsolved.",
                        "strategy_hypothesis": "Increasing passed to match cases maximizes the fixture objective.",
                        "successful_strategy": "Return a wrapper whose build_run_config produces a perfect passed/cases ratio.",
                        "prompt_fragment": "For this fixture, evolve build_run_config toward passed == cases when that preserves the wrapper contract.",
                        "tool_trace_summary": "candidate wrapper -> fixture verifier -> accept perfect objective",
                    }
                ),
            ]
        )
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_discrete_payload(
                task_id="external-score",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
                generation_budget=1,
                candidate_budget=1,
                branching_factor=1,
            )
        run = payload["runs"][0]
        self.assertEqual(run["run_mode"], "llm-required")
        self.assertEqual(run["winner"]["agent"], "candidate-1")
        self.assertEqual(run["winner"]["metrics"]["objective"], 1.0)
        self.assertEqual(run["winner"]["metrics"]["verifier_status"], "pass")
        self.assertEqual(len(run["generations"]), 1)
        self.assertIn("build_run_config", run["winner"]["source_code"])
        self.assertGreater(float(run["delta_primary_score"]), 0.0)

    def test_math_tasks_fail_fast_when_math_verify_is_missing(self) -> None:
        runtime = make_runtime([])
        with tempfile.TemporaryDirectory() as tmp_dir, patch("app.entries.runner.importlib.util.find_spec", return_value=None):
            with self.assertRaises(ConfigError) as raised:
                generate_discrete_payload(
                    task_id="aime-2026",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                    max_items=1,
                )
        self.assertIn("math-verify", str(raised.exception))
        self.assertIn("aime-2026", str(raised.exception))

    def test_invalid_llm_output_fails_immediately(self) -> None:
        runtime = make_runtime([chat_response({"candidates": [{"name": "bad"}]})])
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_proposal_prompt_requests_exact_candidate_budget_and_json_only_output(self) -> None:
        task = _fixture_task("contains-duplicates")
        task["candidate_budget"] = 1
        candidate = {
            "candidate_summary": "Checked-in baseline.",
            "metrics": {"objective": 0.0, "objective_score": 0.0, "primary_score": 0.0, "tie_break_score": 0.42, "gate_passed": True},
            "baseline_source": "def contains_duplicates(values):\n    return False\n",
            "source_code": "def contains_duplicates(values):\n    return False\n",
        }
        system_prompt, user_prompt = _proposal_prompt(
            task=task,
            generation=1,
            parent_candidate=candidate,
            current_best=candidate,
            candidate_history=[],
            memories=[],
        )
        self.assertIn("Return exactly 1 candidate.", system_prompt)
        self.assertIn("Return only a JSON object.", system_prompt)
        self.assertIn("Do not include Markdown code fences", system_prompt)
        self.assertIn("file_body is the only field that may be long.", system_prompt)
        self.assertNotIn("Return between 1 and 3 candidates.", system_prompt)
        self.assertIn("Task mode: answer", user_prompt)
        self.assertIn("Optimization scope: implementation", user_prompt)

    def test_truncated_llm_output_surfaces_parse_details(self) -> None:
        truncated_response = raw_content_response(
            "```json\n{\"candidates\": [{\"name\": \"cut off\"",
            completion_tokens=4096,
        )
        runtime = make_runtime([truncated_response, truncated_response, truncated_response])
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError) as raised:
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )
        payload = raised.exception.as_payload()
        self.assertIn("appears truncated", payload["error"])
        details = payload.get("details")
        self.assertIsInstance(details, dict)
        assert isinstance(details, dict)
        self.assertEqual(details["parse_status"], "truncated")
        self.assertEqual(details["completion_tokens"], 4096)
        self.assertEqual(details["max_tokens"], 4096)
        self.assertEqual(details["attempt"], 3)
        self.assertTrue(details["response_truncated"])
        self.assertIn("cut off", str(details["raw_preview"]))

    def test_llm_concurrency_gate_limits_inflight_requests(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def transport(_request_body, _config):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return chat_response(PROPOSAL_PAYLOAD)
            finally:
                with lock:
                    active -= 1

        runtime = ProposalRuntime(
            RuntimeConfig(
                api_key="test-key",
                api_base="https://api.test/v1",
                primary_model="deepseek-chat",
                available_models=("deepseek-chat",),
                temperature=0.2,
                max_tokens=4096,
                timeout_s=45,
                llm_concurrency=2,
            ),
            transport=transport,
        )
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    runtime.complete_json,
                    purpose="concurrency-test",
                    system_prompt="Return strict JSON only.",
                    user_prompt='Return {"candidates":[{"name":"n","strategy":"s","rationale":"r","file_body":"def f():\\n    return 1","candidate_summary":"c"}]}',
                )
                for _ in range(5)
            ]
            for future in futures:
                payload, trace = future.result()
                self.assertIn("candidates", payload)
                self.assertEqual(trace["parse_status"], "ok")
        self.assertLessEqual(max_active, 2)

    def test_llm_queue_prioritizes_lower_generation_requests(self) -> None:
        order: list[str] = []
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        def transport(request_body, _config):
            label = str(request_body["messages"][1]["content"])
            order.append(label)
            if label == "blocker":
                blocker_started.set()
                if not release_blocker.wait(timeout=2):
                    raise AssertionError("blocker was not released")
            return raw_content_response('{"ok": true}')

        runtime = ProposalRuntime(
            RuntimeConfig(
                api_key="test-key",
                api_base="https://api.test/v1",
                primary_model="deepseek-chat",
                available_models=("deepseek-chat",),
                temperature=0.2,
                max_tokens=4096,
                timeout_s=45,
                llm_concurrency=1,
            ),
            transport=transport,
        )
        with ThreadPoolExecutor(max_workers=4) as executor:
            blocker = executor.submit(
                runtime.complete_json,
                purpose="queue-test",
                system_prompt="Return strict JSON only.",
                user_prompt="blocker",
                queue_priority=999,
            )
            self.assertTrue(blocker_started.wait(timeout=1))
            generation_two = executor.submit(
                runtime.complete_json,
                purpose="queue-test",
                system_prompt="Return strict JSON only.",
                user_prompt="generation-2",
                queue_priority=20,
            )
            generation_one_a = executor.submit(
                runtime.complete_json,
                purpose="queue-test",
                system_prompt="Return strict JSON only.",
                user_prompt="generation-1-a",
                queue_priority=10,
            )
            generation_one_b = executor.submit(
                runtime.complete_json,
                purpose="queue-test",
                system_prompt="Return strict JSON only.",
                user_prompt="generation-1-b",
                queue_priority=10,
            )
            release_blocker.set()
            for future in (blocker, generation_two, generation_one_a, generation_one_b):
                payload, trace = future.result()
                self.assertEqual(payload["ok"], True)
                self.assertEqual(trace["parse_status"], "ok")
        self.assertEqual(order[0], "blocker")
        self.assertEqual(order[1:], ["generation-1-a", "generation-1-b", "generation-2"])

    def test_success_path_writes_payload_and_memory_without_handoff_bundle(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(FAILURE_PROPOSAL_PAYLOAD),
                chat_response(FAILURE_REFLECTION_PAYLOAD),
                chat_response(FAILURE_PROPOSAL_PAYLOAD),
                chat_response(FAILURE_REFLECTION_PAYLOAD),
            ]
        )
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            artifact_path = write_discrete_artifacts(
                task_id="contains-duplicates",
                proposal_runtime=runtime,
                runs_root=tmp,
                branching_factor=1,
            )
            payload = json.loads(artifact_path.read_text())
            latest_payload = json.loads((tmp / "latest_run.json").read_text())
            run = payload["runs"][0]
            self.assertEqual(artifact_path.name, "codegen-contains-duplicates.json")
            self.assertTrue((tmp / "codegen-contains-duplicates.json").exists())
            self.assertEqual(latest_payload["summary"]["generated_at"], payload["summary"]["generated_at"])
            self.assertEqual(latest_payload["runs"][0]["task"]["id"], "contains-duplicates")
            self.assertEqual(payload["run_mode"], "llm-required")
            self.assertEqual(payload["summary"]["active_model"], "deepseek-chat")
            self.assertTrue((tmp / "codegen_working_memory.md").exists())
            self.assertNotIn("handoff_bundle", run)
            self.assertFalse((tmp / "handoff").exists())
            self.assertTrue(run["session_id"])
            self.assertTrue(run["generated_at"])
            self.assertEqual(payload["summary"]["write_backs"], 0)
            self.assertEqual(payload["summary"]["experiment_write_backs"], 2)
            self.assertEqual(run["memory_before_count"], 2)
            self.assertEqual(run["memory_after_count"], 4)
            self.assertEqual(run["positive_experiences_added"], 1)
            self.assertEqual(run["negative_experiences_added"], 1)
            self.assertEqual(len(run["added_experiences"]), 2)
            self.assertEqual(run["added_experiences"][0]["generation"], 1)
            self.assertEqual(run["added_experiences"][0]["experience_outcome"], "success")
            memories = json.loads((tmp / "codegen_working_memory.json").read_text())
            task_memories = [item for item in memories if item.get("source_task") == "contains-duplicates"]
            self.assertTrue(any(item.get("experience_outcome") == "success" for item in task_memories))
            self.assertTrue(any(item.get("experience_outcome") == "failure" for item in task_memories))
            self.assertIn("failure_memories:", (tmp / "codegen_working_memory.md").read_text())

    def test_timeout_failure_aborts_run(self) -> None:
        runtime = make_runtime([TimeoutError("timed out")] * 3)
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmTransportError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_invalid_http_json_aborts_run(self) -> None:
        runtime = make_runtime(["not-json"] * 3)
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_reflection_failure_is_skipped_for_robustness(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response({"failure_pattern": "missing most fields"}),
            ]
        )
        with patch_runner_fixture_catalog(), tempfile.TemporaryDirectory() as tmp_dir:
            payload = generate_discrete_payload(
                task_id="contains-duplicates",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
                generation_budget=1,
                branching_factor=1,
            )
        run = payload["runs"][0]
        self.assertEqual(run["winner"]["metrics"]["status"], "pass")
        self.assertEqual(run["positive_experiences_added"], 0)
        self.assertEqual(run["negative_experiences_added"], 0)
        self.assertTrue(any(trace.get("phase") == "memory_reflection_failed" for trace in run["llm_traces"]))

    def test_non_improving_passing_candidate_skips_failure_memory_writeback(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(NON_IMPROVING_PASS_PROPOSAL_PAYLOAD),
            ]
        )
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 2
        task["branching_factor"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())

            with patch(
                "app.codegen.trainer._select_branch_parents",
                side_effect=lambda _frontier, current_best, _accepted_history, _branching_factor: [current_best],
            ):
                result = run_codegen_task(
                    task,
                    store,
                    proposal_runtime=runtime,
                    workspace_root=tmp / "workspace",
                    session_id="non-improving-pass",
                )

            self.assertEqual(len(result["memory_events"]), 1)
            self.assertEqual(result["positive_experiences_added"], 1)
            self.assertEqual(result["negative_experiences_added"], 0)
            self.assertFalse(result["generations"][1]["wrote_memory"])
            self.assertEqual(result["generations"][1]["experience_outcome"], "failure")
            self.assertEqual(len(result["llm_traces"]), 3)

    def test_full_editable_file_payloads_are_accepted(self) -> None:
        runtime = make_runtime(
            [
                chat_response(FULL_FILE_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 1
        task["branching_factor"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            result = run_codegen_task(
                task,
                store,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                session_id="wrapped-function",
            )
            self.assertEqual(result["winner"]["metrics"]["status"], "pass")
            self.assertTrue(result["generations"][0]["winner_accepted"])

    def test_planbench_task_accepts_valid_planner_candidate(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PLANBENCH_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = _fixture_task("planbench-lite")
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                memory_root=tmp / "item-memory",
                session_id="planbench-success",
                max_items=1,
            )
            self.assertEqual(result["dataset_summary"]["total_items"], 1)
            self.assertEqual(result["dataset_summary"]["winner_passed"], 1)
            self.assertEqual(result["winner"]["metrics"]["status"], "pass")
            self.assertGreater(result["winner"]["metrics"]["objective"], 0.0)
            self.assertEqual(len(result["item_runs"]), 1)
            self.assertEqual(result["item_runs"][0]["winner"]["metrics"]["status"], "pass")

    def test_model_content_parse_error_is_retried(self) -> None:
        runtime = make_runtime(
            [
                raw_content_response("not valid json content"),
                chat_response(FULL_FILE_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 1
        task["branching_factor"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            result = run_codegen_task(
                task,
                store,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                session_id="retry-parse-error",
            )
            self.assertEqual(result["winner"]["metrics"]["status"], "pass")
            self.assertEqual(result["llm_traces"][0]["attempt"], 2)

    def test_model_transport_error_is_retried(self) -> None:
        runtime = make_runtime(
            [
                LlmTransportError("Model request timed out.", model="deepseek-chat"),
                chat_response(FULL_FILE_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 1
        task["branching_factor"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            events: list[dict[str, object]] = []
            result = run_codegen_task(
                task,
                store,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                session_id="retry-transport-error",
                progress_callback=events.append,
            )
            self.assertEqual(result["winner"]["metrics"]["status"], "pass")
            self.assertEqual(result["llm_traces"][0]["attempt"], 2)
            retry_events = [event for event in events if event.get("phase") == "llm_retry"]
            self.assertEqual(len(retry_events), 1)
            self.assertEqual(retry_events[0]["retry_attempt"], 2)
            self.assertEqual(retry_events[0]["max_attempts"], 3)
            self.assertEqual(retry_events[0]["branch_id"], "g1-b1")

    def test_branching_factor_triggers_parallel_proposals_and_branch_events(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(PARALLEL_PROPOSAL_PAYLOAD),
                chat_response(PARALLEL_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 2
        task["branching_factor"] = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            events: list[dict[str, object]] = []
            result = run_codegen_task(
                task,
                store,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                session_id="parallel-branches",
                progress_callback=events.append,
            )
            second_generation = result["generations"][1]
            self.assertEqual(len(second_generation["branches"]), 2)
            self.assertTrue(all(branch["branch_id"].startswith("g2-b") for branch in second_generation["branches"]))
            self.assertIn("objective_score", result["winner"]["metrics"])
            self.assertTrue(any(event.get("branch_id") == "g2-b1" for event in events))
            self.assertTrue(any(event.get("branch_id") == "g2-b2" for event in events))

    def test_branch_transport_error_does_not_abort_other_parallel_branches(self) -> None:
        runtime = make_runtime([])
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 2
        task["candidate_budget"] = 1
        task["branching_factor"] = 2

        candidate_specs = [
            {
                "agent": "candidate-1",
                "label": "Full editable file",
                "strategy": "Return the full editable file for the fixture task.",
                "rationale": "A valid branch should still complete even if a sibling branch fails in transport.",
                "imports": [],
                "file_body": FULL_FILE_PROPOSAL_PAYLOAD["candidates"][0]["file_body"],
                "candidate_summary": "Set-cardinality duplicate detector returned as a full editable file.",
                "run_mode": "llm-required",
                "proposal_model": "deepseek-chat",
            }
        ]
        proposal_lock = threading.Lock()
        proposal_calls = 0

        def flaky_proposals(*_args, **_kwargs):
            nonlocal proposal_calls
            with proposal_lock:
                proposal_calls += 1
                call_number = proposal_calls
            if call_number == 2:
                raise LlmTransportError("Model request timed out.", model="deepseek-chat")
            return candidate_specs, {"selected_model": "deepseek-chat", "parse_status": "ok", "attempt": 1}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            with patch("app.codegen.trainer.propose_code_candidates", side_effect=flaky_proposals), patch(
                "app.codegen.trainer.reflect_strategy_experience",
                return_value=(REFLECTION_PAYLOAD, {"selected_model": "deepseek-chat", "attempt": 1}),
            ):
                result = run_codegen_task(
                    task,
                    store,
                    proposal_runtime=runtime,
                    workspace_root=tmp / "workspace",
                    session_id="branch-transport-isolated",
                )
        self.assertEqual(result["winner"]["metrics"]["status"], "pass")
        self.assertEqual(len(result["generations"][1]["branches"]), 2)
        self.assertTrue(any(branch["winner"]["metrics"]["status"] == "error" for branch in result["generations"][1]["branches"]))
        self.assertTrue(any(trace.get("phase") == "proposal_generation_failed" for trace in result["llm_traces"]))

    def test_reflection_transport_error_does_not_abort_verified_candidate(self) -> None:
        runtime = make_runtime([chat_response(FULL_FILE_PROPOSAL_PAYLOAD)])
        task = _fixture_task("contains-duplicates")
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            with patch(
                "app.codegen.trainer.reflect_strategy_experience",
                side_effect=LlmTransportError("Reflection request timed out.", model="deepseek-chat"),
            ):
                result = run_codegen_task(
                    task,
                    store,
                    proposal_runtime=runtime,
                    workspace_root=tmp / "workspace",
                    session_id="reflection-transport-isolated",
                )
        self.assertEqual(result["winner"]["metrics"]["status"], "pass")
        self.assertEqual(result["positive_experiences_added"], 0)
        self.assertEqual(result["negative_experiences_added"], 0)
        self.assertTrue(any(trace.get("phase") == "memory_reflection_failed" for trace in result["llm_traces"]))

    def test_min_objective_uses_normalized_objective_score_for_selection(self) -> None:
        baseline_metrics = {
            "status": "pass",
            "verifier_status": "pass",
            "objective": 10.0,
            "objective_score": -10.0,
            "primary_score": -10.0,
            "tie_break_score": 0.0,
            "gate_passed": True,
            "benchmark_ms": 10.0,
            "passed_tests": 1,
            "total_tests": 1,
            "speedup_vs_baseline": 1.0,
        }
        candidate_metrics = {
            "status": "pass",
            "verifier_status": "pass",
            "objective": 6.0,
            "objective_score": -6.0,
            "primary_score": -6.0,
            "tie_break_score": 0.0,
            "gate_passed": True,
            "benchmark_ms": 6.0,
            "passed_tests": 1,
            "total_tests": 1,
            "speedup_vs_baseline": 1.0,
        }
        runtime = make_runtime(
            [
                chat_response(
                    {
                        "candidates": [
                            {
                                "name": "Lower latency path",
                                "strategy": "Return a full file candidate for the synthetic min-objective task.",
                                "rationale": "Selection should rely on normalized objective_score for min-direction tasks.",
                                "imports": [],
                                "file_body": _file_with_entry("solve", "values", "return values[0]"),
                                "candidate_summary": "Synthetic min-objective candidate.",
                            }
                        ]
                    }
                ),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            editable_path = tmp / "editable.py"
            editable_path.write_text(_file_with_entry("solve", "values", "return values[0]"))
            task = {
                "id": "synthetic-min-task",
                "title": "Synthetic min task",
                "description": "Select lower objective values by normalized score.",
                "family": "numeric",
                "function_name": "solve",
                "entry_symbol": "solve",
                "editable_file": "editable.py",
                "editable_filename": "editable.py",
                "editable_path": str(editable_path),
                "verifier_path": str(editable_path),
                "answer_metric": "latency_ms",
                "benchmark_tier": "comparable",
                "track": "synthetic",
                "dataset_id": "synthetic_min_v1",
                "included_in_main_comparison": True,
                "objective_label": "latency_ms",
                "objective_direction": "min",
                "objective_spec": {
                    "display_name": "Latency",
                    "direction": "min",
                    "unit": "ms",
                    "summary_template": "Lower latency is better.",
                    "formula": "latency_ms = elapsed_ms",
                },
                "task_signature": ["python-codegen", "synthetic", "min-objective"],
                "runtime_backend": "dataset",
                "task_mode": "answer",
                "optimization_scope": "implementation",
                "generation_budget": 1,
                "candidate_budget": 1,
                "branching_factor": 1,
                "epsilon": 0.0,
                "baseline_summary": "Baseline synthetic solver.",
            }
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            with (
                patch("app.codegen.trainer.evaluate_materialized_candidate", side_effect=[baseline_metrics, candidate_metrics]),
                patch("app.codegen.trainer.materialize_candidate", return_value=(tmp / "candidate.py", _file_with_entry("solve", "values", "return values[0]"))),
            ):
                result = run_codegen_task(
                    task,
                    store,
                    proposal_runtime=runtime,
                    workspace_root=tmp / "workspace",
                    session_id="min-objective",
                )
            self.assertEqual(result["winner"]["metrics"]["objective"], 6.0)
            self.assertEqual(result["winner"]["metrics"]["objective_score"], -6.0)
            self.assertTrue(result["generations"][0]["winner_accepted"])

    def test_skipping_baseline_verifier_still_accepts_the_first_passing_candidate(self) -> None:
        candidate_metrics = {
            "status": "pass",
            "verifier_status": "pass",
            "objective": 0.0,
            "objective_score": 0.0,
            "primary_score": 0.0,
            "tie_break_score": 0.0,
            "gate_passed": True,
            "benchmark_ms": 1.0,
            "passed_tests": 1,
            "total_tests": 1,
        }
        runtime = make_runtime(
            [
                chat_response(
                    {
                        "candidates": [
                            {
                                "name": "Reference-only baseline skip candidate",
                                "strategy": "Return a full file candidate after skipping the checked-in baseline verifier.",
                                "rationale": "Heavy verifier tasks should still promote the first verified candidate even when the baseline is reference-only.",
                                "imports": [],
                                "file_body": _file_with_entry("solve", "values", "return values[0]"),
                                "candidate_summary": "Synthetic candidate accepted against a reference-only baseline.",
                            }
                        ]
                    }
                ),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            editable_path = tmp / "editable.py"
            editable_path.write_text(_file_with_entry("solve", "values", "return values[0]"))
            task = {
                "id": "synthetic-reference-only-baseline",
                "title": "Synthetic reference-only baseline",
                "description": "Skip the baseline verifier but still accept the first passing candidate.",
                "family": "numeric",
                "function_name": "solve",
                "entry_symbol": "solve",
                "editable_file": "editable.py",
                "editable_filename": "editable.py",
                "editable_path": str(editable_path),
                "verifier_path": str(editable_path),
                "answer_metric": "score",
                "benchmark_tier": "experiment",
                "track": "synthetic",
                "dataset_id": "synthetic_reference_only_v1",
                "included_in_main_comparison": False,
                "objective_label": "score",
                "objective_direction": "max",
                "objective_spec": {
                    "display_name": "Score",
                    "direction": "max",
                    "unit": "score",
                    "summary_template": "Higher is better.",
                    "formula": "score = objective",
                },
                "task_signature": ["python-codegen", "synthetic", "reference-only-baseline"],
                "runtime_backend": "dataset",
                "task_mode": "answer",
                "optimization_scope": "implementation",
                "generation_budget": 1,
                "candidate_budget": 1,
                "branching_factor": 1,
                "epsilon": 0.0,
                "baseline_summary": "Baseline synthetic solver.",
                "run_baseline_verifier": False,
            }
            store = MemoryStore(tmp / "memory.json", markdown_path=tmp / "memory.md")
            store.ensure_seed_records(seed_strategy_experiences())
            with patch("app.codegen.trainer.evaluate_materialized_candidate", return_value=candidate_metrics) as evaluate_candidate:
                result = run_codegen_task(
                    task,
                    store,
                    proposal_runtime=runtime,
                    workspace_root=tmp / "workspace",
                    session_id="reference-only-baseline",
                )

            evaluate_candidate.assert_called_once()
            self.assertEqual(result["baseline"]["metrics"]["verifier_status"], "not-run")
            self.assertEqual(result["winner"]["candidate_id"], "synthetic-reference-only-baseline-g1-b1-c1")
            self.assertTrue(result["generations"][0]["winner_accepted"])

    def test_full_sequence_only_runs_comparable_tasks(self) -> None:
        comparable_tasks = [dict(task) for task in load_codegen_tasks(included_in_main_comparison=True)]
        for task in comparable_tasks:
            task["generation_budget"] = 1
            task["candidate_budget"] = 1
            task["branching_factor"] = 1
            task["item_workers"] = 1
        runtime = make_runtime([chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD) for _ in comparable_tasks])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for task in comparable_tasks:
                if task["id"] == "livecodebench":
                    manifest_path = tmp / "livecodebench-questions.json"
                    manifest_item = {
                        "item_id": "livecodebench-test-item",
                        "name": "livecodebench-test-item",
                        "prompt": "Read two integers and print their sum.",
                        "context": "Synthetic LiveCodeBench stdin sample.",
                        "expected_answer": "Pass all public and private tests.",
                        "metadata": {
                            "problem_file": str(ROOT / "tests" / "fixtures" / "livecodebench" / "problems" / "stdin_problem.json"),
                            "platform": "atcoder",
                            "evaluation_mode": "stdin",
                        },
                    }
                elif task.get("lazy_item_manifest"):
                    manifest_path = tmp / f"{task['id']}-questions.json"
                    manifest_item = {
                        "item_id": f"{task['id']}-test-item",
                        "name": f"{task['id']}-test-item",
                        "prompt": f"Synthetic prompt for {task['id']}.",
                        "context": "Synthetic dataset sample.",
                        "choices": ["alpha", "beta", "gamma", "delta"],
                        "expected_answer": "alpha",
                        "metadata": {
                            "correct_choice_index": 0,
                            "answer_aliases": ["alpha"],
                        },
                    }
                else:
                    continue
                manifest_path.write_text(json.dumps({"items": [manifest_item]}, indent=2))
                task["item_manifest_path"] = str(manifest_path)
                task["dataset_size"] = 1
                task["lazy_item_manifest"] = False
            with (
                patch("app.entries.runner.load_codegen_tasks", return_value=comparable_tasks),
                patch(
                    "app.codegen.trainer.reflect_strategy_experience",
                    return_value=(REFLECTION_PAYLOAD, {"selected_model": "deepseek-chat", "attempt": 1}),
                ),
            ):
                payload = generate_discrete_payload(
                    proposal_runtime=runtime,
                    runs_root=tmp,
                    generation_budget=1,
                    candidate_budget=1,
                    branching_factor=1,
                    max_items=1,
                )
        self.assertEqual(payload["summary"]["num_tasks"], len(comparable_tasks))
        self.assertEqual(payload["summary"]["total_runs"], len(comparable_tasks))
        self.assertEqual(payload["summary"]["experiment_runs"], 0)
        self.assertTrue(payload["runs"])
        self.assertTrue(all(run["included_in_main_comparison"] for run in payload["runs"]))

    def test_dataset_task_fanout_runs_each_question_independently(self) -> None:
        runtime = make_runtime(
            [
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "olymmath")
        task = dict(task)
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items = [
                {
                    "item_id": "olymmath-r1",
                    "raw_item_id": "olymmath-r1",
                    "id": "olymmath-r1",
                    "question_id": "olymmath-r1",
                    "name": "Remainders mod 2 through 6",
                    "prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5, and 6, and leaves remainder 0 when divided by 7?",
                    "raw_prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5, and 6, and leaves remainder 0 when divided by 7?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "301",
                    "raw_expected_answer": "301",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric"},
                },
                {
                    "item_id": "olymmath-triangle",
                    "raw_item_id": "olymmath-triangle",
                    "id": "olymmath-triangle",
                    "question_id": "olymmath-triangle",
                    "name": "Triangle sides 13, 14, 15",
                    "prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "raw_prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "84",
                    "raw_expected_answer": "84",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric"},
                },
            ]
            manifest = tmp / "questions.json"
            manifest.write_text(json.dumps(items, indent=2))
            task["item_manifest_path"] = str(manifest)
            task["dataset_size"] = len(items)
            events: list[dict[str, object]] = []
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                memory_root=tmp / "item-memory",
                session_id="dataset-fanout",
                progress_callback=events.append,
            )
            self.assertEqual(result["dataset_summary"]["total_items"], 2)
            self.assertEqual(result["dataset_summary"]["winner_passed"], 2)
            self.assertEqual(len(result["item_runs"]), 2)
            self.assertEqual({item_run["item_id"] for item_run in result["item_runs"]}, {item["item_id"] for item in items})
            self.assertTrue(all(item_run["winner"]["metrics"]["status"] == "pass" for item_run in result["item_runs"]))
            self.assertTrue(all(item_run["memory_before_count"] == 2 for item_run in result["item_runs"]))
            self.assertTrue(any(event.get("item_id") == items[0]["item_id"] for event in events))
            self.assertTrue(any(event.get("item_id") == items[1]["item_id"] for event in events))
            self.assertTrue(any(event.get("item_brief") for event in events if event.get("item_id") == items[0]["item_id"]))
            self.assertTrue(any(event.get("expected_answer") == items[0]["expected_answer"] for event in events if event.get("item_id") == items[0]["item_id"]))

    def test_dataset_task_isolates_single_item_transport_failure(self) -> None:
        def transport(request_body, _config):
            prompt = str(request_body["messages"][1]["content"])
            if "olymmath-r1" in prompt:
                raise LlmTransportError("Model request timed out.", model="deepseek-chat")
            if "Outcome:" in prompt:
                return chat_response(REFLECTION_PAYLOAD)
            return chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD)

        runtime = ProposalRuntime(
            RuntimeConfig(
                api_key="test-key",
                api_base="https://api.test/v1",
                primary_model="deepseek-chat",
                available_models=("deepseek-chat",),
                temperature=0.2,
                max_tokens=4096,
                timeout_s=45,
                llm_concurrency=2,
            ),
            transport=transport,
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "olymmath")
        task = dict(task)
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items = [
                {
                    "item_id": "olymmath-r1",
                    "raw_item_id": "olymmath-r1",
                    "id": "olymmath-r1",
                    "question_id": "olymmath-r1",
                    "name": "Remainders mod 2 through 6",
                    "prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5, and 6, and leaves remainder 0 when divided by 7?",
                    "raw_prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5, and 6, and leaves remainder 0 when divided by 7?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "301",
                    "raw_expected_answer": "301",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric"},
                },
                {
                    "item_id": "olymmath-r2",
                    "raw_item_id": "olymmath-r2",
                    "id": "olymmath-r2",
                    "question_id": "olymmath-r2",
                    "name": "Triangle sides 13, 14, 15",
                    "prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "raw_prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "84",
                    "raw_expected_answer": "84",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric"},
                },
            ]
            manifest = tmp / "questions.json"
            manifest.write_text(json.dumps(items, indent=2))
            task["item_manifest_path"] = str(manifest)
            task["dataset_size"] = len(items)
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                memory_root=tmp / "item-memory",
                session_id="dataset-item-transport-isolated",
            )
        self.assertEqual(result["dataset_summary"]["total_items"], 2)
        self.assertEqual(result["dataset_summary"]["winner_passed"], 1)
        self.assertEqual(result["dataset_summary"]["failure_count"], 1)
        item_runs = {item_run["item_id"]: item_run for item_run in result["item_runs"]}
        self.assertEqual(item_runs["olymmath-r2"]["winner"]["metrics"]["status"], "pass")
        self.assertEqual(item_runs["olymmath-r1"]["winner"]["metrics"]["status"], "error")
        self.assertEqual(item_runs["olymmath-r1"]["error_payload"]["error_type"], "llm_transport_error")

    def test_dataset_task_can_run_selected_item_ids_only(self) -> None:
        runtime = make_runtime(
            [
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "olymmath")
        task = dict(task)
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items = [
                {
                    "item_id": "olymmath-r1",
                    "raw_item_id": "olymmath-r1",
                    "id": "olymmath-r1",
                    "question_id": "olymmath-r1",
                    "name": "Remainders mod 2 through 6",
                    "prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2 through 6 and 0 mod 7?",
                    "raw_prompt": "What is the smallest positive integer that leaves remainder 1 when divided by 2 through 6 and 0 mod 7?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "301",
                    "raw_expected_answer": "301",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric", "source_id": "1", "source_index": 0},
                },
                {
                    "item_id": "olymmath-r2",
                    "raw_item_id": "olymmath-r2",
                    "id": "olymmath-r2",
                    "question_id": "olymmath-r2",
                    "name": "Triangle area",
                    "prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "raw_prompt": "A triangle has side lengths 13, 14, and 15. What is its area?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "84",
                    "raw_expected_answer": "84",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric", "source_id": "2", "source_index": 1},
                },
                {
                    "item_id": "olymmath-r3",
                    "raw_item_id": "olymmath-r3",
                    "id": "olymmath-r3",
                    "question_id": "olymmath-r3",
                    "name": "Square count",
                    "prompt": "How many squares are on a standard chessboard?",
                    "raw_prompt": "How many squares are on a standard chessboard?",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "204",
                    "raw_expected_answer": "204",
                    "metadata": {"dataset": "olymmath", "answer_format": "numeric", "source_id": "3", "source_index": 2},
                },
            ]
            manifest = tmp / "questions.json"
            manifest.write_text(json.dumps(items, indent=2))
            task["item_manifest_path"] = str(manifest)
            task["dataset_size"] = len(items)
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                memory_root=tmp / "item-memory",
                session_id="dataset-selected-items",
                selected_item_ids=["2", "olymmath-r1"],
            )
            self.assertEqual(result["dataset_summary"]["total_items"], 2)
            self.assertEqual({item_run["item_id"] for item_run in result["item_runs"]}, {"olymmath-r1", "olymmath-r2"})

    def test_numeric_item_selection_is_1_based_from_first_question(self) -> None:
        runtime = make_runtime(
            [
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "planbench")
        task = dict(task)
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items = [
                {
                    "item_id": "planbench-00002",
                    "raw_item_id": "planbench-00002",
                    "id": "planbench-00002",
                    "question_id": "planbench-00002",
                    "name": "obfuscated_deceptive_logistics / oneshot / 2",
                    "prompt": "Synthetic planning prompt 2.",
                    "raw_prompt": "Synthetic planning prompt 2.",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "move a b",
                    "raw_expected_answer": "move a b",
                    "metadata": {"dataset": "planbench", "answer_format": "plan", "domain": "obfuscated_deceptive_logistics"},
                },
                {
                    "item_id": "planbench-00003",
                    "raw_item_id": "planbench-00003",
                    "id": "planbench-00003",
                    "question_id": "planbench-00003",
                    "name": "obfuscated_deceptive_logistics / oneshot / 3",
                    "prompt": "Synthetic planning prompt 3.",
                    "raw_prompt": "Synthetic planning prompt 3.",
                    "context": None,
                    "raw_context": None,
                    "choices": [],
                    "raw_choices": [],
                    "expected_answer": "move b c",
                    "raw_expected_answer": "move b c",
                    "metadata": {"dataset": "planbench", "answer_format": "plan", "domain": "obfuscated_deceptive_logistics"},
                },
            ]
            manifest = tmp / "questions.json"
            manifest.write_text(json.dumps(items, indent=2))
            task["item_manifest_path"] = str(manifest)
            task["dataset_size"] = len(items)
            result = run_dataset_task(
                task,
                proposal_runtime=runtime,
                workspace_root=tmp / "workspace",
                memory_root=tmp / "item-memory",
                session_id="dataset-first-question-indexing",
                selected_item_ids=["1"],
            )
            self.assertEqual(result["dataset_summary"]["total_items"], 1)
            self.assertEqual(result["item_runs"][0]["item_id"], "planbench-00002")

    def test_dataset_runs_stay_inline_without_handoff_artifacts(self) -> None:
        runtime = make_runtime(
            [
                chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
            ]
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "sciq")
        task = dict(task)
        task["generation_budget"] = 1
        task["candidate_budget"] = 1
        task["branching_factor"] = 1
        task["item_workers"] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items = load_question_manifest(task)[:2]
            manifest = tmp / "questions.json"
            manifest.write_text(json.dumps(items, indent=2))
            task["item_manifest_path"] = str(manifest)
            task["dataset_size"] = len(items)
            with patch("app.entries.runner.load_codegen_tasks", return_value=[task]):
                artifact_path = write_discrete_artifacts(
                    task_id="sciq",
                    proposal_runtime=runtime,
                    runs_root=tmp,
                    generation_budget=1,
                    candidate_budget=1,
                    branching_factor=1,
                    max_items=1,
                )
            payload = json.loads(artifact_path.read_text())
            run = payload["runs"][0]
            self.assertEqual(run["dataset_summary"]["total_items"], 1)
            self.assertEqual(len(run["item_runs"]), 1)
            self.assertNotIn("handoff_bundle", run)
            self.assertFalse((Path(tmp) / "handoff").exists())
            item_run = run["item_runs"][0]
            self.assertEqual(item_run["question"]["id"], items[0]["item_id"])
            self.assertEqual(item_run["question"]["question_id"], items[0]["item_id"])
            self.assertEqual(item_run["question"]["raw_prompt"], items[0]["raw_prompt"])

    def test_external_item_files_can_supply_large_context_without_bloating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            items_dir = tmp / "items"
            items_dir.mkdir(parents=True, exist_ok=True)
            long_context = "context-line-" * 400
            (items_dir / "item-1.json").write_text(json.dumps({"context": long_context}))
            manifest = tmp / "questions.json"
            manifest.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "longbench-v2-0001",
                                "name": "LongBench synthetic 1",
                                "prompt": "Which choice is correct?",
                                "choices": ["alpha", "beta", "gamma", "delta"],
                                "expected_answer": "beta",
                                "item_file": "items/item-1.json",
                                "metadata": {
                                    "dataset": "longbench-v2",
                                    "correct_choice_index": 1,
                                    "answer_aliases": ["beta"],
                                },
                            }
                        ]
                    }
                )
            )
            task = {
                "id": "longbench-v2",
                "title": "LongBench v2",
                "track": "longcontext_verified",
                "item_manifest_path": str(manifest),
                "prompt_context": "Edit editable.py only.",
                "prompt_context_max_chars": 64,
                "result_context_max_chars": 48,
            }
            items = load_question_manifest(task)
            self.assertEqual(items[0]["raw_context"], long_context)
            micro_task = build_micro_task(task, items[0])
            self.assertIn("full context is still available to solve(question) at runtime", micro_task["prompt_context"])
            question = _question_payload_for_result(task, items[0])
            self.assertIsInstance(question["raw_context"], str)
            self.assertIn("truncated from", question["raw_context"])
            self.assertLess(len(question["raw_context"]), len(long_context))


if __name__ == "__main__":
    unittest.main()
