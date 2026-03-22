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
from app.codegen.dataset_runner import run_dataset_task
from app.codegen.dataset_support import load_question_manifest
from app.codegen.llm import _proposal_prompt
from app.codegen.config import RuntimeConfig
from app.codegen.llm import ProposalRuntime
from app.codegen.trainer import run_codegen_task
from app.entries.discrete_demo import generate_discrete_payload, write_discrete_artifacts
from app.memory.store import MemoryStore
from tests.helpers import chat_response, make_runtime


def _file_with_entry(symbol: str, args: str, body: str) -> str:
    return f"def {symbol}({args}):\n{textwrap.indent(body.strip(), '    ')}\n"


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
            "strategy": "Return the entire editable file as required by the single-file benchmark contract.",
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
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                generate_discrete_payload(task_id="contains-duplicates", runs_root=Path(tmp_dir), env_root=Path(tmp_dir))

    def test_invalid_llm_output_fails_immediately(self) -> None:
        runtime = make_runtime([chat_response({"candidates": [{"name": "bad"}]})])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_proposal_prompt_requests_exact_candidate_budget_and_json_only_output(self) -> None:
        task = dict(next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates"))
        task["candidate_budget"] = 1
        candidate = {
            "candidate_summary": "Checked-in baseline.",
            "metrics": {"objective": 0.0, "objective_score": 0.0, "J": 0.42},
            "baseline_source": "def contains_duplicates(values):\n    return False\n",
            "source_code": "def contains_duplicates(values):\n    return False\n",
        }
        system_prompt, _ = _proposal_prompt(
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

    def test_truncated_llm_output_surfaces_parse_details(self) -> None:
        truncated_response = raw_content_response(
            "```json\n{\"candidates\": [{\"name\": \"cut off\"",
            completion_tokens=4096,
        )
        runtime = make_runtime([truncated_response, truncated_response, truncated_response])
        with tempfile.TemporaryDirectory() as tmp_dir:
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = write_discrete_artifacts(
                task_id="contains-duplicates",
                proposal_runtime=runtime,
                runs_root=Path(tmp_dir),
                branching_factor=1,
            )
            payload = json.loads(artifact_path.read_text())
            run = payload["runs"][0]
            self.assertEqual(payload["run_mode"], "llm-required")
            self.assertEqual(payload["summary"]["active_model"], "deepseek-chat")
            self.assertTrue((Path(tmp_dir) / "codegen_working_memory.md").exists())
            self.assertNotIn("handoff_bundle", run)
            self.assertFalse((Path(tmp_dir) / "handoff").exists())
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
            memories = json.loads((Path(tmp_dir) / "codegen_working_memory.json").read_text())
            task_memories = [item for item in memories if item.get("source_task") == "contains-duplicates"]
            self.assertTrue(any(item.get("experience_outcome") == "success" for item in task_memories))
            self.assertTrue(any(item.get("experience_outcome") == "failure" for item in task_memories))
            self.assertIn("failure_memories:", (Path(tmp_dir) / "codegen_working_memory.md").read_text())

    def test_timeout_failure_aborts_run(self) -> None:
        runtime = make_runtime([TimeoutError("timed out")] * 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmTransportError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_invalid_http_json_aborts_run(self) -> None:
        runtime = make_runtime(["not-json"] * 3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_reflection_failure_aborts_run(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response({"failure_pattern": "missing most fields"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LlmResponseError):
                generate_discrete_payload(
                    task_id="contains-duplicates",
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
                    branching_factor=1,
                )

    def test_non_improving_passing_candidate_skips_failure_memory_writeback(self) -> None:
        runtime = make_runtime(
            [
                chat_response(PROPOSAL_PAYLOAD),
                chat_response(REFLECTION_PAYLOAD),
                chat_response(NON_IMPROVING_PASS_PROPOSAL_PAYLOAD),
            ]
        )
        task = next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates")
        task = dict(task)
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
        task = next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates")
        task = dict(task)
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
        task = next(item for item in load_codegen_tasks() if item["id"] == "planbench-lite")
        task = dict(task)
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
        task = next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates")
        task = dict(task)
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
        task = next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates")
        task = dict(task)
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
                session_id="retry-transport-error",
            )
            self.assertEqual(result["winner"]["metrics"]["status"], "pass")
            self.assertEqual(result["llm_traces"][0]["attempt"], 2)

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
        task = next(item for item in load_codegen_tasks() if item["id"] == "contains-duplicates")
        task = dict(task)
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

    def test_min_objective_uses_normalized_objective_score_for_selection(self) -> None:
        baseline_metrics = {
            "status": "pass",
            "verifier_status": "pass",
            "objective": 10.0,
            "objective_score": -10.0,
            "J": 1.0,
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
            "J": 1.2,
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
                "source_type": "benchmark-task",
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

    def test_full_sequence_only_runs_comparable_tasks(self) -> None:
        comparable_tasks = [dict(task) for task in load_codegen_tasks(included_in_main_comparison=True)]
        for task in comparable_tasks:
            task["generation_budget"] = 1
            task["candidate_budget"] = 1
            task["branching_factor"] = 1
            task["item_workers"] = 1
        runtime = make_runtime([chat_response(QUESTION_SOLVER_PROPOSAL_PAYLOAD) for _ in comparable_tasks])
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch("app.entries.discrete_demo.load_codegen_tasks", return_value=comparable_tasks),
                patch(
                    "app.codegen.trainer.reflect_strategy_experience",
                    return_value=(REFLECTION_PAYLOAD, {"selected_model": "deepseek-chat", "attempt": 1}),
                ),
            ):
                payload = generate_discrete_payload(
                    proposal_runtime=runtime,
                    runs_root=Path(tmp_dir),
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
            with patch("app.entries.discrete_demo.load_codegen_tasks", return_value=[task]):
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


if __name__ == "__main__":
    unittest.main()
