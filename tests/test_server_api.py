from __future__ import annotations

import json
import threading
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from app.codegen.errors import LlmResponseError, LlmTransportError
from app.entries import server
from tests.helpers import make_runtime


def _fetch_json(url: str, *, method: str = "GET") -> tuple[int, dict]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class ServerApiTest(unittest.TestCase):
    def _serve(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DemoHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, thread

    def test_latest_run_reads_cached_payload_without_starting_a_run(self) -> None:
        cached_payload = {"summary": {"generated_at": "cached"}, "runs": [], "task_catalog": []}
        with (
            patch.object(server, "load_cached_discrete_payload", return_value=cached_payload) as load_cached,
            patch.object(server, "write_discrete_artifacts", side_effect=AssertionError("should not run")),
        ):
            httpd, thread = self._serve()
            try:
                status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/latest-run")
                self.assertEqual(status, 200)
                self.assertEqual(payload["summary"]["generated_at"], "cached")
                load_cached.assert_called_once()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_async_job_surfaces_terminal_failure(self) -> None:
        with patch.object(server, "write_discrete_artifacts", side_effect=LlmTransportError("boom", model="deepseek-chat")):
            httpd, thread = self._serve()
            try:
                status, start_payload = _fetch_json(
                    f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=contains-duplicates",
                    method="POST",
                )
                self.assertEqual(status, 202)
                job_id = start_payload["job_id"]
                deadline = time.time() + 5
                payload = {}
                while time.time() < deadline:
                    _, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                    if payload["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(payload["status"], "failed")
                self.assertTrue(payload["terminal"])
                self.assertEqual(payload["error_type"], "llm_transport_error")
                self.assertEqual(payload["model"], "deepseek-chat")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_runtime_endpoint_reports_available_models(self) -> None:
        with patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])):
            httpd, thread = self._serve()
            try:
                status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/runtime")
                self.assertEqual(status, 200)
                self.assertEqual(payload["primary_model"], "deepseek-chat")
                self.assertIn("glm-5", payload["available_models"])
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_tasks_endpoint_returns_benchmark_metadata(self) -> None:
        httpd, thread = self._serve()
        try:
            status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/tasks")
            self.assertEqual(status, 200)
            contains_duplicates = next(task for task in payload["tasks"] if task["id"] == "contains-duplicates")
            olymmath = next(task for task in payload["tasks"] if task["id"] == "olymmath")
            math_500 = next(task for task in payload["tasks"] if task["id"] == "math-500")
            amc = next(task for task in payload["tasks"] if task["id"] == "amc")
            planbench = next(task for task in payload["tasks"] if task["id"] == "planbench")
            sciq = next(task for task in payload["tasks"] if task["id"] == "sciq")
            qasc = next(task for task in payload["tasks"] if task["id"] == "qasc")
            scienceqa = next(task for task in payload["tasks"] if task["id"] == "scienceqa")
            planbench_lite = next(task for task in payload["tasks"] if task["id"] == "planbench-lite")
            self.assertEqual(contains_duplicates["benchmark_tier"], "experiment")
            self.assertEqual(contains_duplicates["track"], "small_experiments")
            self.assertEqual(contains_duplicates["dataset_id"], "contains-duplicates-v1")
            self.assertFalse(contains_duplicates["included_in_main_comparison"])
            self.assertEqual(olymmath["dataset_id"], "olymmath")
            self.assertEqual(olymmath["dataset_size"], 100)
            self.assertTrue(olymmath["local_dataset_only"])
            self.assertEqual(math_500["track"], "math_verified")
            self.assertEqual(math_500["split"], "test")
            self.assertEqual(amc["dataset_size"], 4)
            self.assertEqual(planbench["dataset_size"], 2270)
            self.assertTrue(planbench["local_dataset_only"])
            self.assertEqual(planbench["track"], "planning_verified")
            self.assertTrue(planbench["included_in_main_comparison"])
            self.assertEqual(sciq["track"], "science_verified")
            self.assertEqual(sciq["split"], "validation")
            self.assertEqual(sciq["dataset_size"], 1000)
            self.assertEqual(qasc["dataset_size"], 926)
            self.assertEqual(qasc["split"], "validation")
            self.assertEqual(scienceqa["dataset_size"], 768)
            self.assertEqual(scienceqa["split"], "validation:natural-science:text-only:biology-chemistry-physics")
            self.assertEqual(planbench_lite["dataset_size"], 4)
            self.assertEqual(planbench_lite["track"], "small_experiments")
            self.assertFalse(planbench_lite["included_in_main_comparison"])
            self.assertEqual([task["id"] for task in payload["tasks"][:4]], ["olymmath", "math-500", "aime", "amc"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_invalid_model_override_fails_fast(self) -> None:
        with patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])):
            httpd, thread = self._serve()
            try:
                status, payload = _fetch_json(
                    f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=contains-duplicates&model=unknown-model",
                    method="POST",
                )
                self.assertEqual(status, 400)
                self.assertTrue(payload["terminal"])
                self.assertEqual(payload["error_type"], "config_error")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_branching_factor_is_forwarded_to_job_runner(self) -> None:
        payload = {"summary": {"generated_at": "now"}, "runs": [], "task_catalog": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path) as write_artifacts,
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=contains-duplicates&branching_factor=6",
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["branching_factor"], 6)
                    self.assertEqual(write_artifacts.call_args.kwargs["branching_factor"], 6)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_generation_and_candidate_budgets_are_forwarded_to_job_runner(self) -> None:
        payload = {"summary": {"generated_at": "now"}, "runs": [], "task_catalog": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path) as write_artifacts,
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        (
                            f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                            "?task_id=contains-duplicates&generation_budget=10&candidate_budget=1"
                        ),
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    completed = {}
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["generation_budget"], 10)
                    self.assertEqual(completed["candidate_budget"], 1)
                    self.assertEqual(write_artifacts.call_args.kwargs["generation_budget"], 10)
                    self.assertEqual(write_artifacts.call_args.kwargs["candidate_budget"], 1)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_item_workers_is_forwarded_to_job_runner(self) -> None:
        payload = {"summary": {"generated_at": "now"}, "runs": [], "task_catalog": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path) as write_artifacts,
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        (
                            f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                            "?task_id=math-500&item_workers=50&max_items=50"
                        ),
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    completed = {}
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["item_workers"], 50)
                    self.assertEqual(write_artifacts.call_args.kwargs["item_workers"], 50)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_max_items_is_forwarded_to_job_runner(self) -> None:
        payload = {"summary": {"generated_at": "now"}, "runs": [], "task_catalog": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path) as write_artifacts,
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=math-500&max_items=100",
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["max_items"], 100)
                    self.assertEqual(write_artifacts.call_args.kwargs["max_items"], 100)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_run_sequence_forwards_generation_and_candidate_budgets(self) -> None:
        payload = {"summary": {"generated_at": "now"}, "runs": [], "task_catalog": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path) as write_artifacts,
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        (
                            f"http://127.0.0.1:{httpd.server_port}/api/run-sequence"
                            "?generation_budget=10&candidate_budget=1&branching_factor=4&max_items=50"
                        ),
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    completed = {}
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["generation_budget"], 10)
                    self.assertEqual(completed["candidate_budget"], 1)
                    self.assertEqual(completed["branching_factor"], 4)
                    self.assertEqual(completed["max_items"], 50)
                    self.assertEqual(write_artifacts.call_args.kwargs["generation_budget"], 10)
                    self.assertEqual(write_artifacts.call_args.kwargs["candidate_budget"], 1)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_invalid_generation_and_candidate_budgets_fail_fast(self) -> None:
        with patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])):
            httpd, thread = self._serve()
            try:
                status, _ = _fetch_json(
                    (
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                        "?task_id=contains-duplicates&generation_budget=0"
                    ),
                    method="POST",
                )
                self.assertEqual(status, 400)
                status, _ = _fetch_json(
                    (
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                        "?task_id=contains-duplicates&candidate_budget=bad"
                    ),
                    method="POST",
                )
                self.assertEqual(status, 400)
                status, _ = _fetch_json(
                    (
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                        "?task_id=olymmath&item_workers=0"
                    ),
                    method="POST",
                )
                self.assertEqual(status, 400)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_failed_job_preserves_llm_response_diagnostics(self) -> None:
        error = LlmResponseError(
            "Model response appears truncated before completing a valid JSON object.",
            model="deepseek-chat",
            details={
                "parse_status": "truncated",
                "completion_tokens": 4096,
                "max_tokens": 4096,
                "response_truncated": True,
            },
        )
        with patch.object(server, "write_discrete_artifacts", side_effect=error):
            httpd, thread = self._serve()
            try:
                status, start_payload = _fetch_json(
                    f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=contains-duplicates",
                    method="POST",
                )
                self.assertEqual(status, 202)
                job_id = start_payload["job_id"]
                deadline = time.time() + 5
                payload = {}
                while time.time() < deadline:
                    _, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                    if payload["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error_type"], "llm_response_error")
                self.assertIn("details", payload)
                self.assertEqual(payload["details"]["parse_status"], "truncated")
                self.assertTrue(payload["details"]["response_truncated"])
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_completed_job_payload_includes_memory_summary_fields(self) -> None:
        payload = {
            "summary": {"generated_at": "now"},
            "runs": [
                {
                    "task": {"id": "contains-duplicates"},
                    "memory_before_count": 2,
                    "memory_after_count": 5,
                    "positive_experiences_added": 1,
                    "negative_experiences_added": 2,
                    "added_experiences": [
                        {
                            "experience_id": "exp-1",
                            "generation": 1,
                            "experience_outcome": "success",
                            "verifier_status": "pass",
                            "delta_J": 0.4,
                            "prompt_fragment": "Prefer set membership.",
                            "strategy_hypothesis": "Hash lookup dominates nested scans.",
                            "candidate_summary": "Streaming set check.",
                            "proposal_model": "deepseek-chat",
                        }
                    ],
                }
            ],
            "task_catalog": [],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "payload.json"
            artifact_path.write_text(json.dumps(payload))
            with (
                patch.object(server.ProposalRuntime, "from_env", return_value=make_runtime([])),
                patch.object(server, "write_discrete_artifacts", return_value=artifact_path),
            ):
                httpd, thread = self._serve()
                try:
                    status, start_payload = _fetch_json(
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=contains-duplicates",
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    completed = {}
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    run = completed["payload"]["runs"][0]
                    self.assertEqual(run["memory_before_count"], 2)
                    self.assertEqual(run["memory_after_count"], 5)
                    self.assertEqual(run["positive_experiences_added"], 1)
                    self.assertEqual(run["negative_experiences_added"], 2)
                    self.assertEqual(len(run["added_experiences"]), 1)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_latest_run_payload_preserves_main_and_experiment_summary_split(self) -> None:
        cached_payload = {
            "summary": {
                "generated_at": "cached",
                "num_tasks": 1,
                "total_runs": 2,
                "experiment_runs": 1,
                "total_generations": 3,
                "write_backs": 2,
                "experiment_write_backs": -1,
            },
            "runs": [
                {"task": {"id": "math-500"}, "included_in_main_comparison": True},
                {"task": {"id": "contains-duplicates"}, "included_in_main_comparison": False},
            ],
            "task_catalog": [],
        }
        with patch.object(server, "load_cached_discrete_payload", return_value=cached_payload):
            httpd, thread = self._serve()
            try:
                status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/latest-run")
                self.assertEqual(status, 200)
                self.assertEqual(payload["summary"]["num_tasks"], 1)
                self.assertEqual(payload["summary"]["total_runs"], 2)
                self.assertEqual(payload["summary"]["experiment_runs"], 1)
                self.assertEqual(payload["summary"]["write_backs"], 2)
                self.assertEqual(payload["summary"]["experiment_write_backs"], -1)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
