from __future__ import annotations

import errno
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

from app.bench.personalization_references import load_personalization_reference_benchmarks
from app.codegen.catalog import list_codegen_task_summaries
from app.codegen.errors import LlmResponseError, LlmTransportError
from app.entries import server
from tests.helpers import enabled_registry_task_ids, make_runtime, runnable_personalization_reference_ids


def _fetch_json(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url, method=method, data=body, headers=headers or {})
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

    def test_stalled_job_fails_fast_without_http_server(self) -> None:
        def hang(*_args, **_kwargs):
            time.sleep(10)
            raise AssertionError("stalled job should have been terminated before this returns")

        with (
            patch.object(server, "JOB_STALL_TIMEOUT_S", 0.1),
            patch.object(server, "JOB_PROCESS_START_METHOD", "fork"),
            patch.object(server, "write_discrete_artifacts", side_effect=hang),
        ):
            job_id = server._start_job(
                "contains-duplicates",
                make_runtime([]),
                None,
                branching_factor=None,
                generation_budget=None,
                candidate_budget=None,
                item_workers=None,
                max_items=None,
                selected_item_ids=None,
                suite_config=None,
            )
            try:
                deadline = time.time() + 5
                payload = {}
                while time.time() < deadline:
                    with server.JOB_LOCK:
                        payload = dict(server.JOBS[job_id])
                    if payload["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error_type"], "runtime_error")
                self.assertIn("stalled", str(payload["error"]).lower())
            finally:
                with server.JOB_LOCK:
                    server.JOBS.pop(job_id, None)

    def test_stall_timeout_tracks_runtime_timeout_and_reasoner_models(self) -> None:
        runtime = make_runtime([], model="deepseek-reasoner")
        timeout_s = server._job_stall_timeout_s(runtime)
        self.assertGreater(timeout_s, runtime.config.timeout_s)
        self.assertGreater(timeout_s, server.DEFAULT_JOB_STALL_TIMEOUT_S)

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

    def test_task_scoped_latest_run_prefers_task_specific_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir)
            (runs_root / "latest_run.json").write_text(
                json.dumps({"summary": {"generated_at": "global"}, "runs": [{"task": {"id": "olymmath"}}], "task_catalog": []})
            )
            (runs_root / "codegen-livecodebench.json").write_text(
                json.dumps({"summary": {"generated_at": "task"}, "runs": [{"task": {"id": "livecodebench"}}], "task_catalog": []})
            )
            payload = server.load_cached_discrete_payload(task_id="livecodebench", runs_root=runs_root)
            self.assertEqual(payload["summary"]["generated_at"], "task")
            self.assertEqual(payload["runs"][0]["task"]["id"], "livecodebench")

    def test_polling_access_logs_are_suppressed_by_default(self) -> None:
        self.assertTrue(server._should_suppress_request_logging("/api/latest-run?task_id=planbench-t1"))
        self.assertTrue(server._should_suppress_request_logging("/api/job?job_id=abc123"))
        self.assertTrue(server._should_suppress_request_logging("/api/health"))
        self.assertFalse(server._should_suppress_request_logging("/api/tasks"))

    def test_polling_access_logs_can_be_reenabled_via_env_var(self) -> None:
        with patch.dict("os.environ", {"AUTORESEARCH_LOG_POLLING": "1"}):
            self.assertFalse(server._should_suppress_request_logging("/api/latest-run?task_id=planbench-t1"))
            self.assertFalse(server._should_suppress_request_logging("/api/job?job_id=abc123"))

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

    def test_stalled_job_is_terminated_after_timeout(self) -> None:
        def hang(*_args, **_kwargs):
            time.sleep(10)
            raise AssertionError("stalled job should have been terminated before this returns")

        with (
            patch.object(server, "JOB_STALL_TIMEOUT_S", 0.1),
            patch.object(server, "write_discrete_artifacts", side_effect=hang),
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
                payload = {}
                while time.time() < deadline:
                    _, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                    if payload["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error_type"], "runtime_error")
                self.assertIn("stalled", str(payload["error"]).lower())
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
                self.assertEqual(payload["default_model"], "deepseek-chat")
                self.assertIn("deepseek-reasoner", payload["available_models"])
                self.assertIn("glm-5", payload["available_models"])
                self.assertEqual(payload["transport"], "openai-compatible")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

    def test_tasks_endpoint_returns_benchmark_metadata(self) -> None:
        httpd, thread = self._serve()
        try:
            status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/tasks")
            self.assertEqual(status, 200)
            expected_summaries = list_codegen_task_summaries()
            self.assertEqual(payload.get("dataset_warnings"), server.list_missing_local_dataset_warnings())
            self.assertEqual({task["id"] for task in payload["tasks"]}, set(enabled_registry_task_ids()))
            self.assertEqual([task["id"] for task in payload["tasks"]], [task["id"] for task in expected_summaries])

            payload_by_id = {task["id"]: task for task in payload["tasks"]}
            for expected in expected_summaries:
                actual = payload_by_id[expected["id"]]
                for key, value in expected.items():
                    self.assertEqual(actual[key], value, f"{expected['id']} field mismatch: {key}")

            expected_references = load_personalization_reference_benchmarks()
            actual_references = payload.get("personalization_reference_benchmarks", [])
            self.assertEqual(actual_references, expected_references)
            self.assertEqual(
                {entry["id"] for entry in actual_references},
                runnable_personalization_reference_ids(),
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_tasks_endpoint_includes_dataset_warnings(self) -> None:
        warning = {
            "task_id": "longbench-v2",
            "title": "LongBench v2",
            "track": "longcontext_verified",
            "manifest_path": "/tmp/benchmark/longcontext_verified/longbench-v2/data/questions.json",
            "prepare_command": "python benchmark/prepare_datasets.py --task-id longbench-v2",
            "message": "Missing local dataset manifest: /tmp/benchmark/longcontext_verified/longbench-v2/data/questions.json",
        }
        with (
            patch.object(server, "list_codegen_task_summaries", return_value=[]),
            patch.object(server, "list_missing_local_dataset_warnings", return_value=[warning]),
        ):
            httpd, thread = self._serve()
            try:
                status, payload = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/tasks")
                self.assertEqual(status, 200)
                self.assertEqual(payload["tasks"], [])
                self.assertEqual(payload["dataset_warnings"], [warning])
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

    def test_eval_model_is_forwarded_to_job_runner(self) -> None:
        payload = {
            "summary": {"generated_at": "now", "policy_model": "deepseek-chat", "eval_model": "gpt-4.1-mini"},
            "runs": [],
            "task_catalog": [],
        }
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
                            "?task_id=contains-duplicates&eval_model=gpt-4.1-mini"
                        ),
                        method="POST",
                    )
                    self.assertEqual(status, 202)
                    self.assertEqual(start_payload["policy_model"], "deepseek-chat")
                    self.assertEqual(start_payload["eval_model"], "gpt-4.1-mini")
                    job_id = start_payload["job_id"]
                    deadline = time.time() + 5
                    completed = {}
                    while time.time() < deadline:
                        _, completed = _fetch_json(f"http://127.0.0.1:{httpd.server_port}/api/job?job_id={job_id}")
                        if completed["status"] != "running":
                            break
                        time.sleep(0.05)
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["policy_model"], "deepseek-chat")
                    self.assertEqual(completed["eval_model"], "gpt-4.1-mini")
                    self.assertEqual(write_artifacts.call_args.kwargs["eval_model"], "gpt-4.1-mini")
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_suite_config_is_forwarded_to_job_runner(self) -> None:
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
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=harmbench-text-harmful",
                        method="POST",
                        body=json.dumps({"suite_config": {"n_tasks": 2, "agent_name": "custom-agent"}}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
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
                    self.assertEqual(completed["suite_config"], {"n_tasks": 2, "agent_name": "custom-agent"})
                    self.assertEqual(write_artifacts.call_args.kwargs["suite_config"], {"n_tasks": 2, "agent_name": "custom-agent"})
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_runtime_split_suite_config_is_forwarded_to_job_runner(self) -> None:
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
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=livecodebench",
                        method="POST",
                        body=json.dumps({"suite_config": {"split": "v6"}}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
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
                    self.assertEqual(completed["suite_config"], {"split": "v6"})
                    self.assertEqual(write_artifacts.call_args.kwargs["suite_config"], {"split": "v6"})
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

    def test_llm_concurrency_overrides_runtime_for_job(self) -> None:
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
                            "?task_id=math-500&llm_concurrency=7"
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
                    self.assertEqual(completed["llm_concurrency"], 7)
                    self.assertEqual(write_artifacts.call_args.kwargs["proposal_runtime"].config.llm_concurrency, 7)
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_item_workers_becomes_default_llm_concurrency_when_unspecified(self) -> None:
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
                            "?task_id=olymmath&item_workers=11"
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
                    self.assertEqual(completed["llm_concurrency"], 11)
                    self.assertEqual(write_artifacts.call_args.kwargs["proposal_runtime"].config.llm_concurrency, 11)
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

    def test_item_ids_are_forwarded_to_job_runner(self) -> None:
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
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=aime-2026&item_ids=10,1",
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
                    self.assertEqual(completed["item_ids"], ["10", "1"])
                    self.assertEqual(write_artifacts.call_args.kwargs["selected_item_ids"], ["10", "1"])
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    thread.join(timeout=5)

    def test_skill_options_are_forwarded_to_job_runner(self) -> None:
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
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task?task_id=olymmath",
                        method="POST",
                        body=json.dumps(
                            {
                                "record_skill": True,
                                "skill_item_limit": 3,
                                "selected_skill_id": "olymmath/olymmath-gpt-5-4-task3-20260402_120000.md",
                            }
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
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
                    self.assertTrue(completed["record_skill"])
                    self.assertEqual(completed["skill_item_limit"], 3)
                    self.assertEqual(
                        completed["selected_skill_id"],
                        "olymmath/olymmath-gpt-5-4-task3-20260402_120000.md",
                    )
                    self.assertTrue(write_artifacts.call_args.kwargs["record_skill"])
                    self.assertEqual(write_artifacts.call_args.kwargs["skill_item_limit"], 3)
                    self.assertEqual(
                        write_artifacts.call_args.kwargs["selected_skill_id"],
                        "olymmath/olymmath-gpt-5-4-task3-20260402_120000.md",
                    )
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
                status, _ = _fetch_json(
                    (
                        f"http://127.0.0.1:{httpd.server_port}/api/run-task"
                        "?task_id=olymmath&llm_concurrency=bad"
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
                            "delta_primary_score": 0.4,
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

    def test_bind_server_reuses_port_after_stopping_stale_autoresearch_listener(self) -> None:
        sentinel_httpd = object()
        with (
            patch.object(
                server,
                "ThreadingHTTPServer",
                side_effect=[OSError(errno.EADDRINUSE, "Address already in use"), sentinel_httpd],
            ) as http_server,
            patch.object(server, "_stop_managed_listener_for_port", return_value=[4321]) as stop_listener,
        ):
            httpd, bound_port, note = server._bind_server("127.0.0.1", 8000, "auto")
            self.assertIs(httpd, sentinel_httpd)
            self.assertEqual(bound_port, 8000)
            self.assertIn("4321", note or "")
            stop_listener.assert_called_once_with(8000)
            self.assertEqual(http_server.call_count, 2)

    def test_bind_server_moves_to_next_port_when_requested_port_is_busy(self) -> None:
        sentinel_httpd = object()
        with (
            patch.object(
                server,
                "ThreadingHTTPServer",
                side_effect=[OSError(errno.EADDRINUSE, "Address already in use"), sentinel_httpd],
            ) as http_server,
            patch.object(server, "_stop_managed_listener_for_port", return_value=[]),
            patch.object(server, "_next_available_port", return_value=8001) as next_port,
        ):
            httpd, bound_port, note = server._bind_server("127.0.0.1", 8000, "next")
            self.assertIs(httpd, sentinel_httpd)
            self.assertEqual(bound_port, 8001)
            self.assertIn("8001", note or "")
            next_port.assert_called_once_with("127.0.0.1", 8001)
            self.assertEqual(http_server.call_count, 2)

    def test_bind_server_kill_mode_fails_if_no_managed_listener_can_be_stopped(self) -> None:
        with (
            patch.object(
                server,
                "ThreadingHTTPServer",
                side_effect=OSError(errno.EADDRINUSE, "Address already in use"),
            ),
            patch.object(server, "_stop_managed_listener_for_port", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "no stale autoresearch server could be stopped"):
                server._bind_server("127.0.0.1", 8000, "kill")

    def test_stop_managed_listener_only_kills_current_workspace_server(self) -> None:
        with (
            patch.object(server, "_listening_pids", return_value=[1111, 2222]),
            patch.object(server, "_command_for_pid", side_effect=["python3 -m app serve", "python3 -m app serve"]),
            patch.object(server, "_cwd_for_pid", side_effect=[str(server.ROOT), "/tmp/other-project"]),
            patch.object(server, "_terminate_pid", return_value=True) as terminate_pid,
        ):
            stopped = server._stop_managed_listener_for_port(8000)
            self.assertEqual(stopped, [1111])
            terminate_pid.assert_called_once_with(1111)


if __name__ == "__main__":
    unittest.main()
