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

from app.codegen.errors import LlmTransportError
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


if __name__ == "__main__":
    unittest.main()
