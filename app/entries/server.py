from __future__ import annotations

import argparse
import errno
import json
import multiprocessing
import os
import queue as queue_module
import signal
import socket
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

from app.codegen.catalog import list_codegen_task_summaries
from app.codegen.errors import AutoresearchError, ConfigError
from app.codegen.llm import ProposalRuntime
from app.entries.runner import ROOT, load_cached_discrete_payload, write_discrete_artifacts


UI_DIR = ROOT / "ui" / "dist"
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, object]] = {}
PORT_CONFLICT_MODES = ("auto", "next", "kill", "error")
JOB_PROCESS_START_METHOD = os.getenv("AUTORESEARCH_JOB_PROCESS_START_METHOD", "spawn").strip().lower() or "spawn"
JOB_STALL_TIMEOUT_S = 180.0
QUIET_ACCESS_LOG_PATHS = frozenset({"/api/latest-run", "/api/job", "/api/health"})


def _runtime_for_request(model: str | None = None) -> ProposalRuntime:
    runtime = ProposalRuntime.from_env()
    return runtime.with_model(model)


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, AutoresearchError):
        return exc.as_payload()
    return {
        "terminal": True,
        "error_type": "runtime_error",
        "error": str(exc),
        "model": None,
    }


def _parse_positive_int(raw_value: str | None, field: str) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _parse_port(raw_value: str) -> int:
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _listening_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t", "-n", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pids.append(int(stripped))
        except ValueError:
            continue
    return pids


def _command_for_pid(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def _cwd_for_pid(pid: int) -> str:
    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return line[1:].strip()
    return ""


def _is_autoresearch_server_process(command: str) -> bool:
    normalized = command.strip().lower()
    if not normalized:
        return False
    markers = (
        "-m app serve",
        "-m app.run serve",
        "-m app.entries.server",
        "app/entries/server.py",
    )
    return any(marker in normalized for marker in markers)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, timeout_s: float = 2.0) -> bool:
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.05)
    return not _process_exists(pid)


def _stop_managed_listener_for_port(port: int) -> list[int]:
    stopped: list[int] = []
    for pid in _listening_pids(port):
        command = _command_for_pid(pid)
        cwd = _cwd_for_pid(pid)
        in_workspace = cwd == str(ROOT) or str(ROOT) in command
        if not in_workspace:
            continue
        if not _is_autoresearch_server_process(command):
            continue
        if _terminate_pid(pid):
            stopped.append(pid)
    return stopped


def _next_available_port(host: str, start_port: int, *, max_attempts: int = 100) -> int:
    port = max(0, start_port)
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                port += 1
                continue
        return port
    raise RuntimeError(f"Could not find an available port after trying {max_attempts} ports starting at {start_port}.")


def _bind_server(host: str, port: int, port_conflict: str) -> tuple[ThreadingHTTPServer, int, str | None]:
    try:
        return ThreadingHTTPServer((host, port), DemoHandler), port, None
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise

    managed_pids: list[int] = []
    if port_conflict in {"auto", "kill"}:
        managed_pids = _stop_managed_listener_for_port(port)
        if managed_pids:
            try:
                return (
                    ThreadingHTTPServer((host, port), DemoHandler),
                    port,
                    f"Port {port} was occupied by stale autoresearch server pid(s) {', '.join(str(pid) for pid in managed_pids)}. Reused the same port.",
                )
            except OSError as retry_exc:
                if retry_exc.errno != errno.EADDRINUSE:
                    raise
                if port_conflict == "kill":
                    raise RuntimeError(
                        f"Stopped stale autoresearch server pid(s) {', '.join(str(pid) for pid in managed_pids)}, but port {port} is still busy."
                    ) from retry_exc
    if port_conflict in {"auto", "next"}:
        next_port = _next_available_port(host, port + 1)
        return (
            ThreadingHTTPServer((host, next_port), DemoHandler),
            next_port,
            f"Port {port} is already in use. Using port {next_port} instead.",
        )
    if port_conflict == "kill":
        raise RuntimeError(
            f"Port {port} is already in use and no stale autoresearch server could be stopped. Use --port-conflict next or choose another --port."
        )
    raise RuntimeError(
        f"Port {port} is already in use. Re-run with --port-conflict next, --port-conflict kill, or choose another --port."
    )


def _should_suppress_request_logging(path: str) -> bool:
    flag = os.getenv("AUTORESEARCH_LOG_POLLING", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    return urlparse(path).path in QUIET_ACCESS_LOG_PATHS


def _job_process_context() -> multiprocessing.context.BaseContext:
    preferred_method = JOB_PROCESS_START_METHOD
    # Unit tests patch job entrypoints; prefer fork there so child processes inherit the mocks.
    if preferred_method == "spawn" and (
        isinstance(write_discrete_artifacts, Mock) or isinstance(ProposalRuntime.from_env, Mock)
    ):
        preferred_method = "fork"
    start_methods = multiprocessing.get_all_start_methods()
    if preferred_method in start_methods:
        return multiprocessing.get_context(preferred_method)
    if "forkserver" in start_methods:
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def _should_run_job_inline() -> bool:
    if not isinstance(write_discrete_artifacts, Mock):
        return False
    side_effect = write_discrete_artifacts.side_effect
    return not callable(side_effect)


def _run_job_process(
    event_queue,
    task_id: str | None,
    model: str | None,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
) -> None:
    def progress(event: dict) -> None:
        event_queue.put({"type": "event", "event": event})

    try:
        proposal_runtime = _runtime_for_request(model)
        artifact = write_discrete_artifacts(
            task_id=task_id,
            progress_callback=progress,
            pace_ms=120,
            proposal_runtime=proposal_runtime,
            generation_budget=generation_budget,
            candidate_budget=candidate_budget,
            branching_factor=branching_factor,
            item_workers=item_workers,
            max_items=max_items,
        )
        event_queue.put({"type": "completed", "artifact_path": str(artifact)})
    except Exception as exc:  # noqa: BLE001
        event_queue.put({"type": "failed", "payload": _error_payload(exc)})


def _run_job(
    job_id: str,
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
) -> None:
    if _should_run_job_inline():
        def progress(event: dict) -> None:
            with JOB_LOCK:
                JOBS[job_id]["events"].append(event)
                JOBS[job_id]["last_progress_at"] = time.time()

        try:
            artifact = write_discrete_artifacts(
                task_id=task_id,
                progress_callback=progress,
                pace_ms=120,
                proposal_runtime=proposal_runtime,
                generation_budget=generation_budget,
                candidate_budget=candidate_budget,
                branching_factor=branching_factor,
                item_workers=item_workers,
                max_items=max_items,
            )
            payload = json.loads(Path(artifact).read_text())
            with JOB_LOCK:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["payload"] = payload
                JOBS[job_id]["last_progress_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            with JOB_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id].update(_error_payload(exc))
                JOBS[job_id]["last_progress_at"] = time.time()
        return

    context = _job_process_context()
    event_queue = context.Queue()
    process = context.Process(
        target=_run_job_process,
        args=(
            event_queue,
            task_id,
            proposal_runtime.active_model,
            branching_factor,
            generation_budget,
            candidate_budget,
            item_workers,
            max_items,
        ),
        daemon=True,
    )
    process.start()
    last_progress_at = time.monotonic()

    try:
        while True:
            try:
                message = event_queue.get(timeout=0.5)
            except queue_module.Empty:
                if time.monotonic() - last_progress_at > JOB_STALL_TIMEOUT_S:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive() and hasattr(process, "kill"):
                            process.kill()
                            process.join(timeout=1)
                    with JOB_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id].update(
                            {
                                "terminal": True,
                                "error_type": "runtime_error",
                                "error": (
                                    f"Job stalled for more than {int(JOB_STALL_TIMEOUT_S)} seconds without progress and was terminated."
                                ),
                                "model": proposal_runtime.active_model,
                            }
                        )
                    return
                if not process.is_alive():
                    with JOB_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id].update(
                            {
                                "terminal": True,
                                "error_type": "runtime_error",
                                "error": f"Job exited unexpectedly with code {process.exitcode}.",
                                "model": proposal_runtime.active_model,
                            }
                        )
                    return
                continue

            last_progress_at = time.monotonic()
            message_type = str(message.get("type") or "")
            if message_type == "event":
                with JOB_LOCK:
                    JOBS[job_id]["events"].append(message["event"])
                    JOBS[job_id]["last_progress_at"] = time.time()
                continue

            if message_type == "completed":
                artifact_path = str(message["artifact_path"])
                payload = json.loads(Path(artifact_path).read_text())
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "completed"
                    JOBS[job_id]["payload"] = payload
                    JOBS[job_id]["last_progress_at"] = time.time()
                return

            if message_type == "failed":
                with JOB_LOCK:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id].update(dict(message["payload"]))
                    JOBS[job_id]["last_progress_at"] = time.time()
                return
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
        with JOB_LOCK:
            if "last_progress_at" not in JOBS[job_id]:
                JOBS[job_id]["last_progress_at"] = time.time()
        event_queue.close()
        event_queue.join_thread()


def _start_job(
    task_id: str | None,
    proposal_runtime: ProposalRuntime,
    branching_factor: int | None,
    generation_budget: int | None,
    candidate_budget: int | None,
    item_workers: int | None,
    max_items: int | None,
) -> str:
    job_id = uuid.uuid4().hex[:10]
    with JOB_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "task_id": task_id,
            "branching_factor": branching_factor,
            "generation_budget": generation_budget,
            "candidate_budget": candidate_budget,
            "item_workers": item_workers,
            "max_items": max_items,
            "events": [],
            "payload": None,
            "terminal": False,
            "error_type": None,
            "error": None,
            "model": proposal_runtime.active_model,
            "started_at": time.time(),
            "last_progress_at": time.time(),
            "stall_timeout_s": JOB_STALL_TIMEOUT_S,
        }
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, task_id, proposal_runtime, branching_factor, generation_budget, candidate_budget, item_workers, max_items),
        daemon=True,
    )
    thread.start()
    return job_id


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if _should_suppress_request_logging(self.path):
            return
        super().log_request(code, size)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/latest-run":
            task_id = query.get("task_id", [None])[0]
            try:
                payload = load_cached_discrete_payload(task_id=task_id)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, payload)
            return

        if parsed.path == "/api/tasks":
            _json_response(self, {"tasks": list_codegen_task_summaries()})
            return

        if parsed.path == "/api/runtime":
            try:
                runtime = _runtime_for_request()
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, runtime.describe())
            return

        if parsed.path == "/api/job":
            job_id = query.get("job_id", [None])[0]
            if job_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "job_id is required")
                return
            with JOB_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "job not found")
                return
            _json_response(self, job)
            return

        if parsed.path == "/api/health":
            _json_response(self, {"ok": True})
            return

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/run-task":
            task_id = query.get("task_id", [None])[0]
            model = query.get("model", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            if task_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "task_id is required")
                return
            try:
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        task_id,
                        runtime,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                    ),
                    "model": runtime.active_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        if parsed.path == "/api/run-sequence":
            model = query.get("model", [None])[0]
            branching_value = query.get("branching_factor", [None])[0]
            generation_value = query.get("generation_budget", [None])[0]
            candidate_value = query.get("candidate_budget", [None])[0]
            item_workers_value = query.get("item_workers", [None])[0]
            max_items_value = query.get("max_items", [None])[0]
            try:
                branching_factor = _parse_positive_int(branching_value, "branching_factor")
                generation_budget = _parse_positive_int(generation_value, "generation_budget")
                candidate_budget = _parse_positive_int(candidate_value, "candidate_budget")
                item_workers = _parse_positive_int(item_workers_value, "item_workers")
                max_items = _parse_positive_int(max_items_value, "max_items")
            except ValueError as exc:
                _json_response(self, ConfigError(str(exc)).as_payload(), status=HTTPStatus.BAD_REQUEST)
                return
            try:
                runtime = _runtime_for_request(model)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                {
                    "job_id": _start_job(
                        None,
                        runtime,
                        branching_factor,
                        generation_budget,
                        candidate_budget,
                        item_workers,
                        max_items,
                    ),
                    "model": runtime.active_model,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the autoresearch UI and API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=_parse_port, default=8000, help="TCP port to bind. Defaults to 8000.")
    parser.add_argument(
        "--port-conflict",
        choices=PORT_CONFLICT_MODES,
        default="auto",
        help="How to handle an occupied port: auto stops stale autoresearch servers or moves to the next free port.",
    )
    args = parser.parse_args(argv)

    ProposalRuntime.from_env()
    if not (UI_DIR / "index.html").exists():
        raise RuntimeError("UI build missing. Run `cd ui && npm install && npm run build` before starting the server.")
    server, bound_port, conflict_note = _bind_server(args.host, args.port, args.port_conflict)
    if conflict_note is not None:
        print(conflict_note)
    print(f"serving autoresearch UI at http://{args.host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
