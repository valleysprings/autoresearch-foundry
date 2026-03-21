from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app.codegen.catalog import list_codegen_task_summaries
from app.codegen.errors import AutoresearchError
from app.codegen.llm import ProposalRuntime
from app.entries.discrete_demo import ROOT, write_discrete_artifacts


UI_DIR = ROOT / "ui"
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, object]] = {}


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


def _run_job(job_id: str, task_id: str | None) -> None:
    def progress(event: dict) -> None:
        with JOB_LOCK:
            JOBS[job_id]["events"].append(event)

    try:
        artifact = write_discrete_artifacts(
            task_id=task_id,
            progress_callback=progress,
            pace_ms=120,
        )
        payload = json.loads(artifact.read_text())
        with JOB_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["payload"] = payload
    except Exception as exc:  # noqa: BLE001
        with JOB_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id].update(_error_payload(exc))


def _start_job(task_id: str | None) -> str:
    job_id = uuid.uuid4().hex[:10]
    with JOB_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "task_id": task_id,
            "events": [],
            "payload": None,
            "terminal": False,
            "error_type": None,
            "error": None,
            "model": None,
        }
    thread = threading.Thread(target=_run_job, args=(job_id, task_id), daemon=True)
    thread.start()
    return job_id


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/latest-run":
            task_id = query.get("task_id", [None])[0]
            try:
                artifact = write_discrete_artifacts(task_id=task_id)
            except Exception as exc:  # noqa: BLE001
                _json_response(self, _error_payload(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, json.loads(artifact.read_text()))
            return

        if parsed.path == "/api/tasks":
            _json_response(self, {"tasks": list_codegen_task_summaries()})
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
            if task_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "task_id is required")
                return
            _json_response(self, {"job_id": _start_job(task_id)}, status=HTTPStatus.ACCEPTED)
            return

        if parsed.path == "/api/run-sequence":
            _json_response(self, {"job_id": _start_job(None)}, status=HTTPStatus.ACCEPTED)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def main() -> None:
    ProposalRuntime.from_env()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("serving codegen demo at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
