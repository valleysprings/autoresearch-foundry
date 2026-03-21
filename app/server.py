from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app.demo_run import ROOT, write_demo_artifacts
from app.task_catalog import list_task_summaries

UI_DIR = ROOT / "ui"


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/latest-run":
            task_id = query.get("task_id", [None])[0]
            artifact = write_demo_artifacts(task_id=task_id)
            _json_response(self, json.loads(artifact.read_text()))
            return

        if parsed.path == "/api/tasks":
            _json_response(self, {"tasks": list_task_summaries()})
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
            artifact = write_demo_artifacts(task_id=task_id)
            _json_response(self, json.loads(artifact.read_text()))
            return

        if parsed.path == "/api/run-sequence":
            artifact = write_demo_artifacts(task_id=None)
            _json_response(self, json.loads(artifact.read_text()))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def main() -> None:
    write_demo_artifacts(task_id="contains-duplicates")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DemoHandler)
    print("serving demo at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
