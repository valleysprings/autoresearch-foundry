from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.demo_run import ROOT, RUNS, write_demo_artifacts

UI_DIR = ROOT / "ui"


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/latest-run":
            artifact = write_demo_artifacts()
            payload = artifact.read_text()
            body = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/health":
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path
        super().do_GET()


def main() -> None:
    write_demo_artifacts()
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
