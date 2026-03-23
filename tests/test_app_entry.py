from __future__ import annotations

import unittest
from unittest.mock import patch

from app import run


class AppEntryTest(unittest.TestCase):
    def test_default_mode_dispatches_to_cli_runner(self) -> None:
        with (
            patch.object(run, "runner_main") as runner_main,
            patch.object(run, "server_main") as server_main,
        ):
            run.main(["--task", "olymmath", "--max-items", "5"])
            runner_main.assert_called_once_with(["--task", "olymmath", "--max-items", "5"])
            server_main.assert_not_called()

    def test_serve_mode_dispatches_to_http_server(self) -> None:
        with (
            patch.object(run, "runner_main") as runner_main,
            patch.object(run, "server_main") as server_main,
        ):
            run.main(["serve", "--port", "9000"])
            server_main.assert_called_once_with(["--port", "9000"])
            runner_main.assert_not_called()
