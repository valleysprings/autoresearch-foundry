from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiSmokeTest(unittest.TestCase):
    def test_app_mentions_llm_required_and_terminal_failure(self) -> None:
        source = (ROOT / "ui" / "app.js").read_text()
        self.assertIn("llm-required", source)
        self.assertIn("terminal failure", source)
        self.assertIn("Deterministic verification around direct code generation.", source)

    def test_index_loads_codegen_workbench(self) -> None:
        source = (ROOT / "ui" / "index.html").read_text()
        self.assertIn("Strict LLM-Required Codegen Workbench", source)
        self.assertIn("Loading strict codegen workbench", source)


if __name__ == "__main__":
    unittest.main()
