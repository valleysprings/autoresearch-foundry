from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiSmokeTest(unittest.TestCase):
    def test_react_app_mentions_llm_required_and_dynamic_runtime(self) -> None:
        source = (ROOT / "ui" / "src" / "App.tsx").read_text()
        self.assertIn("llm-required", source)
        self.assertIn("terminal failure", source)
        self.assertIn("Deterministic verifier and model runtime", source)
        self.assertIn("Run selected task", source)
        self.assertIn("Memory fragments", source)
        self.assertIn("Positive fragments", source)
        self.assertIn("Negative fragments", source)
        self.assertIn("Theme mode", source)
        self.assertIn("run overview", source)

    def test_index_loads_react_entry(self) -> None:
        source = (ROOT / "ui" / "index.html").read_text()
        self.assertIn("Strict LLM-Required Codegen Workbench", source)
        self.assertIn("/src/main.tsx", source)


if __name__ == "__main__":
    unittest.main()
