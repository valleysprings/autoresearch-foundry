from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiSmokeTest(unittest.TestCase):
    def test_react_app_mentions_llm_required_and_dynamic_runtime(self) -> None:
        source = (ROOT / "ui" / "src" / "App.tsx").read_text()
        self.assertIn("llm-required", source)
        self.assertIn("terminal failure", source)
        self.assertIn("Verifier and proposal runtime", source)
        self.assertIn("Run selected task", source)
        self.assertIn("Per-generation net change", source)
        self.assertIn("Task-scoped runtime trace", source)
        self.assertIn("Task summaries and generation details", source)
        self.assertIn("Main benchmark comparison", source)
        self.assertIn("Small Experiments", source)
        self.assertIn("Theme mode", source)
        self.assertIn("Generational deltas", source)
        self.assertIn("Per-question results", source)
        self.assertIn("dataset total questions", source)
        self.assertIn("Dataset manifest and item artifacts", source)
        self.assertIn("Max Items", source)

    def test_index_loads_react_entry(self) -> None:
        source = (ROOT / "ui" / "index.html").read_text()
        self.assertIn("Strict LLM-Required Codegen Workbench", source)
        self.assertIn("/src/main.tsx", source)


if __name__ == "__main__":
    unittest.main()
