from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiSmokeTest(unittest.TestCase):
    def test_react_app_mentions_llm_required_and_dynamic_runtime(self) -> None:
        source = (ROOT / "ui" / "src" / "App.tsx").read_text()
        self.assertIn('planning_verified: "Planning"', source)
        self.assertIn('deepsearch_verified: "Deep Search"', source)
        self.assertIn("llm-required", source)
        self.assertIn("Auto Research Console", source)
        self.assertIn("Auto Research Operations Console", source)
        self.assertIn("Start verified run", source)
        self.assertIn("Open dataset brief", source)
        self.assertIn("run failure", source)
        self.assertIn("Runtime and verifier configuration", source)
        self.assertIn("Live execution trace", source)
        self.assertIn("Latest cached report", source)
        self.assertIn("No cached report for this task yet", source)
        self.assertIn('aria-label="Theme"', source)
        self.assertIn("selection rules", source)
        self.assertIn("Item-level outcomes", source)
        self.assertIn("items scheduled", source)
        self.assertIn("Generation {item.latestGeneration || 0}/{task.generationBudget || \"?\"}", source)
        self.assertIn("Response", source)
        self.assertIn("Question.", source)
        self.assertIn("fail -> pass", source)
        self.assertIn("Run memory log", source)
        self.assertIn('selectedTaskIsCoding ? "Problem Count" : "Item Cap"', source)
        self.assertIn("Dataset size:", source)
        self.assertIn("Demo warning: running more than 50 LiveCodeBench items here is not recommended.", source)
        self.assertIn("Single-item task: cap disabled", source)
        self.assertIn("dataset brief", source)
        self.assertIn("Frontier Parents", source)
        self.assertIn("Max Search Rounds", source)
        self.assertIn("Candidates Count per Branch", source)
        self.assertIn("Parallel Item Workers", source)
        self.assertNotIn("LLM Queue", source)
        self.assertNotIn("Run main benchmark sequence", source)
        self.assertNotIn("Main benchmark comparison", source)
        self.assertNotIn("Branching evolution, task by task.", source)
        self.assertNotIn("EvoAlgo: Benchmark Workbench", source)
        self.assertNotIn("Autoresearch Benchmark Console", source)
        self.assertNotIn("Run verified search on benchmark tasks.", source)
        self.assertNotIn("Mainline Benchmarks", source)
        self.assertNotIn("mainline benchmark", source)
        self.assertNotIn("comparison slate", source)
        self.assertNotIn("sidecar", source)
        self.assertNotIn("Cached run reports", source)
        self.assertNotIn("Benchmark Reports", source)
        self.assertNotIn('reasoning_verified: "Reasoning"', source)
        self.assertNotIn('longcontext_verified: "Long Context"', source)
        self.assertNotIn("Multihop Reasoning", source)
        self.assertNotIn("Smoke / Regression", source)
        self.assertNotIn("Smoke / Regression Reports", source)
        self.assertNotIn("Generation Cap", source)
        self.assertNotIn("Candidates / Branch", source)
        self.assertNotIn("items opened", source)
        self.assertNotIn('<span className="badge">{runtimeInfo.mode}</span>', source)
        self.assertNotIn('<span className="badge">runtime {runtimeInfo.active_model}</span>', source)

    def test_index_loads_react_entry(self) -> None:
        source = (ROOT / "ui" / "index.html").read_text()
        self.assertIn("Auto Research Console", source)
        self.assertNotIn("Autoresearch Benchmark Console", source)
        self.assertNotIn("EvoAlgo: Benchmark Workbench", source)
        self.assertIn("/src/main.tsx", source)

    def test_styles_and_polling_guards_prevent_layout_blowout(self) -> None:
        app_source = (ROOT / "ui" / "src" / "App.tsx").read_text()
        styles = (ROOT / "ui" / "src" / "styles.css").read_text()
        self.assertIn('document.addEventListener("visibilitychange"', app_source)
        self.assertIn("IDLE_LATEST_RUN_POLL_MS", app_source)
        self.assertIn("LIVE_JOB_BACKGROUND_POLL_MS", app_source)
        self.assertIn("text-size-adjust: 100%", styles)
        self.assertIn("scrollbar-gutter: stable;", styles)
        self.assertIn("min-width: 0;", styles)
        self.assertIn("width: 100%;", styles)


if __name__ == "__main__":
    unittest.main()
