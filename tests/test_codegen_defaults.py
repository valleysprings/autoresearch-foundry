from __future__ import annotations

import unittest

from app.codegen.catalog import seed_strategy_experiences
from app.configs.codegen import (
    DEFAULT_SESSION_ID,
    DEFAULT_SPEEDUP_OBJECTIVE_SPEC,
    DELTA_FORMULA,
    PRIMARY_FORMULA,
    RUN_DELTA_FORMULA,
    TIE_BREAK_FORMULA,
)


class CodegenDefaultsTest(unittest.TestCase):
    def test_seed_strategy_experiences_returns_copy(self) -> None:
        experiences = seed_strategy_experiences()
        self.assertEqual(len(experiences), 2)
        experiences[0]["experience_id"] = "mutated"
        self.assertNotEqual(seed_strategy_experiences()[0]["experience_id"], "mutated")

    def test_selection_formulas_are_exposed(self) -> None:
        self.assertIn("primary_score", PRIMARY_FORMULA)
        self.assertIn("tie_break_score", TIE_BREAK_FORMULA)
        self.assertIn("delta_primary_score", DELTA_FORMULA)
        self.assertIn("run_delta_primary_score", RUN_DELTA_FORMULA)
        self.assertEqual(DEFAULT_SESSION_ID, "session-current")

    def test_default_objective_spec_exposes_speedup_template(self) -> None:
        self.assertEqual(DEFAULT_SPEEDUP_OBJECTIVE_SPEC["display_name"], "Speedup vs baseline")
        self.assertEqual(
            DEFAULT_SPEEDUP_OBJECTIVE_SPEC["formula"],
            "speedup_vs_baseline = baseline_ms / candidate_ms",
        )


if __name__ == "__main__":
    unittest.main()
