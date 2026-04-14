from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.entries import runner


SAMPLE_TASK = {
    "id": "livecodebench",
    "title": "LiveCodeBench",
    "description": "Synthetic coding benchmark summary for CLI tests.",
    "family": "coding",
    "function_name": "solve",
    "entry_symbol": "solve",
    "editable_file": "editable.py",
    "answer_metric": "test_pass_rate",
    "objective_label": "Test pass rate",
    "objective_direction": "max",
    "objective_spec": {
        "display_name": "Test pass rate",
        "direction": "max",
        "unit": "ratio",
        "formula": "test_pass_rate = passed_cases / total_cases",
    },
    "selection_spec": {
        "primary_formula": "primary_score = objective_score",
        "gate_summary": "gate: verifier_status == 'pass'",
        "tie_break_formula": "tie_break_score = 0.0 (no auxiliary tie-break metrics configured)",
        "archive_summary": "archive_features = none",
    },
    "generation_budget": 3,
    "candidate_budget": 2,
    "branching_factor": 3,
    "item_workers": 6,
    "benchmark_tier": "comparable",
    "track": "coding_verified",
    "dataset_id": "livecodebench_all",
    "dataset_size": 1055,
    "local_dataset_only": True,
    "split": "v1+v2+v3+v4+v5+v6:test",
    "task_mode": "artifact",
    "interaction_mode": "single_turn",
    "included_in_main_comparison": True,
    "supports_runtime_config": False,
    "suite_run_config": None,
    "runtime_split_selector": {
        "label": "Split",
        "default_value": "all",
        "options": [
            {"value": "all", "title": "All Releases", "item_count": 1055},
            {"value": "v6", "title": "v6", "item_count": 175, "match_tags_any": ["release:v6"]},
        ],
    },
    "supports_max_items": True,
    "default_max_items": 1055,
    "supports_max_episodes": False,
    "default_max_episodes": None,
}


class RunnerCliTest(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            runner.main(argv)
        return buffer.getvalue()

    def test_tasks_command_returns_api_shaped_json(self) -> None:
        with patch.object(runner, "list_codegen_task_summaries", return_value=[dict(SAMPLE_TASK)]):
            output = self._run_cli(["tasks"])
        payload = json.loads(output)
        self.assertIn("tasks", payload)
        self.assertEqual(payload["tasks"][0]["id"], "livecodebench")
        self.assertEqual(payload["tasks"][0]["task_mode"], "artifact")

    def test_tasks_pretty_mode_renders_single_task_detail(self) -> None:
        with patch.object(runner, "list_codegen_task_summaries", return_value=[dict(SAMPLE_TASK)]):
            output = self._run_cli(["tasks", "--task-id", "livecodebench", "--pretty"])
        self.assertIn("task_mode_summary", output)
        self.assertIn("Artifact task", output)
        self.assertIn("objective_formula", output)
        self.assertIn("test_pass_rate = passed_cases / total_cases", output)

    def test_latest_run_command_renders_cached_summary(self) -> None:
        payload = {
            "summary": {
                "generated_at": "2026-03-25T12:00:00+08:00",
                "active_model": "deepseek-chat",
                "num_tasks": 1,
                "total_runs": 1,
                "total_generations": 3,
                "write_backs": 1,
                "experiment_runs": 0,
            },
            "audit": {
                "session_id": "20260325_120000",
                "workspace_root": "runs/workspace/20260325_120000",
                "max_items": 5,
                "max_episodes": None,
            },
            "runs": [
                {
                    "task": {"id": "livecodebench"},
                    "winner": {"metrics": {"objective": 0.75, "primary_score": 0.75}},
                    "delta_primary_score": 0.12,
                }
            ],
        }
        with patch.object(runner, "load_cached_discrete_payload", return_value=payload) as load_cached:
            output = self._run_cli(["latest-run", "--task-id", "livecodebench", "--pretty"])
        load_cached.assert_called_once_with(task_id="livecodebench")
        self.assertIn("generated_at", output)
        self.assertIn("deepseek-chat", output)
        self.assertIn("livecodebench", output)
        self.assertIn("delta_primary_score=0.12", output)

    def test_run_task_returns_payload_json(self) -> None:
        runtime = object()
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "codegen-livecodebench.json"
            artifact_path.write_text(json.dumps({"summary": {"generated_at": "now"}, "runs": []}))
            with (
                patch.object(runner, "_runtime_for_cli", return_value=runtime) as runtime_for_cli,
                patch.object(runner, "write_discrete_artifacts", return_value=artifact_path) as write_discrete_artifacts,
            ):
                output = self._run_cli(["run-task", "--task-id", "livecodebench", "--max-items", "3", "--llm-concurrency", "7"])
        runtime_for_cli.assert_called_once_with(None, 7, None)
        write_discrete_artifacts.assert_called_once_with(
            task_id="livecodebench",
            proposal_runtime=runtime,
            generation_budget=None,
            candidate_budget=None,
            branching_factor=None,
            item_workers=None,
            max_items=3,
            max_episodes=None,
            suite_config=None,
            eval_model=None,
        )
        payload = json.loads(output)
        self.assertEqual(payload["summary"]["generated_at"], "now")

    def test_plan_dataset_smoke_applies_cap_and_small_dataset_rules(self) -> None:
        tasks = [
            {
                "id": "livecodebench",
                "title": "LiveCodeBench",
                "track": "coding_verified",
                "local_dataset_only": True,
                "dataset_size": 1055,
                "prepared_item_count": 80,
                "included_in_main_comparison": True,
                "requires_eval_model": False,
                "default_eval_model": None,
                "description": "Large coding benchmark.",
                "baseline_summary": "Placeholder baseline implementation.",
            },
            {
                "id": "aime-2026",
                "title": "AIME 2026",
                "track": "math_verified",
                "local_dataset_only": True,
                "dataset_size": 30,
                "included_in_main_comparison": True,
                "requires_eval_model": False,
                "default_eval_model": None,
                "description": "Real small benchmark slice.",
                "baseline_summary": "Placeholder solver baseline.",
            },
            {
                "id": "rmtbench",
                "title": "RMTBench",
                "track": "personalization_verified",
                "local_dataset_only": True,
                "dataset_size": 3,
                "included_in_main_comparison": False,
                "requires_eval_model": True,
                "default_eval_model": None,
                "description": "Hidden judged proxy for the official RMTBench full-dialogue role-play protocol.",
                "baseline_summary": "A tiny hidden proxy slice.",
            },
        ]
        with patch.object(runner, "load_codegen_tasks", return_value=tasks):
            output = self._run_cli(["plan-dataset-smoke"])
        payload = json.loads(output)
        by_id = {row["task_id"]: row for row in payload["rows"]}
        self.assertEqual(by_id["livecodebench"]["max_items"], 80)
        self.assertEqual(by_id["livecodebench"]["action"], "run")
        self.assertEqual(by_id["livecodebench"]["dataset_size"], 1055)
        self.assertEqual(by_id["livecodebench"]["prepared_count"], 80)
        self.assertEqual(by_id["aime-2026"]["max_items"], 30)
        self.assertEqual(by_id["aime-2026"]["action"], "run")
        self.assertEqual(by_id["rmtbench"]["max_items"], 0)
        self.assertEqual(by_id["rmtbench"]["action"], "skip")

    def test_smoke_test_datasets_requires_eval_model_for_required_tasks(self) -> None:
        tasks = [
            {
                "id": "incharacter",
                "title": "InCharacter",
                "track": "personalization_verified",
                "local_dataset_only": True,
                "dataset_size": 44,
                "included_in_main_comparison": True,
                "requires_eval_model": True,
                "default_eval_model": None,
                "description": "Role-play benchmark.",
                "baseline_summary": "Standard baseline.",
            }
        ]
        with patch.object(runner, "load_codegen_tasks", return_value=tasks):
            with self.assertRaises(SystemExit) as exc:
                self._run_cli(["smoke-test-datasets"])
        self.assertEqual(exc.exception.code, 1)

    def test_audit_datasets_reports_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            verifier_path = task_dir / "verifier.py"
            verifier_path.write_text("VALUE = 1\n")
            readme_path = task_dir / "README.md"
            readme_path.write_text("# test\n")
            task_path = task_dir / "task.json"
            task_path.write_text("{}\n")
            (task_dir / "prepare.py").write_text("print('ok')\n")
            payload_task = {
                "id": "demo-task",
                "track": "science_verified",
                "local_dataset_only": True,
                "dataset_size": 100,
                "prepared_item_count": 80,
                "task_dir": str(task_dir),
                "task_path": str(task_path),
                "readme_path": str(readme_path),
                "editable_path": str(task_dir / "editable.py"),
                "verifier_path": str(verifier_path),
                "item_manifest": "data/questions.json",
            }
            (task_dir / "editable.py").write_text("def solve(question):\n    return ''\n")
            (task_dir / "data").mkdir()
            (task_dir / "data" / "questions.json").write_text(json.dumps({"items": [{} for _ in range(80)]}))
            with patch.object(runner, "load_codegen_tasks", return_value=[payload_task]):
                output = self._run_cli(["audit-datasets"])
        payload = json.loads(output)
        self.assertEqual(payload["summary"]["count_mismatches"], ["demo-task"])
        self.assertEqual(payload["rows"][0]["verifier_import"], True)


if __name__ == "__main__":
    unittest.main()
