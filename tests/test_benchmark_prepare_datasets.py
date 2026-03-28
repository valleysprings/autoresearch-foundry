from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "prepare_datasets.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("benchmark_prepare_datasets", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkPrepareDatasetsTest(unittest.TestCase):
    def test_list_reports_prepare_status(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_root = Path(tmp_dir) / "benchmark"
            registry_path = benchmark_root / "registry.json"
            task_with_prepare = benchmark_root / "track" / "task-with-prepare"
            task_without_prepare = benchmark_root / "track" / "task-without-prepare"
            task_with_prepare.mkdir(parents=True, exist_ok=True)
            task_without_prepare.mkdir(parents=True, exist_ok=True)
            (task_with_prepare / "prepare.py").write_text("print('ok')\n")
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "task-with-prepare", "path": "track/task-with-prepare", "enabled": True},
                            {"id": "task-without-prepare", "path": "track/task-without-prepare", "enabled": True},
                        ]
                    }
                )
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.main(["--benchmark-root", str(benchmark_root), "--list"])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("task-with-prepare\tprepare=yes", output)
            self.assertIn("task-without-prepare\tprepare=no", output)

    def test_selected_prepare_script_is_executed(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_root = Path(tmp_dir) / "benchmark"
            registry_path = benchmark_root / "registry.json"
            task_dir = benchmark_root / "track" / "co-bench"
            marker_path = task_dir / "prepared.txt"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "prepare.py").write_text(
                "from pathlib import Path\n"
                "Path(__file__).with_name('prepared.txt').write_text('done')\n"
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "co-bench", "path": "track/co-bench", "enabled": True},
                        ]
                    }
                )
            )

            exit_code = module.main(["--benchmark-root", str(benchmark_root), "--task-id", "co-bench"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(marker_path.read_text(), "done")

    def test_missing_enabled_local_dataset_without_prepare_fails(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_root = Path(tmp_dir) / "benchmark"
            registry_path = benchmark_root / "registry.json"
            task_dir = benchmark_root / "track" / "missing-data"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "missing-data",
                        "local_dataset_only": True,
                        "item_manifest": "data/questions.json",
                    }
                )
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "missing-data", "path": "track/missing-data", "enabled": True},
                        ]
                    }
                )
            )

            exit_code = module.main(["--benchmark-root", str(benchmark_root)])

            self.assertEqual(exit_code, 1)

    def test_prepared_local_dataset_without_prepare_is_skipped(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_root = Path(tmp_dir) / "benchmark"
            registry_path = benchmark_root / "registry.json"
            task_dir = benchmark_root / "track" / "ready-data"
            manifest_path = task_dir / "data" / "questions.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"items": []}))
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "ready-data",
                        "local_dataset_only": True,
                        "item_manifest": "data/questions.json",
                    }
                )
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "ready-data", "path": "track/ready-data", "enabled": True},
                        ]
                    }
                )
            )

            exit_code = module.main(["--benchmark-root", str(benchmark_root)])

            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
