from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.codegen.or_benchmarks import _dataset_rows, _load_local_or_dataset_rows


class OrBenchmarksTest(unittest.TestCase):
    def test_load_local_or_dataset_rows_normalizes_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "questions.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "nl4opt-test-0001",
                                "prompt": "Solve this OR model.",
                                "expected_answer": 42,
                            }
                        ]
                    }
                )
            )

            self.assertEqual(
                _load_local_or_dataset_rows(manifest_path),
                [
                    {
                        "item_id": "nl4opt-test-0001",
                        "en_question": "Solve this OR model.",
                        "en_answer": 42,
                    }
                ],
            )

    def test_dataset_rows_prefers_local_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir) / "benchmark" / "or_verified" / "nl4opt"
            manifest_path = task_dir / "data" / "questions.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "item_id": "nl4opt-test-0001",
                                "prompt": "Question 1",
                                "expected_answer": 7,
                            },
                            {
                                "item_id": "nl4opt-test-0002",
                                "prompt": "Question 2",
                                "expected_answer": 8,
                            },
                        ]
                    }
                )
            )

            rows, source = _dataset_rows(
                task={"task_dir": str(task_dir)},
                config={},
                max_items=1,
            )

            self.assertEqual(source, str(manifest_path))
            self.assertEqual(
                rows,
                [
                    {
                        "item_id": "nl4opt-test-0001",
                        "en_question": "Question 1",
                        "en_answer": 7,
                    }
                ],
            )

    def test_dataset_rows_runs_prepare_when_local_manifest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir) / "benchmark" / "or_verified" / "industryor"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "prepare.py").write_text(
                "from pathlib import Path\n"
                "import json\n"
                "data_dir = Path(__file__).with_name('data')\n"
                "data_dir.mkdir(parents=True, exist_ok=True)\n"
                "(data_dir / 'questions.json').write_text(json.dumps({'items': [{'item_id': 'x', 'prompt': 'Prepared question', 'expected_answer': 9}]}))\n"
            )

            rows, source = _dataset_rows(
                task={"id": "industryor", "task_dir": str(task_dir)},
                config={},
                max_items=1,
            )

            self.assertEqual(source, str(task_dir / "data" / "questions.json"))
            self.assertEqual(
                rows,
                [
                    {
                        "item_id": "x",
                        "en_question": "Prepared question",
                        "en_answer": 9,
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
