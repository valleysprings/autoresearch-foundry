from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
FULL_DATASET_SIZE = 90


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize aggregate AIME 2024/2025/2026 rows into questions.json.")
    parser.add_argument("--output", default=str(MANIFEST_PATH))
    return parser.parse_args()


def _aime_2024_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_dataset("HuggingFaceH4/aime_2024", split="train"):
        item = dict(row)
        item["year"] = 2024
        item["source_split"] = str(item.get("source_split") or "train")
        rows.append(item)
    return rows


def _aime_2025_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_name, config_name in (("I", "AIME2025-I"), ("II", "AIME2025-II")):
        for row in load_dataset("opencompass/AIME2025", config_name, split="test"):
            item = dict(row)
            item["year"] = 2025
            item["round"] = round_name
            item["source_split"] = f"{config_name}:test"
            rows.append(item)
    return rows


def _aime_2026_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_dataset("math-ai/aime26", split="test"):
        item = dict(row)
        item["year"] = 2026
        item["source_split"] = str(item.get("source_split") or "test")
        rows.append(item)
    return rows


def main() -> None:
    args = _parse_args()
    grouped_rows = [
        (2024, _aime_2024_rows()),
        (2025, _aime_2025_rows()),
        (2026, _aime_2026_rows()),
    ]
    items: list[dict[str, Any]] = []
    for year, rows in grouped_rows:
        for year_index, row in enumerate(rows, start=1):
            index = len(items)
            prompt = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
            answer = row.get("answer")
            if not prompt or answer is None:
                raise ValueError(f"AIME {year} row {index + 1} must declare prompt and answer.")
            metadata = {
                "dataset": "aime",
                "source_split": str(row.get("source_split") or ""),
                "source_index": index,
                "subject": str(row.get("subject") or "competition-math"),
                "answer_format": "numeric",
                "year": year,
                "runtime_split_tags": [f"year:{year}"],
            }
            round_name = str(row.get("round") or "").strip()
            if round_name:
                metadata["round"] = round_name
            source_id = row.get("id")
            if source_id is not None:
                metadata["source_id"] = str(source_id)
            items.append(
                {
                    "item_id": f"aime-{year}-{year_index:02d}",
                    "name": f"AIME {year} Question {year_index}",
                    "prompt": prompt,
                    "expected_answer": str(answer).strip(),
                    "metadata": metadata,
                }
            )

    payload = {
        "dataset_id": "aime_all_years",
        "split": "2024+2025+2026",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(Path(args.output), payload)
    print(f"Wrote {len(items)} aggregate AIME items to {args.output}.")


if __name__ == "__main__":
    main()
