from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
DATASET_ID = "mmlu_pro"
DATASET_NAME = "TIGER-Lab/MMLU-Pro"
DATASET_CONFIG = "default"
FULL_DATASET_SIZE = 12032


def _slug(value: str) -> str:
    return str(value).strip().lower().replace(" ", "-")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize MMLU-Pro into questions.json.")
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--split", default="test")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _answer_aliases(correct_text: str, answer_index: int) -> list[str]:
    aliases = [correct_text]
    if 0 <= answer_index < len(string.ascii_uppercase):
        aliases.append(string.ascii_uppercase[answer_index])
    return aliases


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    dataset = load_dataset(DATASET_NAME, args.config, split=args.split)

    items = []
    for source_index, row in enumerate(dataset):
        if source_index >= requested_items:
            break
        choices = [str(choice).strip() for choice in list(row["options"])]
        correct_choice_index = int(row["answer_index"])
        if correct_choice_index < 0 or correct_choice_index >= len(choices):
            raise ValueError(f"MMLU-Pro row {source_index} has invalid answer index {correct_choice_index}.")
        correct = choices[correct_choice_index]
        question_id = row.get("question_id")
        item_id_suffix = str(question_id if question_id is not None else source_index)
        category = str(row.get("category") or "").strip()
        items.append(
            {
                "item_id": f"mmlu-pro-{args.config}-{args.split}-{item_id_suffix}",
                "question_id": question_id,
                "name": f"MMLU-Pro {category or 'uncategorized'} {source_index + 1}",
                "prompt": str(row["question"]).strip(),
                "choices": choices,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "mmlu-pro",
                    "source_config": args.config,
                    "source_split": args.split,
                    "source_index": source_index,
                    "category": category,
                    "src": str(row.get("src") or "").strip(),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": _answer_aliases(correct, correct_choice_index),
                    "runtime_split_tags": [f"category:{_slug(category)}"] if category else [],
                },
            }
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "config": args.config,
        "split": args.split,
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} MMLU-Pro items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
