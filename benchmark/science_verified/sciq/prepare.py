from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize SciQ validation into questions.json.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    dataset = load_dataset("allenai/sciq", split=args.split)
    items = []
    for index, row in enumerate(dataset, start=1):
        correct = str(row["correct_answer"]).strip()
        choices = [
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
            correct,
        ]
        items.append(
            {
                "item_id": f"sciq-{args.split}-{index - 1}",
                "name": f"SciQ {args.split} {index}",
                "prompt": str(row["question"]).strip(),
                "context": str(row.get("support") or "").strip() or None,
                "choices": choices,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "sciq",
                    "source_split": args.split,
                    "source_index": index - 1,
                    "correct_choice_index": 3,
                    "answer_aliases": [correct],
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
