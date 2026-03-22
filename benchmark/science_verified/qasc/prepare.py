from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize QASC validation into questions.json.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    dataset = load_dataset("allenai/qasc", split=args.split)
    items = []
    for index, row in enumerate(dataset, start=1):
        choice_labels = [str(label).strip() for label in row["choices"]["label"]]
        choice_texts = [str(text).strip() for text in row["choices"]["text"]]
        answer_key = str(row["answerKey"]).strip()
        try:
            correct_choice_index = choice_labels.index(answer_key)
        except ValueError as exc:
            raise ValueError(f"QASC row {index} is missing answerKey {answer_key!r} in choices.") from exc
        correct = choice_texts[correct_choice_index]
        fact1 = str(row.get("fact1") or "").strip()
        fact2 = str(row.get("fact2") or "").strip()
        context_parts = []
        if fact1:
            context_parts.append(f"Fact 1: {fact1}")
        if fact2:
            context_parts.append(f"Fact 2: {fact2}")
        context = "\n".join(context_parts) or None
        items.append(
            {
                "item_id": f"qasc-{args.split}-{index - 1}",
                "name": f"QASC {args.split} {index}",
                "prompt": str(row["question"]).strip(),
                "context": context,
                "choices": choice_texts,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "qasc",
                    "source_split": args.split,
                    "source_index": index - 1,
                    "source_id": str(row.get("id") or ""),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": [correct],
                    "hop_count": 2,
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
