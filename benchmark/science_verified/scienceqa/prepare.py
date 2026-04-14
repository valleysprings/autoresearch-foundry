from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize a text-only ScienceQA subset into questions.json.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--topics", default="biology,chemistry,physics")
    args = parser.parse_args()

    allowed_topics = {topic.strip() for topic in args.topics.split(",") if topic.strip()}
    dataset = load_dataset("derek-thomas/ScienceQA", split=args.split)
    items = []
    for source_index, row in enumerate(dataset):
        if str(row.get("subject") or "").strip() != "natural science":
            continue
        topic = str(row.get("topic") or "").strip()
        if topic not in allowed_topics:
            continue
        if row.get("image") is not None:
            continue
        choices = [str(choice).strip() for choice in row["choices"]]
        correct_choice_index = int(row["answer"])
        if correct_choice_index < 0 or correct_choice_index >= len(choices):
            raise ValueError(f"ScienceQA row {source_index} has invalid answer index {correct_choice_index}.")
        correct = choices[correct_choice_index]
        hint = str(row.get("hint") or "").strip()
        lecture = str(row.get("lecture") or "").strip()
        context_parts = []
        if hint:
            context_parts.append(f"Hint: {hint}")
        if lecture:
            context_parts.append(f"Lecture: {lecture}")
        context = "\n\n".join(context_parts) or None
        item_index = len(items)
        items.append(
            {
                "item_id": f"scienceqa-{args.split}-{item_index}",
                "name": f"ScienceQA {topic} {item_index + 1}",
                "prompt": str(row["question"]).strip(),
                "context": context,
                "choices": choices,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "scienceqa",
                    "source_split": args.split,
                    "source_index": source_index,
                    "subject": "natural science",
                    "topic": topic,
                    "category": str(row.get("category") or "").strip(),
                    "skill": str(row.get("skill") or "").strip(),
                    "grade": str(row.get("grade") or "").strip(),
                    "task": str(row.get("task") or "").strip(),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": [correct],
                    "runtime_split_tags": [f"topic:{topic}"],
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
