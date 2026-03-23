from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize PlanBench task_1 into questions.json.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset = load_dataset("tasksource/planbench", "task_1_plan_generation", split=args.split)
    items = []
    for index, row in enumerate(dataset, start=1):
        item_id = f"planbench-{row['domain']}-{row['prompt_type']}-{int(row['instance_id']):05d}"
        items.append(
            {
                "item_id": item_id,
                "name": f"{row['domain']} / {row['prompt_type']} / {row['instance_id']}",
                "prompt": str(row["query"]).strip(),
                "context": {
                    "domain": str(row["domain"]),
                    "prompt_type": str(row["prompt_type"]),
                    "instance_id": int(row["instance_id"]),
                },
                "expected_answer": str(row["ground_truth_plan"]).strip(),
                "metadata": {
                    "dataset": "planbench",
                    "config": "task_1_plan_generation",
                    "source_split": args.split,
                    "source_index": index - 1,
                    "domain": str(row["domain"]),
                    "prompt_type": str(row["prompt_type"]),
                    "answer_format": "plan",
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
