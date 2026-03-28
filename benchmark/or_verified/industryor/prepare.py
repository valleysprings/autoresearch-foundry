from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "CardinalOperations/IndustryOR"
ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize IndustryOR into data/questions.json.")
    parser.add_argument("--output", default=str(ROOT / "data" / "questions.json"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME, split=args.split)
    items = []
    for index, row in enumerate(dataset, start=1):
        item_id = f"industryor-{args.split}-{index:04d}"
        items.append(
            {
                "item_id": item_id,
                "name": item_id,
                "prompt": str(row.get("en_question") or "").strip(),
                "expected_answer": row.get("en_answer"),
                "metadata": {
                    "dataset_name": DATASET_NAME,
                    "source_split": args.split,
                    "source_index": index - 1,
                },
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_NAME,
                "split": args.split,
                "prepared_count": len(items),
                "items": items,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    print(f"Wrote {len(items)} IndustryOR rows to {output_path}")


if __name__ == "__main__":
    main()
