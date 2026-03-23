from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize OlymMATH en-hard into questions.json.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--config", default="en-hard")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    dataset = load_dataset("RUC-AIBOX/OlymMATH", args.config, split=args.split)
    items = []
    for index, row in enumerate(dataset, start=1):
        items.append(
            {
                "item_id": str(row.get("unique_id") or f"olymmath-{index:03d}"),
                "name": str(row.get("unique_id") or f"OlymMATH {index}"),
                "prompt": str(row["problem"]).strip(),
                "expected_answer": str(row["answer"]).strip(),
                "metadata": {
                    "dataset": "olymmath",
                    "source_split": f"{args.config}:{args.split}",
                    "source_index": index - 1,
                    "subject": str(row.get("subject") or "unknown"),
                    "answer_format": "numeric",
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
