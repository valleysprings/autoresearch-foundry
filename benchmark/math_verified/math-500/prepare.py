from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize MATH-500 rows into questions.json.")
    parser.add_argument("--input", help="Optional path to a local JSON list with problem/answer fields.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    args = parser.parse_args()

    if args.input:
        payload = json.loads(Path(args.input).read_text())
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON list of rows.")
    else:
        payload = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))

    items = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        prompt = str(row.get("problem") or row.get("prompt") or "").strip()
        answer = row.get("answer")
        if not prompt or answer is None:
            raise ValueError(f"Row {index} must declare problem/prompt and answer.")
        items.append(
            {
                "item_id": str(row.get("item_id") or f"math-500-{index:03d}"),
                "name": str(row.get("name") or f"MATH-500 seed {index}"),
                "prompt": prompt,
                "expected_answer": str(answer).strip(),
                "metadata": {
                    "dataset": "math-500",
                    "source_split": str(row.get("source_split") or "test"),
                    "source_index": index - 1,
                    "subject": str(row.get("subject") or row.get("year") or "unknown"),
                    "answer_format": str(row.get("answer_format") or "symbolic"),
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
