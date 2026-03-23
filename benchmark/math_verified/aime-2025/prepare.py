from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _default_payload() -> list[dict]:
    rows: list[dict] = []
    for round_name, config_name in (("I", "AIME2025-I"), ("II", "AIME2025-II")):
        dataset = load_dataset("opencompass/AIME2025", config_name, split="test")
        for row in dataset:
            item = dict(row)
            item["round"] = round_name
            item["source_split"] = f"{config_name}:test"
            rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize AIME 2025 rows into questions.json.")
    parser.add_argument("--input", help="Optional path to a local JSON list with question/answer fields.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    args = parser.parse_args()

    if args.input:
        payload = json.loads(Path(args.input).read_text())
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON list of rows.")
    else:
        payload = _default_payload()

    items = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        prompt = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
        answer = row.get("answer")
        if not prompt or answer is None:
            raise ValueError(f"Row {index} must declare question/prompt and answer.")
        items.append(
            {
                "item_id": str(row.get("item_id") or f"aime-2025-{index:02d}"),
                "name": str(row.get("name") or f"AIME 2025 Question {index}"),
                "prompt": prompt,
                "expected_answer": str(answer).strip(),
                "metadata": {
                    "dataset": "aime-2025",
                    "source_split": str(row.get("source_split") or "test"),
                    "source_index": index - 1,
                    "subject": str(row.get("subject") or "competition-math"),
                    "answer_format": "numeric",
                    "year": 2025,
                    "round": str(row.get("round") or ""),
                },
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
