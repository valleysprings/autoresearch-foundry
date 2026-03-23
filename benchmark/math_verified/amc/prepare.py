from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


LETTER_TO_INDEX = {letter: index for index, letter in enumerate("ABCDE")}
OPTION_PATTERN = re.compile(
    r"(?:\\(?:textbf|text|mathrm)\s*\{?\s*)\(([A-E])\)\s*(.*?)(?=(?:\\(?:textbf|text|mathrm)\s*\{?\s*\([A-E]\))|$)",
    re.S,
)


def _extract_choices(prompt: str) -> tuple[str, list[str]]:
    matches = list(OPTION_PATTERN.finditer(prompt))
    if len(matches) < 5:
        raise ValueError("Could not parse AMC choices from question text.")
    stem = prompt[: matches[0].start()].strip()
    choices: list[str] = []
    for match in matches[:5]:
        choice_text = match.group(2)
        choice_text = choice_text.replace("\\qquad", " ").replace("$", " ").replace("{", " ").replace("}", " ").strip()
        choices.append(" ".join(choice_text.split()))
    return stem, choices


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize AMC rows into questions.json.")
    parser.add_argument("--input", help="Optional path to a local JSON list with problem/choices fields.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    args = parser.parse_args()

    if args.input:
        payload = json.loads(Path(args.input).read_text())
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON list of rows.")
    else:
        payload = list(load_dataset("edev2000/amc12-full", split="train"))

    items = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        prompt = str(row.get("problem") or row.get("prompt") or row.get("question") or "").strip()
        choices = list(row.get("choices") or [])
        parsed_prompt = prompt
        if not choices and prompt:
            try:
                parsed_prompt, choices = _extract_choices(prompt)
            except ValueError:
                choices = []
        if not prompt:
            raise ValueError(f"Row {index} must declare problem/prompt.")
        correct_choice_index = row.get("correct_choice_index")
        answer_letter = str(row.get("answer_letter") or row.get("answer") or "").strip().upper()
        if correct_choice_index is None:
            correct_choice_index = LETTER_TO_INDEX.get(answer_letter)
        if choices and (not isinstance(correct_choice_index, int) or correct_choice_index < 0 or correct_choice_index >= len(choices)):
            raise ValueError(f"Row {index} must declare a valid correct choice.")
        items.append(
            {
                "item_id": str(row.get("item_id") or f"amc-2024-{index:02d}"),
                "name": str(row.get("name") or f"AMC 12 2024 seed {index}"),
                "prompt": parsed_prompt,
                "choices": choices,
                "expected_answer": answer_letter,
                "metadata": {
                    "dataset": "amc",
                    "source_split": str(row.get("source_split") or "train"),
                    "source_index": index - 1,
                    "subject": str(row.get("subject") or "competition-math"),
                    "answer_format": "choice",
                    **({"correct_choice_index": correct_choice_index} if choices else {}),
                    "answer_aliases": list(
                        row.get("answer_aliases")
                        or ([choices[correct_choice_index]] if choices and isinstance(correct_choice_index, int) else [answer_letter])
                    )
                }
            }
        )
    Path(args.output).write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
