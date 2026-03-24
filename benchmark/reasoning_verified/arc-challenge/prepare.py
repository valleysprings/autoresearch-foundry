from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
FULL_DATASET_SIZE = 299


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local ARC-Challenge manifest prefix.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="ARC-Challenge")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _load_existing_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    payload = json.loads(MANIFEST_PATH.read_text())
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return None


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    existing = _load_existing_manifest()
    existing_items = list(existing.get("items") or []) if isinstance(existing, dict) else []
    if len(existing_items) >= requested_items:
        print(f"Manifest already covers {len(existing_items)} items; requested {requested_items}.")
        return

    dataset = load_dataset("allenai/ai2_arc", args.config, split=args.split)
    items = []
    for source_index, row in enumerate(dataset):
        if source_index >= requested_items:
            break
        choice_labels = [str(label).strip() for label in row["choices"]["label"]]
        choice_texts = [str(text).strip() for text in row["choices"]["text"]]
        answer_key = str(row["answerKey"]).strip()
        try:
            correct_choice_index = choice_labels.index(answer_key)
        except ValueError as exc:
            raise ValueError(f"ARC-Challenge row {source_index} is missing answerKey {answer_key!r} in choices.") from exc
        correct = choice_texts[correct_choice_index]
        items.append(
            {
                "item_id": f"arc-challenge-{args.split}-{source_index}",
                "name": f"ARC-Challenge {args.split} {source_index + 1}",
                "prompt": str(row["question"]).strip(),
                "choices": choice_texts,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "arc-challenge",
                    "source_config": args.config,
                    "source_split": args.split,
                    "source_index": source_index,
                    "source_id": str(row.get("id") or ""),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": [correct],
                },
            }
        )

    manifest = {
        "dataset_id": "ai2_arc_arc_challenge",
        "config": args.config,
        "split": args.split,
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} ARC-Challenge items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
