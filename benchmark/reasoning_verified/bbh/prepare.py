from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
FULL_DATASET_SIZE = 6511
DATASET_ID = "bigbenchhard_all"
DATASET_URL_TEMPLATE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{config}.json"
CONFIGS = [
    "tracking_shuffled_objects_seven_objects",
    "salient_translation_error_detection",
    "tracking_shuffled_objects_three_objects",
    "geometric_shapes",
    "object_counting",
    "word_sorting",
    "logical_deduction_five_objects",
    "hyperbaton",
    "sports_understanding",
    "logical_deduction_seven_objects",
    "multistep_arithmetic_two",
    "ruin_names",
    "causal_judgement",
    "logical_deduction_three_objects",
    "formal_fallacies",
    "snarks",
    "boolean_expressions",
    "reasoning_about_colored_objects",
    "dyck_languages",
    "navigate",
    "disambiguation_qa",
    "temporal_sequences",
    "web_of_lies",
    "tracking_shuffled_objects_five_objects",
    "penguins_in_a_table",
    "movie_recommendation",
    "date_understanding",
]
OPTION_LINE_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$")
BULLET_OPTION_RE = re.compile(r"^-\s+(.+?)\s*$")
OPTION_LABEL_RE = re.compile(r"^\(?([A-Z])\)?$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize BIG-Bench Hard into questions.json.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    parser.add_argument("--subset", action="append", default=[], help="Restrict to one or more BBH configs.")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _normalize_answer_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split()).strip()
    return text.casefold()


def _selected_configs(subsets: list[str]) -> list[str]:
    if not subsets:
        return list(CONFIGS)
    requested = [str(value).strip() for value in subsets if str(value).strip()]
    unknown = [value for value in requested if value not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown BBH subset(s): {', '.join(unknown)}")
    return requested


def _load_subset(config: str) -> list[dict[str, Any]]:
    url = DATASET_URL_TEMPLATE.format(config=config)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"BBH subset {config} did not return an examples list.")
    return [dict(row) for row in examples]


def _extract_choices(prompt: str) -> tuple[list[str], dict[str, int] | None]:
    lines = prompt.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "options:":
            start_index = index + 1
            break
    if start_index is None:
        return [], None

    lettered: list[tuple[str, str]] = []
    bullets: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            continue
        option_match = OPTION_LINE_RE.match(stripped)
        if option_match:
            lettered.append((option_match.group(1), option_match.group(2).strip()))
            continue
        bullet_match = BULLET_OPTION_RE.match(stripped)
        if bullet_match:
            bullets.append(bullet_match.group(1).strip())
            continue
        if lettered or bullets:
            break

    if lettered:
        labels = {label: idx for idx, (label, _) in enumerate(lettered)}
        return [text for _, text in lettered], labels
    if bullets:
        return bullets, None
    return [], None


def _normalize_item(prompt: str, target: str) -> tuple[str, list[str], dict[str, Any]]:
    choices, labeled_choices = _extract_choices(prompt)
    metadata: dict[str, Any] = {"answer_aliases": [target]}
    expected_answer = target

    if choices:
        metadata["answer_format"] = "choice"
        option_match = OPTION_LABEL_RE.match(target.strip())
        if option_match and labeled_choices:
            label = option_match.group(1)
            correct_choice_index = labeled_choices.get(label)
            if correct_choice_index is not None:
                metadata["correct_choice_index"] = correct_choice_index
                metadata["correct_choice_text"] = choices[correct_choice_index]
                return expected_answer, choices, metadata

        normalized_target = _normalize_answer_text(target)
        for index, choice in enumerate(choices):
            if _normalize_answer_text(choice) == normalized_target:
                metadata["correct_choice_index"] = index
                metadata["correct_choice_text"] = choice
                return expected_answer, choices, metadata
        return expected_answer, choices, metadata

    metadata["answer_format"] = "short_text"
    return expected_answer, [], metadata


def main() -> None:
    args = _parse_args()
    configs = _selected_configs(list(args.subset))
    requested_items = max(1, int(args.items or FULL_DATASET_SIZE))

    items: list[dict[str, Any]] = []
    for config in configs:
        rows = _load_subset(config)
        for source_index, row in enumerate(rows):
            if len(items) >= requested_items:
                break
            prompt = str(row.get("input") or "").strip()
            target = str(row.get("target") or "").strip()
            expected_answer, choices, extra_metadata = _normalize_item(prompt, target)
            item = {
                "item_id": f"bbh-{config}-{source_index:03d}",
                "name": f"BBH {config} {source_index + 1}",
                "prompt": prompt,
                "expected_answer": expected_answer,
                "metadata": {
                    "dataset": "bbh",
                    "source_config": config,
                    "source_split": "train",
                    "source_index": source_index,
                    "runtime_split_tags": [f"config:{config}"],
                    **extra_metadata,
                },
            }
            if choices:
                item["choices"] = choices
            items.append(item)
        if len(items) >= requested_items:
            break

    manifest = {
        "dataset_id": DATASET_ID,
        "split": "train",
        "dataset_size": FULL_DATASET_SIZE if configs == CONFIGS else len(items),
        "prepared_count": len(items),
        "configs": configs,
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} BIG-Bench Hard items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
