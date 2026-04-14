from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
DATASET_ID = "acpbench_bool_mcq"
DATASET_REPO = "ibm-research/acp_bench"

BOOL_CONFIGS = {
    "acp_app_bool": 130,
    "acp_areach_bool": 120,
    "acp_just_bool": 130,
    "acp_land_bool": 130,
    "acp_prog_bool": 130,
    "acp_reach_bool": 130,
    "acp_val_bool": 130,
}
MCQ_CONFIGS = {
    "acp_app_mcq": 130,
    "acp_areach_mcq": 120,
    "acp_just_mcq": 130,
    "acp_land_mcq": 130,
    "acp_prog_mcq": 130,
    "acp_reach_mcq": 130,
    "acp_val_mcq": 130,
}
CONFIG_ORDER = list(BOOL_CONFIGS) + list(MCQ_CONFIGS)
FULL_DATASET_SIZE = sum(BOOL_CONFIGS.values()) + sum(MCQ_CONFIGS.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize ACPBench bool/mcq test items.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _manifest_item(config_name: str, source_index: int, row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    context = str(row.get("context") or "").strip()
    answer = row.get("answer")
    group = str(row.get("group") or "").strip()
    if not question or answer in (None, ""):
        raise ValueError(f"ACPBench {config_name} row {source_index} is missing question or answer.")
    format_name = "bool" if config_name.endswith("_bool") else "mcq"
    task_name = config_name.removeprefix("acp_")
    metadata = {
        "dataset": "acpbench",
        "source_config": config_name,
        "source_split": "test",
        "source_index": source_index,
        "group": group,
        "query": str(row.get("query") or "").strip(),
        "answer_format": format_name,
        "runtime_split_tags": [f"format:{format_name}", f"config:{config_name}", f"task:{task_name}"],
    }
    item = {
        "item_id": f"acpbench-{config_name}-{source_index:04d}",
        "name": f"ACPBench {config_name} {source_index + 1}",
        "prompt": question,
        "context": context,
        "expected_answer": str(answer).strip() if format_name == "bool" else "",
        "metadata": metadata,
    }
    if format_name == "bool":
        normalized = str(answer).strip().lower()
        if normalized not in {"yes", "no"}:
            raise ValueError(f"ACPBench {config_name} row {source_index} has invalid boolean answer {answer!r}.")
        item["expected_answer"] = normalized
        metadata["answer_aliases"] = [normalized]
        return item

    choices = dict(row.get("choices") or {})
    labels = [str(label).strip() for label in list(choices.get("label") or [])]
    texts = [str(text).strip() for text in list(choices.get("text") or [])]
    answer_label = str(answer).strip()
    if not labels or len(labels) != len(texts):
        raise ValueError(f"ACPBench {config_name} row {source_index} is missing aligned multiple-choice labels/text.")
    try:
        correct_choice_index = labels.index(answer_label)
    except ValueError as exc:
        raise ValueError(f"ACPBench {config_name} row {source_index} is missing answer label {answer_label!r}.") from exc
    correct_text = texts[correct_choice_index]
    item["choices"] = texts
    item["expected_answer"] = correct_text
    metadata["correct_choice_index"] = correct_choice_index
    metadata["answer_aliases"] = [answer_label, correct_text]
    return item


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    items: list[dict[str, Any]] = []
    for config_name in CONFIG_ORDER:
        dataset = load_dataset(DATASET_REPO, name=config_name, split="test")
        for source_index, row in enumerate(dataset):
            items.append(_manifest_item(config_name, source_index, dict(row)))
            if len(items) >= requested_items:
                break
        if len(items) >= requested_items:
            break

    manifest = {
        "dataset_id": DATASET_ID,
        "split": "test:bool+mcq",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "configs": CONFIG_ORDER,
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} ACPBench bool/mcq items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
