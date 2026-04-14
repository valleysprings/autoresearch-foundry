from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from app.bench.planbench_support import (
    derive_verification_verdict_from_query,
    normalize_verification_verdict,
    plan_step_count,
    verification_answer_aliases,
)


DATASET_ID = "tasksource/planbench"
SUPPORTED_CONFIGS = {
    "planbench-t1": "task_1_plan_generation",
    "planbench-t2": "task_2_plan_optimality",
    "planbench-t3": "task_3_plan_verification",
}
PARQUET_FILENAMES = {
    "task_1_plan_generation": {"train": ("task_1_plan_generation/train-00000-of-00001-f765a1b29ae17c5a.parquet",)},
    "task_2_plan_optimality": {"train": ("task_2_plan_optimality/train-00000-of-00001-4f4028566ce201a0.parquet",)},
    "task_3_plan_verification": {"train": ("task_3_plan_verification/train-00000-of-00001-7e91305090db7039.parquet",)},
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _normalize_row_value(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _normalize_row_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _normalize_row_value(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_normalize_row_value(inner) for inner in value]
    if isinstance(value, list):
        return [_normalize_row_value(inner) for inner in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _parquet_files(config_name: str, split: str) -> list[str]:
    matches = list(PARQUET_FILENAMES.get(config_name, {}).get(split, ()))
    if not matches:
        raise FileNotFoundError(f"Unable to locate parquet shards for {DATASET_ID}:{config_name}:{split}")
    return matches


def load_rows(task_id: str, *, split: str = "train") -> list[dict[str, Any]]:
    config_name = SUPPORTED_CONFIGS[task_id]
    rows: list[dict[str, Any]] = []
    for filename in _parquet_files(config_name, split):
        parquet_path = hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=filename)
        frame = pd.read_parquet(parquet_path)
        for raw_row in frame.to_dict(orient="records"):
            rows.append({str(key): _normalize_row_value(value) for key, value in raw_row.items()})
    return rows


def _item_id(task_id: str, row: dict[str, Any], *, config_name: str) -> str:
    prompt_type = str(row.get("prompt_type") or "default").strip()
    item_id = f"{task_id}-{row['domain']}-{prompt_type}-{int(row['instance_id']):05d}"
    source_task = str(row.get("task") or config_name).strip()
    if source_task and source_task != config_name:
        item_id = f"{item_id}-{source_task}"
    return item_id


def _base_item(task_id: str, row: dict[str, Any], *, config_name: str, split: str, index: int) -> dict[str, Any]:
    source_task = str(row.get("task") or config_name)
    domain = str(row["domain"])
    prompt_type = str(row.get("prompt_type") or "default")
    return {
        "item_id": _item_id(task_id, row, config_name=config_name),
        "name": f"{domain} / {prompt_type} / {row['instance_id']}",
        "context": {
            "domain": domain,
            "prompt_type": prompt_type,
            "instance_id": int(row["instance_id"]),
            "task": source_task,
        },
        "metadata": {
            "dataset": "planbench",
            "benchmark_task_id": task_id,
            "config": config_name,
            "source_split": split,
            "source_index": index,
            "domain": domain,
            "prompt_type": prompt_type,
            "task": source_task,
            "source_task": source_task,
            "runtime_split_tags": [
                f"domain:{domain}",
                f"prompt_type:{prompt_type}",
            ],
        },
    }


def _plan_item(task_id: str, row: dict[str, Any], *, config_name: str, split: str, index: int, optimality: bool) -> dict[str, Any]:
    item = _base_item(task_id, row, config_name=config_name, split=split, index=index)
    metadata = dict(item["metadata"])
    metadata["answer_format"] = "plan"
    example_instance_ids = row.get("example_instance_ids")
    if isinstance(example_instance_ids, list) and example_instance_ids:
        metadata["example_instance_ids"] = [int(value) for value in example_instance_ids]
    if optimality:
        metadata["optimal_plan_steps"] = plan_step_count(str(row["ground_truth_plan"]).strip())
    item["metadata"] = metadata
    item["prompt"] = str(row["query"]).strip()
    item["expected_answer"] = str(row["ground_truth_plan"]).strip()
    return item


def _verification_item(task_id: str, row: dict[str, Any], *, config_name: str, split: str, index: int) -> dict[str, Any]:
    item = _base_item(task_id, row, config_name=config_name, split=split, index=index)
    official_verification_raw = row.get("ground_truth_plan")
    if official_verification_raw is None or not str(official_verification_raw).strip():
        verdict, semantic_detail = derive_verification_verdict_from_query(row.get("query"), item)
        verdict_source = "semantic_derivation"
        official_verification = None
    else:
        verdict = normalize_verification_verdict(official_verification_raw)
        semantic_detail = None
        verdict_source = "dataset_label"
        official_verification = str(official_verification_raw).strip()
    aliases = verification_answer_aliases(verdict)
    metadata = dict(item["metadata"])
    metadata.update(
        {
            "answer_format": "choice",
            "correct_choice_index": 0 if verdict == "yes" else 1,
            "answer_aliases": aliases,
            "official_verification": official_verification,
            "verdict_source": verdict_source,
        }
    )
    if semantic_detail:
        metadata["semantic_verification_detail"] = semantic_detail
    item["metadata"] = metadata
    item["prompt"] = (
        f"{str(row['query']).strip()}\n\n"
        "Answer only yes or no: is the plan valid?"
    )
    item["choices"] = ["yes", "no"]
    item["expected_answer"] = verdict
    return item


def build_items(task_id: str, *, split: str = "train") -> list[dict[str, Any]]:
    config_name = SUPPORTED_CONFIGS[task_id]
    items: list[dict[str, Any]] = []
    for index, raw_row in enumerate(load_rows(task_id, split=split)):
        if task_id == "planbench-t1":
            item = _plan_item(task_id, raw_row, config_name=config_name, split=split, index=index, optimality=False)
        elif task_id == "planbench-t2":
            item = _plan_item(task_id, raw_row, config_name=config_name, split=split, index=index, optimality=True)
        elif task_id == "planbench-t3":
            item = _verification_item(task_id, raw_row, config_name=config_name, split=split, index=index)
        else:
            raise ValueError(f"Unsupported PlanBench task id: {task_id}")
        items.append(item)
    return items


def write_manifest(task_id: str, output_path: Path, *, split: str = "train") -> int:
    items = build_items(task_id, split=split)
    payload = {
        "dataset_id": f"planbench_{SUPPORTED_CONFIGS[task_id]}",
        "split": f"{SUPPORTED_CONFIGS[task_id]}:{split}",
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(output_path, payload)
    return len(items)
