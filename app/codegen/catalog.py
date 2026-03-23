from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.configs.codegen import (
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_EDITABLE_FILE,
    DEFAULT_ENTRY_SYMBOL,
    DEFAULT_SOURCE_TYPE,
    REQUIRED_TASK_FIELDS,
    SEED_STRATEGY_EXPERIENCES,
    VALID_BENCHMARK_TIERS,
    speedup_objective_spec,
)
from app.codegen.selection import selection_spec_for_task
from app.configs.paths import BENCHMARK_ROOT as CONFIG_BENCHMARK_ROOT
from app.configs.paths import REGISTRY_PATH as CONFIG_REGISTRY_PATH

BENCHMARK_ROOT = CONFIG_BENCHMARK_ROOT
REGISTRY_PATH = CONFIG_REGISTRY_PATH

TRACK_ORDER = {
    "math_verified": 0,
    "planning_verified": 1,
    "deepsearch_verified": 2,
    "browse_snapshot": 3,
    "science_verified": 4,
    "multihop_qa_snapshot": 5,
    "terminal_verified": 6,
    "coding_verified": 7,
    "small_experiments": 8,
}
TASK_ORDER = {
    "olymmath": 0,
    "math-500": 1,
    "aime-2024": 2,
    "aime-2025": 3,
    "aime-2026": 4,
    "planbench": 5,
    "sciq": 6,
    "qasc": 7,
    "scienceqa": 8,
    "livecodebench": 9,
}


def _speedup_objective_spec() -> dict[str, str]:
    return speedup_objective_spec()


def _count_manifest_items(path: Path) -> int:
    payload = json.loads(path.read_text())
    rows = payload.get("items") if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Question manifest must contain a list of items: {path}")
    return len(rows)


def _normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task)
    missing = [field for field in REQUIRED_TASK_FIELDS if not isinstance(normalized.get(field), str) or not str(normalized.get(field)).strip()]
    if missing:
        raise ValueError(f"Task {normalized.get('id') or '<unknown>'} is missing required fields: {', '.join(missing)}")

    benchmark_tier = str(normalized["benchmark_tier"]).strip()
    if benchmark_tier not in VALID_BENCHMARK_TIERS:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid benchmark_tier={benchmark_tier!r}; "
            f"expected one of {sorted(VALID_BENCHMARK_TIERS)}."
        )

    objective_spec = dict(task.get("objective_spec") or {})
    if not objective_spec:
        objective_spec = _speedup_objective_spec()
    normalized["objective_spec"] = objective_spec
    normalized["objective_label"] = normalized.get("objective_label") or objective_spec["display_name"]
    normalized["objective_direction"] = normalized.get("objective_direction") or objective_spec["direction"]
    normalized["branching_factor"] = int(normalized.get("branching_factor", DEFAULT_BRANCHING_FACTOR))
    normalized["benchmark_tier"] = benchmark_tier
    normalized["track"] = str(normalized["track"]).strip()
    normalized["answer_metric"] = str(normalized["answer_metric"]).strip()
    normalized["dataset_id"] = str(normalized.get("dataset_id") or normalized["id"])
    normalized["dataset_size"] = int(normalized.get("dataset_size") or 0)
    normalized["entry_symbol"] = str(normalized.get("entry_symbol") or normalized.get("function_name") or DEFAULT_ENTRY_SYMBOL)
    normalized["function_name"] = str(normalized.get("function_name") or normalized["entry_symbol"])
    normalized["editable_file"] = str(normalized.get("editable_file") or DEFAULT_EDITABLE_FILE)
    normalized["editable_filename"] = Path(normalized["editable_file"]).name
    normalized["included_in_main_comparison"] = normalized["benchmark_tier"] == "comparable"
    normalized["source_type"] = str(normalized.get("source_type") or DEFAULT_SOURCE_TYPE)
    normalized["local_dataset_only"] = bool(normalized.get("local_dataset_only"))
    split = normalized.get("split")
    normalized["split"] = str(split).strip() if isinstance(split, str) and split.strip() else None
    item_manifest = normalized.get("item_manifest")
    normalized["item_manifest"] = str(item_manifest).strip() if isinstance(item_manifest, str) and item_manifest.strip() else None
    normalized["lazy_item_manifest"] = bool(normalized.get("lazy_item_manifest"))
    normalized["prompt_context"] = str(normalized.get("prompt_context") or "")
    normalized["allow_browsing"] = bool(normalized.get("allow_browsing", False))
    normalized["verifier_path"] = str(normalized["verifier_path"])
    normalized["editable_path"] = str(normalized["editable_path"])
    normalized["selection_spec"] = selection_spec_for_task(normalized)
    if normalized["local_dataset_only"]:
        if normalized["dataset_size"] <= 0:
            raise ValueError(f"Dataset task {normalized['id']} must declare dataset_size > 0.")
        if normalized["item_manifest"] is None:
            raise ValueError(f"Dataset task {normalized['id']} must declare item_manifest.")
    return normalized


def _registry_entries() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Benchmark registry is missing: {REGISTRY_PATH}")
    payload = json.loads(REGISTRY_PATH.read_text())
    entries = payload.get("tasks")
    if not isinstance(entries, list):
        raise ValueError("benchmark/registry.json must contain a top-level 'tasks' list.")
    return [dict(entry) for entry in entries]


def _load_task(entry: dict[str, Any]) -> dict[str, Any]:
    relative_path = entry.get("path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Every benchmark registry entry must declare a non-empty path.")
    task_dir = BENCHMARK_ROOT / relative_path
    task_path = task_dir / "task.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_path}")
    task = json.loads(task_path.read_text())
    if not isinstance(task, dict):
        raise ValueError(f"Task spec must be a JSON object: {task_path}")
    track_from_path = Path(relative_path).parts[0]
    if isinstance(task.get("track"), str) and task["track"] != track_from_path:
        raise ValueError(
            f"Task {task.get('id') or '<unknown>'} declares track={task['track']!r} "
            f"but registry path lives under {track_from_path!r}."
        )
    merged = {**task, "task_dir": str(task_dir), "task_path": str(task_path)}
    merged["editable_path"] = str(task_dir / str(task.get("editable_file") or ""))
    merged["verifier_path"] = str(task_dir / str(task.get("verifier") or ""))
    if not Path(merged["editable_path"]).exists():
        raise FileNotFoundError(f"Editable file not found: {merged['editable_path']}")
    if not Path(merged["verifier_path"]).exists():
        raise FileNotFoundError(f"Verifier file not found: {merged['verifier_path']}")
    item_manifest = task.get("item_manifest")
    if isinstance(item_manifest, str) and item_manifest.strip():
        item_manifest_path = task_dir / item_manifest
        if not item_manifest_path.exists() and not bool(task.get("lazy_item_manifest")):
            raise FileNotFoundError(f"Question manifest not found: {item_manifest_path}")
        merged["item_manifest_path"] = str(item_manifest_path)
        if item_manifest_path.exists():
            prepared_item_count = _count_manifest_items(item_manifest_path)
            merged["prepared_item_count"] = prepared_item_count
            if not bool(task.get("lazy_item_manifest")):
                merged["dataset_size"] = prepared_item_count
    data_file = task.get("data_file")
    if isinstance(data_file, str) and data_file.strip():
        merged["data_path"] = str(task_dir / data_file)
        merged["data"] = json.loads((task_dir / data_file).read_text())
    readme_path = task_dir / "README.md"
    if readme_path.exists():
        merged["readme_path"] = str(readme_path)
    return _normalize_task(merged)


def _sort_key(task: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        0 if task["included_in_main_comparison"] else 1,
        TRACK_ORDER.get(task["track"], len(TRACK_ORDER)),
        TASK_ORDER.get(task["id"], 999),
        task["track"],
        task["id"],
    )

def load_codegen_tasks(
    task_id: str | None = None,
    *,
    included_in_main_comparison: bool | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for entry in _registry_entries():
        if not bool(entry.get("enabled", True)):
            continue
        try:
            tasks.append(_load_task(entry))
        except FileNotFoundError:
            continue
    if task_id is not None:
        tasks = [task for task in tasks if task["id"] == task_id]
    if included_in_main_comparison is not None:
        tasks = [task for task in tasks if task["included_in_main_comparison"] == included_in_main_comparison]
    return sorted(tasks, key=_sort_key)


def list_codegen_task_summaries() -> list[dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "function_name": task["function_name"],
            "entry_symbol": task["entry_symbol"],
            "editable_file": task["editable_file"],
            "answer_metric": task["answer_metric"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "objective_spec": task["objective_spec"],
            "selection_spec": task["selection_spec"],
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "branching_factor": task["branching_factor"],
            "item_workers": int(task.get("item_workers") or 20),
            "benchmark_tier": task["benchmark_tier"],
            "track": task["track"],
            "dataset_id": task["dataset_id"],
            "dataset_size": task["dataset_size"],
            "local_dataset_only": task["local_dataset_only"],
            "split": task["split"],
            "included_in_main_comparison": task["included_in_main_comparison"],
        }
        for task in load_codegen_tasks()
    ]


def seed_strategy_experiences() -> list[dict[str, Any]]:
    return [dict(item) for item in SEED_STRATEGY_EXPERIENCES]
