from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "benchmark"
REGISTRY_PATH = BENCHMARK_ROOT / "registry.json"
DEFAULT_BRANCHING_FACTOR = 4
VALID_BENCHMARK_TIERS = {"experiment", "comparable"}
REQUIRED_TASK_FIELDS = (
    "benchmark_tier",
    "track",
    "answer_metric",
    "editable_file",
    "entry_symbol",
    "verifier",
)


def _speedup_objective_spec() -> dict[str, str]:
    return {
        "display_name": "Speedup vs baseline",
        "direction": "max",
        "unit": "x",
        "summary_template": "Higher speedup is better. This task maximizes runtime gain over the checked-in baseline.",
        "formula": "speedup_vs_baseline = baseline_ms / candidate_ms",
    }


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
    normalized["entry_symbol"] = str(normalized.get("entry_symbol") or normalized.get("function_name") or "solve")
    normalized["function_name"] = str(normalized.get("function_name") or normalized["entry_symbol"])
    normalized["editable_file"] = str(normalized.get("editable_file") or "editable.py")
    normalized["editable_filename"] = Path(normalized["editable_file"]).name
    normalized["included_in_main_comparison"] = normalized["benchmark_tier"] == "comparable"
    normalized["source_type"] = str(normalized.get("source_type") or "benchmark-task")
    normalized["local_dataset_only"] = bool(normalized.get("local_dataset_only"))
    split = normalized.get("split")
    normalized["split"] = str(split).strip() if isinstance(split, str) and split.strip() else None
    item_manifest = normalized.get("item_manifest")
    normalized["item_manifest"] = str(item_manifest).strip() if isinstance(item_manifest, str) and item_manifest.strip() else None
    normalized["prompt_context"] = str(normalized.get("prompt_context") or "")
    normalized["verifier_path"] = str(normalized["verifier_path"])
    normalized["editable_path"] = str(normalized["editable_path"])
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
        if not item_manifest_path.exists():
            raise FileNotFoundError(f"Question manifest not found: {item_manifest_path}")
        merged["item_manifest_path"] = str(item_manifest_path)
    data_file = task.get("data_file")
    if isinstance(data_file, str) and data_file.strip():
        merged["data_path"] = str(task_dir / data_file)
        merged["data"] = json.loads((task_dir / data_file).read_text())
    readme_path = task_dir / "README.md"
    if readme_path.exists():
        merged["readme_path"] = str(readme_path)
    return _normalize_task(merged)


def _sort_key(task: dict[str, Any]) -> tuple[int, str, str]:
    return (0 if task["included_in_main_comparison"] else 1, task["track"], task["id"])


SEED_STRATEGY_EXPERIENCES: list[dict[str, Any]] = [
    {
        "experience_id": "exp-correctness-first",
        "experience_type": "strategy_experience",
        "experience_outcome": "success",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["single-file-optimization", "deterministic-eval", "correctness-first"],
        "verifier_status": "pass",
        "rejection_reason": "",
        "failure_pattern": "benchmark-only selection promoted fast but incorrect candidates",
        "strategy_hypothesis": "Run deterministic correctness gates before trusting optimization gains.",
        "successful_strategy": "preserve the public contract first, then optimize the editable file under the verifier",
        "prompt_fragment": "Preserve correctness first, then optimize only under the deterministic verifier contract.",
        "tool_trace_summary": "materialize candidate file -> deterministic verifier -> score -> selective write-back",
        "delta_J": 0.18,
        "proposal_model": "seed",
        "candidate_summary": "Valid candidates only enter the comparison lane.",
        "reusable_rules": ["correctness_first", "deterministic_scoring", "single_file_mutation"],
        "supporting_memory_ids": [],
    },
    {
        "experience_id": "exp-semantics-before-shortcuts",
        "experience_type": "strategy_experience",
        "experience_outcome": "failure",
        "source_task": "seed",
        "source_session_id": "seed-catalog",
        "family": "agnostic",
        "task_signature": ["single-file-optimization", "deterministic-eval", "semantics-preservation"],
        "verifier_status": "fail",
        "rejection_reason": "A shortcut violated the contract and failed the deterministic checks.",
        "failure_pattern": "Aggressive rewrites improved one metric while breaking task semantics.",
        "strategy_hypothesis": "When the verifier is strict, local semantics-preserving rewrites dominate speculative shortcuts.",
        "successful_strategy": "Keep the contract fixed and prefer rewrites that remain faithful to the benchmark spec.",
        "prompt_fragment": "Do not trade away task semantics for a shortcut; keep the benchmark contract intact.",
        "tool_trace_summary": "shortcut rewrite -> deterministic failure -> reject -> prefer semantics-preserving rewrite",
        "delta_J": -0.24,
        "proposal_model": "seed",
        "candidate_summary": "A fast-looking rewrite that violated the verifier contract.",
        "reusable_rules": ["preserve_semantics", "respect_verifier_contract"],
        "supporting_memory_ids": [],
    },
]


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
            "generation_budget": task["generation_budget"],
            "candidate_budget": task["candidate_budget"],
            "branching_factor": task["branching_factor"],
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
