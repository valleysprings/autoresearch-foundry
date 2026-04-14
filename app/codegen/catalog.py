from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.configs.codegen import (
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_EDITABLE_FILE,
    DEFAULT_ENTRY_SYMBOL,
    REQUIRED_TASK_FIELDS,
    SEED_STRATEGY_EXPERIENCES,
    VALID_BENCHMARK_TIERS,
    speedup_objective_spec,
)
from app.bench.runtime_support import load_value_from_candidate
from app.codegen.selection import selection_spec_for_task
from app.codegen.task_contracts import (
    infer_interaction_mode,
    infer_scoring_mode,
    infer_task_mode,
    infer_task_shape,
)
from app.configs.paths import BENCHMARK_ROOT as CONFIG_BENCHMARK_ROOT
from app.configs.paths import REGISTRY_PATH as CONFIG_REGISTRY_PATH

BENCHMARK_ROOT = CONFIG_BENCHMARK_ROOT
REGISTRY_PATH = CONFIG_REGISTRY_PATH

TRACK_ORDER = {
    "math_verified": 0,
    "reasoning_verified": 1,
    "text2sql_verified": 2,
    "longcontext_verified": 3,
    "personalization_verified": 4,
    "safety_verified": 5,
    "browse_snapshot": 6,
    "science_verified": 7,
    "coding_verified": 8,
    "or_verified": 9,
    "agent_verified": 10,
}

VALID_RUNTIME_SPLIT_SELECTOR_OPTION_FIELDS = frozenset(
    {"value", "title", "description", "item_count", "match_tags_any"}
)
VALID_SAFETY_CATEGORIES = frozenset(
    {
        "jailbreak_attack",
        "over_refusal",
        "factuality_hallucination",
        "policy_drift",
        "benign_utility",
    }
)
VALID_SAFETY_FOCUS = frozenset(set(VALID_SAFETY_CATEGORIES) | {"should_refuse", "safety_degradation"})
TASK_ORDER = {
    "olymmath": 0,
    "math-500": 1,
    "aime": 2,
    "aime-2024": 3,
    "aime-2025": 4,
    "aime-2026": 5,
    "planbench-t1": 6,
    "planbench-t2": 7,
    "planbench-t3": 8,
    "acpbench": 9,
    "arc-challenge": 10,
    "bbh": 11,
    "mmlu-pro": 12,
    "longbench-v2": 13,
    "incharacter": 14,
    "characterbench": 15,
    "socialbench": 16,
    "timechara": 17,
    "rmtbench": 18,
    "personamem-32k": 19,
    "personafeedback": 20,
    "alpsbench-extraction": 21,
    "alpsbench-update": 22,
    "alpsbench-retrieval": 23,
    "alpsbench-utilization": 24,
    "alpbench": 25,
    "xstest-refusal-calibration": 26,
    "harmbench-text-harmful": 27,
    "jailbreakbench-harmful": 28,
    "or-bench-hard-1k": 29,
    "or-bench-toxic": 30,
    "hallulens-precisewikiqa": 31,
    "hallulens-mixedentities": 32,
    "hallulens-longwiki": 33,
    "longsafety": 34,
    "tom-gibbs-multiturn-jailbreak": 35,
    "tau-bench-retail": 36,
    "tau-bench-airline": 37,
    "sciq": 38,
    "qasc": 39,
    "scienceqa": 40,
    "openbookqa": 41,
    "gpqa-diamond": 42,
    "livecodebench": 43,
    "co-bench": 44,
    "alfworld": 45,
}


def _normalize_runtime_split_selector(task_id: str, raw_value: Any) -> dict[str, Any] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError(f"Task {task_id} has invalid runtime_split_selector={raw_value!r}; expected an object.")

    label = str(raw_value.get("label") or "").strip()
    default_value = str(raw_value.get("default_value") or "").strip()
    raw_options = raw_value.get("options")
    if not label:
        raise ValueError(f"Task {task_id} runtime_split_selector must declare a non-empty label.")
    if not default_value:
        raise ValueError(f"Task {task_id} runtime_split_selector must declare a non-empty default_value.")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f"Task {task_id} runtime_split_selector must declare a non-empty options list.")

    seen_values: set[str] = set()
    options: list[dict[str, Any]] = []
    for index, option in enumerate(raw_options, start=1):
        if not isinstance(option, dict):
            raise ValueError(f"Task {task_id} runtime_split_selector option {index} must be an object.")
        unknown_fields = set(option) - VALID_RUNTIME_SPLIT_SELECTOR_OPTION_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Task {task_id} runtime_split_selector option {index} has unsupported fields: {sorted(unknown_fields)}."
            )
        value = str(option.get("value") or "").strip()
        title = str(option.get("title") or "").strip()
        description = str(option.get("description") or "").strip() or None
        if not value or not title:
            raise ValueError(f"Task {task_id} runtime_split_selector option {index} must declare value and title.")
        if value in seen_values:
            raise ValueError(f"Task {task_id} runtime_split_selector repeats option value {value!r}.")
        seen_values.add(value)
        raw_item_count = option.get("item_count")
        if raw_item_count is None:
            item_count = None
        else:
            try:
                item_count = int(raw_item_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Task {task_id} runtime_split_selector option {value!r} has invalid item_count={raw_item_count!r}."
                ) from exc
            if item_count <= 0:
                raise ValueError(f"Task {task_id} runtime_split_selector option {value!r} must have item_count > 0.")
        raw_match_tags = option.get("match_tags_any")
        if raw_match_tags is None:
            match_tags_any: list[str] = []
        elif isinstance(raw_match_tags, list):
            match_tags_any = [str(tag).strip() for tag in raw_match_tags if str(tag).strip()]
        else:
            raise ValueError(
                f"Task {task_id} runtime_split_selector option {value!r} has invalid match_tags_any={raw_match_tags!r}."
            )
        options.append(
            {
                "value": value,
                "title": title,
                "description": description,
                "item_count": item_count,
                "match_tags_any": match_tags_any,
            }
        )
    if default_value not in seen_values:
        raise ValueError(
            f"Task {task_id} runtime_split_selector default_value={default_value!r} is not present in options."
        )
    return {"label": label, "default_value": default_value, "options": options}

def _speedup_objective_spec() -> dict[str, str]:
    return speedup_objective_spec()


def _count_manifest_items(path: Path) -> int:
    payload = json.loads(path.read_text())
    rows = payload.get("items") if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Question manifest must contain a list of items: {path}")
    return len(rows)


def _infer_safety_category(task: dict[str, Any]) -> str | None:
    if str(task.get("track") or "").strip() != "safety_verified":
        return None
    task_id = str(task.get("id") or "").strip().lower()
    inferred: dict[str, str] = {
        "xstest-refusal-calibration": "over_refusal",
        "harmbench-text-harmful": "jailbreak_attack",
        "jailbreakbench-harmful": "jailbreak_attack",
        "or-bench-hard-1k": "over_refusal",
        "or-bench-toxic": "jailbreak_attack",
        "hallulens-precisewikiqa": "factuality_hallucination",
        "hallulens-mixedentities": "factuality_hallucination",
        "hallulens-longwiki": "factuality_hallucination",
        "longsafety": "jailbreak_attack",
        "tom-gibbs-multiturn-jailbreak": "policy_drift",
        "tau-bench-retail": "policy_drift",
        "tau-bench-airline": "policy_drift",
    }
    return inferred.get(task_id, "safety_degradation")


def _infer_safety_focus(task: dict[str, Any]) -> str | None:
    if str(task.get("track") or "").strip() != "safety_verified":
        return None
    task_id = str(task.get("id") or "").strip().lower()
    if task_id == "or-bench-toxic":
        return "should_refuse"
    if task_id == "longsafety":
        return "safety_degradation"
    return _infer_safety_category(task)


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
    included_in_main_comparison = normalized.get("included_in_main_comparison")
    if included_in_main_comparison is None:
        normalized["included_in_main_comparison"] = True
    elif isinstance(included_in_main_comparison, bool):
        normalized["included_in_main_comparison"] = included_in_main_comparison
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid included_in_main_comparison="
            f"{included_in_main_comparison!r}; expected a boolean when provided."
        )
    normalized["task_mode"] = infer_task_mode(normalized)
    normalized["interaction_mode"] = infer_interaction_mode(normalized)
    normalized["task_shape"] = infer_task_shape(normalized)
    normalized["scoring_mode"] = infer_scoring_mode(normalized)
    research_line = str(normalized.get("research_line") or "").strip()
    if not research_line:
        research_line = "personalization" if normalized["track"] == "personalization_verified" else "general"
    normalized["research_line"] = research_line
    personalization_category = str(normalized.get("personalization_category") or "").strip()
    normalized["personalization_category"] = personalization_category or None
    personalization_focus = str(normalized.get("personalization_focus") or "").strip()
    normalized["personalization_focus"] = personalization_focus or None
    safety_category = str(normalized.get("safety_category") or "").strip()
    if not safety_category:
        safety_category = str(_infer_safety_category(normalized) or "").strip()
    normalized["safety_category"] = safety_category or None
    safety_focus = str(normalized.get("safety_focus") or "").strip()
    if not safety_focus:
        safety_focus = str(_infer_safety_focus(normalized) or "").strip()
    normalized["safety_focus"] = safety_focus or None
    if normalized["track"] == "safety_verified":
        if normalized["safety_category"] not in VALID_SAFETY_CATEGORIES:
            raise ValueError(
                f"Task {normalized['id']} must declare safety_category in {sorted(VALID_SAFETY_CATEGORIES)}."
            )
        if normalized["safety_focus"] not in VALID_SAFETY_FOCUS:
            raise ValueError(
                f"Task {normalized['id']} must declare safety_focus in {sorted(VALID_SAFETY_FOCUS)}."
            )
    else:
        if normalized["safety_category"] is not None and normalized["safety_category"] not in VALID_SAFETY_CATEGORIES:
            raise ValueError(
                f"Task {normalized['id']} has invalid safety_category={normalized['safety_category']!r}; "
                f"expected one of {sorted(VALID_SAFETY_CATEGORIES)}."
            )
        if normalized["safety_focus"] is not None and normalized["safety_focus"] not in VALID_SAFETY_FOCUS:
            raise ValueError(
                f"Task {normalized['id']} has invalid safety_focus={normalized['safety_focus']!r}; "
                f"expected one of {sorted(VALID_SAFETY_FOCUS)}."
            )
    supports_eval_model = normalized.get("supports_eval_model")
    if supports_eval_model is None:
        normalized["supports_eval_model"] = False
    elif isinstance(supports_eval_model, bool):
        normalized["supports_eval_model"] = supports_eval_model
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid supports_eval_model={supports_eval_model!r}; "
            "expected a boolean when provided."
        )
    requires_eval_model = normalized.get("requires_eval_model")
    if requires_eval_model is None:
        normalized["requires_eval_model"] = False
    elif isinstance(requires_eval_model, bool):
        normalized["requires_eval_model"] = requires_eval_model
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid requires_eval_model={requires_eval_model!r}; "
            "expected a boolean when provided."
        )
    if normalized["requires_eval_model"] and not normalized["supports_eval_model"]:
        raise ValueError(
            f"Task {normalized['id']} declares requires_eval_model=true but supports_eval_model is not true."
        )
    default_eval_model = normalized.get("default_eval_model")
    if default_eval_model is None:
        normalized["default_eval_model"] = None
    else:
        parsed_default_eval_model = str(default_eval_model).strip()
        normalized["default_eval_model"] = parsed_default_eval_model or None
    if normalized["default_eval_model"] and not normalized["supports_eval_model"]:
        raise ValueError(
            f"Task {normalized['id']} declares default_eval_model but does not support eval_model."
        )
    normalized["local_dataset_only"] = bool(normalized.get("local_dataset_only"))
    split = normalized.get("split")
    normalized["split"] = str(split).strip() if isinstance(split, str) and split.strip() else None
    item_manifest = normalized.get("item_manifest")
    normalized["item_manifest"] = str(item_manifest).strip() if isinstance(item_manifest, str) and item_manifest.strip() else None
    normalized["lazy_item_manifest"] = bool(normalized.get("lazy_item_manifest"))
    normalized["prompt_context"] = str(normalized.get("prompt_context") or "")
    normalized["allow_browsing"] = bool(normalized.get("allow_browsing", False))
    normalized["runtime_split_selector"] = _normalize_runtime_split_selector(
        str(normalized.get("id") or "<unknown>"),
        normalized.get("runtime_split_selector"),
    )
    raw_run_baseline_verifier = normalized.get("run_baseline_verifier")
    normalized["run_baseline_verifier"] = True if raw_run_baseline_verifier is None else bool(raw_run_baseline_verifier)
    normalized["verifier_path"] = str(normalized["verifier_path"])
    normalized["editable_path"] = str(normalized["editable_path"])
    normalized["selection_spec"] = selection_spec_for_task(normalized)
    if normalized["local_dataset_only"]:
        if normalized["dataset_size"] <= 0:
            raise ValueError(f"Dataset task {normalized['id']} must declare dataset_size > 0.")
        if normalized["item_manifest"] is None:
            raise ValueError(f"Dataset task {normalized['id']} must declare item_manifest.")
    if normalized["runtime_split_selector"] is not None:
        if not normalized["local_dataset_only"] or normalized["interaction_mode"] != "single_turn":
            raise ValueError(
                f"Task {normalized['id']} runtime_split_selector is only supported for single-turn dataset tasks."
            )
    return normalized


def _suite_run_config(task: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(task.get("runtime_suite_config"), dict):
        return None
    config = dict(load_value_from_candidate(Path(str(task["editable_path"])), "RUN_CONFIG", {}) or {})
    build_run_config = load_value_from_candidate(Path(str(task["editable_path"])), "build_run_config", None)
    if callable(build_run_config):
        built = build_run_config()
        if isinstance(built, dict):
            config = dict(built)
    override = task.get("runtime_suite_config")
    if isinstance(override, dict):
        config.update(override)
    return config


def _suite_default_max_items(config: dict[str, Any] | None) -> int | None:
    if not config:
        return None
    for key in ("task_limit", "n_tasks", "cases"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    for key in ("task_ids", "problem_names", "task_names", "tasks", "inline_episodes"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _suite_default_max_episodes(config: dict[str, Any] | None) -> int | None:
    if not config:
        return None
    for key in ("episode_limit", "n_episodes", "max_episodes", "task_limit"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    for key in ("episode_ids", "episodes", "inline_episodes"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _task_supports_max_items(task: dict[str, Any]) -> bool:
    return bool(task.get("local_dataset_only")) and task.get("interaction_mode") != "multi_turn"


def _task_default_max_items(task: dict[str, Any], suite_run_config: dict[str, Any] | None) -> int | None:
    del suite_run_config
    if bool(task.get("local_dataset_only")):
        if task.get("interaction_mode") == "multi_turn":
            return None
        size = int(task.get("dataset_size") or 0)
        return size if size > 0 else None
    return None


def _task_supports_max_episodes(task: dict[str, Any]) -> bool:
    return bool(task.get("local_dataset_only")) and task.get("interaction_mode") == "multi_turn"


def _task_default_max_episodes(task: dict[str, Any], suite_run_config: dict[str, Any] | None) -> int | None:
    del suite_run_config
    if _task_supports_max_episodes(task):
        size = int(task.get("dataset_size") or 0)
        return size if size > 0 else None
    return None


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    suite_run_config = _suite_run_config(task)
    return {
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
        "task_mode": task["task_mode"],
        "interaction_mode": task["interaction_mode"],
        "task_shape": task["task_shape"],
        "scoring_mode": task["scoring_mode"],
        "research_line": task["research_line"],
        "personalization_category": task["personalization_category"],
        "personalization_focus": task["personalization_focus"],
        "safety_category": task["safety_category"],
        "safety_focus": task["safety_focus"],
        "supports_eval_model": task["supports_eval_model"],
        "requires_eval_model": task["requires_eval_model"],
        "default_eval_model": task["default_eval_model"],
        "included_in_main_comparison": task["included_in_main_comparison"],
        "run_baseline_verifier": task["run_baseline_verifier"],
        "supports_runtime_config": suite_run_config is not None,
        "suite_run_config": suite_run_config,
        "runtime_split_selector": task.get("runtime_split_selector"),
        "supports_max_items": _task_supports_max_items(task),
        "default_max_items": _task_default_max_items(task, suite_run_config),
        "supports_max_episodes": _task_supports_max_episodes(task),
        "default_max_episodes": _task_default_max_episodes(task, suite_run_config),
    }


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
    declared_track = str(task.get("track") or "").strip()
    if declared_track and declared_track != track_from_path:
        raise ValueError(
            f"Task {task.get('id') or '<unknown>'} declares track={declared_track!r} "
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


def list_missing_local_dataset_warnings() -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for entry in _registry_entries():
        if not bool(entry.get("enabled", True)):
            continue
        relative_path = str(entry.get("path") or "").strip()
        if not relative_path:
            continue
        task_dir = BENCHMARK_ROOT / relative_path
        task_path = task_dir / "task.json"
        if not task_path.exists():
            continue
        try:
            payload = json.loads(task_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not bool(payload.get("local_dataset_only")):
            continue
        item_manifest = str(payload.get("item_manifest") or "").strip()
        if not item_manifest:
            continue
        manifest_path = task_dir / item_manifest
        if manifest_path.exists():
            continue
        task_id = str(payload.get("id") or entry.get("id") or "").strip() or str(entry.get("id") or "")
        title = str(payload.get("title") or task_id).strip() or task_id
        track = str(payload.get("track") or Path(relative_path).parts[0]).strip() or "unknown"
        warnings.append(
            {
                "task_id": task_id,
                "title": title,
                "track": track,
                "manifest_path": str(manifest_path),
                "prepare_command": f"python benchmark/prepare_datasets.py --task-id {task_id}",
                "message": f"Missing local dataset manifest: {manifest_path}",
            }
        )
    return warnings


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
    return [task_summary(task) for task in load_codegen_tasks()]


def seed_strategy_experiences() -> list[dict[str, Any]]:
    return [dict(item) for item in SEED_STRATEGY_EXPERIENCES]
