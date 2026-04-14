from __future__ import annotations

import argparse
import base64
import json
import pickle
import urllib.request
import zlib
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


HF_DATASET_REPO = "livecodebench/code_generation_lite"
HF_DATASET_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"
EXPECTED_ANSWER = "Pass all public and private tests."
CACHE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "coding_verified" / "_downloads" / "livecodebench"
RELEASE_FILES = {
    "v1": ["test.jsonl"],
    "v2": ["test2.jsonl"],
    "v3": ["test3.jsonl"],
    "v4": ["test4.jsonl"],
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local LiveCodeBench manifest prefix.")
    parser.add_argument(
        "--items",
        type=int,
        default=None,
        help="How many items to materialize locally.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return None


def _public_tests(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [dict(item) for item in parsed]
    if isinstance(value, list):
        return [dict(item) for item in value]
    return []


def _private_tests(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    raw = base64.b64decode(str(value))
    decoded = zlib.decompress(raw, 15)
    payload = pickle.loads(decoded)
    if isinstance(payload, str):
        parsed = json.loads(payload)
    else:
        parsed = payload
    if not isinstance(parsed, list):
        raise ValueError("Decoded private test cases were not a list.")
    return [dict(item) for item in parsed]


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return dict(parsed)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _evaluation_mode(public_tests: list[dict[str, Any]], metadata: dict[str, Any], platform: str) -> str:
    if isinstance(metadata.get("func_name"), str) and str(metadata["func_name"]).strip():
        return "functional"
    first = public_tests[0] if public_tests else {}
    if str(first.get("testtype") or "").strip().lower() == "functional":
        return "functional"
    if platform.lower() == "leetcode":
        return "functional"
    return "stdin"


def _download_dataset_file(file_name: str) -> Path:
    if hf_hub_download is not None:
        return Path(hf_hub_download(repo_id=HF_DATASET_REPO, filename=file_name, repo_type="dataset"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / file_name
    if destination.exists():
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    url = f"{HF_DATASET_RESOLVE_BASE}/{file_name}?download=1"
    with urllib.request.urlopen(url, timeout=120) as response:
        temp_path.write_bytes(response.read())
    temp_path.replace(destination)
    return destination


def _iter_remote_rows(release_version: str):
    file_names = RELEASE_FILES.get(release_version)
    if not file_names:
        supported = ", ".join(sorted(RELEASE_FILES))
        raise ValueError(f"Unsupported LiveCodeBench shard: {release_version}. Expected one of: {supported}")
    for file_name in file_names:
        local_path = _download_dataset_file(file_name)
        with local_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                yield file_name, json.loads(line)


def _starter_preview(starter_code: str) -> str:
    compact = " ".join(str(starter_code or "").split())
    if len(compact) <= 220:
        return compact
    return compact[:217].rstrip() + "..."


def _tests_preview(cases: list[dict[str, Any]], limit: int = 2) -> str:
    if not cases:
        return "[]"
    preview = cases[:limit]
    return json.dumps(preview, ensure_ascii=True)


def _build_context(problem: dict[str, Any]) -> str:
    parts = [
        f"Platform: {problem['platform']}.",
        f"Difficulty: {problem['difficulty'] or 'unknown'}.",
        f"Evaluation mode: {problem['evaluation_mode']}.",
    ]
    if problem["question_title"]:
        parts.append(f"Title: {problem['question_title']}.")
    if problem["contest_id"]:
        parts.append(f"Contest id: {problem['contest_id']}.")
    if problem["contest_date"]:
        parts.append(f"Contest date: {problem['contest_date']}.")
    if problem["function_name"]:
        parts.append(f"Required method: Solution.{problem['function_name']}.")
    if problem["starter_code"]:
        parts.append(f"Starter code: {_starter_preview(problem['starter_code'])}")
    parts.append(
        f"Public tests ({len(problem['public_test_cases'])}): {_tests_preview(problem['public_test_cases'])}"
    )
    parts.append(f"Hidden tests: {len(problem['private_test_cases'])}.")
    return " ".join(part for part in parts if part)


def _build_problem_record(item_id: str, row: dict[str, Any], *, source_file: str, source_row_index: int) -> dict[str, Any]:
    metadata = _metadata(row.get("metadata"))
    public_test_cases = _public_tests(row.get("public_test_cases"))
    private_test_cases = _private_tests(row.get("private_test_cases"))
    platform = str(row.get("platform") or "")
    evaluation_mode = _evaluation_mode(public_test_cases, metadata, platform)
    function_name = str(metadata.get("func_name") or "").strip() or None
    return {
        "item_id": item_id,
        "source_file": source_file,
        "source_row_index": source_row_index,
        "question_title": str(row.get("question_title") or "").strip(),
        "question_content": str(row.get("question_content") or "").strip(),
        "platform": platform,
        "question_id": str(row.get("question_id") or "").strip(),
        "contest_id": str(row.get("contest_id") or "").strip(),
        "contest_date": str(row.get("contest_date") or "").strip(),
        "difficulty": str(row.get("difficulty") or "").strip(),
        "starter_code": str(row.get("starter_code") or ""),
        "metadata": metadata,
        "evaluation_mode": evaluation_mode,
        "function_name": function_name,
        "public_test_cases": public_test_cases,
        "private_test_cases": private_test_cases,
    }


def _build_manifest_item(problem: dict[str, Any], *, metadata_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    extra = dict(metadata_extra or {})
    problem_file = f"problems/{problem['item_id']}.json"
    return {
        "item_id": problem["item_id"],
        "name": problem["question_title"] or problem["item_id"],
        "prompt": problem["question_content"] or problem["question_title"] or problem["item_id"],
        "context": _build_context(problem),
        "expected_answer": EXPECTED_ANSWER,
        "metadata": {
            "problem_file": problem_file,
            "platform": problem["platform"],
            "evaluation_mode": problem["evaluation_mode"],
            "function_name": problem["function_name"],
            "question_title": problem["question_title"],
            "difficulty": problem["difficulty"],
            "contest_id": problem["contest_id"],
            "contest_date": problem["contest_date"],
            "public_test_count": len(problem["public_test_cases"]),
            "private_test_count": len(problem["private_test_cases"]),
            **extra,
        },
    }


def prepare_livecodebench_shard(
    *,
    task_root: Path,
    task_id: str,
    release_version: str,
    full_dataset_size: int,
) -> None:
    data_dir = task_root / "data"
    problems_dir = data_dir / "problems"
    manifest_path = data_dir / "questions.json"
    source_info_path = data_dir / "source_info.json"

    args = _parse_args()
    requested_items = full_dataset_size if args.items is None else int(args.items or full_dataset_size)
    requested_items = max(1, min(requested_items, full_dataset_size))

    existing = _load_existing_manifest(manifest_path)
    existing_items = list(existing.get("items") or []) if isinstance(existing, dict) else []
    if len(existing_items) >= requested_items:
        print(f"Manifest already covers {len(existing_items)} items; requested {requested_items}.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    problems_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for source_row_index, (file_name, row) in enumerate(_iter_remote_rows(release_version), start=1):
        item_id = f"{task_id}-{source_row_index:04d}"
        problem = _build_problem_record(
            item_id,
            row,
            source_file=f"{HF_DATASET_REPO}:{release_version}:{file_name}",
            source_row_index=source_row_index - 1,
        )
        _write_json(problems_dir / f"{item_id}.json", problem)
        items.append(_build_manifest_item(problem))
        if source_row_index >= requested_items:
            break

    manifest = {
        "dataset_id": f"livecodebench_{release_version}",
        "release": release_version,
        "dataset_size": full_dataset_size,
        "prepared_count": len(items),
        "items": items,
    }
    source_info = {
        "benchmark": task_id,
        "sources": {
            "dataset": f"https://huggingface.co/datasets/{HF_DATASET_REPO}",
            "official_repo": "https://github.com/LiveCodeBench/LiveCodeBench",
        },
        "release": release_version,
        "prepared_count": len(items),
        "dataset_size": full_dataset_size,
        "evaluator_semantics": "official_lcb_runner_testing_util",
    }
    _write_json(manifest_path, manifest)
    _write_json(source_info_path, source_info)
    print(f"Wrote {len(items)} LiveCodeBench items to {manifest_path}.")


def prepare_livecodebench_collection(
    *,
    task_root: Path,
    task_id: str,
    releases: list[tuple[str, int]],
) -> None:
    data_dir = task_root / "data"
    problems_dir = data_dir / "problems"
    manifest_path = data_dir / "questions.json"
    source_info_path = data_dir / "source_info.json"

    args = _parse_args()
    full_dataset_size = sum(size for _, size in releases)
    requested_items = full_dataset_size if args.items is None else int(args.items or full_dataset_size)
    requested_items = max(1, min(requested_items, full_dataset_size))

    existing = _load_existing_manifest(manifest_path)
    existing_items = list(existing.get("items") or []) if isinstance(existing, dict) else []
    if len(existing_items) >= requested_items:
        print(f"Manifest already covers {len(existing_items)} items; requested {requested_items}.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    problems_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    global_index = 0
    release_counts: dict[str, int] = {}
    for release_version, release_size in releases:
        release_counts[release_version] = release_size
        for release_index, (file_name, row) in enumerate(_iter_remote_rows(release_version), start=1):
            global_index += 1
            item_id = f"{task_id}-{global_index:04d}"
            problem = _build_problem_record(
                item_id,
                row,
                source_file=f"{HF_DATASET_REPO}:{release_version}:{file_name}",
                source_row_index=release_index - 1,
            )
            _write_json(problems_dir / f"{item_id}.json", problem)
            items.append(
                _build_manifest_item(
                    problem,
                    metadata_extra={
                        "release_version": release_version,
                        "runtime_split_tags": [f"release:{release_version}"],
                    },
                )
            )
            if global_index >= requested_items:
                break
        if global_index >= requested_items:
            break

    manifest = {
        "dataset_id": "livecodebench_all",
        "split": "+".join(release for release, _ in releases),
        "dataset_size": full_dataset_size,
        "prepared_count": len(items),
        "release_counts": release_counts,
        "items": items,
    }
    source_info = {
        "benchmark": task_id,
        "sources": {
            "dataset": f"https://huggingface.co/datasets/{HF_DATASET_REPO}",
            "official_repo": "https://github.com/LiveCodeBench/LiveCodeBench",
        },
        "releases": [release for release, _ in releases],
        "release_counts": release_counts,
        "prepared_count": len(items),
        "dataset_size": full_dataset_size,
        "evaluator_semantics": "official_lcb_runner_testing_util",
    }
    _write_json(manifest_path, manifest)
    _write_json(source_info_path, source_info)
    print(f"Wrote {len(items)} LiveCodeBench items to {manifest_path}.")
