from __future__ import annotations

import argparse
import base64
import json
import pickle
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
MANIFEST_PATH = DATA_DIR / "questions.json"
FULL_DATASET_SIZE = 1055
RAW_BASE_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
RAW_FILES = (
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
)
EXPECTED_ANSWER = "Pass all public and private tests."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local LiveCodeBench manifest prefix.")
    parser.add_argument(
        "--items",
        type=int,
        default=FULL_DATASET_SIZE,
        help="How many items to materialize locally.",
    )
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


def _iter_remote_rows(filename: str) -> Iterator[dict[str, Any]]:
    request = urllib.request.Request(
        f"{RAW_BASE_URL}/{filename}",
        headers={"User-Agent": "autoresearcher/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            yield json.loads(raw_line.decode("utf-8"))


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


def _build_manifest_item(problem: dict[str, Any]) -> dict[str, Any]:
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
        },
    }


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    existing = _load_existing_manifest()
    existing_items = list(existing.get("items") or []) if isinstance(existing, dict) else []
    if len(existing_items) >= requested_items:
        print(f"Manifest already covers {len(existing_items)} items; requested {requested_items}.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    global_index = 0
    for filename in RAW_FILES:
        for source_row_index, row in enumerate(_iter_remote_rows(filename)):
            global_index += 1
            item_id = f"livecodebench-{global_index:04d}"
            problem = _build_problem_record(
                item_id,
                row,
                source_file=filename,
                source_row_index=source_row_index,
            )
            _write_json(PROBLEMS_DIR / f"{item_id}.json", problem)
            items.append(_build_manifest_item(problem))
            if global_index >= requested_items:
                break
        if global_index >= requested_items:
            break

    manifest = {
        "dataset_id": "livecodebench_release_v6",
        "release": "release_v6",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} LiveCodeBench items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
