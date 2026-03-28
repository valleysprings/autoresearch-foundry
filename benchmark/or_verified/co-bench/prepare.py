from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "questions.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.codegen.co_benchmarks import (  # noqa: E402
    build_co_bench_manifest,
    co_bench_problem_names,
    normalize_co_bench_problem_names,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local CO-Bench dataset snapshot.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--download-max-workers", type=int, default=2)
    parser.add_argument("--items", type=int, default=None, help="Optionally prepare only the first N official problems.")
    parser.add_argument(
        "--problem-name",
        action="append",
        dest="problem_names",
        default=[],
        help="Optional official CO-Bench problem name. Repeat to limit the local snapshot.",
    )
    return parser.parse_args()


def _allow_patterns(problem_names: list[str]) -> list[str] | None:
    if not problem_names:
        return None
    return ["README.md", ".gitattributes", *[f"{name}/**" for name in problem_names]]


def _existing_manifest_problem_names(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        names.append(str(metadata.get("problem_name") or item.get("name") or item.get("item_id") or ""))
    return normalize_co_bench_problem_names(names)


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest_path)
    all_problem_names = co_bench_problem_names(ROOT)
    problem_names = normalize_co_bench_problem_names(list(args.problem_names)) if args.problem_names else list(all_problem_names)
    if args.problem_names and manifest_path.exists() and not isinstance(args.items, int):
        problem_names = normalize_co_bench_problem_names(_existing_manifest_problem_names(manifest_path) + problem_names)
    if isinstance(args.items, int) and args.items > 0:
        problem_names = problem_names[: args.items]

    from huggingface_hub import snapshot_download

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="CO-Bench/CO-Bench",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=_allow_patterns(problem_names),
        max_workers=max(1, int(args.download_max_workers or 2)),
    )
    manifest = build_co_bench_manifest(
        task_dir=ROOT,
        data_dir=data_dir,
        problem_names=problem_names,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")

    scope = ", ".join(problem_names) if problem_names else "all official CO-Bench tasks"
    print(f"Prepared CO-Bench dataset ({scope}) in {data_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print("Evaluation framework lives in benchmark/or_verified/co-bench/evaluation")
    print("Official README notes Docker support is 'coming soon'; the evaluator itself does not currently require Docker.")


if __name__ == "__main__":
    main()
