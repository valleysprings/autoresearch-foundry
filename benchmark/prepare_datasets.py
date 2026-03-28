from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark-local dataset prepare.py scripts.")
    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT))
    parser.add_argument("--registry")
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--list", action="store_true", help="List benchmark tasks and whether they expose prepare.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print the prepare commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep preparing later tasks even if one prepare.py fails.",
    )
    return parser.parse_args(argv)


def _load_registry_entries(registry_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(registry_path.read_text())
    entries = payload.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"Registry must contain a top-level tasks list: {registry_path}")
    return [dict(entry) for entry in entries]


def _select_entries(entries: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(entry.get("id")): entry for entry in entries}
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {', '.join(missing)}")
        return [by_id[task_id] for task_id in task_ids]
    return [entry for entry in entries if bool(entry.get("enabled", True))]


def _prepare_script_path(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        raise ValueError(f"Registry entry is missing path: {entry}")
    return benchmark_root / relative_path / "prepare.py"


def _task_dir(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        raise ValueError(f"Registry entry is missing path: {entry}")
    return benchmark_root / relative_path


def _task_spec_path(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    return _task_dir(benchmark_root, entry) / "task.json"


def _maybe_load_task_spec(benchmark_root: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    task_path = _task_spec_path(benchmark_root, entry)
    if not task_path.exists():
        return None
    payload = json.loads(task_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Task spec must contain an object: {task_path}")
    return payload


def _local_dataset_manifest_path(benchmark_root: Path, entry: dict[str, Any], task_spec: dict[str, Any] | None) -> Path | None:
    if not isinstance(task_spec, dict) or not bool(task_spec.get("local_dataset_only")):
        return None
    item_manifest = str(task_spec.get("item_manifest") or "").strip()
    if not item_manifest:
        return None
    return _task_dir(benchmark_root, entry) / item_manifest


def _dataset_status(benchmark_root: Path, entry: dict[str, Any]) -> tuple[str, Path | None]:
    task_spec = _maybe_load_task_spec(benchmark_root, entry)
    manifest_path = _local_dataset_manifest_path(benchmark_root, entry, task_spec)
    if manifest_path is None:
        return "n/a", None
    return ("yes" if manifest_path.exists() else "no"), manifest_path


def _run_prepare_script(script_path: Path, python_executable: str, *, dry_run: bool) -> None:
    command = [python_executable, str(script_path)]
    print(f"[prepare] {' '.join(command)}")
    if dry_run:
        return
    completed = subprocess.run(
        command,
        cwd=script_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{script_path} exited with code {completed.returncode}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    benchmark_root = Path(args.benchmark_root).resolve()
    registry_path = Path(args.registry).resolve() if args.registry else benchmark_root / "registry.json"
    entries = _load_registry_entries(registry_path)
    selected = _select_entries(entries, list(args.task_ids))

    if args.list:
        for entry in selected:
            script_path = _prepare_script_path(benchmark_root, entry)
            prepare_status = "yes" if script_path.exists() else "no"
            dataset_status, manifest_path = _dataset_status(benchmark_root, entry)
            status_parts = [
                str(entry["id"]),
                f"prepare={prepare_status}",
                f"local_dataset_ready={dataset_status}",
                f"path={entry['path']}",
            ]
            if manifest_path is not None:
                status_parts.append(f"manifest={manifest_path}")
            print("\t".join(status_parts))
        return 0

    prepared = 0
    skipped = 0
    failures: list[str] = []
    for entry in selected:
        script_path = _prepare_script_path(benchmark_root, entry)
        if not script_path.exists():
            dataset_status, manifest_path = _dataset_status(benchmark_root, entry)
            if dataset_status == "no":
                detail = (
                    f"enabled local dataset manifest is missing at {manifest_path} and no prepare.py was found"
                    if manifest_path is not None
                    else "enabled local dataset manifest is missing and no prepare.py was found"
                )
                failures.append(f"{entry['id']}: {detail}")
                print(f"[error] {entry['id']}: {detail}", file=sys.stderr)
                if not args.continue_on_error:
                    break
                continue
            skipped += 1
            print(f"[skip] {entry['id']}: no prepare.py")
            continue
        try:
            _run_prepare_script(script_path, args.python, dry_run=bool(args.dry_run))
            prepared += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{entry['id']}: {exc}")
            print(f"[error] {entry['id']}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(f"prepared={prepared} skipped={skipped} failed={len(failures)}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
