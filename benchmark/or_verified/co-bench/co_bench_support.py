from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from app.bench.runtime_support import effective_suite_run_config


CO_BENCH_DATASET_ID = "CO-Bench/CO-Bench"
CO_BENCH_SPLIT = "official:test"
CO_BENCH_PROBLEM_ALIASES = {
    "MIS": "Maximal independent set",
    "TSP": "Travelling salesman problem",
}


@contextlib.contextmanager
def _prepend_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _task_data_is_present(data_dir: Path, task_name: str) -> bool:
    task_dir = data_dir / task_name
    return task_dir.exists() and (task_dir / "config.py").exists() and any(task_dir.iterdir())


def _canonical_problem_name(name: str) -> str:
    text = str(name).strip()
    return CO_BENCH_PROBLEM_ALIASES.get(text, text)


def _problem_slug(name: str) -> str:
    text = _canonical_problem_name(name).strip().lower()
    return "-".join(part for part in text.replace(":", " ").replace("/", " ").split() if part)


def _normalized_problem_names(names: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = _canonical_problem_name(str(name))
        if not canonical or canonical in seen:
            continue
        normalized.append(canonical)
        seen.add(canonical)
    return normalized


def _co_bench_task_dir(task: dict[str, Any]) -> Path:
    return Path(str(task["task_dir"]))


def _co_bench_data_dir(task: dict[str, Any], config: dict[str, Any]) -> Path:
    configured = str(config.get("data_dir") or "").strip()
    if configured:
        return Path(configured)
    return _co_bench_task_dir(task) / "data"


def _co_bench_evaluation_dir(task: dict[str, Any], config: dict[str, Any]) -> Path:
    configured = str(config.get("evaluation_dir") or "").strip()
    if configured:
        return Path(configured)
    return _co_bench_task_dir(task) / "evaluation"


def _controller_task_names(controller_module: Any) -> list[str]:
    return _normalized_problem_names(list(getattr(controller_module, "TASK_LIST") or []))


def _python_ast(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return source, ast.parse(source, filename=str(path))


def _controller_task_names_from_source(evaluation_dir: Path) -> list[str]:
    controller_path = evaluation_dir / "controller.py"
    if not controller_path.exists():
        raise FileNotFoundError(
            f"CO-Bench evaluation framework is missing at {controller_path}. "
            "The repo should include benchmark/or_verified/co-bench/evaluation."
        )
    _, tree = _python_ast(controller_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_LIST" for target in node.targets):
            continue
        values = ast.literal_eval(node.value)
        if not isinstance(values, list):
            raise ValueError(f"TASK_LIST must be a list in {controller_path}")
        return _normalized_problem_names([str(value) for value in values])
    raise ValueError(f"TASK_LIST was not found in {controller_path}")


def co_bench_problem_names(task_dir: Path) -> list[str]:
    return _controller_task_names_from_source(Path(task_dir) / "evaluation")


def normalize_co_bench_problem_names(names: list[str] | tuple[str, ...]) -> list[str]:
    return _normalized_problem_names(list(names))


def _extract_string_assignment(path: Path, name: str) -> str:
    _, tree = _python_ast(path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string literal in {path}")
        return value
    raise ValueError(f"{name} was not found in {path}")


def _extract_function_source(path: Path, function_name: str) -> str:
    source, tree = _python_ast(path)
    source_lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if not hasattr(node, "end_lineno"):
                raise RuntimeError("Python 3.8+ is required for AST end_lineno support.")
            return "\n".join(source_lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Function {function_name!r} was not found in {path}")


def _problem_description_from_config(config_path: Path) -> str:
    description = _extract_string_assignment(config_path, "DESCRIPTION")
    solve_template = _extract_function_source(config_path, "solve")
    return f"{description}\n\n# Implement in Solve Function\n\n{solve_template}"


def _problem_case_files(problem_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in problem_dir.iterdir()
        if path.is_file() and path.suffix != ".py" and path.name != "__pycache__" and not path.name.startswith(".")
    )


def _load_problem_data_loader(config_path: Path):
    module_name = f"co_bench_config_{config_path.parent.name.lower().replace(' ', '_').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CO-Bench config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_data = getattr(module, "load_data", None)
    return load_data if callable(load_data) else None


def _problem_case_instance_counts(problem_dir: Path) -> tuple[int, int]:
    config_path = problem_dir / "config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"CO-Bench task config was not found: {config_path}")
    load_data = _load_problem_data_loader(config_path)
    case_files = _problem_case_files(problem_dir)
    if load_data is None:
        return len(case_files), 0
    instance_count = 0
    for case_file in case_files:
        instance_count += len(list(load_data(str(case_file))))
    return len(case_files), instance_count


def build_co_bench_manifest(
    *,
    task_dir: Path,
    data_dir: Path,
    problem_names: list[str] | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    task_dir = Path(task_dir)
    data_dir = Path(data_dir)
    all_problem_names = co_bench_problem_names(task_dir)
    selected_problem_names = _normalized_problem_names(problem_names or all_problem_names)
    if isinstance(max_items, int) and max_items > 0:
        selected_problem_names = selected_problem_names[:max_items]

    items: list[dict[str, Any]] = []
    for index, problem_name in enumerate(selected_problem_names, start=1):
        problem_dir = data_dir / problem_name
        config_path = problem_dir / "config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"CO-Bench task config was not found: {config_path}")
        problem_description = _problem_description_from_config(config_path)
        case_count, instance_count = _problem_case_instance_counts(problem_dir)
        items.append(
            {
                "item_id": problem_name,
                "name": problem_name,
                "prompt": f"Generate a solve(**kwargs) implementation for the CO-Bench problem: {problem_name}",
                "context": problem_description,
                "expected_answer": "official_test_score",
                "metadata": {
                    "problem_name": problem_name,
                    "source_index": index - 1,
                    "source_split": CO_BENCH_SPLIT,
                    "config_path": str(config_path.relative_to(task_dir)),
                    "case_count": case_count,
                    "instance_count": instance_count,
                    "runtime_split_tags": [f"problem:{_problem_slug(problem_name)}"],
                },
            }
        )

    return {
        "dataset_id": CO_BENCH_DATASET_ID,
        "dataset_size": len(all_problem_names),
        "prepared_count": len(items),
        "items": items,
    }


def _prepare_datasets_hint(task: dict[str, Any]) -> str:
    return f"Run `python benchmark/prepare_datasets.py --task-id {task['id']}` first."


def _run_co_bench_prepare(
    task: dict[str, Any],
    *,
    problem_names: list[str] | None = None,
    min_items: int | None = None,
) -> None:
    task_dir = _co_bench_task_dir(task).resolve()
    prepare_path = task_dir / "prepare.py"
    if not prepare_path.exists():
        raise FileNotFoundError(
            f"CO-Bench prepare.py is missing at {prepare_path}. {_prepare_datasets_hint(task)}"
        )

    args = [sys.executable, str(prepare_path)]
    if isinstance(min_items, int) and min_items > 0:
        args.extend(["--items", str(min_items)])
    for problem_name in _normalized_problem_names(list(problem_names or [])):
        args.extend(["--problem-name", problem_name])

    completed = subprocess.run(
        args,
        cwd=task_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"returncode={completed.returncode}"
        raise RuntimeError(f"{prepare_path} failed: {details}. {_prepare_datasets_hint(task)}")


def _ensure_problem_data(task: dict[str, Any], config: dict[str, Any], problem_name: str) -> Path:
    data_dir = _co_bench_data_dir(task, config)
    if _task_data_is_present(data_dir, problem_name):
        return data_dir
    _run_co_bench_prepare(task, problem_names=[problem_name])
    if _task_data_is_present(data_dir, problem_name):
        return data_dir
    raise FileNotFoundError(
        f"CO-Bench data for {problem_name!r} is missing under {data_dir}. {_prepare_datasets_hint(task)}"
    )


def _co_bench_question_item(task: dict[str, Any]) -> dict[str, Any]:
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("CO-Bench dataset task must provide question_item.")
    return item


def _question_problem_name(item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    name = (
        metadata.get("problem_name")
        or item.get("raw_item_id")
        or item.get("name")
        or item.get("item_id")
    )
    canonical = _canonical_problem_name(str(name or ""))
    if not canonical:
        raise ValueError("CO-Bench question item is missing metadata.problem_name.")
    return canonical


def _co_bench_feedback(
    results: dict[str, tuple[list[Any], str | None]],
    avg_score: float,
    *,
    feedback_length: int = 64,
) -> str:
    rows: list[str] = []
    for case, (scores, error_message) in results.items():
        if error_message:
            rows.append(f"{case} -> Caught Error: {error_message}")
            continue
        rendered_scores = [value if isinstance(value, str) else f"{float(value):.3f}" for value in scores][:feedback_length]
        rows.append(f"{case} -> Scores: {rendered_scores}")
    feedback = "\n".join(rows[:feedback_length])
    return feedback + f"\nAvg Score {avg_score}"


def _evaluate_with_official_scoring(
    data: Any,
    python_code: str,
    *,
    timeout_s: int,
) -> dict[str, Any]:
    utils = importlib.import_module("evaluation.utils")
    evaluator_module = importlib.import_module("evaluation.evaluate")
    runtime = utils.ParallelRun(evaluator_module.evaluate_instance)
    raw_results: dict[str, tuple[list[Any], str | None]] = {}

    with utils.FileLock():
        for case in data.test_cases:
            file_path = Path(str(data.src_dir)) / str(data.task) / str(case)
            try:
                instances = list(data.load_data(str(file_path)))
                instance_results = [
                    runtime.run_instance_with_timeout(instance, python_code, data.config_path, timeout_s)
                    for instance in instances
                ]
                raw_results[str(case)] = (instance_results, None)
            except Exception as exc:  # noqa: BLE001
                raw_results[str(case)] = ([], f"Exception: {exc}")

    results = data.norm_score(raw_results)
    dev_results = utils.filter_dev(results, data.get_dev())
    test_results = utils.filter_test(results, data.get_dev())
    score = utils.average_score(results, data.test_cases)
    dev_score = utils.average_score(dev_results, data.test_cases)
    test_score = utils.average_score(test_results, data.test_cases)
    return {
        "score": score,
        "dev_score": dev_score,
        "test_score": test_score,
        "feedback": _co_bench_feedback(results, score),
        "dev_feedback": _co_bench_feedback(dev_results, dev_score),
        "test_feedback": _co_bench_feedback(test_results, test_score),
        "results": results,
    }


def evaluate_co_bench_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
) -> dict[str, Any]:
    item = _co_bench_question_item(task)
    config = effective_suite_run_config(task, candidate_path)
    evaluation_dir = _co_bench_evaluation_dir(task, config)
    if not (evaluation_dir / "controller.py").exists():
        raise FileNotFoundError(
            f"CO-Bench evaluation framework is missing under {evaluation_dir}. "
            "It should be checked into benchmark/or_verified/co-bench/evaluation."
        )

    problem_name = _question_problem_name(item)
    data_dir = _ensure_problem_data(task, config, problem_name)
    timeout_s = int(config.get("timeout_s") or 10)
    item_name = str(item.get("name") or problem_name)
    item_id = str(item.get("item_id") or problem_name)
    started = time.perf_counter()

    with _prepend_sys_path(evaluation_dir.parent):
        controller = importlib.import_module("evaluation.controller")
        data = controller.get_data(problem_name, src_dir=str(data_dir))
        try:
            feedback = _evaluate_with_official_scoring(data, source_code, timeout_s=timeout_s)
            test_score = float(feedback["test_score"])
            test_row = {
                "name": item_name,
                "expected": 1.0,
                "actual": test_score,
                "passed": test_score > 0.0,
                "actual_raw": {
                    "problem_name": problem_name,
                    "dev_score": feedback["dev_score"],
                    "test_score": feedback["test_score"],
                    "test_feedback": feedback["test_feedback"],
                },
            }
        except Exception as exc:  # noqa: BLE001
            test_score = 0.0
            test_row = {
                "name": item_name,
                "expected": 1.0,
                "actual": 0.0,
                "passed": False,
                "actual_raw": {
                    "problem_name": problem_name,
                    "error": str(exc),
                },
            }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    passed = 1 if test_row["passed"] else 0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": test_score,
        "passed_tests": passed,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": test_score,
        "objective_score": test_score,
        "objective_signal": test_score,
        "error": None,
        "co_bench_problem_name": problem_name,
        "data_dir": str(data_dir),
        "evaluation_dir": str(evaluation_dir),
        "test_results": [test_row],
    }
