from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from app.codegen.external import (
    build_external_candidate,
    build_external_result,
    effective_external_run_config,
    emit_progress,
    load_candidate_module,
    render_external_run_config_source,
    runtime_for_external_task,
)
from app.codegen.llm import ProposalRuntime


OR_MODELING_SYSTEM_PROMPT = (
    "You are an expert in operations research modeling. "
    "Return only a JSON object with keys modeling_summary and python_code. "
    "python_code must be directly executable Python using coptpy, without Markdown fences. "
    "It must define the complete optimization model needed to solve the question."
)
_ADD_SCRIPT = (
    "\nif model.status == COPT.OPTIMAL:\n"
    '    print(f"Just print the best solution: {model.objval}")\n'
    'else:\n'
    '    print("No Best Solution")\n'
)
def _local_or_dataset_path(task: dict[str, Any], config: dict[str, Any]) -> Path:
    configured = str(config.get("dataset_path") or "").strip()
    if configured:
        return Path(configured)
    return Path(str(task["task_dir"])) / "data" / "questions.json"


def _prepare_datasets_hint(task: dict[str, Any]) -> str:
    return f"Run `python benchmark/prepare_datasets.py --task-id {task['id']}` first."


def _run_local_or_prepare(task: dict[str, Any]) -> None:
    task_dir = Path(str(task["task_dir"])).resolve()
    prepare_path = task_dir / "prepare.py"
    if not prepare_path.exists():
        raise FileNotFoundError(
            f"Local dataset is missing for {task['id']} and no prepare.py was found at {prepare_path}. "
            f"{_prepare_datasets_hint(task)}"
        )
    completed = subprocess.run(
        [sys.executable, str(prepare_path)],
        cwd=task_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"returncode={completed.returncode}"
        raise RuntimeError(f"{prepare_path} failed: {details}")


def _load_local_or_dataset_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Local OR dataset manifest must contain a list of items: {path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Local OR dataset row {index} is not a JSON object: {path}")
        question = str(row.get("prompt") or row.get("question") or row.get("en_question") or "").strip()
        if not question:
            raise ValueError(f"Local OR dataset row {index} is missing a question prompt: {path}")
        normalized.append(
            {
                "item_id": str(row.get("item_id") or f"or-item-{index:04d}"),
                "en_question": question,
                "en_answer": row.get("expected_answer", row.get("answer", row.get("en_answer"))),
            }
        )
    return normalized
def _dataset_rows(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    max_items: int | None,
) -> tuple[list[dict[str, Any]], str]:
    local_path = _local_or_dataset_path(task, config)
    configured_local_path = str(config.get("dataset_path") or "").strip()
    if not local_path.exists() and not configured_local_path:
        _run_local_or_prepare(task)
    if local_path.exists():
        rows = _load_local_or_dataset_rows(local_path)
        source_label = str(local_path)
    else:
        raise FileNotFoundError(
            f"Local OR dataset manifest not found: {local_path}. {_prepare_datasets_hint(task)}"
        )
    if isinstance(max_items, int) and max_items > 0:
        rows = rows[:max_items]
    return rows, source_label


def _or_prompt(question: str) -> str:
    return (
        "Below is an operations research question. Build a mathematical model and corresponding "
        "python code using `coptpy` that appropriately addresses the question.\n\n"
        f"# Question:\n{question.strip()}\n\n"
        "# Response:\n"
    )


def _prompt_python_code(
    proposal_runtime: ProposalRuntime,
    *,
    candidate_path: Path,
    question: str,
    purpose: str,
    queue_priority: int,
) -> tuple[str, str, dict[str, Any]]:
    module = load_candidate_module(candidate_path)
    system_prompt = str(getattr(module, "SYSTEM_PROMPT", OR_MODELING_SYSTEM_PROMPT) or OR_MODELING_SYSTEM_PROMPT).strip()
    build_user_prompt = getattr(module, "build_user_prompt", None)
    if callable(build_user_prompt):
        user_prompt = str(build_user_prompt(question) or "").strip()
        if not user_prompt:
            raise ValueError("build_user_prompt(question) must return a non-empty string.")
    else:
        user_prompt = _or_prompt(question)
    payload, trace = proposal_runtime.complete_json(
        purpose=purpose,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=queue_priority,
    )
    code = str(payload.get("python_code") or "").strip()
    summary = str(payload.get("modeling_summary") or "").strip()
    if not code:
        raise ValueError("Model response did not include python_code.")
    return code, summary, trace


def _execute_coptpy_code(
    python_code: str,
    *,
    script_path: Path,
    timeout_s: int,
) -> dict[str, Any]:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(python_code.rstrip() + "\n" + _ADD_SCRIPT)
    try:
        completed = subprocess.run(
            ["python3", str(script_path)],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "execution_state": "Execution Failed: Timeout",
            "execution_best_solution": None,
            "execution_result": "",
        }

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return {
            "execution_state": f"Execution Failed: returncode={completed.returncode}",
            "execution_best_solution": None,
            "execution_result": (stderr or stdout).strip(),
        }

    marker = "Just print the best solution:"
    if marker in stdout:
        value = stdout.split(marker, 1)[1].strip().splitlines()[0].strip()
        return {
            "execution_state": "Execution Successful and Best Solution Found",
            "execution_best_solution": value,
            "execution_result": stdout.strip(),
        }
    if "No Best Solution" in stdout:
        return {
            "execution_state": "Execution Successful but No Best Solution Found",
            "execution_best_solution": "No Best Solution",
            "execution_result": stdout.strip(),
        }
    return {
        "execution_state": "Execution Successful but Output Could Not Be Parsed",
        "execution_best_solution": None,
        "execution_result": stdout.strip(),
    }


def _answers_match(expected: Any, predicted: Any, *, tolerance: float) -> bool:
    expected_text = str(expected).strip()
    if predicted is None:
        return False
    predicted_text = str(predicted).strip()
    if expected_text == "No Best Solution":
        return predicted_text == expected_text
    try:
        expected_value = round(float(expected_text))
        predicted_value = round(float(predicted_text))
    except (TypeError, ValueError):
        return expected_text == predicted_text
    if expected_value == 0:
        return abs(predicted_value) <= tolerance
    return math.isfinite(predicted_value) and abs((predicted_value - expected_value) / expected_value) <= tolerance


def evaluate_coptpy_value_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    workspace_root: Path,
    max_items: int | None,
    dataset_name: str,
    dataset_split: str,
    summary_label: str,
    proposal_runtime: ProposalRuntime | None = None,
    progress_callback=None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    runtime = proposal_runtime or runtime_for_external_task(task)
    execution_timeout_s = int(config.get("execution_timeout_s") or 600)
    tolerance = float(config.get("numerical_err_tolerance") or 1e-4)
    dataset_name = str(config.get("dataset_name") or dataset_name)
    dataset_split = str(config.get("dataset_split") or dataset_split)
    rows, dataset_source = _dataset_rows(
        task=task,
        config=config,
        max_items=max_items,
    )

    emit_progress(
        progress_callback,
        task_id=str(task["id"]),
        phase="external_dataset_loading",
        message=f"Loading {dataset_source}",
        pace_ms=pace_ms,
    )
    llm_traces: list[dict[str, Any]] = []
    item_results: list[dict[str, Any]] = []
    passed = 0
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        item_id = f"{task['id']}-{index:04d}"
        question = str(row.get("en_question") or "").strip()
        expected = row.get("en_answer")
        emit_progress(
            progress_callback,
            task_id=str(task["id"]),
            phase="external_item_started",
            message=f"[{item_id}] prompting model",
            pace_ms=pace_ms,
            item_id=item_id,
        )
        try:
            python_code, modeling_summary, trace = _prompt_python_code(
                runtime,
                candidate_path=candidate_path,
                question=question,
                purpose=f"{task['id']}::{item_id}",
                queue_priority=1000 + index,
            )
            llm_traces.append({**trace, "item_id": item_id, "dataset_name": dataset_name})
        except Exception as exc:  # noqa: BLE001
            item_results.append(
                {
                    "name": item_id,
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "error": str(exc),
                }
            )
            continue

        execution = _execute_coptpy_code(
            python_code,
            script_path=workspace_root / "generated" / f"{item_id}.py",
            timeout_s=execution_timeout_s,
        )
        predicted = execution["execution_best_solution"]
        matched = _answers_match(expected, predicted, tolerance=tolerance)
        if matched:
            passed += 1
        item_results.append(
            {
                "name": item_id,
                "expected": expected,
                "actual": predicted,
                "passed": matched,
                "answer_format": "numeric",
                "actual_raw": {
                    "question": question,
                    "modeling_summary": modeling_summary,
                    "execution_state": execution["execution_state"],
                    "execution_result": execution["execution_result"],
                },
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    total = len(rows)
    objective = passed / total if total else 0.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": passed,
        "total_tests": total,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "test_results": item_results,
        "external_summary": {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "dataset_source": dataset_source,
            "passed": passed,
            "total": total,
        },
    }


def run_coptpy_value_benchmark(
    *,
    task: dict[str, Any],
    candidate_path: Path,
    source_code: str,
    proposal_runtime: ProposalRuntime,
    workspace_root: Path,
    max_items: int | None,
    dataset_name: str,
    dataset_split: str,
    summary_label: str,
    progress_callback,
    pace_ms: int,
) -> dict[str, Any]:
    config = effective_external_run_config(task, candidate_path)
    rendered_source = render_external_run_config_source(config)
    raw_metrics = evaluate_coptpy_value_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        max_items=max_items,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        summary_label=summary_label,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    objective = float(raw_metrics.get("objective") or 0.0)
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)

    baseline = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent="checked-in-config",
        label="checked-in-config",
        strategy="Use the checked-in benchmark configuration.",
        rationale="External benchmark runs compare the current configured model against a neutral checked-in config baseline.",
        candidate_summary="Checked-in external benchmark configuration.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_external_candidate(
        task=task,
        source_code=rendered_source,
        agent=proposal_runtime.active_model,
        label=proposal_runtime.active_model,
        strategy="Generate coptpy formulations directly from each benchmark question.",
        rationale="Each question is solved by prompting the configured model for an executable coptpy program and comparing the resulting optimal value.",
        candidate_summary=f"{summary_label} evaluated by executing model-generated coptpy code.",
        raw_metrics=raw_metrics,
        workspace_path=str(workspace_root),
        proposal_model=proposal_runtime.active_model,
    )
    return build_external_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"{summary_label} finished with {passed}/{total} exact optimal-value matches.",
        llm_traces=[],
        extra_fields={"external_summary": dict(raw_metrics.get("external_summary") or {})},
    )
