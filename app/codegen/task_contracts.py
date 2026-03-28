from __future__ import annotations

from typing import Any


TASK_MODE_VALUES = frozenset({"answer", "artifact", "agent"})
OPTIMIZATION_SCOPE_VALUES = frozenset({"prompt", "wrapper", "implementation"})
RUNTIME_BACKEND_VALUES = frozenset({"dataset", "external"})


def _normalized_string(value: Any) -> str:
    return str(value or "").strip()


def infer_task_mode(task: dict[str, Any]) -> str:
    explicit = _normalized_string(task.get("task_mode"))
    if not explicit:
        raise ValueError("Task must declare task_mode.")
    if explicit not in TASK_MODE_VALUES:
        raise ValueError(f"Invalid task_mode={explicit!r}; expected one of {sorted(TASK_MODE_VALUES)}.")
    return explicit


def infer_optimization_scope(task: dict[str, Any]) -> str:
    explicit = _normalized_string(task.get("optimization_scope"))
    if not explicit:
        raise ValueError("Task must declare optimization_scope.")
    if explicit not in OPTIMIZATION_SCOPE_VALUES:
        raise ValueError(
            f"Invalid optimization_scope={explicit!r}; expected one of {sorted(OPTIMIZATION_SCOPE_VALUES)}."
        )
    return explicit


def infer_runtime_backend(task: dict[str, Any]) -> str:
    explicit = _normalized_string(task.get("runtime_backend"))
    if not explicit:
        raise ValueError("Task must declare runtime_backend.")
    if explicit not in RUNTIME_BACKEND_VALUES:
        raise ValueError(f"Invalid runtime_backend={explicit!r}; expected one of {sorted(RUNTIME_BACKEND_VALUES)}.")
    return explicit


def task_mode_summary(mode: str) -> str:
    if mode == "answer":
        return "Direct input-output task: candidate code returns the final answer/output for one item."
    if mode == "artifact":
        return "Artifact task: candidate code produces or defines a program, script, or other artifact that the verifier executes or consumes."
    if mode == "agent":
        return "Agent task: candidate code defines an interaction policy or harness wrapper that is evaluated over a multi-step environment."
    return "Task contract summary unavailable."


def optimization_scope_summary(scope: str) -> str:
    if scope == "prompt":
        return "Mutate prompt templates only; keep the surrounding wrapper and implementation fixed."
    if scope == "wrapper":
        return "Mutate the Python wrapper or policy layer, including prompts, preprocessing, postprocessing, and run configuration."
    if scope == "implementation":
        return "Mutate the task implementation itself; the editable file is the actual solver/program under evaluation."
    return "Optimization scope summary unavailable."
