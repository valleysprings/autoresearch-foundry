from __future__ import annotations

import json
import queue
import re
import threading
import time
import textwrap
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.configs.prompts import (
    CANDIDATE_LABEL_LIMIT,
    PROPOSAL_CANDIDATE_COUNT_TEMPLATE,
    PROPOSAL_CONCISE_FIELDS_INSTRUCTION,
    CANDIDATE_RESPONSE_REQUIRED_FIELDS,
    FAILURE_REFLECTION_OUTCOME_INSTRUCTIONS,
    FAILURE_REFLECTION_SYSTEM_PROMPT,
    MODEL_COMPLETION_MAX_ATTEMPTS,
    PROPOSAL_JSON_ONLY_INSTRUCTION,
    PROPOSAL_RESULT_INSTRUCTION,
    PROPOSAL_SYSTEM_PROMPT,
    RAW_PREVIEW_LIMIT,
    REFLECTION_FIELD_LIMIT,
    REFLECTION_FRAGMENT_INSTRUCTION,
    REFLECTION_REQUIRED_FIELDS,
    REQUEST_PREVIEW_LIMIT,
    SUCCESS_REFLECTION_OUTCOME_INSTRUCTIONS,
    SUCCESS_REFLECTION_SYSTEM_PROMPT,
    TRIM_DEFAULT_LIMIT,
)
from app.codegen.selection import prompt_summary
from app.configs.codegen import PROPOSAL_SELECTION_GUIDANCE
from app.codegen.config import ROOT, RuntimeConfig, load_runtime_config
from app.codegen.errors import LlmResponseError, LlmTransportError


Transport = Callable[[dict[str, Any], RuntimeConfig], str]
_TRANSPORT_GATE_LOCK = threading.Lock()
_TRANSPORT_DISPATCHERS: dict[str, tuple[int, "_TransportDispatcher"]] = {}


class _TransportDispatcher:
    def __init__(self, *, max_workers: int) -> None:
        self._queue: queue.PriorityQueue[tuple[int, int, Transport, dict[str, Any], RuntimeConfig, Future[str]]] = (
            queue.PriorityQueue()
        )
        self._lock = threading.Lock()
        self._sequence = 0
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"autoresearch-llm-{index}",
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        *,
        priority: int,
        sender: Transport,
        request_body: dict[str, Any],
        config: RuntimeConfig,
    ) -> Future[str]:
        future: Future[str] = Future()
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        self._queue.put((priority, sequence, sender, request_body, config, future))
        return future

    def _worker(self) -> None:
        while True:
            _priority, _sequence, sender, request_body, config, future = self._queue.get()
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(sender(request_body, config))
                    except BaseException as exc:  # noqa: BLE001
                        future.set_exception(exc)
            finally:
                self._queue.task_done()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return json.dumps(content)


def _transport_dispatcher(config: RuntimeConfig) -> _TransportDispatcher:
    with _TRANSPORT_GATE_LOCK:
        dispatcher_entry = _TRANSPORT_DISPATCHERS.get(config.api_base)
        if dispatcher_entry is None or dispatcher_entry[0] != config.llm_concurrency:
            dispatcher_entry = (
                config.llm_concurrency,
                _TransportDispatcher(max_workers=config.llm_concurrency),
            )
            _TRANSPORT_DISPATCHERS[config.api_base] = dispatcher_entry
        return dispatcher_entry[1]


def _proposal_queue_priority(generation: int) -> int:
    return max(1, generation) * 10


def _reflection_queue_priority(generation: int) -> int:
    return _proposal_queue_priority(generation) + 5


def _extract_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        raise LlmResponseError("Model response was empty.")
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL)
    for candidate in fenced:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for candidate in _balanced_json_objects(normalized):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LlmResponseError("Model response did not contain a valid JSON object.")


def _balanced_json_objects(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if escaped:
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start_index : index + 1])
                start_index = None
    return candidates


def _trim(value: Any, *, limit: int = TRIM_DEFAULT_LIMIT) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _proposal_candidate_noun(count: int) -> str:
    return "candidate" if count == 1 else "candidates"


def _proposal_system_prompt(candidate_budget: int) -> str:
    return " ".join(
        (
            PROPOSAL_SYSTEM_PROMPT,
            PROPOSAL_JSON_ONLY_INSTRUCTION,
            PROPOSAL_CANDIDATE_COUNT_TEMPLATE.format(
                count=candidate_budget,
                noun=_proposal_candidate_noun(candidate_budget),
            ),
            PROPOSAL_CONCISE_FIELDS_INSTRUCTION,
            "file_body must contain the full contents of the editable file and must preserve the declared entry symbol.",
        )
    )


def _looks_like_truncated_response(text: str, completion_tokens: Any, max_tokens: int) -> bool:
    completion_hit_limit = isinstance(completion_tokens, int) and completion_tokens >= max_tokens
    normalized = text.rstrip()
    if not normalized:
        return completion_hit_limit
    incomplete_tail = (
        normalized.startswith("```")
        and not normalized.endswith("```")
        or normalized.count("{") > normalized.count("}")
        or normalized.count("[") > normalized.count("]")
        or normalized.endswith("\\")
    )
    return completion_hit_limit or incomplete_tail


def _parse_failure_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    usage: dict[str, Any],
    text: str,
    attempt: int,
    fallback_message: str,
) -> LlmResponseError:
    completion_tokens = usage.get("completion_tokens")
    response_truncated = _looks_like_truncated_response(text, completion_tokens, runtime.config.max_tokens)
    details = {
        "purpose": purpose,
        "selected_model": runtime.active_model,
        "parse_status": "truncated" if response_truncated else "invalid_json",
        "api_base": runtime.config.api_base,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "max_tokens": runtime.config.max_tokens,
        "request_preview": _request_preview(messages),
        "raw_preview": _trim(text, limit=RAW_PREVIEW_LIMIT),
        "attempt": attempt,
        "response_truncated": response_truncated,
    }
    if response_truncated:
        message = (
            "Model response appears truncated before completing a valid JSON object "
            f"(completion_tokens={completion_tokens}, max_tokens={runtime.config.max_tokens})."
        )
    else:
        message = fallback_message
    return LlmResponseError(message, model=runtime.active_model, details=details)


def _transport_failure_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    attempt: int,
    exc: Exception,
) -> LlmTransportError:
    details = {
        "purpose": purpose,
        "selected_model": runtime.active_model,
        "parse_status": "transport_error",
        "api_base": runtime.config.api_base,
        "request_preview": _request_preview(messages),
        "attempt": attempt,
        "max_attempts": MODEL_COMPLETION_MAX_ATTEMPTS,
    }
    if isinstance(exc, LlmTransportError) and exc.details is not None:
        details.update(exc.details)
    return LlmTransportError(str(exc), model=runtime.active_model, details=details)


def _response_envelope_error(
    *,
    purpose: str,
    runtime: "ProposalRuntime",
    messages: list[dict[str, str]],
    raw_response: str,
    attempt: int,
    message: str,
) -> LlmResponseError:
    return LlmResponseError(
        message,
        model=runtime.active_model,
        details={
            "purpose": purpose,
            "selected_model": runtime.active_model,
            "parse_status": "invalid_http_json",
            "api_base": runtime.config.api_base,
            "request_preview": _request_preview(messages),
            "raw_preview": _trim(raw_response, limit=RAW_PREVIEW_LIMIT),
            "attempt": attempt,
        },
    )


def _normalize_imports(raw_imports: Any) -> list[str]:
    if raw_imports is None:
        return []
    if not isinstance(raw_imports, list):
        raise LlmResponseError("Candidate imports must be a list of strings.")
    imports: list[str] = []
    for item in raw_imports:
        if not isinstance(item, str):
            raise LlmResponseError("Candidate imports must contain only strings.")
        stripped = item.strip()
        if stripped:
            imports.append(stripped)
    return list(dict.fromkeys(imports))


def _normalize_candidate_payload(payload: dict[str, Any], task: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LlmResponseError("Model response must contain a non-empty candidates list.", model=trace.get("selected_model"))

    normalized: list[dict[str, Any]] = []
    max_candidates = int(task["candidate_budget"])
    for index, item in enumerate(candidates[:max_candidates], start=1):
        if not isinstance(item, dict):
            raise LlmResponseError("Each candidate must be an object.", model=trace.get("selected_model"))
        missing = [
            key for key in CANDIDATE_RESPONSE_REQUIRED_FIELDS if not isinstance(item.get(key), str) or not item.get(key).strip()
        ]
        if missing:
            raise LlmResponseError(
                f"Candidate {index} is missing required string fields: {', '.join(missing)}.",
                model=trace.get("selected_model"),
            )
        file_body = item["file_body"].strip("\n")
        if not file_body.strip():
            raise LlmResponseError("Candidates must return a non-empty editable file.", model=trace.get("selected_model"))
        normalized.append(
            {
                "agent": f"candidate-{index}",
                "label": _trim(item["name"], limit=CANDIDATE_LABEL_LIMIT),
                "strategy": _trim(item["strategy"]),
                "rationale": _trim(item["rationale"]),
                "imports": _normalize_imports(item.get("imports")),
                "file_body": file_body,
                "candidate_summary": _trim(item["candidate_summary"]),
                "run_mode": "llm-required",
                "proposal_model": trace.get("selected_model"),
            }
        )
    if not normalized:
        raise LlmResponseError("Model response did not yield any valid candidates.", model=trace.get("selected_model"))
    return normalized


def _normalize_reflection_payload(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in REFLECTION_REQUIRED_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise LlmResponseError(f"Reflection response is missing required field {field}.", model=trace.get("selected_model"))
        normalized[field] = _trim(value, limit=REFLECTION_FIELD_LIMIT)
    return normalized


def _request_preview(messages: list[dict[str, str]]) -> str:
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    if not user_messages:
        return ""
    return _trim(user_messages[-1], limit=REQUEST_PREVIEW_LIMIT)


def _default_transport(request_body: dict[str, Any], config: RuntimeConfig) -> str:
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url=f"{config.api_base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LlmTransportError(f"Model request failed with HTTP {exc.code}: {body[:240]}", model=config.active_model) from exc
    except urllib.error.URLError as exc:
        raise LlmTransportError(f"Model request failed: {exc.reason}", model=config.active_model) from exc
    except TimeoutError as exc:
        raise LlmTransportError("Model request timed out.", model=config.active_model) from exc


@dataclass(slots=True)
class ProposalRuntime:
    config: RuntimeConfig
    transport: Transport | None = None

    @classmethod
    def from_env(cls, root: Path | None = None) -> "ProposalRuntime":
        return cls(load_runtime_config(root or ROOT))

    @property
    def active_model(self) -> str:
        return self.config.active_model

    def with_model(self, model: str | None) -> "ProposalRuntime":
        return ProposalRuntime(config=self.config.with_model(model), transport=self.transport)

    def describe(self) -> dict[str, object]:
        return self.config.describe()

    def complete_json(
        self,
        *,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        queue_priority: int = 1000,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_body = {
            "model": self.active_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        sender = self.transport or _default_transport
        last_parse_error: LlmResponseError | None = None
        last_transport_error: LlmTransportError | None = None
        for attempt in range(1, MODEL_COMPLETION_MAX_ATTEMPTS + 1):
            try:
                raw_response = _transport_dispatcher(self.config).submit(
                    priority=queue_priority,
                    sender=sender,
                    request_body=request_body,
                    config=self.config,
                ).result()
            except LlmTransportError as exc:
                last_transport_error = _transport_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    attempt=attempt,
                    exc=exc,
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_transport_error from exc
                time.sleep(min(2**(attempt - 1), 5))
                continue
            except TimeoutError as exc:
                last_transport_error = _transport_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    attempt=attempt,
                    exc=LlmTransportError("Model request timed out.", model=self.active_model),
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_transport_error from exc
                time.sleep(min(2**(attempt - 1), 5))
                continue
            except urllib.error.URLError as exc:
                last_transport_error = _transport_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    attempt=attempt,
                    exc=LlmTransportError(f"Model request failed: {exc.reason}", model=self.active_model),
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_transport_error from exc
                time.sleep(min(2**(attempt - 1), 5))
                continue
            try:
                parsed_response = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                last_parse_error = _response_envelope_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    raw_response=raw_response,
                    attempt=attempt,
                    message="Model HTTP response was not valid JSON.",
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_parse_error from exc
                continue

            try:
                message = parsed_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                last_parse_error = _response_envelope_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    raw_response=raw_response,
                    attempt=attempt,
                    message="Model response did not contain choices[0].message.content.",
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_parse_error from exc
                continue

            text = _message_text(message)
            usage = parsed_response.get("usage", {})
            try:
                payload = _extract_json_object(text)
            except LlmResponseError as exc:
                last_parse_error = _parse_failure_error(
                    purpose=purpose,
                    runtime=self,
                    messages=messages,
                    usage=usage,
                    text=text,
                    attempt=attempt,
                    fallback_message=str(exc),
                )
                if attempt >= MODEL_COMPLETION_MAX_ATTEMPTS:
                    raise last_parse_error from exc
                continue
            trace = {
                "purpose": purpose,
                "selected_model": self.active_model,
                "parse_status": "ok",
                "api_base": self.config.api_base,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": self.config.max_tokens,
                "request_preview": _request_preview(messages),
                "raw_preview": _trim(text, limit=RAW_PREVIEW_LIMIT),
                "attempt": attempt,
            }
            return payload, trace
        if last_transport_error is not None:
            raise last_transport_error
        if last_parse_error is not None:
            raise last_parse_error
        raise LlmResponseError("Model response did not contain a valid JSON object.", model=self.active_model)


def _proposal_prompt(
    *,
    task: dict[str, Any],
    generation: int,
    parent_candidate: dict[str, Any],
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> tuple[str, str]:
    objective_spec = task.get("objective_spec") or {}
    selection_spec = task.get("selection_spec") or {}
    objective_name = objective_spec.get("display_name") or task.get("objective_label") or "objective"
    objective_direction = objective_spec.get("direction") or task.get("objective_direction") or "max"
    objective_formula = objective_spec.get("formula") or task.get("objective_label") or "objective"
    objective_summary = objective_spec.get("summary_template") or ""
    selection_summary = prompt_summary(selection_spec)
    memory_lines = [
        "- "
        + json.dumps(
            {
                "experience_id": memory.get("experience_id"),
                "experience_outcome": memory.get("experience_outcome", "success"),
                "verifier_status": memory.get("verifier_status"),
                "rejection_reason": memory.get("rejection_reason"),
                "failure_pattern": memory.get("failure_pattern"),
                "strategy_hypothesis": memory.get("strategy_hypothesis"),
                "successful_strategy": memory.get("successful_strategy"),
                "prompt_fragment": memory.get("prompt_fragment"),
                "candidate_summary": memory.get("candidate_summary"),
                "delta_primary_score": memory.get("delta_primary_score"),
            }
        )
        for memory in memories
    ]
    history_lines = [
        "- "
        + json.dumps(
            {
                "generation": item.get("generation"),
                "candidate": item.get("agent"),
                "objective": item.get("metrics", {}).get("objective"),
                "primary_score": item.get("metrics", {}).get("primary_score"),
                "tie_break_score": item.get("metrics", {}).get("tie_break_score"),
                "gate_passed": item.get("metrics", {}).get("gate_passed"),
                "status": item.get("metrics", {}).get("status"),
                "candidate_summary": item.get("candidate_summary"),
                "strategy": item.get("strategy"),
            }
        )
        for item in candidate_history[-6:]
    ]
    system_prompt = _proposal_system_prompt(max(1, int(task["candidate_budget"])))
    user_prompt = (
        f"Task id: {task['id']}\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
        f"Benchmark tier: {task['benchmark_tier']}\n"
        f"Track: {task['track']}\n"
        f"Dataset id: {task['dataset_id']}\n"
        f"Editable file: {task['editable_file']}\n"
        f"Entry symbol: {task['entry_symbol']}\n"
        f"Objective: {objective_name}\n"
        f"Objective direction: {objective_direction}\n"
        f"Objective formula: {objective_formula}\n"
        f"Objective summary: {objective_summary}\n"
        f"{PROPOSAL_SELECTION_GUIDANCE}\n"
        f"{selection_summary}\n"
        f"Prompt context: {task.get('prompt_context') or 'n/a'}\n"
        f"Generation: {generation}\n"
        f"Selected parent summary: {parent_candidate['candidate_summary']}\n"
        f"Selected parent objective: {parent_candidate['metrics']['objective']}\n"
        f"Selected parent objective_score: {parent_candidate['metrics'].get('objective_score')}\n"
        f"Selected parent primary_score: {parent_candidate['metrics'].get('primary_score')}\n"
        f"Selected parent tie_break_score: {parent_candidate['metrics'].get('tie_break_score')}\n"
        f"Selected parent gate_passed: {parent_candidate['metrics'].get('gate_passed')}\n"
        f"Global best summary: {current_best['candidate_summary']}\n"
        f"Global best objective: {current_best['metrics']['objective']}\n"
        f"Global best objective_score: {current_best['metrics'].get('objective_score')}\n"
        f"Global best primary_score: {current_best['metrics'].get('primary_score')}\n"
        f"Global best tie_break_score: {current_best['metrics'].get('tie_break_score')}\n"
        f"Global best gate_passed: {current_best['metrics'].get('gate_passed')}\n"
        "Baseline source:\n"
        f"{current_best['baseline_source']}\n"
        "Selected parent editable file:\n"
        f"{parent_candidate['source_code']}\n"
        "Global best editable file:\n"
        f"{current_best['source_code']}\n"
        "Retrieved strategy experiences (successful wins and failed attempts to avoid):\n"
        + ("\n".join(memory_lines) if memory_lines else "- none")
        + "\nPrevious candidate summaries:\n"
        + ("\n".join(history_lines) if history_lines else "- none")
        + f"\n{PROPOSAL_RESULT_INSTRUCTION}"
    )
    return system_prompt, user_prompt


def propose_code_candidates(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    generation: int,
    parent_candidate: dict[str, Any],
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_prompt, user_prompt = _proposal_prompt(
        task=task,
        generation=generation,
        parent_candidate=parent_candidate,
        current_best=current_best,
        candidate_history=candidate_history,
        memories=memories,
    )
    payload, trace = runtime.complete_json(
        purpose="generation_proposals",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=_proposal_queue_priority(generation),
    )
    candidates = _normalize_candidate_payload(payload, task, trace)
    trace["candidate_count"] = len(candidates)
    return candidates, trace


def reflect_strategy_experience(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    generation: int,
    previous_best: dict[str, Any],
    winner: dict[str, Any],
    delta_primary_score: float,
    outcome: str,
    rejection_reason: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if outcome == "success":
        system_prompt = SUCCESS_REFLECTION_SYSTEM_PROMPT
        outcome_instructions = SUCCESS_REFLECTION_OUTCOME_INSTRUCTIONS
    else:
        system_prompt = FAILURE_REFLECTION_SYSTEM_PROMPT
        outcome_instructions = FAILURE_REFLECTION_OUTCOME_INSTRUCTIONS
    failed_tests = [result["name"] for result in winner["metrics"].get("test_results", []) if not result.get("passed")]
    user_prompt = (
        f"Task id: {task['id']}\n"
        f"Generation: {generation}\n"
        f"Outcome: {outcome}\n"
        f"Previous best summary: {previous_best['candidate_summary']}\n"
        f"Previous best objective: {previous_best['metrics']['objective']}\n"
        f"Winner summary: {winner['candidate_summary']}\n"
        f"Winner strategy: {winner['strategy']}\n"
        f"Winner rationale: {winner['rationale']}\n"
        f"Winner verifier_status: {winner['metrics']['verifier_status']}\n"
        f"Winner objective: {winner['metrics']['objective']}\n"
        f"Winner primary_score: {winner['metrics'].get('primary_score')}\n"
        f"Winner tie_break_score: {winner['metrics'].get('tie_break_score')}\n"
        f"Winner error: {winner['metrics'].get('error')}\n"
        f"Failed tests: {json.dumps(failed_tests)}\n"
        f"Rejection reason: {rejection_reason or 'n/a'}\n"
        f"delta_primary_score: {delta_primary_score}\n"
        f"{outcome_instructions}\n"
        f"{REFLECTION_FRAGMENT_INSTRUCTION}"
    )
    payload, trace = runtime.complete_json(
        purpose="memory_reflection",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=_reflection_queue_priority(generation),
    )
    return _normalize_reflection_payload(payload, trace), trace
