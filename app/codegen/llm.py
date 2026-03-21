from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.codegen.config import ROOT, RuntimeConfig, load_runtime_config
from app.codegen.errors import LlmResponseError, LlmTransportError


Transport = Callable[[dict[str, Any], RuntimeConfig], str]


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
    raise LlmResponseError("Model response did not contain a valid JSON object.")


def _trim(value: Any, *, limit: int = 240) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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
        required_keys = ("name", "strategy", "rationale", "function_body", "candidate_summary")
        missing = [key for key in required_keys if not isinstance(item.get(key), str) or not item.get(key).strip()]
        if missing:
            raise LlmResponseError(
                f"Candidate {index} is missing required string fields: {', '.join(missing)}.",
                model=trace.get("selected_model"),
            )
        function_body = item["function_body"].strip("\n")
        if function_body.lstrip().startswith("def "):
            raise LlmResponseError("Candidates must return function_body only, without a def line.", model=trace.get("selected_model"))
        normalized.append(
            {
                "agent": f"candidate-{index}",
                "label": _trim(item["name"], limit=72),
                "strategy": _trim(item["strategy"]),
                "rationale": _trim(item["rationale"]),
                "imports": _normalize_imports(item.get("imports")),
                "function_body": function_body,
                "candidate_summary": _trim(item["candidate_summary"]),
                "run_mode": "llm-required",
                "proposal_model": trace.get("selected_model"),
            }
        )
    if not normalized:
        raise LlmResponseError("Model response did not yield any valid candidates.", model=trace.get("selected_model"))
    return normalized


def _normalize_reflection_payload(payload: dict[str, Any], trace: dict[str, Any]) -> dict[str, str]:
    required = ("failure_pattern", "strategy_hypothesis", "successful_strategy", "prompt_fragment", "tool_trace_summary")
    normalized: dict[str, str] = {}
    for field in required:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise LlmResponseError(f"Reflection response is missing required field {field}.", model=trace.get("selected_model"))
        normalized[field] = _trim(value, limit=320)
    return normalized


def _request_preview(messages: list[dict[str, str]]) -> str:
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    if not user_messages:
        return ""
    return _trim(user_messages[-1], limit=280)


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
        raise LlmTransportError(f"Model request failed with HTTP {exc.code}: {body[:240]}", model=config.primary_model) from exc
    except urllib.error.URLError as exc:
        raise LlmTransportError(f"Model request failed: {exc.reason}", model=config.primary_model) from exc
    except TimeoutError as exc:
        raise LlmTransportError("Model request timed out.", model=config.primary_model) from exc


@dataclass(slots=True)
class ProposalRuntime:
    config: RuntimeConfig
    transport: Transport | None = None

    @classmethod
    def from_env(cls, root: Path | None = None) -> "ProposalRuntime":
        return cls(load_runtime_config(root or ROOT))

    @property
    def active_model(self) -> str:
        return self.config.primary_model

    def describe(self) -> dict[str, object]:
        return self.config.describe()

    def complete_json(self, *, purpose: str, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_body = {
            "model": self.config.primary_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        sender = self.transport or _default_transport
        try:
            raw_response = sender(request_body, self.config)
        except LlmTransportError:
            raise
        except TimeoutError as exc:
            raise LlmTransportError("Model request timed out.", model=self.active_model) from exc
        except urllib.error.URLError as exc:
            raise LlmTransportError(f"Model request failed: {exc.reason}", model=self.active_model) from exc
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LlmResponseError("Model HTTP response was not valid JSON.", model=self.active_model) from exc

        try:
            message = parsed_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmResponseError("Model response did not contain choices[0].message.content.", model=self.active_model) from exc

        text = _message_text(message)
        try:
            payload = _extract_json_object(text)
        except LlmResponseError as exc:
            raise LlmResponseError(str(exc), model=self.active_model) from exc
        usage = parsed_response.get("usage", {})
        trace = {
            "purpose": purpose,
            "selected_model": self.active_model,
            "parse_status": "ok",
            "api_base": self.config.api_base,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "request_preview": _request_preview(messages),
            "raw_preview": _trim(text, limit=280),
        }
        return payload, trace


def _proposal_prompt(
    *,
    task: dict[str, Any],
    generation: int,
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> tuple[str, str]:
    memory_lines = [
        "- "
        + json.dumps(
            {
                "experience_id": memory.get("experience_id"),
                "failure_pattern": memory.get("failure_pattern"),
                "strategy_hypothesis": memory.get("strategy_hypothesis"),
                "prompt_fragment": memory.get("prompt_fragment"),
                "candidate_summary": memory.get("candidate_summary"),
                "delta_J": memory.get("delta_J"),
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
                "J": item.get("metrics", {}).get("J"),
                "status": item.get("metrics", {}).get("status"),
                "candidate_summary": item.get("candidate_summary"),
                "strategy": item.get("strategy"),
            }
        )
        for item in candidate_history[-6:]
    ]
    system_prompt = (
        "You are the only proposal model in a strict outer-loop Python code optimization system. "
        "Return strict JSON with shape "
        '{"candidates":[{"name":"short label","strategy":"one sentence","rationale":"why it should win",'
        '"imports":["import line"],"function_body":"body only, no def line","candidate_summary":"brief code summary"}]}. '
        "Return between 1 and 3 candidates. The function_body must contain only the indented body contents, not the signature."
    )
    user_prompt = (
        f"Task id: {task['id']}\n"
        f"Title: {task['title']}\n"
        f"Description: {task['description']}\n"
        f"Function signature: {task['function_signature']}\n"
        f"Objective: {task['objective_direction']} {task['objective_label']}\n"
        f"Generation: {generation}\n"
        f"Current best summary: {current_best['candidate_summary']}\n"
        f"Current best objective: {current_best['metrics']['objective']}\n"
        f"Current best J: {current_best['metrics']['J']}\n"
        "Baseline source:\n"
        f"{current_best['baseline_source']}\n"
        "Current best source:\n"
        f"{current_best['source_code']}\n"
        f"Benchmark spec: {json.dumps(task['benchmark'])}\n"
        f"Tests: {json.dumps(task['tests'])}\n"
        "Retrieved strategy experiences:\n"
        + ("\n".join(memory_lines) if memory_lines else "- none")
        + "\nPrevious candidate summaries:\n"
        + ("\n".join(history_lines) if history_lines else "- none")
        + "\nGenerate Python candidates that preserve correctness and improve runtime."
    )
    return system_prompt, user_prompt


def propose_code_candidates(
    runtime: ProposalRuntime,
    *,
    task: dict[str, Any],
    generation: int,
    current_best: dict[str, Any],
    candidate_history: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_prompt, user_prompt = _proposal_prompt(
        task=task,
        generation=generation,
        current_best=current_best,
        candidate_history=candidate_history,
        memories=memories,
    )
    payload, trace = runtime.complete_json(
        purpose="generation_proposals",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
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
    delta_j: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    system_prompt = (
        "You compress successful Python code mutations into reusable strategy memory. "
        "Return strict JSON with fields failure_pattern, strategy_hypothesis, successful_strategy, prompt_fragment, tool_trace_summary."
    )
    user_prompt = (
        f"Task id: {task['id']}\n"
        f"Generation: {generation}\n"
        f"Previous best summary: {previous_best['candidate_summary']}\n"
        f"Previous best objective: {previous_best['metrics']['objective']}\n"
        f"Winner summary: {winner['candidate_summary']}\n"
        f"Winner strategy: {winner['strategy']}\n"
        f"Winner rationale: {winner['rationale']}\n"
        f"Winner objective: {winner['metrics']['objective']}\n"
        f"Winner J: {winner['metrics']['J']}\n"
        f"delta_J: {delta_j}\n"
        "Write a compact strategy fragment that can be pasted into the next proposal prompt."
    )
    payload, trace = runtime.complete_json(
        purpose="memory_reflection",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return _normalize_reflection_payload(payload, trace), trace
