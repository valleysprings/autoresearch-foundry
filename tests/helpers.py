from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from app.codegen import catalog
from app.codegen.config import RuntimeConfig
from app.codegen.llm import ProposalRuntime

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_BENCHMARK_ROOT = ROOT / "tests" / "fixtures" / "benchmarks"
FIXTURE_REGISTRY_PATH = FIXTURE_BENCHMARK_ROOT / "registry.json"


def chat_response(payload: dict, *, model: str = "deepseek-chat") -> str:
    return json.dumps(
        {
            "id": "resp-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
            "model": model,
        }
    )


class QueueTransport:
    def __init__(self, responses: Sequence[object]):
        self.responses = list(responses)
        self._lock = threading.Lock()

    def __call__(self, _request_body, _config) -> str:
        with self._lock:
            if not self.responses:
                raise AssertionError("No mocked LLM responses remain.")
            response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


def make_runtime(responses: Sequence[object], *, model: str = "deepseek-chat") -> ProposalRuntime:
    return ProposalRuntime(
        RuntimeConfig(
            api_key="test-key",
            api_base="https://api.test/v1",
            primary_model=model,
            available_models=(model, "kimi-k2.5", "glm-5"),
            temperature=0.2,
            max_tokens=4096,
            timeout_s=45,
            llm_concurrency=20,
        ),
        transport=QueueTransport(responses),
    )


@contextmanager
def patched_fixture_catalog():
    with (
        patch.object(catalog, "BENCHMARK_ROOT", FIXTURE_BENCHMARK_ROOT),
        patch.object(catalog, "REGISTRY_PATH", FIXTURE_REGISTRY_PATH),
    ):
        yield


def load_fixture_codegen_tasks(
    task_id: str | None = None,
    *,
    included_in_main_comparison: bool | None = None,
) -> list[dict[str, object]]:
    with patched_fixture_catalog():
        return catalog.load_codegen_tasks(
            task_id,
            included_in_main_comparison=included_in_main_comparison,
        )


def list_fixture_codegen_task_summaries() -> list[dict[str, object]]:
    with patched_fixture_catalog():
        return catalog.list_codegen_task_summaries()


@contextmanager
def patch_runner_fixture_catalog():
    with (
        patch("app.entries.runner.load_codegen_tasks", side_effect=load_fixture_codegen_tasks),
        patch("app.entries.runner.list_codegen_task_summaries", side_effect=list_fixture_codegen_task_summaries),
    ):
        yield
