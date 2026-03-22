from __future__ import annotations

import json
import threading
from collections.abc import Sequence

from app.codegen.config import RuntimeConfig
from app.codegen.llm import ProposalRuntime


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
            max_tokens=1400,
            timeout_s=45,
        ),
        transport=QueueTransport(responses),
    )
