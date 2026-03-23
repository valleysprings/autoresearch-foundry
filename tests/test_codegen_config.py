from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.config import load_runtime_config
from app.codegen.errors import ConfigError
from app.configs.runtime import (
    DEFAULT_AVAILABLE_MODELS,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_RUNTIME_MAX_TOKENS,
    DEFAULT_RUNTIME_TEMPERATURE,
    DEFAULT_RUNTIME_TIMEOUT_S,
)


class CodegenConfigTest(unittest.TestCase):
    def test_dotenv_loads_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.primary_model, DEFAULT_PRIMARY_MODEL)
            self.assertEqual(config.available_models, DEFAULT_AVAILABLE_MODELS)
            self.assertEqual(config.temperature, DEFAULT_RUNTIME_TEMPERATURE)
            self.assertEqual(config.max_tokens, DEFAULT_RUNTIME_MAX_TOKENS)
            self.assertEqual(config.timeout_s, DEFAULT_RUNTIME_TIMEOUT_S)
            self.assertEqual(config.llm_concurrency, 20)

    def test_shell_env_can_override_primary_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"AUTORESEARCH_PRIMARY_MODEL": "shell-model"},
            clear=True,
        ):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.primary_model, "shell-model")
            self.assertEqual(config.available_models[0], "shell-model")

    def test_missing_required_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.max_tokens, DEFAULT_RUNTIME_MAX_TOKENS)

    def test_available_models_env_is_parsed_and_keeps_primary_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_AVAILABLE_MODELS=kimi-k2.5, glm-5",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.available_models, ("deepseek-chat", "kimi-k2.5", "glm-5"))
            self.assertEqual(config.with_model("glm-5").active_model, "glm-5")

    def test_invalid_url_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=not-a-url",
                    ]
                )
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(Path(tmp_dir))

    def test_llm_concurrency_env_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_LLM_CONCURRENCY=7",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.llm_concurrency, 7)

    def test_runtime_knobs_can_still_be_overridden_by_env_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {
                "AUTORESEARCH_TEMPERATURE": "0.4",
                "AUTORESEARCH_MAX_TOKENS": "2048",
                "AUTORESEARCH_TIMEOUT_S": "90",
            },
            clear=True,
        ):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.temperature, 0.4)
            self.assertEqual(config.max_tokens, 2048)
            self.assertEqual(config.timeout_s, 90)


if __name__ == "__main__":
    unittest.main()
