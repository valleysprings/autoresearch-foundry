from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.codegen.config import load_runtime_config
from app.codegen.errors import ConfigError


class CodegenConfigTest(unittest.TestCase):
    def test_dotenv_loads_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=4096",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.primary_model, "deepseek-chat")
            self.assertEqual(config.available_models, ("deepseek-chat",))
            self.assertEqual(config.max_tokens, 4096)
            self.assertEqual(config.llm_concurrency, 20)

    def test_shell_env_overrides_dotenv(self) -> None:
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
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=4096",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.primary_model, "shell-model")

    def test_missing_required_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_TIMEOUT_S=45",
                    ]
                )
            )
            with self.assertRaises(ConfigError):
                load_runtime_config(Path(tmp_dir))

    def test_available_models_env_is_parsed_and_keeps_primary_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "AUTORESEARCH_API_KEY=test-key",
                        "AUTORESEARCH_API_BASE=https://api.example.com/v1",
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_AVAILABLE_MODELS=kimi-k2.5, glm-5",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=4096",
                        "AUTORESEARCH_TIMEOUT_S=45",
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
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=4096",
                        "AUTORESEARCH_TIMEOUT_S=45",
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
                        "AUTORESEARCH_PRIMARY_MODEL=deepseek-chat",
                        "AUTORESEARCH_TEMPERATURE=0.2",
                        "AUTORESEARCH_MAX_TOKENS=4096",
                        "AUTORESEARCH_TIMEOUT_S=45",
                        "AUTORESEARCH_LLM_CONCURRENCY=7",
                    ]
                )
            )
            config = load_runtime_config(Path(tmp_dir))
            self.assertEqual(config.llm_concurrency, 7)


if __name__ == "__main__":
    unittest.main()
